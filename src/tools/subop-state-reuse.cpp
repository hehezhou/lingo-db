#include "features.h"

#include "lingodb/catalog/Catalog.h"
#include "lingodb/compiler/Conversion/ArrowToStd/ArrowToStd.h"
#include "lingodb/compiler/Conversion/DBToStd/DBToStd.h"
#include "lingodb/compiler/Conversion/SubOpToControlFlow/SubOpToControlFlowPass.h"
#include "lingodb/compiler/Conversion/RelAlgToSubOp/RelAlgToSubOpPass.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/CrossQueryStateReuse.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateExtraction.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/helper.h"
#include "lingodb/compiler/mlir-support/eval.h"
#include "lingodb/execution/LLVMBackends.h"
#include "lingodb/execution/ResultProcessing.h"
#include "lingodb/execution/Frontend.h"
#include "lingodb/runtime/ExecutionContext.h"
#include "lingodb/runtime/Session.h"
#include "lingodb/scheduler/Scheduler.h"
#include "lingodb/scheduler/Tasks.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Transforms/Passes.h"

#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

#include <chrono>
#include <cassert>
#include <cctype>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char kToolName[] = "subop-state-reuse";

/// Interpret \p envVar as an explicit boolean. Unset or empty → false.
/// Tokens `0`, `false`, `no`, `off` (case-insensitive) → false so e.g. `LINGODB_*=0` does not enable the flag.
/// Tokens `1`, `true`, `yes`, `on` → true.
static bool envFlagEnabled(const char* envVar) {
   const char* v = std::getenv(envVar);
   if (!v || v[0] == '\0') return false;
   llvm::StringRef s(v);
   if (s.equals_insensitive("0") || s.equals_insensitive("false") || s.equals_insensitive("no") ||
       s.equals_insensitive("off")) {
      return false;
   }
   return s.equals_insensitive("1") || s.equals_insensitive("true") || s.equals_insensitive("yes") ||
          s.equals_insensitive("on");
}

/// Elapsed milliseconds since \p start (same pattern as `LLVMBackends.cpp`).
static double millisSince(std::chrono::high_resolution_clock::time_point start) {
   auto end = std::chrono::high_resolution_clock::now();
   return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
}

static void dumpModuleToFile(mlir::ModuleOp module, const llvm::Twine& path) {
   std::error_code ec;
   llvm::raw_fd_ostream os(path.str(), ec);
   if (ec) {
      llvm::errs() << kToolName << ": failed to write " << path << ": " << ec.message() << "\n";
      return;
   }
   module.print(os);
}

const llvm::json::Array& getQueriesArray(const llvm::json::Object& obj) {
   if (auto* arr = obj.getArray("queries")) {
      return *arr;
   }
   auto* arr = obj.getArray("sql");
   assert(arr && "JSON input must contain 'queries' or 'sql' array");
   return *arr;
}

void runPasses(mlir::ModuleOp moduleOp, lingodb::catalog::Catalog* catalog) {
   using namespace lingodb::compiler::dialect;

   // 1) (Optional) query optimization on RelAlg.
   {
      mlir::PassManager qoptPm(moduleOp.getContext());
      qoptPm.enableVerifier(true);
      relalg::createQueryOptPipeline(qoptPm, catalog);
      assert(succeeded(qoptPm.run(moduleOp)));
   }

   // 2) Lower RelAlg -> SubOp.
   {
      mlir::PassManager lowerRelAlgPm(moduleOp.getContext());
      lowerRelAlgPm.enableVerifier(true);
      relalg::createLowerRelAlgToSubOpPipeline(lowerRelAlgPm);
      assert(succeeded(lowerRelAlgPm.run(moduleOp)));
   }

   // 3) SubOp "opt" pipeline (matching Execution.cpp up to PrepareLoweringPass).
   {
      mlir::PassManager optSubOpPm(moduleOp.getContext());
      optSubOpPm.enableVerifier(true);

      optSubOpPm.addPass(subop::createFoldColumnsPass());
      optSubOpPm.addPass(subop::createCommonPiplineEliminationPass());
      optSubOpPm.addPass(subop::createReuseLocalPass());
      optSubOpPm.addPass(subop::createSpecializeSubOpPass(true));
      optSubOpPm.addPass(subop::createNormalizeSubOpPass());
      optSubOpPm.addPass(subop::createPullGatherUpPass());
      optSubOpPm.addPass(subop::createEnforceOrderPass());
      optSubOpPm.addPass(subop::createInlineNestedMapPass());
      optSubOpPm.addPass(subop::createFinalizePass());
      optSubOpPm.addPass(subop::createSplitIntoExecutionStepsPass());
      if (!moduleOp->hasAttr("subop.sequential")) {
         optSubOpPm.addNestedPass<mlir::func::FuncOp>(subop::createParallelizePass());
         optSubOpPm.addPass(subop::createSpecializeParallelPass());
      }
      optSubOpPm.addPass(subop::createPrepareLoweringPass());

      assert(succeeded(optSubOpPm.run(moduleOp)));
   }
}

/// Lower from SubOp layer through tail PM. Returns false if lowerDB fails.
static bool lowerFromSubOpLayer(mlir::ModuleOp subopModule, const char* snapshotDir = nullptr,
                                llvm::StringRef snapshotLabel = {}) {
   using namespace lingodb::compiler::dialect;

   auto snap = [&](llvm::StringRef stage, mlir::ModuleOp mod) {
      if (!snapshotDir) return;
      llvm::SmallString<256> filePath;
      if (snapshotLabel.empty()) {
         llvm::sys::path::append(filePath, snapshotDir, "snapshots", (stage + ".mlir").str());
      } else {
         llvm::sys::path::append(filePath, snapshotDir, "snapshots", snapshotLabel, (stage + ".mlir").str());
      }
      llvm::sys::fs::create_directories(llvm::sys::path::parent_path(filePath));
      dumpModuleToFile(mod, filePath);
   };

   // Default: keep `parallel` scan/map attrs from `runPasses` so JIT matches normal parallel lowering.
   // Opt-in sequential debugging via `LINGODB_SUBOP_FORCE_SEQUENTIAL=1` (sets module attr + strips `parallel`).
   if (envFlagEnabled("LINGODB_SUBOP_FORCE_SEQUENTIAL")) {
      subopModule->setAttr("subop.sequential", mlir::UnitAttr::get(subopModule.getContext()));
      subopModule.walk([&](mlir::Operation* op) {
         if (op->hasAttr("parallel")) op->removeAttr("parallel");
      });
   }

   // From SubOp layer (after PrepareLoweringPass) to imperative lowering.
   {
      mlir::PassManager lowerSubOpPm(subopModule.getContext());
      lowerSubOpPm.enableVerifier(true);
      lowerSubOpPm.addPass(subop::createLowerSubOpPass());
      lowerSubOpPm.addPass(lingodb::compiler::createCanonicalizerPass());
      lowerSubOpPm.addPass(mlir::createCSEPass());
      if (failed(lowerSubOpPm.run(subopModule))) return false;
   }
   snap("before-lower-db", subopModule);

   // Lower DB imperative ops.
   {
      mlir::PassManager lowerDBPm(subopModule.getContext());
      lowerDBPm.enableVerifier(true);
      // Help downstream pipelines by reconciling unrealized casts early.
      lowerDBPm.addPass(mlir::createReconcileUnrealizedCastsPass());
      // Experimental: fold `util.generic_memref_cast` + unrealized conversion into a single
      // `util.generic_memref_cast` to the final type.
      struct FixGenericMemrefCastTargets
         : public mlir::PassWrapper<FixGenericMemrefCastTargets, mlir::OperationPass<mlir::ModuleOp>> {
         void runOnOperation() override {
            auto module = getOperation();
            module.walk([&](mlir::UnrealizedConversionCastOp castOp) {
               if (castOp.getNumOperands() != 1 || castOp.getNumResults() != 1) return;
               auto gmc = castOp.getOperand(0).getDefiningOp<lingodb::compiler::dialect::util::GenericMemrefCastOp>();
               if (!gmc) return;
               // Only rewrite if the cast result is a util.ref-like type (usually nested ref/tuple).
               mlir::Type dstT = castOp.getResult(0).getType();
               llvm::SmallString<64> s;
               {
                  llvm::raw_svector_ostream os(s);
                  dstT.print(os);
               }
               if (!llvm::StringRef(s).contains("!util.ref")) return;

               mlir::OpBuilder b(gmc);
               auto newCast = b.create<lingodb::compiler::dialect::util::GenericMemrefCastOp>(gmc.getLoc(), dstT, gmc.getVal());
               castOp.getResult(0).replaceAllUsesWith(newCast.getRes());
               castOp.erase();
               // erase old gmc if dead
               if (gmc->use_empty()) gmc.erase();
            });
         }
      };
      lowerDBPm.addPass(std::make_unique<FixGenericMemrefCastTargets>());
      struct FixUnrealizedScalarCasts
         : public mlir::PassWrapper<FixUnrealizedScalarCasts, mlir::OperationPass<mlir::ModuleOp>> {
         void runOnOperation() override {
            auto module = getOperation();
            module.walk([&](mlir::UnrealizedConversionCastOp castOp) {
               if (castOp.getNumOperands() != 1 || castOp.getNumResults() != 1) return;
               mlir::Value in = castOp.getOperand(0);
               mlir::Type srcT = in.getType();
               mlir::Type dstT = castOp.getResult(0).getType();
               // Only handle scalar casts; util.ref and tuples are handled by other conversions.
               llvm::SmallString<64> s1, s2;
               {
                  llvm::raw_svector_ostream os(s1);
                  srcT.print(os);
               }
               {
                  llvm::raw_svector_ostream os(s2);
                  dstT.print(os);
               }
               if (llvm::StringRef(s1).contains("!util.ref") || llvm::StringRef(s2).contains("!util.ref")) return;
               if (llvm::StringRef(s1).contains("tuple<") || llvm::StringRef(s2).contains("tuple<")) return;

               mlir::OpBuilder b(castOp);
               auto dbCast = b.create<lingodb::compiler::dialect::db::CastOp>(castOp.getLoc(), dstT, in);
               castOp.getResult(0).replaceAllUsesWith(dbCast.getRes());
               castOp.erase();
            });
         }
      };
      // Experimental: turn remaining unrealized scalar casts into explicit db.cast ops.
      lowerDBPm.addPass(std::make_unique<FixUnrealizedScalarCasts>());
      db::createLowerDBPipeline(lowerDBPm);
      if (failed(lowerDBPm.run(subopModule))) {
         snap("lower-db-failed", subopModule);
         return false;
      }
   }
   snap("after-lower-db", subopModule);

   // Lower Arrow ops to std.
   {
      mlir::PassManager lowerArrowPm(subopModule.getContext());
      lowerArrowPm.enableVerifier(true);
      lowerArrowPm.addPass(lingodb::compiler::dialect::arrow::createLowerToStdPass());
      lowerArrowPm.addPass(lingodb::compiler::createCanonicalizerPass());
      lowerArrowPm.addPass(mlir::createLoopInvariantCodeMotionPass());
      lowerArrowPm.addPass(mlir::createCSEPass());
      if (failed(lowerArrowPm.run(subopModule))) return false;
   }

   // Cross-query reuse can leave a few `builtin.unrealized_conversion_cast` ops after dialect
   // lowering; reconcile + canonicalize before LLVM translation (mirrors LLVMBackends.cpp).
   {
      mlir::PassManager tailPm(subopModule.getContext());
      tailPm.enableVerifier(true);
      tailPm.addPass(mlir::createReconcileUnrealizedCastsPass());
      tailPm.addPass(lingodb::compiler::createCanonicalizerPass());
      tailPm.addPass(mlir::createCSEPass());
      if (failed(tailPm.run(subopModule))) return false;
   }
   snap("after-tail-pm", subopModule);
   return true;
}

static void executeFromSubOpLayer(mlir::ModuleOp subopModule,
                                  lingodb::runtime::ExecutionContext* executionContext) {
   using namespace lingodb::compiler::dialect;

   if (!lowerFromSubOpLayer(subopModule)) {
      subopModule.dump();
      assert(0 && "lowerFromSubOpLayer failed (dumped module above)");
   }

   // Execute using default LLVM backend and print result.
   auto backend = std::shared_ptr<lingodb::execution::ExecutionBackend>(
      lingodb::execution::createDefaultLLVMBackend(/*optimize*/ true).release());
   auto printer = std::shared_ptr<lingodb::execution::ResultProcessor>(
      lingodb::execution::createTablePrinter().release());

   struct RunWithContextTask : public lingodb::scheduler::TaskWithContext {
      std::function<void()> fn;
      RunWithContextTask(lingodb::runtime::ExecutionContext* ctx, std::function<void()> fn)
         : TaskWithContext(ctx), fn(std::move(fn)) {}
      bool allocateWork() override {
         if (workExhausted.exchange(true)) return false;
         return true;
      }
      void performWork() override { fn(); }
   };

   lingodb::scheduler::awaitEntryTask(std::make_unique<RunWithContextTask>(executionContext, [&]() {
      lingodb::runtime::ExecutionContext::clearResult(0);
      backend->execute(subopModule, executionContext);
      printer->process(executionContext);
   }));
}

} // namespace

int main(int argc, char** argv) {
   if (argc == 2 && std::string(argv[1]) == "--features") {
      printFeatures();
      return 0;
   }

   // Usage:
   // - subop-state-reuse <db_dir> <sql_or_json>
   // - subop-state-reuse <db_dir> <sql_file_a> <sql_file_b>
   assert(argc == 3 || argc == 4);
   std::string dbDir = argv[1];

   lingodb::compiler::support::eval::init();
   // Execution backends expect a running scheduler (even if queries are forced sequential).
   auto schedulerHandle = lingodb::scheduler::startScheduler(/*numWorkers*/ 0);
   std::shared_ptr<lingodb::catalog::Catalog> catalog = lingodb::catalog::Catalog::create(dbDir, /*eagerLoading*/ true);
   auto session = lingodb::runtime::Session::createSession(dbDir, /*eagerLoading*/ true);

   llvm::SmallVector<std::string, 16> queries;
   if (argc == 4) {
      for (int fi = 2; fi <= 3; fi++) {
         auto fileOrErr = llvm::MemoryBuffer::getFile(argv[fi]);
         assert(!fileOrErr.getError());
         queries.push_back((*fileOrErr)->getBuffer().str());
      }
   } else {
      std::string input = argv[2];
      auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(input);
      assert(!fileOrErr.getError());

      auto buf = (*fileOrErr)->getBuffer();

      // Experimental parsing rule:
      // - If the input starts with '{', treat it as JSON and assert it contains a "queries"/"sql" array.
      // - Otherwise, treat it as a plain SQL string.
      size_t off = 0;
      while (off < buf.size() && std::isspace(static_cast<unsigned char>(buf[off]))) off++;
      if (off < buf.size() && buf[off] == '{') {
         auto parsed = llvm::json::parse(buf);
         assert(parsed && "failed to parse JSON input");
         auto* obj = parsed->getAsObject();
         assert(obj && "JSON input must be an object");
         const llvm::json::Array& arr = getQueriesArray(*obj);
         for (auto& v : arr) {
            auto s = v.getAsString();
            assert(s && "query array entries must be strings");
            queries.push_back(s->str());
         }
         assert(!queries.empty() && "query array must not be empty");
      } else {
         queries.push_back(buf.str());
      }
   }

   struct QueryRun {
      std::unique_ptr<mlir::MLIRContext> ctx;
      std::unique_ptr<lingodb::execution::Frontend> frontend;
      mlir::ModuleOp module;
   };
   std::vector<QueryRun> runs;
   runs.reserve(queries.size());

   double optimizationMs = 0;
   llvm::SmallVector<double, 8> queryCompileMs;
   queryCompileMs.reserve(queries.size());

   for (size_t i = 0; i < queries.size(); i++) {
      std::string sql = queries[i];

      QueryRun run;
      run.ctx = std::make_unique<mlir::MLIRContext>();
      lingodb::execution::initializeContext(*run.ctx, /*includeLLVM*/ true);

      run.frontend = lingodb::execution::createSQLFrontend();
      run.frontend->setCatalog(catalog.get());
      run.frontend->setContext(run.ctx.get());
      run.frontend->loadFromString(sql);

      auto& feErr = run.frontend->getError();
      assert(!feErr);

      mlir::ModuleOp* moduleOpPtr = run.frontend->getModule();
      assert(moduleOpPtr);
      run.module = *moduleOpPtr;

      {
         auto t0 = std::chrono::high_resolution_clock::now();
         runPasses(run.module, catalog.get());
         const double ms = millisSince(t0);
         queryCompileMs.push_back(ms);
         optimizationMs += ms;
      }
      runs.push_back(std::move(run));

      // 1) Print SubOp layer IR (previous json-sql-to-subop output).
      llvm::outs() << "\n// ============================\n";
      llvm::outs() << "// query[" << i << "] subop layer\n";
      llvm::outs() << "// ============================\n";
      runs.back().module.print(llvm::outs());
      llvm::outs() << "\n";

      // 2) Print execution-step/state debug.
      lingodb::compiler::dialect::subop::printExecutionSteps(runs.back().module, llvm::outs());
      llvm::outs() << "\n";
   }

   // If we have at least two queries, detect matches and inject cache_put/cache_get to reuse states.
   assert(runs.size() == 2);
   llvm::SmallVector<std::pair<int, mlir::ModuleOp>, 8> qmods;
   for (size_t i = 0; i < runs.size(); i++) {
      qmods.push_back({static_cast<int>(i), runs[i].module});
   }
   auto matches = lingodb::compiler::dialect::subop::collectCrossQueryStateMatchPairs(qmods);
   // Print matches on the unmodified modules.
   lingodb::compiler::dialect::subop::printCrossQueryStateMatches(qmods, llvm::outs());
   // Only inject reuse for the first two modules for now.
   llvm::SmallVector<lingodb::compiler::dialect::subop::CrossQueryStateMatchPair, 64> firstPair;
   for (auto& m : matches) {
      if ((m.queryA == 0 && m.queryB == 1) || (m.queryA == 1 && m.queryB == 0)) {
         // Normalize to (0,1).
         if (m.queryA == 0) firstPair.push_back(m);
         else {
            auto mm = m;
            std::swap(mm.queryA, mm.queryB);
            std::swap(mm.stateA, mm.stateB);
            firstPair.push_back(mm);
         }
      }
   }
   auto tRewrite = std::chrono::high_resolution_clock::now();
   auto rewriteRes = lingodb::compiler::dialect::subop::rewritePlansWithSyntheticQuery0(
      runs[0].module, runs[1].module, firstPair);
   const double rewriteMs = millisSince(tRewrite);
   optimizationMs += rewriteMs;
   if (mlir::failed(mlir::verify(runs[0].module)) || mlir::failed(mlir::verify(runs[1].module)) ||
       (rewriteRes.query0 && mlir::failed(mlir::verify(*rewriteRes.query0)))) {
      llvm::errs() << kToolName << ": MLIR verification failed after cross-query reuse rewrite"
                   << " (optimization_ms=" << optimizationMs << ")\n";
      return 1;
   }
   llvm::outs() << "\n// reuse_targets: query[0]=" << rewriteRes.numTargetsQuery0
                  << " query[1]=" << rewriteRes.numTargetsQuery1 << "\n";
   llvm::outs() << "\n// reuse_targets_q0_mapped: " << rewriteRes.numTargetsQuery0Mapped << "\n";

   const char* dumpSubOpDir = std::getenv("LINGODB_DUMP_SUBOP_DIR");
   const bool dumpLowering = envFlagEnabled("LINGODB_DUMP_LOWERING");
   if (dumpLowering && dumpSubOpDir) {
      auto dumpScenario = [&](mlir::ModuleOp mod, const llvm::Twine& outDir, llvm::StringRef scenarioName) {
         const std::string dir = outDir.str();
         llvm::sys::fs::create_directories(dir);
         dumpModuleToFile(mod, dir + "/consumer-subop.mlir");
         mlir::OwningOpRef<mlir::ModuleOp> clone = mlir::cast<mlir::ModuleOp>(mod->clone());
         const bool ok = lowerFromSubOpLayer(*clone, dir.c_str(), /*snapshotLabel=*/{});
         llvm::errs() << "[dump] " << scenarioName << " lowerDB " << (ok ? "OK" : "FAILED") << "\n";
         return ok;
      };

      const std::string base(dumpSubOpDir);
      if (rewriteRes.query0 && rewriteRes.numTargetsQuery0Mapped > 0) {
         dumpScenario(*rewriteRes.query0, base + "/rewrite-synthetic-query0", "rewrite-synthetic-query0");
      }
      dumpScenario(runs[0].module, base + "/rewrite-query0-test.sql", "rewrite-query0-test.sql");
      dumpScenario(runs[1].module, base + "/rewrite-query1-test2.sql", "rewrite-query1-test2.sql");
   } else if (dumpSubOpDir) {
      const std::string base(dumpSubOpDir);
      if (rewriteRes.query0 && rewriteRes.numTargetsQuery0Mapped > 0) {
         llvm::sys::fs::create_directories(base + "/rewrite-synthetic-query0");
         dumpModuleToFile(*rewriteRes.query0, base + "/rewrite-synthetic-query0/consumer-subop.mlir");
      }
      llvm::sys::fs::create_directories(base + "/rewrite-query0-test.sql");
      llvm::sys::fs::create_directories(base + "/rewrite-query1-test2.sql");
      dumpModuleToFile(runs[0].module, base + "/rewrite-query0-test.sql/consumer-subop.mlir");
      dumpModuleToFile(runs[1].module, base + "/rewrite-query1-test2.sql/consumer-subop.mlir");
   }

   // Optional heavy debug printing (can be huge / sometimes crashes when IR is malformed).
   // Enable via env var: LINGODB_REUSE_PRINT_REWRITTEN=1
   const bool printRewritten = (std::getenv("LINGODB_REUSE_PRINT_REWRITTEN") != nullptr);
   if (printRewritten) {
      if (rewriteRes.numTargetsQuery0Mapped > 0) {
         assert(rewriteRes.query0);
         llvm::outs() << "\n// ============================\n";
         llvm::outs() << "// query[0] (synthetic) subop layer (post-rewrite)\n";
         llvm::outs() << "// ============================\n";
         (*rewriteRes.query0)->print(llvm::outs());
         llvm::outs() << "\n";
      }
      llvm::outs() << "\n// ============================\n";
      llvm::outs() << "// Step layout vs SubOpToControlFlow stderr ordinal\n";
      llvm::outs() << "// ============================\n";
      if (rewriteRes.query0 && rewriteRes.numTargetsQuery0Mapped > 0) {
         llvm::outs() << "// query[0] (synthetic) — same indices as lowering uses for that module\n";
         lingodb::compiler::dialect::subop::printTopLevelExecutionStepLayout(*rewriteRes.query0, llvm::outs());
      }
      for (size_t i = 0; i < 2; i++) {
         llvm::outs() << "\n// --- query[" << i << "] execution_step layout (post-rewrite IR) ---\n";
         lingodb::compiler::dialect::subop::printTopLevelExecutionStepLayout(runs[i].module, llvm::outs());
      }

      for (size_t i = 0; i < 2; i++) {
         llvm::outs() << "\n// ============================\n";
         llvm::outs() << "// query[" << i << "] subop layer (post-rewrite)\n";
         llvm::outs() << "// ============================\n";
         runs[i].module.print(llvm::outs());
         llvm::outs() << "\n";
      }
   }

   // Execute query0 first, then the original queries.
   // Use a shared ExecutionContext so cache_get/cache_put pointers remain valid across the
   // synthetic producer run and the consumer queries.
   const bool skipExecute = (std::getenv("LINGODB_SKIP_EXECUTE") != nullptr);
   if (skipExecute) {
      llvm::outs() << "\n// (LINGODB_SKIP_EXECUTE set: skipping JIT execution)\n";
      llvm::outs() << "\n// timing: optimization_ms=" << optimizationMs << " execution_ms=0 (skipped)\n";
      return 0;
   }
   double executionMs = 0;
   llvm::SmallVector<double, 8> executionMsPerRun;
   // `putCachedState` stores raw pointers into memory registered on the **allocating**
   // `ExecutionContext`. Destroying that context before consumers run leaves entries in the
   // process-global `gCachedStates` map dangling. Use one context for synthetic + both consumers,
   // then clear the global map after it goes out of scope.
   {
      lingodb::runtime::ExecutionContext::clearAllCachedStates();
      auto sharedExecCtx = session->createExecutionContext();
      auto runOne = [&](mlir::ModuleOp mod, const std::string& label) {
         llvm::outs() << "\n// ============================\n";
         llvm::outs() << label << "\n";
         llvm::outs() << "// ============================\n";
         mlir::OwningOpRef<mlir::ModuleOp> execModule = mlir::cast<mlir::ModuleOp>(mod->clone());
         auto tExec = std::chrono::high_resolution_clock::now();
         executeFromSubOpLayer(*execModule, sharedExecCtx.get());
         const double oneMs = millisSince(tExec);
         executionMs += oneMs;
         executionMsPerRun.push_back(oneMs);
         llvm::outs().flush();
         std::cout.flush();
         llvm::outs() << "\n";
      };
      // When cross-query rewrite could not map any donor state into the synthetic module, it only
      // contains an empty execution_group shell — skip JIT for that shell.
      if (rewriteRes.numTargetsQuery0Mapped > 0) {
         runOne(*rewriteRes.query0, "// query[0] (synthetic) execute");
      } else {
         llvm::outs() << "\n// (skip synthetic execute: reuse_targets_q0_mapped==0)\n";
      }
      for (size_t i = 0; i < runs.size(); i++) {
         runOne(runs[i].module, "// query[" + std::to_string(i) + "] execute");
      }
   }
   lingodb::runtime::ExecutionContext::clearAllCachedStates();
   llvm::outs() << "\n// timing_compile_ms: per_query=[";
   for (size_t i = 0; i < queryCompileMs.size(); i++) {
      if (i) llvm::outs() << ",";
      llvm::outs() << queryCompileMs[i];
   }
   llvm::outs() << "] rewrite=" << rewriteMs << " total_optimization_ms=" << optimizationMs << "\n";
   llvm::outs() << "// timing_execution_ms: per_run=[";
   for (size_t i = 0; i < executionMsPerRun.size(); i++) {
      if (i) llvm::outs() << ",";
      llvm::outs() << executionMsPerRun[i];
   }
   llvm::outs() << "] total_execution_ms=" << executionMs << "\n";
   if (rewriteRes.numTargetsQuery0Mapped > 0 && executionMsPerRun.size() >= 3) {
      double synthMs = executionMsPerRun[0];
      double consumerMs = 0;
      for (size_t i = 1; i < executionMsPerRun.size(); ++i) consumerMs += executionMsPerRun[i];
      llvm::outs() << "// timing_note: synthetic_ms=" << synthMs
                    << " consumers_only_ms=" << consumerMs << "\n";
   }
   llvm::outs() << "// timing: optimization_ms=" << optimizationMs << " execution_ms=" << executionMs << "\n";
   return 0;
}
