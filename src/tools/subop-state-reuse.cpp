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
#include <unordered_map>
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

static double timingOrZero(const std::unordered_map<std::string, double>& timing, llvm::StringRef key) {
   auto it = timing.find(key.str());
   return it != timing.end() ? it->second : 0.0;
}

/// Per-run timings aligned with `run-sql` / `TimingPrinter` backend keys where possible.
struct SubOpExecuteTiming {
   double wallMs = 0;
   /// SubOp → DB/Arrow lowering (not reported by run-sql as a single bucket).
   double lowerMs = 0;
   /// `DefaultCPULLVMBackend`: time inside generated `main()` only (same as run-sql `executionTime`).
   double executionTime = 0;
   double lowerToLLVM = 0;
   double toLLVMIR = 0;
   double llvmOptimize = 0;
   double llvmCodeGen = 0;

   double llvmJitMs() const { return lowerToLLVM + toLLVMIR + llvmOptimize + llvmCodeGen; }

   static SubOpExecuteTiming fromBackend(const std::unordered_map<std::string, double>& backendTiming) {
      SubOpExecuteTiming t;
      t.executionTime = timingOrZero(backendTiming, "executionTime");
      t.lowerToLLVM = timingOrZero(backendTiming, "lowerToLLVM");
      t.toLLVMIR = timingOrZero(backendTiming, "toLLVMIR");
      t.llvmOptimize = timingOrZero(backendTiming, "llvmOptimize");
      t.llvmCodeGen = timingOrZero(backendTiming, "llvmCodeGen");
      return t;
   }

   void add(const SubOpExecuteTiming& o) {
      wallMs += o.wallMs;
      lowerMs += o.lowerMs;
      executionTime += o.executionTime;
      lowerToLLVM += o.lowerToLLVM;
      toLLVMIR += o.toLLVMIR;
      llvmOptimize += o.llvmOptimize;
      llvmCodeGen += o.llvmCodeGen;
   }
};

static void printTimingSegment(llvm::raw_ostream& os, llvm::StringRef name, const SubOpExecuteTiming& t) {
   os << "// timing_segment " << name << ": executionTime=" << t.executionTime << " lower_ms=" << t.lowerMs
      << " llvm_jit_ms=" << t.llvmJitMs() << " (lowerToLLVM=" << t.lowerToLLVM << " toLLVMIR=" << t.toLLVMIR
      << " llvmOptimize=" << t.llvmOptimize << " llvmCodeGen=" << t.llvmCodeGen << ")\n";
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
      auto res = qoptPm.run(moduleOp);
      assert(succeeded(res));
   }

   // 2) Lower RelAlg -> SubOp.
   {
      mlir::PassManager lowerRelAlgPm(moduleOp.getContext());
      lowerRelAlgPm.enableVerifier(true);
      relalg::createLowerRelAlgToSubOpPipeline(lowerRelAlgPm);
      auto res = lowerRelAlgPm.run(moduleOp);
      assert(succeeded(res));
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

      auto res = optSubOpPm.run(moduleOp);
      assert(succeeded(res));
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
      if (!envFlagEnabled("LINGODB_SUBOP_SKIP_LOWER_FIXUPS"))
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
               if (srcT == dstT) {
                  castOp.getResult(0).replaceAllUsesWith(in);
                  castOp.erase();
                  return;
               }
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
      if (!envFlagEnabled("LINGODB_SUBOP_SKIP_LOWER_FIXUPS"))
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

static SubOpExecuteTiming executeFromSubOpLayer(mlir::ModuleOp subopModule,
                                                lingodb::runtime::ExecutionContext* executionContext,
                                                std::optional<unsigned> resultQueryIndex = std::nullopt) {
   using namespace lingodb::compiler::dialect;

   auto wallStart = std::chrono::high_resolution_clock::now();
   SubOpExecuteTiming out;

   auto lowerStart = std::chrono::high_resolution_clock::now();
   if (!lowerFromSubOpLayer(subopModule)) {
      subopModule.dump();
      assert(0 && "lowerFromSubOpLayer failed (dumped module above)");
   }
   out.lowerMs = millisSince(lowerStart);

   // Same backend as run-sql DEFAULT mode (`createDefaultLLVMBackend()` → optimize=true).
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
      // Shared context: required so `cache_put` pointers stay valid for later `cache_get`.
      // Does not affect `executionTime` (measured around `main()` only). Clear printed result
      // slot; do not tear down context between synthetic and consumers.
      lingodb::runtime::ExecutionContext::clearResult(0);
      backend->execute(subopModule, executionContext);
      if (resultQueryIndex) {
         std::cout << "// result_begin: query[" << *resultQueryIndex << "]\n";
         std::cout.flush();
      }
      printer->process(executionContext);
      if (resultQueryIndex) {
         std::cout << "// result_end: query[" << *resultQueryIndex << "]\n";
      }
      std::cout.flush();
   }));

   out.add(SubOpExecuteTiming::fromBackend(backend->getTiming()));
   out.wallMs = millisSince(wallStart);
   return out;
}

} // namespace

int main(int argc, char** argv) {
   if (argc == 2 && std::string(argv[1]) == "--features") {
      printFeatures();
      return 0;
   }

   bool skipReuseRewrite = false;
   llvm::SmallVector<const char*, 8> positionalArgs;
   positionalArgs.push_back(argv[0]);
   for (int i = 1; i < argc; i++) {
      llvm::StringRef arg(argv[i]);
      if (arg == "--no-reuse-rewrite") {
         skipReuseRewrite = true;
         continue;
      }
      positionalArgs.push_back(argv[i]);
   }

   // Usage:
   // - subop-state-reuse <db_dir> <sql_or_json>
   // - subop-state-reuse <db_dir> <sql_file_a> <sql_file_b> [sql_file_c ...]
   assert(positionalArgs.size() >= 3);
   std::string dbDir = positionalArgs[1];

   lingodb::compiler::support::eval::init();
   // Execution backends expect a running scheduler (even if queries are forced sequential).
   auto schedulerHandle = lingodb::scheduler::startScheduler(/*numWorkers*/ 0);
   std::shared_ptr<lingodb::catalog::Catalog> catalog = lingodb::catalog::Catalog::create(dbDir, /*eagerLoading*/ true);
   auto session = lingodb::runtime::Session::createSession(dbDir, /*eagerLoading*/ true);

   llvm::SmallVector<std::string, 16> queries;
   if (positionalArgs.size() >= 4) {
      for (size_t fi = 2; fi < positionalArgs.size(); fi++) {
         auto fileOrErr = llvm::MemoryBuffer::getFile(positionalArgs[fi]);
         assert(!fileOrErr.getError());
         queries.push_back((*fileOrErr)->getBuffer().str());
      }
   } else {
      std::string input = positionalArgs[2];
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
   assert(queries.size() >= 2 && "subop-state-reuse batch input must contain at least two queries");

   struct QueryRun {
      std::unique_ptr<lingodb::execution::Frontend> frontend;
      mlir::ModuleOp module;
   };
   auto sharedMlirContext = std::make_unique<mlir::MLIRContext>();
   lingodb::execution::initializeContext(*sharedMlirContext, /*includeLLVM*/ true);
   std::vector<QueryRun> runs;
   runs.reserve(queries.size());

   double optimizationMs = 0;
   llvm::SmallVector<double, 8> queryCompileMs;
   queryCompileMs.reserve(queries.size());
   const bool printPlan = envFlagEnabled("LINGODB_REUSE_PRINT_PLAN");
   const bool printMatches = envFlagEnabled("LINGODB_REUSE_PRINT_MATCHES");
   const bool verboseTiming = envFlagEnabled("LINGODB_REUSE_VERBOSE_TIMING");

   for (size_t i = 0; i < queries.size(); i++) {
      std::string sql = queries[i];

      QueryRun run;
      run.frontend = lingodb::execution::createSQLFrontend();
      run.frontend->setCatalog(catalog.get());
      run.frontend->setContext(sharedMlirContext.get());
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

      if (printPlan) {
         llvm::outs() << "\n// ============================\n";
         llvm::outs() << "// query[" << i << "] subop layer\n";
         llvm::outs() << "// ============================\n";
         runs.back().module.print(llvm::outs());
         llvm::outs() << "\n";

         lingodb::compiler::dialect::subop::printExecutionSteps(runs.back().module, llvm::outs());
         llvm::outs() << "\n";
      }
   }

   // If we have at least two queries, detect matches and inject cache_put/cache_get to reuse states.
   llvm::SmallVector<std::pair<int, mlir::ModuleOp>, 8> qmods;
   for (size_t i = 0; i < runs.size(); i++) {
      qmods.push_back({static_cast<int>(i), runs[i].module});
   }
   llvm::SmallVector<lingodb::compiler::dialect::subop::CrossQueryStateMatchGroup, 64> groups;
   if (!skipReuseRewrite) groups = lingodb::compiler::dialect::subop::collectCrossQueryStateMatchGroups(qmods);
   if (printMatches) {
      lingodb::compiler::dialect::subop::printCrossQueryStateMatches(qmods, llvm::outs());
   }
   const char* dumpSubOpDir = std::getenv("LINGODB_DUMP_SUBOP_DIR");
   if (dumpSubOpDir) {
      const std::string base(dumpSubOpDir);
      auto dumpPlan = [&](mlir::ModuleOp mod, const llvm::Twine& scenarioDir) {
         const std::string dir = (base + "/" + scenarioDir).str();
         llvm::sys::fs::create_directories(dir);
         dumpModuleToFile(mod, dir + "/consumer-subop.mlir");
      };
      for (size_t i = 0; i < runs.size(); ++i) {
         dumpPlan(runs[i].module, "plan-query" + llvm::Twine(i));
      }
   }

   auto tRewrite = std::chrono::high_resolution_clock::now();
   lingodb::compiler::dialect::subop::BatchReusePlanRewriteResult rewriteRes;
   if (!skipReuseRewrite) {
      llvm::SmallVector<mlir::ModuleOp, 8> modules;
      for (auto& r : runs) modules.push_back(r.module);
      rewriteRes = lingodb::compiler::dialect::subop::rewritePlansWithSyntheticQueryBatch(
         modules, groups, catalog.get());
   } else {
      rewriteRes.numTargetsPerQuery.resize(runs.size(), 0);
      rewriteRes.numTargetsNoTablePerQuery.resize(runs.size(), 0);
   }
   const double rewriteMs = skipReuseRewrite ? 0.0 : millisSince(tRewrite);
   optimizationMs += rewriteMs;
   if (skipReuseRewrite) {
      llvm::outs() << "\n// reuse_rewrite: skipped (--no-reuse-rewrite)\n";
   }
   auto printSizeArray = [](llvm::raw_ostream& os, llvm::StringRef label, llvm::ArrayRef<size_t> values) {
      os << "\n// " << label << ": per_query=[";
      for (size_t i = 0; i < values.size(); ++i) {
         if (i) os << ",";
         os << values[i];
      }
      os << "]\n";
   };
   printSizeArray(llvm::outs(), "reuse_targets", rewriteRes.numTargetsPerQuery);
   printSizeArray(llvm::outs(), "reuse_targets_no_table", rewriteRes.numTargetsNoTablePerQuery);
   llvm::outs() << "\n// reuse_targets_synthetic_mapped: " << rewriteRes.numTargetsSyntheticMapped << "\n";
   llvm::outs() << "\n// reuse_targets_synthetic_mapped_no_table: " << rewriteRes.numTargetsSyntheticMappedNoTable << "\n";

   // With `LINGODB_DUMP_SUBOP_DIR`, replay lowering into `snapshots/` by default.
   // Set `LINGODB_DUMP_LOWERING=0` to write only `consumer-subop.mlir`.
   const char* dumpLoweringEnv = std::getenv("LINGODB_DUMP_LOWERING");
   const bool dumpLoweringSnapshots =
      dumpSubOpDir && (!dumpLoweringEnv || dumpLoweringEnv[0] == '\0' || envFlagEnabled("LINGODB_DUMP_LOWERING"));
   if (dumpSubOpDir) {
      auto dumpScenario = [&](mlir::ModuleOp mod, const llvm::Twine& outDir, llvm::StringRef scenarioName) {
         const std::string dir = outDir.str();
         llvm::sys::fs::create_directories(dir);
         dumpModuleToFile(mod, dir + "/consumer-subop.mlir");
         if (!dumpLoweringSnapshots) return true;
         mlir::OwningOpRef<mlir::ModuleOp> clone = mlir::cast<mlir::ModuleOp>(mod->clone());
         const bool ok = lowerFromSubOpLayer(*clone, dir.c_str(), /*snapshotLabel=*/{});
         llvm::errs() << "[dump] " << scenarioName << " lowering " << (ok ? "OK" : "FAILED") << "\n";
         return ok;
      };

      const std::string base(dumpSubOpDir);
      if (rewriteRes.synthetic && rewriteRes.numTargetsSyntheticMapped > 0) {
         dumpScenario(*rewriteRes.synthetic, base + "/rewrite-synthetic-batch", "rewrite-synthetic-batch");
      }
      for (size_t i = 0; i < runs.size(); ++i) {
         std::string scenarioName = ("rewrite-query" + llvm::Twine(i)).str();
         dumpScenario(runs[i].module, base + "/" + scenarioName, scenarioName);
      }
   }

   bool verifyFailed = false;
   for (auto& r : runs) verifyFailed |= mlir::failed(mlir::verify(r.module));
   verifyFailed |= (rewriteRes.synthetic && mlir::failed(mlir::verify(*rewriteRes.synthetic)));
   if (verifyFailed) {
      llvm::errs() << kToolName << ": MLIR verification failed after cross-query reuse rewrite"
                   << " (optimization_ms=" << optimizationMs << ")\n";
      return 1;
   }

   // Optional heavy debug printing (can be huge / sometimes crashes when IR is malformed).
   // Enable via env var: LINGODB_REUSE_PRINT_REWRITTEN=1
   const bool printRewritten = (std::getenv("LINGODB_REUSE_PRINT_REWRITTEN") != nullptr);
   if (printRewritten) {
      if (rewriteRes.numTargetsSyntheticMapped > 0) {
         assert(rewriteRes.synthetic);
         llvm::outs() << "\n// ============================\n";
         llvm::outs() << "// synthetic batch subop layer (post-rewrite)\n";
         llvm::outs() << "// ============================\n";
         (*rewriteRes.synthetic)->print(llvm::outs());
         llvm::outs() << "\n";
      }
      llvm::outs() << "\n// ============================\n";
      llvm::outs() << "// Step layout vs SubOpToControlFlow stderr ordinal\n";
      llvm::outs() << "// ============================\n";
      if (rewriteRes.synthetic && rewriteRes.numTargetsSyntheticMapped > 0) {
         llvm::outs() << "// synthetic batch — same indices as lowering uses for that module\n";
         lingodb::compiler::dialect::subop::printTopLevelExecutionStepLayout(*rewriteRes.synthetic, llvm::outs());
      }
      for (size_t i = 0; i < runs.size(); i++) {
         llvm::outs() << "\n// --- query[" << i << "] execution_step layout (post-rewrite IR) ---\n";
         lingodb::compiler::dialect::subop::printTopLevelExecutionStepLayout(runs[i].module, llvm::outs());
      }

      for (size_t i = 0; i < runs.size(); i++) {
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
      llvm::outs() << "\n// timing: optimization_ms=" << optimizationMs << " execution_time_ms=0 (skipped)\n";
      return 0;
   }

   SubOpExecuteTiming totalExec;
   SubOpExecuteTiming segmentSynthetic;
   llvm::SmallVector<SubOpExecuteTiming, 8> consumerSegments(runs.size());
   llvm::SmallVector<SubOpExecuteTiming, 8> timingPerRun;

   // `putCachedState` stores raw pointers into memory registered on the **allocating**
   // `ExecutionContext`. Destroying that context before consumers run leaves entries in the
   // process-global `gCachedStates` map dangling. Use one context for synthetic + both consumers,
   // then clear the global map after it goes out of scope.
   {
      lingodb::runtime::ExecutionContext::clearAllCachedStates();
      auto sharedExecCtx = session->createExecutionContext();
      auto runOne = [&](mlir::ModuleOp mod, const std::string& label, SubOpExecuteTiming& segment,
                        std::optional<unsigned> resultQueryIndex) {
         if (verboseTiming) {
            llvm::outs() << "\n// ============================\n";
            llvm::outs() << label << "\n";
            llvm::outs() << "// ============================\n";
            llvm::outs().flush();
         }
         mlir::OwningOpRef<mlir::ModuleOp> execModule = mlir::cast<mlir::ModuleOp>(mod->clone());
         SubOpExecuteTiming one = executeFromSubOpLayer(*execModule, sharedExecCtx.get(), resultQueryIndex);
         segment = one;
         totalExec.add(one);
         timingPerRun.push_back(one);
         // TablePrinter uses std::cout; flush before timing on llvm::outs to preserve log order.
         std::cout.flush();
         if (verboseTiming) {
            llvm::outs() << "// timing_run executionTime_ms=" << one.executionTime << " lower_ms=" << one.lowerMs
                         << " llvm_jit_ms=" << one.llvmJitMs() << " execute_wall_ms=" << one.wallMs << "\n";
            llvm::outs().flush();
            llvm::outs() << "\n";
         }
      };
      // When cross-query rewrite could not map any donor state into the synthetic module, it only
      // contains an empty execution_group shell — skip JIT for that shell.
      if (rewriteRes.numTargetsSyntheticMapped > 0) {
         runOne(*rewriteRes.synthetic, "// synthetic batch execute", segmentSynthetic, std::nullopt);
      } else {
         if (verboseTiming) llvm::outs() << "\n// (skip synthetic execute: reuse_targets_synthetic_mapped==0)\n";
      }
      for (size_t i = 0; i < runs.size(); i++) {
         SubOpExecuteTiming& seg = consumerSegments[i];
         runOne(runs[i].module, "// query[" + std::to_string(i) + "] execute", seg, static_cast<unsigned>(i));
      }
   }
   lingodb::runtime::ExecutionContext::clearAllCachedStates();

   auto printPerRun = [&](llvm::StringRef key, auto getter) {
      llvm::outs() << "// timing_" << key << "_ms: per_run=[";
      for (size_t i = 0; i < timingPerRun.size(); i++) {
         if (i) llvm::outs() << ",";
         llvm::outs() << getter(timingPerRun[i]);
      }
      llvm::outs() << "] total=" << getter(totalExec) << "\n";
   };

   llvm::outs() << "\n// timing_compile_ms: per_query=[";
   for (size_t i = 0; i < queryCompileMs.size(); i++) {
      if (i) llvm::outs() << ",";
      llvm::outs() << queryCompileMs[i];
   }
   llvm::outs() << "] rewrite=" << rewriteMs << " total_optimization_ms=" << optimizationMs << "\n";

   printPerRun("execution_time", [](const SubOpExecuteTiming& t) { return t.executionTime; });

   SubOpExecuteTiming consumersOnly;
   for (const SubOpExecuteTiming& seg : consumerSegments) consumersOnly.add(seg);
   if (verboseTiming) {
      llvm::outs() << "// timing_note: executionTime_ms is run-sql `executionTime` (generated main() only).\n";
      llvm::outs() << "// timing_note: shared ExecutionContext keeps cache_put pointers valid across runs;\n";
      llvm::outs() << "// timing_note: does not change executionTime; per-run clearResult(0) only. Arena/state\n";
      llvm::outs() << "// timing_note: may accumulate on the shared context (memory, not timing).\n";

      printPerRun("lower_imperative", [](const SubOpExecuteTiming& t) { return t.lowerMs; });
      printPerRun("llvm_jit", [](const SubOpExecuteTiming& t) { return t.llvmJitMs(); });
      printPerRun("execute_wall", [](const SubOpExecuteTiming& t) { return t.wallMs; });

      if (rewriteRes.numTargetsSyntheticMapped > 0) {
         printTimingSegment(llvm::outs(), "synthetic_ir", segmentSynthetic);
      }
      for (size_t i = 0; i < consumerSegments.size(); ++i) {
         std::string segmentName = ("consumer_q" + llvm::Twine(i) + "_ir").str();
         printTimingSegment(llvm::outs(), segmentName, consumerSegments[i]);
      }
      printTimingSegment(llvm::outs(), "consumers_only_ir", consumersOnly);
      printTimingSegment(llvm::outs(), "all_execute_runs", totalExec);
   }

   llvm::outs() << "// timing: optimization_ms=" << optimizationMs
                << " execution_time_ms=" << totalExec.executionTime << "\n";
   return 0;
}
