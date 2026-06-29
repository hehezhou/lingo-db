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
#include "lingodb/runtime/ArrowTable.h"
#include "lingodb/runtime/ExecutionContext.h"
#include "lingodb/runtime/Session.h"
#include "lingodb/runtime/Tracing.h"
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

#include <arrow/pretty_print.h>
#include <arrow/table.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

#include "md5.h"

#include <algorithm>
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

unsigned char hexval(unsigned char c) {
   if ('0' <= c && c <= '9')
      return c - '0';
   if ('a' <= c && c <= 'f')
      return c - 'a' + 10;
   if ('A' <= c && c <= 'F')
      return c - 'A' + 10;
   abort();
}

static std::string hashArrowTableRows(const std::shared_ptr<arrow::Table>& table) {
   std::vector<std::string> toHash;
   std::vector<std::string> columnReps;
   std::vector<size_t> positions;
   arrow::PrettyPrintOptions options;
   options.indent_size = 0;
   options.window = 1000000;
   options.container_window = 1000000;
   options.element_size_limit = 10000;
   std::vector<bool> convertHex;
   std::vector<bool> isFloat;
   for (auto c : table->columns()) {
      convertHex.push_back(table->schema()->field(positions.size())->type()->id() == arrow::Type::FIXED_SIZE_BINARY);
      isFloat.push_back(table->schema()->field(positions.size())->type()->id() == arrow::Type::DOUBLE);
      std::stringstream sstr;
      [[maybe_unused]] auto status = arrow::PrettyPrint(*c.get(), options, &sstr);
      columnReps.push_back(sstr.str());
      positions.push_back(0);
   }

   bool cont = true;
   while (cont) {
      cont = false;
      for (size_t column = 0; column < columnReps.size(); column++) {
         char32_t currChar = U'\0';
         uint8_t currCharSize = 0;

         bool first = true;
         bool afterComma = false;
         size_t digits = 0;
         std::stringstream out;
         while (positions[column] < columnReps[column].size()) {
            cont = true;
            char curr = columnReps[column][positions[column]];
            char next = columnReps[column][positions[column] + 1];
            positions[column]++;
            if (first && (curr == '[' || curr == ']' || curr == ',')) continue;
            if (curr == ',' && next == '\n') continue;
            if (curr == '\n') break;
            if (isFloat[column]) {
               if (std::isdigit(static_cast<unsigned char>(curr))) {
                  if (afterComma && digits < 3) {
                     digits++;
                     out << curr;
                  } else if (!afterComma) {
                     out << curr;
                     first = false;
                  }
               } else if (curr == '.') {
                  afterComma = true;
                  out << curr;
                  digits = 0;
               } else {
                  afterComma = false;
                  digits = 0;
                  first = false;
                  out << curr;
               }
            } else if (convertHex[column]) {
               first = false;
               if (std::isxdigit(static_cast<unsigned char>(curr))) {
                  if (currCharSize % 2 == 0)
                     currChar |= hexval(curr) << (currCharSize++ * 4 + 4);
                  else
                     currChar |= hexval(curr) << (currCharSize++ * 4 - 4);
               } else {
                  out << curr;
               }
            } else {
               first = false;
               out << curr;
            }
         }
         if (currChar != U'\0') {
            assert(currChar <= 0xFF && "Only ASCII characters supported for result hashing");
            out << static_cast<char>(currChar);
         }
         if (!first) toHash.push_back(out.str());
      }
   }
   for (std::string& s : toHash) {
      if (s == "null") s = "NULL";
      if (s == "true") s = "t";
      if (s == "false") s = "f";
      if (s.starts_with("\"") && s.ends_with("\"")) s = s.substr(1, s.size() - 2);
   }
   const size_t numColumns = table->num_columns();
   std::vector<std::vector<std::string>> rows;
   std::vector<std::string> row;
   for (auto& s : toHash) {
      row.push_back(s);
      if (row.size() == numColumns) {
         rows.push_back(row);
         row.clear();
      }
   }
   std::sort(rows.begin(), rows.end());
   toHash.clear();
   for (auto& r : rows)
      for (auto& v : r) toHash.push_back(v);
   return md5Strings(toHash);
}

static std::string hashStrings(llvm::ArrayRef<std::string> values) {
   std::vector<std::string> copy(values.begin(), values.end());
   return md5Strings(copy);
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
      [[maybe_unused]] bool ok = succeeded(qoptPm.run(moduleOp));
      assert(ok);
   }

   // 2) Lower RelAlg -> SubOp.
   {
      mlir::PassManager lowerRelAlgPm(moduleOp.getContext());
      lowerRelAlgPm.enableVerifier(true);
      relalg::createLowerRelAlgToSubOpPipeline(lowerRelAlgPm);
      [[maybe_unused]] bool ok = succeeded(lowerRelAlgPm.run(moduleOp));
      assert(ok);
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

      [[maybe_unused]] bool ok = succeeded(optSubOpPm.run(moduleOp));
      assert(ok);
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
      snap("before-lower-subop", subopModule);
      mlir::PassManager lowerSubOpPm(subopModule.getContext());
      lowerSubOpPm.enableVerifier(true);
      lowerSubOpPm.addPass(subop::createLowerSubOpPass());
      lowerSubOpPm.addPass(mlir::createCSEPass());
      if (failed(lowerSubOpPm.run(subopModule))) {
         snap("lower-subop-failed", subopModule);
         return false;
      }
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
                                                std::optional<unsigned> resultQueryIndex = std::nullopt,
                                                std::vector<std::string>* resultRowsetHashes = nullptr) {
   using namespace lingodb::compiler::dialect;

   auto wallStart = std::chrono::high_resolution_clock::now();
   SubOpExecuteTiming out;

   auto lowerStart = std::chrono::high_resolution_clock::now();
   if (!lowerFromSubOpLayer(subopModule)) {
      subopModule.dump();
      llvm_unreachable("lowerFromSubOpLayer failed (dumped module above)");
   }
   out.lowerMs = millisSince(lowerStart);

   // Same backend as run-sql DEFAULT mode (`createDefaultLLVMBackend()` → optimize=true).
   auto backend = std::shared_ptr<lingodb::execution::ExecutionBackend>(
      lingodb::execution::createDefaultLLVMBackend(/*optimize*/ true).release());
   std::shared_ptr<::arrow::Table> retrievedTable;
   auto resultProcessor = std::shared_ptr<lingodb::execution::ResultProcessor>(
      resultRowsetHashes ? lingodb::execution::createTableRetriever(retrievedTable).release()
                         : lingodb::execution::createTablePrinter().release());

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
      if (lingodb::runtime::RuntimeScanCpuProfiler::enabled()) {
         lingodb::runtime::RuntimeScanCpuProfiler::reset();
      }
      uint64_t processCpuBegin = lingodb::runtime::RuntimeScanCpuProfiler::enabled()
                                    ? lingodb::runtime::RuntimeScanCpuProfiler::processCpuNs()
                                    : 0;
      backend->execute(subopModule, executionContext);
      if (lingodb::runtime::RuntimeScanCpuProfiler::enabled()) {
         uint64_t processCpuEnd = lingodb::runtime::RuntimeScanCpuProfiler::processCpuNs();
         uint64_t totalCpuNs = processCpuEnd >= processCpuBegin ? processCpuEnd - processCpuBegin : 0;
         uint64_t scanCpuNs = lingodb::runtime::RuntimeScanCpuProfiler::scanCpuNs();
         double totalCpuMs = static_cast<double>(totalCpuNs) / 1000000.0;
         double scanCpuMs = static_cast<double>(scanCpuNs) / 1000000.0;
         double pct = totalCpuNs ? (static_cast<double>(scanCpuNs) * 100.0 / static_cast<double>(totalCpuNs)) : 0.0;
         llvm::outs() << "// scan_cpu_profile: scan_cpu_ms=" << scanCpuMs
                      << " total_cpu_ms=" << totalCpuMs
                      << " percent=" << pct
                      << " scan_calls=" << lingodb::runtime::RuntimeScanCpuProfiler::scanCalls()
                      << "\n";
         llvm::outs().flush();
      }
      if (resultQueryIndex && !resultRowsetHashes) {
         std::cout << "// result_begin: query[" << *resultQueryIndex << "]\n";
         std::cout.flush();
      }
      resultProcessor->process(executionContext);
      if (resultRowsetHashes && retrievedTable) {
         resultRowsetHashes->push_back(hashArrowTableRows(retrievedTable));
      }
      if (resultQueryIndex && !resultRowsetHashes) {
         std::cout << "// result_end: query[" << *resultQueryIndex << "]\n";
      }
      std::cout.flush();
   }));

   out.add(SubOpExecuteTiming::fromBackend(backend->getTiming()));
   out.wallMs = millisSince(wallStart);
   return out;
}

struct SubOpStateReuseOptions {
   bool skipReuseRewrite = false;
   bool skipExecute = false;
   bool captureResultHashes = false;
   bool printPlan = false;
   bool printMatches = false;
   bool printRewritten = false;
   bool verboseTiming = false;
   const char* dumpSubOpDir = nullptr;
   bool dumpLoweringSnapshots = false;
};

struct SubOpStateReuseBatchResult {
   int exitCode = 0;
   double optimizationMs = 0;
   double rewriteMs = 0;
   llvm::SmallVector<double, 8> queryCompileMs;
   SubOpExecuteTiming totalExec;
   SubOpExecuteTiming segmentSynthetic;
   llvm::SmallVector<SubOpExecuteTiming, 8> consumerSegments;
   llvm::SmallVector<SubOpExecuteTiming, 8> timingPerRun;
   llvm::SmallVector<size_t, 8> numTargetsPerQuery;
   llvm::SmallVector<size_t, 8> numTargetsNoTablePerQuery;
   llvm::SmallVector<size_t, 8> numUnionTargetsPerQuery;
   llvm::SmallVector<size_t, 8> numUnionTargetsNoTablePerQuery;
   llvm::SmallVector<size_t, 8> numBuildStepTargetsPerQuery;
   llvm::SmallVector<size_t, 8> numBuildStepTargetsNoTablePerQuery;
   size_t numTargetsSyntheticMapped = 0;
   size_t numTargetsSyntheticMappedNoTable = 0;
   size_t numUnionTargetsSyntheticMapped = 0;
   size_t numUnionTargetsSyntheticMappedNoTable = 0;
   size_t numBuildStepTargetsSyntheticMapped = 0;
   size_t numBuildStepTargetsSyntheticMappedNoTable = 0;
   std::vector<std::string> resultRowsetBlockHashes;
   bool verifyFailed = false;
};

struct QueryRun {
   std::unique_ptr<lingodb::execution::Frontend> frontend;
   mlir::ModuleOp module;
};

struct CachedStateCleanupGuard {
   CachedStateCleanupGuard() { lingodb::runtime::ExecutionContext::clearAllCachedStates(); }
   ~CachedStateCleanupGuard() { lingodb::runtime::ExecutionContext::clearAllCachedStates(); }
};

static SubOpStateReuseBatchResult runSubOpStateReuseBatch(
   llvm::ArrayRef<std::string> queries,
   lingodb::catalog::Catalog* catalog,
   lingodb::runtime::Session* session,
   const SubOpStateReuseOptions& opts) {
   assert(queries.size() >= 2 && "subop-state-reuse batch input must contain at least two queries");

   SubOpStateReuseBatchResult result;
   auto sharedMlirContext = std::make_unique<mlir::MLIRContext>();
   lingodb::execution::initializeContext(*sharedMlirContext, /*includeLLVM*/ true);
   std::vector<QueryRun> runs;
   runs.reserve(queries.size());
   result.queryCompileMs.reserve(queries.size());

   for (size_t i = 0; i < queries.size(); i++) {
      std::string sql = queries[i];

      QueryRun run;
      run.frontend = lingodb::execution::createSQLFrontend();
      run.frontend->setCatalog(catalog);
      run.frontend->setContext(sharedMlirContext.get());
      run.frontend->loadFromString(sql);

      auto& feErr = run.frontend->getError();
      (void)feErr;
      assert(!feErr);

      mlir::ModuleOp* moduleOpPtr = run.frontend->getModule();
      assert(moduleOpPtr);
      run.module = *moduleOpPtr;

      {
         auto t0 = std::chrono::high_resolution_clock::now();
         runPasses(run.module, catalog);
         const double ms = millisSince(t0);
         result.queryCompileMs.push_back(ms);
         result.optimizationMs += ms;
      }
      runs.push_back(std::move(run));

      if (opts.printPlan) {
         llvm::outs() << "\n// ============================\n";
         llvm::outs() << "// query[" << i << "] subop layer\n";
         llvm::outs() << "// ============================\n";
         runs.back().module.print(llvm::outs());
         llvm::outs() << "\n";

         lingodb::compiler::dialect::subop::printExecutionSteps(runs.back().module, llvm::outs());
         llvm::outs() << "\n";
      }
   }

   llvm::SmallVector<std::pair<int, mlir::ModuleOp>, 8> qmods;
   for (size_t i = 0; i < runs.size(); i++) {
      qmods.push_back({static_cast<int>(i), runs[i].module});
   }
   llvm::SmallVector<lingodb::compiler::dialect::subop::CrossQueryStateMatchGroup, 64> groups;
   if (!opts.skipReuseRewrite) groups = lingodb::compiler::dialect::subop::collectCrossQueryStateMatchGroups(qmods);
   if (opts.printMatches) {
      lingodb::compiler::dialect::subop::printCrossQueryStateMatches(qmods, llvm::outs());
      llvm::outs().flush();
   }

   if (opts.dumpSubOpDir) {
      const std::string base(opts.dumpSubOpDir);
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
   if (!opts.skipReuseRewrite) {
      llvm::SmallVector<mlir::ModuleOp, 8> modules;
      for (auto& r : runs) modules.push_back(r.module);
      rewriteRes = lingodb::compiler::dialect::subop::rewritePlansWithSyntheticQueryBatch(
         modules, groups, catalog);
   } else {
      rewriteRes.numTargetsPerQuery.resize(runs.size(), 0);
      rewriteRes.numTargetsNoTablePerQuery.resize(runs.size(), 0);
      rewriteRes.numUnionTargetsPerQuery.resize(runs.size(), 0);
      rewriteRes.numUnionTargetsNoTablePerQuery.resize(runs.size(), 0);
      rewriteRes.numBuildStepTargetsPerQuery.resize(runs.size(), 0);
      rewriteRes.numBuildStepTargetsNoTablePerQuery.resize(runs.size(), 0);
   }
   result.numTargetsPerQuery = rewriteRes.numTargetsPerQuery;
   result.numTargetsNoTablePerQuery = rewriteRes.numTargetsNoTablePerQuery;
   result.numUnionTargetsPerQuery = rewriteRes.numUnionTargetsPerQuery;
   result.numUnionTargetsNoTablePerQuery = rewriteRes.numUnionTargetsNoTablePerQuery;
   result.numBuildStepTargetsPerQuery = rewriteRes.numBuildStepTargetsPerQuery;
   result.numBuildStepTargetsNoTablePerQuery = rewriteRes.numBuildStepTargetsNoTablePerQuery;
   result.numTargetsSyntheticMapped = rewriteRes.numTargetsSyntheticMapped;
   result.numTargetsSyntheticMappedNoTable = rewriteRes.numTargetsSyntheticMappedNoTable;
   result.numUnionTargetsSyntheticMapped = rewriteRes.numUnionTargetsSyntheticMapped;
   result.numUnionTargetsSyntheticMappedNoTable = rewriteRes.numUnionTargetsSyntheticMappedNoTable;
   result.numBuildStepTargetsSyntheticMapped = rewriteRes.numBuildStepTargetsSyntheticMapped;
   result.numBuildStepTargetsSyntheticMappedNoTable = rewriteRes.numBuildStepTargetsSyntheticMappedNoTable;
   result.rewriteMs = opts.skipReuseRewrite ? 0.0 : millisSince(tRewrite);
   result.optimizationMs += result.rewriteMs;
   if (opts.skipReuseRewrite) {
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
   printSizeArray(llvm::outs(), "reuse_targets_union", rewriteRes.numUnionTargetsPerQuery);
   printSizeArray(llvm::outs(), "reuse_targets_union_no_table", rewriteRes.numUnionTargetsNoTablePerQuery);
   printSizeArray(llvm::outs(), "reuse_targets_build_step", rewriteRes.numBuildStepTargetsPerQuery);
   printSizeArray(llvm::outs(), "reuse_targets_build_step_no_table",
                  rewriteRes.numBuildStepTargetsNoTablePerQuery);
   llvm::outs() << "\n// reuse_targets_synthetic_mapped: " << rewriteRes.numTargetsSyntheticMapped << "\n";
   llvm::outs() << "\n// reuse_targets_synthetic_mapped_no_table: "
                << rewriteRes.numTargetsSyntheticMappedNoTable << "\n";
   llvm::outs() << "\n// reuse_targets_synthetic_mapped_union: "
                << rewriteRes.numUnionTargetsSyntheticMapped << "\n";
   llvm::outs() << "\n// reuse_targets_synthetic_mapped_union_no_table: "
                << rewriteRes.numUnionTargetsSyntheticMappedNoTable << "\n";
   llvm::outs() << "\n// reuse_targets_synthetic_mapped_build_step: "
                << rewriteRes.numBuildStepTargetsSyntheticMapped << "\n";
   llvm::outs() << "\n// reuse_targets_synthetic_mapped_build_step_no_table: "
                << rewriteRes.numBuildStepTargetsSyntheticMappedNoTable << "\n";
   if (opts.printMatches && !opts.skipReuseRewrite) {
      llvm::SmallVector<std::pair<int, mlir::ModuleOp>, 8> postRewriteQmods;
      for (size_t i = 0; i < runs.size(); i++) {
         postRewriteQmods.push_back({static_cast<int>(i), runs[i].module});
      }
      llvm::outs() << "\n// ==== post-rewrite cross-query state matches ====\n";
      lingodb::compiler::dialect::subop::printCrossQueryStateMatches(postRewriteQmods, llvm::outs());
   }

   if (opts.dumpSubOpDir) {
      auto dumpScenario = [&](mlir::ModuleOp mod, const llvm::Twine& outDir, llvm::StringRef scenarioName) {
         const std::string dir = outDir.str();
         llvm::sys::fs::create_directories(dir);
         dumpModuleToFile(mod, dir + "/consumer-subop.mlir");
         if (!opts.dumpLoweringSnapshots) return true;
         mlir::OwningOpRef<mlir::ModuleOp> clone = mlir::cast<mlir::ModuleOp>(mod->clone());
         const bool ok = lowerFromSubOpLayer(*clone, dir.c_str(), /*snapshotLabel=*/{});
         llvm::errs() << "[dump] " << scenarioName << " lowering " << (ok ? "OK" : "FAILED") << "\n";
         return ok;
      };

      const std::string base(opts.dumpSubOpDir);
      if (rewriteRes.synthetic && rewriteRes.numTargetsSyntheticMapped > 0) {
         dumpScenario(*rewriteRes.synthetic, base + "/rewrite-synthetic-batch", "rewrite-synthetic-batch");
      }
      for (size_t i = 0; i < runs.size(); ++i) {
         std::string scenarioName = ("rewrite-query" + llvm::Twine(i)).str();
         dumpScenario(runs[i].module, base + "/" + scenarioName, scenarioName);
      }
   }

   for (size_t i = 0; i < runs.size(); ++i) {
      if (mlir::failed(mlir::verify(runs[i].module))) {
         llvm::errs() << kToolName << ": MLIR verification failed in rewritten query[" << i << "]\n";
         result.verifyFailed = true;
      }
   }
   if (rewriteRes.synthetic && mlir::failed(mlir::verify(*rewriteRes.synthetic))) {
      llvm::errs() << kToolName << ": MLIR verification failed in rewritten synthetic query\n";
      result.verifyFailed = true;
   }
   if (result.verifyFailed) {
      llvm::errs() << kToolName << ": MLIR verification failed after cross-query reuse rewrite"
                   << " (optimization_ms=" << result.optimizationMs << ")\n";
      result.exitCode = 1;
      return result;
   }

   if (opts.printRewritten) {
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

   if (opts.skipExecute) {
      llvm::outs() << "\n// (LINGODB_SKIP_EXECUTE set: skipping JIT execution)\n";
      llvm::outs() << "\n// timing: optimization_ms=" << result.optimizationMs
                   << " execution_time_ms=0 (skipped)\n";
      return result;
   }

   result.consumerSegments.resize(runs.size());
   {
      CachedStateCleanupGuard cachedStateCleanup;
      auto sharedExecCtx = session->createExecutionContext();
      auto runOne = [&](mlir::ModuleOp mod, const std::string& label, SubOpExecuteTiming& segment,
                        std::optional<unsigned> resultQueryIndex) {
         if (opts.verboseTiming) {
            llvm::outs() << "\n// ============================\n";
            llvm::outs() << label << "\n";
            llvm::outs() << "// ============================\n";
            llvm::outs().flush();
         }
         mlir::OwningOpRef<mlir::ModuleOp> execModule = mlir::cast<mlir::ModuleOp>(mod->clone());
         SubOpExecuteTiming one = executeFromSubOpLayer(
            *execModule, sharedExecCtx.get(), resultQueryIndex,
            opts.captureResultHashes && resultQueryIndex ? &result.resultRowsetBlockHashes : nullptr);
         segment = one;
         result.totalExec.add(one);
         result.timingPerRun.push_back(one);
         std::cout.flush();
         if (opts.verboseTiming) {
            llvm::outs() << "// timing_run executionTime_ms=" << one.executionTime << " lower_ms=" << one.lowerMs
                         << " llvm_jit_ms=" << one.llvmJitMs() << " execute_wall_ms=" << one.wallMs << "\n";
            llvm::outs().flush();
            llvm::outs() << "\n";
         }
      };
      if (rewriteRes.numTargetsSyntheticMapped > 0) {
         runOne(*rewriteRes.synthetic, "// synthetic batch execute", result.segmentSynthetic, std::nullopt);
      } else {
         if (opts.verboseTiming) llvm::outs() << "\n// (skip synthetic execute: reuse_targets_synthetic_mapped==0)\n";
      }
      for (size_t i = 0; i < runs.size(); i++) {
         SubOpExecuteTiming& seg = result.consumerSegments[i];
         runOne(runs[i].module, "// query[" + std::to_string(i) + "] execute", seg, static_cast<unsigned>(i));
      }
   }

   auto printPerRun = [&](llvm::StringRef key, auto getter) {
      llvm::outs() << "// timing_" << key << "_ms: per_run=[";
      for (size_t i = 0; i < result.timingPerRun.size(); i++) {
         if (i) llvm::outs() << ",";
         llvm::outs() << getter(result.timingPerRun[i]);
      }
      llvm::outs() << "] total=" << getter(result.totalExec) << "\n";
   };

   llvm::outs() << "\n// timing_compile_ms: per_query=[";
   for (size_t i = 0; i < result.queryCompileMs.size(); i++) {
      if (i) llvm::outs() << ",";
      llvm::outs() << result.queryCompileMs[i];
   }
   llvm::outs() << "] rewrite=" << result.rewriteMs
                << " total_optimization_ms=" << result.optimizationMs << "\n";

   printPerRun("execution_time", [](const SubOpExecuteTiming& t) { return t.executionTime; });

   SubOpExecuteTiming consumersOnly;
   for (const SubOpExecuteTiming& seg : result.consumerSegments) consumersOnly.add(seg);
   if (opts.verboseTiming) {
      llvm::outs() << "// timing_note: executionTime_ms is run-sql `executionTime` (generated main() only).\n";
      llvm::outs() << "// timing_note: shared ExecutionContext keeps cache_put pointers valid across runs;\n";
      llvm::outs() << "// timing_note: does not change executionTime; per-run clearResult(0) only. Arena/state\n";
      llvm::outs() << "// timing_note: may accumulate on the shared context (memory, not timing).\n";

      printPerRun("lower_imperative", [](const SubOpExecuteTiming& t) { return t.lowerMs; });
      printPerRun("llvm_jit", [](const SubOpExecuteTiming& t) { return t.llvmJitMs(); });
      printPerRun("execute_wall", [](const SubOpExecuteTiming& t) { return t.wallMs; });

      if (rewriteRes.numTargetsSyntheticMapped > 0) {
         printTimingSegment(llvm::outs(), "synthetic_ir", result.segmentSynthetic);
      }
      for (size_t i = 0; i < result.consumerSegments.size(); ++i) {
         std::string segmentName = ("consumer_q" + llvm::Twine(i) + "_ir").str();
         printTimingSegment(llvm::outs(), segmentName, result.consumerSegments[i]);
      }
      printTimingSegment(llvm::outs(), "consumers_only_ir", consumersOnly);
      printTimingSegment(llvm::outs(), "all_execute_runs", result.totalExec);
   }

   llvm::outs() << "// timing: optimization_ms=" << result.optimizationMs
                << " execution_time_ms=" << result.totalExec.executionTime << "\n";
   return result;
}

static llvm::SmallVector<int, 16> parseIntList(llvm::StringRef spec) {
   llvm::SmallVector<int, 16> out;
   llvm::SmallVector<llvm::StringRef, 16> parts;
   spec.split(parts, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
   for (llvm::StringRef p : parts) {
      p = p.trim();
      auto range = p.split('-');
      if (!range.second.empty()) {
         int lo = 0;
         int hi = 0;
         if (range.first.trim().getAsInteger(10, lo) || range.second.trim().getAsInteger(10, hi) || lo > hi)
            llvm::report_fatal_error("invalid integer range list argument");
         for (int v = lo; v <= hi; ++v) out.push_back(v);
         continue;
      }
      int v = 0;
      if (p.getAsInteger(10, v))
         llvm::report_fatal_error("invalid integer list argument");
      out.push_back(v);
   }
   return out;
}

static std::string dbLabel(llvm::StringRef db) {
   return llvm::sys::path::filename(db).str();
}

static std::string readSqlFile(const llvm::Twine& path) {
   auto fileOrErr = llvm::MemoryBuffer::getFile(path.str());
   assert(!fileOrErr.getError());
   return (*fileOrErr)->getBuffer().str();
}

static llvm::json::Array jsonSizeArray(llvm::ArrayRef<size_t> values) {
   llvm::json::Array arr;
   for (size_t v : values) arr.push_back(static_cast<int64_t>(v));
   return arr;
}

static llvm::json::Array jsonStringArray(llvm::ArrayRef<std::string> values) {
   llvm::json::Array arr;
   for (const std::string& v : values) arr.push_back(v);
   return arr;
}

static void writeJsonFile(const llvm::Twine& path, const llvm::json::Value& value) {
   std::error_code ec;
   llvm::raw_fd_ostream os(path.str(), ec);
   assert(!ec && "failed to open json output");
   os << llvm::formatv("{0:2}", value) << "\n";
}

enum class BatchNightlyMode {
   ReuseOff,
   ReuseOnBloomOn,
   ReuseOnBloomOff,
   ReuseOnBloomOnNoHivDisjoint,
   ReuseOnBloomOnNoAggregateDisjoint,
   ReuseOnBloomOnNoDisjoint,
};

static BatchNightlyMode parseBatchNightlyMode(llvm::StringRef mode) {
   if (mode == "reuse_off") return BatchNightlyMode::ReuseOff;
   if (mode == "reuse_on_bloom_on") return BatchNightlyMode::ReuseOnBloomOn;
   if (mode == "reuse_on_bloom_off") return BatchNightlyMode::ReuseOnBloomOff;
   if (mode == "reuse_on_bloom_on_no_hiv_disjoint") return BatchNightlyMode::ReuseOnBloomOnNoHivDisjoint;
   if (mode == "reuse_on_bloom_on_no_aggregate_disjoint")
      return BatchNightlyMode::ReuseOnBloomOnNoAggregateDisjoint;
   if (mode == "reuse_on_bloom_on_no_disjoint") return BatchNightlyMode::ReuseOnBloomOnNoDisjoint;
   llvm_unreachable("unknown batch nightly mode");
}

static void setEnvFlag(const char* name, bool enabled) {
   if (enabled)
      setenv(name, "1", /*overwrite=*/1);
   else
      unsetenv(name);
}

static int runBatchNightlyMain(int argc, char** argv) {
   assert(argc >= 4 && "usage: subop-state-reuse --batch-nightly <db_dir> --mode <mode> [options]");
   std::string dbDir = argv[2];
   std::string mode;
   std::string queryDir = "queries";
   std::string outPath;
   std::string templatesSpec = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22";
   std::string batchSizesSpec = "2,4,8,16,32,64,128";
   int repetitions = 5;
   bool forceSequential = false;

   for (int i = 3; i < argc; i++) {
      llvm::StringRef arg(argv[i]);
      auto requireValue = [&]() -> const char* {
         assert(i + 1 < argc && "missing option value");
         return argv[++i];
      };
      if (arg == "--mode") mode = requireValue();
      else if (arg == "--query-dir") queryDir = requireValue();
      else if (arg == "--out") outPath = requireValue();
      else if (arg == "--templates") templatesSpec = requireValue();
      else if (arg == "--batch-sizes") batchSizesSpec = requireValue();
      else if (arg == "--repetitions") {
         llvm::StringRef v(requireValue());
         bool bad = v.getAsInteger(10, repetitions);
         (void)bad;
         assert(!bad && "invalid repetitions");
      } else if (arg == "--force-sequential") {
         forceSequential = true;
      } else if (arg == "--no-force-sequential") {
         forceSequential = false;
      } else {
         llvm_unreachable("unknown --batch-nightly option");
      }
   }
   assert(!mode.empty() && "--mode is required");
   assert(!outPath.empty() && "--out is required");
   BatchNightlyMode parsedMode = parseBatchNightlyMode(mode);

   setEnvFlag("LINGODB_SUBOP_FORCE_SEQUENTIAL", forceSequential);
   setEnvFlag("LINGODB_DISABLE_FILTER_PRED_BLOOM_ADAPTATION",
              parsedMode == BatchNightlyMode::ReuseOnBloomOff);
   setEnvFlag("LINGODB_DISABLE_HIV_DISJOINT_CLUSTERING",
              parsedMode == BatchNightlyMode::ReuseOnBloomOnNoHivDisjoint ||
                 parsedMode == BatchNightlyMode::ReuseOnBloomOnNoDisjoint);
   setEnvFlag("LINGODB_DISABLE_AGGREGATE_DISJOINT_CLUSTERING",
              parsedMode == BatchNightlyMode::ReuseOnBloomOnNoAggregateDisjoint ||
                 parsedMode == BatchNightlyMode::ReuseOnBloomOnNoDisjoint);

   llvm::SmallVector<int, 16> templates = parseIntList(templatesSpec);
   llvm::SmallVector<int, 16> batchSizes = parseIntList(batchSizesSpec);

   lingodb::compiler::support::eval::init();
   auto schedulerHandle = lingodb::scheduler::startScheduler(/*numWorkers*/ 0);
   std::shared_ptr<lingodb::catalog::Catalog> catalog =
      lingodb::catalog::Catalog::create(dbDir, /*eagerLoading*/ true);
   auto session = lingodb::runtime::Session::createSession(dbDir, /*eagerLoading*/ true);

   llvm::json::Array records;
   llvm::json::Array templateJson;
   for (int v : templates) templateJson.push_back(v);
   llvm::json::Array batchJson;
   for (int v : batchSizes) batchJson.push_back(v);

   const size_t totalCases = templates.size() * batchSizes.size() * static_cast<size_t>(repetitions);
   size_t completed = 0;
   auto allStart = std::chrono::high_resolution_clock::now();

   for (int templ : templates) {
      for (int batchSize : batchSizes) {
         llvm::SmallVector<std::string, 128> queries;
         queries.reserve(batchSize);
         for (int i = 1; i <= batchSize; i++) {
            llvm::SmallString<256> path;
            llvm::sys::path::append(path, queryDir, ("q" + llvm::Twine(templ) + "_" + llvm::Twine(i) + ".sql").str());
            queries.push_back(readSqlFile(path));
         }
         for (int rep = 0; rep < repetitions; rep++) {
            SubOpStateReuseOptions opts;
            opts.skipReuseRewrite = parsedMode == BatchNightlyMode::ReuseOff;
            opts.skipExecute = (std::getenv("LINGODB_SKIP_EXECUTE") != nullptr);
            opts.captureResultHashes = true;
            auto t0 = std::chrono::high_resolution_clock::now();
            SubOpStateReuseBatchResult res = runSubOpStateReuseBatch(queries, catalog.get(), session.get(), opts);
            const double wallMs = millisSince(t0);

            llvm::json::Object record;
            record["db"] = dbDir;
            record["db_label"] = dbLabel(dbDir);
            record["template"] = templ;
            record["batch_size"] = batchSize;
            record["mode"] = mode;
            record["rep"] = rep;
            record["returncode"] = res.exitCode;
            record["wall_ms"] = wallMs;
            record["optimization_ms"] = res.optimizationMs;
            record["execution_time_ms_total"] = res.totalExec.executionTime;
            record["reuse_targets"] = jsonSizeArray(res.numTargetsPerQuery);
            record["reuse_targets_no_table"] = jsonSizeArray(res.numTargetsNoTablePerQuery);
            record["reuse_targets_union"] = jsonSizeArray(res.numUnionTargetsPerQuery);
            record["reuse_targets_union_no_table"] = jsonSizeArray(res.numUnionTargetsNoTablePerQuery);
            record["reuse_targets_build_step"] = jsonSizeArray(res.numBuildStepTargetsPerQuery);
            record["reuse_targets_build_step_no_table"] = jsonSizeArray(res.numBuildStepTargetsNoTablePerQuery);
            record["reuse_targets_synthetic_mapped"] = static_cast<int64_t>(res.numTargetsSyntheticMapped);
            record["reuse_targets_synthetic_mapped_no_table"] =
               static_cast<int64_t>(res.numTargetsSyntheticMappedNoTable);
            record["reuse_targets_synthetic_mapped_union"] =
               static_cast<int64_t>(res.numUnionTargetsSyntheticMapped);
            record["reuse_targets_synthetic_mapped_union_no_table"] =
               static_cast<int64_t>(res.numUnionTargetsSyntheticMappedNoTable);
            record["reuse_targets_synthetic_mapped_build_step"] =
               static_cast<int64_t>(res.numBuildStepTargetsSyntheticMapped);
            record["reuse_targets_synthetic_mapped_build_step_no_table"] =
               static_cast<int64_t>(res.numBuildStepTargetsSyntheticMappedNoTable);
            record["result_block_count"] = static_cast<int64_t>(res.resultRowsetBlockHashes.size());
            record["result_rowset_block_hashes"] = jsonStringArray(res.resultRowsetBlockHashes);
            record["result_rowset_hash"] = hashStrings(res.resultRowsetBlockHashes);
            records.push_back(std::move(record));

            completed++;
            llvm::errs() << "[" << completed << "/" << totalCases << "] db=" << dbLabel(dbDir)
                         << " q" << templ << " batch=" << batchSize << " mode=" << mode
                         << " rep=" << rep << " rc=" << res.exitCode
                         << " time_ms=" << res.totalExec.executionTime
                         << " elapsed_s=" << (millisSince(allStart) / 1000.0) << "\n";
            llvm::errs().flush();
         }
      }
   }

   llvm::json::Object meta;
   meta["db"] = dbDir;
   meta["db_label"] = dbLabel(dbDir);
   meta["query_dir"] = queryDir;
   meta["mode"] = mode;
   meta["templates"] = std::move(templateJson);
   meta["batch_sizes"] = std::move(batchJson);
   meta["repetitions"] = repetitions;
   meta["force_sequential"] = forceSequential;
   llvm::json::Object root;
   root["meta"] = std::move(meta);
   root["records"] = std::move(records);
   writeJsonFile(outPath, llvm::json::Value(std::move(root)));
   llvm::errs() << "wrote " << outPath << "\n";
   llvm::errs().flush();
   return 0;
}

} // namespace

int main(int argc, char** argv) {
   if (argc == 2 && std::string(argv[1]) == "--features") {
      printFeatures();
      return 0;
   }
   if (argc >= 2 && std::string(argv[1]) == "--batch-nightly") {
      return runBatchNightlyMain(argc, argv);
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

   const char* dumpSubOpDir = std::getenv("LINGODB_DUMP_SUBOP_DIR");
   const char* dumpLoweringEnv = std::getenv("LINGODB_DUMP_LOWERING");
   SubOpStateReuseOptions opts;
   opts.skipReuseRewrite = skipReuseRewrite;
   opts.skipExecute = (std::getenv("LINGODB_SKIP_EXECUTE") != nullptr);
   opts.printPlan = envFlagEnabled("LINGODB_REUSE_PRINT_PLAN");
   opts.printMatches = envFlagEnabled("LINGODB_REUSE_PRINT_MATCHES");
   opts.printRewritten = (std::getenv("LINGODB_REUSE_PRINT_REWRITTEN") != nullptr);
   opts.verboseTiming = envFlagEnabled("LINGODB_REUSE_VERBOSE_TIMING");
   opts.dumpSubOpDir = dumpSubOpDir;
   opts.dumpLoweringSnapshots =
      dumpSubOpDir && (!dumpLoweringEnv || dumpLoweringEnv[0] == '\0' || envFlagEnabled("LINGODB_DUMP_LOWERING"));

   SubOpStateReuseBatchResult result = runSubOpStateReuseBatch(queries, catalog.get(), session.get(), opts);
   return result.exitCode;
}
