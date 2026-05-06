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
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

#include <cassert>
#include <cctype>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

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

static void executeFromSubOpLayer(mlir::ModuleOp subopModule,
                                  lingodb::runtime::ExecutionContext* executionContext) {
   using namespace lingodb::compiler::dialect;

   // Force sequential execution in this debug tool to reduce scheduler/parallelism sensitivity.
   subopModule->setAttr("subop.sequential", mlir::UnitAttr::get(subopModule.getContext()));

   // From SubOp layer (after PrepareLoweringPass) to imperative lowering.
   {
      mlir::PassManager lowerSubOpPm(subopModule.getContext());
      lowerSubOpPm.enableVerifier(true);
      lowerSubOpPm.addPass(subop::createLowerSubOpPass());
      lowerSubOpPm.addPass(lingodb::compiler::createCanonicalizerPass());
      lowerSubOpPm.addPass(mlir::createCSEPass());
      assert(succeeded(lowerSubOpPm.run(subopModule)));
   }

   // Lower DB imperative ops.
   {
      mlir::PassManager lowerDBPm(subopModule.getContext());
      lowerDBPm.enableVerifier(true);
      db::createLowerDBPipeline(lowerDBPm);
      assert(succeeded(lowerDBPm.run(subopModule)));
   }

   // Lower Arrow ops to std.
   {
      mlir::PassManager lowerArrowPm(subopModule.getContext());
      lowerArrowPm.enableVerifier(true);
      lowerArrowPm.addPass(lingodb::compiler::dialect::arrow::createLowerToStdPass());
      lowerArrowPm.addPass(lingodb::compiler::createCanonicalizerPass());
      lowerArrowPm.addPass(mlir::createLoopInvariantCodeMotionPass());
      lowerArrowPm.addPass(mlir::createCSEPass());
      assert(succeeded(lowerArrowPm.run(subopModule)));
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
   // - subop-print-steps <db_dir> <sql_or_json>
   // - subop-print-steps <db_dir> <sql_file_a> <sql_file_b>
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

      runPasses(run.module, catalog.get());
      runs.push_back(std::move(run));

      // 1) Print SubOp layer IR (previous json-sql-to-subop output).
      llvm::outs() << "\n// ============================\n";
      llvm::outs() << "// query[" << i << "] subop layer\n";
      llvm::outs() << "// ============================\n";
      runs.back().module.print(llvm::outs());
      llvm::outs() << "\n";

      // 2) Print execution-step/state debug (previous subop-print-steps output).
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
   auto rewriteRes = lingodb::compiler::dialect::subop::rewritePlansWithSyntheticQuery0(
      runs[0].module, runs[1].module, firstPair);
   llvm::outs() << "\n// reuse_targets: query[0]=" << rewriteRes.numTargetsQuery0
                  << " query[1]=" << rewriteRes.numTargetsQuery1 << "\n";
   llvm::outs() << "\n// reuse_targets_q0_mapped: " << rewriteRes.numTargetsQuery0Mapped << "\n";

   // Optional heavy debug printing (can be huge / sometimes crashes when IR is malformed).
   // Enable via env var: LINGODB_REUSE_PRINT_REWRITTEN=1
   const bool printRewritten = (std::getenv("LINGODB_REUSE_PRINT_REWRITTEN") != nullptr);
   if (printRewritten) {
      llvm::outs() << "\n// ============================\n";
      llvm::outs() << "// query[0] (synthetic) subop layer (post-rewrite)\n";
      llvm::outs() << "// ============================\n";
      (*rewriteRes.query0)->print(llvm::outs());
      llvm::outs() << "\n";
      llvm::outs() << "\n// ============================\n";
      llvm::outs() << "// Step layout vs SubOpToControlFlow stderr ordinal\n";
      llvm::outs() << "// ============================\n";
      if (rewriteRes.numTargetsQuery0Mapped == 0) {
         llvm::outs() << "// (synthetic query[0] has no cloned steps: reuse_targets_q0_mapped==0; "
                          "see reuse_targets above. Layout below is from real query[0]/[1] modules.)\n";
      } else {
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
   auto sharedExecCtx = session->createExecutionContext();
   auto runOne = [&](mlir::ModuleOp mod, const std::string& label) {
      llvm::outs() << "\n// ============================\n";
      llvm::outs() << label << "\n";
      llvm::outs() << "// ============================\n";
      mlir::OwningOpRef<mlir::ModuleOp> execModule = mlir::cast<mlir::ModuleOp>(mod->clone());
      executeFromSubOpLayer(*execModule, sharedExecCtx.get());
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
