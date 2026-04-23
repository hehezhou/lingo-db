#include "features.h"

#include "lingodb/catalog/Catalog.h"
#include "lingodb/compiler/Conversion/RelAlgToSubOp/RelAlgToSubOpPass.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateExtraction.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/mlir-support/eval.h"
#include "lingodb/execution/Frontend.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

#include <cassert>
#include <cctype>
#include <memory>
#include <string>

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

} // namespace

int main(int argc, char** argv) {
   if (argc == 2 && std::string(argv[1]) == "--features") {
      printFeatures();
      return 0;
   }

   // Usage:
   // - subop-print-steps <db_dir> <sql_file>
   // - subop-print-steps <db_dir> <json_file_or_dash>
   assert(argc >= 3);
   std::string dbDir = argv[1];
   std::string input = argv[2];

   lingodb::compiler::support::eval::init();
   std::shared_ptr<lingodb::catalog::Catalog> catalog = lingodb::catalog::Catalog::create(dbDir, false);

   auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(input);
   assert(!fileOrErr.getError());

   auto buf = (*fileOrErr)->getBuffer();
   llvm::SmallVector<std::string, 16> queries;

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

   for (size_t i = 0; i < queries.size(); i++) {
      std::string sql = queries[i];

      mlir::MLIRContext ctx;
      lingodb::execution::initializeContext(ctx, /*includeLLVM*/ false);

      auto frontend = lingodb::execution::createSQLFrontend();
      frontend->setCatalog(catalog.get());
      frontend->setContext(&ctx);
      frontend->loadFromString(sql);

      auto& feErr = frontend->getError();
      assert(!feErr);

      mlir::ModuleOp* moduleOpPtr = frontend->getModule();
      assert(moduleOpPtr);

      runPasses(*moduleOpPtr, catalog.get());

      // 1) Print SubOp layer IR (previous json-sql-to-subop output).
      llvm::outs() << "\n// ============================\n";
      llvm::outs() << "// query[" << i << "] subop layer\n";
      llvm::outs() << "// ============================\n";
      moduleOpPtr->print(llvm::outs());
      llvm::outs() << "\n";

      // 2) Print execution-step/state debug (previous subop-print-steps output).
      lingodb::compiler::dialect::subop::printExecutionSteps(*moduleOpPtr, llvm::outs());
      llvm::outs() << "\n";
   }
   return 0;
}

