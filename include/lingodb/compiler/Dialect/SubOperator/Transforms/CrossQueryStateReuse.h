#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_CROSSQUERYSTATEREUSE_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_CROSSQUERYSTATEREUSE_H

#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateExtraction.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <string>

namespace lingodb::compiler::dialect::subop {

struct ReusePlanRewriteResult {
   mlir::OwningOpRef<mlir::ModuleOp> query0; // synthetic producer that cache_puts reusable states
   size_t numTargetsQuery0 = 0;
   size_t numTargetsQuery1 = 0;
   size_t numTargetsQuery0Mapped = 0;
   /// `execution_step` ops present after `addSsaPredecessorSteps` but not in the state-closure core
   /// (`collectCreateAndWriteStepsForStates`); nonzero means the SSA worklist found extra steps.
   size_t numStepOpsAddedOnlyBySsaPredecessorClosure = 0;
   /// When `numStepOpsAddedOnlyBySsaPredecessorClosure > 0`, multi-line `// ...` report (per-step
   /// detail + aggregate histogram) for debugging; printed by tools after the scalar count.
   std::string ssaExtraClosureStepsReport;
};

// Three-stage pipeline:
// 1) analysis: provided by StateExtraction (collectModuleReuseInfo, buildStateMatchProfiles)
// 2) match: provided by StateExtraction (collectCrossQueryStateMatchPairs)
// 3) rewrite: implemented here
//
// Rewrites both queries to use cache_get, and returns a synthetic query0 module that
// computes/cache_puts the reusable states (cloned from query0's construction steps).
ReusePlanRewriteResult rewritePlansWithSyntheticQuery0(
   mlir::ModuleOp query0,
   mlir::ModuleOp query1,
   llvm::ArrayRef<CrossQueryStateMatchPair> matches);

} // namespace lingodb::compiler::dialect::subop

#endif

