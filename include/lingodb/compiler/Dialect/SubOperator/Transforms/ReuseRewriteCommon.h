#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_REUSEREWRITECOMMON_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_REUSEREWRITECOMMON_H

#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateExtraction.h"

#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"

#include <cstdint>

namespace lingodb::compiler::dialect::subop {

struct CacheTarget {
   mlir::Value state;
   uint64_t cacheKey;
};

/// Cloned synthetic join-buffer build site (table \c scan_refs → \c materialize into buffer), recorded after
/// \c cloneExecutionStepsToQuery0 for post–column-union \c filter_pred insertion.
struct ClonedJoinBufferBuildSite {
   uint64_t cacheKey = 0;
   mlir::Value syntheticHiv;
   mlir::Value syntheticMergedBuffer;
   subop::ExecutionStepOp syntheticBuildStep;
};

using ClonedJoinBufferBuildSitesByKey = llvm::DenseMap<uint64_t, ClonedJoinBufferBuildSite>;

bool isPipelineStateValue(mlir::Value v);

mlir::Value peelBlockArgsToEnclosingOperands(mlir::Value v);

/// `ModuleReuseInfo` maps are keyed by values from `collectModuleReuseInfo` / `canonicalizeStateValueForReuse`.
template <typename MapT>
auto findReuseMap(const MapT& m, mlir::Value st) -> decltype(m.find(st)) {
   mlir::Value c = canonicalizeStateValueForReuse(st);
   auto it = m.find(c);
   if (it != m.end()) return it;
   if (c != st) return m.find(st);
   return m.end();
}

llvm::DenseMap<mlir::Operation*, const ModuleReuseInfo::StepRW*> buildRwByStepOpMap(const ModuleReuseInfo& reuse);

ExecutionGroupOp getSingleExecutionGroup(mlir::ModuleOp module);

} // namespace lingodb::compiler::dialect::subop

#endif
