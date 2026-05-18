#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_REUSEJOINSUPERSET_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_REUSEJOINSUPERSET_H

#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseRewriteCommon.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateExtraction.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

namespace lingodb::compiler::dialect::subop {

/// Widen join-buffer / HIV construction in \p synthetic so stored payload columns are the semantic
/// union of matched HIV states in \p query0 and \p query1. Updates scan_refs / gather / materialize /
/// merge / create_hash_indexed_view in the cloned producer IR. Does not change consumer cache_get layouts.
void extendSyntheticJoinBuffersToColumnUnion(mlir::ModuleOp synthetic, mlir::ModuleOp query0, mlir::ModuleOp query1,
                                             llvm::ArrayRef<CrossQueryStateMatchPair> matches,
                                             llvm::ArrayRef<CacheTarget> targetsInSynthetic,
                                             const mlir::IRMapping& donorToSynthetic);

} // namespace lingodb::compiler::dialect::subop

#endif
