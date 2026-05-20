#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_REUSEJOINSUPERSET_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_REUSEJOINSUPERSET_H

#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseRewriteCommon.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateExtraction.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

#include <optional>

namespace lingodb::compiler::dialect::subop {

/// Producer HIV layout after \c extendSyntheticJoinBuffersToColumnUnion (union slot order + semantics).
struct CachedJoinBufferLayout {
   subop::HashIndexedViewType producerHiv;
   llvm::SmallVector<std::string> payloadSemanticKeys;
   llvm::SmallVector<subop::Member> payloadMembers;
   llvm::SmallVector<mlir::Type> payloadColumnTypes;
};

using CachedJoinBufferLayoutsByKey = llvm::DenseMap<uint64_t, CachedJoinBufferLayout>;

/// Widen join-buffer / HIV construction in \p synthetic so stored payload columns are the semantic
/// union of matched HIV states in \p query0 and \p query1. Updates scan_refs / gather / materialize /
/// merge / create_hash_indexed_view in the cloned producer IR.
/// When \p outLayouts is non-null, records per-cache-key producer HIV layout for consumer alignment.
void extendSyntheticJoinBuffersToColumnUnion(mlir::ModuleOp synthetic, mlir::ModuleOp query0, mlir::ModuleOp query1,
                                             llvm::ArrayRef<CrossQueryStateMatchPair> matches,
                                             llvm::ArrayRef<CacheTarget> targetsInSynthetic,
                                             const mlir::IRMapping& donorToSynthetic,
                                             CachedJoinBufferLayoutsByKey* outLayouts = nullptr);

/// Align a consumer module to the producer union HIV: types from \c cache_get through execution steps /
/// nested groups, and gather member slots at \c scan_list sites on that HIV's use chain only.
/// When \p cacheKey is set, only the matching \c subop.cache_get is used as the reuse root.
void alignConsumerModulesToCachedJoinLayout(mlir::ModuleOp consumer, const CachedJoinBufferLayout& layout,
                                            std::optional<uint64_t> cacheKey = std::nullopt);

/// After \c insertCachePutsForTargets on the synthetic module, re-record layouts from \c cache_put
/// state types (includes `filter_pred$0` when join-buffer pred reuse is enabled).
void refreshCachedJoinLayoutsFromSyntheticCachePuts(mlir::ModuleOp synthetic,
                                                    llvm::ArrayRef<CacheTarget> targetsInSynthetic,
                                                    CachedJoinBufferLayoutsByKey& layoutsByKey);

} // namespace lingodb::compiler::dialect::subop

#endif
