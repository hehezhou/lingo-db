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

/// Align a consumer to the union column layout: HIV payload order matches the cached producer, but existing
/// columns keep the consumer's \c member$N names and types; union-only columns are inserted (not type-replaced).
/// Then walk the \c cache_get HIV use chain and update embedded carrier types. Gather mappings are unchanged.
/// When \p cacheKey is set, only that \c cache_get root is processed.
void alignConsumerModulesToCachedJoinLayout(mlir::ModuleOp consumer, const CachedJoinBufferLayout& layout,
                                            std::optional<uint64_t> cacheKey = std::nullopt,
                                            std::optional<unsigned> consumerReuseQueryIndex = std::nullopt);

/// After union widening, materialize each per-query \c filter_pred$N on the synthetic join-buffer writer
/// using that query's table-scan pushdown filters.
void patchSyntheticJoinBufferFilterPredsFromMatchedQueries(
   mlir::ModuleOp synthetic, mlir::ModuleOp query0, mlir::ModuleOp query1,
   llvm::ArrayRef<CrossQueryStateMatchPair> matches, llvm::ArrayRef<CacheTarget> targetsInSynthetic,
   const CachedJoinBufferLayoutsByKey& layoutsByKey);

/// After \c insertCachePutsForTargets on the synthetic module, re-record layouts from \c cache_put
/// state types (includes `filter_pred$0` when join-buffer pred reuse is enabled).
void refreshCachedJoinLayoutsFromSyntheticCachePuts(mlir::ModuleOp synthetic,
                                                    llvm::ArrayRef<CacheTarget> targetsInSynthetic,
                                                    CachedJoinBufferLayoutsByKey& layoutsByKey);

/// After probe-side \c filter_pred$0 insertion, re-push the \c cache_get HIV layout through the reuse
/// SSA closure so nested \c lookup_entry_ref carriers match the aligned consumer member layout.
void resyncConsumerCachedHivCarrierTypesFromCacheGet(mlir::ModuleOp consumer,
                                                     std::optional<uint64_t> cacheKey = std::nullopt);

/// Fix lookup list / nested probe types to match each \c LookupOp's HIV (after union or pred layout passes).
void syncLookupCarrierAttrsFromState(mlir::ModuleOp module);

} // namespace lingodb::compiler::dialect::subop

#endif
