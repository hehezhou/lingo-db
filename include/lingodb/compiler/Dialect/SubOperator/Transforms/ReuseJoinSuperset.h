#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_REUSEJOINSUPERSET_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_REUSEJOINSUPERSET_H

#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseRewriteCommon.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateExtraction.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/StringSet.h"

#include <optional>

namespace lingodb::compiler::dialect::subop {

/// Producer HIV layout after \c extendSyntheticJoinBuffersToColumnUnion (union slot order + semantics).
struct CachedJoinBufferLayout {
   subop::HashIndexedViewType producerHiv;
   /// Union-ordered payload columns (matches \c cache_put / synthetic producer layout).
   llvm::SmallVector<std::string> payloadSemanticKeys;
   llvm::SmallVector<subop::Member> payloadMembers;
   llvm::SmallVector<mlir::Type> payloadColumnTypes;
   /// Per matched query: semantic keys present in that query's join-buffer build (pre-union).
   llvm::SmallVector<std::string> query0SemanticKeys;
   llvm::SmallVector<std::string> query1SemanticKeys;
   /// For each entry in \c query0SemanticKeys / \c query1SemanticKeys, index into the union-ordered arrays above.
   llvm::SmallVector<unsigned> query0SlotInUnion;
   llvm::SmallVector<unsigned> query1SlotInUnion;
};

using CachedJoinBufferLayoutsByKey = llvm::DenseMap<uint64_t, CachedJoinBufferLayout>;

/// Union payload semantic key \c reuse_filter_pred\x1fN → query index \p N.
bool parseReuseFilterPredSemanticKey(llvm::StringRef semanticKey, unsigned& reuseQueryIndex);

/// Per \c cache_get: aligned HIV + SSA closure of probe-side uses (lookup → scan_list), same discovery as
/// \c alignConsumerModulesToCachedJoinLayout.
struct ConsumerCacheGetProbeClosure {
   std::optional<uint64_t> cacheKey;
   /// \c cache_get result SSA root used to rediscover probe \c scan_list sites after layout align.
   mlir::Value cacheGetRoot;
   std::optional<unsigned> consumerReuseQueryIndex;
   subop::HashIndexedViewType alignedHiv = nullptr;
   subop::HashIndexedViewType consumerHivBeforeAlign = nullptr;
   CachedJoinBufferLayout consumerLayout;
   llvm::DenseSet<void*> ssaClosure;
   ::llvm::StringSet<> probeLookupScopes;
   llvm::SmallVector<subop::ScanListOp, 16> scanListsFromTraverse;
};

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
/// Then walk the \c cache_get HIV use chain, remap probe \c gather member keys to the aligned layout, and update
/// embedded carrier types.
/// When \p cacheKey is set, only that \c cache_get root is processed.
void alignConsumerModulesToCachedJoinLayout(mlir::ModuleOp consumer, const CachedJoinBufferLayout& layout,
                                            std::optional<uint64_t> cacheKey = std::nullopt,
                                            std::optional<unsigned> consumerReuseQueryIndex = std::nullopt,
                                            llvm::SmallVectorImpl<ConsumerCacheGetProbeClosure>* outProbeClosures =
                                               nullptr);

/// After \c extendSyntheticJoinBuffersToColumnUnion, record table→buffer build steps in cloned synthetic IR.
ClonedJoinBufferBuildSitesByKey recordClonedJoinBufferBuildSites(mlir::ModuleOp synthetic,
                                                                 llvm::ArrayRef<CacheTarget> targetsInSynthetic,
                                                                 const ModuleReuseInfo& reuseSynthetic);

/// Materialize each per-query \c filter_pred$N on the synthetic join-buffer writer (table descr → MLIR after
/// \c scan_refs). Requires union layout and \p buildSites from \c recordClonedJoinBufferBuildSites.
void insertSyntheticFilterPredsAfterColumnUnion(
   mlir::ModuleOp synthetic, mlir::ModuleOp query0, mlir::ModuleOp query1,
   llvm::ArrayRef<CrossQueryStateMatchPair> matches, llvm::ArrayRef<CacheTarget> targetsInSynthetic,
   const CachedJoinBufferLayoutsByKey& layoutsByKey, const ClonedJoinBufferBuildSitesByKey& buildSites);

/// Insert \c gather filter_pred + \c filter(all_true) on \c scan_list ops discovered via \c cache_get closure.
void applyProbePredFiltersForConsumerClosures(mlir::ModuleOp consumer,
                                              llvm::MutableArrayRef<ConsumerCacheGetProbeClosure> probeClosures);

/// After \c insertCachePutsForTargets on the synthetic module, re-record layouts from \c cache_put
/// state types (includes `filter_pred$0` when join-buffer pred reuse is enabled).
void refreshCachedJoinLayoutsFromSyntheticCachePuts(mlir::ModuleOp synthetic,
                                                    llvm::ArrayRef<CacheTarget> targetsInSynthetic,
                                                    CachedJoinBufferLayoutsByKey& layoutsByKey);

/// After probe-side \c filter_pred$0 insertion, re-push the \c cache_get HIV layout through the reuse
/// SSA closure so nested \c lookup_entry_ref carriers match the aligned consumer member layout.
void resyncConsumerCachedHivCarrierTypesFromCacheGet(mlir::ModuleOp consumer,
                                                     std::optional<uint64_t> cacheKey = std::nullopt);

/// Reconcile probe \c scan_list / \c gather column metadata with the aligned \c cache_get HIV, only within the
/// SSA closure rooted at that \c cache_get (run after pred reapply / resync).
void finalizeConsumerCachedJoinProbeColumnAttrs(mlir::ModuleOp consumer, ConsumerCacheGetProbeClosure& probe);

/// Fix lookup list / nested probe types to match each \c LookupOp's HIV (after union or pred layout passes).
void syncLookupCarrierAttrsFromState(mlir::ModuleOp module);

/// Align \c gather member keys with embedded \c lookup_entry_ref HIV slots (whole module; for synthetic producer).
void syncProbeGatherMappingsInModule(mlir::ModuleOp module);

} // namespace lingodb::compiler::dialect::subop

#endif
