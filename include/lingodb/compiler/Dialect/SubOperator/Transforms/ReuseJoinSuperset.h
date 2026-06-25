#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_REUSEJOINSUPERSET_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_REUSEJOINSUPERSET_H

#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseRewriteCommon.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateExtraction.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/StringSet.h"

#include <optional>

namespace lingodb::catalog {
class Catalog;
} // namespace lingodb::catalog

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
   llvm::DenseMap<unsigned, llvm::SmallVector<std::string, 8>> querySemanticKeysById;
   /// For each entry in \c query0SemanticKeys / \c query1SemanticKeys, index into the union-ordered arrays above.
   llvm::SmallVector<unsigned> query0SlotInUnion;
   llvm::SmallVector<unsigned> query1SlotInUnion;
   llvm::DenseMap<unsigned, llvm::SmallVector<unsigned, 8>> querySlotInUnionById;
   /// Pre–\c cache_get probe \c gather member slots remapped to aligned \c cache_get HIV members.
   llvm::DenseMap<subop::Member, subop::Member> probeGatherMemberRemap;
};

using CachedJoinBufferLayoutsByKey = llvm::DenseMap<uint64_t, CachedJoinBufferLayout>;

struct CachedAggregateLayout {
   subop::PreAggrHtType producerHt;
   bool mixedByQueryId = false;
   subop::Member queryIdMember;
   llvm::SmallVector<std::string> payloadSemanticKeys;
   llvm::SmallVector<subop::Member> payloadMembers;
   llvm::SmallVector<mlir::Type> payloadColumnTypes;
   llvm::SmallVector<std::string> query0SemanticKeys;
   llvm::SmallVector<subop::Member> query0Members;
   llvm::SmallVector<std::string> query1SemanticKeys;
   llvm::SmallVector<subop::Member> query1Members;
   llvm::DenseMap<unsigned, llvm::SmallVector<std::string, 8>> querySemanticKeysById;
   llvm::DenseMap<unsigned, llvm::SmallVector<subop::Member, 8>> queryMembersById;
};

using CachedAggregateLayoutsByKey = llvm::DenseMap<uint64_t, CachedAggregateLayout>;

/// Union payload semantic key \c reuse_filter_pred\x1fN → query index \p N.
bool parseReuseFilterPredSemanticKey(llvm::StringRef semanticKey, unsigned& reuseQueryIndex);

/// True when matched HIV peers' donor \c get_external pushdown filters are identical (no \c filter_pred reuse).
bool joinMatchPeerExternalFiltersIdentical(mlir::ModuleOp query0, mlir::ModuleOp query1, mlir::Value hivA,
                                           mlir::Value hivB, const ModuleReuseInfo& reuse0,
                                           const ModuleReuseInfo& reuse1);

/// Estimate the row count of the OR-merged donor-table pushdown filters for a matched HIV pair.
/// Experimental: assumes both HIVs resolve to the same external table and that catalog sample metadata exists.
double estimateMergedHivExternalFilterRows(mlir::ModuleOp query0, mlir::ModuleOp query1, mlir::Value hivA,
                                           mlir::Value hivB, const ModuleReuseInfo& reuse0,
                                           const ModuleReuseInfo& reuse1,
                                           lingodb::catalog::Catalog& catalog);

/// Current aggregate union rewrite supports a single top-level reduce build step. Nested aggregate builds
/// require explicit tuple-column threading before they can be safely reused across disjoint filters.
bool aggregateHashTablePayloadUnionSupported(mlir::Value aggregateState, const ModuleReuseInfo& reuse);

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

/// Batch form of \c extendSyntheticJoinBuffersToColumnUnion. Each reuse group is widened once and may carry
/// \c filter_pred$N slots for every participating query index.
void extendSyntheticJoinBuffersToColumnUnionForGroups(
   mlir::ModuleOp synthetic, llvm::ArrayRef<mlir::ModuleOp> queries,
   llvm::ArrayRef<CrossQueryStateMatchGroup> groups,
   llvm::ArrayRef<CacheTarget> targetsInSynthetic,
   CachedJoinBufferLayoutsByKey* outLayouts = nullptr);

/// Some reused HIVs are built by probing an upstream mixed HIV. In that case the downstream HIV must
/// inherit the upstream `filter_pred$N` slots so consumers can keep selecting their own slice.
void extendSyntheticJoinBuffersWithInheritedMixedPreds(
   mlir::ModuleOp synthetic, llvm::ArrayRef<CacheTarget> targetsInSynthetic,
   CachedJoinBufferLayoutsByKey* outLayouts = nullptr,
   llvm::DenseMap<uint64_t, llvm::SmallVector<uint64_t, 4>>* outInheritedDepsByCacheKey = nullptr,
   llvm::DenseMap<uint64_t, llvm::DenseMap<unsigned, unsigned>>* consumerSlotByCacheKeyAndQuery = nullptr);

/// Widen scan_refs-only buffer reuse targets with per-query `filter_pred$N` members, materialize
/// those predicates on the synthetic writer, and OR-filter the synthetic buffer stream.
void extendSyntheticScanRefsBuffersWithFilterPredsForGroups(
   mlir::ModuleOp synthetic, llvm::ArrayRef<CrossQueryStateMatchGroup> groups,
   llvm::ArrayRef<CacheTarget> targetsInSynthetic,
   const llvm::DenseMap<uint64_t, llvm::DenseMap<unsigned, unsigned>>* consumerSlotByCacheKeyAndQuery =
      nullptr);

/// Align a consumer `cache_get` buffer to the synthetic predicate layout and filter each scan_refs
/// of that cached buffer by the explicitly selected `filter_pred$N` slot.
void alignConsumerScanRefsBufferCacheGetWithFilterPreds(
   mlir::ModuleOp consumer, uint64_t cacheKey, subop::StateMembersAttr producerMembers,
   llvm::ArrayRef<unsigned> predSlots,
   unsigned consumerReuseSlot);

void extendSyntheticAggregateHashTablesToPayloadUnion(mlir::ModuleOp synthetic, mlir::ModuleOp query0,
                                                      mlir::ModuleOp query1,
                                                      llvm::ArrayRef<CrossQueryStateMatchPair> matches,
                                                      llvm::ArrayRef<CacheTarget> targetsInSynthetic,
                                                      const mlir::IRMapping& donorToSynthetic,
                                                      CachedAggregateLayoutsByKey* outLayouts = nullptr);

void extendSyntheticAggregateHashTablesToPayloadUnionForGroups(
   mlir::ModuleOp synthetic, llvm::ArrayRef<mlir::ModuleOp> queries,
   llvm::ArrayRef<CrossQueryStateMatchGroup> groups,
   llvm::ArrayRef<CacheTarget> targetsInSynthetic,
   CachedAggregateLayoutsByKey* outLayouts = nullptr);

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

void alignConsumerModulesToCachedAggregateLayout(mlir::ModuleOp consumer, const CachedAggregateLayout& layout,
                                                 std::optional<uint64_t> cacheKey = std::nullopt,
                                                 std::optional<unsigned> consumerReuseQueryIndex = std::nullopt,
                                                 std::optional<unsigned> consumerReuseFilterSlot = std::nullopt);

/// After \c extendSyntheticJoinBuffersToColumnUnion, record table→buffer build steps in cloned synthetic IR.
ClonedJoinBufferBuildSitesByKey recordClonedJoinBufferBuildSites(mlir::ModuleOp synthetic,
                                                                 llvm::ArrayRef<CacheTarget> targetsInSynthetic,
                                                                 const ModuleReuseInfo& reuseSynthetic);

/// Refresh \c syntheticHiv values after layout/type rewrites while preserving the already chosen build step.
void refreshClonedJoinBufferBuildSiteStates(mlir::ModuleOp synthetic,
                                            ClonedJoinBufferBuildSitesByKey& buildSites,
                                            const ModuleReuseInfo& reuseSynthetic);

/// Materialize each per-query \c filter_pred$N on the synthetic join-buffer writer (table descr → MLIR after
/// \c scan_refs). Requires union layout and \p buildSites from \c recordClonedJoinBufferBuildSites.
void insertSyntheticFilterPredsAfterColumnUnion(
   mlir::ModuleOp synthetic, mlir::ModuleOp query0, mlir::ModuleOp query1,
   llvm::ArrayRef<CrossQueryStateMatchPair> matches, llvm::ArrayRef<CacheTarget> targetsInSynthetic,
   const CachedJoinBufferLayoutsByKey& layoutsByKey, const ClonedJoinBufferBuildSitesByKey& buildSites);

void insertSyntheticFilterPredsAfterColumnUnionForGroups(
   mlir::ModuleOp synthetic, llvm::ArrayRef<mlir::ModuleOp> queries,
   llvm::ArrayRef<CrossQueryStateMatchGroup> groups,
   llvm::ArrayRef<CacheTarget> targetsInSynthetic,
   const CachedJoinBufferLayoutsByKey& layoutsByKey,
   const ClonedJoinBufferBuildSitesByKey& buildSites);

/// Retag cached HIV probe closures to MixedHIV so scan_list applies the stored filter_pred$N slot directly.
void applyProbePredFiltersForConsumerClosures(mlir::ModuleOp consumer,
                                              llvm::MutableArrayRef<ConsumerCacheGetProbeClosure> probeClosures);

/// For mixed HIV lookups whose state value comes from the given \c cache_get key, use \c filter_pred$slot as
/// the lookup predicate key. The state source is found through execution-step/nested-group def-use ports.
void setMixedLookupPredSlotForCacheGet(mlir::ModuleOp module, uint64_t cacheKey, unsigned slot);

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
