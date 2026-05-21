#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseJoinSuperset.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/CrossQueryStateReuse.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseRewriteCommon.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseFilterPredInsert.h"
#include "lingodb/compiler/Dialect/DB/IR/DBTypes.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseStateClosure.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsAttributes.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateExtraction.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/runtime/ExternalDataSourceProperty.h"
#include "lingodb/utility/Serialization.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include <algorithm>
#include <cassert>
#include <unordered_set>

namespace lingodb::compiler::dialect::subop {
namespace {

using lingodb::runtime::ExternalDatasourceProperty;

static llvm::StringRef stripMemberSuffix(llvm::StringRef name) {
   size_t pos = name.find('$');
   if (pos == llvm::StringRef::npos) return name;
   return name.take_front(pos);
}

static std::string columnSemanticKey(llvm::StringRef scope, llvm::StringRef leaf) {
   return (scope + "\x1f" + leaf).str();
}

static constexpr llvm::StringRef kReuseFilterPredScope = "reuse_filter_pred";

static bool isFilterPredPayloadColumn(llvm::StringRef memberName, llvm::StringRef leaf) {
   return memberName.starts_with("filter_pred") || leaf == "filter_pred";
}

static std::optional<unsigned> parseFilterPredMemberSlot(llvm::StringRef memberName) {
   if (!memberName.consume_front("filter_pred$")) return std::nullopt;
   unsigned slot = 0;
   if (memberName.getAsInteger(10, slot)) return std::nullopt;
   return slot;
}

/// Union payload key for per-query write-side predicates on a shared table→HIV chain.
static std::string reuseFilterPredSemanticKey(unsigned reuseQueryIndex) {
   return columnSemanticKey(kReuseFilterPredScope, llvm::Twine(reuseQueryIndex).str());
}

static bool parseReuseFilterPredSemanticKey(llvm::StringRef semanticKey, unsigned& reuseQueryIndex) {
   size_t sep = semanticKey.find('\x1f');
   if (sep == llvm::StringRef::npos) return false;
   if (semanticKey.take_front(sep) != kReuseFilterPredScope) return false;
   return !semanticKey.drop_front(sep + 1).getAsInteger(10, reuseQueryIndex);
}

static bool parseFilterPredLayoutSemanticKey(llvm::StringRef semanticKey, unsigned& predIndex) {
   if (parseReuseFilterPredSemanticKey(semanticKey, predIndex)) return true;
   return static_cast<bool>(parseFilterPredMemberSlot(semanticKey));
}

static llvm::StringRef normalizeColumnIdentifier(llvm::StringRef identifier) {
   return stripMemberSuffix(identifier);
}

struct PayloadColumnSpec {
   std::string semanticKey;
   std::string scope;
   std::string leaf;
   mlir::Type colType;
   bool isJoinKey = false;
};

struct JoinBufferUnionPlan {
   subop::Member linkMember;
   subop::Member hashMember;
   llvm::SmallVector<PayloadColumnSpec, 8> payloadColumns;
   llvm::SmallVector<mlir::Type, 8> payloadMemberTypes;
   /// Buffer/HIV payload members aligned with \p payloadColumns (semantic union, not slot `member$N`).
   llvm::SmallVector<subop::Member, 8> payloadMembers;
};

static subop::CreateHashIndexedView findCreateHashIndexedViewForState(
   mlir::Value hiv, const llvm::DenseMap<mlir::Value, llvm::SmallVector<subop::ExecutionStepOp, 8>>& writerStepsByState) {
   auto itW = writerStepsByState.find(hiv);
   if (itW == writerStepsByState.end()) return {};
   for (subop::ExecutionStepOp ws : itW->second) {
      subop::CreateHashIndexedView found;
      ws.walk([&](subop::CreateHashIndexedView op) { found = op; });
      if (found) return found;
   }
   return {};
}

static subop::BufferType getInnerBufferTypeForMaterializeState(mlir::Type stateTy) {
   if (auto b = mlir::dyn_cast<subop::BufferType>(stateTy)) return b;
   if (auto tl = mlir::dyn_cast<subop::ThreadLocalType>(stateTy))
      return mlir::dyn_cast<subop::BufferType>(tl.getWrapped());
   return nullptr;
}

static bool materializeTargetsJoinBuffer(subop::MaterializeOp mat, mlir::Value mergedBuffer,
                                        const ModuleReuseInfo& reuse) {
   subop::BufferType targetBuf = mlir::dyn_cast<subop::BufferType>(mergedBuffer.getType());
   if (!targetBuf) return false;
   subop::BufferType matBuf = getInnerBufferTypeForMaterializeState(mat.getState().getType());
   if (!matBuf) return false;
   if (matBuf == targetBuf) return true;
   mlir::Value canonBuf = canonicalizeStateValueForReuse(mergedBuffer);
   if (auto itTL = reuse.mergedFromThreadLocal.find(canonBuf); itTL != reuse.mergedFromThreadLocal.end()) {
      if (mat.getState() == itTL->second) return true;
   }
   return false;
}

static subop::ExecutionStepOp findBufferBuildStepWithTableMaterialize(mlir::Value mergedBuffer,
                                                                        const ModuleReuseInfo& reuse) {
   for (const ModuleReuseInfo::StepRW& rw : reuse.steps) {
      subop::ExecutionStepOp step = rw.step;
      bool hasTableScan = false;
      bool hasJoinMat = false;
      step->walk([&](subop::ScanRefsOp scan) {
         if (mlir::isa<subop::TableType>(scan.getState().getType())) hasTableScan = true;
      });
      step->walk([&](subop::MaterializeOp mat) {
         if (materializeTargetsJoinBuffer(mat, mergedBuffer, reuse)) hasJoinMat = true;
      });
      if (hasTableScan && hasJoinMat) return step;
   }
   return {};
}

static subop::ExecutionStepOp findBufferBuildStepWithTableScan(mlir::ModuleOp module) {
   subop::ExecutionStepOp found;
   module.walk([&](subop::ExecutionStepOp step) {
      if (found) return;
      bool hasTableScan = false;
      bool hasBufferMat = false;
      step->walk([&](subop::ScanRefsOp scan) {
         if (mlir::isa<subop::TableType>(scan.getState().getType())) hasTableScan = true;
      });
      step->walk([&](subop::MaterializeOp mat) {
         if (getInnerBufferTypeForMaterializeState(mat.getState().getType())) hasBufferMat = true;
      });
      if (hasTableScan && hasBufferMat) found = step;
   });
   return found;
}

static void collectPayloadFromMaterialize(
   subop::MaterializeOp mat, llvm::StringRef linkMemberName, llvm::StringRef hashMemberName, subop::MemberManager& mm,
   lingodb::compiler::dialect::tuples::ColumnManager& cm, llvm::StringRef joinKeyMemberName,
   llvm::StringMap<PayloadColumnSpec>& out, std::optional<unsigned> reuseQueryIndex) {
   for (auto& [member, colRef] : mat.getMapping().getMapping()) {
      llvm::StringRef memName = mm.getName(member);
      if (memName == linkMemberName || memName == hashMemberName) continue;
      auto [scope, leaf] = cm.getName(&colRef.getColumn());
      PayloadColumnSpec spec;
      spec.colType = colRef.getColumn().type;
      spec.isJoinKey = !joinKeyMemberName.empty() && memName == joinKeyMemberName;
      if (reuseQueryIndex && isFilterPredPayloadColumn(memName, leaf)) {
         spec.scope = kReuseFilterPredScope.str();
         spec.leaf = llvm::Twine(*reuseQueryIndex).str();
         spec.semanticKey = reuseFilterPredSemanticKey(*reuseQueryIndex);
      } else {
         spec.scope = scope;
         spec.leaf = leaf;
         spec.semanticKey = columnSemanticKey(scope, leaf);
      }
      out.try_emplace(spec.semanticKey, spec);
   }
}

static mlir::Type memberTypeForIdentifier(subop::TableType tableTy, subop::MemberManager& mm, llvm::StringRef identifier);

static ExternalDatasourceProperty mergeExternalDatasource(const ExternalDatasourceProperty& a,
                                                        const ExternalDatasourceProperty& b);

/// One \c get_external per module, indexed by \c ExternalDatasourceProperty::tableName (single module walk).
struct ExternalTableCatalog {
   struct Entry {
      subop::GetExternalOp op;
      ExternalDatasourceProperty ds;
      subop::TableType tableTy;
   };
   llvm::StringMap<llvm::SmallVector<Entry>> byTableName;

   llvm::ArrayRef<Entry> entries(llvm::StringRef tableName) const {
      auto it = byTableName.find(tableName);
      if (it == byTableName.end()) return {};
      return it->second;
   }

   subop::TableType representativeTableTy(llvm::StringRef tableName) const {
      auto refs = entries(tableName);
      if (refs.empty()) return {};
      return refs.front().tableTy;
   }
};

static ExternalTableCatalog buildExternalTableCatalog(mlir::ModuleOp module) {
   ExternalTableCatalog catalog;
   if (!module) return catalog;
   module.walk([&](subop::GetExternalOp ge) {
      auto ds = lingodb::utility::deserializeFromHexString<ExternalDatasourceProperty>(ge.getDescr());
      auto tableTy = mlir::dyn_cast<subop::TableType>(ge.getResult().getType());
      if (!tableTy) return;
      catalog.byTableName[ds.tableName].push_back({ge, std::move(ds), tableTy});
   });
   return catalog;
}

static void mergeExternalDatasourceForTable(ExternalDatasourceProperty& merged, bool& haveMerged,
                                            const ExternalTableCatalog& catalog, llvm::StringRef tableName) {
   for (const ExternalTableCatalog::Entry& entry : catalog.entries(tableName)) {
      if (!haveMerged) {
         merged = entry.ds;
         haveMerged = true;
      } else {
         merged = mergeExternalDatasource(merged, entry.ds);
      }
   }
}

static void ingestExternalTableColumnsForUnionScopes(const ExternalTableCatalog& catalog,
                                                     llvm::StringMap<PayloadColumnSpec>& unionCols) {
   if (catalog.byTableName.empty()) return;
   llvm::StringSet<> scopesInUnion;
   for (auto& it : unionCols) {
      if (!it.getValue().scope.empty()) scopesInUnion.insert(it.getValue().scope);
   }
   auto* ctx = catalog.byTableName.begin()->second.front().op->getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   for (const auto& tableIt : catalog.byTableName) {
      if (!scopesInUnion.contains(tableIt.getKey())) continue;
      for (const ExternalTableCatalog::Entry& entry : tableIt.getValue()) {
         for (const auto& map : entry.ds.mapping) {
            llvm::StringRef leaf = normalizeColumnIdentifier(map.identifier);
            PayloadColumnSpec spec;
            spec.scope = entry.ds.tableName;
            spec.leaf = leaf.str();
            spec.colType = memberTypeForIdentifier(entry.tableTy, mm, leaf);
            if (!spec.colType) continue;
            spec.semanticKey = columnSemanticKey(spec.scope, spec.leaf);
            unionCols.try_emplace(spec.semanticKey, spec);
         }
      }
   }
}

static mlir::Type columnTypeForIdentifierInCatalog(const ExternalTableCatalog& catalog, llvm::StringRef tableName,
                                                   llvm::StringRef identifier) {
   if (catalog.byTableName.empty()) return {};
   auto* ctx = catalog.byTableName.begin()->second.front().op->getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   for (const ExternalTableCatalog::Entry& entry : catalog.entries(tableName)) {
      if (mlir::Type ty = memberTypeForIdentifier(entry.tableTy, mm, identifier)) return ty;
   }
   return {};
}

static JoinBufferUnionPlan buildUnionPlan(mlir::Value hivA, mlir::Value hivB, const ModuleReuseInfo& reuseA,
                                        const ModuleReuseInfo& reuseB, mlir::ModuleOp modA, mlir::ModuleOp modB) {
   JoinBufferUnionPlan plan;
   mlir::MLIRContext* ctxA = hivA.getContext();
   auto* subDialectA = ctxA->getLoadedDialect<subop::SubOperatorDialect>();
   auto* tupleDialectA = ctxA->getLoadedDialect<tuples::TupleStreamDialect>();
   assert(subDialectA && tupleDialectA);
   auto& mm = subDialectA->getMemberManager();

   mlir::Value hiv = resolveCacheTargetStateForReuse(hivA, reuseA);
   auto hivTy = mlir::cast<subop::HashIndexedViewType>(hiv.getType());
   subop::CreateHashIndexedView chiv = findCreateHashIndexedViewForState(hiv, reuseA.writerStepsByState);
   assert(chiv && "join superset: HIV must have create_hash_indexed_view writer");
   plan.linkMember = chiv.getLinkMember().getMember();
   plan.hashMember = chiv.getHashMember().getMember();
   llvm::StringRef joinKeyMemberName;
   if (!hivTy.getValueMembers().getMembers().empty()) {
      joinKeyMemberName = mm.getName(hivTy.getValueMembers().getMembers().front());
   }
   llvm::StringRef linkMemberName = mm.getName(plan.linkMember);
   llvm::StringRef hashMemberName = mm.getName(plan.hashMember);

   llvm::StringMap<PayloadColumnSpec> unionCols;
   auto ingestHiv = [&](mlir::Value h, const ModuleReuseInfo& reuse, unsigned reuseQueryIndex) {
      auto& cm = h.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      mlir::Value canon = resolveCacheTargetStateForReuse(h, reuse);
      mlir::Value buf = canon;
      if (auto it = reuse.hashIndexedViewFromMergedBuffer.find(canon);
          it != reuse.hashIndexedViewFromMergedBuffer.end()) {
         buf = it->second;
      }
      subop::ExecutionStepOp buildStep = findBufferBuildStepWithTableMaterialize(buf, reuse);
      if (!buildStep) {
         if (mlir::Operation* def = h.getDefiningOp()) {
            if (auto mod = def->getParentOfType<mlir::ModuleOp>()) buildStep = findBufferBuildStepWithTableScan(mod);
         }
      }
      if (!buildStep) return;
      buildStep.walk([&](subop::MaterializeOp mat) {
         if (!getInnerBufferTypeForMaterializeState(mat.getState().getType())) return;
         collectPayloadFromMaterialize(mat, linkMemberName, hashMemberName, mm, cm, joinKeyMemberName, unionCols,
                                       reuseQueryIndex);
      });
      if (kEnableReuseStateFilterPredReapply && buildStep) {
         PayloadColumnSpec predSpec;
         predSpec.scope = kReuseFilterPredScope.str();
         predSpec.leaf = llvm::Twine(reuseQueryIndex).str();
         predSpec.colType = mlir::IntegerType::get(h.getContext(), 1);
         predSpec.semanticKey = reuseFilterPredSemanticKey(reuseQueryIndex);
         unionCols.try_emplace(predSpec.semanticKey, predSpec);
      }
   };
   ingestHiv(hivA, reuseA, 0);
   ingestHiv(hivB, reuseB, 1);
   ExternalTableCatalog catalogA = buildExternalTableCatalog(modA);
   ExternalTableCatalog catalogB = buildExternalTableCatalog(modB);
   ingestExternalTableColumnsForUnionScopes(catalogA, unionCols);
   ingestExternalTableColumnsForUnionScopes(catalogB, unionCols);

   llvm::SmallVector<PayloadColumnSpec*, 8> ordered;
   ordered.reserve(unionCols.size());
   for (auto& it : unionCols) ordered.push_back(&it.second);
   llvm::sort(ordered, [](const PayloadColumnSpec* a, const PayloadColumnSpec* b) {
      if (a->isJoinKey != b->isJoinKey) return a->isJoinKey > b->isJoinKey;
      return a->semanticKey < b->semanticKey;
   });
   for (PayloadColumnSpec* p : ordered) {
      plan.payloadColumns.push_back(*p);
      plan.payloadMemberTypes.push_back(p->colType);
   }
   return plan;
}

static std::optional<unsigned> parseMemberSlot(llvm::StringRef name) {
   if (!name.consume_front("member$")) return std::nullopt;
   unsigned slot = 0;
   if (name.getAsInteger(10, slot)) return std::nullopt;
   return slot;
}

/// Next `member$N` slot after \p bySemanticKey reuse set and already-assigned payload members.
static unsigned nextPayloadMemberSlot(subop::MemberManager& mm, const llvm::StringMap<subop::Member>& bySemanticKey,
                                        llvm::ArrayRef<subop::Member> assigned) {
   unsigned maxSlot = 0;
   auto bump = [&](subop::Member m) {
      if (auto slot = parseMemberSlot(mm.getName(m))) maxSlot = std::max(maxSlot, *slot + 1);
      if (auto predSlot = parseFilterPredMemberSlot(mm.getName(m))) maxSlot = std::max(maxSlot, *predSlot + 1);
   };
   for (const auto& it : bySemanticKey) bump(it.second);
   for (subop::Member m : assigned) bump(m);
   return maxSlot;
}

static subop::Member allocUnusedPayloadMemberSlot(subop::MemberManager& mm, mlir::Type colType, unsigned& nextSlot) {
   for (;; ++nextSlot) {
      std::string name = "member$" + std::to_string(nextSlot);
      if (!mm.hasMemberDirect(name)) return mm.createMemberDirect(name, colType);
   }
}

static mlir::Type cloneTypeToContext(mlir::Type ty, mlir::MLIRContext* ctx);

static void collectSemanticKeyToMemberFromMaterialize(
   subop::MaterializeOp mat, subop::Member linkM, subop::Member hashM, subop::MemberManager& mm,
   lingodb::compiler::dialect::tuples::ColumnManager& cm, llvm::StringMap<subop::Member>& out) {
   if (!mat) return;
   for (auto& [member, colRef] : mat.getMapping().getMapping()) {
      if (member == linkM || member == hashM) continue;
      auto [scope, leaf] = cm.getName(&colRef.getColumn());
      if (mm.getType(member) != colRef.getColumn().type) continue;
      out.try_emplace(columnSemanticKey(scope, leaf), member);
   }
}

/// Assign \p plan.payloadMembers from cloned build-step materialize mappings + fresh members for union-only columns.
/// Reuses existing buffer members (e.g. \c member$0, \c member$1) from materialize; new union columns get the next
/// \c member$N slot via \c createMemberDirect (not semantic leaf names like \c s_comment$0).
static void assignPayloadMembersForPlan(subop::MemberManager& mm, lingodb::compiler::dialect::tuples::ColumnManager& cm,
                                        subop::MaterializeOp matOp, JoinBufferUnionPlan& plan) {
   llvm::StringMap<subop::Member> bySemanticKey;
   collectSemanticKeyToMemberFromMaterialize(matOp, plan.linkMember, plan.hashMember, mm, cm, bySemanticKey);
   plan.payloadMembers.clear();
   plan.payloadMembers.reserve(plan.payloadColumns.size());
   unsigned nextSlot = nextPayloadMemberSlot(mm, bySemanticKey, plan.payloadMembers);
   if (subop::BufferType bufTy = getInnerBufferTypeForMaterializeState(matOp.getState().getType())) {
      for (subop::Member m : bufTy.getMembers().getMembers()) {
         if (m == plan.linkMember || m == plan.hashMember) continue;
         if (auto slot = parseMemberSlot(mm.getName(m))) nextSlot = std::max(nextSlot, *slot + 1);
         if (auto predSlot = parseFilterPredMemberSlot(mm.getName(m))) nextSlot = std::max(nextSlot, *predSlot + 1);
      }
   }
   mlir::MLIRContext* synthCtx = matOp.getContext();
   subop::Member canonicalPredMember = makeOrGetPredMemberForSlot(matOp.getContext(), 0);
   mlir::Type canonicalPredTy = mm.getType(canonicalPredMember);
   for (const PayloadColumnSpec& spec : plan.payloadColumns) {
      unsigned predIdx = 0;
      if (parseReuseFilterPredSemanticKey(spec.semanticKey, predIdx)) {
         plan.payloadMembers.push_back(mm.getOrCreateMemberDirect(
            "filter_pred$" + llvm::Twine(predIdx).str(), canonicalPredTy, /*allowTypeUpdate=*/false));
         continue;
      }
      if (auto it = bySemanticKey.find(spec.semanticKey); it != bySemanticKey.end()) {
         mlir::Type wantTy = cloneTypeToContext(spec.colType, synthCtx);
         assert(mm.getType(it->second) == wantTy && "join superset: reused member type must match union column");
         plan.payloadMembers.push_back(it->second);
         continue;
      }
      mlir::Type slotTy = cloneTypeToContext(spec.colType, synthCtx);
      plan.payloadMembers.push_back(allocUnusedPayloadMemberSlot(mm, slotTy, nextSlot));
   }
}

static subop::StateMembersAttr bufferMembersForPlan(mlir::MLIRContext* ctx, const JoinBufferUnionPlan& plan) {
   assert(plan.payloadMembers.size() == plan.payloadColumns.size() &&
          "join superset: payload member list must match union columns");
   llvm::SmallVector<subop::Member> members;
   members.push_back(plan.linkMember);
   members.push_back(plan.hashMember);
   members.append(plan.payloadMembers.begin(), plan.payloadMembers.end());
   return subop::StateMembersAttr::get(ctx, members);
}

static void syncCreateHashIndexedViewFromBuffer(subop::CreateHashIndexedView chiv) {
   auto* ctx = chiv.getContext();
   mlir::Value src = chiv.getSource();
   auto bufTy = mlir::dyn_cast<subop::BufferType>(src.getType());
   if (!bufTy) return;
   subop::Member linkM = chiv.getLinkMember().getMember();
   subop::Member hashM = chiv.getHashMember().getMember();
   llvm::SmallVector<subop::Member> vals;
   for (subop::Member m : bufTy.getMembers().getMembers()) {
      if (m == linkM || m == hashM) continue;
      vals.push_back(m);
   }
   auto oldHiv = mlir::cast<subop::HashIndexedViewType>(chiv.getType());
   auto keyMs = subop::StateMembersAttr::get(ctx, llvm::SmallVector<subop::Member>{hashM});
   auto valMs = subop::StateMembersAttr::get(ctx, vals);
   auto newHiv = subop::HashIndexedViewType::get(ctx, keyMs, valMs, oldHiv.getCompareHashForLookup());
   chiv.getResult().setType(newHiv);
}

static void applyBufferLayoutToSsaClosure(mlir::ModuleOp module, llvm::ArrayRef<mlir::Value> canonicalBuffers,
                                          subop::StateMembersAttr targetMembers, const ModuleReuseInfo& reuse) {
   JoinBufferHivSsaClosure joinClosure = computeJoinBufferHivSsaClosure(canonicalBuffers, reuse);
   llvm::DenseSet<void*>& closure = joinClosure.opaque;
   expandClosureThroughExecutionStepPorts(module, closure);

   auto* ctx = module.getContext();
   auto targetBufTy = subop::BufferType::get(ctx, targetMembers);
   auto targetTlTy = subop::ThreadLocalType::get(ctx, mlir::cast<subop::State>(targetBufTy));

   auto widenCarrierValue = [&](mlir::Value v) {
      if (!opaqueClosureContains(closure, v)) return;
      if (mlir::isa<subop::BufferType>(v.getType())) {
         v.setType(targetBufTy);
      } else if (auto tl = mlir::dyn_cast<subop::ThreadLocalType>(v.getType())) {
         if (mlir::isa<subop::BufferType>(tl.getWrapped())) v.setType(targetTlTy);
      }
   };

   module.walk([&](mlir::Operation* op) {
      for (mlir::Value v : op->getOperands()) widenCarrierValue(v);
      for (mlir::Value v : op->getResults()) widenCarrierValue(v);
   });
   module.walk([&](mlir::Operation* op) {
      for (mlir::Region& reg : op->getRegions()) {
         for (mlir::Block& block : reg) {
            for (mlir::BlockArgument a : block.getArguments()) widenCarrierValue(a);
         }
      }
   });

   module.walk([&](subop::MergeOp merge) {
      if (!opaqueClosureContains(closure, merge.getRes()) && !opaqueClosureContains(closure, merge.getThreadLocal()))
         return;
      merge.getRes().setType(targetBufTy);
      merge.getThreadLocal().setType(targetTlTy);
   });
   module.walk([&](subop::GenericCreateOp create) {
      if (!opaqueClosureContains(closure, create.getRes())) return;
      if (mlir::isa<subop::ThreadLocalType>(create.getType()) &&
          mlir::isa<subop::BufferType>(mlir::cast<subop::ThreadLocalType>(create.getType()).getWrapped())) {
         create.getResult().setType(targetTlTy);
      }
   });
   module.walk([&](subop::MaterializeOp mat) {
      if (!opaqueClosureContains(closure, mat.getState())) return;
      if (mlir::isa<subop::BufferType>(mat.getState().getType())) {
         mat.getState().setType(targetBufTy);
      } else if (mlir::isa<subop::ThreadLocalType>(mat.getState().getType()) &&
                 mlir::isa<subop::BufferType>(mlir::cast<subop::ThreadLocalType>(mat.getState().getType()).getWrapped())) {
         mat.getState().setType(targetTlTy);
      }
   });
   module.walk([&](subop::CreateHashIndexedView chiv) {
      if (!opaqueClosureContains(closure, chiv.getSource())) return;
      syncCreateHashIndexedViewFromBuffer(chiv);
   });
   synchronizeExecutionStepPortTypes(module, &closure);
}

static void alignBufferMergeThreadLocalsWithMergeResult(mlir::ModuleOp module,
                                                        const llvm::DenseSet<void*>* closureFilter) {
   auto* ctx = module.getContext();
   module.walk([&](subop::MergeOp merge) {
      if (closureFilter && !opaqueClosureContains(*closureFilter, merge.getRes()) &&
          !opaqueClosureContains(*closureFilter, merge.getThreadLocal())) {
         return;
      }
      auto resBuf = mlir::dyn_cast<subop::BufferType>(merge.getRes().getType());
      if (!resBuf) return;
      auto targetTl = subop::ThreadLocalType::get(ctx, mlir::cast<subop::State>(resBuf));
      llvm::SmallDenseSet<void*> seen;
      llvm::SmallVector<mlir::Value, 16> worklist;
      worklist.push_back(merge.getThreadLocal());
      while (!worklist.empty()) {
         mlir::Value v = worklist.back();
         worklist.pop_back();
         void* k = v.getAsOpaquePointer();
         if (!seen.insert(k).second) continue;
         v.setType(targetTl);
         if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(v)) {
            mlir::Block* owner = ba.getOwner();
            mlir::Operation* parentOp = owner->getParentOp();
            if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(parentOp)) {
               unsigned idx = ba.getArgNumber();
               if (idx < step.getNumOperands()) worklist.push_back(step.getOperand(idx));
            }
         }
      }
   });
}

static ExternalDatasourceProperty mergeExternalDatasource(const ExternalDatasourceProperty& a,
                                                        const ExternalDatasourceProperty& b) {
   assert(a.tableName == b.tableName && "join superset: table name must match");
   ExternalDatasourceProperty out = a;
   std::unordered_set<std::string> have;
   for (const auto& m : out.mapping) have.insert(m.identifier);
   for (const auto& m : b.mapping) {
      if (have.insert(m.identifier).second) out.mapping.push_back(m);
   }
   llvm::sort(out.mapping, [](const auto& x, const auto& y) { return x.memberName < y.memberName; });
   return out;
}

static mlir::Type memberTypeForIdentifier(subop::TableType tableTy, subop::MemberManager& mm,
                                          llvm::StringRef identifier) {
   if (!tableTy) return {};
   llvm::StringRef id = normalizeColumnIdentifier(identifier);
   for (subop::Member m : tableTy.getMembers().getMembers()) {
      if (stripMemberSuffix(mm.getName(m)) == id) return mm.getType(m);
   }
   return {};
}

static subop::Member tableMemberForIdentifier(subop::TableType tableTy, subop::MemberManager& mm,
                                              llvm::StringRef identifier);

static subop::TableType tableTypeFromMergedExternal(mlir::MLIRContext* ctx, subop::MemberManager& mm,
                                                    const ExternalDatasourceProperty& ds, subop::TableType hintTy,
                                                    llvm::function_ref<mlir::Type(llvm::StringRef)> lookupPeerType) {
   llvm::SmallVector<subop::Member> members;
   for (const auto& map : ds.mapping) {
      mlir::Type ty = memberTypeForIdentifier(hintTy, mm, map.identifier);
      if (!ty) ty = lookupPeerType(map.identifier);
      assert(ty && "join superset: missing column type in merged external layout");
      if (subop::Member existing = tableMemberForIdentifier(hintTy, mm, map.identifier)) {
         assert(mm.getType(existing) == ty && "join superset: column type mismatch for reused table member");
         members.push_back(existing);
      } else {
         members.push_back(mm.createMember(map.identifier, ty));
      }
   }
   return subop::TableType::get(ctx, subop::StateMembersAttr::get(ctx, members), hintTy.getFiltered());
}

static subop::Member tableMemberForIdentifier(subop::TableType tableTy, subop::MemberManager& mm,
                                              llvm::StringRef identifier) {
   if (!tableTy) return {};
   llvm::StringRef id = normalizeColumnIdentifier(identifier);
   for (subop::Member m : tableTy.getMembers().getMembers()) {
      if (stripMemberSuffix(mm.getName(m)) == id) return m;
   }
   return {};
}

static llvm::StringRef semanticKeyLeaf(llvm::StringRef semanticKey) {
   return semanticKey.split('\x1f').second;
}

static subop::GatherOp findPeerGatherForSemanticKey(subop::ExecutionStepOp peerBuild, llvm::StringRef semanticKey,
                                                    tuples::ColumnManager& peerCm) {
   llvm::StringRef wantLeaf = normalizeColumnIdentifier(semanticKeyLeaf(semanticKey));
   subop::GatherOp found;
   peerBuild->walk([&](subop::GatherOp g) {
      for (auto& [mem, def] : g.getMapping().getMapping()) {
         auto [scope, leaf] = peerCm.getName(&def.getColumn());
         (void)scope;
         if (normalizeColumnIdentifier(leaf) == wantLeaf) found = g;
      }
   });
   return found;
}

static subop::MapOp findJoinHashMapBeforeMaterialize(mlir::Block& body, subop::MaterializeOp matOp) {
   subop::MapOp mapOp;
   for (mlir::Operation& op : body.without_terminator()) {
      if (!op.isBeforeInBlock(matOp.getOperation())) continue;
      if (auto m = mlir::dyn_cast<subop::MapOp>(&op)) mapOp = m;
   }
   return mapOp;
}

static subop::GatherOp findTableGatherOnStream(mlir::Block& body, mlir::Value stream, subop::MaterializeOp matOp) {
   subop::GatherOp found;
   for (mlir::Operation& op : body.without_terminator()) {
      if (!op.isBeforeInBlock(matOp.getOperation())) continue;
      if (auto g = mlir::dyn_cast<subop::GatherOp>(&op)) {
         if (g.getStream() == stream) found = g;
      }
   }
   return found;
}

static bool extendGatherMapping(subop::GatherOp gather, subop::Member tableMem, tuples::ColumnDefAttr colDef,
                                mlir::MLIRContext* ctx) {
   auto m = gather.getMapping();
   for (auto& [mem, def] : m.getMapping()) {
      if (mem == tableMem) return false;
      (void)def;
   }
   llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> pairs;
   for (auto& p : m.getMapping()) pairs.push_back(p);
   pairs.push_back({tableMem, colDef});
   gather.setMappingAttr(subop::ColumnDefMemberMappingAttr::get(ctx, pairs));
   return true;
}

/// Propagate a widened \c !subop.table type along SSA values and \c execution_step operand/block-arg ports.
static void refreshTableStateTypesInModule(mlir::ModuleOp module, mlir::Value tableStateRoot, subop::TableType newTy) {
   if (!tableStateRoot || !newTy) return;

   llvm::DenseSet<void*> closure;
   llvm::SmallVector<mlir::Value, 16> worklist;
   auto seed = [&](mlir::Value v) {
      if (!v) return;
      closure.insert(v.getAsOpaquePointer());
      worklist.push_back(v);
   };
   seed(tableStateRoot);
   seed(peelBlockArgsToEnclosingOperands(tableStateRoot));

   expandClosureThroughExecutionStepPorts(module, closure);

   llvm::DenseSet<void*> visited;
   while (!worklist.empty()) {
      mlir::Value v = worklist.pop_back_val();
      if (!visited.insert(v.getAsOpaquePointer()).second) continue;
      v.setType(newTy);

      if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(v)) {
         if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(ba.getOwner()->getParentOp())) {
            unsigned i = ba.getArgNumber();
            if (i < step.getNumOperands()) seed(step.getOperand(i));
         }
      } else if (mlir::Operation* def = v.getDefiningOp()) {
         if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(def)) {
            if (auto ret = mlir::dyn_cast<subop::ExecutionStepReturnOp>(step.getSubOps().front().getTerminator())) {
               for (unsigned i = 0; i < ret.getNumOperands() && i < step.getNumResults(); ++i) {
                  if (ret.getOperand(i) == v) seed(step.getResult(i));
               }
            }
         }
      }

      for (mlir::OpOperand& use : v.getUses()) {
         mlir::Operation* user = use.getOwner();
         if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(user)) {
            unsigned i = use.getOperandNumber();
            mlir::Block& body = step.getSubOps().front();
            if (i < body.getNumArguments()) seed(body.getArgument(i));
         }
      }
   }

   synchronizeExecutionStepPortTypes(module, &closure);

   auto* ctx = module.getContext();
   llvm::SmallVector<subop::Member> refCols;
   for (subop::Member m : newTy.getMembers().getMembers()) refCols.push_back(m);
   auto entryRefTy = subop::TableEntryRefType::get(ctx, subop::StateMembersAttr::get(ctx, refCols));
   module.walk([&](subop::ScanRefsOp scan) {
      if (!opaqueClosureContains(closure, scan.getState())) return;
      auto ref = scan.getRef();
      ref.getColumn().type = entryRefTy;
      scan.setRefAttr(ref);
   });
}

static void propagateJoinSupersetColumnAttrs(mlir::ModuleOp module,
                                             const llvm::DenseSet<void*>* closureFilter,
                                             subop::HashIndexedViewType producerHiv, bool syncGatherOps) {
   auto* ctx = module.getContext();
   auto expectedLer = subop::LookupEntryRefType::get(ctx, producerHiv);
   auto expectedListTy = subop::ListType::get(ctx, expectedLer);
   auto shouldUpdateOp = [&](mlir::Operation* op) {
      if (!closureFilter) return true;
      return opOperandsOrNestedBlockArgsTouchClosure(op, *closureFilter);
   };
   auto syncLookupEntryRefToHiv = [&](tuples::ColumnRefAttr cref) -> bool {
      bool changed = false;
      if (mlir::isa<subop::LookupEntryRefType>(cref.getColumn().type)) {
         if (cref.getColumn().type != expectedLer) {
            cref.getColumn().type = expectedLer;
            changed = true;
         }
      } else if (mlir::isa<subop::ListType>(cref.getColumn().type)) {
         if (cref.getColumn().type != expectedListTy) {
            cref.getColumn().type = expectedListTy;
            changed = true;
         }
      }
      return changed;
   };
   auto syncLookupEntryDefToHiv = [&](tuples::ColumnDefAttr def) -> bool {
      if (!mlir::isa<subop::LookupEntryRefType>(def.getColumn().type)) return false;
      if (def.getColumn().type == expectedLer) return false;
      def.getColumn().type = expectedLer;
      return true;
   };
   if (syncGatherOps) {
      module.walk([&](subop::GatherOp op) {
         if (!shouldUpdateOp(op.getOperation())) return;
         auto r = op.getRef();
         bool changed = syncLookupEntryRefToHiv(r);
         auto m = op.getMapping();
         llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> out;
         for (auto [mem, def] : m.getMapping()) {
            tuples::ColumnDefAttr d = def;
            if (syncLookupEntryDefToHiv(d)) changed = true;
            out.push_back({mem, d});
         }
         if (changed) {
            op.setRefAttr(r);
            op.setMappingAttr(subop::ColumnDefMemberMappingAttr::get(ctx, out));
         }
      });
   }
   module.walk([&](subop::MaterializeOp op) {
      if (!shouldUpdateOp(op.getOperation())) return;
      auto m = op.getMapping();
      llvm::SmallVector<std::pair<subop::Member, tuples::ColumnRefAttr>> out;
      bool changed = false;
      for (auto [mem, cref] : m.getMapping()) {
         tuples::ColumnRefAttr c = cref;
         if (syncLookupEntryRefToHiv(c)) changed = true;
         out.push_back({mem, c});
      }
      if (changed) op.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, out));
   });
   module.walk([&](subop::LookupOp op) {
      if (!shouldUpdateOp(op.getOperation())) return;
      if (closureFilter && !opaqueClosureContains(*closureFilter, op.getState())) return;
      auto r = op.getRef();
      if (r.getColumn().type != expectedListTy) {
         r.getColumn().type = expectedListTy;
         op.setRefAttr(r);
      }
   });
   module.walk([&](subop::ScanListOp op) {
      if (!shouldUpdateOp(op.getOperation())) return;
      auto elem = op.getElem();
      if (syncLookupEntryDefToHiv(elem)) op.setElemAttr(elem);
   });
}

/// Resolve the external table backing \c scan_refs: block-arg → step operand → reuse map / \c get_external.
static bool resolveScannedTableExternal(subop::ExecutionStepOp buildStep, mlir::Value tableStateInBody,
                                       const ModuleReuseInfo& reuse, llvm::StringRef& tableName,
                                       ExternalDatasourceProperty& ds, bool& haveDs, subop::TableType& tableTy) {
   mlir::Value external = tableStateInBody;
   if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(external)) {
      if (ba.getOwner() == &buildStep.getSubOps().front() && ba.getArgNumber() < buildStep.getNumOperands()) {
         external = buildStep.getOperand(ba.getArgNumber());
      }
   }
   external = peelBlockArgsToEnclosingOperands(external);

   if (auto ge = mlir::dyn_cast_or_null<subop::GetExternalOp>(external.getDefiningOp())) {
      ds = lingodb::utility::deserializeFromHexString<ExternalDatasourceProperty>(ge.getDescr());
      tableName = ds.tableName;
      haveDs = true;
      tableTy = mlir::cast<subop::TableType>(ge.getResult().getType());
      assert(!tableName.empty());
      return true;
   }

   mlir::Value canon = canonicalizeStateValueForReuse(external);
   if (auto it = findReuseMap(reuse.externalDatasourceByTableState, canon);
       it != reuse.externalDatasourceByTableState.end()) {
      ds = it->second;
      tableName = ds.tableName;
      haveDs = true;
      tableTy = mlir::cast<subop::TableType>(external.getType());
      assert(!tableName.empty());
      return true;
   }

   if (auto tableStep = mlir::dyn_cast_or_null<subop::ExecutionStepOp>(external.getDefiningOp())) {
      for (mlir::Operation& op : tableStep.getSubOps().front().without_terminator()) {
         if (auto ge = mlir::dyn_cast<subop::GetExternalOp>(&op)) {
            ds = lingodb::utility::deserializeFromHexString<ExternalDatasourceProperty>(ge.getDescr());
            tableName = ds.tableName;
            haveDs = true;
            tableTy = mlir::cast<subop::TableType>(ge.getResult().getType());
            assert(!tableName.empty());
            return true;
         }
      }
   }
   return false;
}

static void patchBufferBuildStepForUnion(mlir::ModuleOp synthetic, subop::ExecutionStepOp buildStep,
                                         const JoinBufferUnionPlan& plan, subop::BufferType targetBufTy,
                                         const ModuleReuseInfo& reuseSynthetic,
                                         llvm::ArrayRef<std::pair<mlir::ModuleOp, mlir::Value>> peerHivs,
                                         llvm::ArrayRef<const ModuleReuseInfo*> peerReuses) {
   mlir::Block& body = buildStep.getSubOps().front();
   auto* ctx = synthetic.getContext();
   auto* subDialect = ctx->getLoadedDialect<subop::SubOperatorDialect>();
   auto* tupleDialect = ctx->getLoadedDialect<tuples::TupleStreamDialect>();
   auto& mm = subDialect->getMemberManager();
   auto& cm = tupleDialect->getColumnManager();

   subop::ScanRefsOp scanOp;
   for (mlir::Operation& op : body.without_terminator()) {
      auto s = mlir::dyn_cast<subop::ScanRefsOp>(&op);
      if (!s) continue;
      if (mlir::isa<subop::TableType>(s.getState().getType())) {
         scanOp = s;
         break;
      }
   }
   assert(scanOp && "join superset: buffer build step must contain scan_refs on a table state");

   mlir::Value tableState = scanOp.getState();
   subop::TableType mergedSupplierTableTy;
   ExternalDatasourceProperty mergedDs;
   bool haveDs = false;
   llvm::StringRef donorTableName;
   subop::TableType donorTableTy;
   assert(resolveScannedTableExternal(buildStep, tableState, reuseSynthetic, donorTableName, mergedDs, haveDs,
                                      donorTableTy) &&
          "join superset: scan_refs must resolve to an external table");

   assert(llvm::any_of(plan.payloadColumns, [&](const PayloadColumnSpec& spec) {
             return spec.scope == donorTableName;
          }) &&
          "join superset: union plan must include donor external table scope");

   ExternalTableCatalog syntheticCatalog = buildExternalTableCatalog(synthetic);
   llvm::SmallVector<ExternalTableCatalog, 2> peerCatalogs;
   peerCatalogs.reserve(peerHivs.size());
   for (auto [peerMod, peerHiv] : peerHivs) {
      (void)peerHiv;
      peerCatalogs.push_back(peerMod ? buildExternalTableCatalog(peerMod) : ExternalTableCatalog{});
   }
   for (const ExternalTableCatalog& peerCat : peerCatalogs) {
      mergeExternalDatasourceForTable(mergedDs, haveDs, peerCat, donorTableName);
   }
   mergeExternalDatasourceForTable(mergedDs, haveDs, syntheticCatalog, donorTableName);
   assert(haveDs && "join superset: merged external datasource required for donor table");

   auto lookupPeerColumnType = [&](llvm::StringRef identifier) -> mlir::Type {
      for (const ExternalTableCatalog& peerCat : peerCatalogs) {
         if (mlir::Type ty = columnTypeForIdentifierInCatalog(peerCat, donorTableName, identifier)) return ty;
      }
      return {};
   };
   subop::TableType hintTy = donorTableTy;
   assert(hintTy && "join superset: resolveScannedTableExternal must provide donor table type");
   auto newTableTy = tableTypeFromMergedExternal(ctx, mm, mergedDs, hintTy, lookupPeerColumnType);
   for (auto& map : mergedDs.mapping) {
      if (subop::Member m = tableMemberForIdentifier(newTableTy, mm, map.identifier))
         map.memberName = mm.getName(m);
   }
   llvm::sort(mergedDs.mapping, [](const auto& x, const auto& y) { return x.memberName < y.memberName; });
   std::string hex = lingodb::utility::serializeToHexString(mergedDs);
   auto geIt = syntheticCatalog.byTableName.find(donorTableName);
   assert(geIt != syntheticCatalog.byTableName.end() && !geIt->second.empty() &&
          "join superset: synthetic module must contain get_external for donor table");
   for (ExternalTableCatalog::Entry& entry : geIt->second) {
      entry.op.setDescrAttr(mlir::StringAttr::get(ctx, hex));
      refreshTableStateTypesInModule(synthetic, entry.op.getResult(), newTableTy);
   }

   refreshTableStateTypesInModule(synthetic, tableState, newTableTy);
   mergedSupplierTableTy = newTableTy;

   subop::MaterializeOp matOp;
   buildStep.walk([&](subop::MaterializeOp m) {
      if (getInnerBufferTypeForMaterializeState(m.getState().getType())) matOp = m;
   });
   assert(matOp && "join superset: buffer build step must materialize into join buffer");

   auto collectMaterializedKeys = [&]() {
      std::unordered_set<std::string> keys;
      for (auto& [member, colRef] : matOp.getMapping().getMapping()) {
         if (member == plan.linkMember || member == plan.hashMember) continue;
         auto [scope, leaf] = cm.getName(&colRef.getColumn());
         keys.insert(columnSemanticKey(scope, leaf));
      }
      return keys;
   };
   std::unordered_set<std::string> materializedKeys = collectMaterializedKeys();

   subop::MapOp hashMapOp = findJoinHashMapBeforeMaterialize(body, matOp);
   assert(hashMapOp);

   auto appendMaterializeMember = [&](subop::Member bufMem, tuples::ColumnDefAttr colDef) {
      llvm::SmallVector<subop::RefMappingPairT> matPairs;
      for (auto pr : matOp.getMapping().getMapping()) matPairs.push_back(pr);
      matPairs.push_back({bufMem, cm.createRef(&colDef.getColumn())});
      matOp.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, matPairs));
   };

   auto isPayloadMaterialized = [&](const PayloadColumnSpec& spec) {
      if (materializedKeys.contains(spec.semanticKey)) return true;
      llvm::StringRef wantLeaf = normalizeColumnIdentifier(spec.leaf);
      for (const std::string& k : materializedKeys) {
         if (normalizeColumnIdentifier(semanticKeyLeaf(k)) == wantLeaf) return true;
      }
      return false;
   };

   int payloadSlotInPlan = 0;
   for (const PayloadColumnSpec& spec : plan.payloadColumns) {
      unsigned reusePredQueryIdx = 0;
      if (parseReuseFilterPredSemanticKey(spec.semanticKey, reusePredQueryIdx)) {
         ++payloadSlotInPlan;
         continue;
      }
      if (isPayloadMaterialized(spec)) {
         ++payloadSlotInPlan;
         continue;
      }

      subop::GatherOp templateGather;
      tuples::ColumnManager* templateCm = nullptr;
      for (size_t pi = 0; pi < peerHivs.size(); ++pi) {
         auto [peerMod, peerHiv] = peerHivs[pi];
         if (!peerMod || !peerHiv) continue;
         const ModuleReuseInfo& reusePeer = *peerReuses[pi];
         mlir::Value peerCanon = resolveCacheTargetStateForReuse(peerHiv, reusePeer);
         mlir::Value peerBuf = peerCanon;
         if (auto it = reusePeer.hashIndexedViewFromMergedBuffer.find(peerCanon);
             it != reusePeer.hashIndexedViewFromMergedBuffer.end()) {
            peerBuf = it->second;
         }
         subop::ExecutionStepOp peerBuild = findBufferBuildStepWithTableMaterialize(peerBuf, reusePeer);
         if (!peerBuild) peerBuild = findBufferBuildStepWithTableScan(peerMod);
         if (!peerBuild) continue;
         auto& peerCm = peerMod.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
         templateGather = findPeerGatherForSemanticKey(peerBuild, spec.semanticKey, peerCm);
         if (templateGather) {
            templateCm = &peerCm;
            break;
         }
      }
      subop::TableType tableTy = mergedSupplierTableTy;
      if (!tableTy) tableTy = mlir::dyn_cast<subop::TableType>(scanOp.getState().getType());
      assert(tableTy && "join superset: table type required for payload column gather");
      subop::Member tableMem = tableMemberForIdentifier(tableTy, mm, spec.leaf);
      assert(tableMem && "join superset: union payload column must exist in merged table layout");

      llvm::StringRef scope = spec.scope;
      if (scope.empty() && templateGather && templateCm) {
         auto def0 = templateGather.getMapping().getMapping().begin()->second;
         auto [scope0, leaf0] = templateCm->getName(&def0.getColumn());
         (void)leaf0;
         scope = scope0;
      }
      assert(!scope.empty() && "join superset: payload column scope required");
      tuples::ColumnDefAttr colDef = cm.createDef(spec.scope, spec.leaf);
      colDef.getColumn().type = spec.colType;
      assert(static_cast<size_t>(payloadSlotInPlan) < plan.payloadMembers.size() &&
             "join superset: payload slot out of range");
      subop::Member bufMem = plan.payloadMembers[payloadSlotInPlan];

      mlir::Value mapStream = hashMapOp.getResult();
      mlir::OpBuilder gb(hashMapOp);
      gb.setInsertionPointAfter(hashMapOp);
      auto mapping = subop::ColumnDefMemberMappingAttr::get(
         ctx, llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>>{{tableMem, colDef}});
      auto gatherTy = mapStream.getType();
      auto refDef = scanOp.getRef();
      auto gatherRef = cm.createRef(&refDef.getColumn());
      auto newGather = gb.create<subop::GatherOp>(mlir::UnknownLoc::get(ctx), gatherTy, mapStream,
                                                   gatherRef, mapping);
      matOp->setOperand(0, newGather.getRes());
      appendMaterializeMember(bufMem, colDef);

      materializedKeys.insert(spec.semanticKey);
      ++payloadSlotInPlan;
   }
}

static void applyUnionPlanToSyntheticHiv(mlir::ModuleOp synthetic, mlir::Value syntheticHiv, JoinBufferUnionPlan& plan,
                                         const ModuleReuseInfo& reuseSynthetic,
                                         mlir::ModuleOp query0Module, mlir::Value hivA, const ModuleReuseInfo& reuseA,
                                         mlir::ModuleOp query1Module, mlir::Value hivB,
                                         const ModuleReuseInfo& reuseB) {
   auto* ctx = synthetic.getContext();
   auto* subDialect = ctx->getLoadedDialect<subop::SubOperatorDialect>();
   auto& mm = subDialect->getMemberManager();

   mlir::Value mergedBuf = syntheticHiv;
   if (auto it = reuseSynthetic.hashIndexedViewFromMergedBuffer.find(syntheticHiv);
       it != reuseSynthetic.hashIndexedViewFromMergedBuffer.end()) {
      mergedBuf = it->second;
   }

   subop::ExecutionStepOp buildStep = findBufferBuildStepWithTableMaterialize(mergedBuf, reuseSynthetic);
   subop::MaterializeOp matOp;
   if (buildStep) {
      buildStep.walk([&](subop::MaterializeOp m) {
         if (getInnerBufferTypeForMaterializeState(m.getState().getType())) matOp = m;
      });
   }
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   assignPayloadMembersForPlan(mm, cm, matOp, plan);
   auto targetMembers = bufferMembersForPlan(ctx, plan);
   auto targetBufTy = subop::BufferType::get(ctx, targetMembers);
   llvm::SmallVector<mlir::Value, 4> roots = {mergedBuf};

   applyBufferLayoutToSsaClosure(synthetic, roots, targetMembers, reuseSynthetic);

   if (buildStep) {
      llvm::SmallVector<std::pair<mlir::ModuleOp, mlir::Value>, 2> peerHivs = {
         {query1Module, hivB},
         {query0Module, hivA},
      };
      const ModuleReuseInfo* peerReuses[] = {&reuseB, &reuseA};
      patchBufferBuildStepForUnion(synthetic, buildStep, plan, targetBufTy, reuseSynthetic, peerHivs, peerReuses);
   }

   applyBufferLayoutToSsaClosure(synthetic, roots, targetMembers, reuseSynthetic);
   JoinBufferHivSsaClosure joinClosure = computeJoinBufferHivSsaClosure(roots, reuseSynthetic);
   alignBufferMergeThreadLocalsWithMergeResult(synthetic, &joinClosure.opaque);
   subop::HashIndexedViewType prodHiv;
   for (mlir::Value v : joinClosure.values) {
      if (auto h = mlir::dyn_cast<subop::HashIndexedViewType>(v.getType())) {
         prodHiv = h;
         break;
      }
   }
   if (prodHiv) propagateJoinSupersetColumnAttrs(synthetic, &joinClosure.opaque, prodHiv, true);
   synchronizeExecutionStepPortTypes(synthetic, &joinClosure.opaque);
}

static bool typeEmbedsHashIndexedView(mlir::Type t) {
   if (mlir::isa<subop::HashIndexedViewType>(t)) return true;
   if (auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(t)) return !!ler.getState();
   if (auto list = mlir::dyn_cast<subop::ListType>(t)) return typeEmbedsHashIndexedView(list.getT());
   return false;
}

static bool lookupEntryRefEmbedsHashIndexedView(subop::LookupEntryRefType ler) {
   return mlir::isa<subop::HashIndexedViewType>(ler.getState());
}

static mlir::Type replaceEmbeddedHivInType(mlir::MLIRContext* ctx, mlir::Type t,
                                           subop::HashIndexedViewType producerHiv,
                                           subop::HashIndexedViewType consumerHivBeforeAlign) {
   if (!t) return t;
   if (mlir::isa<subop::HashIndexedViewType>(t)) {
      if (t == producerHiv) return t;
      if (consumerHivBeforeAlign && t != consumerHivBeforeAlign) return t;
      return producerHiv;
   }
   if (auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(t)) {
      if (!lookupEntryRefEmbedsHashIndexedView(ler)) return t;
      if (ler.getState() == producerHiv) return t;
      if (consumerHivBeforeAlign && ler.getState() != consumerHivBeforeAlign) return t;
      return subop::LookupEntryRefType::get(ctx, producerHiv);
   }
   if (auto list = mlir::dyn_cast<subop::ListType>(t)) {
      mlir::Type nt = replaceEmbeddedHivInType(ctx, list.getT(), producerHiv, consumerHivBeforeAlign);
      if (nt == list.getT()) return t;
      return subop::ListType::get(ctx, mlir::cast<subop::StateEntryReference>(nt));
   }
   return t;
}

static void setValueCarrierType(mlir::Value v, subop::HashIndexedViewType producerHiv,
                                subop::HashIndexedViewType consumerHivBeforeAlign) {
   mlir::MLIRContext* ctx = v.getContext();
   if (mlir::Type nt = replaceEmbeddedHivInType(ctx, v.getType(), producerHiv, consumerHivBeforeAlign);
       nt != v.getType()) {
      v.setType(nt);
   }
}

struct ConsumerCachedHivSites {
   llvm::DenseSet<void*> ssaClosure;
   subop::HashIndexedViewType consumerHivBeforeAlign = nullptr;
};

static void addToSsaClosure(mlir::Value v, ConsumerCachedHivSites& sites) {
   if (!v) return;
   sites.ssaClosure.insert(v.getAsOpaquePointer());
}

static bool isJoinProbeCompilerScope(llvm::StringRef scope) { return scope.starts_with("lookup_u_"); }

static CachedJoinBufferLayout layoutFromUnionPlan(subop::HashIndexedViewType producerHiv,
                                                  const JoinBufferUnionPlan& plan) {
   CachedJoinBufferLayout out;
   out.producerHiv = producerHiv;
   out.payloadMembers.assign(plan.payloadMembers.begin(), plan.payloadMembers.end());
   out.payloadColumnTypes.assign(plan.payloadMemberTypes.begin(), plan.payloadMemberTypes.end());
   out.payloadSemanticKeys.reserve(plan.payloadColumns.size());
   for (const PayloadColumnSpec& spec : plan.payloadColumns) {
      out.payloadSemanticKeys.push_back(spec.semanticKey);
   }
   return out;
}

static unsigned operandIndexOf(mlir::Operation* op, mlir::Value v) {
   for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      if (op->getOperand(i) == v) return i;
   }
   llvm_unreachable("operand not found");
}

static mlir::Type cloneTypeToContext(mlir::Type ty, mlir::MLIRContext* ctx) {
   if (!ty || ty.getContext() == ctx) return ty;
   if (auto i = mlir::dyn_cast<mlir::IntegerType>(ty)) {
      return mlir::IntegerType::get(ctx, i.getWidth(), i.getSignedness());
   }
   if (mlir::isa<mlir::IndexType>(ty)) return mlir::IndexType::get(ctx);
   if (auto c = mlir::dyn_cast<db::CharType>(ty)) return db::CharType::get(ctx, c.getLen());
   if (mlir::isa<db::StringType>(ty)) return db::StringType::get(ctx);
   llvm_unreachable("cloneTypeToContext: unsupported type for cross-context layout clone");
}

static subop::Member cloneMemberToContext(subop::Member srcMember, mlir::MLIRContext* srcCtx, mlir::MLIRContext* dstCtx,
                                         bool allowMemberTypeUpdate) {
   if (srcCtx == dstCtx) return srcMember;
   auto& srcMm = srcCtx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto& dstMm = dstCtx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   return dstMm.getOrCreateMemberDirect(srcMm.getName(srcMember), cloneTypeToContext(srcMm.getType(srcMember), dstCtx),
                                        allowMemberTypeUpdate);
}

static void handleScanListSite(subop::ScanListOp scanList, subop::HashIndexedViewType consumerHiv,
                               subop::HashIndexedViewType consumerHivBeforeAlign) {
   setValueCarrierType(scanList.getList(), consumerHiv, consumerHivBeforeAlign);
   auto* ctx = scanList.getContext();
   auto expectedLer = subop::LookupEntryRefType::get(ctx, consumerHiv);
   auto& col = scanList.getElem().getColumn();
   if (mlir::isa<subop::LookupEntryRefType>(col.type) && col.type != expectedLer) col.type = expectedLer;
}

/// From \c cache_get HIV: recurse through port args; at \c scan_list update list/entry-ref types only.
static void traverseConsumerHivUsesFromRoot(mlir::Value root, subop::HashIndexedViewType consumerHiv,
                                            subop::HashIndexedViewType consumerHivBeforeAlign,
                                            ConsumerCachedHivSites& sites) {
   llvm::DenseSet<void*> visited;
   llvm::SmallVector<mlir::Value> worklist;
   auto enqueue = [&](mlir::Value val) {
      if (!val || !visited.insert(val.getAsOpaquePointer()).second) return;
      addToSsaClosure(val, sites);
      worklist.push_back(val);
   };

   setValueCarrierType(root, consumerHiv, consumerHivBeforeAlign);
   enqueue(root);

   while (!worklist.empty()) {
      mlir::Value v = worklist.pop_back_val();
      if (typeEmbedsHashIndexedView(v.getType())) setValueCarrierType(v, consumerHiv, consumerHivBeforeAlign);

      for (mlir::Operation* user : v.getUsers()) {
         if (auto ret = mlir::dyn_cast<subop::ExecutionStepReturnOp>(user)) {
            auto step = mlir::dyn_cast<subop::ExecutionStepOp>(ret->getParentOp());
            if (!step) continue;
            for (unsigned i = 0; i < ret.getNumOperands() && i < step.getNumResults(); ++i) {
               if (ret.getOperand(i) != v) continue;
               setValueCarrierType(step.getResult(i), consumerHiv, consumerHivBeforeAlign);
               enqueue(step.getResult(i));
            }
            continue;
         }

         if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(user)) {
            unsigned i = operandIndexOf(step.getOperation(), v);
            mlir::Block& body = step.getSubOps().front();
            assert(i < body.getNumArguments() && "execution_step operand without block argument");
            mlir::BlockArgument barg = body.getArgument(i);
            setValueCarrierType(barg, consumerHiv, consumerHivBeforeAlign);
            if (i < step.getNumResults()) setValueCarrierType(step.getResult(i), consumerHiv, consumerHivBeforeAlign);
            enqueue(barg);
            if (i < step.getNumResults()) enqueue(step.getResult(i));
            continue;
         }

         if (auto neg = mlir::dyn_cast<subop::NestedExecutionGroupOp>(user)) {
            unsigned i = operandIndexOf(neg.getOperation(), v);
            mlir::Block& body = neg.getSubOps().front();
            assert(i < body.getNumArguments());
            mlir::BlockArgument barg = body.getArgument(i);
            setValueCarrierType(barg, consumerHiv, consumerHivBeforeAlign);
            enqueue(barg);
            continue;
         }

         if (auto lookup = mlir::dyn_cast<subop::LookupOp>(user)) {
            assert(lookup.getState() == v && "lookup state operand must be the HIV value");
            auto* ctx = v.getContext();
            auto listRef = lookup.getRef();
            mlir::Type expectedListTy =
               subop::ListType::get(ctx, subop::LookupEntryRefType::get(ctx, consumerHiv));
            if (listRef.getColumn().type != expectedListTy) listRef.getColumn().type = expectedListTy;
            for (mlir::Operation* streamUser : lookup.getResult().getUsers()) {
               auto nm = mlir::dyn_cast<subop::NestedMapOp>(streamUser);
               assert(nm && "lookup HIV stream must feed nested_map");
               mlir::Region& reg = nm.getRegion();
               assert(!reg.empty());
               for (mlir::BlockArgument barg : reg.front().getArguments()) {
                  if (!typeEmbedsHashIndexedView(barg.getType())) continue;
                  setValueCarrierType(barg, consumerHiv, consumerHivBeforeAlign);
                  enqueue(barg);
               }
            }
            continue;
         }

         if (auto scanList = mlir::dyn_cast<subop::ScanListOp>(user)) {
            assert(scanList.getList() == v && "scan_list list operand must be the carrier value");
            handleScanListSite(scanList, consumerHiv, consumerHivBeforeAlign);
            continue;
         }

         llvm::errs() << "alignConsumer: unhandled HIV use: " << *user << "\n";
         assert(false && "alignConsumer: unhandled HIV/value use (extend traversal)");
      }
   }
}

static void refreshAllEmbeddedHivCarrierTypes(mlir::ModuleOp module, subop::HashIndexedViewType canonicalHiv) {
   auto* ctx = module.getContext();
   for (;;) {
      llvm::SmallVector<std::pair<mlir::Value, mlir::Type>> updates;
      module.walk([&](mlir::Operation* op) {
         auto consider = [&](mlir::Value val) {
            if (!val) return;
            if (mlir::Type nt = replaceEmbeddedHivInType(ctx, val.getType(), canonicalHiv, nullptr);
                nt != val.getType()) {
               updates.push_back({val, nt});
            }
         };
         for (mlir::Value r : op->getResults()) consider(r);
         for (mlir::Region& reg : op->getRegions()) {
            for (mlir::Block& block : reg) {
               for (mlir::BlockArgument a : block.getArguments()) consider(a);
            }
         }
      });
      if (updates.empty()) break;
      for (auto [val, nt] : updates) val.setType(nt);
   }
}

static void syncExecutionStepPortsForModule(mlir::ModuleOp module, const llvm::DenseSet<void*>* closureFilter) {
   synchronizeExecutionStepPortTypes(module, closureFilter);
}

struct ConsumerColumnBinding {
   subop::Member member;
   mlir::Type columnType;
};

/// Resolve a consumer member for a union payload slot from probe-side gather/materialize mappings.
/// Matches by semantic column identity only (not producer union types).
static std::optional<subop::Member> findConsumerMemberForPayloadSlot(
   llvm::StringRef semanticKey, const llvm::StringMap<ConsumerColumnBinding>& semanticToConsumer,
   subop::MemberManager& mm) {
   if (!semanticKey.starts_with("member$")) {
      if (auto it = semanticToConsumer.find(semanticKey); it != semanticToConsumer.end()) {
         return it->second.member;
      }
      llvm::StringRef wantLeaf = normalizeColumnIdentifier(semanticKeyLeaf(semanticKey));
      llvm::SmallVector<subop::Member, 4> byLeaf;
      for (const auto& entry : semanticToConsumer) {
         if (normalizeColumnIdentifier(semanticKeyLeaf(entry.getKey())) != wantLeaf) continue;
         byLeaf.push_back(entry.second.member);
      }
      if (byLeaf.size() == 1) return byLeaf.front();
   }

   // Layout metadata may still carry producer `member$N` labels; fall back to unique probe column types.
   if (semanticKey == "member$0") {
      for (const auto& entry : semanticToConsumer) {
         if (mm.getName(entry.second.member) == "member$0") return entry.second.member;
      }
   }
   return std::nullopt;
}

static std::optional<subop::Member> findConsumerMemberByName(llvm::StringRef name,
                                                             subop::HashIndexedViewType consumerHivBeforeAlign) {
   if (!consumerHivBeforeAlign) return std::nullopt;
   auto& mm = consumerHivBeforeAlign.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   for (subop::Member m : consumerHivBeforeAlign.getValueMembers().getMembers()) {
      if (mm.getName(m) == name) return m;
   }
   return std::nullopt;
}

/// Map semantic table columns to members used on the consumer's cached HIV probe path (pre-align gathers).
static void collectConsumerPayloadSemanticMembers(mlir::ModuleOp consumer,
                                                  llvm::StringMap<ConsumerColumnBinding>& out) {
   auto& cm = consumer.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto record = [&](subop::Member mem, llvm::StringRef scope, llvm::StringRef leaf, mlir::Type colTy) {
      if (isJoinProbeCompilerScope(scope)) return;
      out.try_emplace(columnSemanticKey(scope, leaf), ConsumerColumnBinding{mem, colTy});
   };

   consumer.walk([&](subop::GatherOp gather) {
      auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(gather.getRef().getColumn().type);
      if (!ler || !mlir::isa<subop::HashIndexedViewType>(ler.getState())) return;
      for (auto [mem, def] : gather.getMapping().getMapping()) {
         auto [scope, leaf] = cm.getName(&def.getColumn());
         record(mem, scope, leaf, def.getColumn().type);
      }
   });
   consumer.walk([&](subop::MaterializeOp mat) {
      if (!mlir::isa<subop::BufferType>(mat.getState().getType())) return;
      for (auto& [mem, colRef] : mat.getMapping().getMapping()) {
         auto [scope, leaf] = cm.getName(&colRef.getColumn());
         record(mem, scope, leaf, colRef.getColumn().type);
      }
   });
}

/// Per-consumer HIV for \c cache_get: payload physical order follows the cached producer layout, but each slot
/// reuses the consumer's own \c member$N name and column type when that query already had the column; union-only
/// columns are inserted with producer types under fresh \c member$N slots (no type overwrite on existing members).
static subop::HashIndexedViewType buildConsumerAlignedHivType(mlir::ModuleOp consumer,
                                                                subop::HashIndexedViewType producerHiv,
                                                                const CachedJoinBufferLayout& layout,
                                                                subop::HashIndexedViewType consumerHivBeforeAlign,
                                                                std::optional<unsigned> consumerReuseQueryIndex,
                                                                CachedJoinBufferLayout& outConsumerLayout) {
   auto* ctx = consumer.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();

   llvm::StringMap<ConsumerColumnBinding> semanticToConsumer;
   collectConsumerPayloadSemanticMembers(consumer, semanticToConsumer);

   llvm::DenseMap<subop::Member, subop::Member> producerPayloadToConsumer;
   llvm::DenseSet<subop::Member> excludedProducerPredSlots;
   llvm::StringMap<subop::Member> assignedBySemantic;
   assert(layout.payloadSemanticKeys.size() == layout.payloadMembers.size());

   auto ensureConsumerMember = [&](std::optional<subop::Member> reused, mlir::Type producerSlotTy) -> subop::Member {
      if (reused) {
         mlir::Type consumerTy = mm.getType(*reused);
         return mm.getOrCreateMemberDirect(mm.getName(*reused), consumerTy, /*allowTypeUpdate=*/false);
      }
      llvm::SmallVector<subop::Member, 8> bump;
      if (consumerHivBeforeAlign) {
         for (subop::Member m : consumerHivBeforeAlign.getValueMembers().getMembers()) bump.push_back(m);
      }
      for (const auto& assigned : assignedBySemantic) bump.push_back(assigned.second);
      unsigned nextSlot = nextPayloadMemberSlot(mm, assignedBySemantic, bump);
      mlir::Type slotTy = cloneTypeToContext(producerSlotTy, ctx);
      return allocUnusedPayloadMemberSlot(mm, slotTy, nextSlot);
   };

   for (size_t i = 0; i < layout.payloadSemanticKeys.size(); ++i) {
      mlir::Type producerSlotTy = layout.payloadColumnTypes[i];
      llvm::StringRef semKey = layout.payloadSemanticKeys[i];
      subop::Member producerMem = layout.payloadMembers[i];

      unsigned predIdx = 0;
      if (parseFilterPredLayoutSemanticKey(semKey, predIdx)) {
         if (consumerReuseQueryIndex && predIdx != *consumerReuseQueryIndex) {
            excludedProducerPredSlots.insert(producerMem);
            continue;
         }
         subop::Member consumerMem = makeOrGetPredMemberForSlot(ctx, predIdx);
         assignedBySemantic[semKey] = consumerMem;
         producerPayloadToConsumer[producerMem] = consumerMem;
         continue;
      }

      std::optional<subop::Member> reused = findConsumerMemberForPayloadSlot(semKey, semanticToConsumer, mm);
      if (!reused && i == 0) {
         reused = findConsumerMemberByName("member$0", consumerHivBeforeAlign);
      }
      subop::Member consumerMem = ensureConsumerMember(reused, producerSlotTy);
      assignedBySemantic[semKey] = consumerMem;
      producerPayloadToConsumer[layout.payloadMembers[i]] = consumerMem;
   }

   llvm::SmallVector<subop::Member> valueMembers;
   valueMembers.reserve(producerHiv.getValueMembers().getMembers().size());
   for (subop::Member m : producerHiv.getValueMembers().getMembers()) {
      if (excludedProducerPredSlots.contains(m)) continue;
      if (auto it = producerPayloadToConsumer.find(m); it != producerPayloadToConsumer.end()) {
         valueMembers.push_back(it->second);
      } else if (consumerHivBeforeAlign) {
         if (auto reused = findConsumerMemberByName(mm.getName(m), consumerHivBeforeAlign)) {
            valueMembers.push_back(
               mm.getOrCreateMemberDirect(mm.getName(*reused), mm.getType(*reused), /*allowTypeUpdate=*/false));
         } else {
            valueMembers.push_back(cloneMemberToContext(m, producerHiv.getContext(), ctx, /*allowMemberTypeUpdate=*/false));
         }
      } else {
         valueMembers.push_back(cloneMemberToContext(m, producerHiv.getContext(), ctx, /*allowMemberTypeUpdate=*/false));
      }
   }

   mlir::MLIRContext* producerCtx = producerHiv.getContext();
   llvm::SmallVector<subop::Member> keyMembers;
   for (subop::Member m : producerHiv.getKeyMembers().getMembers()) {
      keyMembers.push_back(cloneMemberToContext(m, producerCtx, ctx, /*allowMemberTypeUpdate=*/false));
   }
   auto consumerHiv = subop::HashIndexedViewType::get(
      ctx, subop::StateMembersAttr::get(ctx, keyMembers), subop::StateMembersAttr::get(ctx, valueMembers),
      producerHiv.getCompareHashForLookup());

   outConsumerLayout = layout;
   outConsumerLayout.producerHiv = consumerHiv;
   outConsumerLayout.payloadMembers.clear();
   outConsumerLayout.payloadColumnTypes.clear();
   outConsumerLayout.payloadSemanticKeys.clear();
   outConsumerLayout.payloadMembers.reserve(layout.payloadMembers.size());
   outConsumerLayout.payloadColumnTypes.reserve(layout.payloadMembers.size());
   outConsumerLayout.payloadSemanticKeys.reserve(layout.payloadMembers.size());
   for (size_t i = 0; i < layout.payloadMembers.size(); ++i) {
      subop::Member m = layout.payloadMembers[i];
      if (excludedProducerPredSlots.contains(m)) continue;
      subop::Member consumerMem;
      if (auto it = producerPayloadToConsumer.find(m); it != producerPayloadToConsumer.end()) {
         consumerMem = it->second;
      } else {
         consumerMem = cloneMemberToContext(m, producerCtx, ctx, /*allowMemberTypeUpdate=*/false);
      }
      outConsumerLayout.payloadMembers.push_back(consumerMem);
      outConsumerLayout.payloadColumnTypes.push_back(mm.getType(consumerMem));
      outConsumerLayout.payloadSemanticKeys.push_back(layout.payloadSemanticKeys[i]);
   }
   return consumerHiv;
}

static void patchSyntheticJoinBufferFilterPredsImpl(
   mlir::ModuleOp synthetic, mlir::ModuleOp query0, mlir::ModuleOp query1,
   llvm::ArrayRef<CrossQueryStateMatchPair> matches, llvm::ArrayRef<CacheTarget> targetsInSynthetic,
   const CachedJoinBufferLayoutsByKey& layoutsByKey) {
   if (!kEnableReuseStateFilterPredReapply) return;

   auto reuse0 = collectModuleReuseInfo(query0);
   auto reuse1 = collectModuleReuseInfo(query1);
   auto reuseSynthetic = collectModuleReuseInfo(synthetic);

   llvm::DenseMap<uint64_t, const CrossQueryStateMatchPair*> matchByKey;
   for (const auto& m : matches) {
      if (!m.stateA || !m.stateB) continue;
      matchByKey[m.cacheKey] = &m;
   }

   for (const CacheTarget& t : targetsInSynthetic) {
      auto itL = layoutsByKey.find(t.cacheKey);
      auto itM = matchByKey.find(t.cacheKey);
      if (itL == layoutsByKey.end() || itM == matchByKey.end()) continue;
      const CachedJoinBufferLayout& layout = itL->second;
      const CrossQueryStateMatchPair& match = *itM->second;

      mlir::Value synthHiv = t.state;
      if (!mlir::isa<subop::HashIndexedViewType>(synthHiv.getType())) continue;
      mlir::Value mergedBuf = synthHiv;
      if (auto it = reuseSynthetic.hashIndexedViewFromMergedBuffer.find(synthHiv);
          it != reuseSynthetic.hashIndexedViewFromMergedBuffer.end()) {
         mergedBuf = it->second;
      }
      subop::ExecutionStepOp buildStep = findBufferBuildStepWithTableMaterialize(mergedBuf, reuseSynthetic);
      if (!buildStep) continue;

      auto& mm = synthetic.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
      mlir::Value hivs[] = {resolveCacheTargetStateForReuse(match.stateA, reuse0),
                            resolveCacheTargetStateForReuse(match.stateB, reuse1)};
      ModuleReuseInfo* reuses[] = {&reuse0, &reuse1};

      for (size_t i = 0; i < layout.payloadSemanticKeys.size(); ++i) {
         unsigned qIdx = 0;
         if (!parseReuseFilterPredSemanticKey(layout.payloadSemanticKeys[i], qIdx)) continue;
         llvm::StringRef predName = mm.getName(layout.payloadMembers[i]);
         auto filters = decodeFiltersFromTableScanInExecutionStep(buildStep);
         if (filters.empty()) {
            filters = decodeFiltersForStateFromWriterSteps(hivs[qIdx], *reuses[qIdx]);
            filters = restrictFiltersToTableScanInExecutionStep(buildStep, filters);
         }
         insertWriteSidePredIntoBufferConstructionStepForPredMember(buildStep, filters, predName);
      }
   }
}

} // namespace

void patchSyntheticJoinBufferFilterPredsFromMatchedQueries(
   mlir::ModuleOp synthetic, mlir::ModuleOp query0, mlir::ModuleOp query1,
   llvm::ArrayRef<CrossQueryStateMatchPair> matches, llvm::ArrayRef<CacheTarget> targetsInSynthetic,
   const CachedJoinBufferLayoutsByKey& layoutsByKey) {
   patchSyntheticJoinBufferFilterPredsImpl(synthetic, query0, query1, matches, targetsInSynthetic, layoutsByKey);
}

void alignConsumerModulesToCachedJoinLayout(mlir::ModuleOp consumer, const CachedJoinBufferLayout& layout,
                                            std::optional<uint64_t> cacheKey,
                                            std::optional<unsigned> consumerReuseQueryIndex) {
   if (!layout.producerHiv) return;

   subop::HashIndexedViewType producerHiv = layout.producerHiv;
   ConsumerCachedHivSites sites;
   subop::HashIndexedViewType consumerHiv = nullptr;

   consumer.walk([&](subop::CacheGetOp get) {
      if (cacheKey && static_cast<uint64_t>(get.getKey()) != *cacheKey) return;

      sites.consumerHivBeforeAlign = mlir::dyn_cast<subop::HashIndexedViewType>(get.getResult().getType());
      CachedJoinBufferLayout consumerLayout;
      consumerHiv = buildConsumerAlignedHivType(consumer, producerHiv, layout, sites.consumerHivBeforeAlign,
                                                  consumerReuseQueryIndex, consumerLayout);
      get.getResult().setType(consumerHiv);
      traverseConsumerHivUsesFromRoot(get.getResult(), consumerHiv, sites.consumerHivBeforeAlign, sites);
   });

   if (sites.ssaClosure.empty() || !consumerHiv) return;

   expandClosureThroughExecutionStepPorts(consumer, sites.ssaClosure);
   syncExecutionStepPortsForModule(consumer, &sites.ssaClosure);
}

void extendSyntheticJoinBuffersToColumnUnion(mlir::ModuleOp synthetic, mlir::ModuleOp query0, mlir::ModuleOp query1,
                                             llvm::ArrayRef<CrossQueryStateMatchPair> matches,
                                             llvm::ArrayRef<CacheTarget> targetsInSynthetic,
                                             const mlir::IRMapping& donorToSynthetic,
                                             CachedJoinBufferLayoutsByKey* outLayouts) {
   (void)donorToSynthetic;
   if (matches.empty() || targetsInSynthetic.empty()) return;

   auto reuse0 = collectModuleReuseInfo(query0);
   auto reuse1 = collectModuleReuseInfo(query1);

   llvm::DenseMap<uint64_t, const CrossQueryStateMatchPair*> matchByKey;
   for (const auto& m : matches) {
      if (!m.stateA || !m.stateB) continue;
      matchByKey[m.cacheKey] = &m;
   }

   for (const CacheTarget& t : targetsInSynthetic) {
      auto itM = matchByKey.find(t.cacheKey);
      if (itM == matchByKey.end()) continue;
      const CrossQueryStateMatchPair& m = *itM->second;

      mlir::Value hivA = resolveCacheTargetStateForReuse(m.stateA, reuse0);
      mlir::Value hivB = resolveCacheTargetStateForReuse(m.stateB, reuse1);
      if (!mlir::isa<subop::HashIndexedViewType>(hivA.getType()) ||
          !mlir::isa<subop::HashIndexedViewType>(hivB.getType())) {
         continue;
      }

      JoinBufferUnionPlan plan = buildUnionPlan(hivA, hivB, reuse0, reuse1, query0, query1);
      if (plan.payloadColumns.empty()) continue;

      unsigned reuseFilterPredSlots = 0;
      for (const PayloadColumnSpec& spec : plan.payloadColumns) {
         unsigned qIdx = 0;
         if (parseReuseFilterPredSemanticKey(spec.semanticKey, qIdx)) ++reuseFilterPredSlots;
      }
      auto fpA = reuse0.joinBuildStoredValueMembersByState.find(hivA);
      auto fpB = reuse1.joinBuildStoredValueMembersByState.find(hivB);
      if (fpA != reuse0.joinBuildStoredValueMembersByState.end() &&
          fpB != reuse1.joinBuildStoredValueMembersByState.end() && fpA->second == fpB->second &&
          reuseFilterPredSlots < 2) {
         continue;
      }

      mlir::Value synthHiv = t.state;
      if (!mlir::isa<subop::HashIndexedViewType>(synthHiv.getType())) continue;

      auto reuseSynthetic = collectModuleReuseInfo(synthetic);
      applyUnionPlanToSyntheticHiv(synthetic, synthHiv, plan, reuseSynthetic, query0, hivA, reuse0, query1, hivB,
                                   reuse1);

      if (outLayouts) {
         mlir::Value canonHiv = synthHiv;
         synthetic.walk([&](subop::CachePutOp put) {
            if (put.getKey() != t.cacheKey) return;
            canonHiv = put.getState();
         });
         if (auto hivTy = mlir::dyn_cast<subop::HashIndexedViewType>(canonHiv.getType())) {
            (*outLayouts)[t.cacheKey] = layoutFromUnionPlan(hivTy, plan);
         }
      }
   }

   if (outLayouts) {
      for (const CacheTarget& t : targetsInSynthetic) {
         if (outLayouts->contains(t.cacheKey)) continue;
         auto hivTy = mlir::dyn_cast<subop::HashIndexedViewType>(t.state.getType());
         if (!hivTy) continue;
         CachedJoinBufferLayout layout;
         layout.producerHiv = hivTy;
         auto& mm = synthetic.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
         for (subop::Member m : hivTy.getValueMembers().getMembers()) {
            layout.payloadMembers.push_back(m);
            layout.payloadColumnTypes.push_back(mm.getType(m));
            layout.payloadSemanticKeys.push_back(std::string(mm.getName(m)));
         }
         (*outLayouts)[t.cacheKey] = std::move(layout);
      }
   }
}

void resyncConsumerCachedHivCarrierTypesFromCacheGet(mlir::ModuleOp consumer,
                                                     std::optional<uint64_t> cacheKey) {
   consumer.walk([&](subop::CacheGetOp get) {
      if (cacheKey && static_cast<uint64_t>(get.getKey()) != *cacheKey) return;
      auto canonicalHiv = mlir::dyn_cast<subop::HashIndexedViewType>(get.getResult().getType());
      if (!canonicalHiv) return;

      ConsumerCachedHivSites sites;
      traverseConsumerHivUsesFromRoot(get.getResult(), canonicalHiv, nullptr, sites);
      expandClosureThroughExecutionStepPorts(consumer, sites.ssaClosure);
      syncExecutionStepPortsForModule(consumer, &sites.ssaClosure);
      // `filter_pred` insertion can touch carriers outside the cache_get use closure.
      refreshAllEmbeddedHivCarrierTypes(consumer, canonicalHiv);
      syncExecutionStepPortsForModule(consumer, nullptr);
   });
}

void refreshCachedJoinLayoutsFromSyntheticCachePuts(mlir::ModuleOp synthetic,
                                                    llvm::ArrayRef<CacheTarget> targetsInSynthetic,
                                                    CachedJoinBufferLayoutsByKey& layoutsByKey) {
   auto& mm = synthetic.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   for (const CacheTarget& t : targetsInSynthetic) {
      synthetic.walk([&](subop::CachePutOp put) {
         if (static_cast<uint64_t>(put.getKey()) != t.cacheKey) return;
         auto hivTy = mlir::dyn_cast<subop::HashIndexedViewType>(put.getState().getType());
         if (!hivTy) return;
         const CachedJoinBufferLayout* prev = nullptr;
         if (auto it = layoutsByKey.find(t.cacheKey); it != layoutsByKey.end()) prev = &it->second;

         CachedJoinBufferLayout layout;
         layout.producerHiv = hivTy;
         size_t idx = 0;
         for (subop::Member m : hivTy.getValueMembers().getMembers()) {
            layout.payloadMembers.push_back(m);
            layout.payloadColumnTypes.push_back(mm.getType(m));
            if (prev && idx < prev->payloadSemanticKeys.size()) {
               layout.payloadSemanticKeys.push_back(prev->payloadSemanticKeys[idx]);
            } else {
               layout.payloadSemanticKeys.push_back(std::string(mm.getName(m)));
            }
            ++idx;
         }
         layoutsByKey[t.cacheKey] = std::move(layout);
      });
   }
}

} // namespace lingodb::compiler::dialect::subop
