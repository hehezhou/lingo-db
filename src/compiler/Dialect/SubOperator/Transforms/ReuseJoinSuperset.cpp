#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseJoinSuperset.h"
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

static void ingestExternalTableColumnsForUnionScopes(mlir::ModuleOp module,
                                                     llvm::StringMap<PayloadColumnSpec>& unionCols);

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
   llvm::StringMap<PayloadColumnSpec>& out) {
   for (auto& [member, colRef] : mat.getMapping().getMapping()) {
      llvm::StringRef memName = mm.getName(member);
      if (memName == linkMemberName || memName == hashMemberName) continue;
      auto [scope, leaf] = cm.getName(&colRef.getColumn());
      PayloadColumnSpec spec;
      spec.scope = scope;
      spec.leaf = leaf;
      spec.colType = colRef.getColumn().type;
      spec.semanticKey = columnSemanticKey(scope, leaf);
      spec.isJoinKey = !joinKeyMemberName.empty() && memName == joinKeyMemberName;
      out.try_emplace(spec.semanticKey, spec);
   }
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
   auto ingestHiv = [&](mlir::Value h, const ModuleReuseInfo& reuse) {
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
         collectPayloadFromMaterialize(mat, linkMemberName, hashMemberName, mm, cm, joinKeyMemberName, unionCols);
      });
   };
   ingestHiv(hivA, reuseA);
   ingestHiv(hivB, reuseB);
   ingestExternalTableColumnsForUnionScopes(modA, unionCols);
   ingestExternalTableColumnsForUnionScopes(modB, unionCols);

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
   };
   for (const auto& it : bySemanticKey) bump(it.second);
   for (subop::Member m : assigned) bump(m);
   return maxSlot;
}

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
   for (const PayloadColumnSpec& spec : plan.payloadColumns) {
      if (auto it = bySemanticKey.find(spec.semanticKey); it != bySemanticKey.end()) {
         assert(mm.getType(it->second) == spec.colType && "join superset: reused member type must match union column");
         plan.payloadMembers.push_back(it->second);
         continue;
      }
      std::string name = "member$" + std::to_string(nextSlot++);
      plan.payloadMembers.push_back(mm.createMemberDirect(name, spec.colType));
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

/// Widen union with full external-table columns only for table scopes already present in \p unionCols
/// (from materialize mappings), keyed by \c ExternalDatasourceProperty::tableName — not a fixed table name.
static void ingestExternalTableColumnsForUnionScopes(mlir::ModuleOp module,
                                                     llvm::StringMap<PayloadColumnSpec>& unionCols) {
   if (!module) return;
   llvm::StringSet<> scopesInUnion;
   for (auto& it : unionCols) {
      if (!it.getValue().scope.empty()) scopesInUnion.insert(it.getValue().scope);
   }
   if (scopesInUnion.empty()) return;
   auto& mm = module.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   module.walk([&](subop::GetExternalOp ge) {
      auto ds = lingodb::utility::deserializeFromHexString<ExternalDatasourceProperty>(ge.getDescr());
      if (!scopesInUnion.contains(ds.tableName)) return;
      auto tableTy = mlir::dyn_cast<subop::TableType>(ge.getResult().getType());
      if (!tableTy) return;
      for (const auto& map : ds.mapping) {
         llvm::StringRef leaf = normalizeColumnIdentifier(map.identifier);
         PayloadColumnSpec spec;
         spec.scope = ds.tableName;
         spec.leaf = leaf.str();
         spec.colType = memberTypeForIdentifier(tableTy, mm, leaf);
         if (!spec.colType) continue;
         spec.semanticKey = columnSemanticKey(spec.scope, spec.leaf);
         unionCols.try_emplace(spec.semanticKey, spec);
      }
   });
}

static mlir::Type columnTypeForIdentifierInModule(mlir::ModuleOp module, llvm::StringRef tableName,
                                                llvm::StringRef identifier) {
   auto& mm = module.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   mlir::Type found;
   module.walk([&](subop::GetExternalOp ge) {
      if (found) return;
      auto ds = lingodb::utility::deserializeFromHexString<ExternalDatasourceProperty>(ge.getDescr());
      if (ds.tableName != tableName) return;
      auto tableTy = mlir::cast<subop::TableType>(ge.getResult().getType());
      if (mlir::Type ty = memberTypeForIdentifier(tableTy, mm, identifier)) found = ty;
   });
   return found;
}

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

static void mergeSupplierExternalFromModule(mlir::ModuleOp module, llvm::StringRef tableName,
                                            ExternalDatasourceProperty& merged, bool& haveMerged) {
   module.walk([&](subop::GetExternalOp ge) {
      auto ds = lingodb::utility::deserializeFromHexString<ExternalDatasourceProperty>(ge.getDescr());
      if (ds.tableName != tableName) return;
      if (!haveMerged) {
         merged = std::move(ds);
         haveMerged = true;
      } else {
         merged = mergeExternalDatasource(merged, ds);
      }
   });
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

static void refreshTableStateTypesInModule(mlir::ModuleOp module, mlir::Value tableState, subop::TableType newTy) {
   tableState.setType(newTy);
   module.walk([&](mlir::Operation* op) {
      for (mlir::OpResult r : op->getResults()) {
         if (r == tableState) r.setType(newTy);
      }
      for (mlir::Region& reg : op->getRegions()) {
         for (mlir::Block& b : reg) {
            for (mlir::BlockArgument a : b.getArguments()) {
               if (a == tableState) a.setType(newTy);
            }
         }
      }
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
   if (!scanOp) return;

   mlir::Value tableState = scanOp.getState();
   subop::TableType mergedSupplierTableTy;
   ExternalDatasourceProperty mergedDs;
   bool haveDs = false;
   llvm::StringRef donorTableName;
   subop::TableType donorTableTy = mlir::dyn_cast<subop::TableType>(tableState.getType());
   if (auto ge = mlir::dyn_cast_or_null<subop::GetExternalOp>(tableState.getDefiningOp())) {
      auto donorDs = lingodb::utility::deserializeFromHexString<ExternalDatasourceProperty>(ge.getDescr());
      donorTableName = donorDs.tableName;
      mergedDs = std::move(donorDs);
      haveDs = true;
   }
   bool unionTouchesDonorTable = false;
   if (!donorTableName.empty()) {
      for (const PayloadColumnSpec& spec : plan.payloadColumns) {
         if (spec.scope == donorTableName) {
            unionTouchesDonorTable = true;
            break;
         }
      }
   }
   if (unionTouchesDonorTable && !donorTableName.empty()) {
      for (auto [peerMod, peerHiv] : peerHivs) {
         if (!peerMod) continue;
         (void)peerHiv;
         mergeSupplierExternalFromModule(peerMod, donorTableName, mergedDs, haveDs);
      }
      if (donorTableTy && haveDs) {
         synthetic.walk([&](subop::GetExternalOp ge) {
            auto donorDs = lingodb::utility::deserializeFromHexString<ExternalDatasourceProperty>(ge.getDescr());
            if (donorDs.tableName != donorTableName) return;
            mergedDs = mergeExternalDatasource(mergedDs, donorDs);
         });
      }
   }
   if (unionTouchesDonorTable && haveDs) {
      auto lookupPeerColumnType = [&](llvm::StringRef identifier) -> mlir::Type {
         for (auto [peerMod, peerHiv] : peerHivs) {
            if (!peerMod) continue;
            (void)peerHiv;
            if (mlir::Type ty = columnTypeForIdentifierInModule(peerMod, donorTableName, identifier)) return ty;
         }
         return {};
      };
      subop::TableType hintTy = donorTableTy;
      if (!hintTy) {
         for (auto [peerMod, peerHiv] : peerHivs) {
            if (!peerMod) continue;
            (void)peerHiv;
            peerMod.walk([&](subop::GetExternalOp ge) {
               if (hintTy) return;
               auto ds = lingodb::utility::deserializeFromHexString<ExternalDatasourceProperty>(ge.getDescr());
               if (ds.tableName != donorTableName) return;
               hintTy = mlir::cast<subop::TableType>(ge.getResult().getType());
            });
            if (hintTy) break;
         }
      }
      assert(hintTy && "join superset: external table type required for external merge");
      auto newTableTy = tableTypeFromMergedExternal(ctx, mm, mergedDs, hintTy, lookupPeerColumnType);
      for (auto& map : mergedDs.mapping) {
         if (subop::Member m = tableMemberForIdentifier(newTableTy, mm, map.identifier))
            map.memberName = mm.getName(m);
      }
      llvm::sort(mergedDs.mapping, [](const auto& x, const auto& y) { return x.memberName < y.memberName; });
      std::string hex = lingodb::utility::serializeToHexString(mergedDs);
      synthetic.walk([&](subop::GetExternalOp ge) {
         auto ds = lingodb::utility::deserializeFromHexString<ExternalDatasourceProperty>(ge.getDescr());
         if (ds.tableName != mergedDs.tableName) return;
         ge.setDescrAttr(mlir::StringAttr::get(ctx, hex));
         refreshTableStateTypesInModule(synthetic, ge.getResult(), newTableTy);
      });

      auto refDef = scanOp.getRef();
      llvm::SmallVector<subop::Member> refCols;
      for (subop::Member m : newTableTy.getMembers().getMembers()) refCols.push_back(m);
      refDef.getColumn().type = subop::TableEntryRefType::get(ctx, subop::StateMembersAttr::get(ctx, refCols));
      scanOp.setRefAttr(refDef);
      mergedSupplierTableTy = newTableTy;
      if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(scanOp->getParentOp())) {
         for (mlir::Value operand : step->getOperands()) {
            if (mlir::isa<subop::TableType>(operand.getType())) operand.setType(newTableTy);
         }
      }
   }

   subop::GatherOp firstTableGather;
   for (mlir::Operation& op : body.without_terminator()) {
      if (!scanOp->isBeforeInBlock(&op)) continue;
      auto g = mlir::dyn_cast<subop::GatherOp>(&op);
      if (!g) continue;
      if (g.getStream() == scanOp.getRes()) {
         firstTableGather = g;
         break;
      }
   }
   subop::MaterializeOp matOp;
   buildStep.walk([&](subop::MaterializeOp m) {
      if (getInnerBufferTypeForMaterializeState(m.getState().getType())) matOp = m;
   });
   if (!matOp) return;

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
      subop::Member tableMem = tableMemberForIdentifier(tableTy, mm, spec.leaf);
      if (!tableMem) {
         ++payloadSlotInPlan;
         continue;
      }

      llvm::StringRef scope = spec.scope;
      if (scope.empty() && templateGather && templateCm) {
         auto def0 = templateGather.getMapping().getMapping().begin()->second;
         auto [scope0, leaf0] = templateCm->getName(&def0.getColumn());
         (void)leaf0;
         scope = scope0;
      }
      if (scope.empty()) {
         ++payloadSlotInPlan;
         continue;
      }
      tuples::ColumnDefAttr colDef = cm.createDef(spec.scope, spec.leaf);
      colDef.getColumn().type = spec.colType;
      assert(static_cast<size_t>(payloadSlotInPlan) < plan.payloadMembers.size() &&
             "join superset: payload slot out of range");
      subop::Member bufMem = plan.payloadMembers[payloadSlotInPlan];

      auto tableScanRefForGather = [&]() -> tuples::ColumnRefAttr {
         if (firstTableGather) return firstTableGather.getRef();
         if (templateGather) return templateGather.getRef();
         auto refDef = scanOp.getRef();
         return cm.createRef(&refDef.getColumn());
      };

      bool grafted = false;
      if (hashMapOp) {
         mlir::Value mapStream = hashMapOp.getResult();
         subop::GatherOp gatherOnMap = findTableGatherOnStream(body, mapStream, matOp);
         if (gatherOnMap && extendGatherMapping(gatherOnMap, tableMem, colDef, ctx)) {
            matOp->setOperand(0, gatherOnMap.getRes());
            appendMaterializeMember(bufMem, colDef);
            grafted = true;
         } else {
            mlir::OpBuilder gb(hashMapOp);
            gb.setInsertionPointAfter(hashMapOp);
            auto mapping = subop::ColumnDefMemberMappingAttr::get(
               ctx, llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>>{{tableMem, colDef}});
            auto gatherTy = firstTableGather ? firstTableGather.getRes().getType() : mapStream.getType();
            auto gatherLoc = templateGather ? templateGather.getLoc() : hashMapOp.getLoc();
            auto newGather = gb.create<subop::GatherOp>(gatherLoc, gatherTy, mapStream, tableScanRefForGather(), mapping);
            matOp->setOperand(0, newGather.getRes());
            appendMaterializeMember(bufMem, colDef);
            grafted = true;
         }
      }

      if (!grafted) {
         mlir::Operation* insertAfter = scanOp;
         for (mlir::Operation& op : body.without_terminator()) {
            if (&op == scanOp.getOperation()) continue;
            if (!scanOp->isBeforeInBlock(&op)) continue;
            if (mlir::isa<subop::GatherOp, subop::MapOp, subop::ReduceOp>(&op) && op.isBeforeInBlock(matOp)) {
               insertAfter = &op;
            }
         }
         mlir::OpBuilder gb(insertAfter);
         gb.setInsertionPointAfter(insertAfter);
         auto mapping = subop::ColumnDefMemberMappingAttr::get(
            ctx, llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>>{{tableMem, colDef}});
         mlir::Value streamIn = insertAfter->getResult(0);
         auto gatherTy = firstTableGather ? firstTableGather.getRes().getType() : streamIn.getType();
         auto gatherLoc = templateGather ? templateGather.getLoc() : scanOp.getLoc();
         auto newGather = gb.create<subop::GatherOp>(gatherLoc, gatherTy, streamIn, tableScanRefForGather(), mapping);
         matOp->setOperand(0, newGather.getRes());
         appendMaterializeMember(bufMem, colDef);
      }

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
   llvm::DenseSet<void*> scanListOps;
   /// Consumer HIV layout before union alignment (from \c cache_get), for LER/HIV type guards.
   subop::HashIndexedViewType consumerHivBeforeAlign = nullptr;
};

static void addToSsaClosure(mlir::Value v, ConsumerCachedHivSites& sites) {
   if (!v) return;
   sites.ssaClosure.insert(v.getAsOpaquePointer());
}

static bool ssaClosureContains(const ConsumerCachedHivSites& sites, mlir::Value v) {
   return v && sites.ssaClosure.contains(v.getAsOpaquePointer());
}

static bool isGatherUnderCachedScanList(subop::GatherOp gather, const ConsumerCachedHivSites& sites) {
   mlir::Block* b = gather->getBlock();
   if (!b) return false;
   for (void* p : sites.scanListOps) {
      auto* scanOp = static_cast<mlir::Operation*>(p);
      if (scanOp->getBlock() != b) continue;
      if (scanOp->isBeforeInBlock(gather.getOperation())) return true;
   }
   return false;
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

static CachedJoinBufferLayout cloneLayoutForContext(const CachedJoinBufferLayout& src, mlir::MLIRContext* dstCtx) {
   if (!src.producerHiv) return {};
   mlir::MLIRContext* srcCtx = src.producerHiv.getContext();
   if (srcCtx == dstCtx) return src;

   subop::HashIndexedViewType srcHiv = src.producerHiv;
   llvm::SmallVector<subop::Member> keyMembers;
   llvm::SmallVector<subop::Member> valMembers;
   const bool allowMemberTypeUpdate = (srcCtx != dstCtx);
   for (subop::Member m : srcHiv.getKeyMembers().getMembers()) {
      keyMembers.push_back(cloneMemberToContext(m, srcCtx, dstCtx, allowMemberTypeUpdate));
   }
   for (subop::Member m : srcHiv.getValueMembers().getMembers()) {
      valMembers.push_back(cloneMemberToContext(m, srcCtx, dstCtx, allowMemberTypeUpdate));
   }

   CachedJoinBufferLayout out;
   out.producerHiv = subop::HashIndexedViewType::get(
      dstCtx, subop::StateMembersAttr::get(dstCtx, keyMembers), subop::StateMembersAttr::get(dstCtx, valMembers),
      srcHiv.getCompareHashForLookup());
   out.payloadSemanticKeys = src.payloadSemanticKeys;
   out.payloadMembers.reserve(src.payloadMembers.size());
   out.payloadColumnTypes.reserve(src.payloadColumnTypes.size());
   for (size_t i = 0; i < src.payloadMembers.size(); ++i) {
      out.payloadMembers.push_back(cloneMemberToContext(src.payloadMembers[i], srcCtx, dstCtx, allowMemberTypeUpdate));
      out.payloadColumnTypes.push_back(cloneTypeToContext(src.payloadColumnTypes[i], dstCtx));
   }
   return out;
}

/// Sync lookup-entry-ref column types only on \c scan_list / gather tied to \p sites.
static void syncConsumerLookupEntryRefColumnTypes(mlir::ModuleOp consumer, subop::HashIndexedViewType producerHiv,
                                                  const ConsumerCachedHivSites& sites) {
   auto* ctx = consumer.getContext();
   auto expectedLer = subop::LookupEntryRefType::get(ctx, producerHiv);
   consumer.walk([&](subop::ScanListOp scan) {
      if (!ssaClosureContains(sites, scan.getList())) return;
      auto& col = scan.getElem().getColumn();
      auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(col.type);
      if (!ler || !lookupEntryRefEmbedsHashIndexedView(ler)) return;
      if (col.type != expectedLer) col.type = expectedLer;
   });
   consumer.walk([&](subop::GatherOp gather) {
      if (!isGatherUnderCachedScanList(gather, sites)) return;
      auto& col = gather.getRef().getColumn();
      auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(col.type);
      if (!ler || !lookupEntryRefEmbedsHashIndexedView(ler)) return;
      if (col.type != expectedLer) col.type = expectedLer;
   });
}

static void syncConsumerLookupListColumnAttrs(mlir::ModuleOp consumer, subop::HashIndexedViewType producerHiv,
                                            const ConsumerCachedHivSites& sites) {
   auto* ctx = consumer.getContext();
   auto expectedListTy =
      subop::ListType::get(ctx, subop::LookupEntryRefType::get(ctx, producerHiv));
   consumer.walk([&](subop::LookupOp op) {
      if (!ssaClosureContains(sites, op.getState())) return;
      if (op.getRef().getColumn().type != expectedListTy) op.getRef().getColumn().type = expectedListTy;
   });
}

static void remapSupplierPayloadGatherMembersInBlock(mlir::MLIRContext* ctx, mlir::Block& block,
                                                      const CachedJoinBufferLayout& layout,
                                                      const ConsumerCachedHivSites& sites) {
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   llvm::StringMap<subop::Member> semanticToMember;
   assert(layout.payloadSemanticKeys.size() == layout.payloadMembers.size());
   for (size_t i = 0; i < layout.payloadSemanticKeys.size(); ++i) {
      semanticToMember[layout.payloadSemanticKeys[i]] = layout.payloadMembers[i];
   }
   block.walk([&](subop::GatherOp gather) {
      if (!isGatherUnderCachedScanList(gather, sites)) return;
      if (!mlir::isa<subop::LookupEntryRefType>(gather.getRef().getColumn().type)) return;
      auto mapping = gather.getMapping();
      llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> out;
      bool changed = false;
      for (auto [mem, def] : mapping.getMapping()) {
         subop::Member newMem = mem;
         auto [scope, leaf] = cm.getName(&def.getColumn());
         if (!isJoinProbeCompilerScope(scope)) {
            if (auto it = semanticToMember.find(columnSemanticKey(scope, leaf));
                it != semanticToMember.end() && newMem != it->second) {
               newMem = it->second;
               changed = true;
            }
         }
         out.push_back({newMem, def});
      }
      if (changed) gather.setMappingAttr(subop::ColumnDefMemberMappingAttr::get(ctx, out));
   });
}

static void remapSupplierPayloadGatherMembers(mlir::ModuleOp consumer, const CachedJoinBufferLayout& layout,
                                              const ConsumerCachedHivSites& sites) {
   auto* ctx = consumer.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   llvm::StringMap<subop::Member> semanticToMember;
   assert(layout.payloadSemanticKeys.size() == layout.payloadMembers.size());
   for (size_t i = 0; i < layout.payloadSemanticKeys.size(); ++i) {
      semanticToMember[layout.payloadSemanticKeys[i]] = layout.payloadMembers[i];
   }
   consumer.walk([&](subop::GatherOp gather) {
      if (!isGatherUnderCachedScanList(gather, sites)) return;
      if (!mlir::isa<subop::LookupEntryRefType>(gather.getRef().getColumn().type)) return;
      auto mapping = gather.getMapping();
      llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> out;
      bool changed = false;
      for (auto [mem, def] : mapping.getMapping()) {
         subop::Member newMem = mem;
         auto [scope, leaf] = cm.getName(&def.getColumn());
         if (!isJoinProbeCompilerScope(scope)) {
            if (auto it = semanticToMember.find(columnSemanticKey(scope, leaf));
                it != semanticToMember.end() && newMem != it->second) {
               newMem = it->second;
               changed = true;
            }
         }
         out.push_back({newMem, def});
      }
      if (changed) gather.setMappingAttr(subop::ColumnDefMemberMappingAttr::get(ctx, out));
   });
}

static void handleScanListSite(subop::ScanListOp scanList, subop::HashIndexedViewType producerHiv,
                               subop::HashIndexedViewType consumerHivBeforeAlign, const CachedJoinBufferLayout& layout,
                               ConsumerCachedHivSites& sites) {
   setValueCarrierType(scanList.getList(), producerHiv, consumerHivBeforeAlign);
   sites.scanListOps.insert(scanList.getOperation());
   remapSupplierPayloadGatherMembersInBlock(scanList->getContext(), *scanList->getBlock(), layout, sites);
}

/// Walk every use of cached HIV SSA: execution_step / nested_execution_group ports (recurse),
/// \c scan_list (fix types + gather slots), \c lookup (list carrier). Other uses assert.
static void buildConsumerCachedHivSitesFromRoot(mlir::Value root, subop::HashIndexedViewType producerHiv,
                                                subop::HashIndexedViewType consumerHivBeforeAlign,
                                                const CachedJoinBufferLayout& layout, ConsumerCachedHivSites& sites) {
   llvm::DenseSet<void*> visited;
   llvm::SmallVector<mlir::Value> worklist;
   auto enqueue = [&](mlir::Value val) {
      if (!val || !visited.insert(val.getAsOpaquePointer()).second) return;
      addToSsaClosure(val, sites);
      worklist.push_back(val);
   };

   setValueCarrierType(root, producerHiv, consumerHivBeforeAlign);
   enqueue(root);

   while (!worklist.empty()) {
      mlir::Value v = worklist.pop_back_val();
      if (typeEmbedsHashIndexedView(v.getType())) setValueCarrierType(v, producerHiv, consumerHivBeforeAlign);

      for (mlir::Operation* user : v.getUsers()) {
         if (auto ret = mlir::dyn_cast<subop::ExecutionStepReturnOp>(user)) {
            auto step = mlir::dyn_cast<subop::ExecutionStepOp>(ret->getParentOp());
            if (!step) continue;
            for (unsigned i = 0; i < ret.getNumOperands() && i < step.getNumResults(); ++i) {
               if (ret.getOperand(i) != v) continue;
               setValueCarrierType(step.getResult(i), producerHiv, consumerHivBeforeAlign);
               enqueue(step.getResult(i));
            }
            continue;
         }

         if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(user)) {
            unsigned i = operandIndexOf(step.getOperation(), v);
            mlir::Block& body = step.getSubOps().front();
            assert(i < body.getNumArguments() && "execution_step operand without block argument");
            mlir::BlockArgument barg = body.getArgument(i);
            setValueCarrierType(barg, producerHiv, consumerHivBeforeAlign);
            if (i < step.getNumResults()) setValueCarrierType(step.getResult(i), producerHiv, consumerHivBeforeAlign);
            enqueue(barg);
            if (i < step.getNumResults()) enqueue(step.getResult(i));
            continue;
         }

         if (auto neg = mlir::dyn_cast<subop::NestedExecutionGroupOp>(user)) {
            unsigned i = operandIndexOf(neg.getOperation(), v);
            mlir::Block& body = neg.getSubOps().front();
            assert(i < body.getNumArguments());
            mlir::BlockArgument barg = body.getArgument(i);
            setValueCarrierType(barg, producerHiv, consumerHivBeforeAlign);
            enqueue(barg);
            continue;
         }

         if (auto lookup = mlir::dyn_cast<subop::LookupOp>(user)) {
            assert(lookup.getState() == v && "lookup state operand must be the HIV value");
            auto* ctx = v.getContext();
            auto listRef = lookup.getRef();
            mlir::Type expectedListTy =
               subop::ListType::get(ctx, subop::LookupEntryRefType::get(ctx, producerHiv));
            if (listRef.getColumn().type != expectedListTy) listRef.getColumn().type = expectedListTy;
            for (mlir::Operation* streamUser : lookup.getResult().getUsers()) {
               auto nm = mlir::dyn_cast<subop::NestedMapOp>(streamUser);
               assert(nm && "lookup HIV stream must feed nested_map");
               mlir::Region& reg = nm.getRegion();
               assert(!reg.empty());
               for (mlir::BlockArgument barg : reg.front().getArguments()) {
                  if (!typeEmbedsHashIndexedView(barg.getType())) continue;
                  setValueCarrierType(barg, producerHiv, consumerHivBeforeAlign);
                  enqueue(barg);
               }
            }
            continue;
         }

         if (auto scanList = mlir::dyn_cast<subop::ScanListOp>(user)) {
            assert(scanList.getList() == v && "scan_list list operand must be the carrier value");
            handleScanListSite(scanList, producerHiv, consumerHivBeforeAlign, layout, sites);
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

static void refreshEmbeddedHivCarrierTypesInClosure(mlir::ModuleOp module, subop::HashIndexedViewType producerHiv,
                                                    const ConsumerCachedHivSites& sites) {
   auto* ctx = module.getContext();
   auto refreshValue = [&](mlir::Value val) {
      if (!ssaClosureContains(sites, val)) return;
      if (mlir::Type nt = replaceEmbeddedHivInType(ctx, val.getType(), producerHiv, nullptr);
          nt != val.getType()) {
         val.setType(nt);
      }
   };
   module.walk([&](mlir::Operation* op) {
      if (!opOperandsOrNestedBlockArgsTouchClosure(op, sites.ssaClosure)) return;
      for (mlir::Value r : op->getResults()) refreshValue(r);
      for (mlir::Region& reg : op->getRegions()) {
         for (mlir::Block& block : reg) {
            for (mlir::BlockArgument a : block.getArguments()) refreshValue(a);
         }
      }
   });
   for (void* p : sites.ssaClosure) {
      refreshValue(mlir::Value::getFromOpaquePointer(p));
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
static std::optional<subop::Member> findConsumerMemberForPayloadSlot(
   llvm::StringRef semanticKey, mlir::Type slotTy,
   const llvm::StringMap<ConsumerColumnBinding>& semanticToConsumer, subop::MemberManager& mm) {
   auto matchesSlot = [&](const ConsumerColumnBinding& b) { return b.columnType == slotTy; };

   if (!semanticKey.starts_with("member$")) {
      if (auto it = semanticToConsumer.find(semanticKey); it != semanticToConsumer.end()) {
         if (matchesSlot(it->second)) return it->second.member;
      }
      llvm::StringRef wantLeaf = normalizeColumnIdentifier(semanticKeyLeaf(semanticKey));
      for (const auto& entry : semanticToConsumer) {
         if (normalizeColumnIdentifier(semanticKeyLeaf(entry.getKey())) != wantLeaf) continue;
         if (matchesSlot(entry.second)) return entry.second.member;
      }
   }

   // Layout metadata may still carry producer `member$N` labels; fall back to unique probe column types.
   llvm::SmallVector<subop::Member, 4> byType;
   for (const auto& entry : semanticToConsumer) {
      if (matchesSlot(entry.second)) byType.push_back(entry.second.member);
   }
   if (byType.size() == 1) return byType.front();
   if (byType.size() > 1 && semanticKey == "member$0") {
      for (subop::Member m : byType) {
         if (mm.getName(m) == "member$0") return m;
      }
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

/// Per-consumer HIV for \c cache_get: same payload column types and physical order as the producer union layout,
/// but member names follow the consumer's pre-reuse HIV when that query already had the column, otherwise a fresh
/// \c member$N slot.
static subop::HashIndexedViewType buildConsumerAlignedHivType(mlir::ModuleOp consumer,
                                                                subop::HashIndexedViewType producerHiv,
                                                                const CachedJoinBufferLayout& layout,
                                                                subop::HashIndexedViewType consumerHivBeforeAlign,
                                                                CachedJoinBufferLayout& outConsumerLayout) {
   auto* ctx = consumer.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();

   llvm::StringMap<ConsumerColumnBinding> semanticToConsumer;
   collectConsumerPayloadSemanticMembers(consumer, semanticToConsumer);

   llvm::DenseMap<subop::Member, subop::Member> producerPayloadToConsumer;
   llvm::StringMap<subop::Member> assignedBySemantic;
   assert(layout.payloadSemanticKeys.size() == layout.payloadMembers.size());

   for (size_t i = 0; i < layout.payloadSemanticKeys.size(); ++i) {
      mlir::Type ty = layout.payloadColumnTypes[i];
      llvm::StringRef semKey = layout.payloadSemanticKeys[i];
      if (semKey.starts_with("filter_pred")) continue;
      subop::Member consumerMem;
      std::optional<subop::Member> reused;
      // Union plan always places the join-key column first (i32 suppkey for supplier HIV).
      if (i == 0 && mlir::isa<mlir::IntegerType>(ty)) {
         consumerMem = mm.getOrCreateMemberDirect("member$0", ty, /*allowTypeUpdate=*/true);
      } else {
      reused =
         findConsumerMemberForPayloadSlot(semKey, ty, semanticToConsumer, mm);
      if (reused) {
         consumerMem = mm.getOrCreateMemberDirect(mm.getName(*reused), ty, /*allowTypeUpdate=*/true);
      } else {
         llvm::SmallVector<subop::Member, 8> bump;
         if (consumerHivBeforeAlign) {
            for (subop::Member m : consumerHivBeforeAlign.getValueMembers().getMembers()) bump.push_back(m);
         }
         for (const auto& assigned : assignedBySemantic) bump.push_back(assigned.second);
         unsigned nextSlot = nextPayloadMemberSlot(mm, assignedBySemantic, bump);
         consumerMem = mm.createMemberDirect("member$" + std::to_string(nextSlot), ty);
      }
      }
      assignedBySemantic[semKey] = consumerMem;
      producerPayloadToConsumer[layout.payloadMembers[i]] = consumerMem;
   }

   llvm::SmallVector<subop::Member> valueMembers;
   valueMembers.reserve(producerHiv.getValueMembers().getMembers().size());
   for (subop::Member m : producerHiv.getValueMembers().getMembers()) {
      if (auto it = producerPayloadToConsumer.find(m); it != producerPayloadToConsumer.end()) {
         valueMembers.push_back(it->second);
      } else {
         valueMembers.push_back(m);
      }
   }

   mlir::MLIRContext* producerCtx = producerHiv.getContext();
   llvm::SmallVector<subop::Member> keyMembers;
   for (subop::Member m : producerHiv.getKeyMembers().getMembers()) {
      keyMembers.push_back(cloneMemberToContext(m, producerCtx, ctx, /*allowMemberTypeUpdate=*/true));
   }
   auto consumerHiv = subop::HashIndexedViewType::get(
      ctx, subop::StateMembersAttr::get(ctx, keyMembers), subop::StateMembersAttr::get(ctx, valueMembers),
      producerHiv.getCompareHashForLookup());

   outConsumerLayout = layout;
   outConsumerLayout.producerHiv = consumerHiv;
   outConsumerLayout.payloadMembers.clear();
   outConsumerLayout.payloadMembers.reserve(layout.payloadMembers.size());
   for (subop::Member m : layout.payloadMembers) {
      outConsumerLayout.payloadMembers.push_back(producerPayloadToConsumer[m]);
   }
   return consumerHiv;
}

} // namespace

void alignConsumerModulesToCachedJoinLayout(mlir::ModuleOp consumer, const CachedJoinBufferLayout& layout,
                                            std::optional<uint64_t> cacheKey) {
   if (!layout.producerHiv) return;

   // Keep producer \c Member handles in the synthetic/producer context only. Cloning payload members into the
   // consumer \c MemberManager would reuse names like \c member$1 with conflicting types across queries.
   subop::HashIndexedViewType producerHiv = layout.producerHiv;
   ConsumerCachedHivSites sites;
   CachedJoinBufferLayout consumerLayout;
   subop::HashIndexedViewType consumerHiv = nullptr;

   consumer.walk([&](subop::CacheGetOp get) {
      if (cacheKey && static_cast<uint64_t>(get.getKey()) != *cacheKey) return;

      sites.consumerHivBeforeAlign = mlir::dyn_cast<subop::HashIndexedViewType>(get.getResult().getType());
      consumerHiv = buildConsumerAlignedHivType(consumer, producerHiv, layout, sites.consumerHivBeforeAlign,
                                                consumerLayout);
      get.getResult().setType(consumerHiv);
      buildConsumerCachedHivSitesFromRoot(get.getResult(), consumerHiv, nullptr, consumerLayout, sites);
   });

   if (sites.ssaClosure.empty() || !consumerHiv) return;

   expandClosureThroughExecutionStepPorts(consumer, sites.ssaClosure);

   // Force every embedded HIV/LER in the cached closure to `consumerHiv`. Intermediate layouts
   // (e.g. filter_pred-extended pre-align carriers on nested_map block args) must not survive.
   refreshEmbeddedHivCarrierTypesInClosure(consumer, consumerHiv, sites);
   refreshAllEmbeddedHivCarrierTypes(consumer, consumerHiv);
   syncExecutionStepPortsForModule(consumer, &sites.ssaClosure);
   syncConsumerLookupListColumnAttrs(consumer, consumerHiv, sites);
   syncConsumerLookupEntryRefColumnTypes(consumer, consumerHiv, sites);
   remapSupplierPayloadGatherMembers(consumer, consumerLayout, sites);
   syncExecutionStepPortsForModule(consumer, &sites.ssaClosure);
   refreshAllEmbeddedHivCarrierTypes(consumer, consumerHiv);
   syncExecutionStepPortsForModule(consumer, nullptr);
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

      auto fpA = reuse0.joinBuildStoredValueMembersByState.find(hivA);
      auto fpB = reuse1.joinBuildStoredValueMembersByState.find(hivB);
      if (fpA != reuse0.joinBuildStoredValueMembersByState.end() &&
          fpB != reuse1.joinBuildStoredValueMembersByState.end() && fpA->second == fpB->second) {
         continue;
      }

      mlir::Value synthHiv = t.state;
      if (!mlir::isa<subop::HashIndexedViewType>(synthHiv.getType())) continue;

      JoinBufferUnionPlan plan = buildUnionPlan(hivA, hivB, reuse0, reuse1, query0, query1);
      if (plan.payloadColumns.empty()) continue;

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
   (void)collectModuleReuseInfo(consumer);
   consumer.walk([&](subop::CacheGetOp get) {
      if (cacheKey && static_cast<uint64_t>(get.getKey()) != *cacheKey) return;
      auto canonicalHiv = mlir::dyn_cast<subop::HashIndexedViewType>(get.getResult().getType());
      if (!canonicalHiv) return;

      // Probe `filter_pred` insertion can leave narrow pre-align HIV types on `nested_map` block
      // arguments outside the cache_get backward closure; refresh the whole module.
      refreshAllEmbeddedHivCarrierTypes(consumer, canonicalHiv);
      syncExecutionStepPortsForModule(consumer, nullptr);
      auto* ctx = consumer.getContext();
      auto expectedLer = subop::LookupEntryRefType::get(ctx, canonicalHiv);
      auto expectedListTy = subop::ListType::get(ctx, expectedLer);
      consumer.walk([&](subop::LookupOp op) {
         if (op.getRef().getColumn().type != expectedListTy) op.getRef().getColumn().type = expectedListTy;
      });
      consumer.walk([&](subop::ScanListOp scan) {
         auto& col = scan.getElem().getColumn();
         if (col.type != expectedLer) col.type = expectedLer;
      });
      consumer.walk([&](subop::GatherOp gather) {
         if (!mlir::isa<subop::LookupEntryRefType>(gather.getRef().getColumn().type)) return;
         auto& col = gather.getRef().getColumn();
         if (col.type != expectedLer) col.type = expectedLer;
      });
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
