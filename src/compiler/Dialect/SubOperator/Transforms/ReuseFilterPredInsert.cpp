#include "lingodb/compiler/Dialect/SubOperator/Transforms/CrossQueryStateReuse.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseRewriteCommon.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseFilterPredInsert.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseStateClosure.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsAttributes.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateExtraction.h"
#include "lingodb/compiler/Dialect/SubOperator/Utils.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/runtime/ExternalDataSourceProperty.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseSet.h"
#include <cassert>

namespace lingodb::compiler::dialect::subop {

subop::Member makeOrGetPredMember(mlir::MLIRContext* ctx) {
   auto* d = ctx->getLoadedDialect<subop::SubOperatorDialect>();
   assert(d && "SubOperatorDialect must be loaded");
   auto& mm = d->getMemberManager();
   // Stable name. If it already exists, type must match.
   return mm.createMemberDirect("filter_pred$0", mlir::IntegerType::get(ctx, 1));
}

/// `subop.materialize` may write into `!subop.buffer<...>` or `!subop.thread_local<!subop.buffer<...>>`.
static subop::BufferType getInnerBufferTypeForMaterializeState(mlir::Type stateTy) {
   if (auto b = mlir::dyn_cast<subop::BufferType>(stateTy)) return b;
   if (auto tl = mlir::dyn_cast<subop::ThreadLocalType>(stateTy))
      return mlir::dyn_cast<subop::BufferType>(tl.getWrapped());
   return nullptr;
}

static subop::StateMembersAttr appendMember(mlir::MLIRContext* ctx, subop::StateMembersAttr members, subop::Member m) {
   llvm::SmallVector<subop::Member> ms;
   for (auto x : members.getMembers()) ms.push_back(x);
   ms.push_back(m);
   return subop::StateMembersAttr::get(ctx, std::move(ms));
}

bool valueMembersContainMemberNamed(mlir::MLIRContext* ctx, subop::StateMembersAttr members,
                                           llvm::StringRef name) {
   auto* d = ctx->getLoadedDialect<subop::SubOperatorDialect>();
   assert(d);
   auto& mm = d->getMemberManager();
   for (auto m : members.getMembers()) {
      if (mm.getName(m) == name) return true;
   }
   return false;
}

static mlir::Type extendHashMapTypeWithPred(mlir::Type t, subop::Member predMember);
static subop::HashIndexedViewType extendHashIndexedViewWithPredMemberIfMissing(mlir::MLIRContext* ctx,
                                                                              subop::HashIndexedViewType hiv,
                                                                              subop::Member predMember);
static void syncCreateHashIndexedViewResultType(subop::CreateHashIndexedView chiv) {
   auto* ctx = chiv.getContext();
   mlir::Value src = chiv.getSource();
   auto bufTy = mlir::dyn_cast<subop::BufferType>(src.getType());
   if (!bufTy) return;
   if (!valueMembersContainMemberNamed(ctx, bufTy.getMembers(), "filter_pred$0")) return;
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

/// `applyGlobalHashMapPredLayout` updates SSA `HashMapType` values but not every nested
/// `hash_map_entry_ref<...>` / `lookup_entry_ref<...>` carried by tuple column attrs. `reduce` /
/// `gather` / `scatter` / `lookup` still read the old map layout from those attrs while the runtime
/// buffer matches the extended map — lowering then loads the wrong struct field (e.g. i8 vs ptr)
/// and leaves an unreconcilable `builtin.unrealized_conversion_cast` before LLVM translation.
static subop::HashMapType getHashMapTypeForStateValue(mlir::Value v) {
   if (auto hm = mlir::dyn_cast<subop::HashMapType>(v.getType())) return hm;
   if (auto tl = mlir::dyn_cast<subop::ThreadLocalType>(v.getType()))
      return mlir::dyn_cast<subop::HashMapType>(tl.getWrapped());
   return {};
}

/// Join pipelines are keyed by `HashMapType::key_members` + lock flag. After extending a specific
/// join map with `filter_pred$0`, every `lookup_entry_ref` / `hash_map_entry_ref` in the same
/// `execution_step` body that targets the **same join key layout** must use the **same** canonical
/// `HashMapType` object — otherwise `reduce`/`gather` lowering builds mismatched `EntryStorageHelper`
/// layouts vs the actual merged hash state and LLVM ends up with `i8` vs `!llvm.ptr` bridges.
static bool hashMapJoinKeyMatches(subop::HashMapType a, subop::HashMapType b) {
   return a.getKeyMembers() == b.getKeyMembers() && a.getWithLock() == b.getWithLock();
}

static bool alignJoinHashMapColumnRefToCanonical(tuples::ColumnRefAttr& r, subop::HashMapType canonicalHm) {
   auto* ctx = canonicalHm.getContext();
   bool changed = false;
   if (auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(r.getColumn().type)) {
      if (auto hm = mlir::dyn_cast<subop::HashMapType>(ler.getState())) {
         if (hashMapJoinKeyMatches(hm, canonicalHm)) {
            r.getColumn().type = subop::LookupEntryRefType::get(ctx, canonicalHm);
            changed = true;
         }
      }
   }
   if (auto hmer = mlir::dyn_cast<subop::HashMapEntryRefType>(r.getColumn().type)) {
      if (hashMapJoinKeyMatches(hmer.getHashMap(), canonicalHm)) {
         r.getColumn().type = subop::HashMapEntryRefType::get(ctx, canonicalHm);
         changed = true;
      }
   }
   return changed;
}

static bool alignJoinHashMapColumnDefToCanonical(tuples::ColumnDefAttr& def, subop::HashMapType canonicalHm) {
   auto* ctx = canonicalHm.getContext();
   bool changed = false;
   if (auto hmer = mlir::dyn_cast<subop::HashMapEntryRefType>(def.getColumn().type)) {
      if (hashMapJoinKeyMatches(hmer.getHashMap(), canonicalHm)) {
         def.getColumn().type = subop::HashMapEntryRefType::get(ctx, canonicalHm);
         changed = true;
      }
   }
   if (auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(def.getColumn().type)) {
      if (auto hm = mlir::dyn_cast<subop::HashMapType>(ler.getState())) {
         if (hashMapJoinKeyMatches(hm, canonicalHm)) {
            def.getColumn().type = subop::LookupEntryRefType::get(ctx, canonicalHm);
            changed = true;
         }
      }
   }
   return changed;
}

/// Align tuple column attrs in one hash-build `execution_step` to the extended canonical map type.
static void alignJoinHashMapStepColumnAttrs(ExecutionStepOp step, subop::HashMapType canonicalHm) {
   auto* ctx = canonicalHm.getContext();
   step.walk([&](mlir::Operation* op) {
      if (op->getParentOfType<ExecutionStepOp>() != step) return;

      if (auto ro = mlir::dyn_cast<subop::ReduceOp>(op)) {
         auto r = ro.getRef();
         if (alignJoinHashMapColumnRefToCanonical(r, canonicalHm)) ro.setRefAttr(r);
         return;
      }
      if (auto lk = mlir::dyn_cast<subop::LookupOp>(op)) {
         subop::HashMapType hm = getHashMapTypeForStateValue(lk.getState());
         if (hm && hashMapJoinKeyMatches(hm, canonicalHm)) {
            auto r = lk.getRef();
            if (alignJoinHashMapColumnDefToCanonical(r, canonicalHm)) lk.setRefAttr(r);
         }
         return;
      }
      if (auto loi = mlir::dyn_cast<subop::LookupOrInsertOp>(op)) {
         subop::HashMapType hm = getHashMapTypeForStateValue(loi.getState());
         if (hm && hashMapJoinKeyMatches(hm, canonicalHm)) {
            auto r = loi.getRef();
            if (alignJoinHashMapColumnDefToCanonical(r, canonicalHm)) loi.setRefAttr(r);
         }
         return;
      }
      if (auto sr = mlir::dyn_cast<subop::ScanRefsOp>(op)) {
         subop::HashMapType hm = getHashMapTypeForStateValue(sr.getState());
         if (hm && hashMapJoinKeyMatches(hm, canonicalHm)) {
            auto r = sr.getRef();
            if (alignJoinHashMapColumnDefToCanonical(r, canonicalHm)) sr.setRefAttr(r);
         }
         return;
      }
      if (auto go = mlir::dyn_cast<subop::GatherOp>(op)) {
         auto r = go.getRef();
         bool changed = alignJoinHashMapColumnRefToCanonical(r, canonicalHm);
         auto m = go.getMapping();
         llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> out;
         for (auto [mem, def] : m.getMapping()) {
            tuples::ColumnDefAttr d = def;
            if (alignJoinHashMapColumnDefToCanonical(d, canonicalHm)) changed = true;
            out.push_back({mem, d});
         }
         if (changed) {
            go.setRefAttr(r);
            go.setMappingAttr(subop::ColumnDefMemberMappingAttr::get(ctx, out));
         }
         return;
      }
      if (auto sc = mlir::dyn_cast<subop::ScanOp>(op)) {
         auto m = sc.getMapping();
         llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> out;
         bool changed = false;
         for (auto [mem, def] : m.getMapping()) {
            tuples::ColumnDefAttr d = def;
            if (alignJoinHashMapColumnDefToCanonical(d, canonicalHm)) changed = true;
            out.push_back({mem, d});
         }
         if (changed) sc.setMappingAttr(subop::ColumnDefMemberMappingAttr::get(ctx, out));
         return;
      }
      if (auto so = mlir::dyn_cast<subop::ScatterOp>(op)) {
         auto r = so.getRef();
         bool changed = alignJoinHashMapColumnRefToCanonical(r, canonicalHm);
         auto m = so.getMapping();
         llvm::SmallVector<std::pair<subop::Member, tuples::ColumnRefAttr>> out;
         for (auto [mem, cref] : m.getMapping()) {
            tuples::ColumnRefAttr c = cref;
            if (alignJoinHashMapColumnRefToCanonical(c, canonicalHm)) changed = true;
            out.push_back({mem, c});
         }
         if (changed) {
            so.setRefAttr(r);
            so.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, out));
         }
         return;
      }
      if (auto lo = mlir::dyn_cast<subop::LockOp>(op)) {
         auto r = lo.getRef();
         if (alignJoinHashMapColumnRefToCanonical(r, canonicalHm)) lo.setRefAttr(r);
         return;
      }
      if (auto mo = mlir::dyn_cast<subop::MapOp>(op)) {
         auto cols = mo.getInputColsAttr();
         llvm::SmallVector<mlir::Attribute> newCols;
         bool changed = false;
         for (mlir::Attribute a : cols) {
            if (auto cref = mlir::dyn_cast<tuples::ColumnRefAttr>(a)) {
               tuples::ColumnRefAttr r = cref;
               if (alignJoinHashMapColumnRefToCanonical(r, canonicalHm)) {
                  changed = true;
                  newCols.push_back(r);
                  continue;
               }
            }
            newCols.push_back(a);
         }
         if (changed) mo.setInputColsAttr(mlir::ArrayAttr::get(ctx, newCols));
         return;
      }
      if (auto fo = mlir::dyn_cast<subop::FilterOp>(op)) {
         auto conds = fo.getConditionsAttr();
         llvm::SmallVector<mlir::Attribute> out;
         bool changed = false;
         for (mlir::Attribute a : conds) {
            if (auto cref = mlir::dyn_cast<tuples::ColumnRefAttr>(a)) {
               tuples::ColumnRefAttr r = cref;
               if (alignJoinHashMapColumnRefToCanonical(r, canonicalHm)) {
                  changed = true;
                  out.push_back(r);
                  continue;
               }
            }
            out.push_back(a);
         }
         if (changed) fo.setConditionsAttr(mlir::ArrayAttr::get(ctx, out));
         return;
      }
      if (auto mat = mlir::dyn_cast<subop::MaterializeOp>(op)) {
         auto m = mat.getMapping();
         llvm::SmallVector<std::pair<subop::Member, tuples::ColumnRefAttr>> out;
         bool changed = false;
         for (auto [mem, cref] : m.getMapping()) {
            tuples::ColumnRefAttr c = cref;
            if (alignJoinHashMapColumnRefToCanonical(c, canonicalHm)) changed = true;
            out.push_back({mem, c});
         }
         if (changed) mat.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, out));
         return;
      }
      if (auto ins = mlir::dyn_cast<subop::InsertOp>(op)) {
         auto m = ins.getMapping();
         llvm::SmallVector<std::pair<subop::Member, tuples::ColumnRefAttr>> out;
         bool changed = false;
         for (auto [mem, cref] : m.getMapping()) {
            tuples::ColumnRefAttr c = cref;
            if (alignJoinHashMapColumnRefToCanonical(c, canonicalHm)) changed = true;
            out.push_back({mem, c});
         }
         if (changed) ins.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, out));
         return;
      }
      if (auto uo = mlir::dyn_cast<subop::UnwrapOptionalRefOp>(op)) {
         auto pref = uo.getOptionalRef();
         bool c1 = alignJoinHashMapColumnRefToCanonical(pref, canonicalHm);
         auto r = uo.getRef();
         bool c2 = alignJoinHashMapColumnDefToCanonical(r, canonicalHm);
         if (c1) uo.setOptionalRefAttr(pref);
         if (c2) uo.setRefAttr(r);
         return;
      }
      if (auto gb = mlir::dyn_cast<subop::GetBeginReferenceOp>(op)) {
         auto r = gb.getRef();
         if (alignJoinHashMapColumnDefToCanonical(r, canonicalHm)) gb.setRefAttr(r);
         return;
      }
      if (auto ge = mlir::dyn_cast<subop::GetEndReferenceOp>(op)) {
         auto r = ge.getRef();
         if (alignJoinHashMapColumnDefToCanonical(r, canonicalHm)) ge.setRefAttr(r);
         return;
      }
      if (auto nm = mlir::dyn_cast<subop::NestedMapOp>(op)) {
         auto params = nm.getParametersAttr();
         llvm::SmallVector<mlir::Attribute> newParams;
         bool changed = false;
         for (mlir::Attribute a : params) {
            if (auto cref = mlir::dyn_cast<tuples::ColumnRefAttr>(a)) {
               tuples::ColumnRefAttr r = cref;
               if (alignJoinHashMapColumnRefToCanonical(r, canonicalHm)) {
                  changed = true;
                  newParams.push_back(r);
                  continue;
               }
            }
            newParams.push_back(a);
         }
         if (changed) nm.setParametersAttr(mlir::ArrayAttr::get(ctx, newParams));
      }
   });
}

static void applyJoinBufferHivPredToSsaClosure(mlir::ModuleOp module, llvm::ArrayRef<mlir::Value> canonicalBuffers,
                                               const ModuleReuseInfo& reuse) {
   if (canonicalBuffers.empty()) return;

   JoinBufferHivSsaClosure joinClosure = computeJoinBufferHivSsaClosure(canonicalBuffers, reuse);
   llvm::DenseSet<void*>& closure = joinClosure.opaque;
   llvm::SmallVector<mlir::Value, 64>& values = joinClosure.values;

   auto* ctx = module.getContext();
   subop::Member predMember = makeOrGetPredMember(ctx);

   for (mlir::Value v : values) {
      if (auto buf = mlir::dyn_cast<subop::BufferType>(v.getType())) {
         if (valueMembersContainMemberNamed(ctx, buf.getMembers(), "filter_pred$0")) continue;
         auto nt = subop::BufferType::get(ctx, appendMember(ctx, buf.getMembers(), predMember));
         v.setType(nt);
      } else if (auto tl = mlir::dyn_cast<subop::ThreadLocalType>(v.getType())) {
         auto inner = mlir::dyn_cast<subop::BufferType>(tl.getWrapped());
         if (!inner || valueMembersContainMemberNamed(ctx, inner.getMembers(), "filter_pred$0")) continue;
         auto innerNew = subop::BufferType::get(ctx, appendMember(ctx, inner.getMembers(), predMember));
         v.setType(subop::ThreadLocalType::get(ctx, mlir::cast<subop::State>(innerNew)));
      }
   }

   module.walk([&](subop::CreateHashIndexedView chiv) {
      if (!closure.contains(chiv.getSource().getAsOpaquePointer())) return;
      syncCreateHashIndexedViewResultType(chiv);
   });

   expandClosureThroughExecutionStepPorts(module, closure);
   synchronizeExecutionStepPortTypes(module, &closure);
   propagateSubOpColumnAttrsFromSsaStateLayout(module, &closure);
}

static void applyGlobalHashMapPredLayout(mlir::ModuleOp module) {
   auto* ctx = module.getContext();
   subop::Member predMember = makeOrGetPredMember(ctx);
   module.walk([&](mlir::Operation* op) {
      for (mlir::OpResult r : op->getResults()) {
         mlir::Type t = r.getType();
         mlir::Type nt = extendHashMapTypeWithPred(t, predMember);
         if (nt != t) r.setType(nt);
      }
      for (mlir::Region& reg : op->getRegions()) {
         for (mlir::Block& b : reg) {
            for (mlir::BlockArgument a : b.getArguments()) {
               mlir::Type t = a.getType();
               mlir::Type nt = extendHashMapTypeWithPred(t, predMember);
               if (nt != t) a.setType(nt);
            }
         }
      }
   });
}

/// Join pipelines use `!subop.hashmap<...>` (often under `thread_local`). Extend value members with
/// `filter_pred$0 : i1` so table scan filters can be materialized once and re-checked on `scan_refs`.
static mlir::Type extendHashMapTypeWithPred(mlir::Type t, subop::Member predMember) {
   auto* ctx = t.getContext();
   if (auto tl = mlir::dyn_cast<subop::ThreadLocalType>(t)) {
      mlir::Type w = tl.getWrapped();
      mlir::Type nw = extendHashMapTypeWithPred(w, predMember);
      if (nw == w) return t;
      return subop::ThreadLocalType::get(ctx, mlir::cast<subop::State>(nw));
   }
   if (auto hm = mlir::dyn_cast<subop::HashMapType>(t)) {
      if (valueMembersContainMemberNamed(ctx, hm.getValueMembers(), "filter_pred$0")) return t;
      auto newVals = appendMember(ctx, hm.getValueMembers(), predMember);
      return subop::HashMapType::get(ctx, hm.getKeyMembers(), newVals, hm.getWithLock());
   }
   return t;
}

/// Align `lookup_entry_ref<!subop.hash_indexed_view<...>>` with join-buffer `filter_pred$0` layout
/// (same append order as `syncCreateHashIndexedViewResultType` when the pred column is last).
static subop::HashIndexedViewType extendHashIndexedViewWithPredMemberIfMissing(mlir::MLIRContext* ctx,
                                                                               subop::HashIndexedViewType hiv,
                                                                               subop::Member predMember) {
   if (valueMembersContainMemberNamed(ctx, hiv.getValueMembers(), "filter_pred$0")) return hiv;
   auto newVals = appendMember(ctx, hiv.getValueMembers(), predMember);
   return subop::HashIndexedViewType::get(ctx, hiv.getKeyMembers(), newVals, hiv.getCompareHashForLookup());
}

/// HIV gains `filter_pred$0` on join-buffer reuse; nested `!subop.list<!subop.lookup_entry_ref<...>>`
/// (e.g. `nested_execution_group` block args) can keep the **old** HIV inside the value type while column
/// attrs were patched — lowering then disagrees with the widened buffer / HIV and leaves unrealized casts.
///
/// Pure recursive rewrite on a `mlir::Type` only. SSA updates go through `refreshHivListCarrierValueTypes`,
/// which applies this per `OpResult` / `BlockArgument` (entire module, or only values in the HIV closure).
static mlir::Type deepReplaceHivLookupEntryRefWithPredLayout(mlir::MLIRContext* ctx, mlir::Type t,
                                                            subop::Member predMember) {
   if (!t) return t;
   if (auto list = mlir::dyn_cast<subop::ListType>(t)) {
      mlir::Type nt = deepReplaceHivLookupEntryRefWithPredLayout(ctx, list.getT(), predMember);
      if (nt != list.getT()) return subop::ListType::get(ctx, mlir::cast<subop::StateEntryReference>(nt));
      return t;
   }
   if (auto opt = mlir::dyn_cast<subop::OptionalType>(t)) {
      mlir::Type nt = deepReplaceHivLookupEntryRefWithPredLayout(ctx, opt.getT(), predMember);
      if (nt != opt.getT()) return subop::OptionalType::get(ctx, mlir::cast<subop::StateEntryReference>(nt));
      return t;
   }
   if (auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(t)) {
      if (auto hiv = mlir::dyn_cast<subop::HashIndexedViewType>(ler.getState())) {
         auto nhiv = extendHashIndexedViewWithPredMemberIfMissing(ctx, hiv, predMember);
         if (nhiv != hiv) return subop::LookupEntryRefType::get(ctx, nhiv);
      }
      return t;
   }
   return t;
}

/// Apply `deepReplaceHivLookupEntryRefWithPredLayout` to each SSA value's declared type. When
/// \p closureFilter is set, only **values whose opaque pointer is in the closure** are updated
/// (no `opOperandsOrNestedBlockArgsTouchClosure` gate — operands are other values' results/args).
///
/// Important: never call `Value::setType` inside `module.walk` — mutating types while traversing
/// the IR graph can leave `Type` storages in a bad state and crash the next `dyn_cast` on a
/// sibling value. Collect updates first, then apply after the walk completes; repeat until a
/// full pass makes no changes (fixpoint). With consistent IR, `deepReplaceHivLookupEntryRefWithPredLayout`
/// is idempotent at fixpoint; a non-terminating loop would indicate a bug elsewhere, not an
/// intentional "type update cycle".
static void refreshHivListCarrierValueTypes(mlir::ModuleOp module, const llvm::DenseSet<void*>* closureFilter) {
   auto* ctx = module.getContext();
   subop::Member predMember = makeOrGetPredMember(ctx);
   for (;;) {
      llvm::SmallVector<std::pair<mlir::Value, mlir::Type>> updates;
      updates.reserve(128);
      module.walk([&](mlir::Operation* op) {
         for (mlir::OpResult r : op->getResults()) {
            if (closureFilter && !opaqueClosureContains(*closureFilter, r)) continue;
            mlir::Type cur = r.getType();
            if (!cur) continue;
            mlir::Type nt = deepReplaceHivLookupEntryRefWithPredLayout(ctx, cur, predMember);
            if (nt != cur) updates.push_back({r, nt});
         }
         for (mlir::Region& reg : op->getRegions()) {
            for (mlir::Block& b : reg) {
               for (mlir::BlockArgument a : b.getArguments()) {
                  if (closureFilter && !opaqueClosureContains(*closureFilter, a)) continue;
                  mlir::Type cur = a.getType();
                  if (!cur) continue;
                  mlir::Type nt = deepReplaceHivLookupEntryRefWithPredLayout(ctx, cur, predMember);
                  if (nt != cur) updates.push_back({a, nt});
               }
            }
         }
      });
      if (updates.empty()) break;
      for (auto [v, nt] : updates) {
         v.setType(nt);
      }
   }
}

/// After `applyGlobalHashMapPredLayout` / join-buffer HIV extensions, tuple column attrs and nested
/// list/optional carrier types may still embed pre-extension `hash_map_entry_ref` / `lookup_entry_ref`.
/// When \p closureFilter is null, fix the whole module. Otherwise column-attr walks still use
/// `opOperandsOrNestedBlockArgsTouchClosure`, while `refreshHivListCarrierValueTypes` only rewrites
/// **types of SSA values present in the closure** (after `expandClosureThroughExecutionStepPorts`).
void propagateSubOpColumnAttrsFromSsaStateLayout(mlir::ModuleOp module,
                                                        const llvm::DenseSet<void*>* closureFilter) {
   // `alignConsumerHashIndexedViewsWithSyntheticProducer` strips nested lookup/HIV carriers to the
   // synthetic producer layout; this pass re-extends attrs and SSA with `filter_pred$0` via
   // `refreshHivListCarrierValueTypes`. Skip entirely while filter-pred reuse is disabled.
   if (!kEnableReuseStateFilterPredReapply) return;

   auto* ctx = module.getContext();
   subop::Member predMember = makeOrGetPredMember(ctx);
   auto shouldUpdateOp = [&](mlir::Operation* op) {
      return !closureFilter || opOperandsOrNestedBlockArgsTouchClosure(op, *closureFilter);
   };
   auto syncHashMapEntryRef = [&](tuples::ColumnRefAttr cref) -> bool {
      bool changed = false;
      if (auto hmer = mlir::dyn_cast<subop::HashMapEntryRefType>(cref.getColumn().type)) {
         mlir::Type nhm = extendHashMapTypeWithPred(hmer.getHashMap(), predMember);
         if (nhm != hmer.getHashMap()) {
            cref.getColumn().type = subop::HashMapEntryRefType::get(ctx, mlir::cast<subop::HashMapType>(nhm));
            changed = true;
         }
      }
      if (auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(cref.getColumn().type)) {
         if (auto hm = mlir::dyn_cast<subop::HashMapType>(ler.getState())) {
            mlir::Type nhm = extendHashMapTypeWithPred(hm, predMember);
            hm = mlir::dyn_cast<subop::HashMapType>(nhm);
            if (hm) {
               auto expected = subop::LookupEntryRefType::get(ctx, hm);
               if (cref.getColumn().type != expected) {
                  cref.getColumn().type = expected;
                  changed = true;
               }
            }
         } else if (auto hiv = mlir::dyn_cast<subop::HashIndexedViewType>(ler.getState())) {
            auto nhiv = extendHashIndexedViewWithPredMemberIfMissing(ctx, hiv, predMember);
            if (nhiv != hiv) {
               cref.getColumn().type = subop::LookupEntryRefType::get(ctx, nhiv);
               changed = true;
            }
         }
      }
      return changed;
   };
   auto syncHashMapEntryRefInColumnDef = [&](tuples::ColumnDefAttr def) -> bool {
      bool changed = false;
      if (auto hmer = mlir::dyn_cast<subop::HashMapEntryRefType>(def.getColumn().type)) {
         mlir::Type nhm = extendHashMapTypeWithPred(hmer.getHashMap(), predMember);
         if (nhm != hmer.getHashMap()) {
            def.getColumn().type = subop::HashMapEntryRefType::get(ctx, mlir::cast<subop::HashMapType>(nhm));
            changed = true;
         }
      }
      if (auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(def.getColumn().type)) {
         if (auto hm = mlir::dyn_cast<subop::HashMapType>(ler.getState())) {
            mlir::Type nhm = extendHashMapTypeWithPred(hm, predMember);
            hm = mlir::dyn_cast<subop::HashMapType>(nhm);
            if (hm) {
               auto expected = subop::LookupEntryRefType::get(ctx, hm);
               if (def.getColumn().type != expected) {
                  def.getColumn().type = expected;
                  changed = true;
               }
            }
         } else if (auto hiv = mlir::dyn_cast<subop::HashIndexedViewType>(ler.getState())) {
            auto nhiv = extendHashIndexedViewWithPredMemberIfMissing(ctx, hiv, predMember);
            if (nhiv != hiv) {
               def.getColumn().type = subop::LookupEntryRefType::get(ctx, nhiv);
               changed = true;
            }
         }
      }
      return changed;
   };
   module.walk([&](subop::ReduceOp op) {
      if (!shouldUpdateOp(op.getOperation())) return;
      bool changed = false;
      auto r = op.getRef();
      if (syncHashMapEntryRef(r)) {
         op.setRefAttr(r);
         changed = true;
      }
      // `ReduceOpLowering` compares each stream column type to these attrs; stale
      // `hash_map_entry_ref` / `lookup_entry_ref` after pred layout forces unrealized casts.
      llvm::SmallVector<mlir::Attribute> newCols;
      newCols.reserve(op.getColumns().size());
      for (mlir::Attribute a : op.getColumns()) {
         if (auto cref = mlir::dyn_cast<tuples::ColumnRefAttr>(a)) {
            tuples::ColumnRefAttr cr = cref;
            if (syncHashMapEntryRef(cr)) {
               changed = true;
               newCols.push_back(cr);
               continue;
            }
         }
         newCols.push_back(a);
      }
      if (changed) op.setColumnsAttr(mlir::ArrayAttr::get(ctx, newCols));
   });
   module.walk([&](subop::ScatterOp op) {
      if (!shouldUpdateOp(op.getOperation())) return;
      auto r = op.getRef();
      bool changed = syncHashMapEntryRef(r);
      llvm::SmallVector<std::pair<subop::Member, tuples::ColumnRefAttr>> out;
      for (auto [mem, cref] : op.getMapping().getMapping()) {
         tuples::ColumnRefAttr c = cref;
         if (syncHashMapEntryRef(c)) changed = true;
         out.push_back({mem, c});
      }
      if (!changed) return;
      op.setRefAttr(r);
      op.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, out));
   });
   module.walk([&](subop::GatherOp op) {
      if (!shouldUpdateOp(op.getOperation())) return;
      auto r = op.getRef();
      bool changed = syncHashMapEntryRef(r);
      auto m = op.getMapping();
      llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> out;
      for (auto [mem, def] : m.getMapping()) {
         tuples::ColumnDefAttr d = def;
         if (syncHashMapEntryRefInColumnDef(d)) changed = true;
         out.push_back({mem, d});
      }
      if (changed) {
         op.setRefAttr(r);
         op.setMappingAttr(subop::ColumnDefMemberMappingAttr::get(ctx, out));
      }
   });
   module.walk([&](subop::ScanOp op) {
      if (!shouldUpdateOp(op.getOperation())) return;
      auto m = op.getMapping();
      llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> out;
      bool changed = false;
      for (auto [mem, def] : m.getMapping()) {
         tuples::ColumnDefAttr d = def;
         if (syncHashMapEntryRefInColumnDef(d)) changed = true;
         out.push_back({mem, d});
      }
      if (changed) op.setMappingAttr(subop::ColumnDefMemberMappingAttr::get(ctx, out));
   });
   module.walk([&](subop::NestedMapOp op) {
      if (!shouldUpdateOp(op.getOperation())) return;
      auto params = op.getParametersAttr();
      llvm::SmallVector<mlir::Attribute> newParams;
      bool changed = false;
      for (mlir::Attribute a : params) {
         if (auto cref = mlir::dyn_cast<tuples::ColumnRefAttr>(a)) {
            tuples::ColumnRefAttr r = cref;
            if (syncHashMapEntryRef(r)) {
               changed = true;
               newParams.push_back(r);
               continue;
            }
         }
         newParams.push_back(a);
      }
      if (changed) op.setParametersAttr(mlir::ArrayAttr::get(ctx, newParams));
   });
   module.walk([&](subop::MapOp op) {
      if (!shouldUpdateOp(op.getOperation())) return;
      auto cols = op.getInputColsAttr();
      llvm::SmallVector<mlir::Attribute> newCols;
      bool changed = false;
      for (mlir::Attribute a : cols) {
         if (auto cref = mlir::dyn_cast<tuples::ColumnRefAttr>(a)) {
            tuples::ColumnRefAttr r = cref;
            if (syncHashMapEntryRef(r)) {
               changed = true;
               newCols.push_back(r);
               continue;
            }
         }
         newCols.push_back(a);
      }
      if (changed) op.setInputColsAttr(mlir::ArrayAttr::get(ctx, newCols));
   });
   module.walk([&](subop::FilterOp op) {
      if (!shouldUpdateOp(op.getOperation())) return;
      auto conds = op.getConditionsAttr();
      llvm::SmallVector<mlir::Attribute> out;
      bool changed = false;
      for (mlir::Attribute a : conds) {
         if (auto cref = mlir::dyn_cast<tuples::ColumnRefAttr>(a)) {
            tuples::ColumnRefAttr r = cref;
            if (syncHashMapEntryRef(r)) {
               changed = true;
               out.push_back(r);
               continue;
            }
         }
         out.push_back(a);
      }
      if (changed) op.setConditionsAttr(mlir::ArrayAttr::get(ctx, out));
   });
   module.walk([&](subop::LockOp op) {
      if (!shouldUpdateOp(op.getOperation())) return;
      auto r = op.getRef();
      if (!syncHashMapEntryRef(r)) return;
      op.setRefAttr(r);
   });
   module.walk([&](subop::LookupOp op) {
      if (!shouldUpdateOp(op.getOperation())) return;
      auto r = op.getRef();
      if (auto hiv = mlir::dyn_cast<subop::HashIndexedViewType>(op.getState().getType())) {
         auto expected = subop::LookupEntryRefType::get(ctx, hiv);
         if (r.getColumn().type == expected) return;
         r.getColumn().type = expected;
         op.setRefAttr(r);
         return;
      }
      subop::HashMapType hm = getHashMapTypeForStateValue(op.getState());
      if (!hm) return;
      mlir::Type nhm = extendHashMapTypeWithPred(hm, predMember);
      hm = mlir::dyn_cast<subop::HashMapType>(nhm);
      if (!hm) return;
      auto expected = subop::LookupEntryRefType::get(ctx, hm);
      if (r.getColumn().type == expected) return;
      r.getColumn().type = expected;
      op.setRefAttr(r);
   });
   module.walk([&](subop::LookupOrInsertOp op) {
      if (!shouldUpdateOp(op.getOperation())) return;
      subop::HashMapType hm = getHashMapTypeForStateValue(op.getState());
      if (!hm) return;
      mlir::Type nhm = extendHashMapTypeWithPred(hm, predMember);
      hm = mlir::dyn_cast<subop::HashMapType>(nhm);
      if (!hm) return;
      auto expected = subop::LookupEntryRefType::get(ctx, hm);
      auto r = op.getRef();
      if (r.getColumn().type == expected) return;
      r.getColumn().type = expected;
      op.setRefAttr(r);
   });
   module.walk([&](subop::ScanRefsOp op) {
      if (!shouldUpdateOp(op.getOperation())) return;
      subop::HashMapType hm = getHashMapTypeForStateValue(op.getState());
      if (!hm) return;
      mlir::Type nhm = extendHashMapTypeWithPred(hm, predMember);
      hm = mlir::dyn_cast<subop::HashMapType>(nhm);
      if (!hm) return;
      auto expected = subop::HashMapEntryRefType::get(ctx, hm);
      auto r = op.getRef();
      if (r.getColumn().type == expected) return;
      r.getColumn().type = expected;
      op.setRefAttr(r);
   });
   module.walk([&](subop::MaterializeOp op) {
      if (!shouldUpdateOp(op.getOperation())) return;
      auto m = op.getMapping();
      llvm::SmallVector<std::pair<subop::Member, tuples::ColumnRefAttr>> out;
      bool changed = false;
      for (auto [mem, cref] : m.getMapping()) {
         tuples::ColumnRefAttr r = cref;
         if (syncHashMapEntryRef(r)) changed = true;
         out.push_back({mem, r});
      }
      if (changed) op.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, out));
   });
   module.walk([&](subop::InsertOp op) {
      if (!shouldUpdateOp(op.getOperation())) return;
      auto m = op.getMapping();
      llvm::SmallVector<std::pair<subop::Member, tuples::ColumnRefAttr>> out;
      bool changed = false;
      for (auto [mem, cref] : m.getMapping()) {
         tuples::ColumnRefAttr r = cref;
         if (syncHashMapEntryRef(r)) changed = true;
         out.push_back({mem, r});
      }
      if (changed) op.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, out));
   });
   module.walk([&](subop::UnwrapOptionalRefOp op) {
      if (!shouldUpdateOp(op.getOperation())) return;
      auto pref = op.getOptionalRef();
      bool c1 = syncHashMapEntryRef(pref);
      auto r = op.getRef();
      bool c2 = syncHashMapEntryRefInColumnDef(r);
      if (auto optTy = mlir::dyn_cast<subop::OptionalType>(op.getOptionalRef().getColumn().type)) {
         mlir::Type inner = optTy.getT();
         if (r.getColumn().type != inner) {
            r.getColumn().type = inner;
            c2 = true;
         }
      }
      if (c1) op.setOptionalRefAttr(pref);
      if (c2) op.setRefAttr(r);
   });
   module.walk([&](subop::GetBeginReferenceOp op) {
      if (!shouldUpdateOp(op.getOperation())) return;
      auto r = op.getRef();
      if (!syncHashMapEntryRefInColumnDef(r)) return;
      op.setRefAttr(r);
   });
   module.walk([&](subop::GetEndReferenceOp op) {
      if (!shouldUpdateOp(op.getOperation())) return;
      auto r = op.getRef();
      if (!syncHashMapEntryRefInColumnDef(r)) return;
      op.setRefAttr(r);
   });
   module.walk([&](subop::EntriesBetweenOp op) {
      if (!shouldUpdateOp(op.getOperation())) return;
      auto lr = op.getLeftRef();
      if (syncHashMapEntryRef(lr)) op.setLeftRefAttr(lr);
      auto rr = op.getRightRef();
      if (syncHashMapEntryRef(rr)) op.setRightRefAttr(rr);
      auto between = op.getBetween();
      if (syncHashMapEntryRefInColumnDef(between)) op.setBetweenAttr(between);
   });
   module.walk([&](subop::OffsetReferenceBy op) {
      if (!shouldUpdateOp(op.getOperation())) return;
      auto ref = op.getRef();
      if (syncHashMapEntryRef(ref)) op.setRefAttr(ref);
      auto idx = op.getIdx();
      if (syncHashMapEntryRef(idx)) op.setIdxAttr(idx);
      auto newRef = op.getNewRef();
      if (syncHashMapEntryRefInColumnDef(newRef)) op.setNewRefAttr(newRef);
   });

   refreshHivListCarrierValueTypes(module, closureFilter);
   synchronizeExecutionStepPortTypes(module, closureFilter);

   module.walk([&](subop::ScanListOp scan) {
      if (!shouldUpdateOp(scan.getOperation())) return;
      auto listTy = mlir::dyn_cast<subop::ListType>(scan.getList().getType());
      if (!listTy) return;
      mlir::Type inner = listTy.getT();
      auto elem = scan.getElem();
      if (elem.getColumn().type == inner) return;
      auto elem2 = elem;
      elem2.getColumn().type = inner;
      scan.setElemAttr(elem2);
   });
   module.walk([&](subop::LookupOp lookup) {
      if (!shouldUpdateOp(lookup.getOperation())) return;
      auto hiv = mlir::dyn_cast<subop::HashIndexedViewType>(lookup.getState().getType());
      if (!hiv) return;
      auto expected = subop::LookupEntryRefType::get(ctx, hiv);
      auto refDef = lookup.getRef();
      if (refDef.getColumn().type == expected) return;
      refDef.getColumn().type = expected;
      lookup.setRefAttr(refDef);
   });
}

/// Join-buffer reuse appends `filter_pred$0` to merged `!subop.buffer<...>` results. The `subop.merge`
/// lowering assumes the incoming `thread_local` wraps the **same** buffer layout as the merge
/// result; if only the merge result type was widened, the operand chain still names the old
/// three-field buffer and LLVM translation hits stuck `builtin.unrealized_conversion_cast`.
void alignBufferMergeThreadLocalsWithExtendedMergeResult(mlir::ModuleOp module) {
   auto* ctx = module.getContext();
   for (unsigned round = 0; round < 1; ++round) {
      module.walk([&](subop::MergeOp merge) {
         auto resBuf = mlir::dyn_cast<subop::BufferType>(merge.getRes().getType());
         if (!resBuf || !valueMembersContainMemberNamed(ctx, resBuf.getMembers(), "filter_pred$0")) return;
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
               if (!parentOp && owner->getParent()) parentOp = owner->getParent()->getParentOp();
               if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(parentOp)) {
                  unsigned idx = ba.getArgNumber();
                  if (idx < step.getNumOperands()) worklist.push_back(step.getOperand(idx));
               }
            }
         }
      });
   }
}

static constexpr llvm::StringLiteral kHashmapLookupInitPredExtendedAttr = "lingo.hashmap_lookup_init_pred_extended";
static constexpr llvm::StringLiteral kHashmapMergeCombinePredExtendedAttr = "lingo.hashmap_merge_combine_pred_extended";

/// Extend `hashmap` join state globally; extend `buffer` / `hash_indexed_view` only on the SSA closure of
/// matched reuse buffers (requires \p reuseForJoinBuffers from `collectModuleReuseInfo` **before** mutation).
void rewriteHashmapTypesInModule(mlir::ModuleOp module,
                                        llvm::ArrayRef<mlir::Value> extendJoinBufferStates,
                                        const ModuleReuseInfo* reuseForJoinBuffers) {
   applyGlobalHashMapPredLayout(module);

   auto* ctx = module.getContext();

   if (!extendJoinBufferStates.empty()) {
      assert(reuseForJoinBuffers && "rewriteHashmapTypesInModule: reuse info required for join buffer+HIV");
      applyJoinBufferHivPredToSsaClosure(module, extendJoinBufferStates, *reuseForJoinBuffers);
   }

   module.walk([&](subop::ScanRefsOp scan) {
      if (!mlir::isa<subop::HashMapType>(scan.getState().getType())) return;
      auto hmTy = mlir::cast<subop::HashMapType>(scan.getState().getType());
      auto newRefTy = subop::HashMapEntryRefType::get(ctx, hmTy);
      auto refDef = scan.getRef();
      refDef.getColumn().type = newRefTy;
      scan.setRefAttr(refDef);
   });
   module.walk([&](subop::LookupOrInsertOp loi) {
      if (!mlir::isa<subop::HashMapType>(loi.getState().getType())) return;
      auto hmTy = mlir::cast<subop::HashMapType>(loi.getState().getType());
      auto newRefTy = subop::LookupEntryRefType::get(ctx, hmTy);
      auto refDef = loi.getRef();
      refDef.getColumn().type = newRefTy;
      loi.setRefAttr(refDef);
   });

   module.walk([&](subop::LookupOrInsertOp loi) {
      if (!mlir::isa<subop::HashMapType>(loi.getState().getType())) return;
      if (loi->hasAttr(kHashmapLookupInitPredExtendedAttr)) return;
      assert(!loi.getInitFn().empty());
      mlir::Block& b = loi.getInitFn().front();
      auto ret = mlir::cast<tuples::ReturnOp>(b.getTerminator());
      mlir::OpBuilder bb(ret);
      mlir::Value t = bb.create<mlir::arith::ConstantIntOp>(loi.getLoc(), 1, 1);
      llvm::SmallVector<mlir::Value, 32> outs(ret.getOperands().begin(), ret.getOperands().end());
      outs.push_back(t);
      bb.create<tuples::ReturnOp>(loi.getLoc(), outs);
      ret.erase();
      loi->setAttr(kHashmapLookupInitPredExtendedAttr, mlir::UnitAttr::get(ctx));
   });

   module.walk([&](subop::MergeOp mergeOp) {
      if (!mlir::isa<subop::HashMapType>(mergeOp.getResult().getType())) return;
      if (mergeOp->hasAttr(kHashmapMergeCombinePredExtendedAttr)) return;
      if (mergeOp.getCombineFn().empty()) return;
      auto hmTy = mlir::cast<subop::HashMapType>(mergeOp.getResult().getType());
      auto* d = ctx->getLoadedDialect<subop::SubOperatorDialect>();
      assert(d && "SubOperatorDialect must be loaded");
      auto& mm = d->getMemberManager();
      auto mems = hmTy.getValueMembers().getMembers();
      if (mems.empty()) return;
      if (mm.getName(mems.back()) != "filter_pred$0") return;

      // `MergeThreadLocalHashMap` passes combine args as concat(left value map, right value map) in
      // **member order**, so with `filter_pred$0` last the layout is
      //   [L_non_pred..., L_pred, R_non_pred..., R_pred].
      // Historically we appended two i1 at the end (`[L..., R..., L_pred, R_pred]`), which mismatched
      // lowering and produced unreconcilable casts. Insert the predicate slots in the middle instead.
      const unsigned k = mems.size();
      const unsigned nonPredPerSide = k - 1;
      mlir::Block& b = mergeOp.getCombineFn().front();
      const unsigned nArg = b.getNumArguments();
      // Already includes both sides' `filter_pred$0` slots (parser built full 2*k combine args).
      if (nArg == 2 * k) return;
      if (nArg != 2 * nonPredPerSide) return;

      mlir::Type i1 = mlir::IntegerType::get(ctx, 1);
      auto loc = mergeOp.getLoc();
      b.insertArgument(nonPredPerSide, i1, loc);
      b.insertArgument(2 * nonPredPerSide + 1, i1, loc);

      auto ret = mlir::cast<tuples::ReturnOp>(b.getTerminator());
      mlir::OpBuilder bb(ret);
      mlir::Value leftPred = b.getArgument(nonPredPerSide);
      mlir::Value rightPred = b.getArgument(2 * nonPredPerSide + 1);
      mlir::Value mergedPred = bb.create<mlir::arith::AndIOp>(loc, leftPred, rightPred);

      llvm::SmallVector<mlir::Value, 32> outs(ret.getOperands().begin(), ret.getOperands().end());
      outs.push_back(mergedPred);
      bb.create<tuples::ReturnOp>(loc, outs);
      ret.erase();
      mergeOp->setAttr(kHashmapMergeCombinePredExtendedAttr, mlir::UnitAttr::get(ctx));
   });

   propagateSubOpColumnAttrsFromSsaStateLayout(module, nullptr);
   alignBufferMergeThreadLocalsWithExtendedMergeResult(module);
}

// Convert runtime filter descriptions into MLIR subop.map + subop.filter.
// For now: only integer column + int64 constant comparisons. Everything else asserts.
mlir::Value materializeRuntimeFiltersAsSubopFilter(mlir::OpBuilder& b,
                                                          mlir::Location loc,
                                                          mlir::Value stream,
                                                          const llvm::DenseMap<llvm::StringRef, tuples::ColumnRefAttr>& colByName,
                                                          llvm::ArrayRef<runtime::FilterDescription> filters) {
   if (filters.empty()) return stream;

   // Build a single predicate column: AND over all supported filters.
   auto [predDef, predRef] = [&]() -> std::pair<tuples::ColumnDefAttr, tuples::ColumnRefAttr> {
      auto& cm = b.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      std::string scopeName = cm.getUniqueScope("delay_filter");
      tuples::ColumnDefAttr def = cm.createDef(scopeName, "pred");
      def.getColumn().type = b.getI1Type();
      return {def, cm.createRef(&def.getColumn())};
   }();

   subop::MapCreationHelper helper(b.getContext());
   helper.buildBlock(b, [&](mlir::OpBuilder& rb) {
      mlir::Value acc;
      for (auto& f : filters) {
         auto it = colByName.find(f.columnName);
         assert(it != colByName.end() && "delay_filter: missing filter column in gathered columns");
         mlir::Value colV = helper.access(it->second, loc);

         mlir::Value pred;
         if (f.op == runtime::FilterOp::NOTNULL) {
            pred = rb.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
         } else {
            assert(std::holds_alternative<int64_t>(f.value) && "delay_filter: only int64 filter values supported");
            int64_t v = std::get<int64_t>(f.value);
            auto itTy = mlir::dyn_cast<mlir::IntegerType>(colV.getType());
            assert(itTy && "delay_filter: only integer-typed columns supported");
            mlir::Value c = rb.create<mlir::arith::ConstantIntOp>(loc, v, itTy.getWidth());
            using P = mlir::arith::CmpIPredicate;
            P p;
            switch (f.op) {
               case runtime::FilterOp::EQ: p = P::eq; break;
               case runtime::FilterOp::NEQ: p = P::ne; break;
               case runtime::FilterOp::LT: p = P::slt; break;
               case runtime::FilterOp::LTE: p = P::sle; break;
               case runtime::FilterOp::GT: p = P::sgt; break;
               case runtime::FilterOp::GTE: p = P::sge; break;
               default: assert(0 && "delay_filter: unsupported filter op");
            }
            pred = rb.create<mlir::arith::CmpIOp>(loc, p, colV, c);
         }
         acc = acc ? rb.create<mlir::arith::AndIOp>(loc, acc, pred) : pred;
      }
      rb.create<tuples::ReturnOp>(loc, mlir::ValueRange{acc});
   });

   auto mapOp = b.create<subop::MapOp>(loc,
                                       tuples::TupleStreamType::get(b.getContext()),
                                       stream,
                                       b.getArrayAttr({predDef}),
                                       helper.getColRefs());
   mapOp.getFn().push_back(helper.getMapBlock());
   auto filterOp = b.create<subop::FilterOp>(loc,
                                             mapOp.getResult(),
                                             subop::FilterSemantic::all_true,
                                             b.getArrayAttr({predRef}));
   return filterOp.getRes();
}

// Build a predicate column (AND over supported filters) and append it to the stream.
// The returned stream has an additional boolean column named `predLeafName` in a fresh scope.
static std::pair<mlir::Value, tuples::ColumnDefAttr>
materializeRuntimeFiltersAsPredicateColumn(mlir::OpBuilder& b,
                                           mlir::Location loc,
                                           mlir::Value stream,
                                           const llvm::DenseMap<llvm::StringRef, tuples::ColumnRefAttr>& colByName,
                                           llvm::ArrayRef<runtime::FilterDescription> filters,
                                           llvm::StringRef predLeafName) {
   assert(!filters.empty());

   auto& cm = b.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   std::string scopeName = cm.getUniqueScope("delay_filter_pred");
   tuples::ColumnDefAttr predDef = cm.createDef(scopeName, predLeafName.str());
   predDef.getColumn().type = b.getI1Type();
   // Keep predicate column def alive; column ref created where needed.

   subop::MapCreationHelper helper(b.getContext());
   helper.buildBlock(b, [&](mlir::OpBuilder& rb) {
      mlir::Value acc;
      for (auto& f : filters) {
         auto it = colByName.find(f.columnName);
         assert(it != colByName.end() && "delay_filter_pred: missing filter column in gathered columns");
         mlir::Value colV = helper.access(it->second, loc);

         mlir::Value pred;
         if (f.op == runtime::FilterOp::NOTNULL) {
            pred = rb.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
         } else {
            // Minimal support for TPCH Q1: date column compared to string literal.
            if (std::holds_alternative<int64_t>(f.value)) {
               int64_t v = std::get<int64_t>(f.value);
               auto itTy = mlir::dyn_cast<mlir::IntegerType>(colV.getType());
               assert(itTy && "delay_filter_pred: only integer-typed columns supported");
               mlir::Value c = rb.create<mlir::arith::ConstantIntOp>(loc, v, itTy.getWidth());
               using P = mlir::arith::CmpIPredicate;
               P p;
               switch (f.op) {
                  case runtime::FilterOp::EQ: p = P::eq; break;
                  case runtime::FilterOp::NEQ: p = P::ne; break;
                  case runtime::FilterOp::LT: p = P::slt; break;
                  case runtime::FilterOp::LTE: p = P::sle; break;
                  case runtime::FilterOp::GT: p = P::sgt; break;
                  case runtime::FilterOp::GTE: p = P::sge; break;
                  default: assert(0 && "delay_filter_pred: unsupported filter op");
               }
               pred = rb.create<mlir::arith::CmpIOp>(loc, p, colV, c);
            } else if (std::holds_alternative<std::string>(f.value)) {
               auto s = std::get<std::string>(f.value);
               assert((mlir::isa<lingodb::compiler::dialect::db::DateType>(colV.getType()) ||
                       mlir::isa<lingodb::compiler::dialect::db::CharType>(colV.getType())) &&
                      "delay_filter_pred: string literal filters only supported for db.date/db.char");
               auto rhs = rb.create<lingodb::compiler::dialect::db::ConstantOp>(
                  loc, colV.getType(), rb.getStringAttr(s));
               using P = lingodb::compiler::dialect::db::DBCmpPredicate;
               P p;
               switch (f.op) {
                  case runtime::FilterOp::EQ: p = P::eq; break;
                  case runtime::FilterOp::NEQ: p = P::neq; break;
                  case runtime::FilterOp::LT: p = P::lt; break;
                  case runtime::FilterOp::LTE: p = P::lte; break;
                  case runtime::FilterOp::GT: p = P::gt; break;
                  case runtime::FilterOp::GTE: p = P::gte; break;
                  default: assert(0 && "delay_filter_pred: unsupported filter op");
               }
               auto cmp = rb.create<lingodb::compiler::dialect::db::CmpOp>(loc, p, colV, rhs);
               // cmp may be nullable depending on operands; `derive_truth` normalizes to i1.
               pred = rb.create<lingodb::compiler::dialect::db::DeriveTruth>(loc, cmp);
            } else {
               assert(0 && "delay_filter_pred: unsupported filter literal type");
            }
         }
         acc = acc ? rb.create<mlir::arith::AndIOp>(loc, acc, pred) : pred;
      }
      rb.create<tuples::ReturnOp>(loc, mlir::ValueRange{acc});
   });

   auto mapOp = b.create<subop::MapOp>(loc,
                                       tuples::TupleStreamType::get(b.getContext()),
                                       stream,
                                       b.getArrayAttr({predDef}),
                                       helper.getColRefs());
   mapOp.getFn().push_back(helper.getMapBlock());
   return {mapOp.getResult(), predDef};
}

void collectJoinBufferStatesFromTargets(llvm::ArrayRef<CacheTarget> targets,
                                        llvm::SmallVector<mlir::Value, 8>& out,
                                        const ModuleReuseInfo* reuse) {
   llvm::DenseSet<void*> seen;
   for (auto& t : targets) {
      if (!t.state) continue;
      mlir::Value buf;
      if (mlir::isa<subop::BufferType>(t.state.getType())) {
         buf = t.state;
      } else if (reuse) {
         if (auto itG = findReuseMap(reuse->hashIndexedViewFromMergedBuffer, t.state);
             itG != reuse->hashIndexedViewFromMergedBuffer.end()) {
            buf = itG->second;
         }
      }
      if (!buf) continue;
      if (seen.insert(buf.getAsOpaquePointer()).second) out.push_back(buf);
   }
}

llvm::SmallVector<runtime::FilterDescription, 8> decodeFiltersForStateFromWriterSteps(
   mlir::Value state, const ModuleReuseInfo& reuse,
   const llvm::DenseMap<mlir::Operation*, const ModuleReuseInfo::StepRW*>* rwByStepOp) {
   llvm::DenseMap<mlir::Operation*, const ModuleReuseInfo::StepRW*> localRw;
   if (!rwByStepOp) {
      localRw = buildRwByStepOpMap(reuse);
      rwByStepOp = &localRw;
   }
   llvm::SmallVector<runtime::FilterDescription, 8> decoded;
   auto itW = reuse.writerStepsByState.find(canonicalizeStateValueForReuse(state));
   if (itW == reuse.writerStepsByState.end()) itW = reuse.writerStepsByState.find(state);
   if (itW == reuse.writerStepsByState.end()) return decoded;
   for (ExecutionStepOp ws : itW->second) {
      const ModuleReuseInfo::StepRW* rw = rwByStepOp->lookup(ws.getOperation());
      if (!rw) continue;
      for (mlir::Value r : rw->reads) {
         if (!mlir::isa<subop::TableType>(r.getType())) continue;
         mlir::Value tableV = r;
         if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(r)) {
            if (ba.getOwner() == &ws.getSubOps().front()) {
               assert(static_cast<unsigned>(ba.getArgNumber()) < ws.getNumOperands());
               tableV = ws.getOperand(ba.getArgNumber());
            }
         }
         mlir::Value key = canonicalizeStateValueForReuse(tableV);
         auto it = reuse.externalDatasourceByTableState.find(key);
         if (it == reuse.externalDatasourceByTableState.end()) it = reuse.externalDatasourceByTableState.find(tableV);
         if (it == reuse.externalDatasourceByTableState.end()) continue;
         for (auto& f : it->second.filterDescriptions) decoded.push_back(f);
      }
   }
   return decoded;
}

static mlir::Operation* findFirstOpConsumingInBlock(mlir::Block& block, mlir::Value v) {
   for (mlir::Operation& op : block.without_terminator()) {
      for (mlir::Value in : op.getOperands()) {
         if (in == v) return &op;
      }
   }
   return nullptr;
}

// Join-style plans (e.g. TPCH Q2) thread merged state through `nested_execution_group` and inner
// `execution_step` before `lookup` / `reduce`.
// Follow that chain so delay_filter can still reach the TupleStream fed into those ops.
mlir::Operation* drillToStateConsumerSkippingNestedScopes(mlir::Block& startBlock, mlir::Value startState) {
   mlir::Block* curBlock = &startBlock;
   mlir::Value curState = startState;
   for (;;) {
      mlir::Operation* op = findFirstOpConsumingInBlock(*curBlock, curState);
      if (!op) return nullptr;
      if (auto nest = mlir::dyn_cast<subop::NestedExecutionGroupOp>(op)) {
         unsigned idx = 0;
         for (; idx < nest.getNumOperands(); ++idx) {
            if (nest.getOperand(idx) == curState) break;
         }
         if (idx >= nest.getNumOperands()) return nullptr;
         curBlock = &nest.getSubOps().front();
         curState = curBlock->getArgument(idx);
         continue;
      }
      if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(op)) {
         unsigned idx = 0;
         for (; idx < step.getNumOperands(); ++idx) {
            if (step.getOperand(idx) == curState) break;
         }
         if (idx >= step.getNumOperands()) return nullptr;
         curBlock = &step.getSubOps().front();
         curState = curBlock->getArgument(idx);
         continue;
      }
      return op;
   }
}

void insertWriteSidePredIntoHashMapConstructionStep(ExecutionStepOp step,
                                                            llvm::ArrayRef<runtime::FilterDescription> filters) {
   if (filters.empty()) return;
   mlir::Block& body = step.getSubOps().front();

   // Find the table scan_refs (input table) and a gather that follows it.
   subop::ScanRefsOp scanOp;
   for (mlir::Operation& op : body.without_terminator()) {
      auto s = mlir::dyn_cast<subop::ScanRefsOp>(&op);
      if (!s) continue;
      if (mlir::isa<subop::TableType>(s.getState().getType())) { scanOp = s; break; }
   }
   assert(scanOp && "write_pred: expected a scan_refs over table");

   // Ensure scan_refs produced table_entry_ref contains all filter columns.
   {
      auto tableTy = mlir::cast<subop::TableType>(scanOp.getState().getType());
      auto stripSuffix = [](llvm::StringRef s) -> llvm::StringRef {
         size_t pos = s.find('$');
         if (pos == llvm::StringRef::npos) return s;
         return s.take_front(pos);
      };
      auto refDef = scanOp.getRef();
      auto refTy = mlir::dyn_cast<subop::TableEntryRefType>(refDef.getColumn().type);
      assert(refTy && "write_pred: expected scan_refs ref to be table_entry_ref");
      llvm::SmallVector<subop::Member> cols = refTy.getTableColumns().getMembers();
      llvm::DenseSet<subop::Member> have;
      for (auto m : cols) have.insert(m);
      auto& mm = scanOp->getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
      for (auto& f : filters) {
         if (f.op == runtime::FilterOp::NOTNULL) continue;
         subop::Member mem;
         for (auto m : tableTy.getMembers().getMembers()) {
            if (stripSuffix(mm.getName(m)) == f.columnName) { mem = m; break; }
         }
         assert(mem && "write_pred: could not find table member for filter column");
         if (have.insert(mem).second) cols.push_back(mem);
      }
      auto newCols = subop::StateMembersAttr::get(step.getContext(), cols);
      refDef.getColumn().type = subop::TableEntryRefType::get(step.getContext(), newCols);
      scanOp.setRefAttr(refDef);
   }

   subop::GatherOp firstGather;
   for (mlir::Operation& op : body.without_terminator()) {
      if (!scanOp->isBeforeInBlock(&op)) continue;
      auto g = mlir::dyn_cast<subop::GatherOp>(&op);
      if (!g) continue;
      if (g.getStream() == scanOp.getRes()) { firstGather = g; break; }
   }
   assert(firstGather && "write_pred: expected a gather right after table scan_refs");

   // Ensure filter columns exist in the gathered stream by extending the gather mapping (member->coldef).
   auto& cm = firstGather->getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = firstGather->getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   llvm::DenseMap<llvm::StringRef, tuples::ColumnRefAttr> colByName;
   for (auto& p : firstGather.getMapping().getMapping()) {
      auto def = p.second;
      auto [scope, leaf] = cm.getName(&def.getColumn());
      (void)scope;
      colByName[llvm::StringRef(leaf)] = cm.createRef(&def.getColumn());
   }
   auto tableTy = mlir::cast<subop::TableType>(scanOp.getState().getType());
   auto stripSuffix = [](llvm::StringRef s) -> llvm::StringRef {
      size_t pos = s.find('$');
      if (pos == llvm::StringRef::npos) return s;
      return s.take_front(pos);
   };
   bool needExtend = false;
   for (auto& f : filters) {
      if (f.op == runtime::FilterOp::NOTNULL) continue;
      if (colByName.contains(f.columnName)) continue;
      needExtend = true;
      break;
   }
   if (needExtend) {
      llvm::StringRef reusedScope = "write_pred";
      if (!firstGather.getMapping().getMapping().empty()) {
         auto def0 = firstGather.getMapping().getMapping().begin()->second;
         auto [scope0, leaf0] = cm.getName(&def0.getColumn());
         (void)leaf0;
         reusedScope = scope0;
      }
      llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> mappingPairs;
      for (auto& p : firstGather.getMapping().getMapping()) mappingPairs.push_back({p.first, p.second});
      for (auto& f : filters) {
         if (f.op == runtime::FilterOp::NOTNULL) continue;
         if (colByName.contains(f.columnName)) continue;
         subop::Member mem;
         for (auto m : tableTy.getMembers().getMembers()) {
            if (stripSuffix(mm.getName(m)) == f.columnName) { mem = m; break; }
         }
         assert(mem && "write_pred: could not find table member for filter column");
         tuples::ColumnDefAttr def = cm.createDef(reusedScope.str(), f.columnName);
         def.getColumn().type = mm.getType(mem);
         colByName[f.columnName] = cm.createRef(&def.getColumn());
         mappingPairs.push_back({mem, def});
      }
      mlir::OpBuilder gb(firstGather);
      gb.setInsertionPointAfter(firstGather);
      auto newMapping = subop::ColumnDefMemberMappingAttr::get(firstGather.getContext(), mappingPairs);
      auto g2 = gb.create<subop::GatherOp>(firstGather.getLoc(), firstGather.getRes(), firstGather.getRef(), newMapping);
      // Replace uses of old gather stream after it with new gather.
      mlir::Value oldS = firstGather.getRes();
      mlir::Value newS = g2.getRes();
      oldS.replaceUsesWithIf(newS, [&](mlir::OpOperand& ou) {
         if (ou.getOwner() == g2.getOperation()) return false;
         if (ou.getOwner()->getBlock() != &body) return false;
         return firstGather->isBeforeInBlock(ou.getOwner());
      });
      firstGather = g2;
   }

   // Insert a map that computes `filter_pred` bool column.
   mlir::OpBuilder pb(firstGather);
   pb.setInsertionPointAfter(firstGather);
   auto [predStream, predDef] =
      materializeRuntimeFiltersAsPredicateColumn(pb, firstGather.getLoc(), firstGather.getRes(), colByName, filters, "filter_pred");

   // Replace uses of gathered stream with predStream for subsequent ops in this step.
   mlir::Value oldS = firstGather.getRes();
   oldS.replaceUsesWithIf(predStream, [&](mlir::OpOperand& ou) {
      if (ou.getOwner() == predStream.getDefiningOp()) return false;
      if (ou.getOwner()->getBlock() != &body) return false;
      return firstGather->isBeforeInBlock(ou.getOwner());
   });

   // Also reduce the computed predicate into the ht fragment by inserting an extra reduce
   // that only updates `filter_pred$0`. This avoids rewriting the existing big reduce.
   subop::ReduceOp bigReduce;
   for (mlir::Operation& op : body.without_terminator()) {
      auto r = mlir::dyn_cast<subop::ReduceOp>(&op);
      if (!r) continue;
      auto refTy = mlir::dyn_cast<subop::LookupEntryRefType>(r.getRef().getColumn().type);
      if (!refTy) continue;
      if (!mlir::isa<subop::HashMapType>(refTy.getState())) continue;
      bigReduce = r;
      break;
   }
   assert(bigReduce && "write_pred: expected reduce into hashmap");

   // Canonical extended join hashmap for this step: align **every** column attr in the flat step
   // body that refers to the same join key layout, so lowering never sees stale `hash_map_entry_ref`
   // / `lookup_entry_ref` vs SSA buffer types.
   subop::HashMapType hmTySync = mlir::cast<subop::HashMapType>(extendHashMapTypeWithPred(
      mlir::cast<subop::HashMapType>(
         mlir::cast<subop::LookupEntryRefType>(bigReduce.getRef().getColumn().type).getState()),
      makeOrGetPredMember(step.getContext())));
   alignJoinHashMapStepColumnAttrs(step, hmTySync);

   subop::Member predMember;
   {
      auto& mm = bigReduce->getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
      for (auto m : hmTySync.getValueMembers().getMembers()) {
         if (mm.getName(m) == "filter_pred$0") {
            predMember = m;
            break;
         }
      }
      assert(predMember && "write_pred: could not find filter_pred$0 member in hashmap type");
   }

   tuples::ColumnRefAttr predColRef = cm.createRef(&predDef.getColumn());
   {
      mlir::OpBuilder rb(bigReduce);
      rb.setInsertionPointAfter(bigReduce);
      auto cols = mlir::ArrayAttr::get(step.getContext(), mlir::ArrayRef<mlir::Attribute>{predColRef});
      auto mems = mlir::ArrayAttr::get(step.getContext(), mlir::ArrayRef<mlir::Attribute>{subop::MemberAttr::get(step.getContext(), predMember)});
      auto predReduce =
         rb.create<subop::ReduceOp>(bigReduce.getLoc(), bigReduce.getStream(), bigReduce.getRef(), cols, mems);

      // Region 0: (predCol, currPred) -> incomingPred (= predCol)
      {
         mlir::Block* b = new mlir::Block();
         predReduce.getRegion().push_back(b);
         b->addArgument(predDef.getColumn().type, bigReduce.getLoc());
         b->addArgument(mlir::IntegerType::get(step.getContext(), 1), bigReduce.getLoc());
         mlir::OpBuilder bb(step.getContext());
         bb.setInsertionPointToStart(b);
         bb.create<tuples::ReturnOp>(bigReduce.getLoc(), mlir::ValueRange{b->getArgument(0)});
      }
      // Combine region: (currPred, incomingPred) -> (currPred & incomingPred)
      {
         mlir::Block* b = new mlir::Block();
         predReduce.getCombine().push_back(b);
         b->addArgument(mlir::IntegerType::get(step.getContext(), 1), bigReduce.getLoc());
         b->addArgument(mlir::IntegerType::get(step.getContext(), 1), bigReduce.getLoc());
         mlir::OpBuilder bb(step.getContext());
         bb.setInsertionPointToStart(b);
         mlir::Value merged = bb.create<mlir::arith::AndIOp>(bigReduce.getLoc(), b->getArgument(0), b->getArgument(1));
         bb.create<tuples::ReturnOp>(bigReduce.getLoc(), mlir::ValueRange{merged});
      }
   }
}

/// Join hash build via `growing buffer` + `create_hash_indexed_view`: materialize `filter_pred$0` from table scan.
void insertWriteSidePredIntoBufferConstructionStep(ExecutionStepOp step,
                                                          llvm::ArrayRef<runtime::FilterDescription> filters) {
   if (filters.empty()) return;
   mlir::Block& body = step.getSubOps().front();

   subop::ScanRefsOp scanOp;
   for (mlir::Operation& op : body.without_terminator()) {
      auto s = mlir::dyn_cast<subop::ScanRefsOp>(&op);
      if (!s) continue;
      if (mlir::isa<subop::TableType>(s.getState().getType())) {
         scanOp = s;
         break;
      }
   }
   assert(scanOp && "write_pred_buf: expected a scan_refs over table");

   {
      auto tableTy = mlir::cast<subop::TableType>(scanOp.getState().getType());
      auto stripSuffix = [](llvm::StringRef s) -> llvm::StringRef {
         size_t pos = s.find('$');
         if (pos == llvm::StringRef::npos) return s;
         return s.take_front(pos);
      };
      auto refDef = scanOp.getRef();
      auto refTy = mlir::dyn_cast<subop::TableEntryRefType>(refDef.getColumn().type);
      assert(refTy && "write_pred_buf: expected scan_refs ref to be table_entry_ref");
      llvm::SmallVector<subop::Member> cols = refTy.getTableColumns().getMembers();
      llvm::DenseSet<subop::Member> have;
      for (auto m : cols) have.insert(m);
      auto& mm = scanOp->getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
      for (auto& f : filters) {
         if (f.op == runtime::FilterOp::NOTNULL) continue;
         subop::Member mem;
         for (auto m : tableTy.getMembers().getMembers()) {
            if (stripSuffix(mm.getName(m)) == f.columnName) {
               mem = m;
               break;
            }
         }
         assert(mem && "write_pred_buf: could not find table member for filter column");
         if (have.insert(mem).second) cols.push_back(mem);
      }
      auto newCols = subop::StateMembersAttr::get(step.getContext(), cols);
      refDef.getColumn().type = subop::TableEntryRefType::get(step.getContext(), newCols);
      scanOp.setRefAttr(refDef);
   }

   subop::GatherOp firstGather;
   for (mlir::Operation& op : body.without_terminator()) {
      if (!scanOp->isBeforeInBlock(&op)) continue;
      auto g = mlir::dyn_cast<subop::GatherOp>(&op);
      if (!g) continue;
      if (g.getStream() == scanOp.getRes()) {
         firstGather = g;
         break;
      }
   }
   assert(firstGather && "write_pred_buf: expected a gather right after table scan_refs");

   auto& cm = firstGather->getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = firstGather->getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   llvm::DenseMap<llvm::StringRef, tuples::ColumnRefAttr> colByName;
   for (auto& p : firstGather.getMapping().getMapping()) {
      auto def = p.second;
      auto [scope, leaf] = cm.getName(&def.getColumn());
      (void)scope;
      colByName[llvm::StringRef(leaf)] = cm.createRef(&def.getColumn());
   }
   auto tableTy = mlir::cast<subop::TableType>(scanOp.getState().getType());
   auto stripSuffix = [](llvm::StringRef s) -> llvm::StringRef {
      size_t pos = s.find('$');
      if (pos == llvm::StringRef::npos) return s;
      return s.take_front(pos);
   };
   bool needExtend = false;
   for (auto& f : filters) {
      if (f.op == runtime::FilterOp::NOTNULL) continue;
      if (colByName.contains(f.columnName)) continue;
      needExtend = true;
      break;
   }
   if (needExtend) {
      llvm::StringRef reusedScope = "write_pred";
      if (!firstGather.getMapping().getMapping().empty()) {
         auto def0 = firstGather.getMapping().getMapping().begin()->second;
         auto [scope0, leaf0] = cm.getName(&def0.getColumn());
         (void)leaf0;
         reusedScope = scope0;
      }
      llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> mappingPairs;
      for (auto& p : firstGather.getMapping().getMapping()) mappingPairs.push_back({p.first, p.second});
      for (auto& f : filters) {
         if (f.op == runtime::FilterOp::NOTNULL) continue;
         if (colByName.contains(f.columnName)) continue;
         subop::Member mem;
         for (auto m : tableTy.getMembers().getMembers()) {
            if (stripSuffix(mm.getName(m)) == f.columnName) {
               mem = m;
               break;
            }
         }
         assert(mem && "write_pred_buf: could not find table member for filter column");
         tuples::ColumnDefAttr def = cm.createDef(reusedScope.str(), f.columnName);
         def.getColumn().type = mm.getType(mem);
         colByName[f.columnName] = cm.createRef(&def.getColumn());
         mappingPairs.push_back({mem, def});
      }
      mlir::OpBuilder gb(firstGather);
      gb.setInsertionPointAfter(firstGather);
      auto newMapping = subop::ColumnDefMemberMappingAttr::get(firstGather.getContext(), mappingPairs);
      auto g2 = gb.create<subop::GatherOp>(firstGather.getLoc(), firstGather.getRes(), firstGather.getRef(), newMapping);
      mlir::Value oldS = firstGather.getRes();
      mlir::Value newS = g2.getRes();
      oldS.replaceUsesWithIf(newS, [&](mlir::OpOperand& ou) {
         if (ou.getOwner() == g2.getOperation()) return false;
         if (ou.getOwner()->getBlock() != &body) return false;
         return firstGather->isBeforeInBlock(ou.getOwner());
      });
      firstGather = g2;
   }

   mlir::OpBuilder pb(firstGather);
   pb.setInsertionPointAfter(firstGather);
   auto [predStream, predDef] =
      materializeRuntimeFiltersAsPredicateColumn(pb, firstGather.getLoc(), firstGather.getRes(), colByName, filters, "filter_pred");

   mlir::Value oldS = firstGather.getRes();
   oldS.replaceUsesWithIf(predStream, [&](mlir::OpOperand& ou) {
      if (ou.getOwner() == predStream.getDefiningOp()) return false;
      if (ou.getOwner()->getBlock() != &body) return false;
      return firstGather->isBeforeInBlock(ou.getOwner());
   });

   // Materialize may target either `!subop.buffer<...>` or `!subop.thread_local<!subop.buffer<...>>`
   // (parallel / align passes sometimes thread_local-wrap the state SSA). The layout with
   // `filter_pred$0` lives on the inner buffer type in both cases.
   subop::MaterializeOp matOp;
   step.getOperation()->walk([&](subop::MaterializeOp m) {
      subop::BufferType bufTy = getInnerBufferTypeForMaterializeState(m.getState().getType());
      if (!bufTy) return mlir::WalkResult::advance();
      if (!valueMembersContainMemberNamed(step.getContext(), bufTy.getMembers(), "filter_pred$0"))
         return mlir::WalkResult::advance();
      matOp = m;
      return mlir::WalkResult::interrupt();
   });
   assert(matOp && "write_pred_buf: expected materialize into join buffer with filter_pred$0");

   subop::Member predMember;
   {
      subop::BufferType bufTy = getInnerBufferTypeForMaterializeState(matOp.getState().getType());
      assert(bufTy && "write_pred_buf: materialize state must be buffer or thread_local<buffer>");
      for (auto m : bufTy.getMembers().getMembers()) {
         if (mm.getName(m) == "filter_pred$0") {
            predMember = m;
            break;
         }
      }
      assert(predMember && "write_pred_buf: filter_pred$0 missing on buffer type");
   }

   tuples::ColumnRefAttr predColRef = cm.createRef(&predDef.getColumn());
   llvm::SmallVector<subop::RefMappingPairT> pairs;
   for (auto& pr : matOp.getMapping().getMapping()) pairs.push_back(pr);
   pairs.push_back({predMember, predColRef});
   matOp.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(step.getContext(), pairs));
}

/// `rewriteHashmapTypesInModule` can extend join buffers with `filter_pred$0` on the type before any
/// writer fills that member (e.g. no decoded table filters for that buffer, or predicate lowering did
/// not attach to this `materialize`). Uninitialized bits make downstream `filter(all_true ...)` drop
/// all rows. When the buffer layout includes `filter_pred$0` but the `materialize` mapping does not,
/// append a constant-true predicate column and map it into `filter_pred$0`.
void ensureJoinBufferFilterPredMaterializeMappings(mlir::ModuleOp module) {
   auto* ctx = module.getContext();
   auto* subDialect = ctx->getLoadedDialect<subop::SubOperatorDialect>();
   auto* tupleDialect = ctx->getLoadedDialect<tuples::TupleStreamDialect>();
   if (!subDialect || !tupleDialect) return;
   auto& mm = subDialect->getMemberManager();
   auto& cm = tupleDialect->getColumnManager();

   llvm::SmallVector<subop::MaterializeOp, 16> todo;
   module.walk([&](subop::MaterializeOp m) {
      subop::BufferType bufTy = getInnerBufferTypeForMaterializeState(m.getState().getType());
      if (!bufTy) return mlir::WalkResult::advance();
      subop::Member predM;
      for (auto mem : bufTy.getMembers().getMembers()) {
         if (mm.getName(mem) == "filter_pred$0") {
            predM = mem;
            break;
         }
      }
      if (!predM) return mlir::WalkResult::advance();
      for (auto& pr : m.getMapping().getMapping()) {
         if (pr.first == predM) return mlir::WalkResult::advance();
      }
      todo.push_back(m);
      return mlir::WalkResult::advance();
   });

   for (subop::MaterializeOp m : todo) {
      mlir::OpBuilder pb(m);
      mlir::Location loc = m.getLoc();
      std::string sc = cm.getUniqueScope("jp_default_pred");
      tuples::ColumnDefAttr def = cm.createDef(sc, "t");
      def.getColumn().type = mlir::IntegerType::get(ctx, 1);
      tuples::ColumnRefAttr predRef = cm.createRef(&def.getColumn());

      subop::MapCreationHelper helper(ctx);
      helper.buildBlock(pb, [&](mlir::OpBuilder& rb) {
         mlir::Value t = rb.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
         rb.create<tuples::ReturnOp>(loc, mlir::ValueRange{t});
      });

      mlir::Value stream = m.getStream();
      pb.setInsertionPoint(m);
      auto mapOp = pb.create<subop::MapOp>(loc, tuples::TupleStreamType::get(ctx), stream,
                                           pb.getArrayAttr({def}), helper.getColRefs());
      mapOp.getFn().push_back(helper.getMapBlock());
      m->setOperand(0, mapOp.getResult());

      subop::BufferType bufTy = getInnerBufferTypeForMaterializeState(m.getState().getType());
      assert(bufTy);
      subop::Member predM;
      for (auto mem : bufTy.getMembers().getMembers()) {
         if (mm.getName(mem) == "filter_pred$0") {
            predM = mem;
            break;
         }
      }
      assert(predM);
      llvm::SmallVector<subop::RefMappingPairT> pairs;
      for (auto pr : m.getMapping().getMapping()) pairs.push_back(pr);
      pairs.push_back({predM, predRef});
      m.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, pairs));
   }
}

// For scan_refs over join `hashmap`: filter by stored predicate member `filter_pred$0`.
void insertScanRefsPredFilter(ExecutionStepOp step, subop::Member predMember) {
   mlir::Block& body = step.getSubOps().front();
   auto& cm = step.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = step.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();

   subop::ScanRefsOp scanOp;
   for (mlir::Operation& op : body.without_terminator()) {
      auto s = mlir::dyn_cast<subop::ScanRefsOp>(&op);
      if (!s) continue;
      if (mlir::isa<subop::HashMapType>(s.getState().getType())) { scanOp = s; break; }
   }
   if (!scanOp) return;

   // Find first gather after scan, fed by scan result.
   subop::GatherOp gatherOp;
   for (mlir::Operation& op : body.without_terminator()) {
      if (!scanOp->isBeforeInBlock(&op)) continue;
      auto g = mlir::dyn_cast<subop::GatherOp>(&op);
      if (!g) continue;
      if (g.getStream() == scanOp.getRes()) { gatherOp = g; break; }
   }
   assert(gatherOp && "pred_filter: expected gather directly after scan_refs(ht)");

   // Ensure gather includes predicate column.
   bool hasPred = false;
   for (auto& p : gatherOp.getMapping().getMapping()) {
      if (p.first == predMember) { hasPred = true; break; }
   }
   tuples::ColumnRefAttr predRef;
   if (!hasPred) {
      // Reuse existing scope and add a new column def.
      llvm::StringRef scope = "pred_filter";
      if (!gatherOp.getMapping().getMapping().empty()) {
         auto def0 = gatherOp.getMapping().getMapping().begin()->second;
         auto [scope0, leaf0] = cm.getName(&def0.getColumn());
         (void)leaf0;
         scope = scope0;
      }
      tuples::ColumnDefAttr predDef = cm.createDef(scope.str(), "filter_pred");
      predDef.getColumn().type = mm.getType(predMember);
      predRef = cm.createRef(&predDef.getColumn());

      llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> mappingPairs;
      for (auto& p : gatherOp.getMapping().getMapping()) mappingPairs.push_back({p.first, p.second});
      mappingPairs.push_back({predMember, predDef});

      mlir::OpBuilder gb(gatherOp);
      gb.setInsertionPointAfter(gatherOp);
      auto newMapping = subop::ColumnDefMemberMappingAttr::get(gatherOp.getContext(), mappingPairs);
      auto g2 = gb.create<subop::GatherOp>(gatherOp.getLoc(), gatherOp.getRes(), gatherOp.getRef(), newMapping);
      gatherOp = g2;
   } else {
      // Find existing pred column def.
      for (auto& p : gatherOp.getMapping().getMapping()) {
         if (p.first == predMember) {
            predRef = cm.createRef(&p.second.getColumn());
            break;
         }
      }
      assert(predRef);
   }

   // Insert filter(all_true [predRef]) after gatherOp.
   mlir::OpBuilder fb(gatherOp);
   fb.setInsertionPointAfter(gatherOp);
   auto filterOp = fb.create<subop::FilterOp>(gatherOp.getLoc(), gatherOp.getRes(), subop::FilterSemantic::all_true,
                                              fb.getArrayAttr({predRef}));

   // Rewrite uses of gathered stream after insertion point.
   mlir::Value orig = gatherOp.getRes();
   mlir::Value filtered = filterOp.getRes();
   mlir::Block* gatherBlock = gatherOp->getBlock();
   orig.replaceUsesWithIf(filtered, [&](mlir::OpOperand& ou) {
      mlir::Operation* owner = ou.getOwner();
      if (owner == filtered.getDefiningOp()) return false;
      if (owner->getBlock() != gatherBlock) return false;
      return gatherOp->isBeforeInBlock(owner);
   });
}

/// After `lookup` / `scan_list` on `!subop.hash_indexed_view<...>` with `filter_pred$0`, re-check the stored bool.
void insertHashIndexedViewGatherPredFilters(ExecutionStepOp step, subop::Member predMember) {
   auto* ctx = step.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();

   llvm::SmallVector<subop::GatherOp> gathers;
   step.getOperation()->walk([&](subop::GatherOp g) { gathers.push_back(g); });

   for (subop::GatherOp gatherOp : gathers) {
      auto refTy = mlir::dyn_cast<subop::LookupEntryRefType>(gatherOp.getRef().getColumn().type);
      if (!refTy) continue;
      auto hiv = mlir::dyn_cast<subop::HashIndexedViewType>(refTy.getState());
      if (!hiv) continue;
      if (!valueMembersContainMemberNamed(ctx, hiv.getValueMembers(), "filter_pred$0")) continue;

      bool hasPred = false;
      for (auto& p : gatherOp.getMapping().getMapping()) {
         if (p.first == predMember) {
            hasPred = true;
            break;
         }
      }
      tuples::ColumnRefAttr predRef;
      if (!hasPred) {
         llvm::StringRef scope = "pred_filter";
         if (!gatherOp.getMapping().getMapping().empty()) {
            auto def0 = gatherOp.getMapping().getMapping().begin()->second;
            auto [scope0, leaf0] = cm.getName(&def0.getColumn());
            (void)leaf0;
            scope = scope0;
         }
         tuples::ColumnDefAttr predDef = cm.createDef(scope.str(), "filter_pred");
         predDef.getColumn().type = mm.getType(predMember);
         predRef = cm.createRef(&predDef.getColumn());

         llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> mappingPairs;
         for (auto& p : gatherOp.getMapping().getMapping()) mappingPairs.push_back({p.first, p.second});
         mappingPairs.push_back({predMember, predDef});

         mlir::OpBuilder gb(gatherOp);
         gb.setInsertionPointAfter(gatherOp);
         auto newMapping = subop::ColumnDefMemberMappingAttr::get(gatherOp.getContext(), mappingPairs);
         auto g2 = gb.create<subop::GatherOp>(gatherOp.getLoc(), gatherOp.getRes(), gatherOp.getRef(), newMapping);
         gatherOp = g2;
      } else {
         for (auto& p : gatherOp.getMapping().getMapping()) {
            if (p.first == predMember) {
               predRef = cm.createRef(&p.second.getColumn());
               break;
            }
         }
         assert(predRef);
      }

      mlir::OpBuilder fb(gatherOp);
      fb.setInsertionPointAfter(gatherOp);
      auto filterOp = fb.create<subop::FilterOp>(gatherOp.getLoc(), gatherOp.getRes(), subop::FilterSemantic::all_true,
                                                 fb.getArrayAttr({predRef}));

      mlir::Value orig = gatherOp.getRes();
      mlir::Value filtered = filterOp.getRes();
      mlir::Block* gatherBlock = gatherOp->getBlock();
      orig.replaceUsesWithIf(filtered, [&](mlir::OpOperand& ou) {
         mlir::Operation* owner = ou.getOwner();
         if (owner == filtered.getDefiningOp()) return false;
         if (owner->getBlock() != gatherBlock) return false;
         return gatherOp->isBeforeInBlock(owner);
      });
   }
}

/// Forward walk (uses + `merge` + `execution_step` region args) until we see `subop.create_hash_indexed_view`
/// whose `source` is a tracked `!subop.buffer` SSA value.
static bool joinBufferSeedsReachCreateHashIndexedView(llvm::ArrayRef<mlir::Value> seeds) {
   llvm::DenseSet<void*> visited;
   llvm::SmallVector<mlir::Value> stack;
   auto push = [&](mlir::Value v) {
      if (!v) return;
      if (!mlir::isa<subop::BufferType, subop::ThreadLocalType>(v.getType())) return;
      if (!visited.insert(v.getAsOpaquePointer()).second) return;
      stack.push_back(v);
   };
   for (mlir::Value s : seeds) push(s);

   while (!stack.empty()) {
      mlir::Value v = stack.back();
      stack.pop_back();
      for (mlir::OpOperand& use : v.getUses()) {
         mlir::Operation* op = use.getOwner();
         if (mlir::isa<subop::BufferType>(v.getType())) {
            if (auto chiv = mlir::dyn_cast<subop::CreateHashIndexedView>(op)) {
               if (chiv.getSource() == v) return true;
            }
         }
         if (auto merge = mlir::dyn_cast<subop::MergeOp>(op)) {
            if (merge.getThreadLocal() == v) push(merge.getResult());
         }
         if (mlir::Value inner = mapStateThroughExecutionStepOperands(v, op)) push(inner);
      }
   }
   return false;
}

/// Filter `!subop.buffer` reuse targets to those that (per `reuse` RW + SSA forward walk) feed a HIV build.
llvm::SmallVector<mlir::Value, 8> collectJoinBuffersFeedingHashIndexedView(llvm::ArrayRef<mlir::Value> candidates,
                                                                                 const ModuleReuseInfo& reuse) {
   llvm::SmallVector<mlir::Value, 8> out;
   for (mlir::Value cand : candidates) {
      auto bt = mlir::dyn_cast_or_null<subop::BufferType>(cand.getType());
      if (!bt) continue;
      if (valueMembersContainMemberNamed(bt.getContext(), bt.getMembers(), "filter_pred$0")) continue;

      llvm::SmallVector<mlir::Value, 8> seeds;
      collectJoinBufferReachabilitySeeds(cand, reuse, seeds);
      if (joinBufferSeedsReachCreateHashIndexedView(seeds)) out.push_back(cand);
   }
   return out;
}

llvm::DenseMap<mlir::Value, llvm::SmallVector<runtime::FilterDescription, 8>>
decodeFiltersByCacheTargets(llvm::ArrayRef<CacheTarget> targets, const ModuleReuseInfo& reuse) {
   auto rwByStepOp = buildRwByStepOpMap(reuse);
   llvm::DenseMap<mlir::Value, llvm::SmallVector<runtime::FilterDescription, 8>> decodedFiltersByTarget;
   for (auto& t : targets) {
      if (!t.state) continue;
      mlir::Value filterSeed = t.state;
      if (auto itG = findReuseMap(reuse.hashIndexedViewFromMergedBuffer, t.state);
          itG != reuse.hashIndexedViewFromMergedBuffer.end()) {
         filterSeed = itG->second;
      }
      llvm::SmallVector<runtime::FilterDescription, 8> decoded =
         decodeFiltersForStateFromWriterSteps(filterSeed, reuse, &rwByStepOp);
      if (auto itTL = findReuseMap(reuse.mergedFromThreadLocal, filterSeed);
          itTL != reuse.mergedFromThreadLocal.end()) {
         auto fromTl = decodeFiltersForStateFromWriterSteps(itTL->second, reuse, &rwByStepOp);
         decoded.append(fromTl.begin(), fromTl.end());
      }
      if (!decoded.empty()) {
         decodedFiltersByTarget[t.state] = std::move(decoded);
      }
   }
   return decodedFiltersByTarget;
}

void materializeRuntimeFiltersAtCacheGetUses(mlir::Value cached,
                                             llvm::ArrayRef<runtime::FilterDescription> decodedFilters) {
   if (decodedFilters.empty()) return;

   llvm::SmallVector<mlir::OpOperand*, 16> uses;
   for (auto& u : cached.getUses()) {
      uses.push_back(&u);
   }
   for (mlir::OpOperand* u : uses) {
      mlir::Operation* owner = u->getOwner();
      subop::ExecutionStepOp step;
      mlir::Value stateInBody;
      if (auto stepOwner = mlir::dyn_cast<subop::ExecutionStepOp>(owner)) {
         step = stepOwner;
         unsigned argNo = u->getOperandNumber();
         assert(argNo < step.getNumOperands());
         stateInBody = step.getSubOps().front().getArgument(argNo);
      } else {
         step = owner->getParentOfType<subop::ExecutionStepOp>();
         if (!step) continue;
         stateInBody = u->get();
      }

      mlir::Operation* useOp = drillToStateConsumerSkippingNestedScopes(step.getSubOps().front(), stateInBody);
      if (!useOp) continue;

      mlir::OpOperand* streamOperand = nullptr;
      for (mlir::OpOperand& ou : useOp->getOpOperands()) {
         if (mlir::isa<tuples::TupleStreamType>(ou.get().getType())) {
            streamOperand = &ou;
            break;
         }
      }

      subop::GatherOp gatherOp;
      if (streamOperand) {
         mlir::Value inStream = streamOperand->get();
         gatherOp = mlir::dyn_cast_or_null<subop::GatherOp>(inStream.getDefiningOp());
         if (!gatherOp) continue;
      } else if (auto scan = mlir::dyn_cast<subop::ScanRefsOp>(useOp)) {
         assert(scan->getNumResults() == 1);
         mlir::Value scanStream = scan.getResult();
         mlir::Block* scanBlock = scan->getBlock();
         for (mlir::Operation& op : scanBlock->without_terminator()) {
            if (!scan->isBeforeInBlock(&op)) continue;
            auto g = mlir::dyn_cast<subop::GatherOp>(&op);
            if (!g) continue;
            if (g.getStream() == scanStream) {
               gatherOp = g;
               break;
            }
         }
         if (!gatherOp) continue;
      } else {
         continue;
      }

      llvm::DenseMap<llvm::StringRef, tuples::ColumnRefAttr> colByName;
      auto& cm = gatherOp->getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      for (auto& pair : gatherOp.getMapping().getMapping()) {
         auto def = pair.second;
         auto [scope, leaf] = cm.getName(&def.getColumn());
         (void)scope;
         colByName[llvm::StringRef(leaf)] = cm.createRef(&def.getColumn());
      }

      bool anyMissing = false;
      for (auto& f : decodedFilters) {
         if (f.op == runtime::FilterOp::NOTNULL) continue;
         if (colByName.contains(f.columnName)) continue;
         anyMissing = true;
         break;
      }
      if (anyMissing) {
         if (!streamOperand) continue;

         subop::ScanRefsOp scanOp;
         mlir::Block* gBlock = gatherOp->getBlock();
         for (mlir::Operation& op : gBlock->without_terminator()) {
            if (&op == gatherOp.getOperation()) break;
            if (auto s = mlir::dyn_cast<subop::ScanRefsOp>(&op)) {
               if (mlir::isa<subop::TableType>(s.getState().getType())) scanOp = s;
            }
         }
         assert(scanOp && "delay_filter: missing filter columns but could not find scan_refs before gather");
         auto tableTy = mlir::dyn_cast<subop::TableType>(scanOp.getState().getType());
         assert(tableTy && "delay_filter: scan_refs state must be a subop.table");

         llvm::StringRef reusedScope = "delay_filter";
         if (!gatherOp.getMapping().getMapping().empty()) {
            auto def0 = gatherOp.getMapping().getMapping().begin()->second;
            auto [scope0, leaf0] = cm.getName(&def0.getColumn());
            (void)leaf0;
            reusedScope = scope0;
         }

         auto& mm = gatherOp->getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
         llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>, 8> extra;
         auto stripSuffix = [](llvm::StringRef s) -> llvm::StringRef {
            size_t pos = s.find('$');
            if (pos == llvm::StringRef::npos) return s;
            return s.take_front(pos);
         };
         for (auto& f : decodedFilters) {
            if (f.op == runtime::FilterOp::NOTNULL) continue;
            if (colByName.contains(f.columnName)) continue;

            subop::Member mem;
            for (auto m : tableTy.getMembers().getMembers()) {
               if (stripSuffix(mm.getName(m)) == f.columnName) {
                  mem = m;
                  break;
               }
            }
            assert(mem && "delay_filter: could not find table member for filter column");

            tuples::ColumnDefAttr def = cm.createDef(reusedScope.str(), f.columnName);
            def.getColumn().type = mm.getType(mem);
            colByName[f.columnName] = cm.createRef(&def.getColumn());
            extra.push_back({mem, def});
         }

         if (!extra.empty()) {
            llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> mappingPairs;
            for (auto& p : gatherOp.getMapping().getMapping()) {
               mappingPairs.push_back({p.first, p.second});
            }
            for (auto& e : extra) mappingPairs.push_back(e);

            mlir::OpBuilder gb(gatherOp);
            gb.setInsertionPointAfter(gatherOp);
            auto newMapping = subop::ColumnDefMemberMappingAttr::get(gatherOp.getContext(), mappingPairs);
            auto g2 = gb.create<subop::GatherOp>(gatherOp.getLoc(), gatherOp.getRes(), gatherOp.getRef(), newMapping);
            gatherOp = g2;
         }
      }

      mlir::OpBuilder builder(gatherOp);
      builder.setInsertionPointAfter(gatherOp);
      mlir::Value filteredStream = materializeRuntimeFiltersAsSubopFilter(
         builder, gatherOp.getLoc(), gatherOp.getRes(), colByName, decodedFilters);

      if (streamOperand) {
         streamOperand->set(filteredStream);
      } else {
         mlir::Value orig = gatherOp.getRes();
         mlir::Block* gatherBlock = gatherOp->getBlock();
         orig.replaceUsesWithIf(filteredStream, [&](mlir::OpOperand& ou) {
            mlir::Operation* ouOwner = ou.getOwner();
            if (ouOwner == filteredStream.getDefiningOp()) return false;
            if (ouOwner->getBlock() != gatherBlock) return false;
            return gatherOp->isBeforeInBlock(ouOwner);
         });
      }
   }
}

} // namespace lingodb::compiler::dialect::subop
