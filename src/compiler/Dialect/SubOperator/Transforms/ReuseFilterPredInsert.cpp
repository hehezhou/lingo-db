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
#include "lingodb/compiler/Dialect/util/UtilTypes.h"
#include "lingodb/runtime/ExternalDataSourceProperty.h"
#include "lingodb/utility/Serialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include <cassert>
#include <optional>

namespace lingodb::compiler::dialect::subop {

static std::optional<unsigned> parseFilterPredMemberSlot(llvm::StringRef memberName) {
   if (!memberName.consume_front("filter_pred$")) return std::nullopt;
   unsigned slot = 0;
   if (memberName.getAsInteger(10, slot)) return std::nullopt;
   return slot;
}

static mlir::Type cloneFilterPredTypeToContext(mlir::Type ty, mlir::MLIRContext* ctx) {
   if (!ty) return ty;
   if (auto nullable = mlir::dyn_cast<lingodb::compiler::dialect::db::NullableType>(ty))
      return lingodb::compiler::dialect::db::NullableType::get(
         cloneFilterPredTypeToContext(nullable.getType(), ctx));
   if (auto tuple = mlir::dyn_cast<mlir::TupleType>(ty)) {
      llvm::SmallVector<mlir::Type> types;
      for (mlir::Type elem : tuple.getTypes()) types.push_back(cloneFilterPredTypeToContext(elem, ctx));
      return mlir::TupleType::get(ctx, types);
   }
   if (auto i = mlir::dyn_cast<mlir::IntegerType>(ty))
      return mlir::IntegerType::get(ctx, i.getWidth(), i.getSignedness());
   if (mlir::isa<mlir::IndexType>(ty)) return mlir::IndexType::get(ctx);
   if (auto f = mlir::dyn_cast<mlir::FloatType>(ty)) {
      if (f.isF64()) return mlir::Float64Type::get(ctx);
      if (f.isF32()) return mlir::Float32Type::get(ctx);
      if (f.isF16()) return mlir::Float16Type::get(ctx);
   }
   if (auto c = mlir::dyn_cast<lingodb::compiler::dialect::db::CharType>(ty))
      return lingodb::compiler::dialect::db::CharType::get(ctx, c.getLen());
   if (mlir::isa<lingodb::compiler::dialect::db::StringType>(ty))
      return lingodb::compiler::dialect::db::StringType::get(ctx);
   if (auto d = mlir::dyn_cast<lingodb::compiler::dialect::db::DateType>(ty))
      return lingodb::compiler::dialect::db::DateType::get(ctx, d.getUnit());
   if (auto t = mlir::dyn_cast<lingodb::compiler::dialect::db::TimestampType>(ty))
      return lingodb::compiler::dialect::db::TimestampType::get(ctx, t.getUnit());
   if (auto dec = mlir::dyn_cast<lingodb::compiler::dialect::db::DecimalType>(ty))
      return lingodb::compiler::dialect::db::DecimalType::get(ctx, dec.getP(), dec.getS());
   if (auto ref = mlir::dyn_cast<lingodb::compiler::dialect::util::RefType>(ty))
      return lingodb::compiler::dialect::util::RefType::get(
         ctx, cloneFilterPredTypeToContext(ref.getElementType(), ctx));
   if (auto buf = mlir::dyn_cast<lingodb::compiler::dialect::util::BufferType>(ty))
      return lingodb::compiler::dialect::util::BufferType::get(ctx, cloneFilterPredTypeToContext(buf.getT(), ctx));
   if (mlir::isa<lingodb::compiler::dialect::util::VarLen32Type>(ty))
      return lingodb::compiler::dialect::util::VarLen32Type::get(ctx);
   llvm_unreachable("filter pred insert: unsupported cross-context column type");
}

static subop::StateMembersAttr appendMember(mlir::MLIRContext* ctx, subop::StateMembersAttr members,
                                            subop::Member m);
bool valueMembersContainMemberNamed(mlir::MLIRContext* ctx, subop::StateMembersAttr members,
                                    llvm::StringRef name);

static bool hashIndexedViewHasFilterPredMember(mlir::MLIRContext* ctx, subop::HashIndexedViewType hiv) {
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   for (subop::Member m : hiv.getValueMembers().getMembers()) {
      if (parseFilterPredMemberSlot(mm.getName(m))) return true;
   }
   return false;
}

static bool hashIndexedViewLikeHasFilterPredMember(mlir::MLIRContext* ctx, mlir::Type t) {
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   subop::StateMembersAttr valueMembers;
   if (auto hiv = mlir::dyn_cast<subop::HashIndexedViewType>(t)) {
      valueMembers = hiv.getValueMembers();
   } else if (auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(t)) {
      valueMembers = mixed.getValueMembers();
   } else {
      return false;
   }
   for (subop::Member m : valueMembers.getMembers()) {
      if (parseFilterPredMemberSlot(mm.getName(m))) return true;
   }
   return false;
}

static subop::MixedHashIndexedViewType getMixedHashIndexedViewTypeForPredMember(mlir::MLIRContext* ctx,
                                                                                mlir::Type t,
                                                                                subop::Member predMember) {
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   subop::StateMembersAttr keyMembers;
   subop::StateMembersAttr valueMembers;
   bool compareHashForLookup = false;
   if (auto hiv = mlir::dyn_cast<subop::HashIndexedViewType>(t)) {
      keyMembers = hiv.getKeyMembers();
      valueMembers = hiv.getValueMembers();
      compareHashForLookup = hiv.getCompareHashForLookup();
   } else if (auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(t)) {
      keyMembers = mixed.getKeyMembers();
      valueMembers = mixed.getValueMembers();
      compareHashForLookup = mixed.getCompareHashForLookup();
   } else {
      return nullptr;
   }
   if (!valueMembersContainMemberNamed(ctx, valueMembers, mm.getName(predMember))) {
      valueMembers = appendMember(ctx, valueMembers, predMember);
   }
   return subop::MixedHashIndexedViewType::get(ctx, keyMembers, valueMembers, compareHashForLookup,
                                               mlir::StringAttr::get(ctx, mm.getName(predMember)));
}

subop::Member makeOrGetPredMemberForSlot(mlir::MLIRContext* ctx, unsigned slot) {
   auto* d = ctx->getLoadedDialect<subop::SubOperatorDialect>();
   assert(d && "SubOperatorDialect must be loaded");
   auto& mm = d->getMemberManager();
   std::string name = ("filter_pred$" + llvm::Twine(slot)).str();
   return mm.getOrCreateMemberDirect(name, mlir::IntegerType::get(ctx, 1), /*allowTypeUpdate=*/false);
}

subop::Member makeOrGetPredMember(mlir::MLIRContext* ctx) { return makeOrGetPredMemberForSlot(ctx, 0); }

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

static bool valueMembersContainAnyFilterPred(mlir::MLIRContext* ctx, subop::StateMembersAttr members) {
   auto* d = ctx->getLoadedDialect<subop::SubOperatorDialect>();
   assert(d);
   auto& mm = d->getMemberManager();
   for (auto m : members.getMembers()) {
      if (parseFilterPredMemberSlot(mm.getName(m))) return true;
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
   if (!valueMembersContainAnyFilterPred(ctx, bufTy.getMembers())) return;
   subop::Member linkM = chiv.getLinkMember().getMember();
   subop::Member hashM = chiv.getHashMember().getMember();
   llvm::SmallVector<subop::Member> vals;
   for (subop::Member m : bufTy.getMembers().getMembers()) {
      if (m == linkM || m == hashM) continue;
      vals.push_back(m);
   }
   auto oldHiv = mlir::dyn_cast<subop::HashIndexedViewType>(chiv.getType());
   auto oldMixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(chiv.getType());
   if (!oldHiv && !oldMixed) return;
   auto keyMs = subop::StateMembersAttr::get(ctx, llvm::SmallVector<subop::Member>{hashM});
   auto valMs = subop::StateMembersAttr::get(ctx, vals);
   bool compareHashForLookup = oldHiv ? oldHiv.getCompareHashForLookup() : oldMixed.getCompareHashForLookup();
   auto newHiv = subop::HashIndexedViewType::get(ctx, keyMs, valMs, compareHashForLookup);
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
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   if (valueMembersContainMemberNamed(ctx, hiv.getValueMembers(), mm.getName(predMember))) return hiv;
   // Identical-filter reuse keeps `cache_get` HIV without any `filter_pred$N` — do not introduce one.
   if (!hashIndexedViewHasFilterPredMember(ctx, hiv)) return hiv;
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
         if (!hashIndexedViewHasFilterPredMember(ctx, hiv)) return t;
         auto nhiv = extendHashIndexedViewWithPredMemberIfMissing(ctx, hiv, predMember);
         if (nhiv != hiv) return subop::LookupEntryRefType::get(ctx, nhiv);
      } else if (auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(ler.getState())) {
         if (!hashIndexedViewLikeHasFilterPredMember(ctx, mixed)) return t;
         auto nmixed = getMixedHashIndexedViewTypeForPredMember(ctx, mixed, predMember);
         if (nmixed != mixed) return subop::LookupEntryRefType::get(ctx, nmixed);
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
         auto expected =
            subop::ListType::get(ctx, subop::LookupEntryRefType::get(ctx, hiv));
         if (r.getColumn().type == expected) return;
         r.getColumn().type = expected;
         op.setRefAttr(r);
         return;
      } else if (auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(op.getState().getType())) {
         auto expected =
            subop::ListType::get(ctx, subop::LookupEntryRefType::get(ctx, mixed));
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
      auto stateTy = lookup.getState().getType();
      if (!mlir::isa<subop::HashIndexedViewType, subop::MixedHashIndexedViewType>(stateTy)) return;
      auto expected = subop::ListType::get(ctx, subop::LookupEntryRefType::get(ctx, mlir::cast<subop::LookupAbleState>(stateTy)));
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

static subop::GetExternalOp findUniqueGetExternalInTableRefStep(subop::ExecutionStepOp tableStep) {
   assert(isExternalTableRefStep(tableStep) && "expected external table_ref construction step");
   subop::GetExternalOp ge;
   for (mlir::Operation& op : tableStep.getSubOps().front().without_terminator()) {
      if (auto g = mlir::dyn_cast<subop::GetExternalOp>(&op)) {
         assert(!ge && "table_ref step must contain exactly one get_external");
         ge = g;
      }
   }
   assert(ge && "table_ref step must contain get_external");
   return ge;
}

static llvm::SmallVector<runtime::FilterDescription, 8>
decodeExternalFiltersForTableState(mlir::Value tableState) {
   for (;;) {
      if (auto ge = mlir::dyn_cast_or_null<subop::GetExternalOp>(tableState.getDefiningOp())) {
         auto ds = lingodb::utility::deserializeFromHexString<runtime::ExternalDatasourceProperty>(ge.getDescr());
         return llvm::SmallVector<runtime::FilterDescription, 8>(ds.filterDescriptions.begin(),
                                                                 ds.filterDescriptions.end());
      }
      if (auto tableStep = mlir::dyn_cast_or_null<ExecutionStepOp>(tableState.getDefiningOp())) {
         assert(isExternalTableRefStep(tableStep) &&
                "table state must come from external table_ref construction step");
         subop::GetExternalOp ge = findUniqueGetExternalInTableRefStep(tableStep);
         auto ds = lingodb::utility::deserializeFromHexString<runtime::ExternalDatasourceProperty>(ge.getDescr());
         return llvm::SmallVector<runtime::FilterDescription, 8>(ds.filterDescriptions.begin(),
                                                                 ds.filterDescriptions.end());
      }
      mlir::Value peeled = peelBlockArgsToEnclosingOperands(tableState);
      if (peeled == tableState) break;
      tableState = peeled;
   }
   return {};
}

llvm::SmallVector<runtime::FilterDescription, 8>
decodeFiltersFromTableScanInExecutionStep(ExecutionStepOp step) {
   mlir::Block& body = step.getSubOps().front();
   subop::ScanRefsOp scanOp;
   for (mlir::Operation& op : body.without_terminator()) {
      auto s = mlir::dyn_cast<subop::ScanRefsOp>(&op);
      if (!s) continue;
      if (mlir::isa<subop::TableType, subop::SharedTableType>(s.getState().getType())) {
         scanOp = s;
         break;
      }
   }
   if (!scanOp) return {};

   mlir::Value tableState = scanOp.getState();
   if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(tableState)) {
      if (ba.getOwner() == &body) {
         auto inputs = step.getInputs();
         assert(static_cast<unsigned>(ba.getArgNumber()) < inputs.size());
         tableState = inputs[ba.getArgNumber()];
      }
   }
   return decodeExternalFiltersForTableState(tableState);
}

static mlir::Value deriveDbPredicateTruthValue(mlir::OpBuilder& rb, mlir::Location loc, mlir::Value v) {
   return rb.create<lingodb::compiler::dialect::db::DeriveTruth>(loc, v);
}

// Emit one filter predicate (i1). Types aligned with runtime Restrictions::create (table scan).
static mlir::Value emitRuntimeFilterPredicateValue(mlir::OpBuilder& rb, mlir::Location loc,
                                                   subop::MapCreationHelper& helper,
                                                   tuples::ColumnRefAttr colRef,
                                                   const runtime::FilterDescription& f) {
   mlir::Value colV = helper.access(colRef, loc);
   auto emitArithIntConstant = [&](int64_t v) -> mlir::Value {
      auto itTy = mlir::dyn_cast<mlir::IntegerType>(colV.getType());
      assert(itTy && "runtime filter IR: int64 literal requires integer column type");
      return rb.create<mlir::arith::ConstantOp>(loc, rb.getIntegerAttr(itTy, v));
   };
   if (f.op == runtime::FilterOp::NOTNULL) {
      return rb.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
   }
   if (f.op == runtime::FilterOp::IN) {
      if (std::holds_alternative<std::vector<int64_t>>(f.values)) {
         const auto& vals = std::get<std::vector<int64_t>>(f.values);
         assert(!vals.empty() && "runtime filter IR: IN requires a non-empty value list");
         mlir::Value acc;
         for (int64_t v : vals) {
            mlir::Value c = emitArithIntConstant(v);
            mlir::Value eq = rb.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, colV, c);
            acc = acc ? rb.create<mlir::arith::OrIOp>(loc, acc, eq) : eq;
         }
         return acc;
      }
      if (std::holds_alternative<std::vector<std::string>>(f.values)) {
         const auto& vals = std::get<std::vector<std::string>>(f.values);
         assert(!vals.empty() && "runtime filter IR: IN requires a non-empty value list");
         [[maybe_unused]] mlir::Type colTy = getBaseType(colV.getType());
         assert((mlir::isa<lingodb::compiler::dialect::db::DateType>(colTy) ||
                 mlir::isa<lingodb::compiler::dialect::db::CharType>(colTy) ||
                 mlir::isa<lingodb::compiler::dialect::db::StringType>(colTy)) &&
                "runtime filter IR: IN string values require db.date, db.char, or db.string column");
         llvm::SmallVector<mlir::Value> candidates;
         candidates.reserve(vals.size());
         for (const std::string& s : vals) {
            candidates.push_back(
               rb.create<lingodb::compiler::dialect::db::ConstantOp>(loc, colV.getType(), rb.getStringAttr(s)));
         }
         auto oneOf = rb.create<lingodb::compiler::dialect::db::OneOfOp>(loc, colV, candidates);
         return deriveDbPredicateTruthValue(rb, loc, oneOf);
      }
      if (std::holds_alternative<std::vector<double>>(f.values)) {
         const auto& vals = std::get<std::vector<double>>(f.values);
         assert(!vals.empty() && "runtime filter IR: IN requires a non-empty value list");
         auto ft = mlir::dyn_cast<mlir::FloatType>(colV.getType());
         assert(ft && "runtime filter IR: IN double values require float column type");
         mlir::Value acc;
         for (double v : vals) {
            mlir::Value c = rb.create<mlir::arith::ConstantFloatOp>(loc, llvm::APFloat(v), ft);
            mlir::Value eq = rb.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, colV, c);
            acc = acc ? rb.create<mlir::arith::OrIOp>(loc, acc, eq) : eq;
         }
         return acc;
      }
      llvm_unreachable("runtime filter IR: IN filter has unsupported values variant");
   }
   if (std::holds_alternative<int64_t>(f.value)) {
      int64_t v = std::get<int64_t>(f.value);
      mlir::Value c = emitArithIntConstant(v);
      using P = mlir::arith::CmpIPredicate;
      P p = P::eq;
      switch (f.op) {
         case runtime::FilterOp::EQ: p = P::eq; break;
         case runtime::FilterOp::NEQ: p = P::ne; break;
         case runtime::FilterOp::LT: p = P::slt; break;
         case runtime::FilterOp::LTE: p = P::sle; break;
         case runtime::FilterOp::GT: p = P::sgt; break;
         case runtime::FilterOp::GTE: p = P::sge; break;
         default: llvm_unreachable("runtime filter IR: unsupported filter op");
      }
      return rb.create<mlir::arith::CmpIOp>(loc, p, colV, c);
   }
   if (std::holds_alternative<std::string>(f.value)) {
      auto s = std::get<std::string>(f.value);
      [[maybe_unused]] mlir::Type colTy = getBaseType(colV.getType());
      assert((mlir::isa<lingodb::compiler::dialect::db::DateType>(colTy) ||
              mlir::isa<lingodb::compiler::dialect::db::CharType>(colTy) ||
              mlir::isa<lingodb::compiler::dialect::db::StringType>(colTy)) &&
             "runtime filter IR: string literal requires db.date, db.char, or db.string column");
      auto rhs = rb.create<lingodb::compiler::dialect::db::ConstantOp>(loc, colV.getType(), rb.getStringAttr(s));
      using P = lingodb::compiler::dialect::db::DBCmpPredicate;
      P p;
      switch (f.op) {
         case runtime::FilterOp::EQ: p = P::eq; break;
         case runtime::FilterOp::NEQ: p = P::neq; break;
         case runtime::FilterOp::LT: p = P::lt; break;
         case runtime::FilterOp::LTE: p = P::lte; break;
         case runtime::FilterOp::GT: p = P::gt; break;
         case runtime::FilterOp::GTE: p = P::gte; break;
         default: llvm_unreachable("runtime filter IR: unsupported filter op");
      }
      auto cmp = rb.create<lingodb::compiler::dialect::db::CmpOp>(loc, p, colV, rhs);
      return deriveDbPredicateTruthValue(rb, loc, cmp);
   }
   if (std::holds_alternative<double>(f.value)) {
      double v = std::get<double>(f.value);
      auto ft = mlir::dyn_cast<mlir::FloatType>(colV.getType());
      assert(ft && "runtime filter IR: double literal requires float column type");
      mlir::Value c = rb.create<mlir::arith::ConstantFloatOp>(loc, llvm::APFloat(v), ft);
      using P = mlir::arith::CmpFPredicate;
      P p;
      switch (f.op) {
         case runtime::FilterOp::EQ: p = P::OEQ; break;
         case runtime::FilterOp::NEQ: p = P::ONE; break;
         case runtime::FilterOp::LT: p = P::OLT; break;
         case runtime::FilterOp::LTE: p = P::OLE; break;
         case runtime::FilterOp::GT: p = P::OGT; break;
         case runtime::FilterOp::GTE: p = P::OGE; break;
         default: llvm_unreachable("runtime filter IR: unsupported filter op");
      }
      return rb.create<mlir::arith::CmpFOp>(loc, p, colV, c);
   }
   llvm_unreachable("runtime filter IR: unsupported filter literal type (expected int64, double, or string)");
}

static std::string normalizeRuntimeFilterColumnName(llvm::StringRef name) {
   size_t dollar = name.find('$');
   if (dollar != llvm::StringRef::npos) name = name.take_front(dollar);
   std::string out = name.str();
   size_t pos = out.rfind("_u_");
   if (pos == std::string::npos || pos + 3 >= out.size()) return out;
   bool allDigits = true;
   for (char c : llvm::StringRef(out).drop_front(pos + 3)) {
      if (c < '0' || c > '9') {
         allDigits = false;
         break;
      }
   }
   if (allDigits) out.resize(pos);
   return out;
}

static tuples::ColumnRefAttr lookupRuntimeFilterColumn(
   const llvm::DenseMap<llvm::StringRef, tuples::ColumnRefAttr>& colByName,
   llvm::StringRef name) {
   if (auto it = colByName.find(name); it != colByName.end() && it->second) return it->second;
   std::string normalized = normalizeRuntimeFilterColumnName(name);
   for (auto& kv : colByName) {
      if (!kv.second) continue;
      if (normalizeRuntimeFilterColumnName(kv.first) == normalized) return kv.second;
   }
   return {};
}

// Convert runtime filter descriptions into MLIR subop.map + subop.filter.
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
         if (f.op == runtime::FilterOp::NOTNULL) continue;
         tuples::ColumnRefAttr col = lookupRuntimeFilterColumn(colByName, f.columnName);
         assert(col && "delay_filter: missing filter column in gathered columns");
         mlir::Value pred = emitRuntimeFilterPredicateValue(rb, loc, helper, col, f);
         acc = acc ? rb.create<lingodb::compiler::dialect::db::AndOp>(loc, mlir::ValueRange{acc, pred}) : pred;
      }
      if (!acc) acc = rb.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
      acc = deriveDbPredicateTruthValue(rb, loc, acc);
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
   std::string scopeSeed = "delay_filter_pred";
   if (auto slot = parseFilterPredMemberSlot(predLeafName)) scopeSeed = ("delay_filter_pred$" + llvm::Twine(*slot)).str();
   std::string scopeName = cm.getUniqueScope(scopeSeed);
   tuples::ColumnDefAttr predDef = cm.createDef(scopeName, "filter_pred");
   predDef.getColumn().type = b.getI1Type();
   // Keep predicate column def alive; column ref created where needed.

   subop::MapCreationHelper helper(b.getContext());
   helper.buildBlock(b, [&](mlir::OpBuilder& rb) {
      mlir::Value acc;
      for (auto& f : filters) {
         if (f.op == runtime::FilterOp::NOTNULL) continue;
         tuples::ColumnRefAttr col = lookupRuntimeFilterColumn(colByName, f.columnName);
         assert(col && "delay_filter_pred: missing filter column in gathered columns");
         mlir::Value pred = emitRuntimeFilterPredicateValue(rb, loc, helper, col, f);
         acc = acc ? rb.create<lingodb::compiler::dialect::db::AndOp>(loc, mlir::ValueRange{acc, pred}) : pred;
      }
      if (!acc) acc = rb.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
      acc = deriveDbPredicateTruthValue(rb, loc, acc);
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
         if (auto itG = findReuseMap(reuse->mergedFromShadowState, t.state);
             itG != reuse->mergedFromShadowState.end()) {
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
         if (!mlir::isa<subop::TableType, subop::SharedTableType>(r.getType())) continue;
         mlir::Value tableV = r;
         if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(r)) {
            if (ba.getOwner() == &ws.getSubOps().front()) {
               auto inputs = ws.getInputs();
               assert(static_cast<unsigned>(ba.getArgNumber()) < inputs.size());
               tableV = inputs[ba.getArgNumber()];
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

// Some join-style plans thread merged state through `nested_execution_group` and inner
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

static subop::ScanRefsOp findTableScanRefsInStep(subop::ExecutionStepOp step) {
   mlir::Block& body = step.getSubOps().front();
   for (mlir::Operation& op : body.without_terminator()) {
      auto s = mlir::dyn_cast<subop::ScanRefsOp>(&op);
      if (!s) continue;
      if (mlir::isa<subop::TableType, subop::SharedTableType>(s.getState().getType())) return s;
   }
   return {};
}

static subop::StateMembersAttr tableScanDataMembers(mlir::Type type) {
   if (auto tableTy = mlir::dyn_cast<subop::TableType>(type)) return tableTy.getMembers();
   if (auto sharedTy = mlir::dyn_cast<subop::SharedTableType>(type)) return sharedTy.getTableMembers();
   return {};
}

static subop::StateMembersAttr tableScanPredicateMembers(mlir::Type type) {
   if (auto sharedTy = mlir::dyn_cast<subop::SharedTableType>(type)) return sharedTy.getPredicateMembers();
   return subop::StateMembersAttr::get(type.getContext(), {});
}

static bool tableScanFiltered(mlir::Type type) {
   if (auto tableTy = mlir::dyn_cast<subop::TableType>(type)) return tableTy.getFiltered();
   if (auto sharedTy = mlir::dyn_cast<subop::SharedTableType>(type)) return sharedTy.getFiltered();
   llvm_unreachable("expected table-like scan state");
}

static subop::StateMembersAttr tableEntryDataMembers(mlir::Type type) {
   if (auto refTy = mlir::dyn_cast<subop::TableEntryRefType>(type)) return refTy.getTableColumns();
   if (auto refTy = mlir::dyn_cast<subop::SharedTableEntryRefType>(type)) return refTy.getTableColumns();
   return {};
}

static subop::StateMembersAttr tableEntryPredicateMembers(mlir::Type type) {
   if (auto refTy = mlir::dyn_cast<subop::SharedTableEntryRefType>(type)) return refTy.getPredicateColumns();
   return subop::StateMembersAttr::get(type.getContext(), {});
}

llvm::SmallVector<runtime::FilterDescription, 8>
restrictFiltersToTableScanInExecutionStep(ExecutionStepOp step,
                                          llvm::ArrayRef<runtime::FilterDescription> filters) {
   subop::ScanRefsOp scanOp = findTableScanRefsInStep(step);
   if (!scanOp || filters.empty()) return {};

   auto tableMembers = tableScanDataMembers(scanOp.getState().getType());
   auto& mm = step.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto stripSuffix = [](llvm::StringRef s) -> llvm::StringRef {
      size_t pos = s.find('$');
      if (pos == llvm::StringRef::npos) return s;
      return s.take_front(pos);
   };

   llvm::SmallVector<runtime::FilterDescription, 8> out;
   for (const auto& f : filters) {
      bool onTable = false;
      for (auto m : tableMembers.getMembers()) {
         if (stripSuffix(mm.getName(m)) == f.columnName) {
            onTable = true;
            break;
         }
      }
      if (onTable) out.push_back(f);
   }
   return out;
}

static subop::MaterializeOp findBufferMaterializeForPredMember(subop::ExecutionStepOp step,
                                                               llvm::StringRef predMemberName) {
   subop::MaterializeOp matOp;
   step.getOperation()->walk([&](subop::MaterializeOp m) {
      subop::BufferType bufTy = getInnerBufferTypeForMaterializeState(m.getState().getType());
      if (!bufTy) return mlir::WalkResult::advance();
      if (!valueMembersContainMemberNamed(step.getContext(), bufTy.getMembers(), predMemberName))
         return mlir::WalkResult::advance();
      matOp = m;
      return mlir::WalkResult::interrupt();
   });
   return matOp;
}

static void extendTableScanRefTypesForFilters(subop::ScanRefsOp scanOp,
                                              llvm::ArrayRef<runtime::FilterDescription> filters) {
   auto tableMembers = tableScanDataMembers(scanOp.getState().getType());
   auto stripSuffix = [](llvm::StringRef s) -> llvm::StringRef {
      size_t pos = s.find('$');
      if (pos == llvm::StringRef::npos) return s;
      return s.take_front(pos);
   };
   auto refDef = scanOp.getRef();
   llvm::SmallVector<subop::Member> cols =
      tableEntryDataMembers(refDef.getColumn().type).getMembers();
   llvm::DenseSet<subop::Member> have;
   for (auto m : cols) have.insert(m);
   auto& mm = scanOp->getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   for (auto& f : filters) {
      if (f.op == runtime::FilterOp::NOTNULL) continue;
      subop::Member mem;
      for (auto m : tableMembers.getMembers()) {
         if (stripSuffix(mm.getName(m)) == f.columnName) {
            mem = m;
            break;
         }
      }
      assert(mem && "write_pred: could not find table member for filter column");
      if (have.insert(mem).second) cols.push_back(mem);
   }
   auto newCols = subop::StateMembersAttr::get(scanOp.getContext(), cols);
   auto predCols = tableEntryPredicateMembers(refDef.getColumn().type);
   refDef.getColumn().type = predCols.getMembers().empty()
                                ? mlir::Type(subop::TableEntryRefType::get(scanOp.getContext(), newCols))
                                : mlir::Type(subop::SharedTableEntryRefType::get(scanOp.getContext(), newCols,
                                                                                 predCols));
   scanOp.setRefAttr(refDef);
}

static subop::GetExternalOp resolveGetExternalForScanRefs(subop::ScanRefsOp scanOp);

static subop::Member getOrCreatePredicateMember(mlir::MLIRContext* ctx, llvm::StringRef predMemberName) {
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   return mm.getOrCreateMemberDirect(predMemberName.str(), mlir::IntegerType::get(ctx, 1),
                                     /*allowTypeUpdate=*/false);
}

static void upgradeScanToSharedTableWithPredMember(subop::ScanRefsOp scanOp,
                                                   llvm::StringRef predMemberName) {
   auto* ctx = scanOp.getContext();
   subop::Member predMember = getOrCreatePredicateMember(ctx, predMemberName);
   llvm::SmallVector<subop::Member> tableMembers =
      tableScanDataMembers(scanOp.getState().getType()).getMembers();
   llvm::SmallVector<subop::Member> predMembers =
      tableScanPredicateMembers(scanOp.getState().getType()).getMembers();
   if (!llvm::is_contained(predMembers, predMember)) predMembers.push_back(predMember);
   auto newStateTy = subop::SharedTableType::get(ctx, subop::StateMembersAttr::get(ctx, tableMembers),
                                                 subop::StateMembersAttr::get(ctx, predMembers),
                                                 tableScanFiltered(scanOp.getState().getType()));

   subop::GetExternalOp ge = resolveGetExternalForScanRefs(scanOp);
   ge.getResult().setType(newStateTy);
   scanOp.getState().setType(newStateTy);
   if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(scanOp.getState())) {
      mlir::Operation* parent = arg.getOwner()->getParentOp();
      if (auto step = mlir::dyn_cast_or_null<ExecutionStepOp>(parent)) {
         assert(arg.getArgNumber() < step.getNumOperands());
         step.getOperand(arg.getArgNumber()).setType(newStateTy);
      }
   }

   auto refDef = scanOp.getRef();
   llvm::SmallVector<subop::Member> refTableMembers =
      tableEntryDataMembers(refDef.getColumn().type).getMembers();
   if (refTableMembers.empty()) refTableMembers = tableMembers;
   llvm::SmallVector<subop::Member> refPredMembers =
      tableEntryPredicateMembers(refDef.getColumn().type).getMembers();
   if (!llvm::is_contained(refPredMembers, predMember)) refPredMembers.push_back(predMember);
   refDef.getColumn().type = subop::SharedTableEntryRefType::get(
      ctx, subop::StateMembersAttr::get(ctx, refTableMembers),
      subop::StateMembersAttr::get(ctx, refPredMembers));
   scanOp.setRefAttr(refDef);

   synchronizeExecutionStepPortTypes(scanOp->getParentOfType<mlir::ModuleOp>(), nullptr);
}

static void extendTableScanRefTypeWithPredMember(subop::ScanRefsOp scanOp,
                                                 llvm::StringRef predMemberName) {
   upgradeScanToSharedTableWithPredMember(scanOp, predMemberName);
}

static subop::GetExternalOp resolveGetExternalForScanRefs(subop::ScanRefsOp scanOp) {
   mlir::Value tableState = scanOp.getState();
   for (;;) {
      if (auto ge = mlir::dyn_cast_or_null<subop::GetExternalOp>(tableState.getDefiningOp())) return ge;
      if (auto tableStep = mlir::dyn_cast_or_null<ExecutionStepOp>(tableState.getDefiningOp())) {
         return findUniqueGetExternalInTableRefStep(tableStep);
      }
      mlir::Value peeled = peelBlockArgsToEnclosingOperands(tableState);
      assert(peeled != tableState && "shared_scan: table state must resolve to get_external");
      tableState = peeled;
   }
}

static void addSharedPredicateClauseForScan(subop::ScanRefsOp scanOp, llvm::StringRef predMemberName,
                                            llvm::ArrayRef<runtime::FilterDescription> filters) {
   upgradeScanToSharedTableWithPredMember(scanOp, predMemberName);
   auto predMembers = tableScanPredicateMembers(scanOp.getState().getType());
   subop::Member predMember = getOrCreatePredicateMember(scanOp.getContext(), predMemberName);
   std::optional<unsigned> localSlot;
   for (auto indexedMember : llvm::enumerate(predMembers.getMembers())) {
      if (indexedMember.value() != predMember) continue;
      localSlot = indexedMember.index();
      break;
   }
   assert(localSlot && "shared_scan: predicate member must be part of shared table type");

   subop::GetExternalOp ge = resolveGetExternalForScanRefs(scanOp);
   auto ds = lingodb::utility::deserializeFromHexString<runtime::ExternalDatasourceProperty>(ge.getDescr());
   ds.filterDescriptions.clear();
   ds.orFilterClauses.clear();
   if (ds.sharedPredicateClauses.size() < predMembers.getMembers().size())
      ds.sharedPredicateClauses.resize(predMembers.getMembers().size());
   std::vector<runtime::FilterDescription> filterVec(filters.begin(), filters.end());
   ds.sharedPredicateClauses[*localSlot] = std::move(filterVec);
   ge.setDescrAttr(mlir::StringAttr::get(ge.getContext(), lingodb::utility::serializeToHexString(ds)));
}

static std::pair<mlir::Value, tuples::ColumnDefAttr> gatherSharedPredicateColumnAfterScan(
   subop::ScanRefsOp scanOp, llvm::StringRef predMemberName) {
   extendTableScanRefTypeWithPredMember(scanOp, predMemberName);
   auto* ctx = scanOp.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   subop::Member predMember =
      mm.getOrCreateMemberDirect(predMemberName.str(), mlir::IntegerType::get(ctx, 1), /*allowTypeUpdate=*/false);
   tuples::ColumnDefAttr predDef = cm.createDef(cm.getUniqueScope(predMemberName), "filter_pred");
   predDef.getColumn().type = mlir::IntegerType::get(ctx, 1);
   mlir::OpBuilder gb(scanOp);
   gb.setInsertionPointAfter(scanOp);
   auto gather = gb.create<subop::GatherOp>(
      scanOp.getLoc(), scanOp.getRes(), cm.createRef(&scanOp.getRef().getColumn()),
      subop::ColumnDefMemberMappingAttr::get(ctx, {{predMember, predDef}}));
   return {gather.getRes(), predDef};
}

static llvm::DenseMap<llvm::StringRef, tuples::ColumnRefAttr>
buildFilterColByNameFromGather(subop::GatherOp gatherOp) {
   auto* ctx = gatherOp.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto stripSuffix = [](llvm::StringRef s) -> llvm::StringRef {
      size_t pos = s.find('$');
      if (pos == llvm::StringRef::npos) return s;
      return s.take_front(pos);
   };
   llvm::DenseMap<llvm::StringRef, tuples::ColumnRefAttr> colByName;
   for (auto& p : gatherOp.getMapping().getMapping()) {
      llvm::StringRef key = stripSuffix(mm.getName(p.first));
      colByName[key] = cm.createRef(&p.second.getColumn());
   }
   return colByName;
}

static subop::GatherOp insertFilterColumnGatherRightAfterScan(
   subop::ScanRefsOp scanOp, llvm::ArrayRef<runtime::FilterDescription> filters) {
   auto* ctx = scanOp.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto tableMembers = tableScanDataMembers(scanOp.getState().getType());
   auto stripSuffix = [](llvm::StringRef s) -> llvm::StringRef {
      size_t pos = s.find('$');
      if (pos == llvm::StringRef::npos) return s;
      return s.take_front(pos);
   };

   llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> mappingPairs;
   llvm::StringMap<bool> seenCols;
   for (auto& f : filters) {
      if (f.op == runtime::FilterOp::NOTNULL) continue;
      if (seenCols.contains(f.columnName)) continue;
      seenCols[f.columnName] = true;
      subop::Member mem;
      for (auto m : tableMembers.getMembers()) {
         if (stripSuffix(mm.getName(m)) == f.columnName) {
            mem = m;
            break;
         }
      }
      assert(mem && "write_pred: could not find table member for filter column");
      std::string scope = cm.getUniqueScope("reuse_write_pred_col");
      tuples::ColumnDefAttr def = cm.createDef(scope, f.columnName);
      def.getColumn().type = cloneFilterPredTypeToContext(mm.getType(mem), ctx);
      mappingPairs.push_back({mem, def});
   }
   if (mappingPairs.empty()) return {};

   tuples::ColumnRefAttr scanEntryRef = cm.createRef(&scanOp.getRef().getColumn());
   mlir::OpBuilder gb(scanOp);
   gb.setInsertionPointAfter(scanOp);
   return gb.create<subop::GatherOp>(scanOp.getLoc(), scanOp.getRes(), scanEntryRef,
                                   subop::ColumnDefMemberMappingAttr::get(ctx, mappingPairs));
}

static std::pair<mlir::Value, tuples::ColumnDefAttr> materializeConstantTruePredColumnOnStream(
   mlir::OpBuilder& pb, mlir::Location loc, mlir::Value stream, llvm::StringRef predMemberName) {
   auto* ctx = pb.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   std::string scopeSeed = "reuse_write_pred_const";
   if (auto slot = parseFilterPredMemberSlot(predMemberName))
      scopeSeed = ("reuse_write_pred_const$" + llvm::Twine(*slot)).str();
   std::string sc = cm.getUniqueScope(scopeSeed);
   tuples::ColumnDefAttr def = cm.createDef(sc, "filter_pred");
   def.getColumn().type = mlir::IntegerType::get(ctx, 1);
   subop::MapCreationHelper helper(ctx);
   helper.buildBlock(pb, [&](mlir::OpBuilder& rb) {
      mlir::Value t = rb.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
      rb.create<tuples::ReturnOp>(loc, mlir::ValueRange{t});
   });
   auto mapOp = pb.create<subop::MapOp>(loc, tuples::TupleStreamType::get(ctx), stream, pb.getArrayAttr({def}),
                                        helper.getColRefs());
   mapOp.getFn().push_back(helper.getMapBlock());
   return {mapOp.getResult(), def};
}

static void rewireStreamUsesAfterAnchorInBlock(mlir::Value anchorStream, mlir::Value newStream,
                                               mlir::Operation* anchorOp,
                                               llvm::ArrayRef<mlir::Operation*> excludeOps) {
   anchorStream.replaceUsesWithIf(newStream, [&](mlir::OpOperand& ou) {
      mlir::Operation* owner = ou.getOwner();
      for (mlir::Operation* ex : excludeOps) {
         if (owner == ex) return false;
      }
      if (owner->getBlock() != anchorOp->getBlock()) return false;
      return anchorOp->isBeforeInBlock(owner);
   });
}

std::pair<mlir::Value, tuples::ColumnRefAttr> materializeRuntimeFiltersAsPredicateColumnAfterScanRefs(
   subop::ScanRefsOp scanOp, llvm::ArrayRef<runtime::FilterDescription> filters,
   llvm::StringRef predLeafName, bool rewireDownstreamUses) {
   assert(scanOp && "predicate column insertion requires scan_refs");
   llvm::SmallVector<runtime::FilterDescription, 8> restricted =
      restrictFiltersToTableScanInExecutionStep(scanOp->getParentOfType<ExecutionStepOp>(), filters);
   filters = restricted;
   assert(!filters.empty() && "predicate column insertion requires at least one runtime filter");

   extendTableScanRefTypesForFilters(scanOp, filters);

   llvm::SmallVector<mlir::Operation*> excludeOps;
   excludeOps.push_back(scanOp.getOperation());
   mlir::Value predStream;
   tuples::ColumnDefAttr predDef;
   bool needsColumnGather =
      llvm::any_of(filters, [](const runtime::FilterDescription& f) { return f.op != runtime::FilterOp::NOTNULL; });
   if (!needsColumnGather) {
      mlir::OpBuilder b(scanOp);
      b.setInsertionPointAfter(scanOp);
      std::tie(predStream, predDef) =
         materializeConstantTruePredColumnOnStream(b, scanOp.getLoc(), scanOp.getRes(), predLeafName);
   } else {
      subop::GatherOp filterGather = insertFilterColumnGatherRightAfterScan(scanOp, filters);
      assert(filterGather && "predicate column insertion requires gathered filter columns");
      excludeOps.push_back(filterGather.getOperation());
      auto colByName = buildFilterColByNameFromGather(filterGather);
      mlir::OpBuilder b(filterGather);
      b.setInsertionPointAfter(filterGather);
      std::tie(predStream, predDef) =
         materializeRuntimeFiltersAsPredicateColumn(b, filterGather.getLoc(), filterGather.getRes(),
                                                    colByName, filters, predLeafName);
   }
   if (mlir::Operation* predMap = predStream.getDefiningOp()) excludeOps.push_back(predMap);
   if (rewireDownstreamUses) rewireStreamUsesAfterAnchorInBlock(scanOp.getRes(), predStream, scanOp, excludeOps);

   auto& cm = scanOp.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   return {predStream, cm.createRef(&predDef.getColumn())};
}

std::pair<mlir::Value, tuples::ColumnRefAttr> materializeRuntimeFilterClausesAsIdColumnAfterScanRefs(
   subop::ScanRefsOp scanOp, llvm::ArrayRef<RuntimeFilterIdClause> clauses,
   unsigned defaultId, llvm::StringRef idLeafName, bool rewireDownstreamUses) {
   assert(scanOp && "id column insertion requires scan_refs");
   assert(!clauses.empty() && "id column insertion requires at least one clause");

   llvm::SmallVector<RuntimeFilterIdClause, 8> restrictedClauses;
   llvm::SmallVector<runtime::FilterDescription, 32> allFilters;
   for (const RuntimeFilterIdClause& clause : clauses) {
      RuntimeFilterIdClause restricted;
      restricted.id = clause.id;
      restricted.filters = restrictFiltersToTableScanInExecutionStep(
         scanOp->getParentOfType<ExecutionStepOp>(), clause.filters);
      assert(!restricted.filters.empty() && "mixed aggregate id clause must have simple table filters");
      allFilters.append(restricted.filters.begin(), restricted.filters.end());
      restrictedClauses.push_back(std::move(restricted));
   }
   extendTableScanRefTypesForFilters(scanOp, allFilters);

   llvm::SmallVector<mlir::Operation*> excludeOps;
   excludeOps.push_back(scanOp.getOperation());
   mlir::Value stream = scanOp.getRes();
   llvm::DenseMap<llvm::StringRef, tuples::ColumnRefAttr> colByName;
   bool needsColumnGather =
      llvm::any_of(allFilters, [](const runtime::FilterDescription& f) { return f.op != runtime::FilterOp::NOTNULL; });
   if (needsColumnGather) {
      subop::GatherOp gather = insertFilterColumnGatherRightAfterScan(scanOp, allFilters);
      assert(gather && "id column insertion requires gathered filter columns");
      excludeOps.push_back(gather.getOperation());
      colByName = buildFilterColByNameFromGather(gather);
      stream = gather.getRes();
   }

   auto* ctx = scanOp.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   tuples::ColumnDefAttr idDef = cm.createDef(cm.getUniqueScope("agg_reuse_query_id"), idLeafName);
   idDef.getColumn().type = mlir::IntegerType::get(ctx, 64);
   tuples::ColumnRefAttr idRef = cm.createRef(&idDef.getColumn());

   mlir::OpBuilder b(scanOp);
   mlir::Operation* anchor = stream.getDefiningOp();
   assert(anchor && "id column stream must be op-defined");
   b.setInsertionPointAfter(anchor);
   subop::MapCreationHelper helper(ctx);
   mlir::Location loc = scanOp.getLoc();
   helper.buildBlock(b, [&](mlir::OpBuilder& rb) {
      mlir::Type idTy = idDef.getColumn().type;
      mlir::Value id =
         rb.create<lingodb::compiler::dialect::db::ConstantOp>(loc, idTy, rb.getI64IntegerAttr(defaultId));
      for (auto it = restrictedClauses.rbegin(); it != restrictedClauses.rend(); ++it) {
         mlir::Value pred;
         for (const runtime::FilterDescription& f : it->filters) {
            if (f.op == runtime::FilterOp::NOTNULL) continue;
            tuples::ColumnRefAttr col = lookupRuntimeFilterColumn(colByName, f.columnName);
            assert(col && "id column insertion missing filter column");
            mlir::Value one = emitRuntimeFilterPredicateValue(rb, loc, helper, col, f);
            pred = pred ? rb.create<lingodb::compiler::dialect::db::AndOp>(loc, mlir::ValueRange{pred, one}) : one;
         }
         if (!pred) pred = rb.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
         pred = deriveDbPredicateTruthValue(rb, loc, pred);
         mlir::Value clauseId =
            rb.create<lingodb::compiler::dialect::db::ConstantOp>(loc, idTy, rb.getI64IntegerAttr(it->id));
         id = rb.create<mlir::arith::SelectOp>(loc, pred, clauseId, id);
      }
      rb.create<tuples::ReturnOp>(loc, mlir::ValueRange{id});
   });
   auto idMap = b.create<subop::MapOp>(loc, tuples::TupleStreamType::get(ctx), stream,
                                       b.getArrayAttr({idDef}), helper.getColRefs());
   idMap.getFn().push_back(helper.getMapBlock());
   excludeOps.push_back(idMap.getOperation());

   if (rewireDownstreamUses)
      rewireStreamUsesAfterAnchorInBlock(scanOp.getRes(), idMap.getResult(), scanOp, excludeOps);
   return {idMap.getResult(), idRef};
}

static void appendPredMemberToBufferMaterialize(subop::MaterializeOp matOp, subop::Member predMember,
                                                tuples::ColumnDefAttr predDef) {
   auto* ctx = matOp.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   tuples::ColumnRefAttr predColRef = cm.createRef(&predDef.getColumn());
   llvm::SmallVector<subop::RefMappingPairT> pairs;
   for (auto& pr : matOp.getMapping().getMapping()) {
      if (pr.first != predMember) pairs.push_back(pr);
   }
   pairs.push_back({predMember, predColRef});
   matOp.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, pairs));
}

void insertWriteSidePredIntoHashMapConstructionStep(ExecutionStepOp step,
                                                            llvm::ArrayRef<runtime::FilterDescription> filters) {
   llvm::SmallVector<runtime::FilterDescription, 8> restricted =
      restrictFiltersToTableScanInExecutionStep(step, filters);
   filters = restricted;
   if (filters.empty()) return;
   mlir::Block& body = step.getSubOps().front();

   subop::ScanRefsOp scanOp = findTableScanRefsInStep(step);
   assert(scanOp && "write_pred: expected a scan_refs over table");

   mlir::Value predStream;
   tuples::ColumnDefAttr predDef;
   llvm::SmallVector<mlir::Operation*> excludeOps;
   excludeOps.push_back(scanOp.getOperation());

   bool needsColumnGather =
      llvm::any_of(filters, [](const runtime::FilterDescription& f) { return f.op != runtime::FilterOp::NOTNULL; });
   if (!needsColumnGather) {
      mlir::OpBuilder pb(scanOp);
      pb.setInsertionPointAfter(scanOp);
      std::tie(predStream, predDef) =
         materializeConstantTruePredColumnOnStream(pb, scanOp.getLoc(), scanOp.getRes(), "filter_pred");
   } else {
      addSharedPredicateClauseForScan(scanOp, "filter_pred$0", filters);
      std::tie(predStream, predDef) = gatherSharedPredicateColumnAfterScan(scanOp, "filter_pred$0");
      if (mlir::Operation* predGather = predStream.getDefiningOp()) excludeOps.push_back(predGather);
   }
   auto& cm = scanOp.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   if (mlir::Operation* predMap = predStream.getDefiningOp()) excludeOps.push_back(predMap);
   rewireStreamUsesAfterAnchorInBlock(scanOp.getRes(), predStream, scanOp, excludeOps);

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
std::optional<subop::Member> findFilterPredMemberOnHashIndexedView(subop::HashIndexedViewType hiv) {
   if (!hiv) return std::nullopt;
   auto* ctx = hiv.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   for (subop::Member m : hiv.getValueMembers().getMembers()) {
      llvm::StringRef name = mm.getName(m);
      if (name == "filter_pred" || parseFilterPredMemberSlot(name)) return m;
   }
   return std::nullopt;
}

void materializeConstantTruePredMemberOnBufferMaterialize(subop::MaterializeOp matOp,
                                                          llvm::StringRef predMemberName,
                                                          bool updateStreamOperand) {
   auto* ctx = matOp.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   subop::BufferType bufTy = getInnerBufferTypeForMaterializeState(matOp.getState().getType());
   assert(bufTy);
   subop::Member predMember;
   for (auto m : bufTy.getMembers().getMembers()) {
      if (mm.getName(m) == predMemberName) {
         predMember = m;
         break;
      }
   }
   assert(predMember);

   for (auto& pr : matOp.getMapping().getMapping()) {
      if (pr.first == predMember) return;
   }

   mlir::Value feedForPred = matOp.getStream();

   mlir::OpBuilder pb(matOp);
   mlir::Location loc = matOp.getLoc();
   if (mlir::Operation* defOp = feedForPred.getDefiningOp()) {
      pb.setInsertionPointAfter(defOp);
   } else {
      pb.setInsertionPoint(matOp);
   }
   std::string scopeSeed = "reuse_write_pred_const";
   if (auto slot = parseFilterPredMemberSlot(predMemberName)) scopeSeed = ("reuse_write_pred_const$" + llvm::Twine(*slot)).str();
   std::string sc = cm.getUniqueScope(scopeSeed);
   tuples::ColumnDefAttr def = cm.createDef(sc, "t");
   def.getColumn().type = mlir::IntegerType::get(ctx, 1);
   subop::MapCreationHelper helper(ctx);
   helper.buildBlock(pb, [&](mlir::OpBuilder& rb) {
      mlir::Value t = rb.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
      rb.create<tuples::ReturnOp>(loc, mlir::ValueRange{t});
   });
   auto mapOp = pb.create<subop::MapOp>(loc, tuples::TupleStreamType::get(ctx), feedForPred, pb.getArrayAttr({def}),
                                        helper.getColRefs());
   mapOp.getFn().push_back(helper.getMapBlock());
   if (updateStreamOperand) matOp->setOperand(0, mapOp.getResult());

   tuples::ColumnRefAttr predColRef = cm.createRef(&def.getColumn());
   llvm::SmallVector<subop::RefMappingPairT> pairs;
   for (auto& pr : matOp.getMapping().getMapping()) {
      if (pr.first != predMember) pairs.push_back(pr);
   }
   pairs.push_back({predMember, predColRef});
   matOp.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, pairs));
}

void insertWriteSidePredIntoBufferConstructionStepForPredMember(
   ExecutionStepOp step, llvm::ArrayRef<runtime::FilterDescription> filters, llvm::StringRef predMemberName,
   bool allowSharedScanPredicate) {
   llvm::SmallVector<runtime::FilterDescription, 8> restricted =
      restrictFiltersToTableScanInExecutionStep(step, filters);
   filters = restricted;
   subop::ScanRefsOp scanOp = findTableScanRefsInStep(step);
   assert(scanOp && "write_pred_buf: expected a scan_refs over table");
   subop::MaterializeOp matOp = findBufferMaterializeForPredMember(step, predMemberName);
   assert(matOp && "write_pred_buf: expected materialize into join buffer with filter_pred member");

   auto& mm = step.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   subop::Member predMember;
   {
      subop::BufferType bufTy = getInnerBufferTypeForMaterializeState(matOp.getState().getType());
      assert(bufTy);
      for (auto m : bufTy.getMembers().getMembers()) {
         if (mm.getName(m) == predMemberName) {
            predMember = m;
            break;
         }
      }
      assert(predMember && "write_pred_buf: filter_pred member missing on buffer type");
   }

   mlir::Value predStream;
   tuples::ColumnDefAttr predDef;
   tuples::ColumnRefAttr predRef;
   llvm::SmallVector<mlir::Operation*> excludeOps;
   excludeOps.push_back(scanOp.getOperation());
   bool rewiredByHelper = false;

   if (filters.empty()) {
      mlir::OpBuilder pb(scanOp);
      pb.setInsertionPointAfter(scanOp);
      std::tie(predStream, predDef) =
         materializeConstantTruePredColumnOnStream(pb, scanOp.getLoc(), scanOp.getRes(), predMemberName);
   } else {
      bool needsColumnGather =
         llvm::any_of(filters, [](const runtime::FilterDescription& f) { return f.op != runtime::FilterOp::NOTNULL; });
      if (!needsColumnGather) {
         mlir::OpBuilder pb(scanOp);
         pb.setInsertionPointAfter(scanOp);
         std::tie(predStream, predDef) =
            materializeConstantTruePredColumnOnStream(pb, scanOp.getLoc(), scanOp.getRes(), predMemberName);
      } else {
         if (allowSharedScanPredicate) {
            addSharedPredicateClauseForScan(scanOp, predMemberName, filters);
            std::tie(predStream, predDef) = gatherSharedPredicateColumnAfterScan(scanOp, predMemberName);
            if (mlir::Operation* predGather = predStream.getDefiningOp()) excludeOps.push_back(predGather);
         } else {
            std::tie(predStream, predRef) =
               materializeRuntimeFiltersAsPredicateColumnAfterScanRefs(scanOp, filters, predMemberName,
                                                                       /*rewireDownstreamUses=*/true);
            rewiredByHelper = true;
         }
      }
   }
   if (mlir::Operation* predMap = predStream.getDefiningOp()) excludeOps.push_back(predMap);

   if (!rewiredByHelper) rewireStreamUsesAfterAnchorInBlock(scanOp.getRes(), predStream, scanOp, excludeOps);
   if (predDef) {
      appendPredMemberToBufferMaterialize(matOp, predMember, predDef);
   } else {
      assert(predRef && "write_pred_buf: predicate column must be materialized");
      llvm::SmallVector<subop::RefMappingPairT> pairs;
      for (auto& pr : matOp.getMapping().getMapping()) {
         if (pr.first != predMember) pairs.push_back(pr);
      }
      pairs.push_back({predMember, predRef});
      matOp.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(matOp.getContext(), pairs));
   }
}

void insertWriteSidePredIntoBufferConstructionStep(ExecutionStepOp step,
                                                   llvm::ArrayRef<runtime::FilterDescription> filters) {
   insertWriteSidePredIntoBufferConstructionStepForPredMember(step, filters, "filter_pred$0");
}

/// `rewriteHashmapTypesInModule` can extend join buffers with `filter_pred$0` before the writer maps it.
/// Recompute predicates from the table scan's external descriptor (same filters as table-scan runtime)
/// and materialize them into `filter_pred$0`. Steps with no pushdown filters get constant-true.
void ensureJoinBufferFilterPredMaterializeMappings(mlir::ModuleOp module) {
   auto* ctx = module.getContext();
   auto* subDialect = ctx->getLoadedDialect<subop::SubOperatorDialect>();
   auto* tupleDialect = ctx->getLoadedDialect<tuples::TupleStreamDialect>();
   if (!subDialect || !tupleDialect) return;
   auto& mm = subDialect->getMemberManager();
   auto& cm = tupleDialect->getColumnManager();
   ModuleReuseInfo reuse = collectModuleReuseInfo(module);

   auto isPlaceholderPredColumn = [&](tuples::ColumnRefAttr ref) -> bool {
      auto [scope, leaf] = cm.getName(&ref.getColumn());
      (void)leaf;
      llvm::StringRef sc(scope);
      return sc.contains("jp_default_pred") || sc.contains("jp_no_table_filter");
   };

   auto materializeNeedsPredMapping = [&](subop::MaterializeOp m) -> bool {
      subop::BufferType bufTy = getInnerBufferTypeForMaterializeState(m.getState().getType());
      if (!bufTy) return false;
      subop::Member predM;
      for (auto mem : bufTy.getMembers().getMembers()) {
         if (mm.getName(mem) == "filter_pred$0") {
            predM = mem;
            break;
         }
      }
      if (!predM) return false;
      for (auto& pr : m.getMapping().getMapping()) {
         if (pr.first != predM) continue;
         if (isPlaceholderPredColumn(pr.second)) return true;
         return false;
      }
      return true;
   };

   auto stripPlaceholderPredFromMaterialize = [&](subop::MaterializeOp m, subop::Member predM) {
      if (auto mapOp = mlir::dyn_cast<subop::MapOp>(m.getStream().getDefiningOp())) {
         bool placeholder = false;
         for (auto colAttr : mapOp.getComputedCols()) {
            auto def = mlir::cast<tuples::ColumnDefAttr>(colAttr);
            auto [scope, leaf] = cm.getName(&def.getColumn());
            (void)leaf;
            if (llvm::StringRef(scope).contains("jp_default_pred") ||
                llvm::StringRef(scope).contains("jp_no_table_filter")) {
               placeholder = true;
               break;
            }
         }
         if (placeholder) {
            m->setOperand(0, mapOp.getStream());
            mapOp.erase();
         }
      }
      llvm::SmallVector<subop::RefMappingPairT> pairs;
      for (auto pr : m.getMapping().getMapping()) {
         if (pr.first != predM) pairs.push_back(pr);
      }
      m.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, pairs));
   };

   llvm::SmallVector<subop::MaterializeOp, 16> todo;
   module.walk([&](subop::MaterializeOp m) {
      if (materializeNeedsPredMapping(m)) todo.push_back(m);
      return mlir::WalkResult::advance();
   });

   llvm::DenseSet<mlir::Operation*> stepsWithTableFilters;
   for (subop::MaterializeOp m : todo) {
      auto step = m->getParentOfType<ExecutionStepOp>();
      if (!step) continue;

      llvm::SmallVector<runtime::FilterDescription, 8> filters =
         decodeFiltersFromTableScanInExecutionStep(step);
      if (filters.empty()) filters = decodeFiltersForStateFromWriterSteps(m.getState(), reuse);
      if (filters.empty()) continue;

      if (!stepsWithTableFilters.insert(step.getOperation()).second) {
         // Upgrade placeholder pred maps from an earlier ensureJoinBuffer pass.
         subop::Member predM;
         if (subop::BufferType bufTy = getInnerBufferTypeForMaterializeState(m.getState().getType())) {
            for (auto mem : bufTy.getMembers().getMembers()) {
               if (mm.getName(mem) == "filter_pred$0") {
                  predM = mem;
                  break;
               }
            }
         }
         if (predM) stripPlaceholderPredFromMaterialize(m, predM);
         insertWriteSidePredIntoBufferConstructionStep(step, filters);
         continue;
      }

      subop::Member predM;
      if (subop::BufferType bufTy = getInnerBufferTypeForMaterializeState(m.getState().getType())) {
         for (auto mem : bufTy.getMembers().getMembers()) {
            if (mm.getName(mem) == "filter_pred$0") {
               predM = mem;
               break;
            }
         }
      }
      if (predM) stripPlaceholderPredFromMaterialize(m, predM);
      insertWriteSidePredIntoBufferConstructionStep(step, filters);
   }

   for (subop::MaterializeOp m : todo) {
      if (!materializeNeedsPredMapping(m)) continue;

      mlir::OpBuilder pb(m);
      mlir::Location loc = m.getLoc();
      std::string sc = cm.getUniqueScope("jp_no_table_filter");
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

static bool probeScanStreamAlreadyPredFiltered(mlir::Value scanStream) {
   for (mlir::OpOperand& use : scanStream.getUses()) {
      auto gatherOp = mlir::dyn_cast<subop::GatherOp>(use.getOwner());
      if (!gatherOp || gatherOp.getStream() != scanStream) continue;
      for (mlir::OpOperand& gUse : gatherOp.getRes().getUses()) {
         auto filterOp = mlir::dyn_cast<subop::FilterOp>(gUse.getOwner());
         if (!filterOp || filterOp.getStream() != gatherOp.getRes()) continue;
         if (filterOp.getFilterSemantic() == subop::FilterSemantic::all_true) return true;
      }
   }
   return false;
}

void insertProbePredFilterImmediatelyAfterScanProducer(mlir::Operation* anchorOp, mlir::Value scanStream,
                                                       tuples::ColumnRefAttr entryRef, subop::Member predMember) {
   if (probeScanStreamAlreadyPredFiltered(scanStream)) return;

   auto* ctx = anchorOp->getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();

   std::string scopeSeed = "probe_pred_filter";
   if (auto slot = parseFilterPredMemberSlot(mm.getName(predMember)))
      scopeSeed = ("probe_pred_filter$" + llvm::Twine(*slot)).str();
   tuples::ColumnDefAttr predDef = cm.createDef(cm.getUniqueScope(scopeSeed), "filter_pred");
   predDef.getColumn().type = mm.getType(predMember);

   mlir::OpBuilder gb(anchorOp);
   gb.setInsertionPointAfter(anchorOp);
   auto gatherOp = gb.create<subop::GatherOp>(anchorOp->getLoc(), scanStream, entryRef,
                                              subop::ColumnDefMemberMappingAttr::get(ctx, {{predMember, predDef}}));
   tuples::ColumnRefAttr predRef = cm.createRef(&predDef.getColumn());

   mlir::OpBuilder fb(gatherOp);
   fb.setInsertionPointAfter(gatherOp);
   auto filterOp = fb.create<subop::FilterOp>(gatherOp.getLoc(), gatherOp.getRes(), subop::FilterSemantic::all_true,
                                              fb.getArrayAttr({predRef}));

   scanStream.replaceUsesWithIf(filterOp.getRes(), [&](mlir::OpOperand& ou) {
      mlir::Operation* owner = ou.getOwner();
      if (owner == gatherOp.getOperation() || owner == filterOp.getOperation()) return false;
      if (owner->getBlock() != anchorOp->getBlock()) return false;
      return anchorOp->isBeforeInBlock(owner);
   });
}

// For `scan_refs` over join `hashmap`: filter by stored `filter_pred$N` right after the scan.
void insertScanRefsPredFilter(ExecutionStepOp step, subop::Member predMember) {
   mlir::Block& body = step.getSubOps().front();
   subop::ScanRefsOp scanOp;
   for (mlir::Operation& op : body.without_terminator()) {
      auto s = mlir::dyn_cast<subop::ScanRefsOp>(&op);
      if (!s) continue;
      if (mlir::isa<subop::HashMapType>(s.getState().getType())) {
         scanOp = s;
         break;
      }
   }
   if (!scanOp) return;
   auto* ctx = step.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   tuples::ColumnRefAttr entryRef = cm.createRef(&scanOp.getRef().getColumn());
   insertProbePredFilterImmediatelyAfterScanProducer(scanOp.getOperation(), scanOp.getRes(), entryRef, predMember);
}

static llvm::SmallVector<subop::ScanListOp, 4> findHivScanListsInStep(
   subop::ExecutionStepOp step, subop::Member predMember, const llvm::DenseSet<void*>* closureFilter) {
   llvm::SmallVector<subop::ScanListOp, 4> out;
   auto* ctx = step.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   mlir::Block& body = step.getSubOps().front();
   for (mlir::Operation& op : body.without_terminator()) {
      auto scanListOp = mlir::dyn_cast<subop::ScanListOp>(&op);
      if (!scanListOp) continue;
      if (closureFilter && !opaqueClosureContains(*closureFilter, scanListOp.getList())) continue;
      auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(scanListOp.getElem().getColumn().type);
      if (!ler) continue;
      auto hiv = mlir::dyn_cast<subop::HashIndexedViewType>(ler.getState());
      auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(ler.getState());
      if (!hiv && !mixed) continue;
      subop::StateMembersAttr valueMembers = hiv ? hiv.getValueMembers() : mixed.getValueMembers();
      if (!valueMembersContainMemberNamed(ctx, valueMembers, mm.getName(predMember))) continue;
      out.push_back(scanListOp);
   }
   return out;
}

static void retagHivScanListToMixed(subop::ScanListOp scanList, subop::Member predMember) {
   auto* ctx = scanList.getContext();
   auto listTy = mlir::dyn_cast<subop::ListType>(scanList.getList().getType());
   if (!listTy) return;
   auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(listTy.getT());
   if (!ler) return;
   auto mixedTy = getMixedHashIndexedViewTypeForPredMember(ctx, ler.getState(), predMember);
   if (!mixedTy) return;
   auto mixedLer = subop::LookupEntryRefType::get(ctx, mixedTy);
   scanList.getList().setType(subop::ListType::get(ctx, mixedLer));
   auto elem = scanList.getElem();
   elem.getColumn().type = mixedLer;
   scanList.setElemAttr(elem);
}

static mlir::Type retagHashIndexedViewCarrierTypeToMixed(mlir::MLIRContext* ctx, mlir::Type type,
                                                         subop::Member predMember) {
   if (!type) return type;
   if (auto listTy = mlir::dyn_cast<subop::ListType>(type)) {
      mlir::Type inner = retagHashIndexedViewCarrierTypeToMixed(ctx, listTy.getT(), predMember);
      if (inner != listTy.getT()) return subop::ListType::get(ctx, mlir::cast<subop::StateEntryReference>(inner));
      return type;
   }
   if (auto optTy = mlir::dyn_cast<subop::OptionalType>(type)) {
      mlir::Type inner = retagHashIndexedViewCarrierTypeToMixed(ctx, optTy.getT(), predMember);
      if (inner != optTy.getT()) return subop::OptionalType::get(ctx, mlir::cast<subop::StateEntryReference>(inner));
      return type;
   }
   if (auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(type)) {
      auto mixedTy = getMixedHashIndexedViewTypeForPredMember(ctx, ler.getState(), predMember);
      if (mixedTy && mixedTy != ler.getState()) return subop::LookupEntryRefType::get(ctx, mixedTy);
      return type;
   }
   if (auto mixedTy = getMixedHashIndexedViewTypeForPredMember(ctx, type, predMember)) return mixedTy;
   return type;
}

static void retagHivCarrierValuesToMixed(mlir::ModuleOp module, subop::Member predMember,
                                         const llvm::DenseSet<void*>* closureFilter) {
   auto* ctx = module.getContext();
   llvm::SmallVector<std::pair<mlir::Value, mlir::Type>, 16> updates;
   auto maybeRetag = [&](mlir::Value v) {
      if (!v) return;
      if (closureFilter && !opaqueClosureContains(*closureFilter, v)) return;
      mlir::Type newTy = retagHashIndexedViewCarrierTypeToMixed(ctx, v.getType(), predMember);
      if (!newTy || v.getType() == newTy) return;
      updates.push_back({v, newTy});
   };
   module.walk([&](mlir::Operation* op) {
      for (mlir::Value result : op->getResults()) maybeRetag(result);
      for (mlir::Region& region : op->getRegions()) {
         for (mlir::Block& block : region) {
            for (mlir::BlockArgument arg : block.getArguments()) maybeRetag(arg);
         }
      }
   });
   for (auto& [value, type] : updates) value.setType(type);
}

/// After `scan_list` on cached `hash_indexed_view`, use MixedHIV lookup/scan semantics to filter
/// by stored `filter_pred$N` inside the hash table traversal.
void insertHashIndexedViewGatherPredFilters(ExecutionStepOp step, subop::Member predMember,
                                            const llvm::DenseSet<void*>* closureFilter) {
   llvm::SmallVector<subop::ScanListOp, 4> scanListOps = findHivScanListsInStep(step, predMember, closureFilter);
   if (scanListOps.empty()) return;
   for (subop::ScanListOp scanListOp : scanListOps) retagHivScanListToMixed(scanListOp, predMember);

   mlir::ModuleOp module = step->getParentOfType<mlir::ModuleOp>();
   if (!module) return;
   retagHivCarrierValuesToMixed(module, predMember, closureFilter);
   synchronizeExecutionStepPortTypes(module, nullptr);
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
      if (auto itG = findReuseMap(reuse.mergedFromShadowState, t.state);
          itG != reuse.mergedFromShadowState.end()) {
         filterSeed = itG->second;
      }
      llvm::SmallVector<runtime::FilterDescription, 8> decoded =
         decodeFiltersForStateFromWriterSteps(filterSeed, reuse, &rwByStepOp);
      if (auto itTL = findReuseMap(reuse.mergedFromShadowState, filterSeed);
          itTL != reuse.mergedFromShadowState.end()) {
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
               if (mlir::isa<subop::TableType, subop::SharedTableType>(s.getState().getType())) scanOp = s;
            }
         }
         assert(scanOp && "delay_filter: missing filter columns but could not find scan_refs before gather");
         auto tableMembers = tableScanDataMembers(scanOp.getState().getType());
         assert(tableMembers && "delay_filter: scan_refs state must be table-like");

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
            for (auto m : tableMembers.getMembers()) {
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
