#include "lingodb/compiler/Dialect/SubOperator/Transforms/CrossQueryStateReuse.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseRewriteCommon.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseStateClosure.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseFilterPredInsert.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseJoinSuperset.h"

#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsAttributes.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateExtraction.h"
#include "lingodb/compiler/Dialect/SubOperator/Utils.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

#include <cassert>
#include <limits>
#include <optional>

namespace lingodb::compiler::dialect::subop {
namespace {

static uint64_t splitMaterializeOutputCacheKey(uint64_t groupKey, unsigned slot) {
   uint64_t h = static_cast<uint64_t>(llvm::hash_value(llvm::StringRef("split_materialize_output")));
   h = llvm::hash_combine(h, groupKey);
   h = llvm::hash_combine(h, static_cast<uint64_t>(slot));
   return h;
}

static std::optional<unsigned> parseEarlyFilterPredMemberSlot(llvm::StringRef memberName) {
   if (!memberName.consume_front("filter_pred$")) return std::nullopt;
   unsigned slot = 0;
   if (memberName.getAsInteger(10, slot)) return std::nullopt;
   return slot;
}

static void inheritConsumerSlotsFromSingleMixedDep(
   const llvm::DenseMap<uint64_t, llvm::SmallVector<uint64_t, 4>>& inheritedDepsByCacheKey,
   llvm::DenseMap<uint64_t, llvm::DenseMap<unsigned, unsigned>>& consumerSlotByCacheKeyAndQuery) {
   for (const auto& [targetKey, deps] : inheritedDepsByCacheKey) {
      if (deps.empty()) continue;
      llvm::DenseMap<unsigned, unsigned> inheritedSlots;
      bool foundMappedDep = false;
      for (uint64_t depKey : deps) {
         auto itSlots = consumerSlotByCacheKeyAndQuery.find(depKey);
         if (itSlots == consumerSlotByCacheKeyAndQuery.end()) continue;
         if (!foundMappedDep) {
            inheritedSlots = itSlots->second;
            foundMappedDep = true;
            continue;
         }
         assert(inheritedSlots == itSlots->second &&
                "inherited mixed pred target must not inherit conflicting upstream consumer slots");
      }
      if (!foundMappedDep) continue;
      consumerSlotByCacheKeyAndQuery[targetKey] = std::move(inheritedSlots);
   }
}

static void inheritConsumerSlotsFromGroupDeps(
   llvm::ArrayRef<CrossQueryStateMatchGroup> groups,
   llvm::DenseMap<uint64_t, llvm::DenseMap<unsigned, unsigned>>& consumerSlotByCacheKeyAndQuery) {
   for (const CrossQueryStateMatchGroup& group : groups) {
      if (group.cacheDeps.empty()) continue;
      llvm::DenseMap<unsigned, unsigned> inheritedSlots;
      bool foundMappedDep = false;
      for (uint64_t depKey : group.cacheDeps) {
         auto itSlots = consumerSlotByCacheKeyAndQuery.find(depKey);
         if (itSlots == consumerSlotByCacheKeyAndQuery.end()) continue;
         if (!foundMappedDep) {
            inheritedSlots = itSlots->second;
            foundMappedDep = true;
            continue;
         }
         assert(inheritedSlots == itSlots->second &&
                "dependent reuse group must not inherit conflicting upstream consumer slots");
      }
      if (!foundMappedDep) continue;
      consumerSlotByCacheKeyAndQuery[group.cacheKey] = std::move(inheritedSlots);
   }
}

static void recordConsumerMixedCacheGetSlots(
   mlir::ModuleOp module, unsigned queryIdx,
   llvm::DenseMap<uint64_t, llvm::DenseMap<unsigned, unsigned>>& consumerSlotByCacheKeyAndQuery) {
   module.walk([&](subop::CacheGetOp get) {
      auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(get.getResult().getType());
      if (!mixed) return;
      auto slot = parseEarlyFilterPredMemberSlot(mixed.getFilterPredMemberName().getValue());
      assert(slot && "mixed cache_get must select a filter_pred$N member");
      consumerSlotByCacheKeyAndQuery[static_cast<uint64_t>(get.getKey())][queryIdx] = *slot;
   });
}

static mlir::BlockArgument ensureExecutionStepInput(ExecutionStepOp step, mlir::Value value) {
   for (unsigned i = 0; i < step.getNumOperands(); ++i) {
      if (step.getOperand(i) == value) {
         assert(i < step.getSubOps().front().getNumArguments());
         return step.getSubOps().front().getArgument(i);
      }
   }
   step.getInputsMutable().append(value);
   bool isThreadLocal = mlir::isa<ThreadLocalType>(value.getType());
   llvm::SmallVector<mlir::Attribute, 8> threadLocal(step.getIsThreadLocal().begin(),
                                                     step.getIsThreadLocal().end());
   threadLocal.push_back(mlir::BoolAttr::get(step.getContext(), isThreadLocal));
   step.setIsThreadLocalAttr(mlir::ArrayAttr::get(step.getContext(), threadLocal));
   mlir::Type argType = value.getType();
   if (auto tl = mlir::dyn_cast<ThreadLocalType>(argType)) argType = tl.getWrapped();
   return step.getSubOps().front().addArgument(argType, step.getLoc());
}

static ExecutionStepOp topLevelExecutionStepFor(ExecutionStepOp step) {
   ExecutionStepOp cur = step;
   for (;;) {
      auto parentStep = cur->getParentOfType<ExecutionStepOp>();
      if (!parentStep) return cur;
      cur = parentStep;
   }
}

static mlir::BlockArgument ensureNestedExecutionGroupInput(NestedExecutionGroupOp group, mlir::Value value) {
   for (unsigned i = 0; i < group.getNumOperands(); ++i) {
      if (group.getOperand(i) == value) {
         assert(i < group.getSubOps().front().getNumArguments());
         return group.getSubOps().front().getArgument(i);
      }
   }
   group.getInputsMutable().append(value);
   return group.getSubOps().front().addArgument(value.getType(), group.getLoc());
}

static mlir::Value threadStateToNestedMaterializeStep(ExecutionStepOp materializeStep,
                                                      mlir::Value topLevelState) {
   ExecutionStepOp topStep = topLevelExecutionStepFor(materializeStep);
   mlir::Value current = ensureExecutionStepInput(topStep, topLevelState);
   if (topStep == materializeStep) return current;

   llvm::SmallVector<NestedExecutionGroupOp, 4> nestedGroups;
   for (mlir::Operation* op = materializeStep->getParentOp(); op && op != topStep.getOperation();
        op = op->getParentOp()) {
      if (auto neg = mlir::dyn_cast<NestedExecutionGroupOp>(op)) nestedGroups.push_back(neg);
   }
   for (NestedExecutionGroupOp neg : llvm::reverse(nestedGroups)) {
      current = ensureNestedExecutionGroupInput(neg, current);
   }
   return ensureExecutionStepInput(materializeStep, current);
}

static subop::MaterializeOp findUniqueMaterializeWritingState(ExecutionStepOp step, mlir::Value state) {
   state = canonicalizeStateValueForReuse(state);
   subop::MaterializeOp found;
   step.walk([&](subop::MaterializeOp mat) {
      if (canonicalizeStateValueForReuse(mat.getState()) != state) return;
      assert(!found && "split-materialize reuse expects one materialize for the output state");
      found = mat;
   });
   if (!found) llvm_unreachable("split-materialize reuse requires a materialize writer");
   return found;
}

static llvm::SmallVector<subop::Member> stateMembersForType(mlir::Type type) {
   if (auto tl = mlir::dyn_cast<ThreadLocalType>(type)) type = tl.getWrapped();
   if (auto state = mlir::dyn_cast<subop::State>(type)) {
      llvm::SmallVector<subop::Member> members;
      members.append(state.getMembers().getMembers().begin(), state.getMembers().getMembers().end());
      return members;
   }
   return {};
}

static subop::ColumnRefMemberMappingAttr remapMaterializeMappingMembersByOrdinal(
   mlir::MLIRContext* ctx, subop::ColumnRefMemberMappingAttr sourceMapping, mlir::Type targetStateType) {
   llvm::SmallVector<subop::Member> targetMembers = stateMembersForType(targetStateType);
   assert(!targetMembers.empty() && "split-materialize target must have members");
   llvm::SmallVector<subop::RefMappingPairT> pairs;
   unsigned ordinal = 0;
   for (auto& [member, colRef] : sourceMapping.getMapping()) {
      (void)member;
      assert(ordinal < targetMembers.size() && "split-materialize mapping larger than target state");
      pairs.push_back({targetMembers[ordinal++], colRef});
   }
   llvm::SmallVector<subop::RefMappingPairT> attrPairs;
   attrPairs.append(pairs.begin(), pairs.end());
   return subop::ColumnRefMemberMappingAttr::get(ctx, attrPairs);
}

static void clearGetExternalFiltersForState(mlir::Value tableState) {
   auto tableStep = mlir::dyn_cast_or_null<ExecutionStepOp>(tableState.getDefiningOp());
   if (!tableStep) return;
   subop::GetExternalOp ge;
   tableStep.walk([&](subop::GetExternalOp g) {
      assert(!ge && "split-materialize table step must contain one get_external");
      ge = g;
   });
   if (!ge) return;
   auto ds = lingodb::utility::deserializeFromHexString<runtime::ExternalDatasourceProperty>(ge.getDescr());
   ds.filterDescriptions.clear();
   ds.orFilterClauses.clear();
   ge.setDescrAttr(mlir::StringAttr::get(ge.getContext(), lingodb::utility::serializeToHexString(ds)));
}

static subop::ScanRefsOp findFirstScanRefsInStep(ExecutionStepOp step) {
   subop::ScanRefsOp found;
   step.walk([&](subop::ScanRefsOp scan) {
      if (!found) found = scan;
   });
   assert(found && "split-materialize branch filtering requires a scan_refs source");
   return found;
}

static llvm::DenseMap<llvm::StringRef, tuples::ColumnRefAttr>
materializeStreamColumnsByFilterName(ExecutionStepOp step, subop::MaterializeOp mat) {
   llvm::DenseMap<llvm::StringRef, tuples::ColumnRefAttr> out;
   auto& cm = mat.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = mat.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto addName = [&](llvm::StringRef name, tuples::ColumnRefAttr ref) {
      out[name] = ref;
      size_t dollar = name.find('$');
      if (dollar != llvm::StringRef::npos) out[name.take_front(dollar)] = ref;
   };
   for (auto& [member, colRef] : mat.getMapping().getMapping()) {
      (void)member;
      auto [scope, leaf] = cm.getName(&colRef.getColumn());
      (void)scope;
      addName(leaf, colRef);
   }
   step.walk([&](subop::GatherOp gather) {
      for (auto& [member, colDef] : gather.getMapping().getMapping()) {
         tuples::ColumnRefAttr ref = cm.createRef(&colDef.getColumn());
         addName(mm.getName(member), ref);
         auto [scope, leaf] = cm.getName(&colDef.getColumn());
         (void)scope;
         addName(leaf, ref);
      }
   });
   return out;
}

static void assertRuntimeFiltersAvailableOnStream(
   const llvm::DenseMap<llvm::StringRef, tuples::ColumnRefAttr>& colByName,
   llvm::ArrayRef<runtime::FilterDescription> filters) {
   for (const runtime::FilterDescription& f : filters) {
      if (f.op == runtime::FilterOp::NOTNULL) continue;
      if (!colByName.contains(f.columnName)) {
         llvm_unreachable("split-materialize branch filter column must be present on the shared stream");
      }
   }
}

static bool streamValueHasOnlyUseByWithinStep(mlir::Value v, mlir::Operation* expectedUser,
                                              ExecutionStepOp step) {
   mlir::Operation* onlyUser = nullptr;
   auto isWithinStep = [&](mlir::Operation* op) {
      for (mlir::Operation* cur = op; cur; cur = cur->getParentOp())
         if (cur == step.getOperation()) return true;
      return false;
   };
   for (mlir::OpOperand& use : v.getUses()) {
      mlir::Operation* user = use.getOwner();
      if (!isWithinStep(user)) continue;
      if (onlyUser && onlyUser != user) return false;
      onlyUser = user;
   }
   return onlyUser == expectedUser;
}

struct SplitResidualFilter {
   subop::MapOp predMap;
   subop::FilterOp filter;
   mlir::Value inputStream;
};

static bool filterConditionsComeFromMap(subop::FilterOp filter, subop::MapOp map) {
   llvm::DenseSet<const void*> computedCols;
   for (auto attr : map.getComputedCols()) {
      auto def = mlir::cast<tuples::ColumnDefAttr>(attr);
      computedCols.insert(&def.getColumn());
   }
   for (auto attr : filter.getConditions()) {
      auto ref = mlir::cast<tuples::ColumnRefAttr>(attr);
      if (!computedCols.contains(&ref.getColumn())) return false;
   }
   return true;
}

static unsigned computedColumnIndex(subop::MapOp map, tuples::ColumnRefAttr ref);

static void collectSplitResidualBlockArgs(mlir::Value value, mlir::Block* block,
                                          llvm::DenseSet<unsigned>& out) {
   if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      if (arg.getOwner() == block) out.insert(arg.getArgNumber());
      return;
   }
   mlir::Operation* def = value.getDefiningOp();
   if (!def) return;
   for (mlir::Value operand : def->getOperands()) collectSplitResidualBlockArgs(operand, block, out);
}

static llvm::DenseSet<unsigned> splitResidualPredicateInputIndices(subop::MapOp map,
                                                                   subop::FilterOp filter) {
   llvm::DenseSet<unsigned> used;
   mlir::Block& block = map.getFn().front();
   auto ret = mlir::cast<tuples::ReturnOp>(block.getTerminator());
   for (auto attr : filter.getConditions()) {
      auto cond = mlir::cast<tuples::ColumnRefAttr>(attr);
      unsigned idx = computedColumnIndex(map, cond);
      collectSplitResidualBlockArgs(ret.getOperand(idx), &block, used);
   }
   return used;
}

static llvm::StringRef stripSplitNameSuffix(llvm::StringRef name) {
   size_t dollar = name.find('$');
   return dollar == llvm::StringRef::npos ? name : name.take_front(dollar);
}

static mlir::Type splitResidualSourceStateTypeFromStream(ExecutionStepOp step,
                                                         mlir::Operation* user,
                                                         mlir::Value stream) {
   for (;;) {
      mlir::Operation* def = stream.getDefiningOp();
      if (!def) return {};
      if (!streamValueHasOnlyUseByWithinStep(stream, user, step)) return {};
      if (auto scan = mlir::dyn_cast<subop::ScanRefsOp>(def)) return scan.getState().getType();
      if (auto scan = mlir::dyn_cast<subop::ScanListOp>(def)) {
         auto listTy = mlir::dyn_cast<subop::ListType>(scan.getList().getType());
         if (!listTy) return {};
         auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(listTy.getT());
         return ler ? ler.getState() : mlir::Type{};
      }
      if (auto gather = mlir::dyn_cast<subop::GatherOp>(def)) {
         user = def;
         stream = gather.getStream();
         continue;
      }
      if (auto map = mlir::dyn_cast<subop::MapOp>(def)) {
         user = def;
         stream = map.getStream();
         continue;
      }
      if (auto rename = mlir::dyn_cast<subop::RenamingOp>(def)) {
         user = def;
         stream = rename.getStream();
         continue;
      }
      return {};
   }
}

static bool splitResidualSourceHasMember(mlir::Type sourceType,
                                         subop::MemberManager& mm,
                                         llvm::StringRef leaf) {
   subop::StateMembersAttr members;
   if (auto tableTy = mlir::dyn_cast<subop::TableType>(sourceType)) {
      members = tableTy.getMembers();
   } else if (auto hiv = mlir::dyn_cast<subop::HashIndexedViewType>(sourceType)) {
      members = hiv.getValueMembers();
   } else if (auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(sourceType)) {
      members = mixed.getValueMembers();
   } else {
      return false;
   }
   llvm::StringRef baseLeaf = stripSplitNameSuffix(leaf);
   for (subop::Member member : members.getMembers()) {
      llvm::StringRef name = mm.getName(member);
      if (name == leaf || stripSplitNameSuffix(name) == baseLeaf) return true;
   }
   return false;
}

static bool splitResidualPredicateUsesOnlySourceMembers(ExecutionStepOp step,
                                                        subop::MapOp map,
                                                        subop::FilterOp filter) {
   mlir::Type sourceType = splitResidualSourceStateTypeFromStream(step, map.getOperation(), map.getStream());
   if (!sourceType) return false;
   auto& cm = map.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = map.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   llvm::DenseSet<unsigned> usedInputs = splitResidualPredicateInputIndices(map, filter);
   for (unsigned idx : usedInputs) {
      auto ref = mlir::cast<tuples::ColumnRefAttr>(map.getInputCols()[idx]);
      auto [scope, leaf] = cm.getName(&ref.getColumn());
      (void)scope;
      if (!splitResidualSourceHasMember(sourceType, mm, leaf)) return false;
   }
   return true;
}

static std::optional<SplitResidualFilter> findResidualFilterBeforeMaterialize(ExecutionStepOp step,
                                                                              subop::MaterializeOp mat) {
   mlir::Operation* user = mat.getOperation();
   mlir::Value stream = mat.getStream();
   for (;;) {
      mlir::Operation* def = stream.getDefiningOp();
      if (!def) return std::nullopt;
      if (!streamValueHasOnlyUseByWithinStep(stream, user, step)) return std::nullopt;
      if (auto filter = mlir::dyn_cast<subop::FilterOp>(def)) {
         auto map = mlir::dyn_cast_or_null<subop::MapOp>(filter.getStream().getDefiningOp());
         if (map && streamValueHasOnlyUseByWithinStep(map.getResult(), filter.getOperation(), step) &&
             filterConditionsComeFromMap(filter, map) &&
             splitResidualPredicateUsesOnlySourceMembers(step, map, filter)) {
            return SplitResidualFilter{map, filter, map.getStream()};
         }
         user = def;
         stream = filter.getStream();
         continue;
      }
      if (auto gather = mlir::dyn_cast<subop::GatherOp>(def)) {
         user = def;
         stream = gather.getStream();
         continue;
      }
      if (auto map = mlir::dyn_cast<subop::MapOp>(def)) {
         user = def;
         stream = map.getStream();
         continue;
      }
      if (auto rename = mlir::dyn_cast<subop::RenamingOp>(def)) {
         user = def;
         stream = rename.getStream();
         continue;
      }
      return std::nullopt;
   }
}

static llvm::StringRef baseNameForColumnRef(tuples::ColumnRefAttr ref) {
   auto& cm = ref.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto [scope, leaf] = cm.getName(&ref.getColumn());
   (void)scope;
   llvm::StringRef name = leaf;
   size_t dollar = name.find('$');
   return dollar == llvm::StringRef::npos ? name : name.take_front(dollar);
}

static std::string normalizeSplitColumnName(llvm::StringRef name) {
   size_t dollar = name.find('$');
   if (dollar != llvm::StringRef::npos) name = name.take_front(dollar);
   std::string out = name.str();
   size_t pos = out.rfind("_u_");
   if (pos != std::string::npos && pos + 3 < out.size()) {
      bool allDigits = true;
      for (char c : llvm::StringRef(out).drop_front(pos + 3)) {
         if (c < '0' || c > '9') {
            allDigits = false;
            break;
         }
      }
      if (allDigits) out.resize(pos);
   }
   return out;
}

static tuples::ColumnRefAttr lookupSplitColumnByName(
   const llvm::DenseMap<llvm::StringRef, tuples::ColumnRefAttr>& colByName,
   llvm::StringRef name) {
   if (auto it = colByName.find(name); it != colByName.end()) return it->second;
   std::string norm = normalizeSplitColumnName(name);
   for (auto& kv : colByName) {
      if (normalizeSplitColumnName(kv.first) == norm) return kv.second;
   }
   return {};
}

static llvm::SmallVector<tuples::ColumnDefAttr, 4>
makeFreshComputedColumnDefsLike(mlir::MLIRContext* ctx, subop::MapOp map) {
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   llvm::SmallVector<tuples::ColumnDefAttr, 4> out;
   for (auto attr : map.getComputedCols()) {
      auto oldDef = mlir::cast<tuples::ColumnDefAttr>(attr);
      auto [scope, leaf] = cm.getName(&oldDef.getColumn());
      (void)scope;
      tuples::ColumnDefAttr def = cm.createDef(cm.getUniqueScope("split_residual"), leaf);
      def.getColumn().type = oldDef.getColumn().type;
      out.push_back(def);
   }
   return out;
}

static unsigned computedColumnIndex(subop::MapOp map, tuples::ColumnRefAttr ref) {
   for (unsigned i = 0; i < map.getComputedCols().size(); ++i) {
      auto def = mlir::cast<tuples::ColumnDefAttr>(map.getComputedCols()[i]);
      if (&def.getColumn() == &ref.getColumn()) return i;
   }
   llvm_unreachable("split-materialize residual condition must come from predicate map");
}

static mlir::Value cloneResidualFilterBranch(mlir::OpBuilder& b,
                                             mlir::Location loc,
                                             mlir::Value stream,
                                             const llvm::DenseMap<llvm::StringRef, tuples::ColumnRefAttr>& colByName,
                                             SplitResidualFilter residual) {
   llvm::SmallVector<mlir::Attribute, 8> inputRefs;
   inputRefs.reserve(residual.predMap.getInputCols().size());
   for (auto attr : residual.predMap.getInputCols()) {
      auto oldRef = mlir::cast<tuples::ColumnRefAttr>(attr);
      llvm::StringRef name = baseNameForColumnRef(oldRef);
      tuples::ColumnRefAttr ref = lookupSplitColumnByName(colByName, name);
      assert(ref && "split-materialize residual input column must exist on shared stream");
      inputRefs.push_back(ref);
   }

   llvm::SmallVector<tuples::ColumnDefAttr, 4> defs =
      makeFreshComputedColumnDefsLike(b.getContext(), residual.predMap);
   llvm::SmallVector<mlir::Attribute, 4> defAttrs;
   defAttrs.append(defs.begin(), defs.end());

   mlir::IRMapping mapping;
   auto* clonedOp = residual.predMap.getOperation()->clone(mapping);
   b.getInsertionBlock()->getOperations().insert(b.getInsertionPoint(), clonedOp);
   auto clonedMap = mlir::cast<subop::MapOp>(clonedOp);
   clonedMap->setOperand(0, stream);
   clonedMap.setInputColsAttr(b.getArrayAttr(inputRefs));
   clonedMap.setComputedColsAttr(b.getArrayAttr(defAttrs));
   b.setInsertionPointAfter(clonedMap);

   auto& cm = b.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   llvm::SmallVector<mlir::Attribute, 4> condRefs;
   condRefs.reserve(residual.filter.getConditions().size());
   for (auto attr : residual.filter.getConditions()) {
      auto oldRef = mlir::cast<tuples::ColumnRefAttr>(attr);
      unsigned idx = computedColumnIndex(residual.predMap, oldRef);
      assert(idx < defs.size());
      condRefs.push_back(cm.createRef(&defs[idx].getColumn()));
   }
   auto filter = b.create<subop::FilterOp>(loc, clonedMap.getResult(),
                                           residual.filter.getFilterSemantic(),
                                           b.getArrayAttr(condRefs));
   return filter.getRes();
}

static mlir::Value filterSplitBranchByMixedPredMember(mlir::OpBuilder& b,
                                                      mlir::Location loc,
                                                      ExecutionStepOp step,
                                                      mlir::Value stream,
                                                      llvm::StringRef predMemberName) {
   if (predMemberName.empty()) return stream;
   auto* ctx = step.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

   subop::ScanListOp scanList;
   subop::Member predMember;
   step.walk([&](subop::ScanListOp scan) {
      if (scanList) return;
      auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(scan.getElem().getColumn().type);
      if (!ler) return;
      auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(ler.getState());
      if (!mixed) return;
      for (subop::Member member : mixed.getValueMembers().getMembers()) {
         if (mm.getName(member) == predMemberName) {
            scanList = scan;
            predMember = member;
            return;
         }
      }
   });
   if (!scanList || !predMember) return stream;

   tuples::ColumnDefAttr predDef = cm.createDef(cm.getUniqueScope("split_branch_pred"), "filter_pred");
   predDef.getColumn().type = mlir::IntegerType::get(ctx, 1);
   auto gather = b.create<subop::GatherOp>(
      loc, stream.getType(), stream, cm.createRef(&scanList.getElem().getColumn()),
      subop::ColumnDefMemberMappingAttr::get(ctx, {{predMember, predDef}}));
   tuples::ColumnRefAttr predRef = cm.createRef(&predDef.getColumn());
   auto filter = b.create<subop::FilterOp>(loc, gather.getRes(), subop::FilterSemantic::all_true,
                                           b.getArrayAttr({predRef}));
   return filter.getRes();
}

static std::string residualPredicateValueFingerprint(mlir::Value v,
                                                     llvm::DenseMap<mlir::Value, std::string>& memo) {
   if (auto it = memo.find(v); it != memo.end()) return it->second;
   std::string out;
   llvm::raw_string_ostream os(out);
   if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(v)) {
      os << "arg#" << arg.getArgNumber();
      os.flush();
      memo[v] = out;
      return out;
   }
   mlir::Operation* op = v.getDefiningOp();
   assert(op && "residual predicate value must be defined");
   os << op->getName().getStringRef();
   llvm::SmallVector<mlir::NamedAttribute, 8> attrs(op->getAttrs().begin(), op->getAttrs().end());
   llvm::sort(attrs, [](mlir::NamedAttribute a, mlir::NamedAttribute b) {
      return a.getName().strref() < b.getName().strref();
   });
   for (mlir::NamedAttribute attr : attrs) {
      os << "|attr:" << attr.getName().strref() << '=';
      attr.getValue().print(os);
   }
   for (mlir::Value operand : op->getOperands()) {
      os << "|opnd:" << residualPredicateValueFingerprint(operand, memo);
   }
   os.flush();
   memo[v] = out;
   return out;
}

static std::string residualFilterSemanticFingerprint(SplitResidualFilter residual) {
   std::string out;
   llvm::raw_string_ostream os(out);
   os << "inputs:";
   for (auto attr : residual.predMap.getInputCols()) {
      auto ref = mlir::cast<tuples::ColumnRefAttr>(attr);
      os << baseNameForColumnRef(ref) << ',';
   }
   os << "|semantic:";
   residual.filter.getFilterSemanticAttr().print(os);
   auto ret = mlir::cast<tuples::ReturnOp>(residual.predMap.getFn().front().getTerminator());
   llvm::DenseMap<mlir::Value, std::string> memo;
   os << "|conditions:";
   for (auto attr : residual.filter.getConditions()) {
      auto cond = mlir::cast<tuples::ColumnRefAttr>(attr);
      unsigned idx = computedColumnIndex(residual.predMap, cond);
      os << residualPredicateValueFingerprint(ret.getOperand(idx), memo) << ';';
   }
   os.flush();
   return out;
}

static mlir::Value streamInputOfLinearSuffixOp(mlir::Operation* op) {
   if (auto gather = mlir::dyn_cast<subop::GatherOp>(op)) return gather.getStream();
   if (auto map = mlir::dyn_cast<subop::MapOp>(op)) return map.getStream();
   if (auto filter = mlir::dyn_cast<subop::FilterOp>(op)) return filter.getStream();
   if (auto rename = mlir::dyn_cast<subop::RenamingOp>(op)) return rename.getStream();
   llvm_unreachable("split-materialize residual suffix contains unsupported stream op");
}

static mlir::Value streamResultOfLinearSuffixOp(mlir::Operation* op) {
   if (auto gather = mlir::dyn_cast<subop::GatherOp>(op)) return gather.getResult();
   if (auto map = mlir::dyn_cast<subop::MapOp>(op)) return map.getResult();
   if (auto filter = mlir::dyn_cast<subop::FilterOp>(op)) return filter.getRes();
   if (auto rename = mlir::dyn_cast<subop::RenamingOp>(op)) return rename.getResult();
   llvm_unreachable("split-materialize residual suffix contains unsupported stream op");
}

static mlir::Value cloneLinearStreamSuffixBefore(mlir::OpBuilder& b,
                                                 mlir::Value suffixStart,
                                                 mlir::Value suffixEnd,
                                                 mlir::Value newStart) {
   if (suffixStart == suffixEnd) return newStart;

   llvm::SmallVector<mlir::Operation*, 8> reverseOps;
   mlir::Value stream = suffixEnd;
   for (;;) {
      mlir::Operation* def = stream.getDefiningOp();
      assert(def && "split-materialize residual suffix must be local SSA");
      reverseOps.push_back(def);
      mlir::Value input = streamInputOfLinearSuffixOp(def);
      if (input == suffixStart) break;
      stream = input;
   }

   mlir::Value current = newStart;
   mlir::IRMapping mapping;
   for (mlir::Operation* op : llvm::reverse(reverseOps)) {
      auto* cloned = op->clone(mapping);
      b.getInsertionBlock()->getOperations().insert(b.getInsertionPoint(), cloned);
      cloned->setOperand(0, current);
      b.setInsertionPointAfter(cloned);
      current = streamResultOfLinearSuffixOp(cloned);
   }
   return current;
}

static ExecutionStepOp findUniqueMaterializeStepWritingState(mlir::ModuleOp module, mlir::Value state) {
   state = canonicalizeStateValueForReuse(state);
   ExecutionStepOp found;
   module.walk([&](ExecutionStepOp step) {
      bool writes = false;
      step.walk([&](subop::MaterializeOp mat) {
         if (canonicalizeStateValueForReuse(mat.getState()) == state) writes = true;
      });
      if (!writes) return;
      assert(!found && "split-materialize reuse expects one materialize step for the output state");
      found = step;
   });
   if (!found) llvm_unreachable("split-materialize reuse requires a materialize step");
   return found;
}

llvm::SmallVector<CacheTarget, 64> cacheTargetsWithFilterPredReuse(llvm::ArrayRef<CacheTarget> targets) {
   llvm::SmallVector<CacheTarget, 64> out;
   out.reserve(targets.size());
   for (const CacheTarget& t : targets) {
      if (t.enableFilterPredReuse) out.push_back(t);
   }
   return out;
}

/// SSA closure rooted at one \c cache_get result (expanded through execution_step / nested group ports).
static llvm::DenseSet<void*> collectSsaClosureFromCacheGetRoot(mlir::ModuleOp module, mlir::Value cacheGetRoot) {
   llvm::DenseSet<void*> closure;
   llvm::SmallVector<mlir::Value, 64> worklist;
   llvm::DenseSet<void*> seenValues;
   auto enqueueValue = [&](mlir::Value v) {
      if (!v) return;
      void* p = v.getAsOpaquePointer();
      if (!seenValues.insert(p).second) return;
      worklist.push_back(v);
      closure.insert(p);
   };
   enqueueValue(cacheGetRoot);
   while (!worklist.empty()) {
      mlir::Value v = worklist.pop_back_val();
      for (mlir::Operation* user : v.getUsers()) {
         closure.insert(user);
         for (mlir::OpOperand& operand : user->getOpOperands()) {
            if (operand.get() == v) continue;
            enqueueValue(operand.get());
         }
         if (auto step = mlir::dyn_cast<ExecutionStepOp>(user)) {
            mlir::Block& body = step.getSubOps().front();
            for (mlir::BlockArgument arg : body.getArguments()) {
               for (mlir::Value operand : step.getOperands()) {
                  if (operand == v) enqueueValue(arg);
               }
            }
         }
      }
   }
   for (;;) {
      const size_t before = closure.size();
      expandClosureThroughExecutionStepPorts(module, closure);
      expandClosureThroughNestedExecutionGroupPorts(module, closure);
      if (closure.size() == before) break;
   }
   return closure;
}

/// Map any `execution_step` (including nested regions) to the enclosing top-level step that is a
/// direct child of `donor`'s main block — cloning only ever inserts those top-level ops.
///
/// Returns a step different from \p s when \p s lives under `nested_execution_group` (see
/// PrepareForLowering::splitNested): `collectModuleReuseInfo` records those nested writers, but
/// `cloneExecutionStepsToQuery0` only clones top-level steps under the donor `execution_group`.
static ExecutionStepOp liftToTopLevelStepInDonor(ExecutionGroupOp donor, ExecutionStepOp s) {
   for (mlir::Operation* op = s.getOperation(); ;) {
      if (auto es = mlir::dyn_cast<ExecutionStepOp>(op)) {
         if (es->getParentOp() == donor.getOperation()) {
            return es;
         }
      }
      mlir::Region* pr = op->getParentRegion();
      assert(pr);
      op = pr->getParentOp();
      assert(op);
   }
}

/// For every needed state, include:
/// - each top-level `execution_step` listed in `writerStepsByState` (all writes / updates), and
/// - the top-level `execution_step` that **defines** the state SSA result, if any (create path).
/// - the create-only `execution_step` registered in `ModuleReuseInfo::createOnlyStepForState`, if any.
/// - the top-level `execution_step` that returns the state SSA (e.g. `subop.merge` with no RW write edge).
static ExecutionStepOp findTopLevelStepReturningState(ExecutionGroupOp donor, mlir::Value state) {
   ExecutionStepOp found;
   donor.walk([&](ExecutionStepOp step) {
      if (step->getParentOp() != donor.getOperation()) return mlir::WalkResult::advance();
      auto canon = [&](mlir::Value v) { return v ? canonicalizeStateValueForReuse(v) : mlir::Value{}; };
      for (mlir::Value r : step.getResults()) {
         if (canon(r) == state) {
            found = step;
            return mlir::WalkResult::interrupt();
         }
      }
      auto& body = step.getSubOps().front();
      if (auto ret = mlir::dyn_cast<subop::ExecutionStepReturnOp>(body.getTerminator())) {
         for (mlir::Value o : ret.getOperands()) {
            if (canon(o) == state) {
               found = step;
               return mlir::WalkResult::interrupt();
            }
         }
      }
      return mlir::WalkResult::advance();
   });
   return found;
}

static llvm::SmallVector<ExecutionStepOp, 32>
collectCreateAndWriteStepsForStates(ExecutionGroupOp donor, const ModuleReuseInfo& reuse,
                                    const llvm::DenseSet<mlir::Value>& neededStates) {
   llvm::DenseSet<mlir::Operation*> seen;
   llvm::SmallVector<ExecutionStepOp, 32> out;
   for (mlir::Value s : neededStates) {
      mlir::Value key = bufferJoinChainRootForReuse(s, reuse);
      assert(canonicalizeStateValueForReuse(key) == key);
      if (auto itC = reuse.createOnlyStepForState.find(key); itC != reuse.createOnlyStepForState.end()) {
         ExecutionStepOp top = liftToTopLevelStepInDonor(donor, itC->second);
         if (seen.insert(top.getOperation()).second) out.push_back(top);
      }
      auto itW = findReuseMap(reuse.writerStepsByState, key);
      if (itW != reuse.writerStepsByState.end()) {
         for (ExecutionStepOp w : itW->second) {
            ExecutionStepOp top = liftToTopLevelStepInDonor(donor, w);
            if (seen.insert(top.getOperation()).second) out.push_back(top);
         }
      }
      if (auto def = findTopLevelStepReturningState(donor, key)) {
         if (seen.insert(def.getOperation()).second) out.push_back(def);
      }
   }
   llvm::sort(out, [](ExecutionStepOp a, ExecutionStepOp b) { return a->isBeforeInBlock(b); });
   return out;
}

/// Any SSA referenced inside cloned `execution_step` regions must be defined by ops that are also
/// cloned into the synthetic module. `expandNeededStatesFromTargets` only walks SubOp pipeline
/// state values (not external `!subop.table` streams), so clone seeds can otherwise miss the
/// top-level donor steps that host `get_external` / table producers while still using their results
/// in nested maps — leaving operands pointing at SSA in the donor module (undefined behavior at
/// runtime). This closure repeatedly pulls in top-level `execution_step` producers for every
/// reached operand.
static llvm::SmallVector<ExecutionStepOp, 32>
augmentStepsWithOperandProducerClosure(ExecutionGroupOp donor,
                                       llvm::ArrayRef<ExecutionStepOp> seedSteps) {
   llvm::DenseSet<mlir::Operation*> seen;
   llvm::SmallVector<ExecutionStepOp, 32> worklist;
   worklist.reserve(seedSteps.size());
   for (ExecutionStepOp s : seedSteps) {
      if (seen.insert(s.getOperation()).second) {
         worklist.push_back(s);
      }
   }
   for (size_t i = 0; i < worklist.size(); ++i) {
      ExecutionStepOp s = worklist[i];
      s.walk([&](mlir::Operation* op) {
         for (mlir::Value v : op->getOperands()) {
            mlir::Value peeled = peelBlockArgsToEnclosingOperands(v);
            mlir::Operation* def = peeled.getDefiningOp();
            if (!def) continue;
            ExecutionStepOp innerSt;
            if (auto es = mlir::dyn_cast<ExecutionStepOp>(def)) {
               innerSt = es;
            } else {
               innerSt = def->getParentOfType<ExecutionStepOp>();
            }
            if (!innerSt) continue;
            ExecutionStepOp top = liftToTopLevelStepInDonor(donor, innerSt);
            if (seen.insert(top.getOperation()).second) {
               worklist.push_back(top);
            }
         }
      });
   }
   llvm::sort(worklist, [](ExecutionStepOp a, ExecutionStepOp b) {
      return a->isBeforeInBlock(b);
   });
   return worklist;
}

/// Every pipeline state written by \p step (results + `StepRW::writes`) is in \p obsoleteCanonPtrs.
static bool stepWritesOnlyStatesInObsoleteSet(
   ExecutionStepOp step, const llvm::DenseSet<void*>& obsoleteCanonPtrs,
   const llvm::DenseMap<mlir::Operation*, const ModuleReuseInfo::StepRW*>& rwByStepOp) {
   bool sawPipelineWrite = false;
   bool allInObsolete = true;
   auto check = [&](mlir::Value w) {
      w = peelBlockArgsToEnclosingOperands(w);
      if (!isPipelineStateValue(w)) return;
      sawPipelineWrite = true;
      mlir::Value c = canonicalizeStateValueForReuse(w);
      if (!obsoleteCanonPtrs.contains(c.getAsOpaquePointer())) allInObsolete = false;
   };

   for (mlir::OpResult r : step.getResults()) check(r);
   if (const ModuleReuseInfo::StepRW* rw = rwByStepOp.lookup(step.getOperation())) {
      for (mlir::Value w : rw->writes) check(w);
   }

   return sawPipelineWrite && allInObsolete;
}

/// Never erase `execution_step`s that host `cache_get` (inserted at the start of the group): their
/// result SSA may fingerprint like obsolete construction outputs, but removing them corrupts IR.
static bool executionStepContainsCacheGet(ExecutionStepOp step) {
   bool found = false;
   step.walk([&](subop::CacheGetOp) {
      found = true;
      return mlir::WalkResult::interrupt();
   });
   return found;
}

mlir::IRMapping cloneExecutionStepsToQuery0(ExecutionGroupOp dstGroup,
                                            llvm::ArrayRef<ExecutionStepOp> steps) {
   auto& dstBlock = dstGroup.getSubOps().front();
   auto* dstTerminator = dstBlock.getTerminator();
   assert(dstTerminator && "dst execution_group must have a terminator");

   mlir::IRMapping mapping;
   for (auto s : steps) {
      auto* cloned = s.getOperation()->clone(mapping);
      dstBlock.getOperations().insert(mlir::Block::iterator(dstTerminator), cloned);
   }
   return mapping;
}

/// Join-buffer / HIV layout extension and post-layout merge alignment (producer or consumer).
void maybeExtendJoinBufferHashmapLayoutForFilterPred(mlir::ModuleOp module, llvm::ArrayRef<CacheTarget> targets,
                                                     const ModuleReuseInfo& reuse) {
   llvm::SmallVector<mlir::Value, 8> bufferCandidates;
   collectJoinBufferStatesFromTargets(targets, bufferCandidates, &reuse);
   llvm::SmallVector<mlir::Value, 8> extendJoinBuffers =
      collectJoinBuffersFeedingHashIndexedView(bufferCandidates, reuse);
   rewriteHashmapTypesInModule(module, extendJoinBuffers, extendJoinBuffers.empty() ? nullptr : &reuse);
}

void maybeFinalizeModuleAfterJoinBufferFilterPredLayout(mlir::ModuleOp module) {
   propagateSubOpColumnAttrsFromSsaStateLayout(module, nullptr);
   alignBufferMergeThreadLocalsWithExtendedMergeResult(module);
   // `alignBufferMergeThreadLocalsWithExtendedMergeResult` can widen `execution_step` SSA results
   // (join-buffer `filter_pred$0`) without updating the body's `execution_step_return` operands from
   // `subop.create` / `subop.create_thread_local`. Push result types back onto return operands, then
   // re-sync step ports so lowering sees a consistent layout.
   module.walk([&](subop::ExecutionStepReturnOp ret) {
      auto step = mlir::dyn_cast<subop::ExecutionStepOp>(ret->getParentOp());
      if (!step || step.getNumResults() != ret.getNumOperands()) return;
      for (unsigned i = 0; i < ret.getNumOperands(); ++i) {
         mlir::Value out = ret.getOperand(i);
         mlir::Type want = step.getResult(i).getType();
         if (out.getType() != want) out.setType(want);
      }
   });
   synchronizeExecutionStepPortTypes(module, nullptr);
}

llvm::DenseMap<mlir::Value, llvm::SmallVector<runtime::FilterDescription, 8>>
maybeDecodeFiltersByCacheTargets(llvm::ArrayRef<CacheTarget> targets, const ModuleReuseInfo& reuse) {
   return decodeFiltersByCacheTargets(cacheTargetsWithFilterPredReuse(targets), reuse);
}

struct SimpleNumericFilterRange {
   bool hasLower = false;
   double lower = 0.0;
   bool lowerInclusive = true;
   bool hasUpper = false;
   double upper = 0.0;
   bool upperInclusive = true;
   std::optional<double> eq;
};

static std::optional<double> getNumericFilterValue(const runtime::FilterDescription& f) {
   if (const auto* v = std::get_if<int64_t>(&f.value)) return static_cast<double>(*v);
   if (const auto* v = std::get_if<double>(&f.value)) return *v;
   return std::nullopt;
}

static bool isSupportedRangeFilterOp(runtime::FilterOp op) {
   switch (op) {
      case runtime::FilterOp::EQ:
      case runtime::FilterOp::LT:
      case runtime::FilterOp::LTE:
      case runtime::FilterOp::GT:
      case runtime::FilterOp::GTE:
         return true;
      default:
         return false;
   }
}

static void addRangeConstraint(SimpleNumericFilterRange& range, runtime::FilterOp op, double value) {
   switch (op) {
      case runtime::FilterOp::EQ:
         range.eq = value;
         break;
      case runtime::FilterOp::GT:
      case runtime::FilterOp::GTE: {
         bool inclusive = op == runtime::FilterOp::GTE;
         if (!range.hasLower || value > range.lower) {
            range.hasLower = true;
            range.lower = value;
            range.lowerInclusive = inclusive;
         } else if (value == range.lower) {
            range.lowerInclusive = range.lowerInclusive && inclusive;
         }
         break;
      }
      case runtime::FilterOp::LT:
      case runtime::FilterOp::LTE: {
         bool inclusive = op == runtime::FilterOp::LTE;
         if (!range.hasUpper || value < range.upper) {
            range.hasUpper = true;
            range.upper = value;
            range.upperInclusive = inclusive;
         } else if (value == range.upper) {
            range.upperInclusive = range.upperInclusive && inclusive;
         }
         break;
      }
      default:
         break;
   }
}

static bool rangeAllowsValue(const SimpleNumericFilterRange& range, double value) {
   if (range.hasLower) {
      if (value < range.lower) return false;
      if (value == range.lower && !range.lowerInclusive) return false;
   }
   if (range.hasUpper) {
      if (value > range.upper) return false;
      if (value == range.upper && !range.upperInclusive) return false;
   }
   return true;
}

static bool rangeIsEmpty(const SimpleNumericFilterRange& range) {
   if (range.eq && !rangeAllowsValue(range, *range.eq)) return true;
   if (!range.hasLower || !range.hasUpper) return false;
   if (range.lower > range.upper) return true;
   return range.lower == range.upper && !(range.lowerInclusive && range.upperInclusive);
}

static bool rangesAreDisjoint(const SimpleNumericFilterRange& a, const SimpleNumericFilterRange& b) {
   if (rangeIsEmpty(a) || rangeIsEmpty(b)) return true;
   if (a.eq && b.eq) return *a.eq != *b.eq;
   if (a.eq) return !rangeAllowsValue(b, *a.eq);
   if (b.eq) return !rangeAllowsValue(a, *b.eq);

   if (a.hasUpper && b.hasLower) {
      if (a.upper < b.lower) return true;
      if (a.upper == b.lower && !(a.upperInclusive && b.lowerInclusive)) return true;
   }
   if (b.hasUpper && a.hasLower) {
      if (b.upper < a.lower) return true;
      if (b.upper == a.lower && !(b.upperInclusive && a.lowerInclusive)) return true;
   }
   return false;
}

static llvm::StringMap<SimpleNumericFilterRange>
buildSimpleRangeConstraints(llvm::ArrayRef<runtime::FilterDescription> filters) {
   llvm::StringMap<SimpleNumericFilterRange> byColumn;
   for (const runtime::FilterDescription& f : filters) {
      if (!isSupportedRangeFilterOp(f.op)) continue;
      std::optional<double> value = getNumericFilterValue(f);
      if (!value) continue;
      addRangeConstraint(byColumn[f.columnName], f.op, *value);
   }
   return byColumn;
}

static bool filtersAreDefinitelyDisjoint(llvm::ArrayRef<runtime::FilterDescription> filtersA,
                                         llvm::ArrayRef<runtime::FilterDescription> filtersB) {
   llvm::StringMap<SimpleNumericFilterRange> rangesA = buildSimpleRangeConstraints(filtersA);
   llvm::StringMap<SimpleNumericFilterRange> rangesB = buildSimpleRangeConstraints(filtersB);
   for (auto& entryA : rangesA) {
      auto entryB = rangesB.find(entryA.getKey());
      if (entryB == rangesB.end()) continue;
      if (rangesAreDisjoint(entryA.getValue(), entryB->getValue())) return true;
   }
   return false;
}

static llvm::SmallVector<runtime::FilterDescription, 8>
decodeFiltersForPotentialReuseState(mlir::Value state, const ModuleReuseInfo& reuse) {
   llvm::SmallVector<runtime::FilterDescription, 8> filters =
      decodeFiltersForStateFromWriterSteps(state, reuse);
   mlir::Value resolved = resolveCacheTargetStateForReuse(state, reuse);
   if (filters.empty() && resolved && resolved != state) {
      filters = decodeFiltersForStateFromWriterSteps(resolved, reuse);
   }
   if (filters.empty() && resolved) {
      forEachShadowChainPredecessor(resolved, reuse, [&](mlir::Value shadow) {
         if (!filters.empty()) return;
         filters = decodeFiltersForStateFromWriterSteps(shadow, reuse);
      });
   }
   return filters;
}

[[maybe_unused]] static bool reuseMatchFiltersDefinitelyDisjoint(mlir::Value stateA, mlir::Value stateB,
                                                                 const ModuleReuseInfo& reuseA,
                                                                 const ModuleReuseInfo& reuseB) {
   llvm::SmallVector<runtime::FilterDescription, 8> filtersA =
      decodeFiltersForPotentialReuseState(stateA, reuseA);
   llvm::SmallVector<runtime::FilterDescription, 8> filtersB =
      decodeFiltersForPotentialReuseState(stateB, reuseB);
   if (filtersA.empty() || filtersB.empty()) return false;
   return filtersAreDefinitelyDisjoint(filtersA, filtersB);
}

void maybeApplyWriteSideFilterPredOnProducerHashmap(
   mlir::Value st, const ModuleReuseInfo& reuse,
   const llvm::DenseMap<mlir::Operation*, const ModuleReuseInfo::StepRW*>& rwByStepOp) {
   if (!mlir::isa<subop::HashMapType>(st.getType())) return;

   auto itTL = reuse.mergedFromShadowState.find(st);
   assert(itTL != reuse.mergedFromShadowState.end() && "hashmap merge result must have paired thread_local");
   mlir::Value tl = itTL->second;

   auto decoded = decodeFiltersForStateFromWriterSteps(tl, reuse, &rwByStepOp);
   if (decoded.empty()) return;

   auto itW = reuse.writerStepsByState.find(tl);
   assert(itW != reuse.writerStepsByState.end());
   ExecutionStepOp construction;
   for (auto ws : itW->second) {
      mlir::Block& body = ws.getSubOps().front();
      bool hasLoi = false, hasRed = false;
      for (auto& op : body.without_terminator()) {
         if (mlir::isa<subop::LookupOrInsertOp>(&op)) hasLoi = true;
         if (mlir::isa<subop::ReduceOp>(&op)) hasRed = true;
      }
      if (hasLoi && hasRed) {
         construction = ws;
         break;
      }
   }
   assert(construction && "write_pred: could not find construction step for join hashmap");
   insertWriteSidePredIntoHashMapConstructionStep(construction, decoded);
}

void maybePatchJoinBufferWritersWithFilterPred(
   llvm::ArrayRef<CacheTarget> targets,
   const llvm::DenseMap<mlir::Value, llvm::SmallVector<runtime::FilterDescription, 8>>& decodedFiltersByTarget,
   const ModuleReuseInfo& reuse) {
   for (auto& t : targets) {
      if (!t.state) continue;
      if (!mlir::isa<subop::BufferType>(t.state.getType())) continue;
      auto itF = decodedFiltersByTarget.find(t.state);
      if (itF == decodedFiltersByTarget.end() || itF->second.empty()) continue;
      auto itTL = findReuseMap(reuse.mergedFromShadowState, t.state);
      if (itTL == reuse.mergedFromShadowState.end()) continue;
      auto itW = findReuseMap(reuse.writerStepsByState, itTL->second);
      if (itW == reuse.writerStepsByState.end()) continue;
      for (ExecutionStepOp ws : itW->second) {
         bool hasBufMat = false;
         ws.getOperation()->walk([&](subop::MaterializeOp m) {
            if (mlir::isa<subop::BufferType>(m.getState().getType())) hasBufMat = true;
         });
         if (!hasBufMat) continue;
         insertWriteSidePredIntoBufferConstructionStep(ws, itF->second);
      }
   }
}

/// After `state` is replaced by `cached` from `cache_get`, re-apply decoded construction filters.
void applyFilterPredReapplyAfterCacheGetReplacement(
   mlir::Value state, mlir::Value cached, llvm::ArrayRef<runtime::FilterDescription> decodedFilters,
   ExecutionGroupOp group, subop::Member predMember,
   llvm::DenseSet<mlir::Operation*>& joinBufPredProbeInjectedGroups, bool enableFilterPredReuse) {
   if (!enableFilterPredReuse) return;

   const bool isAggHt = mlir::isa<subop::PreAggrHtType>(state.getType());
   const bool isJoinHm = mlir::isa<subop::HashMapType>(state.getType());
   const bool isJoinBuf = mlir::isa<subop::BufferType>(state.getType());
   const bool isJoinHiv = mlir::isa<subop::HashIndexedViewType>(state.getType());

   if (!decodedFilters.empty() && !isAggHt && !isJoinHm && !isJoinBuf) {
      materializeRuntimeFiltersAtCacheGetUses(cached, decodedFilters);
   }

   if (isJoinHm && !decodedFilters.empty()) {
      for (auto& u : cached.getUses()) {
         auto step = mlir::dyn_cast<subop::ExecutionStepOp>(u.getOwner());
         if (!step) continue;
         insertScanRefsPredFilter(step, predMember);
      }
   }

   // HIV probe `filter_pred` checks are retagged to MixedHIV after consumer layout align via
   // `applyProbePredFiltersForConsumerClosures` (see `rewritePlansWithSyntheticQuery0`).
   (void)isJoinBuf;
   (void)isJoinHiv;
   (void)joinBufPredProbeInjectedGroups;
   (void)predMember;
}

mlir::Operation* topLevelExecutionStepOrSelf(mlir::Operation* op) {
   if (!op) return nullptr;
   if (auto parentStep = op->getParentOfType<subop::ExecutionStepOp>()) {
      if (parentStep.getOperation() != op) return parentStep.getOperation();
   }
   return op;
}

} // namespace

void insertCachePutsForTargets(mlir::ModuleOp producerModule, llvm::ArrayRef<CacheTarget> targets,
                               const ModuleReuseInfo* reuseBeforeMutation) {
   if (targets.empty()) return;

   ModuleReuseInfo ownedReuse;
   if (!reuseBeforeMutation) {
      ownedReuse = collectModuleReuseInfo(producerModule);
      reuseBeforeMutation = &ownedReuse;
   }
   const ModuleReuseInfo& reuse = *reuseBeforeMutation;

   maybeExtendJoinBufferHashmapLayoutForFilterPred(producerModule, cacheTargetsWithFilterPredReuse(targets), reuse);

   auto rwByStepOp = buildRwByStepOpMap(reuse);
   llvm::DenseSet<uint64_t> seenKeys;
   for (auto& t : targets) {
      if (!t.state) continue;
      assert(!mlir::isa<ThreadLocalType>(t.state.getType()) &&
             "insertCachePutsForTargets must never be called for thread_local-wrapped states");
      if (!seenKeys.insert(t.cacheKey).second) continue;

      mlir::Value st = t.state;
      mlir::Operation* insertAfter = topLevelExecutionStepOrSelf(st.getDefiningOp());
      if (auto itW = reuse.writerStepsByState.find(st); itW != reuse.writerStepsByState.end()) {
         for (auto s : itW->second) {
            auto* op = topLevelExecutionStepOrSelf(s.getOperation());
            if (!insertAfter) insertAfter = op;
            else if (insertAfter->isBeforeInBlock(op)) insertAfter = op;
         }
      }
      if (!insertAfter) continue;

      maybeApplyWriteSideFilterPredOnProducerHashmap(st, reuse, rwByStepOp);

      auto loc = insertAfter->getLoc();
      mlir::OpBuilder builder(st.getContext());
      builder.setInsertionPointAfter(insertAfter);
      auto resTy = st.getType();
      auto step = builder.create<ExecutionStepOp>(loc, mlir::TypeRange{}, mlir::ValueRange{st},
                                                  builder.getArrayAttr({builder.getBoolAttr(false)}));
      auto& block = step.getSubOps().emplaceBlock();
      block.addArgument(resTy, loc);
      builder.setInsertionPointToStart(&block);
      builder.create<CachePutOp>(loc, builder.getI64IntegerAttr(static_cast<int64_t>(t.cacheKey)), block.getArgument(0));
      builder.create<ExecutionStepReturnOp>(loc, mlir::ValueRange{});
   }
   maybeFinalizeModuleAfterJoinBufferFilterPredLayout(producerModule);
}

void injectCacheGetsAndDeleteConstructionSteps(mlir::ModuleOp consumerModule, llvm::ArrayRef<CacheTarget> targets,
                                               const ModuleReuseInfo* reuseBeforeMutation,
                                               bool joinBufferHashmapLayoutAlreadyApplied,
                                               bool joinBufferWritePredAlreadyApplied) {
   if (targets.empty()) return;

   ModuleReuseInfo ownedReuse;
   if (!reuseBeforeMutation) {
      ownedReuse = collectModuleReuseInfo(consumerModule);
      reuseBeforeMutation = &ownedReuse;
   }
   const ModuleReuseInfo& reuse = *reuseBeforeMutation;

   llvm::DenseMap<void*, mlir::Type> cacheGetStateTypeBeforePredLayout;
   for (const CacheTarget& t : targets) {
      if (!t.state) continue;
      cacheGetStateTypeBeforePredLayout[t.state.getAsOpaquePointer()] = t.state.getType();
   }

   if (!joinBufferHashmapLayoutAlreadyApplied) {
      maybeExtendJoinBufferHashmapLayoutForFilterPred(consumerModule, cacheTargetsWithFilterPredReuse(targets),
                                                    reuse);
   }

   auto findEnclosingExecutionGroup = [&](mlir::Value v) -> ExecutionGroupOp {
      if (auto* defOp = v.getDefiningOp()) {
         if (auto g = defOp->getParentOfType<ExecutionGroupOp>()) return g;
      }
      for (auto& use : v.getUses()) {
         if (auto g = use.getOwner()->getParentOfType<ExecutionGroupOp>()) return g;
      }
      llvm_unreachable("state must be inside an execution_group");
   };

   auto insertCacheGetAtExecutionGroupStart = [&](ExecutionGroupOp group, mlir::Type stateTy, uint64_t key) -> mlir::Value {
      auto loc = group->getLoc();
      mlir::OpBuilder builder(group.getContext());
      builder.setInsertionPointToStart(&group.getSubOps().front());
      auto step = builder.create<ExecutionStepOp>(loc, mlir::TypeRange{stateTy}, mlir::ValueRange{},
                                                  builder.getArrayAttr({builder.getBoolAttr(false)}));
      auto& block = step.getSubOps().emplaceBlock();
      builder.setInsertionPointToStart(&block);
      auto got = builder.create<CacheGetOp>(loc, stateTy, builder.getI64IntegerAttr(static_cast<int64_t>(key)));
      builder.create<ExecutionStepReturnOp>(loc, mlir::ValueRange{got.getRes()});
      return step.getResult(0);
   };

   llvm::DenseMap<uint64_t, mlir::Value> keyToCached;

   auto rwByStepOp = buildRwByStepOpMap(reuse);

   llvm::DenseMap<mlir::Value, llvm::SmallVector<runtime::FilterDescription, 8>> decodedFiltersByTarget =
      maybeDecodeFiltersByCacheTargets(targets, reuse);

   if (!joinBufferWritePredAlreadyApplied) {
      maybePatchJoinBufferWritersWithFilterPred(cacheTargetsWithFilterPredReuse(targets), decodedFiltersByTarget,
                                              reuse);
   }

   // Erase plan from initial `ModuleReuseInfo` only: backward closure from `execution_group_return`
   // pipeline seeds vs. forward prerequisite closure of `cache_get` targets; obsolete = R \ L.
   ExecutionGroupOp egForErasePlan = getSingleExecutionGroup(consumerModule);
   mlir::Block& egBodyForPlan = egForErasePlan.getSubOps().front();
   auto retForPlan = mlir::dyn_cast<subop::ExecutionGroupReturnOp>(egBodyForPlan.getTerminator());
   assert(retForPlan && "expected execution_group_return terminator");
   llvm::SmallVector<mlir::Value, 8> returnPipelineSeedsForErase;
   for (mlir::Value o : retForPlan.getOperands()) {
      mlir::Value p = peelBlockArgsToEnclosingOperands(o);
      if (!isPipelineStateValue(p)) continue;
      returnPipelineSeedsForErase.push_back(o);
   }

   llvm::DenseSet<mlir::Value> replacedClosureValues = expandNeededStatesFromTargets(targets, reuse);

   auto rewriteOne = [&](mlir::Value state, uint64_t cacheKey, bool enableFilterPredReuse) {
      assert(!mlir::isa<ThreadLocalType>(state.getType()) &&
             "rewrite must never target thread_local-wrapped states");

      llvm::SmallVector<mlir::Value, 8> statesToDelete;
      statesToDelete.push_back(state);
      if (auto itG = findReuseMap(reuse.mergedFromShadowState, state);
          itG != reuse.mergedFromShadowState.end()) {
         statesToDelete.push_back(itG->second);
         if (auto itTL = findReuseMap(reuse.mergedFromShadowState, itG->second);
             itTL != reuse.mergedFromShadowState.end()) {
            statesToDelete.push_back(itTL->second);
         }
      } else if (auto itTL = findReuseMap(reuse.mergedFromShadowState, state);
                 itTL != reuse.mergedFromShadowState.end()) {
         statesToDelete.push_back(itTL->second);
      }

      llvm::SmallVector<ExecutionStepOp, 32> writerSteps;
      for (auto st : statesToDelete) {
         auto itW = reuse.writerStepsByState.find(st);
         if (itW == reuse.writerStepsByState.end()) continue;
         for (auto s : itW->second) writerSteps.push_back(s);
      }
      if (writerSteps.empty()) return;

      llvm::ArrayRef<runtime::FilterDescription> decodedFilters;
      if (auto itF = decodedFiltersByTarget.find(state); itF != decodedFiltersByTarget.end()) {
         decodedFilters = itF->second;
      }

      auto group = findEnclosingExecutionGroup(state);
      mlir::Type cacheGetTy = state.getType();
      if (auto itTy = cacheGetStateTypeBeforePredLayout.find(state.getAsOpaquePointer());
          itTy != cacheGetStateTypeBeforePredLayout.end()) {
         cacheGetTy = itTy->second;
      }
      mlir::Value cached;
      if (auto it = keyToCached.find(cacheKey); it != keyToCached.end()) {
         cached = it->second;
      } else {
         cached = insertCacheGetAtExecutionGroupStart(group, cacheGetTy, cacheKey);
         keyToCached[cacheKey] = cached;
      }

      state.replaceAllUsesWith(cached);

      subop::Member predMember;
      if (enableFilterPredReuse) {
         predMember = makeOrGetPredMember(consumerModule.getContext());
      }
      llvm::DenseSet<mlir::Operation*> joinBufPredProbeInjectedGroups;
      applyFilterPredReapplyAfterCacheGetReplacement(state, cached, decodedFilters, group, predMember,
                                                     joinBufPredProbeInjectedGroups, enableFilterPredReuse);
   };

   for (auto& t : targets) {
      if (!t.state) continue;
      rewriteOne(t.state, t.cacheKey, t.enableFilterPredReuse);
   }

   // Recompute live states from `execution_group_return` **after** `cache_get` replacement so the
   // backward closure does not treat replaced HIV/buffer construction as still required.
   llvm::DenseSet<void*> liveFromReturnReuseOnly =
      closureLiveStatesFromReuseReturnSeeds(returnPipelineSeedsForErase, reuse, rwByStepOp);
   llvm::DenseSet<void*> obsoleteStateCanonPtrs;
   for (mlir::Value rv : replacedClosureValues) {
      mlir::Value c = canonicalizeStateValueForReuse(rv);
      if (!liveFromReturnReuseOnly.contains(c.getAsOpaquePointer()))
         obsoleteStateCanonPtrs.insert(c.getAsOpaquePointer());
   }

   // Erase top-level steps that (per initial `reuse`) only write pipeline states in `obsoleteStateCanonPtrs`
   // (construction for matched states that the return-driven closure does not need). Collect candidates
   // first, erase in reverse block order, and never erase steps that host `cache_get` (see
   // `executionStepContainsCacheGet`) or still have live SSA results.
   if (!returnPipelineSeedsForErase.empty() && !obsoleteStateCanonPtrs.empty()) {
      ExecutionGroupOp eg = getSingleExecutionGroup(consumerModule);
      mlir::Operation* egOp = eg.getOperation();
      mlir::Block& egBody = eg.getSubOps().front();
      llvm::SmallVector<ExecutionStepOp, 32> stepsToErase;
      for (mlir::Operation& op : egBody.without_terminator()) {
         auto step = mlir::dyn_cast<ExecutionStepOp>(&op);
         if (!step) continue;
         if (step->getParentRegion()->getParentOp() != egOp || step->getBlock() != &egBody) continue;
         if (executionStepContainsCacheGet(step)) continue;
         if (!stepWritesOnlyStatesInObsoleteSet(step, obsoleteStateCanonPtrs, rwByStepOp)) continue;
         stepsToErase.push_back(step);
      }
      for (ExecutionStepOp s : llvm::reverse(stepsToErase)) {
         bool resultsUnused = true;
         for (mlir::OpResult r : s.getResults()) {
            if (!r.use_empty()) {
               resultsUnused = false;
               break;
            }
         }
         if (!resultsUnused) continue;
         s.erase();
      }
   }

   llvm::DenseSet<void*> filterPredReuseClosure;
   bool anyFilterPredReuseTarget = false;
   for (const CacheTarget& t : targets) {
      if (!t.enableFilterPredReuse) continue;
      anyFilterPredReuseTarget = true;
      auto itCached = keyToCached.find(t.cacheKey);
      if (itCached == keyToCached.end()) continue;
      llvm::DenseSet<void*> perGet =
         collectSsaClosureFromCacheGetRoot(consumerModule, itCached->second);
      filterPredReuseClosure.insert(perGet.begin(), perGet.end());
   }
   if (anyFilterPredReuseTarget) {
      propagateSubOpColumnAttrsFromSsaStateLayout(consumerModule,
                                                  filterPredReuseClosure.empty() ? nullptr
                                                                                 : &filterPredReuseClosure);
      alignBufferMergeThreadLocalsWithExtendedMergeResult(consumerModule);
   }
}

ReusePlanRewriteResult rewritePlansWithSyntheticQuery0(
   mlir::ModuleOp query0,
   mlir::ModuleOp query1,
   llvm::ArrayRef<CrossQueryStateMatchPair> matches,
   lingodb::catalog::Catalog* catalog) {
   (void)catalog;
   ReusePlanRewriteResult res;

   llvm::SmallVector<CrossQueryStateMatchPair, 64> matchesLocal(matches.begin(), matches.end());
   for (auto& m : matchesLocal) {
      if (m.queryA == 1 && m.queryB == 0) {
         std::swap(m.queryA, m.queryB);
         std::swap(m.stateA, m.stateB);
      }
      assert((m.queryA < 0 || m.queryA == 0) && (m.queryB < 0 || m.queryB == 1) &&
             "rewritePlansWithSyntheticQuery0 expects match stateA/stateB to align with query0/query1");
   }

   llvm::SmallVector<CacheTarget, 64> targets0;
   llvm::SmallVector<CacheTarget, 64> targets1;
   targets0.reserve(matchesLocal.size());
   targets1.reserve(matchesLocal.size());

   auto reuse0Early = collectModuleReuseInfo(query0);
   auto reuse1Early = collectModuleReuseInfo(query1);

   llvm::SmallVector<CrossQueryStateMatchPair, 64> keptMatches;
   keptMatches.reserve(matchesLocal.size());
   for (auto& m : matchesLocal) {
      bool enableFilterPredReuse = m.enableFilterPredReuse;
      mlir::Value ta;
      mlir::Value tb;
      if (m.stateA && m.stateB) {
         ta = resolveCacheTargetStateForReuse(m.stateA, reuse0Early);
         tb = resolveCacheTargetStateForReuse(m.stateB, reuse1Early);
         // Cardinality/disjoint pruning is intentionally disabled for now. The old pairwise
         // rewrite path must match the batch path; pruning should come back through the shared
         // match-group cost model instead of filtering pairs here.
         bool bothHiv = mlir::isa<subop::HashIndexedViewType>(ta.getType()) &&
            mlir::isa<subop::HashIndexedViewType>(tb.getType());
         if (bothHiv) {
            enableFilterPredReuse = true;
            if (joinMatchPeerExternalFiltersIdentical(query0, query1, m.stateA, m.stateB, reuse0Early, reuse1Early)) {
               enableFilterPredReuse = false;
            }
         }
      }
      m.enableFilterPredReuse = enableFilterPredReuse;
      keptMatches.push_back(m);

      if (m.stateA) {
         if (!ta) ta = resolveCacheTargetStateForReuse(m.stateA, reuse0Early);
         assert(!mlir::isa<ThreadLocalType>(ta.getType()) &&
                "match pairs must never target thread_local-wrapped states");
         targets0.push_back(CacheTarget{ta, m.cacheKey, enableFilterPredReuse});
      }
      if (m.stateB) {
         if (!tb) tb = resolveCacheTargetStateForReuse(m.stateB, reuse1Early);
         assert(!mlir::isa<ThreadLocalType>(tb.getType()) &&
                "match pairs must never target thread_local-wrapped states");
         targets1.push_back(CacheTarget{tb, m.cacheKey, enableFilterPredReuse});
      }
   }
   matchesLocal = std::move(keptMatches);

   res.numTargetsQuery0 = targets0.size();
   res.numTargetsQuery1 = targets1.size();
   auto countNoTable = [](llvm::ArrayRef<CacheTarget> targets) -> size_t {
      size_t n = 0;
      for (auto& t : targets) {
         if (!mlir::isa<TableType>(t.state.getType())) n++;
      }
      return n;
   };
   res.numTargetsQuery0NoTable = countNoTable(targets0);
   res.numTargetsQuery1NoTable = countNoTable(targets1);

   if (targets0.empty() || targets1.empty()) {
      return res;
   }

   // Build synthetic query0 module from scratch in the same context as query0, then clone steps into it.
   mlir::MLIRContext* q0Ctx = query0.getContext();
   res.query0 = mlir::OwningOpRef<mlir::ModuleOp>(mlir::ModuleOp::create(mlir::UnknownLoc::get(q0Ctx)));

   auto donorGroup = getSingleExecutionGroup(query0);
   auto donorMain = query0.lookupSymbol<mlir::func::FuncOp>("main");
   assert(donorMain && "expected func @main");

   auto q0Main = mlir::func::FuncOp::create(donorMain.getLoc(), "main", donorMain.getFunctionType());
   res.query0->push_back(q0Main);
   auto* entry = q0Main.addEntryBlock();
   mlir::OpBuilder fb = mlir::OpBuilder::atBlockBegin(entry);

   auto q0Group = fb.create<ExecutionGroupOp>(donorGroup.getLoc(), mlir::TypeRange{}, mlir::ValueRange{});
   auto& q0Block = q0Group.getSubOps().emplaceBlock();
   mlir::OpBuilder gb = mlir::OpBuilder::atBlockBegin(&q0Block);
   gb.create<ExecutionGroupReturnOp>(donorGroup.getLoc(), mlir::ValueRange{});
   fb.create<mlir::func::ReturnOp>(donorMain.getLoc());

   // Clone construction into synthetic query0: seed **states** (matched targets), BFS to all
   // prerequisite states via reads on writer steps (`ModuleReuseInfo`), then take every state's
   // defining step, `createOnlyStepForState`, and all writer steps (`writerStepsByState`), then SSA
   // predecessors for remaining operand edges (merge, hash views, …).
   auto reuse0 = reuse0Early;
   auto reuse1 = reuse1Early;
   llvm::DenseSet<mlir::Value> neededStates = expandNeededStatesFromTargets(targets0, reuse0);
   llvm::SmallVector<ExecutionStepOp, 32> stepsToClone =
      collectCreateAndWriteStepsForStates(donorGroup, reuse0, neededStates);
   stepsToClone = augmentStepsWithOperandProducerClosure(donorGroup, stepsToClone);

   auto mapping = cloneExecutionStepsToQuery0(q0Group, stepsToClone);

   // Map targets into query0.
   llvm::SmallVector<CacheTarget, 64> targetsQ0;
   targetsQ0.reserve(targets0.size());
   for (auto& t : targets0) {
      assert(t.state && "reuse target state must be set");
      mlir::Value mapped = mapping.lookupOrNull(t.state);
      if (!mapped && !t.state.getDefiningOp()) {
         if (auto itW = reuse0.writerStepsByState.find(t.state); itW != reuse0.writerStepsByState.end()) {
            if (!itW->second.empty()) {
               auto lastWriter = itW->second.back();
               if (lastWriter.getNumResults() == 1) {
                  mapped = mapping.lookupOrNull(lastWriter.getResult(0));
               }
            }
         }
      }
      assert(mapped && "reuse target must map into synthetic query0 module");
      targetsQ0.push_back(CacheTarget{mapped, t.cacheKey, t.enableFilterPredReuse});
   }
   res.numTargetsQuery0Mapped = targetsQ0.size();
   res.numTargetsQuery0MappedNoTable = countNoTable(targetsQ0);
   if (targetsQ0.empty()) {
      return res;
   }

   CachedJoinBufferLayoutsByKey producerLayoutsByKey;
   extendSyntheticJoinBuffersToColumnUnion(*res.query0, query0, query1, matchesLocal, targetsQ0, mapping,
                                           &producerLayoutsByKey);
   CachedAggregateLayoutsByKey aggregateLayoutsByKey;
   extendSyntheticAggregateHashTablesToPayloadUnion(*res.query0, query0, query1, matchesLocal, targetsQ0, mapping,
                                                   &aggregateLayoutsByKey);
   auto reuseSyntheticAfterLayoutPrep = collectModuleReuseInfo(*res.query0);
   ClonedJoinBufferBuildSitesByKey joinBuildSites =
      recordClonedJoinBufferBuildSites(*res.query0, targetsQ0, reuseSyntheticAfterLayoutPrep);
   insertSyntheticFilterPredsAfterColumnUnion(*res.query0, query0, query1, matchesLocal, targetsQ0,
                                              producerLayoutsByKey, joinBuildSites);

   // Producer: cache_puts (cloned synthetic IR — needs its own reuse snapshot).
   {
      auto reuseSynthetic = collectModuleReuseInfo(*res.query0);
      refreshClonedJoinBufferBuildSiteStates(*res.query0, joinBuildSites, reuseSynthetic);
      llvm::SmallVector<CacheTarget, 16> cachePutTargets;
      cachePutTargets.reserve(targetsQ0.size());
      for (const CacheTarget& t : targetsQ0) {
         mlir::Value state = t.state;
         if (auto it = joinBuildSites.find(t.cacheKey); it != joinBuildSites.end())
            state = it->second.syntheticHiv;
         cachePutTargets.push_back(CacheTarget{state, t.cacheKey, t.enableFilterPredReuse});
      }
      insertCachePutsForTargets(*res.query0, cachePutTargets, &reuseSynthetic);
      extendSyntheticJoinBuffersWithInheritedMixedPreds(*res.query0, cachePutTargets, &producerLayoutsByKey);
      refreshCachedJoinLayoutsFromSyntheticCachePuts(*res.query0, targetsQ0, producerLayoutsByKey);
      for (const CacheTarget& t : cachePutTargets) {
         if (auto it = producerLayoutsByKey.find(t.cacheKey); it != producerLayoutsByKey.end()) {
            alignConsumerModulesToCachedJoinLayout(*res.query0, it->second, t.cacheKey, std::nullopt, nullptr);
            resyncConsumerCachedHivCarrierTypesFromCacheGet(*res.query0, t.cacheKey);
         }
      }
   }

   injectCacheGetsAndDeleteConstructionSteps(query0, targets0, &reuse0, /*joinBufferHashmapLayoutAlreadyApplied=*/true,
                                           /*joinBufferWritePredAlreadyApplied=*/true);
   injectCacheGetsAndDeleteConstructionSteps(query1, targets1, &reuse1, /*joinBufferHashmapLayoutAlreadyApplied=*/true,
                                           /*joinBufferWritePredAlreadyApplied=*/true);

   llvm::SmallVector<ConsumerCacheGetProbeClosure, 4> probeClosuresQ0;
   llvm::SmallVector<ConsumerCacheGetProbeClosure, 4> probeClosuresQ1;

   for (auto& t : targets0) {
      if (auto it = producerLayoutsByKey.find(t.cacheKey); it != producerLayoutsByKey.end()) {
         std::optional<unsigned> consumerQ = 0u;
         alignConsumerModulesToCachedJoinLayout(query0, it->second, t.cacheKey, consumerQ, &probeClosuresQ0);
      }
   }
   for (auto& t : targets1) {
      if (auto it = producerLayoutsByKey.find(t.cacheKey); it != producerLayoutsByKey.end()) {
         std::optional<unsigned> consumerQ = 1u;
         alignConsumerModulesToCachedJoinLayout(query1, it->second, t.cacheKey, consumerQ, &probeClosuresQ1);
      }
   }
   for (auto& t : targets0) {
      if (auto it = aggregateLayoutsByKey.find(t.cacheKey); it != aggregateLayoutsByKey.end()) {
         alignConsumerModulesToCachedAggregateLayout(query0, it->second, t.cacheKey, 0u);
      }
   }
   for (auto& t : targets1) {
      if (auto it = aggregateLayoutsByKey.find(t.cacheKey); it != aggregateLayoutsByKey.end()) {
         alignConsumerModulesToCachedAggregateLayout(query1, it->second, t.cacheKey, 1u);
      }
   }

   for (auto& t : targets0) {
      resyncConsumerCachedHivCarrierTypesFromCacheGet(query0, t.cacheKey);
   }
   for (auto& t : targets1) {
      resyncConsumerCachedHivCarrierTypesFromCacheGet(query1, t.cacheKey);
   }
   for (ConsumerCacheGetProbeClosure& probe : probeClosuresQ0) {
      finalizeConsumerCachedJoinProbeColumnAttrs(query0, probe);
   }
   for (ConsumerCacheGetProbeClosure& probe : probeClosuresQ1) {
      finalizeConsumerCachedJoinProbeColumnAttrs(query1, probe);
   }
   syncProbeGatherMappingsInModule(query0);
   syncProbeGatherMappingsInModule(query1);
   applyProbePredFiltersForConsumerClosures(query0, probeClosuresQ0);
   applyProbePredFiltersForConsumerClosures(query1, probeClosuresQ1);

   return res;
}

static void expandSplitMaterializeTargetsInSynthetic(
   mlir::ModuleOp synthetic,
   llvm::ArrayRef<mlir::ModuleOp> queries,
   llvm::ArrayRef<CrossQueryStateMatchGroup> groups,
   llvm::ArrayRef<ModuleReuseInfo> reuseEarly,
   llvm::MutableArrayRef<mlir::IRMapping> donorMappings,
   const llvm::DenseMap<uint64_t, llvm::DenseMap<unsigned, unsigned>>& consumerSlotByCacheKeyAndQuery,
   llvm::SmallVectorImpl<CacheTarget>& targetsSynthetic);

static BatchReusePlanRewriteResult rewritePlansWithSyntheticQueryBatchOnce(
   llvm::ArrayRef<mlir::ModuleOp> queries,
   llvm::ArrayRef<CrossQueryStateMatchGroup> groups,
   lingodb::catalog::Catalog* catalog) {
   (void)catalog;
   assert(queries.size() >= 2 && "batch reuse needs at least two queries");

   BatchReusePlanRewriteResult res;
   res.numTargetsPerQuery.resize(queries.size(), 0);
   res.numTargetsNoTablePerQuery.resize(queries.size(), 0);

   llvm::SmallVector<ModuleReuseInfo, 8> reuseEarly;
   reuseEarly.reserve(queries.size());
   for (mlir::ModuleOp q : queries) reuseEarly.push_back(collectModuleReuseInfo(q));

   struct EffectiveGroupRewriteFlags {
      bool enableFilterPredReuse = false;
      bool requiresJoinLayoutUnion = false;
   };
   llvm::DenseMap<uint64_t, EffectiveGroupRewriteFlags> flagsByCacheKey;

   auto effectiveFlagsForGroup = [&](const CrossQueryStateMatchGroup& g) {
      EffectiveGroupRewriteFlags flags{g.enableFilterPredReuse, g.requiresJoinLayoutUnion};
      if (!g.enableFilterPredReuse) return flags;
      const CrossQueryStateMatchEntry* donor = nullptr;
      bool allHiv = true;
      for (const CrossQueryStateMatchEntry& e : g.entries) {
         if (e.query < 0 || static_cast<size_t>(e.query) >= queries.size() || !e.state) continue;
         mlir::Value target = resolveCacheTargetStateForReuse(e.state, reuseEarly[e.query]);
         if (!target || !mlir::isa<HashIndexedViewType>(target.getType())) {
            allHiv = false;
            break;
         }
         if (!donor || e.query < donor->query) donor = &e;
      }
      if (!allHiv || !donor) return flags;

      bool allPeerFiltersIdentical = true;
      for (const CrossQueryStateMatchEntry& e : g.entries) {
         if (&e == donor || e.query < 0 || static_cast<size_t>(e.query) >= queries.size() || !e.state) continue;
         if (!joinMatchPeerExternalFiltersIdentical(queries[donor->query], queries[e.query],
                                                    donor->state, e.state,
                                                    reuseEarly[donor->query], reuseEarly[e.query])) {
            allPeerFiltersIdentical = false;
            break;
         }
      }
      if (allPeerFiltersIdentical) flags.enableFilterPredReuse = false;
      flags.requiresJoinLayoutUnion = g.requiresJoinLayoutUnion || flags.enableFilterPredReuse;
      return flags;
   };

   for (const CrossQueryStateMatchGroup& g : groups) {
      flagsByCacheKey[g.cacheKey] = effectiveFlagsForGroup(g);
   }
   llvm::SmallVector<CrossQueryStateMatchGroup, 64> rewriteGroups(groups.begin(), groups.end());
   for (CrossQueryStateMatchGroup& g : rewriteGroups) {
      EffectiveGroupRewriteFlags flags = flagsByCacheKey.lookup(g.cacheKey);
      g.enableFilterPredReuse = flags.enableFilterPredReuse;
      g.requiresJoinLayoutUnion = flags.requiresJoinLayoutUnion;
   }

   llvm::SmallVector<llvm::SmallVector<CacheTarget, 16>, 8> targetsByQuery(queries.size());
   struct DonorGroup {
      const CrossQueryStateMatchGroup* group = nullptr;
      int donorQuery = -1;
      mlir::Value donorState;
      CacheTarget donorTarget;
   };
   llvm::SmallVector<DonorGroup, 32> donorGroups;
   llvm::DenseMap<uint64_t, llvm::DenseMap<unsigned, unsigned>> consumerSlotByCacheKeyAndQuery;

   auto countNoTable = [](llvm::ArrayRef<CacheTarget> targets) -> size_t {
      size_t n = 0;
      for (const CacheTarget& t : targets) {
         if (!mlir::isa<TableType>(t.state.getType())) n++;
      }
      return n;
   };

   for (const CrossQueryStateMatchGroup& g : rewriteGroups) {
      if (g.entries.size() < 2) continue;
      const CrossQueryStateMatchEntry* donor = nullptr;
      bool unsupportedAggregateUnion = false;
      for (const CrossQueryStateMatchEntry& e : g.entries) {
         if (e.query < 0 || static_cast<size_t>(e.query) >= queries.size()) continue;
         if (!e.state) continue;
         mlir::Value targetState = resolveCacheTargetStateForReuse(e.state, reuseEarly[e.query]);
         assert(targetState && "batch reuse target must resolve during aggregate support check");
         if (mlir::isa<subop::PreAggrHtType>(targetState.getType()) &&
             !aggregateHashTablePayloadUnionSupported(e.state, reuseEarly[e.query])) {
            unsupportedAggregateUnion = true;
            break;
         }
         if (!donor || e.query < donor->query) donor = &e;
      }
      if (unsupportedAggregateUnion) continue;
      if (!donor) continue;

      mlir::Value donorTargetState =
         resolveCacheTargetStateForReuse(donor->state, reuseEarly[donor->query]);
      assert(donorTargetState && "batch reuse donor target must resolve");
      assert(!mlir::isa<ThreadLocalType>(donorTargetState.getType()) &&
             "batch reuse must never target thread_local-wrapped states");

      EffectiveGroupRewriteFlags flags = flagsByCacheKey.lookup(g.cacheKey);
      donorGroups.push_back(DonorGroup{
         &g, donor->query, donor->state,
         CacheTarget{donorTargetState, g.cacheKey, flags.enableFilterPredReuse}});

      for (const CrossQueryStateMatchEntry& e : g.entries) {
         if (e.query < 0 || static_cast<size_t>(e.query) >= queries.size()) continue;
         mlir::Value targetState = resolveCacheTargetStateForReuse(e.state, reuseEarly[e.query]);
         assert(targetState && "batch reuse consumer target must resolve");
         assert(!mlir::isa<ThreadLocalType>(targetState.getType()) &&
                "batch reuse must never target thread_local-wrapped states");
         unsigned splitSlot = e.reuseSlot == std::numeric_limits<unsigned>::max()
                                 ? static_cast<unsigned>(e.query)
                                 : e.reuseSlot;
         uint64_t targetCacheKey = g.requiresSplitMaterialize
                                      ? splitMaterializeOutputCacheKey(g.cacheKey, splitSlot)
                                      : g.cacheKey;
         targetsByQuery[e.query].push_back(CacheTarget{targetState, targetCacheKey, flags.enableFilterPredReuse});
         consumerSlotByCacheKeyAndQuery[g.cacheKey][static_cast<unsigned>(e.query)] =
            mlir::isa<subop::PreAggrHtType>(targetState.getType())
               ? splitSlot
               : static_cast<unsigned>(e.query);
      }
   }
   inheritConsumerSlotsFromGroupDeps(rewriteGroups, consumerSlotByCacheKeyAndQuery);

   for (size_t i = 0; i < targetsByQuery.size(); ++i) {
      res.numTargetsPerQuery[i] = targetsByQuery[i].size();
      res.numTargetsNoTablePerQuery[i] = countNoTable(targetsByQuery[i]);
   }
   if (donorGroups.empty()) return res;

   mlir::ModuleOp firstQuery = queries.front();
   mlir::MLIRContext* ctx = firstQuery.getContext();
   res.synthetic = mlir::OwningOpRef<mlir::ModuleOp>(mlir::ModuleOp::create(mlir::UnknownLoc::get(ctx)));

   auto firstGroup = getSingleExecutionGroup(firstQuery);
   auto firstMain = firstQuery.lookupSymbol<mlir::func::FuncOp>("main");
   assert(firstMain && "expected func @main");

   auto synthMain = mlir::func::FuncOp::create(firstMain.getLoc(), "main", firstMain.getFunctionType());
   res.synthetic->push_back(synthMain);
   auto* entry = synthMain.addEntryBlock();
   mlir::OpBuilder fb = mlir::OpBuilder::atBlockBegin(entry);
   auto synthGroup = fb.create<ExecutionGroupOp>(firstGroup.getLoc(), mlir::TypeRange{}, mlir::ValueRange{});
   auto& synthBlock = synthGroup.getSubOps().emplaceBlock();
   mlir::OpBuilder gb = mlir::OpBuilder::atBlockBegin(&synthBlock);
   gb.create<ExecutionGroupReturnOp>(firstGroup.getLoc(), mlir::ValueRange{});
   fb.create<mlir::func::ReturnOp>(firstMain.getLoc());

   llvm::SmallVector<CacheTarget, 64> targetsSynthetic;
   llvm::SmallVector<mlir::IRMapping, 8> donorMappings(queries.size());

   for (size_t qi = 0; qi < queries.size(); ++qi) {
      llvm::SmallVector<CacheTarget, 16> donorTargets;
      for (const DonorGroup& dg : donorGroups) {
         if (static_cast<size_t>(dg.donorQuery) == qi) donorTargets.push_back(dg.donorTarget);
      }
      if (donorTargets.empty()) continue;

      ExecutionGroupOp donorGroup = getSingleExecutionGroup(queries[qi]);
      llvm::DenseSet<mlir::Value> neededStates = expandNeededStatesFromTargets(donorTargets, reuseEarly[qi]);
      llvm::SmallVector<ExecutionStepOp, 32> stepsToClone =
         collectCreateAndWriteStepsForStates(donorGroup, reuseEarly[qi], neededStates);
      stepsToClone = augmentStepsWithOperandProducerClosure(donorGroup, stepsToClone);
      donorMappings[qi] = cloneExecutionStepsToQuery0(synthGroup, stepsToClone);

      for (const CacheTarget& t : donorTargets) {
         mlir::Value mapped = donorMappings[qi].lookupOrNull(t.state);
         if (!mapped && !t.state.getDefiningOp()) {
            if (auto itW = reuseEarly[qi].writerStepsByState.find(t.state);
                itW != reuseEarly[qi].writerStepsByState.end() && !itW->second.empty()) {
               auto lastWriter = itW->second.back();
               if (lastWriter.getNumResults() == 1) mapped = donorMappings[qi].lookupOrNull(lastWriter.getResult(0));
            }
         }
         assert(mapped && "batch reuse target must map into synthetic module");
         targetsSynthetic.push_back(CacheTarget{mapped, t.cacheKey, t.enableFilterPredReuse});
      }
   }

   res.numTargetsSyntheticMapped = targetsSynthetic.size();
   res.numTargetsSyntheticMappedNoTable = countNoTable(targetsSynthetic);
   if (targetsSynthetic.empty()) return res;

   CachedJoinBufferLayoutsByKey producerLayoutsByKey;
   CachedAggregateLayoutsByKey aggregateLayoutsByKey;

   extendSyntheticJoinBuffersToColumnUnionForGroups(*res.synthetic, queries, rewriteGroups, targetsSynthetic,
                                                    &producerLayoutsByKey);

   extendSyntheticAggregateHashTablesToPayloadUnionForGroups(*res.synthetic, queries, rewriteGroups,
                                                             targetsSynthetic, &aggregateLayoutsByKey);

   expandSplitMaterializeTargetsInSynthetic(*res.synthetic, queries, rewriteGroups, reuseEarly,
                                            donorMappings, consumerSlotByCacheKeyAndQuery, targetsSynthetic);

   llvm::DenseMap<uint64_t, llvm::SmallVector<uint64_t, 4>> inheritedMixedDepsByCacheKey;
   auto reuseSyntheticAfterLayoutPrep = collectModuleReuseInfo(*res.synthetic);
   ClonedJoinBufferBuildSitesByKey joinBuildSites =
      recordClonedJoinBufferBuildSites(*res.synthetic, targetsSynthetic, reuseSyntheticAfterLayoutPrep);
   insertSyntheticFilterPredsAfterColumnUnionForGroups(*res.synthetic, queries, rewriteGroups, targetsSynthetic,
                                                       producerLayoutsByKey, joinBuildSites);

   {
      auto reuseSyntheticAfterLayout = collectModuleReuseInfo(*res.synthetic);
      refreshClonedJoinBufferBuildSiteStates(*res.synthetic, joinBuildSites, reuseSyntheticAfterLayout);
      llvm::SmallVector<CacheTarget, 64> cachePutTargets;
      cachePutTargets.reserve(targetsSynthetic.size());
      for (const CacheTarget& t : targetsSynthetic) {
         mlir::Value state = t.state;
         if (auto it = joinBuildSites.find(t.cacheKey); it != joinBuildSites.end())
            state = it->second.syntheticHiv;
         cachePutTargets.push_back(CacheTarget{state, t.cacheKey, /*enableFilterPredReuse=*/false});
      }
      insertCachePutsForTargets(*res.synthetic, cachePutTargets, &reuseSyntheticAfterLayout);
      extendSyntheticJoinBuffersWithInheritedMixedPreds(*res.synthetic, cachePutTargets, &producerLayoutsByKey,
                                                        &inheritedMixedDepsByCacheKey);
      inheritConsumerSlotsFromSingleMixedDep(inheritedMixedDepsByCacheKey, consumerSlotByCacheKeyAndQuery);
      inheritConsumerSlotsFromGroupDeps(rewriteGroups, consumerSlotByCacheKeyAndQuery);
      refreshCachedJoinLayoutsFromSyntheticCachePuts(*res.synthetic, targetsSynthetic, producerLayoutsByKey);
      for (const CacheTarget& t : cachePutTargets) {
         if (auto it = producerLayoutsByKey.find(t.cacheKey); it != producerLayoutsByKey.end()) {
            alignConsumerModulesToCachedJoinLayout(*res.synthetic, it->second, t.cacheKey, std::nullopt, nullptr);
            resyncConsumerCachedHivCarrierTypesFromCacheGet(*res.synthetic, t.cacheKey);
         }
      }
   }

   for (size_t qi = 0; qi < queries.size(); ++qi) {
      injectCacheGetsAndDeleteConstructionSteps(queries[qi], targetsByQuery[qi], &reuseEarly[qi],
                                                /*joinBufferHashmapLayoutAlreadyApplied=*/true,
                                                /*joinBufferWritePredAlreadyApplied=*/true);
      for (const CacheTarget& t : targetsByQuery[qi]) {
         if (auto it = producerLayoutsByKey.find(t.cacheKey); it != producerLayoutsByKey.end()) {
            unsigned slot = static_cast<unsigned>(qi);
            if (auto itByQuery = consumerSlotByCacheKeyAndQuery.find(t.cacheKey);
                itByQuery != consumerSlotByCacheKeyAndQuery.end()) {
               if (auto itSlot = itByQuery->second.find(static_cast<unsigned>(qi));
                   itSlot != itByQuery->second.end()) {
                  slot = itSlot->second;
               }
            }
            alignConsumerModulesToCachedJoinLayout(queries[qi], it->second, t.cacheKey, slot, nullptr);
         }
      }
      for (const CacheTarget& t : targetsByQuery[qi]) {
         resyncConsumerCachedHivCarrierTypesFromCacheGet(queries[qi], t.cacheKey);
      }
      recordConsumerMixedCacheGetSlots(queries[qi], static_cast<unsigned>(qi), consumerSlotByCacheKeyAndQuery);
   }
   inheritConsumerSlotsFromSingleMixedDep(inheritedMixedDepsByCacheKey, consumerSlotByCacheKeyAndQuery);
   inheritConsumerSlotsFromGroupDeps(rewriteGroups, consumerSlotByCacheKeyAndQuery);

   llvm::SmallVector<llvm::SmallVector<ConsumerCacheGetProbeClosure, 4>, 8> probeClosuresByQuery(queries.size());
   for (size_t qi = 0; qi < queries.size(); ++qi) {
      for (const CacheTarget& t : targetsByQuery[qi]) {
         if (auto it = producerLayoutsByKey.find(t.cacheKey); it != producerLayoutsByKey.end()) {
            unsigned slot = static_cast<unsigned>(qi);
            if (auto itByQuery = consumerSlotByCacheKeyAndQuery.find(t.cacheKey);
                itByQuery != consumerSlotByCacheKeyAndQuery.end()) {
               if (auto itSlot = itByQuery->second.find(static_cast<unsigned>(qi));
                   itSlot != itByQuery->second.end()) {
                  slot = itSlot->second;
               }
            }
            std::optional<unsigned> consumerQ = slot;
            alignConsumerModulesToCachedJoinLayout(queries[qi], it->second, t.cacheKey, consumerQ,
                                                   &probeClosuresByQuery[qi]);
         }
         if (auto it = aggregateLayoutsByKey.find(t.cacheKey); it != aggregateLayoutsByKey.end()) {
            unsigned slot = static_cast<unsigned>(qi);
            if (auto itByQuery = consumerSlotByCacheKeyAndQuery.find(t.cacheKey);
                itByQuery != consumerSlotByCacheKeyAndQuery.end()) {
               if (auto itSlot = itByQuery->second.find(static_cast<unsigned>(qi));
                   itSlot != itByQuery->second.end()) {
                  slot = itSlot->second;
               }
            }
            alignConsumerModulesToCachedAggregateLayout(queries[qi], it->second, t.cacheKey,
                                                        static_cast<unsigned>(qi), slot);
         }
      }
      for (const CacheTarget& t : targetsByQuery[qi]) {
         resyncConsumerCachedHivCarrierTypesFromCacheGet(queries[qi], t.cacheKey);
      }
      for (ConsumerCacheGetProbeClosure& probe : probeClosuresByQuery[qi]) {
         finalizeConsumerCachedJoinProbeColumnAttrs(queries[qi], probe);
      }
      syncProbeGatherMappingsInModule(queries[qi]);
      applyProbePredFiltersForConsumerClosures(queries[qi], probeClosuresByQuery[qi]);
   }

   return res;
}

static void expandSplitMaterializeTargetsInSynthetic(
   mlir::ModuleOp synthetic,
   llvm::ArrayRef<mlir::ModuleOp> queries,
   llvm::ArrayRef<CrossQueryStateMatchGroup> groups,
   llvm::ArrayRef<ModuleReuseInfo> reuseEarly,
   llvm::MutableArrayRef<mlir::IRMapping> donorMappings,
   const llvm::DenseMap<uint64_t, llvm::DenseMap<unsigned, unsigned>>& consumerSlotByCacheKeyAndQuery,
   llvm::SmallVectorImpl<CacheTarget>& targetsSynthetic) {
   llvm::DenseMap<uint64_t, CacheTarget> groupTargetByKey;
   for (const CacheTarget& t : targetsSynthetic) {
      groupTargetByKey[t.cacheKey] = t;
   }

   llvm::DenseSet<uint64_t> splitGroupKeys;
   llvm::SmallVector<CacheTarget, 64> expanded;
   ModuleReuseInfo reuseSynthetic = collectModuleReuseInfo(synthetic);

   auto cloneCreateOnlyStepBefore = [](ExecutionStepOp beforeStep, ExecutionStepOp createStep,
                                       mlir::Value originalState) -> mlir::Value {
      mlir::IRMapping mapping;
      auto* cloned = createStep.getOperation()->clone(mapping);
      beforeStep->getBlock()->getOperations().insert(mlir::Block::iterator(beforeStep.getOperation()), cloned);
      mlir::Value mapped = mapping.lookupOrNull(originalState);
      if (!mapped && createStep.getNumResults() == 1) mapped = mlir::cast<ExecutionStepOp>(cloned).getResult(0);
      assert(mapped && "cloned create-only step must map the requested state");
      return mapped;
   };

   auto cloneMergeStepAfter = [](ExecutionStepOp afterStep, ExecutionStepOp mergeStep,
                                 mlir::Value originalShadow, mlir::Value syntheticShadow,
                                 mlir::Value originalFinal) -> mlir::Value {
      mlir::IRMapping mapping;
      mapping.map(originalShadow, syntheticShadow);
      auto* cloned = mergeStep.getOperation()->clone(mapping);
      afterStep->getBlock()->getOperations().insert(std::next(mlir::Block::iterator(afterStep.getOperation())),
                                                    cloned);
      mlir::Value mapped = mapping.lookupOrNull(originalFinal);
      if (!mapped && mergeStep.getNumResults() == 1) mapped = mlir::cast<ExecutionStepOp>(cloned).getResult(0);
      assert(mapped && "cloned merge step must map the requested final state");
      return mapped;
   };

   auto residualForOriginalTarget = [](mlir::Value target, const ModuleReuseInfo& reuse)
      -> std::optional<SplitResidualFilter> {
      mlir::Value buildState = target;
      if (auto itShadow = findReuseMap(reuse.mergedFromShadowState, target);
          itShadow != reuse.mergedFromShadowState.end()) {
         buildState = itShadow->second;
      }
      auto itWriter = findReuseMap(reuse.writerStepsByState, buildState);
      if (itWriter == reuse.writerStepsByState.end()) return std::nullopt;
      assert(itWriter->second.size() == 1 && "split-materialize target must have one materialize writer");
      subop::MaterializeOp mat = findUniqueMaterializeWritingState(itWriter->second.front(), buildState);
      return findResidualFilterBeforeMaterialize(itWriter->second.front(), mat);
   };

   for (const CrossQueryStateMatchGroup& g : groups) {
      if (!g.requiresSplitMaterialize) continue;
      splitGroupKeys.insert(g.cacheKey);
      auto itTarget = groupTargetByKey.find(g.cacheKey);
      assert(itTarget != groupTargetByKey.end() && "split-materialize group must have a cloned donor target");
      CacheTarget donorSyntheticTarget = itTarget->second;
      mlir::Value donorSyntheticBuildState = donorSyntheticTarget.state;
      if (auto itShadow = findReuseMap(reuseSynthetic.mergedFromShadowState, donorSyntheticTarget.state);
          itShadow != reuseSynthetic.mergedFromShadowState.end()) {
         donorSyntheticBuildState = itShadow->second;
      }
      ExecutionStepOp matStep = findUniqueMaterializeStepWritingState(synthetic, donorSyntheticBuildState);
      subop::MaterializeOp donorMat = findUniqueMaterializeWritingState(matStep, donorSyntheticBuildState);
      subop::ScanRefsOp scanRefs = findFirstScanRefsInStep(matStep);
      clearGetExternalFiltersForState(canonicalizeStateValueForReuse(scanRefs.getState()));
      ExecutionStepOp donorMaterializeOwnerStep = donorMat->getParentOfType<ExecutionStepOp>();
      assert(donorMaterializeOwnerStep && "split-materialize materialize op must be inside an execution_step");
      mlir::Value originalMaterializeStream = donorMat.getStream();
      std::optional<SplitResidualFilter> donorResidual =
         findResidualFilterBeforeMaterialize(donorMaterializeOwnerStep, donorMat);
      bool anyResidual = false;
      bool anyMissingResidual = false;
      llvm::StringSet<> residualFingerprints;
      for (const CrossQueryStateMatchEntry& e : g.entries) {
         assert(e.query >= 0 && static_cast<size_t>(e.query) < queries.size());
         mlir::Value entryTarget = resolveCacheTargetStateForReuse(e.state, reuseEarly[e.query]);
         assert(entryTarget && "split-materialize entry target must resolve");
         if (std::optional<SplitResidualFilter> residual =
                residualForOriginalTarget(entryTarget, reuseEarly[e.query])) {
            anyResidual = true;
            residualFingerprints.insert(residualFilterSemanticFingerprint(*residual));
         } else {
            anyMissingResidual = true;
         }
      }
      bool splitResidual =
         anyResidual && (anyMissingResidual || residualFingerprints.size() > 1);
      mlir::Value baseStream = (splitResidual && donorResidual) ? donorResidual->inputStream : originalMaterializeStream;
      auto colByName = materializeStreamColumnsByFilterName(matStep, donorMat);

      unsigned entryOrdinal = 0;
      for (const CrossQueryStateMatchEntry& e : g.entries) {
         assert(e.query >= 0 && static_cast<size_t>(e.query) < queries.size());
         unsigned slot = e.reuseSlot == std::numeric_limits<unsigned>::max()
                            ? static_cast<unsigned>(e.query)
                            : e.reuseSlot;
         unsigned upstreamPredSlot = entryOrdinal;
         ++entryOrdinal;
         bool foundUpstreamPredSlot = false;
         for (uint64_t depKey : g.cacheDeps) {
            auto itByQuery = consumerSlotByCacheKeyAndQuery.find(depKey);
            if (itByQuery == consumerSlotByCacheKeyAndQuery.end()) continue;
            auto itSlot = itByQuery->second.find(static_cast<unsigned>(e.query));
            if (itSlot == itByQuery->second.end()) continue;
            if (foundUpstreamPredSlot) {
               assert(upstreamPredSlot == itSlot->second &&
                      "split-materialize branch must not inherit conflicting upstream pred slots");
            }
            upstreamPredSlot = itSlot->second;
            foundUpstreamPredSlot = true;
         }
         uint64_t outputKey = splitMaterializeOutputCacheKey(g.cacheKey, slot);

         mlir::Value entryTarget = resolveCacheTargetStateForReuse(e.state, reuseEarly[e.query]);
         assert(entryTarget && "split-materialize entry target must resolve");
         mlir::Value syntheticFinalState = donorMappings[e.query].lookupOrNull(entryTarget);
         mlir::Value syntheticBuildState;
         if (syntheticFinalState) {
            if (auto itShadow = findReuseMap(reuseSynthetic.mergedFromShadowState, syntheticFinalState);
                itShadow != reuseSynthetic.mergedFromShadowState.end()) {
               syntheticBuildState = itShadow->second;
            } else {
               syntheticBuildState = syntheticFinalState;
            }
         } else {
            mlir::Value entryBuildState = entryTarget;
            if (auto itShadow = findReuseMap(reuseEarly[e.query].mergedFromShadowState, entryTarget);
                itShadow != reuseEarly[e.query].mergedFromShadowState.end()) {
               entryBuildState = itShadow->second;
            }
            auto itCreate = findReuseMap(reuseEarly[e.query].createOnlyStepForState, entryBuildState);
            assert(itCreate != reuseEarly[e.query].createOnlyStepForState.end() &&
                   "split-materialize peer build state needs a create-only step");
            syntheticBuildState = cloneCreateOnlyStepBefore(matStep, itCreate->second, entryBuildState);

            if (canonicalizeStateValueForReuse(entryBuildState) == canonicalizeStateValueForReuse(entryTarget)) {
               syntheticFinalState = syntheticBuildState;
            } else {
               auto itMerge = findReuseMap(reuseEarly[e.query].writerStepsByState, entryTarget);
               assert(itMerge != reuseEarly[e.query].writerStepsByState.end() && itMerge->second.size() == 1 &&
                      "split-materialize peer final state needs one merge writer");
               syntheticFinalState =
                  cloneMergeStepAfter(matStep, itMerge->second.front(), entryBuildState, syntheticBuildState, entryTarget);
            }
         }

         mlir::Value stateArg = threadStateToNestedMaterializeStep(donorMaterializeOwnerStep, syntheticBuildState);
         llvm::SmallVector<CacheTarget, 1> filterTarget{CacheTarget{entryTarget, outputKey, false}};
         llvm::DenseMap<mlir::Value, llvm::SmallVector<runtime::FilterDescription, 8>> decodedByTarget =
            decodeFiltersByCacheTargets(filterTarget, reuseEarly[e.query]);
         llvm::ArrayRef<runtime::FilterDescription> filters;
         if (auto itFilters = decodedByTarget.find(entryTarget); itFilters != decodedByTarget.end()) {
            filters = itFilters->second;
         }
         llvm::DenseMap<llvm::StringRef, tuples::ColumnRefAttr> branchColByName = colByName;
         for (const runtime::FilterDescription& f : filters) {
            if (branchColByName.contains(f.columnName)) continue;
            llvm::StringRef name(f.columnName);
            size_t dollar = name.find('$');
            if (dollar == llvm::StringRef::npos) continue;
            auto itBase = branchColByName.find(name.take_front(dollar));
            assert(itBase != branchColByName.end() &&
                   "split-materialize branch filter column must exist on shared stream");
            branchColByName[f.columnName] = itBase->second;
         }
         assertRuntimeFiltersAvailableOnStream(branchColByName, filters);
         mlir::OpBuilder predBuilder(donorMat);
         predBuilder.setInsertionPoint(donorMat);
         mlir::Value branchStream = materializeRuntimeFiltersAsSubopFilter(
            predBuilder, donorMat.getLoc(), baseStream, branchColByName, filters);
         if (splitResidual) {
            std::optional<SplitResidualFilter> residual = residualForOriginalTarget(entryTarget, reuseEarly[e.query]);
            if (!residual) {
               // One-sided residual: this query keeps the whole shared stream.
            } else {
               branchStream = cloneResidualFilterBranch(predBuilder, donorMat.getLoc(), branchStream,
                                                        branchColByName, *residual);
               if (donorResidual) {
                  branchStream = cloneLinearStreamSuffixBefore(predBuilder, donorResidual->filter.getRes(),
                                                               originalMaterializeStream, branchStream);
               }
            }
         }
         std::string slotPredName = ("filter_pred$" + llvm::Twine(upstreamPredSlot)).str();
         branchStream = filterSplitBranchByMixedPredMember(
            predBuilder, donorMat.getLoc(), donorMaterializeOwnerStep, branchStream, slotPredName);
         if (canonicalizeStateValueForReuse(stateArg) != canonicalizeStateValueForReuse(donorMat.getState())) {
            mlir::OpBuilder b(donorMat);
            b.setInsertionPointAfter(donorMat);
            auto mapping = remapMaterializeMappingMembersByOrdinal(
               donorMat.getContext(), donorMat.getMapping(), stateArg.getType());
            b.create<subop::MaterializeOp>(donorMat.getLoc(), branchStream, stateArg,
                                           mapping);
         } else {
            donorMat->setOperand(0, branchStream);
         }
         expanded.push_back(CacheTarget{syntheticFinalState, outputKey, /*enableFilterPredReuse=*/false});
      }
   }

   if (splitGroupKeys.empty()) return;
   llvm::SmallVector<CacheTarget, 64> kept;
   kept.reserve(targetsSynthetic.size() + expanded.size());
   for (const CacheTarget& t : targetsSynthetic) {
      if (splitGroupKeys.contains(t.cacheKey)) continue;
      kept.push_back(t);
   }
   kept.append(expanded.begin(), expanded.end());
   targetsSynthetic.assign(kept.begin(), kept.end());
}

static llvm::DenseMap<uint64_t, mlir::Value> collectCachePutStatesByKey(mlir::ModuleOp module) {
   llvm::DenseMap<uint64_t, mlir::Value> out;
   module.walk([&](subop::CachePutOp put) {
      uint64_t key = static_cast<uint64_t>(put.getKey());
      mlir::Value state = put.getState();
      if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(state)) {
         if (auto ownerStep = mlir::dyn_cast_or_null<ExecutionStepOp>(blockArg.getOwner()->getParentOp())) {
            if (blockArg.getArgNumber() < ownerStep.getNumOperands()) {
               state = ownerStep.getOperand(blockArg.getArgNumber());
            }
         }
      }
      auto it = out.find(key);
      if (it == out.end()) {
         out.try_emplace(key, state);
         return;
      }
      assert(it->second == state &&
             "synthetic module must not cache_put different states for the same key");
   });
   return out;
}

static std::optional<unsigned> parseFilterPredMemberSlotLocal(llvm::StringRef memberName) {
   if (!memberName.consume_front("filter_pred$")) return std::nullopt;
   unsigned slot = 0;
   if (memberName.getAsInteger(10, slot)) return std::nullopt;
   return slot;
}

static subop::HashIndexedViewType asHashIndexedViewLayoutTypeLocal(mlir::Type type) {
   if (auto hiv = mlir::dyn_cast<subop::HashIndexedViewType>(type)) return hiv;
   if (auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(type)) {
      return subop::HashIndexedViewType::get(mixed.getContext(), mixed.getKeyMembers(), mixed.getValueMembers(),
                                             mixed.getCompareHashForLookup());
   }
   return nullptr;
}

static std::optional<unsigned> highestFilterPredSlot(subop::HashIndexedViewType hiv) {
   if (!hiv) return std::nullopt;
   auto* ctx = hiv.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   std::optional<unsigned> bestSlot;
   for (subop::Member m : hiv.getValueMembers().getMembers()) {
      llvm::StringRef name = mm.getName(m);
      auto slot = parseFilterPredMemberSlotLocal(name);
      if (!slot) continue;
      if (!bestSlot || *slot > *bestSlot) bestSlot = *slot;
   }
   return bestSlot;
}

static std::optional<std::pair<uint64_t, mlir::Value>>
singleCacheGetReturnedByStep(ExecutionStepOp step) {
   subop::CacheGetOp get;
   bool multiple = false;
   step.walk([&](subop::CacheGetOp op) {
      if (get) {
         multiple = true;
         return mlir::WalkResult::interrupt();
      }
      get = op;
      return mlir::WalkResult::advance();
   });
   if (!get || multiple) return std::nullopt;
   if (step.getNumResults() != 1) return std::nullopt;
   auto& body = step.getSubOps().front();
   auto ret = mlir::dyn_cast<subop::ExecutionStepReturnOp>(body.getTerminator());
   if (!ret || ret.getNumOperands() != 1 || ret.getOperand(0) != get.getRes()) return std::nullopt;
   return std::make_pair(static_cast<uint64_t>(get.getKey()), get.getRes());
}

static void appendSyntheticProducerSteps(mlir::ModuleOp dst, mlir::ModuleOp src) {
   ExecutionGroupOp dstGroup = getSingleExecutionGroup(dst);
   ExecutionGroupOp srcGroup = getSingleExecutionGroup(src);
   mlir::Block& dstBlock = dstGroup.getSubOps().front();
   mlir::Block& srcBlock = srcGroup.getSubOps().front();
   mlir::Operation* dstTerminator = dstBlock.getTerminator();
   assert(dstTerminator && "synthetic execution_group must have a terminator");

   llvm::DenseMap<uint64_t, mlir::Value> cachedStateByKey = collectCachePutStatesByKey(dst);
   mlir::IRMapping mapping;
   auto operandsAreMapped = [&](mlir::Operation& op) {
      for (mlir::Value operand : op.getOperands()) {
         if (!mapping.lookupOrNull(operand)) return false;
      }
      return true;
   };
   for (mlir::Operation& op : srcBlock.without_terminator()) {
      if (auto step = mlir::dyn_cast<ExecutionStepOp>(&op)) {
         if (auto cacheGet = singleCacheGetReturnedByStep(step)) {
            auto it = cachedStateByKey.find(cacheGet->first);
            if (it != cachedStateByKey.end()) {
               if (!asHashIndexedViewLayoutTypeLocal(it->second.getType()) &&
                   it->second.getType() == cacheGet->second.getType()) {
                  mapping.map(cacheGet->second, it->second);
                  mapping.map(step.getResult(0), it->second);
                  continue;
               }
            }
         }
      }
      if (!operandsAreMapped(op)) {
         llvm_unreachable("synthetic producer append requires mapped operands");
      }
      auto* cloned = op.clone(mapping);
      dstBlock.getOperations().insert(mlir::Block::iterator(dstTerminator), cloned);
   }
}

static CachedJoinBufferLayout cachedJoinLayoutFromHiv(subop::HashIndexedViewType hiv) {
   CachedJoinBufferLayout layout;
   layout.producerHiv = hiv;
   auto& mm = hiv.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   for (subop::Member m : hiv.getValueMembers().getMembers()) {
      layout.payloadMembers.push_back(m);
      layout.payloadColumnTypes.push_back(mm.getType(m));
      llvm::StringRef memberName = mm.getName(m);
      if (auto slot = parseFilterPredMemberSlotLocal(memberName)) {
         std::string semKey = "reuse_filter_pred";
         semKey.push_back('\x1f');
         semKey += llvm::Twine(*slot).str();
         layout.payloadSemanticKeys.push_back(std::move(semKey));
      } else {
         layout.payloadSemanticKeys.push_back(std::string(memberName));
      }
   }
   return layout;
}

static void alignSyntheticCacheGetDependenciesToCachedPuts(mlir::ModuleOp synthetic) {
   llvm::DenseMap<uint64_t, subop::HashIndexedViewType> cachedHivByKey;
   synthetic.walk([&](subop::CachePutOp put) {
      if (auto hiv = asHashIndexedViewLayoutTypeLocal(put.getState().getType())) {
         cachedHivByKey[static_cast<uint64_t>(put.getKey())] = hiv;
      }
   });
   if (cachedHivByKey.empty()) return;

   llvm::DenseSet<uint64_t> keysToAlign;
   synthetic.walk([&](subop::CacheGetOp get) {
      uint64_t key = static_cast<uint64_t>(get.getKey());
      if (cachedHivByKey.contains(key)) keysToAlign.insert(key);
   });
   if (keysToAlign.empty()) return;

   llvm::SmallVector<ConsumerCacheGetProbeClosure, 8> probeClosures;
   for (uint64_t key : keysToAlign) {
      subop::HashIndexedViewType hiv = cachedHivByKey.lookup(key);
      std::optional<unsigned> predSlot = highestFilterPredSlot(hiv);
      if (!predSlot || *predSlot < 2) continue;
      CachedJoinBufferLayout layout = cachedJoinLayoutFromHiv(hiv);
      alignConsumerModulesToCachedJoinLayout(synthetic, layout, key, predSlot, &probeClosures);
      setMixedLookupPredSlotForCacheGet(synthetic, key, *predSlot);
      resyncConsumerCachedHivCarrierTypesFromCacheGet(synthetic, key);
   }
   if (probeClosures.empty()) return;
   for (ConsumerCacheGetProbeClosure& probe : probeClosures) {
      finalizeConsumerCachedJoinProbeColumnAttrs(synthetic, probe);
   }
   syncProbeGatherMappingsInModule(synthetic);
   applyProbePredFiltersForConsumerClosures(synthetic, probeClosures);
}

BatchReusePlanRewriteResult rewritePlansWithSyntheticQueryBatch(
   llvm::ArrayRef<mlir::ModuleOp> queries,
   llvm::ArrayRef<CrossQueryStateMatchGroup> groups,
   lingodb::catalog::Catalog* catalog) {
   BatchReusePlanRewriteResult total;
   total.numTargetsPerQuery.resize(queries.size(), 0);
   total.numTargetsNoTablePerQuery.resize(queries.size(), 0);

   llvm::SmallVector<CrossQueryStateMatchGroup, 64> currentGroups(groups.begin(), groups.end());
   constexpr unsigned maxFixedPointIterations = 8;
   for (unsigned iter = 0; iter < maxFixedPointIterations; ++iter) {
      if (currentGroups.empty()) break;
      BatchReusePlanRewriteResult one =
         rewritePlansWithSyntheticQueryBatchOnce(queries, currentGroups, catalog);
      bool madeProgress = one.numTargetsSyntheticMapped > 0;
      if (!madeProgress) break;

      for (size_t qi = 0; qi < queries.size(); ++qi) {
         if (qi < one.numTargetsPerQuery.size()) total.numTargetsPerQuery[qi] += one.numTargetsPerQuery[qi];
         if (qi < one.numTargetsNoTablePerQuery.size())
            total.numTargetsNoTablePerQuery[qi] += one.numTargetsNoTablePerQuery[qi];
      }
      total.numTargetsSyntheticMapped += one.numTargetsSyntheticMapped;
      total.numTargetsSyntheticMappedNoTable += one.numTargetsSyntheticMappedNoTable;

      if (one.synthetic) {
         if (!total.synthetic) {
            total.synthetic = std::move(one.synthetic);
         } else {
            appendSyntheticProducerSteps(*total.synthetic, *one.synthetic);
         }
         alignSyntheticCacheGetDependenciesToCachedPuts(*total.synthetic);
      }

      llvm::SmallVector<std::pair<int, mlir::ModuleOp>, 8> qmods;
      qmods.reserve(queries.size());
      for (size_t qi = 0; qi < queries.size(); ++qi) {
         qmods.push_back({static_cast<int>(qi), queries[qi]});
      }
      currentGroups = collectCrossQueryStateMatchGroups(qmods);
   }

   return total;
}

} // namespace lingodb::compiler::dialect::subop
