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

static std::optional<unsigned> parseFilterPredMemberSlot(llvm::StringRef memberName) {
   if (!memberName.consume_front("filter_pred$")) return std::nullopt;
   unsigned slot = 0;
   if (memberName.getAsInteger(10, slot)) return std::nullopt;
   return slot;
}

static void inheritConsumerSlotsFromDeps(
   uint64_t targetKey, llvm::ArrayRef<uint64_t> deps,
   llvm::DenseMap<uint64_t, llvm::DenseMap<unsigned, unsigned>>& consumerSlotByCacheKeyAndQuery,
   bool includeTargetSlotInSignature) {
   if (deps.empty()) return;
   if (consumerSlotByCacheKeyAndQuery.contains(targetKey)) return;
   llvm::SmallVector<unsigned, 8> queries;
   llvm::DenseSet<unsigned> seenQueries;
   auto addQueries = [&](const llvm::DenseMap<unsigned, unsigned>& slots) {
      for (const auto& [queryIdx, slot] : slots) {
         (void)slot;
         if (seenQueries.insert(queryIdx).second) queries.push_back(queryIdx);
      }
   };
   auto itTarget = consumerSlotByCacheKeyAndQuery.find(targetKey);
   if (includeTargetSlotInSignature && itTarget != consumerSlotByCacheKeyAndQuery.end())
      addQueries(itTarget->second);
   for (uint64_t depKey : deps) {
      auto itSlots = consumerSlotByCacheKeyAndQuery.find(depKey);
      if (itSlots == consumerSlotByCacheKeyAndQuery.end()) continue;
      addQueries(itSlots->second);
   }
   if (queries.empty()) return;
   llvm::sort(queries);

   llvm::StringMap<unsigned> slotBySignature;
   llvm::DenseMap<unsigned, unsigned> mergedSlots;
   unsigned nextSlot = 0;
   for (unsigned queryIdx : queries) {
      std::string sig;
      llvm::raw_string_ostream os(sig);
      if (includeTargetSlotInSignature && itTarget != consumerSlotByCacheKeyAndQuery.end()) {
         if (auto itSlot = itTarget->second.find(queryIdx); itSlot != itTarget->second.end())
            os << "self:" << itSlot->second << '|';
      }
      for (uint64_t depKey : deps) {
         auto itSlots = consumerSlotByCacheKeyAndQuery.find(depKey);
         if (itSlots == consumerSlotByCacheKeyAndQuery.end()) continue;
         auto itSlot = itSlots->second.find(queryIdx);
         if (itSlot == itSlots->second.end()) continue;
         os << "dep:" << depKey << ':' << itSlot->second << '|';
      }
      os.flush();
      if (sig.empty()) continue;
      auto itSlot = slotBySignature.find(sig);
      if (itSlot == slotBySignature.end())
         itSlot = slotBySignature.try_emplace(sig, nextSlot++).first;
      mergedSlots[queryIdx] = itSlot->second;
   }
   if (!mergedSlots.empty()) consumerSlotByCacheKeyAndQuery[targetKey] = std::move(mergedSlots);
}

static void inheritConsumerSlotsFromSingleMixedDep(
   const llvm::DenseMap<uint64_t, llvm::SmallVector<uint64_t, 4>>& inheritedDepsByCacheKey,
   llvm::DenseMap<uint64_t, llvm::DenseMap<unsigned, unsigned>>& consumerSlotByCacheKeyAndQuery) {
   for (const auto& [targetKey, deps] : inheritedDepsByCacheKey)
      inheritConsumerSlotsFromDeps(targetKey, deps, consumerSlotByCacheKeyAndQuery,
                                   /*includeTargetSlotInSignature=*/false);
}

static void inheritConsumerSlotsFromGroupDeps(
   llvm::ArrayRef<CrossQueryStateMatchGroup> groups,
   llvm::DenseMap<uint64_t, llvm::DenseMap<unsigned, unsigned>>& consumerSlotByCacheKeyAndQuery) {
   for (const CrossQueryStateMatchGroup& group : groups)
      inheritConsumerSlotsFromDeps(group.cacheKey, group.cacheDeps, consumerSlotByCacheKeyAndQuery,
                                   /*includeTargetSlotInSignature=*/true);
}

static void recordConsumerMixedCacheGetSlots(
   mlir::ModuleOp module, unsigned queryIdx,
   llvm::DenseMap<uint64_t, llvm::DenseMap<unsigned, unsigned>>& consumerSlotByCacheKeyAndQuery) {
   module.walk([&](subop::CacheGetOp get) {
      auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(get.getResult().getType());
      if (!mixed) return;
      auto slot = parseFilterPredMemberSlot(mixed.getFilterPredMemberName().getValue());
      assert(slot && "mixed cache_get must select a filter_pred$N member");
      consumerSlotByCacheKeyAndQuery[static_cast<uint64_t>(get.getKey())].try_emplace(queryIdx, *slot);
   });
}

static unsigned consumerSlotForCacheKeyQuery(
   const llvm::DenseMap<uint64_t, llvm::DenseMap<unsigned, unsigned>>& consumerSlotByCacheKeyAndQuery,
   uint64_t cacheKey, unsigned queryIdx) {
   if (auto itByQuery = consumerSlotByCacheKeyAndQuery.find(cacheKey);
       itByQuery != consumerSlotByCacheKeyAndQuery.end()) {
      if (auto itSlot = itByQuery->second.find(queryIdx); itSlot != itByQuery->second.end())
         return itSlot->second;
   }
   return queryIdx;
}

struct ReuseRewriteContext {
   llvm::DenseMap<uint64_t, llvm::DenseMap<unsigned, unsigned>> consumerSlotByCacheKeyAndQuery;
   llvm::DenseMap<uint64_t, llvm::SmallVector<uint64_t, 4>> inheritedMixedDepsByCacheKey;

   void recordConsumerSlotsFromModule(mlir::ModuleOp module, unsigned queryIdx) {
      recordConsumerMixedCacheGetSlots(module, queryIdx, consumerSlotByCacheKeyAndQuery);
   }

   void setConsumerSlot(uint64_t cacheKey, unsigned queryIdx, unsigned slot) {
      consumerSlotByCacheKeyAndQuery[cacheKey][queryIdx] = slot;
   }

   uint64_t targetCacheKeyForSlot(uint64_t groupCacheKey, bool usesSplitMaterialize, unsigned slot) const {
      return usesSplitMaterialize ? splitMaterializeOutputCacheKey(groupCacheKey, slot) : groupCacheKey;
   }

   uint64_t registerConsumerTargetSlot(uint64_t groupCacheKey, bool usesSplitMaterialize,
                                       unsigned queryIdx, unsigned slot) {
      uint64_t targetCacheKey = targetCacheKeyForSlot(groupCacheKey, usesSplitMaterialize, slot);
      setConsumerSlot(groupCacheKey, queryIdx, slot);
      setConsumerSlot(targetCacheKey, queryIdx, slot);
      return targetCacheKey;
   }

   void inheritSlotsFromGroups(llvm::ArrayRef<CrossQueryStateMatchGroup> groups) {
      inheritConsumerSlotsFromGroupDeps(groups, consumerSlotByCacheKeyAndQuery);
   }

   void inheritSlotsFromSingleMixedDeps() {
      inheritConsumerSlotsFromSingleMixedDep(inheritedMixedDepsByCacheKey, consumerSlotByCacheKeyAndQuery);
   }

   unsigned consumerSlot(uint64_t cacheKey, unsigned queryIdx) const {
      return consumerSlotForCacheKeyQuery(consumerSlotByCacheKeyAndQuery, cacheKey, queryIdx);
   }

   std::optional<unsigned> lookupConsumerSlot(uint64_t cacheKey, unsigned queryIdx) const {
      auto itByQuery = consumerSlotByCacheKeyAndQuery.find(cacheKey);
      if (itByQuery == consumerSlotByCacheKeyAndQuery.end()) return std::nullopt;
      auto itSlot = itByQuery->second.find(queryIdx);
      if (itSlot == itByQuery->second.end()) return std::nullopt;
      return itSlot->second;
   }

   std::optional<unsigned> inheritedConsumerSlot(llvm::ArrayRef<uint64_t> depKeys, unsigned queryIdx) const {
      for (uint64_t depKey : depKeys) {
         if (std::optional<unsigned> slot = lookupConsumerSlot(depKey, queryIdx)) return slot;
      }
      return std::nullopt;
   }
};

static unsigned consumerSlotForJoinLayout(const ReuseRewriteContext& rewriteCtx,
                                          const CachedJoinBufferLayout& layout,
                                          uint64_t cacheKey, unsigned queryIdx) {
   (void)layout;
   return rewriteCtx.consumerSlot(cacheKey, queryIdx);
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

   llvm::SmallVector<mlir::Operation*, 8> nestedPortOps;
   for (mlir::Operation* op = materializeStep->getParentOp(); op && op != topStep.getOperation();
        op = op->getParentOp()) {
      if (mlir::isa<NestedExecutionGroupOp, ExecutionStepOp>(op)) nestedPortOps.push_back(op);
   }
   for (mlir::Operation* op : llvm::reverse(nestedPortOps)) {
      if (auto neg = mlir::dyn_cast<NestedExecutionGroupOp>(op)) {
         current = ensureNestedExecutionGroupInput(neg, current);
      } else {
         current = ensureExecutionStepInput(mlir::cast<ExecutionStepOp>(op), current);
      }
   }
   return ensureExecutionStepInput(materializeStep, current);
}

static bool valueDefinedInSameBlockBefore(mlir::Value value, mlir::Operation* op) {
   mlir::Operation* def = value.getDefiningOp();
   return def && def->getBlock() == op->getBlock() && def->isBeforeInBlock(op);
}

static mlir::Value stateToNestedBuildStep(ExecutionStepOp buildStep, mlir::Value state) {
   if (valueDefinedInSameBlockBefore(state, buildStep.getOperation()))
      return ensureExecutionStepInput(buildStep, state);
   return threadStateToNestedMaterializeStep(buildStep, state);
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

static subop::ScanRefsOp tryFindFirstScanRefsInStep(ExecutionStepOp step) {
   subop::ScanRefsOp found;
   step.walk([&](subop::ScanRefsOp scan) {
      if (!found) found = scan;
   });
   return found;
}

static std::string stripColumnReuseSuffixToString(llvm::StringRef s) {
   size_t dollar = s.find('$');
   return (dollar == llvm::StringRef::npos ? s : s.take_front(dollar)).str();
}

static bool isGenericNumberedMemberName(llvm::StringRef s) {
   if (!s.consume_front("member$")) return false;
   if (s.empty()) return false;
   for (char c : s) {
      if (c < '0' || c > '9') return false;
   }
   return true;
}

static std::string scopedColumnName(llvm::StringRef scope, llvm::StringRef leaf) {
   if (scope.empty()) return leaf.str();
   std::string out = scope.str();
   out += "::";
   out += leaf.str();
   return out;
}

static std::string fullNameForColumnRef(tuples::ColumnRefAttr ref) {
   auto& cm = ref.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto [scope, leaf] = cm.getName(&ref.getColumn());
   return scopedColumnName(scope, leaf);
}

static void addSplitColumnAliases(llvm::StringMap<tuples::ColumnRefAttr>& out,
                                  llvm::StringRef scope,
                                  llvm::StringRef leaf,
                                  tuples::ColumnRefAttr ref) {
   out[leaf] = ref;
   std::string full = scopedColumnName(scope, leaf);
   out[full] = ref;
   if (isGenericNumberedMemberName(leaf)) return;
   out[stripColumnReuseSuffixToString(leaf)] = ref;
   std::string strippedFull = scopedColumnName(scope, stripColumnReuseSuffixToString(leaf));
   out[strippedFull] = ref;
}

static llvm::StringMap<tuples::ColumnRefAttr>
materializeStreamColumnsByFilterName(ExecutionStepOp step, subop::MaterializeOp mat) {
   llvm::StringMap<tuples::ColumnRefAttr> out;
   auto& cm = mat.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = mat.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto addName = [&](llvm::StringRef name, tuples::ColumnRefAttr ref) {
      out[name] = ref;
      size_t dollar = name.find('$');
      if (dollar != llvm::StringRef::npos && !isGenericNumberedMemberName(name))
         out[name.take_front(dollar)] = ref;
   };
   for (auto& [member, colRef] : mat.getMapping().getMapping()) {
      (void)member;
      auto [scope, leaf] = cm.getName(&colRef.getColumn());
      addName(leaf, colRef);
      addSplitColumnAliases(out, scope, leaf, colRef);
   }
   step.walk([&](subop::GatherOp gather) {
      for (auto& [member, colDef] : gather.getMapping().getMapping()) {
         tuples::ColumnRefAttr ref = cm.createRef(&colDef.getColumn());
         addName(mm.getName(member), ref);
         auto [scope, leaf] = cm.getName(&colDef.getColumn());
         addName(leaf, ref);
         addSplitColumnAliases(out, scope, leaf, ref);
      }
   });
   return out;
}

static void assertRuntimeFiltersAvailableOnStream(
   const llvm::StringMap<tuples::ColumnRefAttr>& colByName,
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

static bool splitResidualFilterIsRelaxable(ExecutionStepOp step,
                                           subop::MapOp map,
                                           subop::FilterOp filter) {
   return map && filterConditionsComeFromMap(filter, map) &&
          splitResidualPredicateUsesOnlySourceMembers(step, map, filter);
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
             splitResidualFilterIsRelaxable(step, map, filter)) {
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

static std::string baseNameForColumnRef(tuples::ColumnRefAttr ref) {
   auto& cm = ref.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto [scope, leaf] = cm.getName(&ref.getColumn());
   (void)scope;
   llvm::StringRef name(leaf);
   if (isGenericNumberedMemberName(name)) return name.str();
   size_t dollar = name.find('$');
   return (dollar == llvm::StringRef::npos ? name : name.take_front(dollar)).str();
}

static std::string normalizeSplitColumnName(llvm::StringRef name) {
   size_t scopeSep = name.find("::");
   if (scopeSep != llvm::StringRef::npos) {
      std::string scope = normalizeSplitColumnName(name.take_front(scopeSep));
      std::string leaf = normalizeSplitColumnName(name.drop_front(scopeSep + 2));
      return scope.empty() ? leaf : (scope + "::" + leaf);
   }
   bool keepNumberedMemberSuffix = isGenericNumberedMemberName(name);
   size_t dollar = name.find('$');
   if (dollar != llvm::StringRef::npos && !keepNumberedMemberSuffix) name = name.take_front(dollar);
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
   const llvm::StringMap<tuples::ColumnRefAttr>& colByName,
   llvm::StringRef name) {
   if (auto it = colByName.find(name); it != colByName.end()) return it->second;
   std::string norm = normalizeSplitColumnName(name);
   tuples::ColumnRefAttr found;
   for (auto& kv : colByName) {
      if (normalizeSplitColumnName(kv.getKey()) != norm) continue;
      if (found && &found.getColumn() != &kv.second.getColumn()) return {};
      found = kv.second;
   }
   return found;
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
                                             const llvm::StringMap<tuples::ColumnRefAttr>& colByName,
                                             SplitResidualFilter residual,
                                             llvm::ArrayRef<tuples::ColumnRefAttr> inputOverrides = {},
                                             subop::ColumnMapping* columnMapping = nullptr) {
   assert((inputOverrides.empty() ||
           inputOverrides.size() == residual.predMap.getInputCols().size()) &&
          "split-materialize residual input overrides must align with map inputs");
   llvm::SmallVector<mlir::Attribute, 8> inputRefs;
   inputRefs.reserve(residual.predMap.getInputCols().size());
   for (auto [idx, attr] : llvm::enumerate(residual.predMap.getInputCols())) {
      tuples::ColumnRefAttr ref;
      if (!inputOverrides.empty()) {
         ref = inputOverrides[idx];
      } else {
         auto oldRef = mlir::cast<tuples::ColumnRefAttr>(attr);
         if (columnMapping) {
            tuples::ColumnRefAttr mapped = columnMapping->remap(oldRef);
            if (mapped.getColumnPtr() != oldRef.getColumnPtr()) ref = mapped;
         }
         if (!ref) ref = lookupSplitColumnByName(colByName, fullNameForColumnRef(oldRef));
         if (!ref) ref = lookupSplitColumnByName(colByName, baseNameForColumnRef(oldRef));
      }
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
   mlir::Block& clonedBody = clonedMap.getFn().front();
   (void)clonedBody;
   assert(clonedBody.getNumArguments() == inputRefs.size() &&
          "split-materialize residual clone must preserve one block argument per input column");
   for (auto [idx, attr] : llvm::enumerate(inputRefs)) {
      auto ref = mlir::cast<tuples::ColumnRefAttr>(attr);
      (void)ref;
      assert(clonedBody.getArgument(idx).getType() == ref.getColumn().type &&
             "split-materialize residual clone input type must match the selected stream column");
   }
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

static llvm::SmallVector<tuples::ColumnRefAttr, 8>
resolveResidualInputRefsOnCurrentStream(
   SplitResidualFilter residual,
   const llvm::StringMap<tuples::ColumnRefAttr>& colByName,
   subop::ColumnMapping* columnMapping = nullptr) {
   llvm::SmallVector<tuples::ColumnRefAttr, 8> inputRefs;
   inputRefs.reserve(residual.predMap.getInputCols().size());
   for (mlir::Attribute attr : residual.predMap.getInputCols()) {
      auto oldRef = mlir::cast<tuples::ColumnRefAttr>(attr);
      tuples::ColumnRefAttr ref;
      if (columnMapping) {
         tuples::ColumnRefAttr mapped = columnMapping->remap(oldRef);
         if (mapped.getColumnPtr() != oldRef.getColumnPtr()) ref = mapped;
      }
      if (!ref) ref = lookupSplitColumnByName(colByName, fullNameForColumnRef(oldRef));
      if (!ref) ref = lookupSplitColumnByName(colByName, baseNameForColumnRef(oldRef));
      assert(ref && "split-materialize residual template input column must exist on shared stream");
      inputRefs.push_back(ref);
   }
   return inputRefs;
}

static subop::MapOp cloneBranchMapWithDonorColumns(mlir::OpBuilder& b,
                                                   mlir::Value stream,
                                                   subop::MapOp donorMap,
                                                   subop::MapOp branchMap,
                                                   subop::ColumnMapping& columnMapping) {
   assert(donorMap.getInputCols().size() == branchMap.getInputCols().size() &&
          "split aggregate branch map replacement requires ordinal-aligned inputs");
   assert(donorMap.getComputedCols().size() == branchMap.getComputedCols().size() &&
          "split aggregate branch map replacement requires ordinal-aligned outputs");
   llvm::SmallVector<mlir::Attribute, 8> inputRefs;
   inputRefs.reserve(branchMap.getInputCols().size());
   for (mlir::Attribute attr : branchMap.getInputCols()) {
      auto ref = mlir::cast<tuples::ColumnRefAttr>(attr);
      inputRefs.push_back(columnMapping.remap(ref));
   }

   llvm::SmallVector<mlir::Attribute, 8> computedCols;
   computedCols.reserve(donorMap.getComputedCols().size());
   for (mlir::Attribute attr : donorMap.getComputedCols()) {
      auto def = mlir::cast<tuples::ColumnDefAttr>(attr);
      computedCols.push_back(columnMapping.clone(def));
   }

   auto clonedMap = b.create<subop::MapOp>(branchMap.getLoc(), stream,
                                           b.getArrayAttr(computedCols),
                                           b.getArrayAttr(inputRefs));
   mlir::IRMapping regionMapping;
   b.cloneRegionBefore(branchMap.getFn(), clonedMap.getFn(), clonedMap.getFn().begin(), regionMapping);
   return clonedMap;
}

static mlir::Value filterSplitBranchByMixedPredMember(mlir::OpBuilder& b,
                                                      mlir::Location loc,
                                                      mlir::Value stream,
                                                      llvm::StringRef predMemberName) {
   if (predMemberName.empty()) return stream;
   auto* ctx = stream.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

   struct PredSource {
      tuples::ColumnRefAttr ref;
      subop::Member member;
   };

   auto stateHasPredMember = [&](mlir::Type type, subop::Member& predMember) {
      if (auto sorted = mlir::dyn_cast<subop::SortedViewType>(type)) type = sorted.getBasedOn();
      if (auto tl = mlir::dyn_cast<subop::ThreadLocalType>(type)) type = tl.getWrapped();
      auto state = mlir::dyn_cast<subop::State>(type);
      if (!state) return false;
      for (subop::Member member : state.getMembers().getMembers()) {
         if (mm.getName(member) != predMemberName) continue;
         predMember = member;
         return true;
      }
      return false;
   };

   auto findPredSourceOnStreamChain = [&]() -> std::optional<PredSource> {
      mlir::Value cur = stream;
      for (;;) {
         mlir::Operation* def = cur.getDefiningOp();
         if (!def) return std::nullopt;
         if (auto scan = mlir::dyn_cast<subop::ScanRefsOp>(def)) {
            subop::Member predMember;
            if (!stateHasPredMember(scan.getState().getType(), predMember)) return std::nullopt;
            return PredSource{cm.createRef(&scan.getRef().getColumn()), predMember};
         }
         if (auto scan = mlir::dyn_cast<subop::ScanListOp>(def)) {
            auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(scan.getElem().getColumn().type);
            if (!ler) return std::nullopt;
            auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(ler.getState());
            if (!mixed) return std::nullopt;
            for (subop::Member member : mixed.getValueMembers().getMembers()) {
               if (mm.getName(member) == predMemberName)
                  return PredSource{cm.createRef(&scan.getElem().getColumn()), member};
            }
            return std::nullopt;
         }
         if (auto gather = mlir::dyn_cast<subop::GatherOp>(def)) {
            cur = gather.getStream();
            continue;
         }
         if (auto map = mlir::dyn_cast<subop::MapOp>(def)) {
            cur = map.getStream();
            continue;
         }
         if (auto filter = mlir::dyn_cast<subop::FilterOp>(def)) {
            cur = filter.getStream();
            continue;
         }
         if (auto rename = mlir::dyn_cast<subop::RenamingOp>(def)) {
            cur = rename.getStream();
            continue;
         }
         return std::nullopt;
      }
   };

   std::optional<PredSource> source = findPredSourceOnStreamChain();
   if (!source) return stream;

   tuples::ColumnDefAttr predDef = cm.createDef(cm.getUniqueScope("split_branch_pred"), "filter_pred");
   predDef.getColumn().type = mlir::IntegerType::get(ctx, 1);
   auto gather = b.create<subop::GatherOp>(
      loc, stream.getType(), stream, source->ref,
      subop::ColumnDefMemberMappingAttr::get(ctx, {{source->member, predDef}}));
   tuples::ColumnRefAttr predRef = cm.createRef(&predDef.getColumn());
   auto filter = b.create<subop::FilterOp>(loc, gather.getRes(), subop::FilterSemantic::all_true,
                                           b.getArrayAttr({predRef}));
   return filter.getRes();
}

static mlir::Value filterScanListBranchByMixedPredMember(mlir::OpBuilder& b,
                                                         mlir::Location loc,
                                                         mlir::Value stream,
                                                         llvm::StringRef predMemberName) {
   if (predMemberName.empty()) return stream;
   auto scan = mlir::dyn_cast_or_null<subop::ScanListOp>(stream.getDefiningOp());
   if (!scan) {
      return filterSplitBranchByMixedPredMember(b, loc, stream, predMemberName);
   }

   auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(scan.getElem().getColumn().type);
   if (!ler) return stream;
   subop::StateMembersAttr valueMembers;
   if (auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(ler.getState())) {
      valueMembers = mixed.getValueMembers();
   } else if (auto hiv = mlir::dyn_cast<subop::HashIndexedViewType>(ler.getState())) {
      valueMembers = hiv.getValueMembers();
   } else {
      return stream;
   }

   auto* ctx = stream.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   subop::Member predMember;
   for (subop::Member member : valueMembers.getMembers()) {
      if (mm.getName(member) == predMemberName) {
         predMember = member;
         break;
      }
   }
   if (!predMember) return stream;

   tuples::ColumnDefAttr predDef = cm.createDef(cm.getUniqueScope("split_branch_pred"), "filter_pred");
   predDef.getColumn().type = mlir::IntegerType::get(ctx, 1);
   auto gather = b.create<subop::GatherOp>(
      loc, stream.getType(), stream, cm.createRef(&scan.getElem().getColumn()),
      subop::ColumnDefMemberMappingAttr::get(ctx, {{predMember, predDef}}));
   tuples::ColumnRefAttr predRef = cm.createRef(&predDef.getColumn());
   auto filter = b.create<subop::FilterOp>(loc, gather.getRes(), subop::FilterSemantic::all_true,
                                           b.getArrayAttr({predRef}));
   return filter.getRes();
}

static mlir::Type scanListLookupStateType(subop::ScanListOp scan) {
   auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(scan.getElem().getColumn().type);
   if (!ler) return {};
   return ler.getState();
}

static bool sameAggregateSplitColumnIdentity(tuples::ColumnRefAttr a, tuples::ColumnRefAttr b,
                                             tuples::ColumnManager& cm) {
   auto [scopeA, nameA] = cm.getName(&a.getColumn());
   auto [scopeB, nameB] = cm.getName(&b.getColumn());
   return scopeA == scopeB && nameA == nameB && a.getColumn().type == b.getColumn().type;
}

static std::optional<unsigned> findAggregateNestedMapParameter(subop::NestedMapOp nested,
                                                               tuples::ColumnRefAttr ref,
                                                               tuples::ColumnManager& cm) {
   for (unsigned i = 0; i < nested.getParameters().size(); ++i) {
      auto existing = mlir::cast<tuples::ColumnRefAttr>(nested.getParameters()[i]);
      if (sameAggregateSplitColumnIdentity(existing, ref, cm)) return i;
   }
   return std::nullopt;
}

static mlir::BlockArgument ensureAggregateNestedMapParameter(subop::NestedMapOp nested,
                                                             tuples::ColumnRefAttr ref,
                                                             tuples::ColumnManager& cm) {
   if (auto idx = findAggregateNestedMapParameter(nested, ref, cm)) {
      assert(nested.getRegion().front().getNumArguments() > *idx + 1 &&
             "nested_map parameter must have a body argument");
      return nested.getRegion().front().getArgument(*idx + 1);
   }
   llvm::SmallVector<mlir::Attribute, 8> params(nested.getParameters().begin(),
                                                nested.getParameters().end());
   params.push_back(ref);
   nested.setParametersAttr(mlir::ArrayAttr::get(nested.getContext(), params));
   return nested.getRegion().front().addArgument(ref.getColumn().type, nested.getLoc());
}

static subop::NestedExecutionGroupOp firstAggregateNestedExecutionGroup(subop::NestedMapOp nested) {
   assert(!nested.getRegion().empty());
   for (mlir::Operation& op : nested.getRegion().front()) {
      if (auto neg = mlir::dyn_cast<subop::NestedExecutionGroupOp>(&op)) return neg;
   }
   llvm_unreachable("aggregate split nested_map must contain nested_execution_group");
}

static bool aggregateOpIsNestedInside(mlir::Operation* maybeAncestor, mlir::Operation* op) {
   for (mlir::Operation* parent = op; parent; parent = parent->getParentOp()) {
      if (parent == maybeAncestor) return true;
   }
   return false;
}

static void materializeAggregateColumnOnStreamBefore(mlir::Operation* anchor,
                                                     mlir::Value& stream,
                                                     tuples::ColumnDefAttr def,
                                                     mlir::Value value) {
   assert(anchor && "aggregate split predicate threading requires an anchor op");
   mlir::OpBuilder b(anchor);
   auto map = b.create<subop::MapOp>(anchor->getLoc(), tuples::TupleStreamType::get(anchor->getContext()),
                                     stream, b.getArrayAttr({def}), b.getArrayAttr({}));
   mlir::Block* block = new mlir::Block();
   map.getFn().push_back(block);
   mlir::OpBuilder rb(anchor->getContext());
   rb.setInsertionPointToStart(block);
   rb.create<tuples::ReturnOp>(anchor->getLoc(), mlir::ValueRange{value});
   stream = map.getResult();
}

static void rewireAggregateStreamUsesAfterAnchor(mlir::Value oldStream, mlir::Value newStream,
                                                 mlir::Operation* anchorOp,
                                                 llvm::ArrayRef<mlir::Operation*> excludeOps) {
   oldStream.replaceUsesWithIf(newStream, [&](mlir::OpOperand& use) {
      mlir::Operation* owner = use.getOwner();
      if (owner->getBlock() != anchorOp->getBlock()) return false;
      if (!anchorOp->isBeforeInBlock(owner)) return false;
      for (mlir::Operation* ex : excludeOps)
         if (owner == ex) return false;
      return true;
   });
}

static subop::Member predMemberForScanList(subop::ScanListOp scan, llvm::StringRef predMemberName) {
   auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(scan.getElem().getColumn().type);
   if (!ler) return {};
   subop::StateMembersAttr valueMembers;
   if (auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(ler.getState())) {
      valueMembers = mixed.getValueMembers();
   } else if (auto hiv = mlir::dyn_cast<subop::HashIndexedViewType>(ler.getState())) {
      valueMembers = hiv.getValueMembers();
   } else {
      return {};
   }
   auto& mm = scan.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   for (subop::Member member : valueMembers.getMembers()) {
      if (mm.getName(member) == predMemberName) return member;
   }
   return {};
}

static std::optional<tuples::ColumnDefAttr>
gatherAggregatePredAfterScanList(subop::ScanListOp scan, llvm::StringRef predMemberName) {
   subop::Member predMember = predMemberForScanList(scan, predMemberName);
   if (!predMember) return std::nullopt;
   auto* ctx = scan.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   tuples::ColumnDefAttr predDef = cm.createDef(cm.getUniqueScope("split_branch_pred"), "filter_pred");
   predDef.getColumn().type = mlir::IntegerType::get(ctx, 1);
   mlir::OpBuilder b(scan);
   b.setInsertionPointAfter(scan);
   auto gather = b.create<subop::GatherOp>(
      scan.getLoc(), scan.getRes().getType(), scan.getRes(), cm.createRef(&scan.getElem().getColumn()),
      subop::ColumnDefMemberMappingAttr::get(ctx, {{predMember, predDef}}));
   rewireAggregateStreamUsesAfterAnchor(scan.getRes(), gather.getRes(), scan.getOperation(),
                                        {scan.getOperation(), gather.getOperation()});
   gather->setOperand(0, scan.getRes());
   return predDef;
}

static std::optional<std::pair<mlir::Value, tuples::ColumnRefAttr>>
threadAggregatePredToCurrentStream(mlir::OpBuilder& b,
                                   subop::ReduceOp reduce,
                                   mlir::Operation* sourceOp,
                                   tuples::ColumnDefAttr predDef,
                                   mlir::Value currentStream) {
   auto& cm = reduce.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   tuples::ColumnRefAttr ref = cm.createRef(&predDef.getColumn());

   auto threadNestedValueToTargetOp = [](subop::NestedExecutionGroupOp neg,
                                         mlir::Value value,
                                         mlir::Operation* targetOp) -> mlir::Value {
      llvm::SmallVector<mlir::Operation*, 8> portOps;
      for (mlir::Operation* op = targetOp->getParentOp(); op && op != neg.getOperation();
           op = op->getParentOp()) {
         if (mlir::isa<subop::NestedExecutionGroupOp, subop::ExecutionStepOp>(op)) portOps.push_back(op);
      }
      assert(!portOps.empty() && "aggregate split value must be used inside a nested port op");
      mlir::Value current = value;
      for (mlir::Operation* op : llvm::reverse(portOps)) {
         if (auto nestedGroup = mlir::dyn_cast<subop::NestedExecutionGroupOp>(op)) {
            current = ensureNestedExecutionGroupInput(nestedGroup, current);
         } else {
            current = ensureExecutionStepInput(mlir::cast<subop::ExecutionStepOp>(op), current);
         }
      }
      return current;
   };

   llvm::SmallVector<subop::NestedMapOp, 4> nestedMaps;
   for (mlir::Operation* parent = reduce->getParentOp(); parent; parent = parent->getParentOp()) {
      if (auto nested = mlir::dyn_cast<subop::NestedMapOp>(parent)) {
         if (!aggregateOpIsNestedInside(nested.getOperation(), sourceOp)) nestedMaps.push_back(nested);
      }
   }
   std::reverse(nestedMaps.begin(), nestedMaps.end());

   mlir::Value currentValue;
   for (auto [idx, nested] : llvm::enumerate(nestedMaps)) {
      if (currentValue && !findAggregateNestedMapParameter(nested, ref, cm)) {
         mlir::Value stream = nested.getStream();
         materializeAggregateColumnOnStreamBefore(nested.getOperation(), stream, predDef, currentValue);
         nested->setOperand(0, stream);
      }

      mlir::BlockArgument nestedArg = ensureAggregateNestedMapParameter(nested, ref, cm);
      subop::NestedExecutionGroupOp neg = firstAggregateNestedExecutionGroup(nested);
      mlir::BlockArgument negArg = ensureNestedExecutionGroupInput(neg, nestedArg);
      mlir::Operation* targetOp = idx + 1 < nestedMaps.size()
                                     ? nestedMaps[idx + 1].getOperation()
                                     : reduce.getOperation();
      currentValue = threadNestedValueToTargetOp(neg, negArg, targetOp);
   }

   if (!currentValue) return std::make_pair(currentStream, ref);

   tuples::ColumnDefAttr localDef = cm.createDef(&predDef.getColumn());
   auto map = b.create<subop::MapOp>(reduce.getLoc(), tuples::TupleStreamType::get(reduce.getContext()),
                                     currentStream, b.getArrayAttr({localDef}), b.getArrayAttr({}));
   mlir::Block* block = new mlir::Block();
   map.getFn().push_back(block);
   mlir::OpBuilder rb(reduce.getContext());
   rb.setInsertionPointToStart(block);
   rb.create<tuples::ReturnOp>(reduce.getLoc(), mlir::ValueRange{currentValue});
   b.setInsertionPointAfter(map);
   return std::make_pair(map.getResult(), cm.createRef(&localDef.getColumn()));
}

static mlir::Value filterCurrentStreamByPredRefs(mlir::OpBuilder& b,
                                                 mlir::Location loc,
                                                 mlir::Value stream,
                                                 llvm::ArrayRef<tuples::ColumnRefAttr> predRefs) {
   if (predRefs.empty()) return stream;
   llvm::SmallVector<mlir::Attribute, 4> conds;
   conds.append(predRefs.begin(), predRefs.end());
   auto filter = b.create<subop::FilterOp>(loc, stream, subop::FilterSemantic::all_true,
                                           b.getArrayAttr(conds));
   b.setInsertionPointAfter(filter);
   return filter.getRes();
}

static llvm::SmallVector<subop::ScanListOp, 4>
scanListsOnAggregateReduceContext(subop::ReduceOp reduce);

static mlir::Value materializeInheritedAggregatePredsBeforeLookup(
   mlir::OpBuilder& b,
   subop::ReduceOp reduce,
   mlir::Value suffixStart,
   const llvm::DenseSet<mlir::Operation*>& suffixOps,
   mlir::Value current,
   llvm::StringRef mixedPredMemberName,
   const llvm::DenseMap<mlir::Type, unsigned>& predSlotByStateType) {
   if (mixedPredMemberName.empty() && predSlotByStateType.empty()) return current;
   llvm::SmallVector<tuples::ColumnRefAttr, 4> inheritedPredRefs;
   mlir::Operation* suffixStartDef = suffixStart.getDefiningOp();
   for (subop::ScanListOp scanList : scanListsOnAggregateReduceContext(reduce)) {
      if (scanList.getOperation() == suffixStartDef) continue;
      if (suffixOps.contains(scanList.getOperation())) continue;
      std::string predMemberName = mixedPredMemberName.str();
      if (auto itSlot = predSlotByStateType.find(scanListLookupStateType(scanList));
          itSlot != predSlotByStateType.end()) {
         predMemberName = ("filter_pred$" + llvm::Twine(itSlot->second)).str();
      }
      if (predMemberName.empty()) continue;
      std::optional<tuples::ColumnDefAttr> predDef =
         gatherAggregatePredAfterScanList(scanList, predMemberName);
      if (!predDef) continue;
      std::optional<std::pair<mlir::Value, tuples::ColumnRefAttr>> threaded =
         threadAggregatePredToCurrentStream(b, reduce, scanList.getOperation(), *predDef, current);
      if (!threaded) continue;
      current = threaded->first;
      inheritedPredRefs.push_back(threaded->second);
   }
   return filterCurrentStreamByPredRefs(b, reduce.getLoc(), current, inheritedPredRefs);
}

static void collectAggregateScanListsOnStreamChain(mlir::Value rootStream,
                                                   llvm::SmallVectorImpl<subop::ScanListOp>& out,
                                                   llvm::DenseSet<mlir::Operation*>& seenScanLists) {
   llvm::SmallVector<mlir::Value, 4> worklist{rootStream};
   llvm::DenseSet<void*> seenStreams;
   while (!worklist.empty()) {
      mlir::Value stream = worklist.pop_back_val();
      if (!stream || !seenStreams.insert(stream.getAsOpaquePointer()).second) continue;
      mlir::Operation* def = stream.getDefiningOp();
      if (!def) continue;
      if (auto scanList = mlir::dyn_cast<subop::ScanListOp>(def)) {
         if (seenScanLists.insert(scanList.getOperation()).second) out.push_back(scanList);
         continue;
      }
      if (auto gather = mlir::dyn_cast<subop::GatherOp>(def)) {
         worklist.push_back(gather.getStream());
      } else if (auto map = mlir::dyn_cast<subop::MapOp>(def)) {
         worklist.push_back(map.getStream());
      } else if (auto filter = mlir::dyn_cast<subop::FilterOp>(def)) {
         worklist.push_back(filter.getStream());
      } else if (auto lookup = mlir::dyn_cast<subop::LookupOp>(def)) {
         worklist.push_back(lookup.getStream());
      } else if (auto lookup = mlir::dyn_cast<subop::LookupOrInsertOp>(def)) {
         worklist.push_back(lookup.getStream());
      } else if (auto rename = mlir::dyn_cast<subop::RenamingOp>(def)) {
         worklist.push_back(rename.getStream());
      }
   }
}

static llvm::SmallVector<subop::ScanListOp, 4>
scanListsOnAggregateReduceContext(subop::ReduceOp reduce) {
   llvm::SmallVector<subop::ScanListOp, 4> out;
   llvm::DenseSet<mlir::Operation*> seenScanLists;
   collectAggregateScanListsOnStreamChain(reduce.getStream(), out, seenScanLists);
   for (mlir::Operation* parent = reduce->getParentOp(); parent; parent = parent->getParentOp()) {
      if (auto nested = mlir::dyn_cast<subop::NestedMapOp>(parent))
         collectAggregateScanListsOnStreamChain(nested.getStream(), out, seenScanLists);
   }
   return out;
}

static mlir::Type appendMembersToStateCarrierType(mlir::MLIRContext* ctx, mlir::Type type,
                                                  llvm::ArrayRef<subop::Member> members) {
   if (members.empty()) return type;
   if (auto tl = mlir::dyn_cast<subop::ThreadLocalType>(type)) {
      mlir::Type wrapped = appendMembersToStateCarrierType(ctx, tl.getWrapped(), members);
      if (wrapped == tl.getWrapped()) return type;
      return subop::ThreadLocalType::get(ctx, mlir::cast<subop::State>(wrapped));
   }
   if (auto sorted = mlir::dyn_cast<subop::SortedViewType>(type)) {
      mlir::Type based = appendMembersToStateCarrierType(ctx, sorted.getBasedOn(), members);
      if (based == sorted.getBasedOn()) return type;
      return subop::SortedViewType::get(ctx, mlir::cast<subop::State>(based));
   }
   auto buffer = mlir::dyn_cast<subop::BufferType>(type);
   if (!buffer) return type;
   llvm::SmallVector<subop::Member, 8> out(buffer.getMembers().getMembers().begin(),
                                          buffer.getMembers().getMembers().end());
   bool changed = false;
   for (subop::Member member : members) {
      if (llvm::is_contained(out, member)) continue;
      out.push_back(member);
      changed = true;
   }
   if (!changed) return type;
   llvm::SmallVector<subop::Member> attrMembers(out.begin(), out.end());
   return subop::BufferType::get(ctx, subop::StateMembersAttr::get(ctx, attrMembers));
}

static void appendPredMembersToMatchingStateCarriers(mlir::ModuleOp module,
                                                     llvm::ArrayRef<subop::Member> predMembers) {
   auto* ctx = module.getContext();
   module.walk([&](mlir::Operation* op) {
      for (mlir::Value result : op->getResults()) {
         mlir::Type newType = appendMembersToStateCarrierType(ctx, result.getType(), predMembers);
         if (newType != result.getType()) result.setType(newType);
      }
      for (mlir::Region& region : op->getRegions()) {
         for (mlir::Block& block : region) {
            for (mlir::BlockArgument arg : block.getArguments()) {
               mlir::Type newType = appendMembersToStateCarrierType(ctx, arg.getType(), predMembers);
               if (newType != arg.getType()) arg.setType(newType);
            }
         }
      }
      if (auto scan = mlir::dyn_cast<subop::ScanRefsOp>(op)) {
         auto ref = scan.getRef();
         mlir::Type refType = ref.getColumn().type;
         if (auto entry = mlir::dyn_cast<subop::EntryRefType>(refType)) {
            mlir::Type newStateType = appendMembersToStateCarrierType(ctx, entry.getState(), predMembers);
            if (newStateType != entry.getState()) {
               ref.getColumn().type = subop::EntryRefType::get(ctx, mlir::cast<subop::State>(newStateType));
               scan.setRefAttr(ref);
            }
         }
      }
   });
}

static llvm::SmallVector<subop::ScanListOp, 4>
scanListsOnSplitMaterializeInputChain(subop::MaterializeOp matOp) {
   llvm::SmallVector<subop::ScanListOp, 4> out;
   llvm::DenseSet<void*> seenStreams;
   llvm::DenseSet<mlir::Operation*> seenScanLists;
   auto addScanList = [&](subop::ScanListOp scanList) {
      if (seenScanLists.insert(scanList.getOperation()).second) out.push_back(scanList);
   };
   llvm::SmallVector<mlir::Value, 4> worklist{matOp.getStream()};
   while (!worklist.empty()) {
      mlir::Value stream = worklist.pop_back_val();
      if (!stream || !seenStreams.insert(stream.getAsOpaquePointer()).second) continue;
      mlir::Operation* def = stream.getDefiningOp();
      if (!def) continue;
      if (auto scanList = mlir::dyn_cast<subop::ScanListOp>(def)) {
         addScanList(scanList);
         continue;
      }
      if (auto gather = mlir::dyn_cast<subop::GatherOp>(def)) {
         worklist.push_back(gather.getStream());
         continue;
      }
      if (auto map = mlir::dyn_cast<subop::MapOp>(def)) {
         worklist.push_back(map.getStream());
         continue;
      }
      if (auto filter = mlir::dyn_cast<subop::FilterOp>(def)) {
         worklist.push_back(filter.getStream());
         continue;
      }
      if (auto rename = mlir::dyn_cast<subop::RenamingOp>(def)) {
         worklist.push_back(rename.getStream());
         continue;
      }
   }
   return out;
}

static void materializeSplitPredMembersFromMixedScanLists(mlir::ModuleOp module,
                                                          llvm::ArrayRef<subop::Member> predMembers) {
   if (predMembers.empty()) return;
   auto* ctx = module.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   module.walk([&](subop::MaterializeOp mat) {
      if (!mlir::isa<subop::BufferType, subop::ThreadLocalType>(mat.getState().getType())) return;
      llvm::DenseSet<subop::Member> alreadyMapped;
      for (auto& [member, col] : mat.getMapping().getMapping()) {
         (void)col;
         alreadyMapped.insert(member);
      }
      llvm::SmallVector<subop::RefMappingPairT, 8> pairs(mat.getMapping().getMapping().begin(),
                                                         mat.getMapping().getMapping().end());
      mlir::Value currentStream = mat.getStream();
      for (subop::Member predMember : predMembers) {
         if (alreadyMapped.contains(predMember)) continue;
         subop::ScanListOp sourceScan;
         for (subop::ScanListOp scan : scanListsOnSplitMaterializeInputChain(mat)) {
            auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(scan.getElem().getColumn().type);
            if (!ler) continue;
            auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(ler.getState());
            if (!mixed) continue;
            if (!valueMembersContainMemberNamed(ctx, mixed.getValueMembers(), mm.getName(predMember))) continue;
            sourceScan = scan;
            break;
         }
         if (!sourceScan) continue;
         tuples::ColumnDefAttr predDef = cm.createDef(cm.getUniqueScope("split_carried_pred"), "filter_pred");
         predDef.getColumn().type = mlir::IntegerType::get(ctx, 1);
         mlir::OpBuilder b(mat);
         auto gather = b.create<subop::GatherOp>(
            mat.getLoc(), currentStream.getType(), currentStream, cm.createRef(&sourceScan.getElem().getColumn()),
            subop::ColumnDefMemberMappingAttr::get(ctx, {{predMember, predDef}}));
         currentStream = gather.getRes();
         pairs.push_back({predMember, cm.createRef(&predDef.getColumn())});
         alreadyMapped.insert(predMember);
      }
      if (currentStream != mat.getStream()) {
         mat->setOperand(0, currentStream);
         llvm::SmallVector<subop::RefMappingPairT> attrPairs(pairs.begin(), pairs.end());
         mat.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, attrPairs));
      }
   });
   synchronizeExecutionStepPortTypes(module, nullptr);
}

static void ensureSplitMaterializeMixedPredCarriers(mlir::ModuleOp module,
                                                    llvm::ArrayRef<unsigned> predSlots) {
   if (predSlots.empty()) return;
   auto* ctx = module.getContext();
   llvm::SmallVector<subop::Member, 8> predMembers;
   llvm::DenseSet<unsigned> seen;
   for (unsigned slot : predSlots) {
      if (!seen.insert(slot).second) continue;
      predMembers.push_back(makeOrGetPredMemberForSlot(ctx, slot));
   }
   appendPredMembersToMatchingStateCarriers(module, predMembers);
   materializeSplitPredMembersFromMixedScanLists(module, predMembers);
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

static std::string mapBodySemanticFingerprint(subop::MapOp map) {
   std::string out;
   llvm::raw_string_ostream os(out);
   os << "inputs:" << map.getInputCols().size() << "|returns:";
   auto ret = mlir::cast<tuples::ReturnOp>(map.getFn().front().getTerminator());
   llvm::DenseMap<mlir::Value, std::string> memo;
   for (mlir::Value operand : ret.getOperands())
      os << residualPredicateValueFingerprint(operand, memo) << ';';
   os.flush();
   return out;
}

static bool mapReturnUsesOnlyMapBlockArgs(subop::MapOp map) {
   llvm::DenseSet<unsigned> used;
   mlir::Block& block = map.getFn().front();
   auto ret = mlir::cast<tuples::ReturnOp>(block.getTerminator());
   for (mlir::Value operand : ret.getOperands())
      collectSplitResidualBlockArgs(operand, &block, used);
   for (unsigned idx : used) {
      if (idx >= map.getInputCols().size()) return false;
   }
   return true;
}

static mlir::Value streamInputOfLinearSuffixOp(mlir::Operation* op) {
   if (auto scanList = mlir::dyn_cast<subop::ScanListOp>(op)) return scanList.getList();
   if (auto gather = mlir::dyn_cast<subop::GatherOp>(op)) return gather.getStream();
   if (auto map = mlir::dyn_cast<subop::MapOp>(op)) return map.getStream();
   if (auto filter = mlir::dyn_cast<subop::FilterOp>(op)) return filter.getStream();
   if (auto rename = mlir::dyn_cast<subop::RenamingOp>(op)) return rename.getStream();
   if (auto lookup = mlir::dyn_cast<subop::LookupOp>(op)) return lookup.getStream();
   if (auto lookup = mlir::dyn_cast<subop::LookupOrInsertOp>(op)) return lookup.getStream();
   llvm_unreachable("split-materialize residual suffix contains unsupported stream op");
}

static mlir::Value streamResultOfLinearSuffixOp(mlir::Operation* op) {
   if (auto scanList = mlir::dyn_cast<subop::ScanListOp>(op)) return scanList.getRes();
   if (auto gather = mlir::dyn_cast<subop::GatherOp>(op)) return gather.getResult();
   if (auto map = mlir::dyn_cast<subop::MapOp>(op)) return map.getResult();
   if (auto filter = mlir::dyn_cast<subop::FilterOp>(op)) return filter.getRes();
   if (auto rename = mlir::dyn_cast<subop::RenamingOp>(op)) return rename.getResult();
   if (auto lookup = mlir::dyn_cast<subop::LookupOp>(op)) return lookup.getRes();
   if (auto lookup = mlir::dyn_cast<subop::LookupOrInsertOp>(op)) return lookup.getRes();
   llvm_unreachable("split-materialize residual suffix contains unsupported stream op");
}

static mlir::Value executionStepOperandForBlockArgument(mlir::Value value);

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

struct SplitAggregateBuild {
   ExecutionStepOp step;
   subop::ScanRefsOp scan;
   mlir::Value suffixStart;
   subop::LookupOrInsertOp lookup;
   subop::LookupOp plainLookup;
   subop::ReduceOp reduce;
};

static mlir::Value findAggregateSuffixStart(subop::ReduceOp reduce) {
   mlir::Value stream = reduce.getStream();
   for (;;) {
      mlir::Operation* def = stream.getDefiningOp();
      if (!def) return {};
      if (mlir::isa<subop::ScanRefsOp, subop::ScanListOp>(def)) return stream;
      stream = streamInputOfLinearSuffixOp(def);
   }
}

static std::optional<SplitAggregateBuild> tryFindUniqueAggregateBuildWritingState(mlir::ModuleOp module,
                                                                                  mlir::Value state) {
   state = canonicalizeStateValueForReuse(state);
   SplitAggregateBuild found;
   module.walk([&](subop::LookupOrInsertOp lookup) {
      mlir::Value lookupState = canonicalizeStateValueForReuse(
         peelBlockArgsToEnclosingOperands(lookup.getState()));
      if (lookupState != state) return;
      assert(!found.lookup && "split-aggregate reuse expects one lookup_or_insert writer");
      found.lookup = lookup;
      found.step = lookup->getParentOfType<ExecutionStepOp>();
      assert(found.step && "split-aggregate lookup_or_insert must be inside an execution_step");
   });
   if (!found.lookup) return std::nullopt;
   tuples::Column* lookupRefColumn = &found.lookup.getRef().getColumn();
   found.step.walk([&](subop::ReduceOp reduce) {
      if (&reduce.getRef().getColumn() != lookupRefColumn) return;
      assert(!found.reduce && "split-aggregate reuse expects one reduce for lookup_or_insert");
      found.reduce = reduce;
   });
   if (!found.reduce) return std::nullopt;
   found.scan = tryFindFirstScanRefsInStep(found.step);
   found.suffixStart = found.scan ? found.scan.getRes() : findAggregateSuffixStart(found.reduce);
   if (!found.suffixStart) return std::nullopt;
   return found;
}

static std::optional<SplitAggregateBuild> tryFindUniquePlainReduceBuildWritingState(mlir::ModuleOp module,
                                                                                    mlir::Value state) {
   state = canonicalizeStateValueForReuse(state);
   SplitAggregateBuild found;
   module.walk([&](subop::LookupOp lookup) {
      mlir::Value lookupState = canonicalizeStateValueForReuse(
         peelBlockArgsToEnclosingOperands(lookup.getState()));
      if (lookupState != state) return;
      assert(!found.plainLookup && "split-reduce reuse expects one lookup writer");
      found.plainLookup = lookup;
      found.step = lookup->getParentOfType<ExecutionStepOp>();
      assert(found.step && "split-reduce lookup must be inside an execution_step");
   });
   if (!found.plainLookup) return std::nullopt;

   tuples::Column* lookupRefColumn = &found.plainLookup.getRef().getColumn();
   found.step.walk([&](subop::ReduceOp reduce) {
      if (&reduce.getRef().getColumn() != lookupRefColumn) return;
      assert(!found.reduce && "split-reduce reuse expects one reduce for lookup");
      found.reduce = reduce;
   });
   if (!found.reduce) return std::nullopt;
   found.scan = tryFindFirstScanRefsInStep(found.step);
   found.suffixStart = found.scan ? found.scan.getRes() : findAggregateSuffixStart(found.reduce);
   if (!found.suffixStart) return std::nullopt;
   return found;
}

static SplitAggregateBuild findUniqueAggregateBuildWritingState(mlir::ModuleOp module, mlir::Value state) {
   std::optional<SplitAggregateBuild> found = tryFindUniqueAggregateBuildWritingState(module, state);
   if (!found) found = tryFindUniquePlainReduceBuildWritingState(module, state);
   assert(found && "split-aggregate reuse requires a lookup_or_insert/reduce or lookup/reduce writer");
   return *found;
}

static llvm::SmallVector<mlir::Operation*, 8>
linearStreamOpsBeforeReduce(mlir::Value suffixStart, subop::ReduceOp reduce) {
   llvm::SmallVector<mlir::Operation*, 8> reverseOps;
   mlir::Value stream = reduce.getStream();
   while (stream != suffixStart) {
      mlir::Operation* def = stream.getDefiningOp();
      assert(def && "split-aggregate suffix must be local SSA");
      reverseOps.push_back(def);
      stream = streamInputOfLinearSuffixOp(def);
   }
   llvm::SmallVector<mlir::Operation*, 8> ops;
   for (mlir::Operation* op : llvm::reverse(reverseOps)) ops.push_back(op);
   return ops;
}

static llvm::SmallVector<subop::MapOp, 8>
linearMapOpsBeforeReduce(mlir::Value suffixStart, subop::ReduceOp reduce) {
   llvm::SmallVector<subop::MapOp, 8> maps;
   for (mlir::Operation* op : linearStreamOpsBeforeReduce(suffixStart, reduce)) {
      if (auto map = mlir::dyn_cast<subop::MapOp>(op)) maps.push_back(map);
   }
   return maps;
}

static std::optional<SplitResidualFilter>
findResidualFilterBeforeReduce(mlir::Value suffixStart, subop::ReduceOp reduce) {
   ExecutionStepOp step = reduce->getParentOfType<ExecutionStepOp>();
   assert(step && "split reduce must live in an execution_step");
   mlir::Value stream = reduce.getStream();
   while (stream != suffixStart) {
      mlir::Operation* def = stream.getDefiningOp();
      if (!def) return std::nullopt;
      if (auto filter = mlir::dyn_cast<subop::FilterOp>(def)) {
         auto map = mlir::dyn_cast_or_null<subop::MapOp>(filter.getStream().getDefiningOp());
         if (splitResidualFilterIsRelaxable(step, map, filter)) {
            return SplitResidualFilter{map, filter, map.getStream()};
         }
         stream = filter.getStream();
         continue;
      }
      stream = streamInputOfLinearSuffixOp(def);
   }
   return std::nullopt;
}

static subop::PreAggrHtFragmentType preAggrFragmentTypeFromBuildState(mlir::Type type) {
   if (auto tl = mlir::dyn_cast<subop::ThreadLocalType>(type)) type = tl.getWrapped();
   return mlir::dyn_cast<subop::PreAggrHtFragmentType>(type);
}

static void alignClonedAggregateRefAndReduceMembers(subop::LookupOrInsertOp lookup,
                                                    subop::ReduceOp reduce,
                                                    mlir::Value targetState) {
   auto fragment = preAggrFragmentTypeFromBuildState(targetState.getType());
   assert(fragment && "split-aggregate target must be an optimistic_ht_fragment");
   auto* ctx = lookup.getContext();
   auto refDef = lookup.getRef();
   refDef.getColumn().type = subop::LookupEntryRefType::get(ctx, fragment);
   lookup.setRefAttr(refDef);
   auto ref = reduce.getRef();
   ref.getColumn().type = refDef.getColumn().type;
   reduce.setRefAttr(ref);

   llvm::SmallVector<mlir::Attribute, 16> members;
   for (subop::Member member : fragment.getValueMembers().getMembers())
      members.push_back(subop::MemberAttr::get(ctx, member));
   assert(members.size() == reduce.getMembers().size() &&
          "split-aggregate reduce member layout must align by ordinal");
   reduce.setMembersAttr(mlir::ArrayAttr::get(ctx, members));
}

static void alignClonedPlainLookupRefAndReduceMembers(subop::LookupOp lookup,
                                                      subop::ReduceOp reduce,
                                                      mlir::Value targetState) {
   auto state = mlir::dyn_cast<subop::State>(targetState.getType());
   assert(state && "split-reduce target must be a state");
   auto lookupState = mlir::dyn_cast<subop::LookupAbleState>(targetState.getType());
   assert(lookupState && "split-reduce target must be lookup-able");
   auto* ctx = lookup.getContext();
   auto refDef = lookup.getRef();
   refDef.getColumn().type = subop::LookupEntryRefType::get(ctx, lookupState);
   lookup.setRefAttr(refDef);
   auto ref = reduce.getRef();
   ref.getColumn().type = refDef.getColumn().type;
   reduce.setRefAttr(ref);

   llvm::SmallVector<mlir::Attribute, 16> members;
   for (subop::Member member : state.getMembers().getMembers())
      members.push_back(subop::MemberAttr::get(ctx, member));
   assert(members.size() == reduce.getMembers().size() &&
          "split-reduce member layout must align by ordinal");
   reduce.setMembersAttr(mlir::ArrayAttr::get(ctx, members));
}

static llvm::StringRef stripReuseSuffix(llvm::StringRef s) {
   size_t pos = s.find('$');
   return pos == llvm::StringRef::npos ? s : s.take_front(pos);
}

static void addLeafAliasIfSafe(llvm::StringMap<tuples::ColumnRefAttr>& byLeaf,
                               llvm::StringRef name,
                               tuples::ColumnRefAttr ref) {
   byLeaf[name] = ref;
   if (!isGenericNumberedMemberName(name)) byLeaf[stripReuseSuffix(name)] = ref;
}

static std::string localColumnSemanticKey(llvm::StringRef scope, llvm::StringRef leaf) {
   return (scope + "\x1f" + leaf).str();
}

static void collectMapOpsOnStreamChainLocal(mlir::Value stream,
                                            llvm::SmallVectorImpl<subop::MapOp>& out) {
   llvm::DenseSet<void*> seen;
   llvm::SmallVector<mlir::Value, 8> worklist{stream};
   while (!worklist.empty()) {
      mlir::Value v = worklist.pop_back_val();
      if (!v || !seen.insert(v.getAsOpaquePointer()).second) continue;
      for (mlir::Operation* user : v.getUsers()) {
         if (auto map = mlir::dyn_cast<subop::MapOp>(user)) {
            if (map.getStream() == v) out.push_back(map);
            continue;
         }
         if (auto nested = mlir::dyn_cast<subop::NestedMapOp>(user)) {
            if (nested.getStream() == v) worklist.push_back(nested.getRes());
         }
      }
   }
}

static void syncMapInputColsFromSyntheticGather(subop::GatherOp gather) {
   auto* ctx = gather.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   llvm::StringMap<tuples::ColumnRefAttr> byKey;
   llvm::StringMap<tuples::ColumnRefAttr> byLeaf;
   for (auto [member, def] : gather.getMapping().getMapping()) {
      auto [scope, leaf] = cm.getName(&def.getColumn());
      tuples::ColumnRefAttr ref = cm.createRef(&def.getColumn());
      byKey[localColumnSemanticKey(scope, leaf)] = ref;
      addLeafAliasIfSafe(byLeaf, leaf, ref);
      addLeafAliasIfSafe(byLeaf, mm.getName(member), ref);
   }

   llvm::SmallVector<subop::MapOp, 8> maps;
   collectMapOpsOnStreamChainLocal(gather.getRes(), maps);
   for (subop::MapOp map : maps) {
      bool changed = false;
      llvm::SmallVector<mlir::Attribute> inputs;
      inputs.reserve(map.getInputCols().size());
      for (mlir::Attribute attr : map.getInputCols()) {
         auto ref = mlir::dyn_cast<tuples::ColumnRefAttr>(attr);
         if (!ref) {
            inputs.push_back(attr);
            continue;
         }
         auto [scope, leaf] = cm.getName(&ref.getColumn());
         tuples::ColumnRefAttr replacement;
         if (auto it = byKey.find(localColumnSemanticKey(scope, leaf)); it != byKey.end()) {
            replacement = it->second;
         } else if (scope.empty()) {
            if (auto it = byLeaf.find(leaf); it != byLeaf.end()) {
               replacement = it->second;
            } else if (auto it = byLeaf.find(stripReuseSuffix(leaf)); it != byLeaf.end()) {
               replacement = it->second;
            }
         }
         if (replacement && replacement.getColumnPtr() != ref.getColumnPtr()) {
            inputs.push_back(replacement);
            changed = true;
         } else {
            inputs.push_back(ref);
         }
      }
      if (changed) map.setInputColsAttr(mlir::ArrayAttr::get(ctx, inputs));
   }
}

static void rememberAggregateBranchColumn(mlir::MLIRContext* ctx,
                                          llvm::StringMap<tuples::ColumnRefAttr>& colByName,
                                          tuples::ColumnDefAttr def) {
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   tuples::ColumnRefAttr ref = cm.createRef(&def.getColumn());
   auto [scope, leaf] = cm.getName(&def.getColumn());
   addSplitColumnAliases(colByName, scope, leaf, ref);
}

static void rememberAggregateBranchGatherColumns(
   subop::GatherOp gather,
   llvm::StringMap<tuples::ColumnRefAttr>& colByName) {
   auto* ctx = gather.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   for (auto& [member, def] : gather.getMapping().getMapping()) {
      tuples::ColumnRefAttr ref = cm.createRef(&def.getColumn());
      llvm::StringRef memberName = mm.getName(member);
      addLeafAliasIfSafe(colByName, memberName, ref);
      auto [scope, leaf] = cm.getName(&def.getColumn());
      addSplitColumnAliases(colByName, scope, leaf, ref);
   }
}

static void freshenClonedAggregateGatherColumns(subop::GatherOp oldGather,
                                                subop::GatherOp clonedGather,
                                                subop::ColumnMapping& columnMapping) {
   auto* ctx = clonedGather.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   llvm::SmallVector<subop::DefMappingPairT> freshPairs;
   auto oldPairs = oldGather.getMapping().getMapping();
   auto clonedPairs = clonedGather.getMapping().getMapping();
   assert(oldPairs.size() == clonedPairs.size() &&
          "split aggregate cloned gather mapping must align with donor gather mapping");
   for (auto [oldPair, clonedPair] : llvm::zip(oldPairs, clonedPairs)) {
      auto oldDef = mlir::cast<tuples::ColumnDefAttr>(oldPair.second);
      auto clonedDef = mlir::cast<tuples::ColumnDefAttr>(clonedPair.second);
      auto [scope, leaf] = cm.getName(&clonedDef.getColumn());
      (void)scope;
      tuples::ColumnDefAttr freshDef = cm.createDef(cm.getUniqueScope("split_agg_gather"), leaf);
      freshDef.getColumn().type = clonedDef.getColumn().type;
      freshPairs.push_back({clonedPair.first, freshDef});
      columnMapping.mapRaw(&oldDef.getColumn(), &freshDef.getColumn());
   }
   clonedGather.setMappingAttr(subop::ColumnDefMemberMappingAttr::get(ctx, freshPairs));
}

[[maybe_unused]] static void freshenLookupEntryRefGatherColumns(mlir::ModuleOp module) {
   auto* ctx = module.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   llvm::DenseMap<tuples::Column*, tuples::Column*> replacement;

   module.walk([&](subop::GatherOp gather) {
      if (!mlir::isa<subop::LookupEntryRefType>(gather.getRef().getColumn().type)) return;
      llvm::SmallVector<subop::DefMappingPairT> freshPairs;
      bool changed = false;
      for (auto [member, def] : gather.getMapping().getMapping()) {
         auto [scope, leaf] = cm.getName(&def.getColumn());
         (void)scope;
         tuples::ColumnDefAttr freshDef = cm.createDef(cm.getUniqueScope("split_lookup_gather"), leaf);
         freshDef.getColumn().type = def.getColumn().type;
         replacement[&def.getColumn()] = &freshDef.getColumn();
         freshPairs.push_back({member, freshDef});
         changed = true;
      }
      if (changed) gather.setMappingAttr(subop::ColumnDefMemberMappingAttr::get(ctx, freshPairs));
   });

   if (replacement.empty()) return;
   auto remapRef = [&](tuples::ColumnRefAttr& ref) {
      auto it = replacement.find(&ref.getColumn());
      if (it == replacement.end()) return false;
      ref = cm.createRef(it->second);
      return true;
   };
   auto remapRefArray = [&](mlir::ArrayAttr arr) {
      llvm::SmallVector<mlir::Attribute> out;
      bool changed = false;
      for (mlir::Attribute attr : arr) {
         if (auto ref = mlir::dyn_cast<tuples::ColumnRefAttr>(attr)) {
            if (remapRef(ref)) changed = true;
            out.push_back(ref);
         } else {
            out.push_back(attr);
         }
      }
      return std::make_pair(changed, mlir::ArrayAttr::get(ctx, out));
   };
   auto remapRefMapping = [&](subop::ColumnRefMemberMappingAttr mapping) {
      llvm::SmallVector<subop::RefMappingPairT> out;
      bool changed = false;
      for (auto [member, ref] : mapping.getMapping()) {
         if (remapRef(ref)) changed = true;
         out.push_back({member, ref});
      }
      return std::make_pair(changed, subop::ColumnRefMemberMappingAttr::get(ctx, out));
   };

   module.walk([&](subop::MapOp op) {
      auto [changed, attr] = remapRefArray(op.getInputColsAttr());
      if (changed) op.setInputColsAttr(attr);
   });
   module.walk([&](subop::FilterOp op) {
      auto [changed, attr] = remapRefArray(op.getConditionsAttr());
      if (changed) op.setConditionsAttr(attr);
   });
   module.walk([&](subop::NestedMapOp op) {
      auto [changed, attr] = remapRefArray(op.getParametersAttr());
      if (changed) op.setParametersAttr(attr);
   });
   module.walk([&](subop::GatherOp op) {
      tuples::ColumnRefAttr ref = op.getRef();
      if (remapRef(ref)) op.setRefAttr(ref);
   });
   module.walk([&](subop::MaterializeOp op) {
      auto [changed, attr] = remapRefMapping(op.getMapping());
      if (changed) op.setMappingAttr(attr);
   });
   module.walk([&](subop::InsertOp op) {
      auto [changed, attr] = remapRefMapping(op.getMapping());
      if (changed) op.setMappingAttr(attr);
   });
   module.walk([&](subop::ReduceOp op) {
      tuples::ColumnRefAttr ref = op.getRef();
      if (remapRef(ref)) op.setRefAttr(ref);
      auto [changed, attr] = remapRefArray(op.getColumnsAttr());
      if (changed) op.setColumnsAttr(attr);
   });
   module.walk([&](subop::LookupOp op) {
      auto [changed, attr] = remapRefArray(op.getKeysAttr());
      if (changed) op.setKeysAttr(attr);
      auto ref = op.getRef();
      if (replacement.contains(&ref.getColumn())) {
         llvm_unreachable("lookup result column must not be a remapped lookup-entry gather output");
      }
   });
   module.walk([&](subop::LookupOrInsertOp op) {
      auto [changed, attr] = remapRefArray(op.getKeysAttr());
      if (changed) op.setKeysAttr(attr);
      auto ref = op.getRef();
      if (replacement.contains(&ref.getColumn())) {
         llvm_unreachable("lookup_or_insert result column must not be a remapped lookup-entry gather output");
      }
   });
}

static void alignDirectSyntheticHivGathersToCachedLayouts(
   mlir::ModuleOp module,
   const CachedJoinBufferLayoutsByKey& layoutsByKey) {
   auto& cm = module.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto asHivLayout = [](mlir::Type type) -> subop::HashIndexedViewType {
      if (auto hiv = mlir::dyn_cast<subop::HashIndexedViewType>(type)) return hiv;
      if (auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(type)) {
         return subop::HashIndexedViewType::get(mixed.getContext(), mixed.getKeyMembers(),
                                                mixed.getValueMembers(), mixed.getCompareHashForLookup());
      }
      return {};
   };

   auto& mm = module.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto valueMembersOf = [](subop::HashIndexedViewType hiv) {
      llvm::DenseSet<subop::Member> out;
      for (subop::Member member : hiv.getValueMembers().getMembers()) out.insert(member);
      return out;
   };
   struct RefGatherGroup {
      subop::HashIndexedViewType hiv;
      llvm::SmallVector<subop::GatherOp, 8> gathers;
   };
   auto columnKey = [](tuples::Column& column) {
      return &column;
   };
   llvm::DenseMap<tuples::Column*, RefGatherGroup> groups;
   module.walk([&](subop::ScanListOp scan) {
      auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(scan.getElem().getColumn().type);
      if (!ler) return;
      auto hiv = asHivLayout(ler.getState());
      if (!hiv) return;
      groups[columnKey(scan.getElem().getColumn())].hiv = hiv;
   });
   module.walk([&](subop::GatherOp gather) {
      auto it = groups.find(columnKey(gather.getRef().getColumn()));
      if (it == groups.end()) return;
      it->second.gathers.push_back(gather);
   });

   auto layoutKey = [&](subop::HashIndexedViewType hiv) {
      std::string out;
      llvm::raw_string_ostream os(out);
      for (subop::Member member : hiv.getValueMembers().getMembers()) {
         if (parseFilterPredMemberSlot(mm.getName(member))) continue;
         os << mm.getName(member) << ':';
         mm.getType(member).print(os);
         os << ';';
      }
      os.flush();
      return out;
   };

   llvm::StringMap<llvm::StringMap<subop::Member>> memberByLeafByLayout;
   auto addMemberSemanticAliases = [&](llvm::StringMap<subop::Member>& memberByLeaf,
                                       llvm::StringRef scope,
                                       llvm::StringRef leaf,
                                       subop::Member member) {
      memberByLeaf[leaf] = member;
      memberByLeaf[scopedColumnName(scope, leaf)] = member;
      if (isGenericNumberedMemberName(leaf)) return;
      memberByLeaf[stripReuseSuffix(leaf)] = member;
      memberByLeaf[scopedColumnName(scope, stripReuseSuffix(leaf))] = member;
   };
   for (auto& entry : groups) {
      RefGatherGroup& group = entry.second;
      if (!group.hiv || group.gathers.empty()) continue;
      llvm::DenseSet<subop::Member> valueMembers = valueMembersOf(group.hiv);
      llvm::StringMap<subop::Member>& memberByLeaf = memberByLeafByLayout[layoutKey(group.hiv)];
      for (subop::Member member : group.hiv.getValueMembers().getMembers()) {
         memberByLeaf[mm.getName(member)] = member;
      }
      for (subop::GatherOp gather : group.gathers) {
         for (auto [member, def] : gather.getMapping().getMapping()) {
            if (!valueMembers.contains(member)) continue;
            auto [scope, leaf] = cm.getName(&def.getColumn());
            addMemberSemanticAliases(memberByLeaf, scope, leaf, member);
         }
      }
      module.walk([&](subop::MaterializeOp mat) {
         for (auto [member, ref] : mat.getMapping().getMapping()) {
            if (!valueMembers.contains(member)) continue;
            auto [scope, leaf] = cm.getName(&ref.getColumn());
            addMemberSemanticAliases(memberByLeaf, scope, leaf, member);
         }
      });
   }

   llvm::DenseMap<tuples::Column*, llvm::DenseMap<subop::Member, subop::Member>> memberRemapByRefKey;
   for (auto& entry : groups) {
      RefGatherGroup& group = entry.second;
      if (!group.hiv || group.gathers.empty()) continue;
      llvm::DenseSet<subop::Member> valueMembers = valueMembersOf(group.hiv);
      llvm::StringMap<subop::Member> memberByLeaf = memberByLeafByLayout[layoutKey(group.hiv)];
      for (const auto& layoutEntry : layoutsByKey) {
         const CachedJoinBufferLayout& layout = layoutEntry.second;
         if (!layout.producerHiv || layoutKey(layout.producerHiv) != layoutKey(group.hiv)) continue;
         for (size_t i = 0; i < layout.payloadMembers.size() && i < layout.payloadSemanticKeys.size(); ++i) {
            subop::Member member = layout.payloadMembers[i];
            llvm::StringRef sem(layout.payloadSemanticKeys[i]);
            memberByLeaf[sem] = member;
            size_t sep = sem.find('\x1f');
            if (sep != llvm::StringRef::npos) memberByLeaf[sem.drop_front(sep + 1)] = member;
            if (!isGenericNumberedMemberName(sem)) memberByLeaf[stripReuseSuffix(sem)] = member;
         }
      }

      llvm::DenseMap<subop::Member, llvm::SmallVector<std::string, 4>> semanticKeysByOldMember;
      for (subop::GatherOp gather : group.gathers) {
         for (auto [member, def] : gather.getMapping().getMapping()) {
            if (valueMembers.contains(member)) continue;
            auto [scope, leaf] = cm.getName(&def.getColumn());
            llvm::StringRef leafRef(leaf);
            semanticKeysByOldMember[member].push_back(localColumnSemanticKey(scope, leafRef));
            semanticKeysByOldMember[member].push_back(leafRef.str());
            if (!isGenericNumberedMemberName(leafRef))
               semanticKeysByOldMember[member].push_back(stripReuseSuffix(leafRef).str());
         }
      }

      llvm::DenseMap<subop::Member, subop::Member> localRemap;
      for (subop::GatherOp gather : group.gathers) {
         for (auto [member, def] : gather.getMapping().getMapping()) {
            if (valueMembers.contains(member) || localRemap.contains(member)) continue;
            auto [scope, leaf] = cm.getName(&def.getColumn());
            (void)scope;
            auto it = memberByLeaf.find(leaf);
            if (it == memberByLeaf.end() && !isGenericNumberedMemberName(leaf))
               it = memberByLeaf.find(stripReuseSuffix(leaf));
            if (it == memberByLeaf.end()) {
               auto itKeys = semanticKeysByOldMember.find(member);
               if (itKeys != semanticKeysByOldMember.end()) {
                  for (llvm::StringRef key : itKeys->second) {
                     it = memberByLeaf.find(key);
                     if (it != memberByLeaf.end()) break;
                  }
               }
            }
            subop::Member mapped;
            if (it != memberByLeaf.end()) {
               mapped = it->second;
            } else {
               for (const auto& layoutEntry : layoutsByKey) {
                  const CachedJoinBufferLayout& layout = layoutEntry.second;
                  if (!layout.producerHiv || layoutKey(layout.producerHiv) != layoutKey(group.hiv))
                     continue;
                  auto itRemap = layout.probeGatherMemberRemap.find(member);
                  if (itRemap == layout.probeGatherMemberRemap.end()) continue;
                  mapped = itRemap->second;
                  break;
               }
            }
            if (!mapped) {
               for (const auto& layoutEntry : layoutsByKey) {
                  const CachedJoinBufferLayout& layout = layoutEntry.second;
                  for (const auto& remap : layout.probeGatherMemberRemap) {
                     if (remap.second != member || !valueMembers.contains(remap.first)) continue;
                     assert(!mapped || mapped == remap.first &&
                            "direct synthetic HIV gather reverse member remap must be unique");
                     mapped = remap.first;
                  }
                  if (mapped) break;
               }
            }
            assert(mapped &&
                   "direct synthetic HIV gather must map old payload by semantic key or column mapping");
            localRemap[member] = mapped;
         }
      }
      memberRemapByRefKey[entry.first] = std::move(localRemap);
   }

   if (memberRemapByRefKey.empty()) return;

   struct GatherRemap {
      subop::GatherOp gather;
      llvm::SmallVector<subop::DefMappingPairT, 8> mapping;
   };
   llvm::SmallVector<GatherRemap, 16> gatherRemaps;

   module.walk([&](subop::GatherOp gather) {
      auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(gather.getRef().getColumn().type);
      if (!ler) return;
      auto itRemap = memberRemapByRefKey.find(columnKey(gather.getRef().getColumn()));
      if (itRemap == memberRemapByRefKey.end()) return;
      llvm::SmallVector<subop::DefMappingPairT, 8> out;
      bool changed = false;
      for (auto [member, def] : gather.getMapping().getMapping()) {
         subop::Member mappedMember = member;
         if (auto it = itRemap->second.find(member); it != itRemap->second.end()) {
            mappedMember = it->second;
            changed = true;
         }
         out.push_back({mappedMember, def});
      }
      if (changed) gatherRemaps.push_back(GatherRemap{gather, std::move(out)});
   });

   for (GatherRemap& remap : gatherRemaps) {
      llvm::SmallVector<subop::DefMappingPairT> attrPairs(remap.mapping.begin(), remap.mapping.end());
      remap.gather.setMappingAttr(
         subop::ColumnDefMemberMappingAttr::get(remap.gather.getContext(), attrPairs));
      syncMapInputColsFromSyntheticGather(remap.gather);
   }
}

static void rememberAggregateBranchMapColumns(
   subop::MapOp map,
   llvm::StringMap<tuples::ColumnRefAttr>& colByName,
   tuples::ColumnRefAttr& tableEntryRef) {
   for (auto attr : map.getComputedCols()) {
      auto def = mlir::cast<tuples::ColumnDefAttr>(attr);
      rememberAggregateBranchColumn(map.getContext(), colByName, def);
      if (mlir::isa<subop::TableEntryRefType>(def.getColumn().type)) {
         auto& cm = map.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
         tableEntryRef = cm.createRef(&def.getColumn());
      }
   }
}

static void rememberSplitStreamColumnsFromChain(mlir::Value stream,
                                                llvm::StringMap<tuples::ColumnRefAttr>& colByName) {
   tuples::ColumnRefAttr ignoredTableEntryRef;
   llvm::DenseSet<void*> seenStreams;
   for (;;) {
      if (!stream || !seenStreams.insert(stream.getAsOpaquePointer()).second) return;
      mlir::Operation* def = stream.getDefiningOp();
      if (!def) return;
      if (auto gather = mlir::dyn_cast<subop::GatherOp>(def)) {
         rememberAggregateBranchGatherColumns(gather, colByName);
         stream = gather.getStream();
         continue;
      }
      if (auto map = mlir::dyn_cast<subop::MapOp>(def)) {
         rememberAggregateBranchMapColumns(map, colByName, ignoredTableEntryRef);
         stream = map.getStream();
         continue;
      }
      if (auto filter = mlir::dyn_cast<subop::FilterOp>(def)) {
         stream = filter.getStream();
         continue;
      }
      if (auto rename = mlir::dyn_cast<subop::RenamingOp>(def)) {
         stream = rename.getStream();
         continue;
      }
      return;
   }
}

static subop::ColumnRefMemberMappingAttr remapResultMaterializeMappingToTargetLayoutAndStreamColumns(
   mlir::MLIRContext* ctx,
   subop::ColumnRefMemberMappingAttr sourceMapping,
   mlir::Type targetStateType,
   const llvm::StringMap<tuples::ColumnRefAttr>& colByName) {
   llvm::SmallVector<subop::Member> targetMembers = stateMembersForType(targetStateType);
   assert(!targetMembers.empty() && "split-materialize target must have members");
   assert(sourceMapping.getMapping().size() == targetMembers.size() &&
          "split-materialize branch result layout must align by ordinal with donor mapping");
   llvm::SmallVector<subop::RefMappingPairT> pairs;
   unsigned ordinal = 0;
   for (auto& [member, colRef] : sourceMapping.getMapping()) {
      (void)member;
      tuples::ColumnRefAttr mappedCol = lookupSplitColumnByName(colByName, fullNameForColumnRef(colRef));
      if (!mappedCol) mappedCol = lookupSplitColumnByName(colByName, baseNameForColumnRef(colRef));
      assert(mappedCol && "split-materialize result column must exist on branch stream");
      pairs.push_back({targetMembers[ordinal++], mappedCol});
   }
   llvm::SmallVector<subop::RefMappingPairT> attrPairs;
   attrPairs.append(pairs.begin(), pairs.end());
   return subop::ColumnRefMemberMappingAttr::get(ctx, attrPairs);
}

static std::optional<subop::Member> findTableEntryMemberForFilter(tuples::ColumnRefAttr tableEntryRef,
                                                                  llvm::StringRef columnName) {
   auto refTy = mlir::dyn_cast<subop::TableEntryRefType>(tableEntryRef.getColumn().type);
   if (!refTy) return std::nullopt;
   auto* ctx = tableEntryRef.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   for (subop::Member member : refTy.getTableColumns().getMembers()) {
      if (stripReuseSuffix(mm.getName(member)) == columnName) return member;
   }
   return std::nullopt;
}

static mlir::Value gatherMissingAggregateBranchFilterColumns(
   mlir::OpBuilder& b,
   mlir::Location loc,
   mlir::Value stream,
   tuples::ColumnRefAttr tableEntryRef,
   llvm::StringMap<tuples::ColumnRefAttr>& colByName,
   llvm::ArrayRef<runtime::FilterDescription> filters) {
   llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>, 8> mappings;
   auto* ctx = b.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   llvm::StringSet<> seen;
   for (const runtime::FilterDescription& f : filters) {
      if (f.op == runtime::FilterOp::NOTNULL || colByName.contains(f.columnName)) continue;
      if (!seen.insert(f.columnName).second) continue;
      assert(tableEntryRef && "split aggregate scan_list branch needs table entry ref for filter columns");
      std::optional<subop::Member> member = findTableEntryMemberForFilter(tableEntryRef, f.columnName);
      assert(member && "split aggregate scan_list branch missing filter member on table entry ref");
      tuples::ColumnDefAttr def = cm.createDef(cm.getUniqueScope("split_agg_filter_col"), f.columnName);
      def.getColumn().type = mm.getType(*member);
      mappings.push_back({*member, def});
   }
   if (mappings.empty()) return stream;
   llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> attrMappings(mappings.begin(), mappings.end());
   auto gather = b.create<subop::GatherOp>(loc, stream, tableEntryRef,
                                           subop::ColumnDefMemberMappingAttr::get(ctx, attrMappings));
   rememberAggregateBranchGatherColumns(gather, colByName);
   b.setInsertionPointAfter(gather);
   return gather.getRes();
}

static mlir::Value materializeAggregateBranchFiltersBeforeLookup(
   mlir::OpBuilder& b,
   mlir::Location loc,
   mlir::Value stream,
   tuples::ColumnRefAttr tableEntryRef,
   llvm::StringMap<tuples::ColumnRefAttr>& colByName,
   llvm::ArrayRef<runtime::FilterDescription> filters) {
   if (filters.empty()) return stream;
   stream = gatherMissingAggregateBranchFilterColumns(b, loc, stream, tableEntryRef, colByName, filters);
   llvm::DenseMap<llvm::StringRef, tuples::ColumnRefAttr> colByNameView;
   for (auto& kv : colByName) colByNameView[kv.getKey()] = kv.second;
   mlir::Value filtered = materializeRuntimeFiltersAsSubopFilter(b, loc, stream, colByNameView, filters);
   if (mlir::Operation* def = filtered.getDefiningOp()) b.setInsertionPointAfter(def);
   return filtered;
}

struct CloneAggregateSuffixOptions {
   llvm::ArrayRef<runtime::FilterDescription> filters = {};
   llvm::StringRef mixedPredMemberName = {};
   std::optional<SplitResidualFilter> donorResidualToReplace = std::nullopt;
   std::optional<SplitResidualFilter> branchResidual = std::nullopt;
   llvm::ArrayRef<subop::MapOp> branchMapOps = {};
   llvm::DenseMap<mlir::Type, unsigned> predSlotByStateType;
};

struct ClonedAggregateLookupRefs {
   subop::LookupOrInsertOp lookupOrInsert;
   subop::LookupOp lookup;
};

static void materializeAggregateBranchPreLookup(
   mlir::OpBuilder& b,
   subop::ReduceOp reduce,
   mlir::Value suffixStart,
   const llvm::DenseSet<mlir::Operation*>& suffixOps,
   mlir::Operation* lookupLikeOp,
   const CloneAggregateSuffixOptions& options,
   mlir::Value& current,
   llvm::StringMap<tuples::ColumnRefAttr>& colByName,
   tuples::ColumnRefAttr tableEntryRef,
   subop::ColumnMapping& columnMapping,
   bool& branchResidualInserted,
   bool& inheritedPredsInserted,
   bool& filtersInserted) {
   if (!branchResidualInserted) {
      current = cloneResidualFilterBranch(b, lookupLikeOp->getLoc(), current, colByName,
                                          *options.branchResidual,
                                          /*inputOverrides=*/{}, &columnMapping);
      if (mlir::Operation* def = current.getDefiningOp()) b.setInsertionPointAfter(def);
      branchResidualInserted = true;
   }
   if (!inheritedPredsInserted) {
      current = materializeInheritedAggregatePredsBeforeLookup(b, reduce, suffixStart, suffixOps,
                                                               current, options.mixedPredMemberName,
                                                               options.predSlotByStateType);
      inheritedPredsInserted = true;
   }
   if (!filtersInserted) {
      current = materializeAggregateBranchFiltersBeforeLookup(b, lookupLikeOp->getLoc(), current,
                                                              tableEntryRef, colByName,
                                                              options.filters);
      filtersInserted = true;
   }
}

static void recordClonedAggregateSuffixOp(
   mlir::Operation* original,
   mlir::Operation* cloned,
   subop::ColumnMapping& columnMapping,
   llvm::StringMap<tuples::ColumnRefAttr>& colByName,
   tuples::ColumnRefAttr& tableEntryRef,
   ClonedAggregateLookupRefs& lookupRefs) {
   if (auto scanList = mlir::dyn_cast<subop::ScanListOp>(cloned)) {
      auto oldScanList = mlir::cast<subop::ScanListOp>(original);
      columnMapping.mapRaw(&oldScanList.getElem().getColumn(), &scanList.getElem().getColumn());
   }
   if (auto lookup = mlir::dyn_cast<subop::LookupOrInsertOp>(cloned)) {
      auto oldLookup = mlir::cast<subop::LookupOrInsertOp>(original);
      columnMapping.mapRaw(&oldLookup.getRef().getColumn(), &lookup.getRef().getColumn());
      lookupRefs.lookupOrInsert = lookup;
   }
   if (auto lookup = mlir::dyn_cast<subop::LookupOp>(cloned)) {
      auto oldLookup = mlir::cast<subop::LookupOp>(original);
      columnMapping.mapRaw(&oldLookup.getRef().getColumn(), &lookup.getRef().getColumn());
      lookupRefs.lookup = lookup;
   }
   if (auto gather = mlir::dyn_cast<subop::GatherOp>(cloned)) {
      freshenClonedAggregateGatherColumns(mlir::cast<subop::GatherOp>(original), gather, columnMapping);
      rememberAggregateBranchGatherColumns(gather, colByName);
   } else if (auto map = mlir::dyn_cast<subop::MapOp>(cloned)) {
      rememberAggregateBranchMapColumns(map, colByName, tableEntryRef);
   }
}

static void cloneAggregateReduceAndAlign(
   mlir::OpBuilder& b,
   subop::ReduceOp reduce,
   mlir::Value currentStream,
   mlir::Value newState,
   mlir::IRMapping& mapping,
   subop::ColumnMapping& columnMapping,
   const ClonedAggregateLookupRefs& clonedLookupRefs) {
   mapping.map(reduce.getStream(), currentStream);
   auto* clonedReduceOp =
      mlir::cast<subop::SubOperator>(reduce.getOperation()).cloneSubOp(b, mapping, columnMapping);
   auto clonedReduce = mlir::cast<subop::ReduceOp>(clonedReduceOp);
   if (clonedLookupRefs.lookupOrInsert) {
      alignClonedAggregateRefAndReduceMembers(clonedLookupRefs.lookupOrInsert, clonedReduce, newState);
   } else {
      assert(clonedLookupRefs.lookup && "split-reduce suffix must clone lookup");
      alignClonedPlainLookupRefAndReduceMembers(clonedLookupRefs.lookup, clonedReduce, newState);
   }
}

static bool cloneAggregateBranchMapReplacementIfNeeded(
   mlir::OpBuilder& b,
   subop::MapOp donorMap,
   subop::MapOp branchMap,
   mlir::Value& current,
   mlir::IRMapping& mapping,
   subop::ColumnMapping& columnMapping,
   llvm::StringMap<tuples::ColumnRefAttr>& colByName,
   tuples::ColumnRefAttr& tableEntryRef) {
   if (!donorMap || !branchMap || donorMap.getInputCols().empty()) return false;
   if (mapBodySemanticFingerprint(donorMap) == mapBodySemanticFingerprint(branchMap)) return false;
   if (!mapReturnUsesOnlyMapBlockArgs(branchMap))
      llvm_unreachable("split aggregate branch map replacement cannot capture outer block arguments");
   auto clonedMap = cloneBranchMapWithDonorColumns(b, current, donorMap, branchMap, columnMapping);
   mapping.map(donorMap.getResult(), clonedMap.getResult());
   current = clonedMap.getResult();
   b.setInsertionPointAfter(clonedMap);
   rememberAggregateBranchMapColumns(clonedMap, colByName, tableEntryRef);
   return true;
}

static void cloneAggregateSuffixToReduce(mlir::OpBuilder& b,
                                         mlir::Value suffixStart,
                                         subop::ReduceOp reduce,
                                         mlir::Value newStart,
                                         mlir::Value newState,
                                         CloneAggregateSuffixOptions options = CloneAggregateSuffixOptions()) {
   llvm::SmallVector<mlir::Operation*, 8> ops = linearStreamOpsBeforeReduce(suffixStart, reduce);
   llvm::DenseSet<mlir::Operation*> suffixOps(ops.begin(), ops.end());
   mlir::IRMapping mapping;
   subop::ColumnMapping columnMapping;
   mlir::Value current = newStart;
   if (!options.mixedPredMemberName.empty()) {
      current = filterScanListBranchByMixedPredMember(b, reduce.getLoc(), current,
                                                      options.mixedPredMemberName);
   }
   mapping.map(suffixStart, current);
   llvm::StringMap<tuples::ColumnRefAttr> colByName;
   tuples::ColumnRefAttr tableEntryRef;
   bool filtersInserted = options.filters.empty();
   bool inheritedPredsInserted = options.mixedPredMemberName.empty();
   bool branchResidualInserted = !options.branchResidual || options.donorResidualToReplace.has_value();
   ClonedAggregateLookupRefs clonedLookupRefs;
   unsigned mapOrdinal = 0;
   for (mlir::Operation* op : ops) {
      subop::MapOp donorMap = mlir::dyn_cast<subop::MapOp>(op);
      subop::MapOp branchMap;
      if (donorMap && !options.branchMapOps.empty()) {
         assert(mapOrdinal < options.branchMapOps.size() &&
                "split aggregate branch map list must align with donor suffix maps");
         branchMap = options.branchMapOps[mapOrdinal++];
      }
      if (options.donorResidualToReplace &&
          op == options.donorResidualToReplace->predMap.getOperation()) {
         if (options.branchResidual) {
            llvm::SmallVector<tuples::ColumnRefAttr, 8> inputOverrides =
               resolveResidualInputRefsOnCurrentStream(*options.branchResidual, colByName, &columnMapping);
            current = cloneResidualFilterBranch(b, op->getLoc(), current, colByName, *options.branchResidual,
                                                inputOverrides, &columnMapping);
            if (mlir::Operation* def = current.getDefiningOp()) b.setInsertionPointAfter(def);
         }
         mapping.map(streamResultOfLinearSuffixOp(op), current);
         continue;
      }
      if (cloneAggregateBranchMapReplacementIfNeeded(b, donorMap, branchMap, current, mapping,
                                                     columnMapping, colByName, tableEntryRef)) {
         continue;
      }
      if (options.donorResidualToReplace &&
          op == options.donorResidualToReplace->filter.getOperation()) {
         mapping.map(streamResultOfLinearSuffixOp(op), current);
         continue;
      }
      if (mlir::isa<subop::LookupOrInsertOp, subop::LookupOp>(op)) {
         materializeAggregateBranchPreLookup(b, reduce, suffixStart, suffixOps, op, options,
                                             current, colByName, tableEntryRef, columnMapping,
                                             branchResidualInserted, inheritedPredsInserted,
                                             filtersInserted);
      }
      mapping.map(streamInputOfLinearSuffixOp(op), current);
      if (auto lookup = mlir::dyn_cast<subop::LookupOrInsertOp>(op))
         mapping.map(lookup.getState(), newState);
      if (auto lookup = mlir::dyn_cast<subop::LookupOp>(op))
         mapping.map(lookup.getState(), newState);
      auto sub = mlir::cast<subop::SubOperator>(op);
      mlir::Operation* cloned = sub.cloneSubOp(b, mapping, columnMapping);
      recordClonedAggregateSuffixOp(op, cloned, columnMapping, colByName, tableEntryRef, clonedLookupRefs);
      current = streamResultOfLinearSuffixOp(cloned);
      if (mlir::isa<subop::ScanListOp>(cloned) && !options.mixedPredMemberName.empty()) {
         b.setInsertionPointAfter(cloned);
         current = filterScanListBranchByMixedPredMember(b, cloned->getLoc(), current,
                                                         options.mixedPredMemberName);
      }
   }
   assert((options.branchMapOps.empty() || mapOrdinal == options.branchMapOps.size()) &&
          "split aggregate branch map list must align with donor suffix maps");
   assert(filtersInserted && "split aggregate branch filters must be inserted before lookup_or_insert");
   assert(inheritedPredsInserted && "split aggregate inherited predicates must be inserted before lookup_or_insert");
   assert(branchResidualInserted && "split aggregate residual predicate must be inserted before lookup_or_insert");

   cloneAggregateReduceAndAlign(b, reduce, current, newState, mapping, columnMapping, clonedLookupRefs);
}

static void eraseOriginalAggregateSuffix(mlir::Value suffixStart, subop::ReduceOp reduce) {
   llvm::SmallVector<mlir::Operation*, 8> ops = linearStreamOpsBeforeReduce(suffixStart, reduce);
   reduce.erase();
   for (mlir::Operation* op : llvm::reverse(ops))
      op->erase();
}

static mlir::Value executionStepOperandForBlockArgument(mlir::Value value) {
   auto arg = mlir::dyn_cast<mlir::BlockArgument>(value);
   if (!arg) return value;
   auto step = mlir::dyn_cast_or_null<ExecutionStepOp>(arg.getOwner()->getParentOp());
   if (!step) return value;
   if (arg.getArgNumber() >= step.getNumOperands()) return value;
   return step.getOperand(arg.getArgNumber());
}

static std::optional<ExecutionStepOp> tryFindUniqueMaterializeStepWritingState(mlir::ModuleOp module,
                                                                               mlir::Value state) {
   state = canonicalizeStateValueForReuse(state);
   ExecutionStepOp found;
   module.walk([&](ExecutionStepOp step) {
      if (step->getParentOfType<ExecutionStepOp>()) return;
      bool writes = false;
      step.walk([&](subop::MaterializeOp mat) {
         if (canonicalizeStateValueForReuse(mat.getState()) == state) writes = true;
      });
      if (!writes) return;
      assert(!found && "split-materialize reuse expects one materialize step for the output state");
      found = step;
   });
   if (!found) return std::nullopt;
   return found;
}

static ExecutionStepOp findUniqueMaterializeStepWritingState(mlir::ModuleOp module, mlir::Value state) {
   std::optional<ExecutionStepOp> found = tryFindUniqueMaterializeStepWritingState(module, state);
   if (!found) llvm_unreachable("split-materialize reuse requires a materialize step");
   return *found;
}

static std::optional<mlir::Value> mergeInputForFinalState(mlir::ModuleOp module, mlir::Value state) {
   state = canonicalizeStateValueForReuse(state);
   mlir::Value found;
   module.walk([&](subop::MergeOp merge) {
      if (canonicalizeStateValueForReuse(merge.getResult()) != state) return;
      assert(!found && "split-materialize reuse expects one merge producer for final state");
      found = executionStepOperandForBlockArgument(merge.getThreadLocal());
   });
   if (!found) return std::nullopt;
   return found;
}

static mlir::Value resolveSplitMaterializeBuildState(mlir::ModuleOp module, mlir::Value target,
                                                     const ModuleReuseInfo& reuse) {
   if (auto itShadow = findReuseMap(reuse.mergedFromShadowState, target);
       itShadow != reuse.mergedFromShadowState.end()) {
      return itShadow->second;
   }
   if (tryFindUniqueMaterializeStepWritingState(module, target)) return target;
   if (std::optional<mlir::Value> mergeInput = mergeInputForFinalState(module, target)) return *mergeInput;
   return target;
}

static bool stepMergesBuildStateToFinalState(ExecutionStepOp step,
                                             mlir::Value buildState,
                                             mlir::Value finalState) {
   buildState = canonicalizeStateValueForReuse(buildState);
   finalState = canonicalizeStateValueForReuse(finalState);
   bool found = false;
   step.walk([&](subop::MergeOp merge) {
      if (canonicalizeStateValueForReuse(merge.getResult()) != finalState) return;
      mlir::Value threadLocal = canonicalizeStateValueForReuse(
         peelBlockArgsToEnclosingOperands(merge.getThreadLocal()));
      if (threadLocal != buildState) return;
      found = true;
   });
   return found;
}

static bool splitAggregateFinalStateHasNoExtraWriters(mlir::Value target,
                                                      mlir::Value buildState,
                                                      const ModuleReuseInfo& reuse) {
   target = canonicalizeStateValueForReuse(target);
   buildState = canonicalizeStateValueForReuse(buildState);
   if (target == buildState) return true;
   auto itW = findReuseMap(reuse.writerStepsByState, target);
   if (itW == reuse.writerStepsByState.end()) return false;
   if (itW->second.size() != 1) return false;
   return stepMergesBuildStateToFinalState(itW->second.front(), buildState, target);
}

static bool splitAggregateBuildRewriteSupported(mlir::ModuleOp module, mlir::Value target,
                                                const ModuleReuseInfo& reuse) {
   mlir::Value buildState = resolveSplitMaterializeBuildState(module, target, reuse);
   if (!splitAggregateFinalStateHasNoExtraWriters(target, buildState, reuse)) return false;
   std::optional<SplitAggregateBuild> build = tryFindUniqueAggregateBuildWritingState(module, buildState);
   if (!build || !build->suffixStart) return false;

   auto itBuildW = findReuseMap(reuse.writerStepsByState, canonicalizeStateValueForReuse(buildState));
   if (itBuildW == reuse.writerStepsByState.end()) return false;
   ExecutionStepOp buildTopStep = topLevelExecutionStepFor(build->step);
   unsigned matchingWriters = 0;
   for (ExecutionStepOp writer : itBuildW->second) {
      if (topLevelExecutionStepFor(writer) == buildTopStep) matchingWriters++;
   }
   return matchingWriters == itBuildW->second.size();
}

static std::optional<llvm::SmallVector<std::string, 8>>
aggregateSplitMapFingerprint(mlir::ModuleOp module, mlir::Value target, const ModuleReuseInfo& reuse) {
   mlir::Value buildState = resolveSplitMaterializeBuildState(module, target, reuse);
   std::optional<SplitAggregateBuild> build = tryFindUniqueAggregateBuildWritingState(module, buildState);
   if (!build) build = tryFindUniquePlainReduceBuildWritingState(module, buildState);
   if (!build) return std::nullopt;
   std::optional<SplitResidualFilter> residual =
      findResidualFilterBeforeReduce(build->suffixStart, build->reduce);
   llvm::SmallVector<std::string, 8> fingerprints;
   for (subop::MapOp map : linearMapOpsBeforeReduce(build->suffixStart, build->reduce)) {
      if (residual && map == residual->predMap) continue;
      fingerprints.push_back(mapBodySemanticFingerprint(map));
   }
   return fingerprints;
}

llvm::SmallVector<CacheTarget, 64> cacheTargetsWithFilterPredReuse(llvm::ArrayRef<CacheTarget> targets) {
   llvm::SmallVector<CacheTarget, 64> out;
   out.reserve(targets.size());
   for (const CacheTarget& t : targets) {
      if (t.enableFilterPredReuse) out.push_back(t);
   }
   return out;
}

static size_t countNonTableTargets(llvm::ArrayRef<CacheTarget> targets) {
   size_t n = 0;
   for (const CacheTarget& t : targets) {
      if (!mlir::isa<TableType>(t.state.getType())) n++;
   }
   return n;
}

static void initBatchReuseCounters(BatchReusePlanRewriteResult& res, size_t numQueries) {
   res.numTargetsPerQuery.resize(numQueries, 0);
   res.numTargetsNoTablePerQuery.resize(numQueries, 0);
   res.numUnionTargetsPerQuery.resize(numQueries, 0);
   res.numUnionTargetsNoTablePerQuery.resize(numQueries, 0);
   res.numBuildStepTargetsPerQuery.resize(numQueries, 0);
   res.numBuildStepTargetsNoTablePerQuery.resize(numQueries, 0);
}

static void addVectorCounts(llvm::MutableArrayRef<size_t> dst,
                            llvm::ArrayRef<size_t> src) {
   assert(dst.size() >= src.size() && "batch reuse counter vectors must be pre-sized");
   for (auto [idx, value] : llvm::enumerate(src)) dst[idx] += value;
}

static void accumulateBatchReuseCounters(BatchReusePlanRewriteResult& total,
                                         const BatchReusePlanRewriteResult& one) {
   addVectorCounts(total.numTargetsPerQuery, one.numTargetsPerQuery);
   addVectorCounts(total.numTargetsNoTablePerQuery, one.numTargetsNoTablePerQuery);
   addVectorCounts(total.numUnionTargetsPerQuery, one.numUnionTargetsPerQuery);
   addVectorCounts(total.numUnionTargetsNoTablePerQuery, one.numUnionTargetsNoTablePerQuery);
   addVectorCounts(total.numBuildStepTargetsPerQuery, one.numBuildStepTargetsPerQuery);
   addVectorCounts(total.numBuildStepTargetsNoTablePerQuery, one.numBuildStepTargetsNoTablePerQuery);

   total.numTargetsSyntheticMapped += one.numTargetsSyntheticMapped;
   total.numTargetsSyntheticMappedNoTable += one.numTargetsSyntheticMappedNoTable;
   total.numUnionTargetsSyntheticMapped += one.numUnionTargetsSyntheticMapped;
   total.numUnionTargetsSyntheticMappedNoTable += one.numUnionTargetsSyntheticMappedNoTable;
   total.numBuildStepTargetsSyntheticMapped += one.numBuildStepTargetsSyntheticMapped;
   total.numBuildStepTargetsSyntheticMappedNoTable += one.numBuildStepTargetsSyntheticMappedNoTable;
}

static void incrementBatchTargetCategoryCounts(BatchReusePlanRewriteResult& res,
                                               unsigned queryIdx,
                                               mlir::Value targetState,
                                               bool isBuildStepReuse) {
   llvm::SmallVector<size_t, 8>& targets =
      isBuildStepReuse ? res.numBuildStepTargetsPerQuery : res.numUnionTargetsPerQuery;
   llvm::SmallVector<size_t, 8>& targetsNoTable =
      isBuildStepReuse ? res.numBuildStepTargetsNoTablePerQuery : res.numUnionTargetsNoTablePerQuery;
   targets[queryIdx]++;
   if (!mlir::isa<TableType>(targetState.getType())) targetsNoTable[queryIdx]++;
}

static void registerSyntheticMappedTarget(BatchReusePlanRewriteResult& res,
                                          llvm::SmallVectorImpl<CacheTarget>& targetsSynthetic,
                                          mlir::Value mapped,
                                          const CacheTarget& target,
                                          bool isBuildStepReuse) {
   targetsSynthetic.push_back(CacheTarget{mapped, target.cacheKey, target.enableFilterPredReuse});
   if (isBuildStepReuse) {
      res.numBuildStepTargetsSyntheticMapped++;
      if (!mlir::isa<TableType>(mapped.getType())) res.numBuildStepTargetsSyntheticMappedNoTable++;
   } else {
      res.numUnionTargetsSyntheticMapped++;
      if (!mlir::isa<TableType>(mapped.getType())) res.numUnionTargetsSyntheticMappedNoTable++;
   }
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

struct SyntheticModuleShell {
   mlir::OwningOpRef<mlir::ModuleOp> module;
   ExecutionGroupOp group;
};

static SyntheticModuleShell createSyntheticModuleShellLike(mlir::ModuleOp templateModule) {
   mlir::MLIRContext* ctx = templateModule.getContext();
   auto templateGroup = getSingleExecutionGroup(templateModule);
   auto templateMain = templateModule.lookupSymbol<mlir::func::FuncOp>("main");
   assert(templateMain && "expected func @main");

   auto module = mlir::OwningOpRef<mlir::ModuleOp>(
      mlir::ModuleOp::create(mlir::UnknownLoc::get(ctx)));
   auto main = mlir::func::FuncOp::create(templateMain.getLoc(), "main",
                                          templateMain.getFunctionType());
   module->push_back(main);
   auto* entry = main.addEntryBlock();
   mlir::OpBuilder fb = mlir::OpBuilder::atBlockBegin(entry);
   auto group = fb.create<ExecutionGroupOp>(templateGroup.getLoc(), mlir::TypeRange{},
                                            mlir::ValueRange{});
   auto& block = group.getSubOps().emplaceBlock();
   mlir::OpBuilder gb = mlir::OpBuilder::atBlockBegin(&block);
   gb.create<ExecutionGroupReturnOp>(templateGroup.getLoc(), mlir::ValueRange{});
   fb.create<mlir::func::ReturnOp>(templateMain.getLoc());
   return SyntheticModuleShell{std::move(module), group};
}

static mlir::Value lookupMappedReuseTarget(mlir::IRMapping& mapping,
                                           mlir::Value state,
                                           const ModuleReuseInfo& reuse) {
   mlir::Value mapped = mapping.lookupOrNull(state);
   if (!mapped && !state.getDefiningOp()) {
      if (auto itW = reuse.writerStepsByState.find(state);
          itW != reuse.writerStepsByState.end() && !itW->second.empty()) {
         ExecutionStepOp lastWriter = itW->second.back();
         if (lastWriter.getNumResults() == 1) mapped = mapping.lookupOrNull(lastWriter.getResult(0));
      }
   }
   return mapped;
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

static bool hasValueFilter(llvm::ArrayRef<runtime::FilterDescription> filters) {
   return llvm::any_of(filters, [](const runtime::FilterDescription& f) {
      return f.op != runtime::FilterOp::NOTNULL;
   });
}

static bool stateBuildHasOwnValueTableFilter(mlir::Value state, const ModuleReuseInfo& reuse) {
   if (!state) return false;

   llvm::SmallVector<mlir::Value, 8> candidates;
   llvm::DenseSet<void*> seenCandidates;
   auto addCandidate = [&](mlir::Value v) {
      if (!v) return;
      v = canonicalizeStateValueForReuse(v);
      if (!seenCandidates.insert(v.getAsOpaquePointer()).second) return;
      candidates.push_back(v);
   };

   addCandidate(state);
   addCandidate(resolveCacheTargetStateForReuse(state, reuse));
   addCandidate(bufferJoinChainRootForReuse(state, reuse));
   for (size_t i = 0; i < candidates.size(); ++i) {
      forEachShadowChainPredecessor(candidates[i], reuse, addCandidate);
   }

   llvm::DenseSet<mlir::Operation*> seenSteps;
   for (mlir::Value candidate : candidates) {
      auto itW = findReuseMap(reuse.writerStepsByState, candidate);
      if (itW == reuse.writerStepsByState.end()) continue;
      for (ExecutionStepOp step : itW->second) {
         if (!seenSteps.insert(step.getOperation()).second) continue;
         if (hasValueFilter(decodeFiltersFromTableScanInExecutionStep(step))) return true;
      }
   }
   return false;
}

static std::string reuseSlotSignatureForEntryInGroup(
   const ReuseRewriteContext& rewriteCtx,
   const CrossQueryStateMatchGroup& group,
   const CrossQueryStateMatchEntry& entry,
   llvm::ArrayRef<ModuleReuseInfo> reuseEarly) {
   assert(entry.query >= 0 && "reuse group entry must have a query id");
   std::string sig;
   llvm::raw_string_ostream os(sig);
   os << "self:";

   const unsigned queryIdx = static_cast<unsigned>(entry.query);
   bool hasOwnTableFilter = false;
   if (group.enableFilterPredReuse && !group.requiresSplitMaterialize && !group.cacheDeps.empty() &&
       static_cast<size_t>(entry.query) < reuseEarly.size() && entry.state) {
      hasOwnTableFilter = stateBuildHasOwnValueTableFilter(entry.state, reuseEarly[entry.query]);
   }
   if (hasOwnTableFilter) {
      os << queryIdx;
   } else if (!group.cacheDeps.empty()) {
      os << "dep_only";
   } else if (entry.reuseSlot != std::numeric_limits<unsigned>::max()) {
      os << entry.reuseSlot;
   } else {
      os << queryIdx;
   }

   for (uint64_t depKey : group.cacheDeps) {
      if (std::optional<unsigned> slot = rewriteCtx.lookupConsumerSlot(depKey, queryIdx))
         os << "|dep:" << depKey << ':' << *slot;
   }
   os.flush();
   return sig;
}

static llvm::DenseMap<unsigned, unsigned> assignReuseSlotsForGroup(
   const ReuseRewriteContext& rewriteCtx,
   const CrossQueryStateMatchGroup& group,
   llvm::ArrayRef<ModuleReuseInfo> reuseEarly,
   size_t numQueries) {
   llvm::StringMap<unsigned> slotBySignature;
   llvm::DenseMap<unsigned, unsigned> slotByQuery;
   unsigned nextReuseSlot = 0;
   for (const CrossQueryStateMatchEntry& entry : group.entries) {
      if (entry.query < 0 || static_cast<size_t>(entry.query) >= numQueries) continue;
      std::string sig = reuseSlotSignatureForEntryInGroup(rewriteCtx, group, entry, reuseEarly);
      auto itSlot = slotBySignature.find(sig);
      if (itSlot == slotBySignature.end())
         itSlot = slotBySignature.try_emplace(sig, nextReuseSlot++).first;
      slotByQuery[static_cast<unsigned>(entry.query)] = itSlot->second;
   }
   return slotByQuery;
}

static void appendSplitBranchFiltersFromWriterSteps(
   llvm::SmallVectorImpl<runtime::FilterDescription>& out,
   mlir::Value state, const ModuleReuseInfo& reuse, std::optional<unsigned> slot,
   const llvm::DenseMap<mlir::Operation*, const ModuleReuseInfo::StepRW*>& rwByStepOp) {
   if (!state) return;
   auto itW = findReuseMap(reuse.writerStepsByState, canonicalizeStateValueForReuse(state));
   if (itW == reuse.writerStepsByState.end()) itW = findReuseMap(reuse.writerStepsByState, state);
   if (itW == reuse.writerStepsByState.end()) return;
   for (ExecutionStepOp ws : itW->second) {
      const ModuleReuseInfo::StepRW* rw = rwByStepOp.lookup(ws.getOperation());
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
         const runtime::ExternalDatasourceProperty& ds = it->second;
         if (!ds.sharedPredicateClauses.empty()) {
            assert(slot && *slot < ds.sharedPredicateClauses.size() &&
                   "split branch must use an existing shared predicate slot");
            out.append(ds.sharedPredicateClauses[*slot].begin(), ds.sharedPredicateClauses[*slot].end());
            continue;
         }
         assert(ds.orFilterClauses.empty() &&
                "split branch filter extraction only supports one conjunctive source clause");
         out.append(ds.filterDescriptions.begin(), ds.filterDescriptions.end());
      }
   }
}

static llvm::SmallVector<runtime::FilterDescription, 8>
decodeSplitBranchFiltersForState(mlir::Value state, const ModuleReuseInfo& reuse, unsigned slot) {
   llvm::DenseMap<mlir::Operation*, const ModuleReuseInfo::StepRW*> rwByStepOp = buildRwByStepOpMap(reuse);
   llvm::SmallVector<runtime::FilterDescription, 8> filters;
   llvm::DenseSet<void*> seen;
   auto tryState = [&](mlir::Value candidate) {
      if (!candidate) return;
      mlir::Value canonical = canonicalizeStateValueForReuse(candidate);
      if (!seen.insert(canonical.getAsOpaquePointer()).second) return;
      size_t before = filters.size();
      appendSplitBranchFiltersFromWriterSteps(filters, canonical, reuse, slot, rwByStepOp);
      if (filters.size() != before) return;
      appendSplitBranchFiltersFromWriterSteps(filters, candidate, reuse, slot, rwByStepOp);
   };
   tryState(state);
   mlir::Value resolved = resolveCacheTargetStateForReuse(state, reuse);
   tryState(resolved);
   if (resolved) {
      forEachShadowChainPredecessor(resolved, reuse, [&](mlir::Value shadow) {
         if (!filters.empty()) return;
         tryState(shadow);
      });
   }
   return filters;
}

static bool runtimeFilterListEquals(llvm::ArrayRef<runtime::FilterDescription> a,
                                    llvm::ArrayRef<runtime::FilterDescription> b) {
   if (a.size() != b.size()) return false;
   llvm::SmallVector<bool, 8> matched(b.size(), false);
   auto sameFilter = [](const runtime::FilterDescription& lhs, const runtime::FilterDescription& rhs) {
      return normalizeSplitColumnName(lhs.columnName) == normalizeSplitColumnName(rhs.columnName) &&
         lhs.columnId == rhs.columnId &&
         lhs.op == rhs.op &&
         lhs.value == rhs.value &&
         lhs.values == rhs.values;
   };
   for (const runtime::FilterDescription& f : a) {
      bool found = false;
      for (size_t i = 0, e = b.size(); i < e; ++i) {
         if (matched[i]) continue;
         if (!sameFilter(f, b[i])) continue;
         matched[i] = true;
         found = true;
         break;
      }
      if (!found) return false;
   }
   return true;
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
   res.numTargetsQuery0NoTable = countNonTableTargets(targets0);
   res.numTargetsQuery1NoTable = countNonTableTargets(targets1);
   res.numUnionTargetsQuery0 = res.numTargetsQuery0;
   res.numUnionTargetsQuery1 = res.numTargetsQuery1;
   res.numUnionTargetsQuery0NoTable = res.numTargetsQuery0NoTable;
   res.numUnionTargetsQuery1NoTable = res.numTargetsQuery1NoTable;

   if (targets0.empty() || targets1.empty()) {
      return res;
   }

   // Build synthetic query0 module from scratch in the same context as query0, then clone steps into it.
   SyntheticModuleShell q0Shell = createSyntheticModuleShellLike(query0);
   res.query0 = std::move(q0Shell.module);
   ExecutionGroupOp q0Group = q0Shell.group;
   auto donorGroup = getSingleExecutionGroup(query0);

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
      mlir::Value mapped = lookupMappedReuseTarget(mapping, t.state, reuse0);
      assert(mapped && "reuse target must map into synthetic query0 module");
      targetsQ0.push_back(CacheTarget{mapped, t.cacheKey, t.enableFilterPredReuse});
   }
   res.numTargetsQuery0Mapped = targetsQ0.size();
   res.numTargetsQuery0MappedNoTable = countNonTableTargets(targetsQ0);
   res.numUnionTargetsQuery0Mapped = res.numTargetsQuery0Mapped;
   res.numUnionTargetsQuery0MappedNoTable = res.numTargetsQuery0MappedNoTable;
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
   const ReuseRewriteContext& rewriteCtx,
   llvm::SmallVectorImpl<CacheTarget>& targetsSynthetic);
static void alignSyntheticCacheGetDependenciesToCachedPuts(
   mlir::ModuleOp synthetic,
   CachedJoinBufferLayoutsByKey* layoutsByKey = nullptr);

static mlir::Value cloneCreateOnlyStepBefore(ExecutionStepOp beforeStep,
                                             ExecutionStepOp createStep,
                                             mlir::Value originalState) {
   mlir::IRMapping mapping;
   auto* cloned = createStep.getOperation()->clone(mapping);
   beforeStep->getBlock()->getOperations().insert(mlir::Block::iterator(beforeStep.getOperation()), cloned);
   mlir::Value mapped = mapping.lookupOrNull(originalState);
   if (!mapped && createStep.getNumResults() == 1) mapped = mlir::cast<ExecutionStepOp>(cloned).getResult(0);
   assert(mapped && "cloned create-only step must map the requested state");
   return mapped;
}

static mlir::Value cloneMergeStepAfter(ExecutionStepOp afterStep,
                                       ExecutionStepOp mergeStep,
                                       mlir::Value originalShadow,
                                       mlir::Value syntheticShadow,
                                       mlir::Value originalFinal) {
   mlir::IRMapping mapping;
   mapping.map(originalShadow, syntheticShadow);
   auto* cloned = mergeStep.getOperation()->clone(mapping);
   afterStep->getBlock()->getOperations().insert(std::next(mlir::Block::iterator(afterStep.getOperation())),
                                                 cloned);
   mlir::Value mapped = mapping.lookupOrNull(originalFinal);
   if (!mapped && mergeStep.getNumResults() == 1) mapped = mlir::cast<ExecutionStepOp>(cloned).getResult(0);
   assert(mapped && "cloned merge step must map the requested final state");
   return mapped;
}

static std::optional<SplitResidualFilter>
residualForOriginalMaterializeTarget(mlir::Value target, const ModuleReuseInfo& reuse) {
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
}

static std::optional<SplitResidualFilter>
residualForOriginalAggregateTarget(mlir::ModuleOp module, mlir::Value target, const ModuleReuseInfo& reuse) {
   mlir::Value buildState = resolveSplitMaterializeBuildState(module, target, reuse);
   std::optional<SplitAggregateBuild> build =
      tryFindUniqueAggregateBuildWritingState(module, buildState);
   if (!build) build = tryFindUniquePlainReduceBuildWritingState(module, buildState);
   if (!build) return std::nullopt;
   return findResidualFilterBeforeReduce(build->suffixStart, build->reduce);
}

struct SyntheticSplitStatePair {
   mlir::Value finalState;
   mlir::Value buildState;
};

struct SplitGroupEntryTarget {
   unsigned slot = 0;
   uint64_t outputKey = 0;
   mlir::Value target;
};

struct PreparedSplitGroupEntry {
   const CrossQueryStateMatchEntry* entry = nullptr;
   SplitGroupEntryTarget resolved;
};

struct SplitSyntheticDonorTarget {
   CacheTarget target;
   mlir::Value buildState;
};

static SplitSyntheticDonorTarget splitSyntheticDonorTargetForGroup(
   mlir::ModuleOp synthetic,
   const CrossQueryStateMatchGroup& group,
   const llvm::DenseMap<uint64_t, CacheTarget>& groupTargetByKey,
   const ModuleReuseInfo& reuseSynthetic) {
   auto itTarget = groupTargetByKey.find(group.cacheKey);
   assert(itTarget != groupTargetByKey.end() && "split-materialize group must have a cloned donor target");
   mlir::Value buildState =
      resolveSplitMaterializeBuildState(synthetic, itTarget->second.state, reuseSynthetic);
   assert(buildState && "split-materialize donor build state must resolve");
   return SplitSyntheticDonorTarget{itTarget->second, buildState};
}

static std::optional<SplitGroupEntryTarget> prepareSplitGroupEntryTarget(
   const CrossQueryStateMatchGroup& group,
   const CrossQueryStateMatchEntry& entry,
   const ReuseRewriteContext& rewriteCtx,
   llvm::ArrayRef<ModuleReuseInfo> reuseEarly,
   size_t numQueries,
   llvm::DenseSet<uint64_t>& emittedOutputKeys) {
   assert(entry.query >= 0 && static_cast<size_t>(entry.query) < numQueries);
   unsigned slot = rewriteCtx.consumerSlot(group.cacheKey, static_cast<unsigned>(entry.query));
   uint64_t outputKey = rewriteCtx.targetCacheKeyForSlot(group.cacheKey, group.requiresSplitMaterialize, slot);
   if (!emittedOutputKeys.insert(outputKey).second) return std::nullopt;

   mlir::Value entryTarget = resolveCacheTargetStateForReuse(entry.state, reuseEarly[entry.query]);
   assert(entryTarget && "split entry target must resolve");
   return SplitGroupEntryTarget{slot, outputKey, entryTarget};
}

static llvm::SmallVector<PreparedSplitGroupEntry, 8> prepareSplitGroupEntries(
   const CrossQueryStateMatchGroup& group,
   const ReuseRewriteContext& rewriteCtx,
   llvm::ArrayRef<ModuleReuseInfo> reuseEarly,
   size_t numQueries) {
   llvm::SmallVector<PreparedSplitGroupEntry, 8> preparedEntries;
   llvm::DenseSet<uint64_t> emittedOutputKeys;
   for (const CrossQueryStateMatchEntry& entry : group.entries) {
      std::optional<SplitGroupEntryTarget> prepared =
         prepareSplitGroupEntryTarget(group, entry, rewriteCtx, reuseEarly, numQueries, emittedOutputKeys);
      if (!prepared) continue;
      preparedEntries.push_back(PreparedSplitGroupEntry{&entry, *prepared});
   }
   return preparedEntries;
}

struct SplitBranchFilterSummary {
   llvm::DenseMap<mlir::Value, llvm::SmallVector<runtime::FilterDescription, 8>> filtersByEntryTarget;
   bool allFiltersIdentical = true;
};

static SplitBranchFilterSummary collectSplitBranchFilters(
   const CrossQueryStateMatchGroup& group,
   llvm::ArrayRef<ModuleReuseInfo> reuseEarly,
   const ReuseRewriteContext& rewriteCtx,
   size_t numQueries) {
   SplitBranchFilterSummary summary;
   std::optional<llvm::SmallVector<runtime::FilterDescription, 8>> firstFilters;
   for (const CrossQueryStateMatchEntry& entry : group.entries) {
      assert(entry.query >= 0 && static_cast<size_t>(entry.query) < numQueries);
      unsigned slot = rewriteCtx.consumerSlot(group.cacheKey, static_cast<unsigned>(entry.query));
      mlir::Value entryTarget = resolveCacheTargetStateForReuse(entry.state, reuseEarly[entry.query]);
      assert(entryTarget && "split branch entry target must resolve");
      llvm::SmallVector<runtime::FilterDescription, 8> filters =
         decodeSplitBranchFiltersForState(entryTarget, reuseEarly[entry.query], slot);
      if (!firstFilters) {
         firstFilters = filters;
      } else if (!runtimeFilterListEquals(*firstFilters, filters)) {
         summary.allFiltersIdentical = false;
      }
      summary.filtersByEntryTarget[entryTarget] = std::move(filters);
   }
   return summary;
}

static llvm::ArrayRef<runtime::FilterDescription>
filtersForSplitBranch(const SplitBranchFilterSummary& summary, mlir::Value entryTarget) {
   if (summary.allFiltersIdentical) return {};
   auto it = summary.filtersByEntryTarget.find(entryTarget);
   assert(it != summary.filtersByEntryTarget.end() && "split branch filters must be decoded for entry target");
   return it->second;
}

struct SplitMaterializeBranchStream {
   mlir::Value stream;
   llvm::StringMap<tuples::ColumnRefAttr> columnsByName;
};

struct SplitMaterializeResidualPlan {
   bool splitResidual = false;
   mlir::Value baseStream;
};

static SplitMaterializeResidualPlan planSplitMaterializeResidualHandling(
   const CrossQueryStateMatchGroup& group,
   llvm::ArrayRef<mlir::ModuleOp> queries,
   llvm::ArrayRef<ModuleReuseInfo> reuseEarly,
   std::optional<SplitResidualFilter> donorResidual,
   mlir::Value originalMaterializeStream) {
   bool anyResidual = false;
   bool anyMissingResidual = false;
   llvm::StringSet<> residualFingerprints;
   for (const CrossQueryStateMatchEntry& entry : group.entries) {
      assert(entry.query >= 0 && static_cast<size_t>(entry.query) < queries.size());
      mlir::Value entryTarget = resolveCacheTargetStateForReuse(entry.state, reuseEarly[entry.query]);
      assert(entryTarget && "split-materialize entry target must resolve");
      if (std::optional<SplitResidualFilter> residual =
             residualForOriginalMaterializeTarget(entryTarget, reuseEarly[entry.query])) {
         anyResidual = true;
         residualFingerprints.insert(residualFilterSemanticFingerprint(*residual));
      } else {
         anyMissingResidual = true;
      }
   }
   bool splitResidual = anyResidual && (anyMissingResidual || residualFingerprints.size() > 1);
   mlir::Value baseStream = (splitResidual && donorResidual) ? donorResidual->inputStream
                                                             : originalMaterializeStream;
   return SplitMaterializeResidualPlan{splitResidual, baseStream};
}

struct SplitMaterializePredSlotPlan {
   llvm::DenseMap<int, unsigned> slotByQuery;
   llvm::SmallVector<unsigned, 8> requiredSlots;
};

static SplitMaterializePredSlotPlan planSplitMaterializePredSlots(
   const CrossQueryStateMatchGroup& group,
   const ReuseRewriteContext& rewriteCtx) {
   SplitMaterializePredSlotPlan plan;
   for (const CrossQueryStateMatchEntry& entry : group.entries) {
      assert(entry.query >= 0 && "split-materialize entry must have query id");
      unsigned slot = rewriteCtx.consumerSlot(group.cacheKey, static_cast<unsigned>(entry.query));
      plan.slotByQuery[entry.query] = slot;
      plan.requiredSlots.push_back(slot);
   }
   return plan;
}

static void addFilterColumnAliasesFromBaseNames(
   llvm::StringMap<tuples::ColumnRefAttr>& columnsByName,
   llvm::ArrayRef<runtime::FilterDescription> filters) {
   for (const runtime::FilterDescription& filter : filters) {
      if (columnsByName.contains(filter.columnName)) continue;
      llvm::StringRef name(filter.columnName);
      size_t dollar = name.find('$');
      if (dollar == llvm::StringRef::npos) continue;
      auto itBase = columnsByName.find(name.take_front(dollar));
      assert(itBase != columnsByName.end() &&
             "split-materialize branch filter column must exist on shared stream");
      columnsByName[filter.columnName] = itBase->second;
   }
}

static SplitMaterializeBranchStream buildSplitMaterializeBranchStream(
   subop::MaterializeOp donorMat,
   mlir::Value baseStream,
   const llvm::StringMap<tuples::ColumnRefAttr>& baseColumnsByName,
   llvm::ArrayRef<runtime::FilterDescription> filters,
   bool splitResidual,
   mlir::Value entryTarget,
   const ModuleReuseInfo& entryReuse,
   std::optional<SplitResidualFilter> donorResidual,
   mlir::Value originalMaterializeStream,
   unsigned branchPredSlot) {
   llvm::StringMap<tuples::ColumnRefAttr> branchColumnsByName = baseColumnsByName;
   addFilterColumnAliasesFromBaseNames(branchColumnsByName, filters);
   assertRuntimeFiltersAvailableOnStream(branchColumnsByName, filters);

   mlir::OpBuilder predBuilder(donorMat);
   predBuilder.setInsertionPoint(donorMat);
   llvm::DenseMap<llvm::StringRef, tuples::ColumnRefAttr> branchColumnView;
   for (auto& kv : branchColumnsByName) branchColumnView[kv.getKey()] = kv.second;
   mlir::Value branchStream = materializeRuntimeFiltersAsSubopFilter(
      predBuilder, donorMat.getLoc(), baseStream, branchColumnView, filters);

   if (splitResidual) {
      std::optional<SplitResidualFilter> residual =
         residualForOriginalMaterializeTarget(entryTarget, entryReuse);
      if (residual) {
         branchStream = cloneResidualFilterBranch(predBuilder, donorMat.getLoc(), branchStream,
                                                  branchColumnsByName, *residual);
         if (donorResidual) {
            branchStream = cloneLinearStreamSuffixBefore(predBuilder, donorResidual->filter.getRes(),
                                                         originalMaterializeStream, branchStream);
         }
      }
   }

   std::string slotPredName = ("filter_pred$" + llvm::Twine(branchPredSlot)).str();
   branchStream = filterSplitBranchByMixedPredMember(
      predBuilder, donorMat.getLoc(), branchStream, slotPredName);
   return SplitMaterializeBranchStream{branchStream, std::move(branchColumnsByName)};
}

static void materializeSplitBranchOutput(subop::MaterializeOp donorMat,
                                         mlir::Value stateArg,
                                         SplitMaterializeBranchStream& branch) {
   if (canonicalizeStateValueForReuse(stateArg) == canonicalizeStateValueForReuse(donorMat.getState())) {
      donorMat->setOperand(0, branch.stream);
      return;
   }

   mlir::OpBuilder builder(donorMat);
   builder.setInsertionPointAfter(donorMat);
   rememberSplitStreamColumnsFromChain(branch.stream, branch.columnsByName);
   auto mapping = remapResultMaterializeMappingToTargetLayoutAndStreamColumns(
      donorMat.getContext(), donorMat.getMapping(), stateArg.getType(), branch.columnsByName);
   builder.create<subop::MaterializeOp>(donorMat.getLoc(), branch.stream, stateArg, mapping);
}

struct SplitAggregateBranchContext {
   std::string slotPredName;
   std::optional<SplitResidualFilter> branchResidual;
   llvm::SmallVector<subop::MapOp, 8> branchMapOps;
   llvm::DenseMap<mlir::Type, unsigned> predSlotByStateType;
};

static mlir::Value buildSplitAggregateBranchStream(SplitAggregateBuild build,
                                                   mlir::Value suffixStart,
                                                   llvm::ArrayRef<runtime::FilterDescription> filters) {
   mlir::Value branchStream = suffixStart;
   if (!build.scan || filters.empty()) return branchStream;

   auto [predStream, predRef] = materializeRuntimeFiltersAsPredicateColumnAfterScanRefs(
      build.scan, filters, "split_agg_pred", /*rewireDownstreamUses=*/false);
   mlir::Operation* predAnchor = predStream.getDefiningOp();
   assert(predAnchor && "split-aggregate predicate stream must be op-defined");
   mlir::OpBuilder b(predAnchor);
   b.setInsertionPointAfter(predAnchor);
   auto filter = b.create<subop::FilterOp>(build.scan.getLoc(), predStream,
                                           subop::FilterSemantic::all_true,
                                           b.getArrayAttr({predRef}));
   return filter.getRes();
}

static SplitAggregateBranchContext prepareSplitAggregateBranchContext(
   const CrossQueryStateMatchGroup& group,
   const CrossQueryStateMatchEntry& entry,
   mlir::Value entryTarget,
   unsigned slot,
   mlir::Value suffixStart,
   const llvm::DenseMap<mlir::Type, uint64_t>& cacheKeyByStateType,
   const ReuseRewriteContext& rewriteCtx,
   llvm::ArrayRef<mlir::ModuleOp> queries,
   llvm::ArrayRef<ModuleReuseInfo> reuseEarly) {
   SplitAggregateBranchContext ctx;
   for (uint64_t depKey : group.cacheDeps) {
      std::optional<unsigned> depSlot =
         rewriteCtx.lookupConsumerSlot(depKey, static_cast<unsigned>(entry.query));
      if (!depSlot) continue;
      for (auto& [stateType, key] : cacheKeyByStateType) {
         if (key == depKey) ctx.predSlotByStateType[stateType] = *depSlot;
      }
   }

   unsigned predSlot =
      rewriteCtx.inheritedConsumerSlot(group.cacheDeps, static_cast<unsigned>(entry.query)).value_or(slot);
   if (auto scan = mlir::dyn_cast_or_null<subop::ScanListOp>(suffixStart.getDefiningOp())) {
      if (auto itSlot = ctx.predSlotByStateType.find(scanListLookupStateType(scan));
          itSlot != ctx.predSlotByStateType.end()) {
         predSlot = itSlot->second;
      }
   }
   ctx.slotPredName = ("filter_pred$" + llvm::Twine(predSlot)).str();
   ctx.branchResidual =
      residualForOriginalAggregateTarget(queries[entry.query], entryTarget, reuseEarly[entry.query]);

   mlir::Value entryBuildState = resolveSplitMaterializeBuildState(queries[entry.query], entryTarget,
                                                                   reuseEarly[entry.query]);
   SplitAggregateBuild entryAggBuild =
      findUniqueAggregateBuildWritingState(queries[entry.query], entryBuildState);
   ctx.branchMapOps = linearMapOpsBeforeReduce(entryAggBuild.suffixStart, entryAggBuild.reduce);
   return ctx;
}

static CloneAggregateSuffixOptions buildCloneAggregateSuffixOptions(
   SplitAggregateBuild build,
   llvm::ArrayRef<runtime::FilterDescription> filters,
   std::optional<SplitResidualFilter> donorResidual,
   const SplitAggregateBranchContext& branchCtx) {
   CloneAggregateSuffixOptions options;
   options.filters = build.scan ? llvm::ArrayRef<runtime::FilterDescription>{} : filters;
   options.mixedPredMemberName = branchCtx.slotPredName;
   options.donorResidualToReplace = donorResidual;
   options.branchResidual = branchCtx.branchResidual;
   options.branchMapOps = branchCtx.branchMapOps;
   options.predSlotByStateType = branchCtx.predSlotByStateType;
   return options;
}

static void replaceSplitGroupTargetsWithExpandedOutputs(
   const llvm::DenseSet<uint64_t>& splitGroupKeys,
   llvm::ArrayRef<CacheTarget> expanded,
   llvm::SmallVectorImpl<CacheTarget>& targetsSynthetic) {
   if (splitGroupKeys.empty()) return;
   llvm::SmallVector<CacheTarget, 64> kept;
   kept.reserve(targetsSynthetic.size() + expanded.size());
   for (const CacheTarget& target : targetsSynthetic) {
      if (splitGroupKeys.contains(target.cacheKey)) continue;
      kept.push_back(target);
   }
   kept.append(expanded.begin(), expanded.end());
   targetsSynthetic.assign(kept.begin(), kept.end());
}

static SyntheticSplitStatePair materializeSyntheticSplitStateForEntry(
   mlir::ModuleOp synthetic,
   mlir::Value entryTarget,
   const ModuleReuseInfo& entryReuse,
   const ModuleReuseInfo& syntheticReuse,
   mlir::IRMapping& donorMapping,
   ExecutionStepOp insertCreateBeforeStep) {
   mlir::Value syntheticFinalState = donorMapping.lookupOrNull(entryTarget);
   mlir::Value syntheticBuildState;
   if (syntheticFinalState) {
      syntheticBuildState = resolveSplitMaterializeBuildState(synthetic, syntheticFinalState, syntheticReuse);
      return SyntheticSplitStatePair{syntheticFinalState, syntheticBuildState};
   }

   mlir::Value entryBuildState = entryTarget;
   if (auto itShadow = findReuseMap(entryReuse.mergedFromShadowState, entryTarget);
       itShadow != entryReuse.mergedFromShadowState.end()) {
      entryBuildState = itShadow->second;
   }
   auto itCreate = findReuseMap(entryReuse.createOnlyStepForState, entryBuildState);
   assert(itCreate != entryReuse.createOnlyStepForState.end() &&
          "split peer build state needs a create-only step");
   syntheticBuildState = cloneCreateOnlyStepBefore(insertCreateBeforeStep, itCreate->second, entryBuildState);

   if (canonicalizeStateValueForReuse(entryBuildState) == canonicalizeStateValueForReuse(entryTarget)) {
      syntheticFinalState = syntheticBuildState;
   } else {
      auto itMerge = findReuseMap(entryReuse.writerStepsByState, entryTarget);
      assert(itMerge != entryReuse.writerStepsByState.end() && itMerge->second.size() == 1 &&
             "split peer final state needs one merge writer");
      syntheticFinalState =
         cloneMergeStepAfter(insertCreateBeforeStep, itMerge->second.front(), entryBuildState,
                             syntheticBuildState, entryTarget);
   }
   return SyntheticSplitStatePair{syntheticFinalState, syntheticBuildState};
}

static void expandSplitAggregateGroupInSynthetic(
   mlir::ModuleOp synthetic,
   llvm::ArrayRef<mlir::ModuleOp> queries,
   const CrossQueryStateMatchGroup& group,
   mlir::Value donorBuildState,
   const ModuleReuseInfo& reuseSynthetic,
   llvm::ArrayRef<ModuleReuseInfo> reuseEarly,
   llvm::MutableArrayRef<mlir::IRMapping> donorMappings,
   const ReuseRewriteContext& rewriteCtx,
   const llvm::DenseMap<mlir::Type, uint64_t>& cacheKeyByStateType,
   llvm::SmallVectorImpl<CacheTarget>& expanded) {
   SplitAggregateBuild aggBuild = findUniqueAggregateBuildWritingState(synthetic, donorBuildState);
   subop::ScanRefsOp aggregateSourceScan = aggBuild.scan;
   if (!aggregateSourceScan) {
      aggregateSourceScan = tryFindFirstScanRefsInStep(topLevelExecutionStepFor(aggBuild.step));
   }
   SplitBranchFilterSummary branchFilters =
      collectSplitBranchFilters(group, reuseEarly, rewriteCtx, queries.size());
   if (aggregateSourceScan && !branchFilters.allFiltersIdentical) {
      clearGetExternalFiltersForState(canonicalizeStateValueForReuse(aggregateSourceScan.getState()));
   }

   mlir::Value suffixStart = aggBuild.suffixStart;
   std::optional<SplitResidualFilter> donorResidual =
      findResidualFilterBeforeReduce(suffixStart, aggBuild.reduce);
   llvm::SmallVector<PreparedSplitGroupEntry, 8> preparedEntries =
      prepareSplitGroupEntries(group, rewriteCtx, reuseEarly, queries.size());

   for (const PreparedSplitGroupEntry& prepared : preparedEntries) {
      const CrossQueryStateMatchEntry& entry = *prepared.entry;

      SyntheticSplitStatePair splitState = materializeSyntheticSplitStateForEntry(
         synthetic, prepared.resolved.target, reuseEarly[entry.query], reuseSynthetic, donorMappings[entry.query],
         topLevelExecutionStepFor(aggBuild.step));
      mlir::Value stateArg = stateToNestedBuildStep(aggBuild.step, splitState.buildState);
      llvm::ArrayRef<runtime::FilterDescription> filters =
         filtersForSplitBranch(branchFilters, prepared.resolved.target);
      mlir::Value branchStream = buildSplitAggregateBranchStream(aggBuild, suffixStart, filters);

      mlir::Operation* lookupAnchor = aggBuild.lookup ? aggBuild.lookup.getOperation()
                                                      : aggBuild.plainLookup.getOperation();
      assert(lookupAnchor && "split reduce build must have a lookup insertion anchor");
      mlir::OpBuilder builder(lookupAnchor);
      builder.setInsertionPoint(lookupAnchor);
      SplitAggregateBranchContext branchCtx = prepareSplitAggregateBranchContext(
         group, entry, prepared.resolved.target, prepared.resolved.slot, suffixStart, cacheKeyByStateType, rewriteCtx, queries,
         reuseEarly);
      CloneAggregateSuffixOptions cloneOptions =
         buildCloneAggregateSuffixOptions(aggBuild, filters, donorResidual, branchCtx);
      cloneAggregateSuffixToReduce(builder, suffixStart, aggBuild.reduce, branchStream, stateArg, cloneOptions);
      expanded.push_back(CacheTarget{splitState.finalState, prepared.resolved.outputKey,
                                     /*enableFilterPredReuse=*/false});
   }

   eraseOriginalAggregateSuffix(suffixStart, aggBuild.reduce);
}

static void expandSplitMaterializeGroupInSynthetic(
   mlir::ModuleOp synthetic,
   llvm::ArrayRef<mlir::ModuleOp> queries,
   const CrossQueryStateMatchGroup& group,
   mlir::Value donorBuildState,
   const ModuleReuseInfo& reuseSynthetic,
   llvm::ArrayRef<ModuleReuseInfo> reuseEarly,
   llvm::MutableArrayRef<mlir::IRMapping> donorMappings,
   const ReuseRewriteContext& rewriteCtx,
   llvm::SmallVectorImpl<CacheTarget>& expanded) {
   ExecutionStepOp matStep = findUniqueMaterializeStepWritingState(synthetic, donorBuildState);
   subop::MaterializeOp donorMat = findUniqueMaterializeWritingState(matStep, donorBuildState);
   subop::ScanRefsOp scanRefs = findFirstScanRefsInStep(matStep);
   ExecutionStepOp donorMaterializeOwnerStep = donorMat->getParentOfType<ExecutionStepOp>();
   assert(donorMaterializeOwnerStep && "split-materialize materialize op must be inside an execution_step");

   mlir::Value originalMaterializeStream = donorMat.getStream();
   std::optional<SplitResidualFilter> donorResidual =
      findResidualFilterBeforeMaterialize(donorMaterializeOwnerStep, donorMat);
   SplitMaterializeResidualPlan residualPlan =
      planSplitMaterializeResidualHandling(group, queries, reuseEarly, donorResidual, originalMaterializeStream);
   SplitMaterializePredSlotPlan predSlotPlan = planSplitMaterializePredSlots(group, rewriteCtx);
   ensureSplitMaterializeMixedPredCarriers(synthetic, predSlotPlan.requiredSlots);

   auto colByName = materializeStreamColumnsByFilterName(matStep, donorMat);
   SplitBranchFilterSummary branchFilters =
      collectSplitBranchFilters(group, reuseEarly, rewriteCtx, queries.size());
   if (!branchFilters.allFiltersIdentical) {
      clearGetExternalFiltersForState(canonicalizeStateValueForReuse(scanRefs.getState()));
   }

   llvm::SmallVector<PreparedSplitGroupEntry, 8> preparedEntries =
      prepareSplitGroupEntries(group, rewriteCtx, reuseEarly, queries.size());
   for (const PreparedSplitGroupEntry& prepared : preparedEntries) {
      const CrossQueryStateMatchEntry& entry = *prepared.entry;
      unsigned branchPredSlot = predSlotPlan.slotByQuery.lookup(entry.query);
      SyntheticSplitStatePair splitState = materializeSyntheticSplitStateForEntry(
         synthetic, prepared.resolved.target, reuseEarly[entry.query], reuseSynthetic, donorMappings[entry.query], matStep);

      mlir::Value stateArg = threadStateToNestedMaterializeStep(donorMaterializeOwnerStep,
                                                                splitState.buildState);
      llvm::ArrayRef<runtime::FilterDescription> filters =
         filtersForSplitBranch(branchFilters, prepared.resolved.target);
      SplitMaterializeBranchStream branch = buildSplitMaterializeBranchStream(
         donorMat, residualPlan.baseStream, colByName, filters, residualPlan.splitResidual, prepared.resolved.target,
         reuseEarly[entry.query], donorResidual, originalMaterializeStream, branchPredSlot);
      materializeSplitBranchOutput(donorMat, stateArg, branch);
      expanded.push_back(CacheTarget{splitState.finalState, prepared.resolved.outputKey,
                                     /*enableFilterPredReuse=*/false});
   }
}

static llvm::DenseMap<uint64_t, CacheTarget>
cacheTargetsByKey(llvm::ArrayRef<CacheTarget> targets) {
   llvm::DenseMap<uint64_t, CacheTarget> byKey;
   for (const CacheTarget& target : targets) byKey[target.cacheKey] = target;
   return byKey;
}

static llvm::DenseMap<mlir::Type, uint64_t>
cacheGetKeysByStateType(mlir::ModuleOp module) {
   llvm::DenseMap<mlir::Type, uint64_t> byType;
   module.walk([&](subop::CacheGetOp get) {
      byType[get.getResult().getType()] = static_cast<uint64_t>(get.getKey());
   });
   return byType;
}

struct EffectiveGroupRewriteFlags {
   bool enableFilterPredReuse = false;
   bool requiresJoinLayoutUnion = false;
};

struct GroupRewriteDecision {
   const CrossQueryStateMatchEntry* donor = nullptr;
   bool usesSplitMaterialize = false;
   bool unsupported = false;
};

struct ResolvedBatchGroupEntry {
   const CrossQueryStateMatchEntry* entry = nullptr;
   unsigned query = 0;
   mlir::Value targetState;
};

struct BatchDonorGroup {
   int donorQuery = -1;
   mlir::Value donorState;
   CacheTarget donorTarget;
   bool isBuildStepReuse = false;
};

static llvm::SmallVector<ResolvedBatchGroupEntry, 8> collectResolvedBatchGroupEntries(
   const CrossQueryStateMatchGroup& group,
   llvm::ArrayRef<mlir::ModuleOp> queries,
   llvm::ArrayRef<ModuleReuseInfo> reuseEarly) {
   llvm::SmallVector<ResolvedBatchGroupEntry, 8> entries;
   for (const CrossQueryStateMatchEntry& e : group.entries) {
      if (e.query < 0 || static_cast<size_t>(e.query) >= queries.size()) continue;
      if (!e.state) continue;
      mlir::Value targetState = resolveCacheTargetStateForReuse(e.state, reuseEarly[e.query]);
      assert(targetState && "batch reuse group entry target must resolve");
      assert(!mlir::isa<ThreadLocalType>(targetState.getType()) &&
             "batch reuse must never target thread_local-wrapped states");
      entries.push_back(ResolvedBatchGroupEntry{&e, static_cast<unsigned>(e.query), targetState});
   }
   return entries;
}

static EffectiveGroupRewriteFlags computeEffectiveGroupRewriteFlags(
   const CrossQueryStateMatchGroup& group,
   llvm::ArrayRef<ResolvedBatchGroupEntry> entries,
   llvm::ArrayRef<mlir::ModuleOp> queries,
   llvm::ArrayRef<ModuleReuseInfo> reuseEarly) {
   EffectiveGroupRewriteFlags flags{group.enableFilterPredReuse, group.requiresJoinLayoutUnion};
   if (!group.enableFilterPredReuse) return flags;

   const ResolvedBatchGroupEntry* donor = nullptr;
   bool allHiv = true;
   for (const ResolvedBatchGroupEntry& e : entries) {
      if (!mlir::isa<HashIndexedViewType>(e.targetState.getType())) {
         allHiv = false;
         break;
      }
      if (!donor || e.query < donor->query) donor = &e;
   }
   if (!allHiv || !donor) return flags;

   bool allPeerFiltersIdentical = true;
   for (const ResolvedBatchGroupEntry& e : entries) {
      if (&e == donor) continue;
      if (!joinMatchPeerExternalFiltersIdentical(queries[donor->query], queries[e.query],
                                                 donor->entry->state, e.entry->state,
                                                 reuseEarly[donor->query], reuseEarly[e.query])) {
         allPeerFiltersIdentical = false;
         break;
      }
   }
   if (allPeerFiltersIdentical) flags.enableFilterPredReuse = false;
   flags.requiresJoinLayoutUnion = group.requiresJoinLayoutUnion || flags.enableFilterPredReuse;
   return flags;
}

static GroupRewriteDecision decideGroupRewrite(
   const CrossQueryStateMatchGroup& group,
   llvm::ArrayRef<ResolvedBatchGroupEntry> entries,
   llvm::ArrayRef<mlir::ModuleOp> queries,
   llvm::ArrayRef<ModuleReuseInfo> reuseEarly) {
   GroupRewriteDecision decision;
   decision.usesSplitMaterialize = group.requiresSplitMaterialize;
   for (const ResolvedBatchGroupEntry& e : entries) {
      if (mlir::isa<subop::PreAggrHtType>(e.targetState.getType())) {
         if (decision.usesSplitMaterialize) {
            if (!splitAggregateBuildRewriteSupported(queries[e.query], e.targetState, reuseEarly[e.query])) {
               decision.unsupported = true;
               break;
            }
         } else if (!group.cacheDeps.empty()) {
            decision.usesSplitMaterialize = true;
         } else if (!aggregateHashTablePayloadUnionSupported(e.entry->state, reuseEarly[e.query])) {
            decision.usesSplitMaterialize = true;
         }
      }
      if (!decision.donor || static_cast<int>(e.query) < decision.donor->query) decision.donor = e.entry;
   }
   if (group.requiresJoinLayoutUnion) decision.usesSplitMaterialize = false;
   if (decision.unsupported) return decision;

   if (decision.usesSplitMaterialize && !group.requiresSplitMaterialize) {
      for (const ResolvedBatchGroupEntry& e : entries) {
         if (!mlir::isa<subop::PreAggrHtType>(e.targetState.getType())) continue;
         if (!splitAggregateBuildRewriteSupported(queries[e.query], e.targetState, reuseEarly[e.query])) {
            decision.unsupported = true;
            break;
         }
      }
   }
   if (decision.unsupported || !decision.usesSplitMaterialize || !decision.donor) return decision;

   mlir::Value donorTarget;
   for (const ResolvedBatchGroupEntry& e : entries) {
      if (e.entry == decision.donor) {
         donorTarget = e.targetState;
         break;
      }
   }
   assert(donorTarget && "batch reuse donor target must have resolved entry");
   if (!mlir::isa<subop::PreAggrHtType>(donorTarget.getType())) return decision;

   std::optional<llvm::SmallVector<std::string, 8>> donorFp =
      aggregateSplitMapFingerprint(queries[decision.donor->query], donorTarget,
                                   reuseEarly[decision.donor->query]);
   assert(donorFp && "split aggregate donor must have map fingerprints");
   for (const ResolvedBatchGroupEntry& e : entries) {
      if (!mlir::isa<subop::PreAggrHtType>(e.targetState.getType())) continue;
      std::optional<llvm::SmallVector<std::string, 8>> fp =
         aggregateSplitMapFingerprint(queries[e.query], e.targetState, reuseEarly[e.query]);
      if (!fp || *fp != *donorFp) {
         decision.unsupported = true;
         break;
      }
   }
   return decision;
}

static void registerBatchConsumerTarget(
   BatchReusePlanRewriteResult& res,
   llvm::MutableArrayRef<llvm::SmallVector<CacheTarget, 16>> targetsByQuery,
   ReuseRewriteContext& rewriteCtx,
   const CrossQueryStateMatchGroup& group,
   const ResolvedBatchGroupEntry& entry,
   unsigned splitSlot,
   const EffectiveGroupRewriteFlags& flags,
   const GroupRewriteDecision& decision) {
   uint64_t targetCacheKey = rewriteCtx.registerConsumerTargetSlot(
      group.cacheKey, decision.usesSplitMaterialize, entry.query, splitSlot);
   targetsByQuery[entry.query].push_back(CacheTarget{entry.targetState, targetCacheKey, flags.enableFilterPredReuse});
   incrementBatchTargetCategoryCounts(res, entry.query, entry.targetState,
                                      decision.usesSplitMaterialize);
}

static void activateBatchRewriteGroup(
   llvm::SmallVectorImpl<CrossQueryStateMatchGroup>& activeRewriteGroups,
   llvm::SmallVectorImpl<BatchDonorGroup>& donorGroups,
   const CrossQueryStateMatchGroup& group,
   const GroupRewriteDecision& decision,
   const EffectiveGroupRewriteFlags& flags,
   mlir::Value donorTargetState,
   const llvm::DenseMap<unsigned, unsigned>& slotByQuery,
   size_t numQueries) {
   CrossQueryStateMatchGroup groupForRewrite = group;
   groupForRewrite.enableFilterPredReuse = flags.enableFilterPredReuse;
   groupForRewrite.requiresJoinLayoutUnion = flags.requiresJoinLayoutUnion;
   groupForRewrite.requiresSplitMaterialize = decision.usesSplitMaterialize;
   for (CrossQueryStateMatchEntry& entry : groupForRewrite.entries) {
      if (entry.query < 0 || static_cast<size_t>(entry.query) >= numQueries) continue;
      entry.reuseSlot = slotByQuery.lookup(static_cast<unsigned>(entry.query));
   }
   activeRewriteGroups.push_back(groupForRewrite);
   donorGroups.push_back(BatchDonorGroup{
      decision.donor->query, decision.donor->state,
      CacheTarget{donorTargetState, group.cacheKey, flags.enableFilterPredReuse}, decision.usesSplitMaterialize});
}

static void cloneBatchDonorStepsIntoSynthetic(
   BatchReusePlanRewriteResult& res,
   ExecutionGroupOp syntheticGroup,
   llvm::ArrayRef<mlir::ModuleOp> queries,
   llvm::ArrayRef<ModuleReuseInfo> reuseEarly,
   llvm::ArrayRef<BatchDonorGroup> donorGroups,
   llvm::MutableArrayRef<mlir::IRMapping> donorMappings,
   llvm::SmallVectorImpl<CacheTarget>& targetsSynthetic) {
   for (size_t qi = 0; qi < queries.size(); ++qi) {
      llvm::SmallVector<CacheTarget, 16> donorTargets;
      for (const BatchDonorGroup& dg : donorGroups) {
         if (static_cast<size_t>(dg.donorQuery) == qi) donorTargets.push_back(dg.donorTarget);
      }
      if (donorTargets.empty()) continue;

      ExecutionGroupOp donorGroup = getSingleExecutionGroup(queries[qi]);
      llvm::DenseSet<mlir::Value> neededStates = expandNeededStatesFromTargets(donorTargets, reuseEarly[qi]);
      llvm::SmallVector<ExecutionStepOp, 32> stepsToClone =
         collectCreateAndWriteStepsForStates(donorGroup, reuseEarly[qi], neededStates);
      stepsToClone = augmentStepsWithOperandProducerClosure(donorGroup, stepsToClone);
      donorMappings[qi] = cloneExecutionStepsToQuery0(syntheticGroup, stepsToClone);

      for (const BatchDonorGroup& dg : donorGroups) {
         if (static_cast<size_t>(dg.donorQuery) != qi) continue;
         const CacheTarget& target = dg.donorTarget;
         mlir::Value mapped = lookupMappedReuseTarget(donorMappings[qi], target.state, reuseEarly[qi]);
         assert(mapped && "batch reuse target must map into synthetic module");
         registerSyntheticMappedTarget(res, targetsSynthetic, mapped, target, dg.isBuildStepReuse);
      }
   }
}

static BatchReusePlanRewriteResult rewritePlansWithSyntheticQueryBatchOnce(
   llvm::ArrayRef<mlir::ModuleOp> queries,
   llvm::ArrayRef<CrossQueryStateMatchGroup> groups,
   lingodb::catalog::Catalog* catalog) {
   (void)catalog;
   assert(queries.size() >= 2 && "batch reuse needs at least two queries");

   BatchReusePlanRewriteResult res;
   initBatchReuseCounters(res, queries.size());

   llvm::SmallVector<ModuleReuseInfo, 8> reuseEarly;
   reuseEarly.reserve(queries.size());
   for (mlir::ModuleOp q : queries) reuseEarly.push_back(collectModuleReuseInfo(q));

   llvm::SmallVector<llvm::SmallVector<CacheTarget, 16>, 8> targetsByQuery(queries.size());
   llvm::SmallVector<BatchDonorGroup, 32> donorGroups;
   llvm::SmallVector<CrossQueryStateMatchGroup, 64> activeRewriteGroups;
   activeRewriteGroups.reserve(groups.size());
   ReuseRewriteContext rewriteCtx;
   for (size_t qi = 0; qi < queries.size(); ++qi)
      rewriteCtx.recordConsumerSlotsFromModule(queries[qi], static_cast<unsigned>(qi));

   for (const CrossQueryStateMatchGroup& g : groups) {
      if (g.entries.size() < 2) continue;
      llvm::SmallVector<ResolvedBatchGroupEntry, 8> resolvedEntries =
         collectResolvedBatchGroupEntries(g, queries, reuseEarly);
      if (resolvedEntries.size() < 2) continue;
      EffectiveGroupRewriteFlags flags =
         computeEffectiveGroupRewriteFlags(g, resolvedEntries, queries, reuseEarly);
      GroupRewriteDecision decision = decideGroupRewrite(g, resolvedEntries, queries, reuseEarly);
      if (decision.unsupported) continue;
      if (!decision.donor) continue;

      mlir::Value donorTargetState;
      for (const ResolvedBatchGroupEntry& entry : resolvedEntries) {
         if (entry.entry == decision.donor) {
            donorTargetState = entry.targetState;
            break;
         }
      }
      assert(donorTargetState && "batch reuse donor target must resolve");

      CrossQueryStateMatchGroup groupForSlots = g;
      groupForSlots.enableFilterPredReuse = flags.enableFilterPredReuse;
      groupForSlots.requiresJoinLayoutUnion = flags.requiresJoinLayoutUnion;
      groupForSlots.requiresSplitMaterialize = decision.usesSplitMaterialize;
      llvm::DenseMap<unsigned, unsigned> slotByQuery =
         assignReuseSlotsForGroup(rewriteCtx, groupForSlots, reuseEarly, queries.size());
      activateBatchRewriteGroup(activeRewriteGroups, donorGroups, g, decision, flags, donorTargetState,
                                slotByQuery, queries.size());

      for (const ResolvedBatchGroupEntry& e : resolvedEntries) {
         unsigned splitSlot = slotByQuery.lookup(e.query);
         registerBatchConsumerTarget(res, targetsByQuery, rewriteCtx, g, e, splitSlot, flags,
                                     decision);
      }
   }
   llvm::SmallVector<CrossQueryStateMatchGroup, 64> rewriteGroups = std::move(activeRewriteGroups);
   rewriteCtx.inheritSlotsFromGroups(rewriteGroups);

   for (size_t i = 0; i < targetsByQuery.size(); ++i) {
      res.numTargetsPerQuery[i] = targetsByQuery[i].size();
      res.numTargetsNoTablePerQuery[i] = countNonTableTargets(targetsByQuery[i]);
   }
   if (donorGroups.empty()) return res;

   mlir::ModuleOp firstQuery = queries.front();
   SyntheticModuleShell synthShell = createSyntheticModuleShellLike(firstQuery);
   res.synthetic = std::move(synthShell.module);
   ExecutionGroupOp synthGroup = synthShell.group;

   llvm::SmallVector<CacheTarget, 64> targetsSynthetic;
   llvm::SmallVector<mlir::IRMapping, 8> donorMappings(queries.size());

   cloneBatchDonorStepsIntoSynthetic(res, synthGroup, queries, reuseEarly, donorGroups, donorMappings,
                                     targetsSynthetic);

   res.numTargetsSyntheticMapped = targetsSynthetic.size();
   res.numTargetsSyntheticMappedNoTable = countNonTableTargets(targetsSynthetic);
   if (targetsSynthetic.empty()) return res;

   CachedJoinBufferLayoutsByKey producerLayoutsByKey;
   CachedAggregateLayoutsByKey aggregateLayoutsByKey;

   extendSyntheticJoinBuffersToColumnUnionForGroups(*res.synthetic, queries, rewriteGroups, targetsSynthetic,
                                                    &producerLayoutsByKey);

   extendSyntheticAggregateHashTablesToPayloadUnionForGroups(*res.synthetic, queries, rewriteGroups,
                                                             targetsSynthetic, &aggregateLayoutsByKey);

   auto reuseSyntheticAfterLayoutPrep = collectModuleReuseInfo(*res.synthetic);
   ClonedJoinBufferBuildSitesByKey joinBuildSites =
      recordClonedJoinBufferBuildSites(*res.synthetic, targetsSynthetic, reuseSyntheticAfterLayoutPrep);
   insertSyntheticFilterPredsAfterColumnUnionForGroups(*res.synthetic, queries, rewriteGroups, targetsSynthetic,
                                                       producerLayoutsByKey, joinBuildSites);

   bool hasSplitMaterializeGroup = llvm::any_of(rewriteGroups, [](const CrossQueryStateMatchGroup& g) {
      return g.requiresSplitMaterialize;
   });
   expandSplitMaterializeTargetsInSynthetic(*res.synthetic, queries, rewriteGroups, reuseEarly,
                                            donorMappings, rewriteCtx, targetsSynthetic);
   if (hasSplitMaterializeGroup) {
      syncProbeGatherMappingsInModule(*res.synthetic);
      synchronizeExecutionStepPortTypes(*res.synthetic, nullptr);
   }

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
                                                        &rewriteCtx.inheritedMixedDepsByCacheKey,
                                                        &rewriteCtx.consumerSlotByCacheKeyAndQuery);
      rewriteCtx.inheritSlotsFromSingleMixedDeps();
      rewriteCtx.inheritSlotsFromGroups(rewriteGroups);
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
            unsigned slot = consumerSlotForJoinLayout(rewriteCtx, it->second, t.cacheKey,
                                                      static_cast<unsigned>(qi));
            alignConsumerModulesToCachedJoinLayout(queries[qi], it->second, t.cacheKey, slot, nullptr);
         }
      }
      for (const CacheTarget& t : targetsByQuery[qi]) {
         resyncConsumerCachedHivCarrierTypesFromCacheGet(queries[qi], t.cacheKey);
      }
      rewriteCtx.recordConsumerSlotsFromModule(queries[qi], static_cast<unsigned>(qi));
   }
   rewriteCtx.inheritSlotsFromSingleMixedDeps();
   rewriteCtx.inheritSlotsFromGroups(rewriteGroups);

   llvm::SmallVector<llvm::SmallVector<ConsumerCacheGetProbeClosure, 4>, 8> probeClosuresByQuery(queries.size());
   for (size_t qi = 0; qi < queries.size(); ++qi) {
      for (const CacheTarget& t : targetsByQuery[qi]) {
         if (auto it = producerLayoutsByKey.find(t.cacheKey); it != producerLayoutsByKey.end()) {
            unsigned slot = consumerSlotForJoinLayout(rewriteCtx, it->second, t.cacheKey,
                                                      static_cast<unsigned>(qi));
            std::optional<unsigned> consumerQ = slot;
            alignConsumerModulesToCachedJoinLayout(queries[qi], it->second, t.cacheKey, consumerQ,
                                                   &probeClosuresByQuery[qi]);
         }
         if (auto it = aggregateLayoutsByKey.find(t.cacheKey); it != aggregateLayoutsByKey.end()) {
            unsigned slot = rewriteCtx.consumerSlot(t.cacheKey, static_cast<unsigned>(qi));
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

   if (res.synthetic) {
      alignSyntheticCacheGetDependenciesToCachedPuts(*res.synthetic, &producerLayoutsByKey);
      alignDirectSyntheticHivGathersToCachedLayouts(*res.synthetic, producerLayoutsByKey);
   }
   res.cachedJoinLayouts = std::move(producerLayoutsByKey);

   return res;
}

static void expandSplitMaterializeTargetsInSynthetic(
   mlir::ModuleOp synthetic,
   llvm::ArrayRef<mlir::ModuleOp> queries,
   llvm::ArrayRef<CrossQueryStateMatchGroup> groups,
   llvm::ArrayRef<ModuleReuseInfo> reuseEarly,
   llvm::MutableArrayRef<mlir::IRMapping> donorMappings,
   const ReuseRewriteContext& rewriteCtx,
   llvm::SmallVectorImpl<CacheTarget>& targetsSynthetic) {
   llvm::DenseMap<uint64_t, CacheTarget> groupTargetByKey = cacheTargetsByKey(targetsSynthetic);
   llvm::DenseSet<uint64_t> splitGroupKeys;
   llvm::SmallVector<CacheTarget, 64> expanded;
   ModuleReuseInfo reuseSynthetic = collectModuleReuseInfo(synthetic);
   llvm::DenseMap<mlir::Type, uint64_t> cacheKeyByStateType = cacheGetKeysByStateType(synthetic);

   for (const CrossQueryStateMatchGroup& g : groups) {
      if (!g.requiresSplitMaterialize) continue;
      splitGroupKeys.insert(g.cacheKey);
      SplitSyntheticDonorTarget donor =
         splitSyntheticDonorTargetForGroup(synthetic, g, groupTargetByKey, reuseSynthetic);
      if (!tryFindUniqueMaterializeStepWritingState(synthetic, donor.buildState)) {
         expandSplitAggregateGroupInSynthetic(synthetic, queries, g, donor.buildState, reuseSynthetic, reuseEarly,
                                              donorMappings, rewriteCtx, cacheKeyByStateType, expanded);
         continue;
      }
      expandSplitMaterializeGroupInSynthetic(synthetic, queries, g, donor.buildState, reuseSynthetic, reuseEarly,
                                             donorMappings, rewriteCtx, expanded);
   }

   replaceSplitGroupTargetsWithExpandedOutputs(splitGroupKeys, expanded, targetsSynthetic);
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

static subop::HashIndexedViewType asHashIndexedViewLayoutTypeLocal(mlir::Type type) {
   if (auto hiv = mlir::dyn_cast<subop::HashIndexedViewType>(type)) return hiv;
   if (auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(type)) {
      return subop::HashIndexedViewType::get(mixed.getContext(), mixed.getKeyMembers(), mixed.getValueMembers(),
                                             mixed.getCompareHashForLookup());
   }
   return nullptr;
}

static mlir::Type demoteMixedHivCarrierType(mlir::MLIRContext* ctx, mlir::Type type) {
   if (!type) return type;
   if (auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(type)) {
      return subop::HashIndexedViewType::get(ctx, mixed.getKeyMembers(), mixed.getValueMembers(),
                                             mixed.getCompareHashForLookup());
   }
   if (auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(type)) {
      mlir::Type demotedState = demoteMixedHivCarrierType(ctx, ler.getState());
      if (demotedState == ler.getState()) return type;
      return subop::LookupEntryRefType::get(ctx, mlir::cast<subop::LookupAbleState>(demotedState));
   }
   if (auto list = mlir::dyn_cast<subop::ListType>(type)) {
      mlir::Type demotedElem = demoteMixedHivCarrierType(ctx, list.getT());
      if (demotedElem == list.getT()) return type;
      return subop::ListType::get(ctx, mlir::cast<subop::StateEntryReference>(demotedElem));
   }
   return type;
}

static bool demoteMixedHivColumnRef(tuples::ColumnRefAttr ref) {
   mlir::Type demoted = demoteMixedHivCarrierType(ref.getContext(), ref.getColumn().type);
   if (demoted == ref.getColumn().type) return false;
   ref.getColumn().type = demoted;
   return true;
}

static bool demoteMixedHivColumnDef(tuples::ColumnDefAttr def) {
   mlir::Type demoted = demoteMixedHivCarrierType(def.getContext(), def.getColumn().type);
   if (demoted == def.getColumn().type) return false;
   def.getColumn().type = demoted;
   return true;
}

static void demoteMixedHivCarriersInSyntheticProducer(mlir::ModuleOp synthetic) {
   auto* ctx = synthetic.getContext();
   for (;;) {
      bool changed = false;
      synthetic.walk([&](mlir::Operation* op) {
         for (mlir::Value result : op->getResults()) {
            mlir::Type demoted = demoteMixedHivCarrierType(ctx, result.getType());
            if (demoted != result.getType()) {
               result.setType(demoted);
               changed = true;
            }
         }
         for (mlir::Region& region : op->getRegions()) {
            for (mlir::Block& block : region) {
               for (mlir::BlockArgument arg : block.getArguments()) {
                  mlir::Type demoted = demoteMixedHivCarrierType(ctx, arg.getType());
                  if (demoted != arg.getType()) {
                     arg.setType(demoted);
                     changed = true;
                  }
               }
            }
         }
      });

      synthetic.walk([&](subop::LookupOp op) {
         auto ref = op.getRef();
         if (demoteMixedHivColumnDef(ref)) {
            op.setRefAttr(ref);
            changed = true;
         }
      });
      synthetic.walk([&](subop::ScanListOp op) {
         auto elem = op.getElem();
         if (demoteMixedHivColumnDef(elem)) {
            op.setElemAttr(elem);
            changed = true;
         }
      });
      synthetic.walk([&](subop::GatherOp op) {
         auto ref = op.getRef();
         bool opChanged = demoteMixedHivColumnRef(ref);
         if (opChanged) op.setRefAttr(ref);
         llvm::SmallVector<subop::DefMappingPairT> pairs;
         pairs.reserve(op.getMapping().getMapping().size());
         for (auto [member, def] : op.getMapping().getMapping()) {
            tuples::ColumnDefAttr outDef = def;
            opChanged |= demoteMixedHivColumnDef(outDef);
            pairs.push_back({member, outDef});
         }
         if (opChanged) {
            op.setMappingAttr(subop::ColumnDefMemberMappingAttr::get(ctx, pairs));
            changed = true;
         }
      });
      synthetic.walk([&](subop::MapOp op) {
         bool opChanged = false;
         llvm::SmallVector<mlir::Attribute> inputRefs;
         inputRefs.reserve(op.getInputCols().size());
         for (mlir::Attribute attr : op.getInputCols()) {
            auto ref = mlir::cast<tuples::ColumnRefAttr>(attr);
            opChanged |= demoteMixedHivColumnRef(ref);
            inputRefs.push_back(ref);
         }
         llvm::SmallVector<mlir::Attribute> computedDefs;
         computedDefs.reserve(op.getComputedCols().size());
         for (mlir::Attribute attr : op.getComputedCols()) {
            auto def = mlir::cast<tuples::ColumnDefAttr>(attr);
            opChanged |= demoteMixedHivColumnDef(def);
            computedDefs.push_back(def);
         }
         if (opChanged) {
            op.setInputColsAttr(mlir::ArrayAttr::get(ctx, inputRefs));
            op.setComputedColsAttr(mlir::ArrayAttr::get(ctx, computedDefs));
            changed = true;
         }
      });
      synthetic.walk([&](subop::MaterializeOp op) {
         bool opChanged = false;
         llvm::SmallVector<subop::RefMappingPairT> pairs;
         pairs.reserve(op.getMapping().getMapping().size());
         for (auto [member, ref] : op.getMapping().getMapping()) {
            tuples::ColumnRefAttr outRef = ref;
            opChanged |= demoteMixedHivColumnRef(outRef);
            pairs.push_back({member, outRef});
         }
         if (opChanged) {
            op.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, pairs));
            changed = true;
         }
      });
      synthetic.walk([&](subop::ReduceOp op) {
         bool opChanged = false;
         auto ref = op.getRef();
         opChanged |= demoteMixedHivColumnRef(ref);
         if (opChanged) op.setRefAttr(ref);
         llvm::SmallVector<mlir::Attribute> cols;
         cols.reserve(op.getColumns().size());
         for (mlir::Attribute attr : op.getColumns()) {
            auto col = mlir::cast<tuples::ColumnRefAttr>(attr);
            opChanged |= demoteMixedHivColumnRef(col);
            cols.push_back(col);
         }
         if (opChanged) {
            op.setColumnsAttr(mlir::ArrayAttr::get(ctx, cols));
            changed = true;
         }
      });

      if (!changed) break;
   }
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

static bool hivLayoutsCompatibleByOrdinal(subop::HashIndexedViewType a,
                                          subop::HashIndexedViewType b) {
   if (!a || !b) return false;
   if (a.getKeyMembers().getMembers().size() != b.getKeyMembers().getMembers().size()) return false;
   if (a.getValueMembers().getMembers().size() != b.getValueMembers().getMembers().size()) return false;
   auto& mm = a.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   for (auto [ma, mb] : llvm::zip(a.getKeyMembers().getMembers(), b.getKeyMembers().getMembers())) {
      if (mm.getType(ma) != mm.getType(mb)) return false;
   }
   for (auto [ma, mb] : llvm::zip(a.getValueMembers().getMembers(), b.getValueMembers().getMembers())) {
      if (mm.getType(ma) != mm.getType(mb)) return false;
   }
   return true;
}

static mlir::Type retargetCompatibleHivCarrierType(mlir::MLIRContext* ctx, mlir::Type type,
                                                   subop::HashIndexedViewType targetHiv) {
   if (!type || !targetHiv) return type;
   if (auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(type)) {
      auto hiv = subop::HashIndexedViewType::get(ctx, mixed.getKeyMembers(), mixed.getValueMembers(),
                                                 mixed.getCompareHashForLookup());
      if (hivLayoutsCompatibleByOrdinal(hiv, targetHiv)) return targetHiv;
      return type;
   }
   if (auto hiv = mlir::dyn_cast<subop::HashIndexedViewType>(type)) {
      if (hivLayoutsCompatibleByOrdinal(hiv, targetHiv)) return targetHiv;
      return type;
   }
   if (auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(type)) {
      mlir::Type retargetedState = retargetCompatibleHivCarrierType(ctx, ler.getState(), targetHiv);
      if (retargetedState == ler.getState()) return type;
      return subop::LookupEntryRefType::get(ctx, mlir::cast<subop::LookupAbleState>(retargetedState));
   }
   if (auto list = mlir::dyn_cast<subop::ListType>(type)) {
      mlir::Type retargetedElem = retargetCompatibleHivCarrierType(ctx, list.getT(), targetHiv);
      if (retargetedElem == list.getT()) return type;
      return subop::ListType::get(ctx, mlir::cast<subop::StateEntryReference>(retargetedElem));
   }
   return type;
}

static bool retargetCompatibleHivColumnRef(tuples::ColumnRefAttr ref,
                                           subop::HashIndexedViewType targetHiv) {
   mlir::Type retargeted =
      retargetCompatibleHivCarrierType(ref.getContext(), ref.getColumn().type, targetHiv);
   if (retargeted == ref.getColumn().type) return false;
   ref.getColumn().type = retargeted;
   return true;
}

static bool retargetCompatibleHivColumnDef(tuples::ColumnDefAttr def,
                                           subop::HashIndexedViewType targetHiv) {
   mlir::Type retargeted =
      retargetCompatibleHivCarrierType(def.getContext(), def.getColumn().type, targetHiv);
   if (retargeted == def.getColumn().type) return false;
   def.getColumn().type = retargeted;
   return true;
}

static void retargetSyntheticCacheGetClosureToCachedHiv(mlir::ModuleOp synthetic,
                                                        mlir::Value cacheGetRoot,
                                                        subop::HashIndexedViewType cachedHiv) {
   llvm::DenseSet<void*> closure = collectSsaClosureFromCacheGetRoot(synthetic, cacheGetRoot);
   auto expandNestedMapPorts = [&]() {
      bool changed = false;
      synthetic.walk([&](subop::NestedMapOp nested) {
         bool touchesClosure = closure.contains(nested.getOperation()) ||
            opaqueClosureContains(closure, nested.getStream());
         for (mlir::Attribute attr : nested.getParameters()) {
            auto ref = mlir::dyn_cast<tuples::ColumnRefAttr>(attr);
            if (ref && closure.contains(&ref.getColumn())) touchesClosure = true;
         }
         if (!touchesClosure) return;
         closure.insert(nested.getOperation());
         closure.insert(nested.getRes().getAsOpaquePointer());
         if (nested.getRegion().empty()) return;
         for (mlir::BlockArgument arg : nested.getRegion().front().getArguments())
            changed |= closure.insert(arg.getAsOpaquePointer()).second;
      });
      return changed;
   };
   while (expandNestedMapPorts()) {
      expandClosureThroughExecutionStepPorts(synthetic, closure);
      expandClosureThroughNestedExecutionGroupPorts(synthetic, closure);
   }
   auto* ctx = synthetic.getContext();
   auto inClosure = [&](mlir::Operation* op) {
      return !op || closure.contains(op) || opOperandsOrNestedBlockArgsTouchClosure(op, closure);
   };

   for (;;) {
      bool changed = false;
      for (void* opaque : closure) {
         auto value = mlir::Value::getFromOpaquePointer(opaque);
         if (!value) continue;
         mlir::Type retargeted = retargetCompatibleHivCarrierType(ctx, value.getType(), cachedHiv);
         if (retargeted != value.getType()) {
            value.setType(retargeted);
            changed = true;
         }
      }
      synthetic.walk([&](mlir::Operation* op) {
         if (!inClosure(op)) return;
         for (mlir::Region& region : op->getRegions()) {
            for (mlir::Block& block : region) {
               for (mlir::BlockArgument arg : block.getArguments()) {
                  mlir::Type retargeted = retargetCompatibleHivCarrierType(ctx, arg.getType(), cachedHiv);
                  if (retargeted != arg.getType()) {
                     arg.setType(retargeted);
                     changed = true;
                  }
               }
            }
         }
      });
      synthetic.walk([&](subop::LookupOp op) {
         if (!inClosure(op.getOperation())) return;
         auto ref = op.getRef();
         if (retargetCompatibleHivColumnDef(ref, cachedHiv)) {
            op.setRefAttr(ref);
            changed = true;
         }
      });
      synthetic.walk([&](subop::ScanListOp op) {
         if (!inClosure(op.getOperation())) return;
         auto elem = op.getElem();
         if (retargetCompatibleHivColumnDef(elem, cachedHiv)) {
            op.setElemAttr(elem);
            changed = true;
         }
      });
      synthetic.walk([&](subop::GatherOp op) {
         if (!inClosure(op.getOperation())) return;
         auto ref = op.getRef();
         bool opChanged = retargetCompatibleHivColumnRef(ref, cachedHiv);
         llvm::SmallVector<subop::DefMappingPairT> pairs;
         pairs.reserve(op.getMapping().getMapping().size());
         for (auto [member, def] : op.getMapping().getMapping()) {
            tuples::ColumnDefAttr outDef = def;
            opChanged |= retargetCompatibleHivColumnDef(outDef, cachedHiv);
            pairs.push_back({member, outDef});
         }
         if (opChanged) {
            op.setRefAttr(ref);
            op.setMappingAttr(subop::ColumnDefMemberMappingAttr::get(ctx, pairs));
            changed = true;
         }
      });
      synthetic.walk([&](subop::MapOp op) {
         if (!inClosure(op.getOperation())) return;
         bool opChanged = false;
         llvm::SmallVector<mlir::Attribute> inputRefs;
         inputRefs.reserve(op.getInputCols().size());
         for (mlir::Attribute attr : op.getInputCols()) {
            auto ref = mlir::cast<tuples::ColumnRefAttr>(attr);
            opChanged |= retargetCompatibleHivColumnRef(ref, cachedHiv);
            inputRefs.push_back(ref);
         }
         llvm::SmallVector<mlir::Attribute> computedDefs;
         computedDefs.reserve(op.getComputedCols().size());
         for (mlir::Attribute attr : op.getComputedCols()) {
            auto def = mlir::cast<tuples::ColumnDefAttr>(attr);
            opChanged |= retargetCompatibleHivColumnDef(def, cachedHiv);
            computedDefs.push_back(def);
         }
         if (opChanged) {
            op.setInputColsAttr(mlir::ArrayAttr::get(ctx, inputRefs));
            op.setComputedColsAttr(mlir::ArrayAttr::get(ctx, computedDefs));
            changed = true;
         }
      });
      synthetic.walk([&](subop::NestedMapOp op) {
         if (!inClosure(op.getOperation())) return;
         bool opChanged = false;
         llvm::SmallVector<mlir::Attribute> params;
         params.reserve(op.getParameters().size());
         for (mlir::Attribute attr : op.getParameters()) {
            auto ref = mlir::cast<tuples::ColumnRefAttr>(attr);
            opChanged |= retargetCompatibleHivColumnRef(ref, cachedHiv);
            params.push_back(ref);
         }
         if (opChanged) {
            op.setParametersAttr(mlir::ArrayAttr::get(ctx, params));
            changed = true;
         }
      });
      synthetic.walk([&](subop::MaterializeOp op) {
         if (!inClosure(op.getOperation())) return;
         bool opChanged = false;
         llvm::SmallVector<subop::RefMappingPairT> pairs;
         pairs.reserve(op.getMapping().getMapping().size());
         for (auto [member, ref] : op.getMapping().getMapping()) {
            tuples::ColumnRefAttr outRef = ref;
            opChanged |= retargetCompatibleHivColumnRef(outRef, cachedHiv);
            pairs.push_back({member, outRef});
         }
         if (opChanged) {
            op.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, pairs));
            changed = true;
         }
      });
      if (!changed) break;
      expandClosureThroughExecutionStepPorts(synthetic, closure);
      expandClosureThroughNestedExecutionGroupPorts(synthetic, closure);
      while (expandNestedMapPorts()) {
         expandClosureThroughExecutionStepPorts(synthetic, closure);
         expandClosureThroughNestedExecutionGroupPorts(synthetic, closure);
      }
   }
   synchronizeExecutionStepPortTypes(synthetic, &closure);
}

static CachedJoinBufferLayout cachedJoinLayoutFromHiv(subop::HashIndexedViewType hiv) {
   CachedJoinBufferLayout layout;
   layout.producerHiv = hiv;
   auto& mm = hiv.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   for (subop::Member m : hiv.getValueMembers().getMembers()) {
      layout.payloadMembers.push_back(m);
      layout.payloadColumnTypes.push_back(mm.getType(m));
      llvm::StringRef memberName = mm.getName(m);
      if (auto slot = parseFilterPredMemberSlot(memberName)) {
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

static void mergeCacheGetPayloadMemberRemap(CachedJoinBufferLayout& layout,
                                            subop::HashIndexedViewType oldHiv,
                                            subop::HashIndexedViewType cachedHiv) {
   if (!oldHiv || !cachedHiv) return;
   if (!layout.producerHiv) layout = cachedJoinLayoutFromHiv(cachedHiv);
   llvm::ArrayRef<subop::Member> oldVals = oldHiv.getValueMembers().getMembers();
   llvm::ArrayRef<subop::Member> cachedVals = cachedHiv.getValueMembers().getMembers();
   assert(oldVals.size() == cachedVals.size() &&
          "synthetic cache_get payload remap requires aligned HIV payload layout");
   for (auto [oldMember, cachedMember] : llvm::zip(oldVals, cachedVals))
      if (oldMember != cachedMember) layout.probeGatherMemberRemap[oldMember] = cachedMember;
}

static void alignSyntheticCacheGetDependenciesToCachedPuts(
   mlir::ModuleOp synthetic,
   CachedJoinBufferLayoutsByKey* layoutsByKey) {
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

   for (uint64_t key : keysToAlign) {
      subop::HashIndexedViewType hiv = cachedHivByKey.lookup(key);
      CachedJoinBufferLayout layout = cachedJoinLayoutFromHiv(hiv);
      if (layoutsByKey) {
         CachedJoinBufferLayout& cachedLayout = (*layoutsByKey)[key];
         if (!cachedLayout.producerHiv) cachedLayout = layout;
         synthetic.walk([&](subop::CacheGetOp get) {
            if (static_cast<uint64_t>(get.getKey()) != key) return;
            mergeCacheGetPayloadMemberRemap(cachedLayout,
                                            asHashIndexedViewLayoutTypeLocal(get.getResult().getType()),
                                            hiv);
         });
      }
      alignConsumerModulesToCachedJoinLayout(synthetic, layout, key, std::nullopt, nullptr);
      resyncConsumerCachedHivCarrierTypesFromCacheGet(synthetic, key);
      synthetic.walk([&](subop::CacheGetOp get) {
         if (static_cast<uint64_t>(get.getKey()) != key) return;
         mlir::Value root = get.getResult();
         if (auto step = get->getParentOfType<subop::ExecutionStepOp>()) {
            if (step.getNumResults() == 1) root = step.getResult(0);
         }
         retargetSyntheticCacheGetClosureToCachedHiv(synthetic, root, hiv);
      });
   }
   demoteMixedHivCarriersInSyntheticProducer(synthetic);
   syncProbeGatherMappingsInModule(synthetic);
   synthetic.walk([&](subop::CacheGetOp get) {
      uint64_t key = static_cast<uint64_t>(get.getKey());
      auto it = cachedHivByKey.find(key);
      if (it == cachedHivByKey.end()) return;
      mlir::Value root = get.getResult();
      if (auto step = get->getParentOfType<subop::ExecutionStepOp>()) {
         if (step.getNumResults() == 1) root = step.getResult(0);
      }
      retargetSyntheticCacheGetClosureToCachedHiv(synthetic, root, it->second);
   });
}

BatchReusePlanRewriteResult rewritePlansWithSyntheticQueryBatch(
   llvm::ArrayRef<mlir::ModuleOp> queries,
   llvm::ArrayRef<CrossQueryStateMatchGroup> groups,
   lingodb::catalog::Catalog* catalog) {
   BatchReusePlanRewriteResult total;
   initBatchReuseCounters(total, queries.size());

   llvm::SmallVector<CrossQueryStateMatchGroup, 64> currentGroups(groups.begin(), groups.end());
   CachedJoinBufferLayoutsByKey accumulatedJoinLayouts;
   constexpr unsigned maxFixedPointIterations = 8;
   for (unsigned iter = 0; iter < maxFixedPointIterations; ++iter) {
      if (currentGroups.empty()) break;
      BatchReusePlanRewriteResult one =
         rewritePlansWithSyntheticQueryBatchOnce(queries, currentGroups, catalog);
      bool madeProgress = one.numTargetsSyntheticMapped > 0;
      if (!madeProgress) break;

      accumulateBatchReuseCounters(total, one);
      for (auto& layoutEntry : one.cachedJoinLayouts)
         accumulatedJoinLayouts[layoutEntry.first] = std::move(layoutEntry.second);

      if (one.synthetic) {
         if (!total.synthetic) {
            total.synthetic = std::move(one.synthetic);
         } else {
            appendSyntheticProducerSteps(*total.synthetic, *one.synthetic);
         }
         alignSyntheticCacheGetDependenciesToCachedPuts(*total.synthetic, &accumulatedJoinLayouts);
         alignDirectSyntheticHivGathersToCachedLayouts(*total.synthetic, accumulatedJoinLayouts);
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
