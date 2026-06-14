#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateExtraction.h"

#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/runtime/ExternalDataSourceProperty.h"
#include "lingodb/utility/Serialization.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <algorithm>
#include <cassert>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>
#include <sstream>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

namespace {

llvm::SmallVector<int, 16> sortedUniqueStepIndices(llvm::ArrayRef<int> stepIndices) {
   llvm::SmallVector<int, 16> out(stepIndices.begin(), stepIndices.end());
   llvm::sort(out);
   out.erase(std::unique(out.begin(), out.end()), out.end());
   return out;
}

} // namespace

namespace lingodb::compiler::dialect::subop {

namespace {

static std::string normalizeMappingMemberName(llvm::StringRef name) {
   size_t dollar = name.rfind('$');
   if (dollar == llvm::StringRef::npos || dollar + 1 >= name.size()) return name.str();
   for (char c : name.drop_front(dollar + 1)) {
      if (c < '0' || c > '9') return name.str();
   }
   return name.take_front(dollar).str();
}

static std::string renderExternalDataSourceDescrMatchString(
   const lingodb::runtime::ExternalDatasourceProperty& ds,
   llvm::ArrayRef<lingodb::runtime::ExternalDatasourceProperty::Mapping> mapping, bool includeFilters = true) {
   using lingodb::runtime::ExternalDatasourceProperty;
   using lingodb::runtime::FilterDescription;
   using lingodb::runtime::FilterOp;

   llvm::SmallVector<ExternalDatasourceProperty::Mapping, 8> mappingSorted(mapping.begin(), mapping.end());
   llvm::sort(mappingSorted, [&](const auto& a, const auto& b) {
      std::string am = normalizeMappingMemberName(a.memberName);
      std::string bm = normalizeMappingMemberName(b.memberName);
      if (am != bm) return am < bm;
      return a.identifier < b.identifier;
   });

   llvm::SmallVector<FilterDescription, 8> filters;
   if (includeFilters) {
      // Normalize filters: de-duplicate then sort by a stable key.
      std::unordered_set<FilterDescription> uniq;
      for (auto& f : ds.filterDescriptions) {
         if (!uniq.insert(f).second) continue;
         filters.push_back(f);
      }
   }
   auto filterOpToStr = [](FilterOp op) -> const char* {
      switch (op) {
         case FilterOp::EQ: return "EQ";
         case FilterOp::NEQ: return "NEQ";
         case FilterOp::LT: return "LT";
         case FilterOp::LTE: return "LTE";
         case FilterOp::GT: return "GT";
         case FilterOp::GTE: return "GTE";
         case FilterOp::NOTNULL: return "NOTNULL";
         case FilterOp::IN: return "IN";
      }
      return "UNKNOWN";
   };
   auto filterValueToStr = [](const FilterDescription& f) -> std::string {
      std::string out;
      std::ostringstream os;
      // value
      os << "v=";
      std::visit([&](auto const& v) { os << v; }, f.value);
      // values (for IN)
      os << ";vs=";
      std::visit([&](auto const& vs) {
         os << "[";
         for (size_t i = 0; i < vs.size(); i++) {
            if (i) os << ",";
            os << vs[i];
         }
         os << "]";
      }, f.values);
      out = os.str();
      return out;
   };
   if (includeFilters) {
      llvm::sort(filters, [&](const FilterDescription& a, const FilterDescription& b) {
         if (a.columnName != b.columnName) return a.columnName < b.columnName;
         if (a.columnId != b.columnId) return a.columnId < b.columnId;
         if (a.op != b.op) return static_cast<uint8_t>(a.op) < static_cast<uint8_t>(b.op);
         auto av = filterValueToStr(a);
         auto bv = filterValueToStr(b);
         return av < bv;
      });
   }

   // Render to a stable string that we still use for matching (for now).
   std::string s;
   llvm::raw_string_ostream ss(s);
   ss << "table=" << ds.tableName;
   ss << ";index=" << ds.index;
   ss << ";indexType=" << ds.indexType;
   ss << ";mapping=[";
   for (size_t i = 0; i < mappingSorted.size(); i++) {
      if (i) ss << ",";
      ss << normalizeMappingMemberName(mappingSorted[i].memberName) << "->" << mappingSorted[i].identifier;
   }
   ss << "]";
   auto renderFilterList = [&](llvm::ArrayRef<FilterDescription> list) {
      for (size_t i = 0; i < list.size(); i++) {
         if (i) ss << ",";
         ss << list[i].columnName << "#" << list[i].columnId << ":" << filterOpToStr(list[i].op) << "{"
            << filterValueToStr(list[i]) << "}";
      }
   };
   ss << ";filters=[";
   renderFilterList(filters);
   ss << "]";
   ss << ";or_filters=[";
   if (includeFilters) {
      for (size_t ci = 0; ci < ds.orFilterClauses.size(); ci++) {
         if (ci) ss << "|";
         ss << "(";
         renderFilterList(ds.orFilterClauses[ci]);
         ss << ")";
      }
   }
   ss << "]";
   ss.flush();
   return s;
}

static std::string normalizeExternalDataSourceDescrHex(llvm::StringRef hexDescr) {
   using lingodb::runtime::ExternalDatasourceProperty;
   ExternalDatasourceProperty ds = lingodb::utility::deserializeFromHexString<ExternalDatasourceProperty>(hexDescr);
   return renderExternalDataSourceDescrMatchString(ds, ds.mapping);
}

bool isCreateLike(mlir::Operation& op) {
   return mlir::isa<
      subop::CreateThreadLocalOp,
      subop::CreateHeapOp,
      subop::GenericCreateOp,
      subop::CreateFrom,
      subop::CreateSimpleStateOp,
      subop::CreateArrayOp>(op);
}

bool isStateType(mlir::Type t) {
   return mlir::isa<subop::State>(t);
}

bool isThreadLocalOfStateType(mlir::Type t) {
   auto tl = mlir::dyn_cast_or_null<subop::ThreadLocalType>(t);
   return tl && isStateType(tl.getWrapped());
}

static bool isFuncBlockArgument(mlir::Value v) {
   auto ba = mlir::dyn_cast<mlir::BlockArgument>(v);
   if (!ba) return false;
   auto* owner = ba.getOwner();
   if (!owner) return false;
   return mlir::isa_and_nonnull<mlir::func::FuncOp>(owner->getParentOp());
}

/// Cross-query reuse may extend value/buffer members with `filter_pred$0` while nested regions still
/// refer to the pre-extension type; treat those as compatible for mapping block args to operands.
static mlir::Type stripTrailingFilterPredMember(mlir::Type t) {
   auto* d = t.getContext()->getLoadedDialect<subop::SubOperatorDialect>();
   if (!d) return t;
   const auto& mm = d->getMemberManager();
   auto isPred = [&](subop::Member m) {
      return llvm::StringRef(mm.getName(m)).contains("filter_pred");
   };
   auto* ctx = t.getContext();
   if (auto buf = mlir::dyn_cast<subop::BufferType>(t)) {
      auto mems = buf.getMembers().getMembers();
      if (mems.empty() || !isPred(mems.back())) return t;
      llvm::SmallVector<subop::Member> pref(mems.begin(), mems.end() - 1);
      return subop::BufferType::get(ctx, subop::StateMembersAttr::get(ctx, std::move(pref)));
   }
   if (auto hm = mlir::dyn_cast<subop::HashMapType>(t)) {
      auto vals = hm.getValueMembers().getMembers();
      if (vals.empty() || !isPred(vals.back())) return t;
      llvm::SmallVector<subop::Member> pref(vals.begin(), vals.end() - 1);
      auto newVals = subop::StateMembersAttr::get(ctx, std::move(pref));
      return subop::HashMapType::get(ctx, hm.getKeyMembers(), newVals, hm.getWithLock());
   }
   if (auto fr = mlir::dyn_cast<subop::PreAggrHtFragmentType>(t)) {
      auto vals = fr.getValueMembers().getMembers();
      if (vals.empty() || !isPred(vals.back())) return t;
      llvm::SmallVector<subop::Member> pref(vals.begin(), vals.end() - 1);
      auto newVals = subop::StateMembersAttr::get(ctx, std::move(pref));
      return subop::PreAggrHtFragmentType::get(ctx, fr.getKeyMembers(), newVals, fr.getWithLock());
   }
   if (auto ht = mlir::dyn_cast<subop::PreAggrHtType>(t)) {
      auto vals = ht.getValueMembers().getMembers();
      if (vals.empty() || !isPred(vals.back())) return t;
      llvm::SmallVector<subop::Member> pref(vals.begin(), vals.end() - 1);
      auto newVals = subop::StateMembersAttr::get(ctx, std::move(pref));
      return subop::PreAggrHtType::get(ctx, ht.getKeyMembers(), newVals, ht.getWithLock());
   }
   return t;
}

static bool canMapBlockArgToOperandByType(mlir::Type operandTy, mlir::Type argTy) {
   if (operandTy == argTy) return true;
   if (stripTrailingFilterPredMember(operandTy) == stripTrailingFilterPredMember(argTy)) return true;
   if (auto tl = mlir::dyn_cast_or_null<subop::ThreadLocalType>(operandTy)) {
      mlir::Type w = tl.getWrapped();
      if (w == argTy || stripTrailingFilterPredMember(w) == stripTrailingFilterPredMember(argTy) ||
          w == stripTrailingFilterPredMember(argTy) || stripTrailingFilterPredMember(w) == argTy)
         return true;
   }
   if (auto tl = mlir::dyn_cast_or_null<subop::ThreadLocalType>(argTy)) {
      mlir::Type w = tl.getWrapped();
      if (w == operandTy || stripTrailingFilterPredMember(w) == stripTrailingFilterPredMember(operandTy) ||
          w == stripTrailingFilterPredMember(operandTy) || stripTrailingFilterPredMember(w) == operandTy)
         return true;
   }
   return false;
}

mlir::Value canonicalizeStateValue(subop::ExecutionStepOp step, mlir::Value v) {
   auto ba = mlir::dyn_cast<mlir::BlockArgument>(v);
   if (!ba) {
      // Map values returned by the step to the step results.
      auto& block = step.getSubOps().front();
      auto ret = mlir::dyn_cast<subop::ExecutionStepReturnOp>(block.getTerminator());
      assert(ret && "execution_step body must terminate with execution_step_return");
      auto inputs = ret.getInputs();
      for (size_t i = 0; i < inputs.size() && i < step.getNumResults(); i++) {
         if (inputs[i] == v) {
            return step.getResult(i);
         }
      }
      // Value is already in outer SSA form (e.g. step input / previous step result).
      return v;
   }
   auto* owner = ba.getOwner();
   assert(owner && "block argument must have an owner block");
   if (owner != &step.getSubOps().front()) {
      // Nested region argument (e.g. inside subop.map/scf/tuples regions).
      // Map region block arguments back to the parent op operands.
      auto* parent = owner->getParentOp();
      assert(parent && "nested region block must have a parent op");
      unsigned idx = ba.getArgNumber();
      if (idx < parent->getNumOperands() && canMapBlockArgToOperandByType(parent->getOperand(idx).getType(), ba.getType())) {
         return canonicalizeStateValue(step, parent->getOperand(idx));
      }
      // Fallback: if there is exactly one operand with the same type, assume that's the mapping.
      mlir::Value candidate;
      for (auto opnd : parent->getOperands()) {
         if (!canMapBlockArgToOperandByType(opnd.getType(), ba.getType())) continue;
         if (candidate) {
            candidate = mlir::Value();
            break;
         }
         candidate = opnd;
      }
      if (candidate) {
         return canonicalizeStateValue(step, candidate);
      }
      // No parent operand matches (e.g. state threaded only through `subop.map` region args).
      // Keep the value so `getMembersForStateValue` can still classify the type; reuse keys may be less canonical.
      return v;
   }
   auto inputs = step.getInputs();
   auto idx = ba.getArgNumber();
   assert(idx < inputs.size() && "block argument index must be within execution_step inputs");
   return inputs[idx];
}

static mlir::Value resolveStateOperandThroughNestedRegions(mlir::Value v);

static subop::ExecutionStepOp findEnclosingExecutionStep(mlir::Value v) {
   if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(v)) {
      auto* b = ba.getOwner();
      for (auto* p = b ? b->getParentOp() : nullptr; p; p = p->getParentOp()) {
         if (auto s = mlir::dyn_cast<subop::ExecutionStepOp>(p)) return s;
      }
      return {};
   }
   if (auto* def = v.getDefiningOp()) {
      for (auto* p = def; p; p = p->getParentOp()) {
         if (auto s = mlir::dyn_cast<subop::ExecutionStepOp>(p)) return s;
      }
   }
   return {};
}

static subop::ExecutionGroupOp getMainExecutionGroup(mlir::ModuleOp moduleOp) {
   subop::ExecutionGroupOp group;
   moduleOp.walk([&](subop::ExecutionGroupOp g) {
      assert(!group && "expected exactly one execution_group per module");
      group = g;
   });
   assert(group && "module must contain an execution_group");
   return group;
}

static bool isTopLevelExecutionStep(subop::ExecutionStepOp step) {
   return step && mlir::isa<subop::ExecutionGroupOp>(step->getParentOp());
}

static void walkTopLevelExecutionSteps(mlir::ModuleOp moduleOp,
                                       llvm::function_ref<void(subop::ExecutionStepOp)> fn) {
   subop::ExecutionGroupOp group = getMainExecutionGroup(moduleOp);
   for (mlir::Operation& op : group.getSubOps().front()) {
      if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(&op)) fn(step);
   }
}

static subop::ExecutionStepOp findTopLevelExecutionStep(subop::ExecutionStepOp step) {
   if (!step) return step;
   if (isTopLevelExecutionStep(step)) return step;
   if (auto outer = step->getParentOfType<subop::ExecutionStepOp>()) return findTopLevelExecutionStep(outer);
   return step;
}

/// Map nested / in-step SSA to the owning top-level `execution_step` result when possible.
static mlir::Value liftToTopLevelGroupState(mlir::Value v) {
   if (!v) return v;
   if (isFuncBlockArgument(v)) return v;

   subop::ExecutionStepOp step = findEnclosingExecutionStep(v);
   if (!step) return v;

   subop::ExecutionStepOp top = findTopLevelExecutionStep(step);
   mlir::Value mapped = resolveStateOperandThroughNestedRegions(v);
   mapped = canonicalizeStateValue(top, mapped);

   for (mlir::Value r : top.getResults()) {
      if (!isStateType(r.getType()) && !isThreadLocalOfStateType(r.getType())) continue;
      auto& block = top.getSubOps().front();
      auto ret = mlir::dyn_cast<subop::ExecutionStepReturnOp>(block.getTerminator());
      if (!ret) continue;
      for (mlir::Value o : ret.getOperands()) {
         if (!o) continue;
         mlir::Value canonRet = canonicalizeStateValue(top, o);
         if (canonRet == r || canonRet == mapped || o == mapped) return r;
      }
   }
   return mapped;
}

static bool executionStepReturnsStateValue(subop::ExecutionStepOp step) {
   auto& block = step.getSubOps().front();
   auto ret = mlir::dyn_cast<subop::ExecutionStepReturnOp>(block.getTerminator());
   if (!ret) return false;
   for (mlir::Value o : ret.getOperands()) {
      if (!o) continue;
      if (isStateType(o.getType()) || isThreadLocalOfStateType(o.getType())) return true;
   }
   return false;
}

// Canonicalize across nested regions/steps until we reach a top-level execution_group state port.
static mlir::Value canonicalizeStateValueDeep(mlir::Value v) {
   llvm::DenseSet<void*> seen;
   while (true) {
      assert(v && "canonicalizeStateValueDeep called with null value");
      void* key = v.getAsOpaquePointer();
      auto seenInsert = seen.insert(key);
      assert(seenInsert.second && "canonicalizeStateValueDeep hit a cycle");
      auto step = findEnclosingExecutionStep(v);
      if (!step) {
         // Outside any execution_step: must be a func argument (external), otherwise missing mapping.
         assert(isFuncBlockArgument(v) && "state value outside any execution_step must be a func argument");
         return liftToTopLevelGroupState(v);
      }
      auto c = canonicalizeStateValue(step, v);
      if (c == v) return liftToTopLevelGroupState(v);
      v = c;
   }
}

llvm::SmallVector<subop::Member> getMembersForStateValue(mlir::Value v) {
   auto t = v.getType();
   if (auto s = mlir::dyn_cast_or_null<subop::State>(t)) {
      return s.getMembers().getMembers();
   }
   if (auto tl = mlir::dyn_cast_or_null<subop::ThreadLocalType>(t)) {
      if (auto s = mlir::dyn_cast_or_null<subop::State>(tl.getWrapped())) {
         return s.getMembers().getMembers();
      }
   }
   return {};
}

bool intersectsMembers(const llvm::SmallVector<subop::Member>& a, const llvm::SmallVector<subop::Member>& b) {
   llvm::DenseSet<subop::Member> s;
   for (auto m : a) s.insert(m);
   for (auto m : b) {
      if (s.contains(m)) return true;
   }
   return false;
}

bool isCreateOnlyExecutionStep(subop::ExecutionStepOp step) {
   auto& block = step.getSubOps().front();
   for (auto& op : block.without_terminator()) {
      if (!isCreateLike(op)) {
         return false;
      }
   }
   return true;
}

/// Type-side predicate for "transparent" dependency carriers.
/// Actual transparency is computed from RW usage and stored on a per-value basis.
static bool isTransparentDepCarrierType(mlir::Type t) {
   if (mlir::isa<subop::BufferType>(t)) return true;
   if (mlir::isa<subop::SortedViewType>(t)) return true;
   if (isThreadLocalOfStateType(t)) return true;
   return false;
}

/// State produced by a lone `get_external` in an execution_step (table or external hash index, etc.).
static bool isGetExternalLeafState(mlir::Value v);
static std::optional<uint64_t> getCacheGetKeyForState(mlir::Value v);

struct RWFlags {
   bool read = false;
   bool write = false;
};

static bool isTransparentStateValue(mlir::Value v, const llvm::DenseSet<mlir::Value>& transparentStates) {
   if (!v) return false;
   if (!isTransparentDepCarrierType(v.getType())) return false;
   return transparentStates.contains(canonicalizeStateValueDeep(v));
}

static bool isSpecialNonReuseStateValue(mlir::Value v, const llvm::DenseSet<mlir::Value>& transparentStates) {
   if (!v) return false;
   if (mlir::isa<subop::TableType>(v.getType())) return true;
   return isTransparentStateValue(v, transparentStates);
}

/// Compute "transparent" states (a) type is buffer/thread_local/sorted_view AND
/// (b) it has exactly one successor step: used as pure read in exactly one step, and that step has
///     exactly one write state.
static llvm::DenseSet<mlir::Value> computeTransparentStatesFromRw(
   const llvm::DenseMap<int, llvm::DenseMap<mlir::Value, RWFlags>>& rwByStep) {
   llvm::DenseMap<mlir::Value, llvm::SmallVector<int, 4>> pureReadStepsByState;

   llvm::DenseMap<int, unsigned> writeCountByStep;
   for (auto& it : rwByStep) {
      unsigned wc = 0;
      for (auto& kv : it.second) if (kv.second.write) wc++;
      writeCountByStep[it.first] = wc;
   }

   for (auto& it : rwByStep) {
      int stepIdx = it.first;
      for (auto& kv : it.second) {
         mlir::Value s = canonicalizeStateValueDeep(kv.first);
         const RWFlags& f = kv.second;
         if (!isTransparentDepCarrierType(s.getType())) continue;
         if (f.read && !f.write) {
            pureReadStepsByState[s].push_back(stepIdx);
         }
      }
   }

   llvm::DenseSet<mlir::Value> out;
   for (auto& it : pureReadStepsByState) {
      mlir::Value s = it.first;
      auto& steps = it.second;
      llvm::sort(steps);
      steps.erase(std::unique(steps.begin(), steps.end()), steps.end());
      if (steps.size() != 1) continue;
      int onlyStep = steps[0];
      if (writeCountByStep.lookup(onlyStep) != 1) continue;
      out.insert(s);
   }
   return out;
}

/// Same-step RW edges: predecessor `r` must be available before writer `w` is constructed in that step.
struct StateDependencyGraph {
   llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value, 8>> predecessors;
};

static StateDependencyGraph buildStateDependencyGraphFromStepRw(
   const llvm::DenseMap<int, llvm::DenseMap<mlir::Value, RWFlags>>& rwByStep) {
   StateDependencyGraph dag;
   for (auto& stepIt : rwByStep) {
      llvm::SmallVector<mlir::Value, 16> reads;
      llvm::SmallVector<mlir::Value, 16> writes;
      reads.reserve(stepIt.second.size());
      writes.reserve(stepIt.second.size());
      for (auto& kv : stepIt.second) {
         dag.predecessors.try_emplace(kv.first);
         if (kv.second.read) reads.push_back(kv.first);
         if (kv.second.write) writes.push_back(kv.first);
      }
      for (mlir::Value w : writes) {
         for (mlir::Value r : reads) {
            if (r == w) continue;
            dag.predecessors[w].push_back(r);
         }
      }
   }
   for (auto& kv : dag.predecessors) {
      llvm::sort(kv.second, [](mlir::Value a, mlir::Value b) {
         return a.getAsOpaquePointer() < b.getAsOpaquePointer();
      });
      kv.second.erase(std::unique(kv.second.begin(), kv.second.end()), kv.second.end());
   }
   return dag;
}

static llvm::ArrayRef<mlir::Value> directPredecessorsInDepGraph(mlir::Value state,
                                                                const StateDependencyGraph& dag) {
   mlir::Value canon = canonicalizeStateValueDeep(state);
   auto it = dag.predecessors.find(canon);
   if (it == dag.predecessors.end()) return {};
   return it->second;
}

struct StateDepEligibility {
   bool eligible = false;
   llvm::SmallVector<std::string, 8> depTokensSorted;
};

static StateDepEligibility resolveDepEligibilityRec(
   mlir::Value state, const StateDependencyGraph& dag,
   const llvm::DenseSet<mlir::Value>& transparentStates,
   const llvm::DenseMap<mlir::Value, std::string>& tableDescrByTableState, llvm::DenseSet<mlir::Value>& visiting) {
   mlir::Value stateCanon = canonicalizeStateValueDeep(state);
   auto visitingInsert = visiting.insert(stateCanon);
   assert(visitingInsert.second && "state dependency graph must be acyclic");

   llvm::ArrayRef<mlir::Value> preds = directPredecessorsInDepGraph(stateCanon, dag);

   if (preds.empty()) {
      visiting.erase(stateCanon);
      if (isTransparentStateValue(stateCanon, transparentStates)) {
         return {false, {}};
      }
      return {true, {}};
   }

   unsigned nOther = 0;
   llvm::SmallVector<mlir::Value, 4> transparentPreds;
   llvm::SmallVector<mlir::Value, 4> leafPreds;
   for (mlir::Value p : preds) {
      if (isTransparentStateValue(p, transparentStates)) {
         transparentPreds.push_back(p);
         continue;
      }
      if (isGetExternalLeafState(p)) {
         leafPreds.push_back(p);
         continue;
      }
      if (getCacheGetKeyForState(p)) {
         leafPreds.push_back(p);
         continue;
      }
      nOther++;
   }

   if (nOther > 0) {
      visiting.erase(stateCanon);
      return {false, {}};
   }

   StateDepEligibility out;
   out.eligible = true;
   out.depTokensSorted.reserve(leafPreds.size() + 4);

   for (mlir::Value p : leafPreds) {
      auto itD = tableDescrByTableState.find(p);
      if (mlir::isa<subop::TableType>(p.getType())) {
         assert(itD != tableDescrByTableState.end() && "get_external table leaf predecessor must have descr");
         out.depTokensSorted.push_back(std::string("table:") + itD->second);
      } else if (mlir::isa<subop::ExternalHashIndexType>(p.getType())) {
         assert(itD != tableDescrByTableState.end() && "get_external hash-index leaf predecessor must have descr");
         out.depTokensSorted.push_back(std::string("externalhashindex:") + itD->second);
      } else if (std::optional<uint64_t> cacheKey = getCacheGetKeyForState(p)) {
         out.depTokensSorted.push_back(std::string("cache:") + std::to_string(*cacheKey));
      } else {
         llvm_unreachable("leaf predecessor must be get_external table/externalhashindex or cache_get");
      }
   }

   for (mlir::Value transparentPred : transparentPreds) {
      StateDepEligibility inner =
         resolveDepEligibilityRec(transparentPred, dag, transparentStates, tableDescrByTableState, visiting);
      if (!inner.eligible) {
         visiting.erase(stateCanon);
         return {false, {}};
      }
      out.depTokensSorted.append(inner.depTokensSorted.begin(), inner.depTokensSorted.end());
   }

   llvm::sort(out.depTokensSorted);
   out.depTokensSorted.erase(std::unique(out.depTokensSorted.begin(), out.depTokensSorted.end()),
                             out.depTokensSorted.end());
   visiting.erase(stateCanon);
   return out;
}

static StateDepEligibility evaluateStateDepEligibility(
   mlir::Value reuseTarget, const StateDependencyGraph& dag,
   const llvm::DenseSet<mlir::Value>& transparentStates,
   const llvm::DenseMap<mlir::Value, std::string>& tableDescrByTableState) {
   mlir::Value canon = canonicalizeStateValueDeep(reuseTarget);
   assert(!isSpecialNonReuseStateValue(canon, transparentStates) && "special states are not reuse targets");
   llvm::DenseSet<mlir::Value> visiting;
   return resolveDepEligibilityRec(canon, dag, transparentStates, tableDescrByTableState, visiting);
}

/// Trace SSA backward through nested regions / tuple pipelines to the owning `!subop.state` value.
static mlir::Value findUpstreamLookupHashIndexedView(mlir::Value v);

llvm::DenseMap<mlir::Value, RWFlags> analyzeStepStateRW(subop::ExecutionStepOp step) {
   llvm::DenseMap<mlir::Value, RWFlags> res;
   auto& block = step.getSubOps().front();

   if (isExternalTableRefStep(step)) {
      assert(step.getNumResults() == 1 && "external table ref step should have exactly one result");
      res[canonicalizeStateValueDeep(step.getResult(0))].write = true;
      return res;
   }

   for (auto& op : block.without_terminator()) {
      auto sub = mlir::dyn_cast<subop::SubOperator>(op);
      if (!sub) {
         continue;
      }

      // `materialize`: always mark state written; skip generic member-intersection (sink vs stream).
      if (auto mat = mlir::dyn_cast<subop::MaterializeOp>(&op)) {
         mlir::Value stv = mat.getState();
         assert(isStateType(stv.getType()) || isThreadLocalOfStateType(stv.getType()));
         res[canonicalizeStateValueDeep(stv)].write = true;
         continue;
      }
         if (auto hiv = mlir::dyn_cast<subop::CreateHashIndexedView>(&op)) {
            mlir::Value src = hiv.getSource();
            assert(isStateType(src.getType()) || isThreadLocalOfStateType(src.getType()));
            res[canonicalizeStateValueDeep(src)].read = true;
            mlir::Value out = hiv.getResult();
            assert(isStateType(out.getType()) || isThreadLocalOfStateType(out.getType()));
            res[canonicalizeStateValueDeep(out)].write = true;
            continue;
         }
         if (auto lock = mlir::dyn_cast<subop::LockOp>(&op)) {
            mlir::Value st = findUpstreamLookupHashIndexedView(lock.getStream());
            auto key = canonicalizeStateValueDeep(st);
            res[key].read = true;
            res[key].write = true;
            continue;
         }
      if (auto sort = mlir::dyn_cast<subop::CreateSortedViewOp>(&op)) {
         mlir::Value src = sort.getToSort();
         assert(isStateType(src.getType()) || isThreadLocalOfStateType(src.getType()));
         res[canonicalizeStateValueDeep(src)].read = true;
         mlir::Value out = sort.getResult();
         assert(isStateType(out.getType()) || isThreadLocalOfStateType(out.getType()));
         res[canonicalizeStateValueDeep(out)].write = true;
         continue;
      }
      if (auto lookup = mlir::dyn_cast<subop::LookupOp>(&op)) {
         mlir::Value st = lookup.getState();
         assert(isStateType(st.getType()) || isThreadLocalOfStateType(st.getType()));
         res[canonicalizeStateValueDeep(st)].read = true;
         continue;
      }
      if (auto scanList = mlir::dyn_cast<subop::ScanListOp>(&op)) {
         mlir::Value st = findUpstreamLookupHashIndexedView(scanList.getList());
         res[canonicalizeStateValueDeep(st)].read = true;
         continue;
      }
      if (auto scatter = mlir::dyn_cast<subop::ScatterOp>(&op)) {
         mlir::Value st = findUpstreamLookupHashIndexedView(scatter.getStream());
         res[canonicalizeStateValueDeep(st)].write = true;
         continue;
      }
      if (auto reduce = mlir::dyn_cast<subop::ReduceOp>(&op)) {
         mlir::Value st = findUpstreamLookupHashIndexedView(reduce.getStream());
         auto key = canonicalizeStateValueDeep(st);
         res[key].read = true;
         res[key].write = true;
         continue;
      }
      if (auto from = mlir::dyn_cast<subop::CreateFrom>(&op)) {
         mlir::Value src = from.getState();
         assert(isStateType(src.getType()) || isThreadLocalOfStateType(src.getType()));
         res[canonicalizeStateValueDeep(src)].read = true;
         mlir::Value dst = from.getResult();
         assert(isStateType(dst.getType()) || isThreadLocalOfStateType(dst.getType()));
         res[canonicalizeStateValueDeep(dst)].write = true;
         continue;
      }
      if (auto merge = mlir::dyn_cast<subop::MergeOp>(&op)) {
         res[canonicalizeStateValueDeep(merge.getThreadLocal())].read = true;
         res[canonicalizeStateValueDeep(merge.getResult())].write = true;
         continue;
      }

      auto readMembers = sub.getReadMembers();
      auto writtenMembers = sub.getWrittenMembers();

      llvm::SmallVector<mlir::Value, 4> stateOperands;
      for (auto v : op.getOperands()) {
         auto t = v.getType();
         if (isStateType(t) || isThreadLocalOfStateType(t)) {
            stateOperands.push_back(v);
         }
      }

      for (auto s : stateOperands) {
         auto key = canonicalizeStateValueDeep(s);
         auto operandMembers = getMembersForStateValue(key);
         auto& f = res[key];
         if (intersectsMembers(readMembers, operandMembers) || (readMembers.empty() && writtenMembers.empty())) f.read = true;
         if (intersectsMembers(writtenMembers, operandMembers)) f.write = true;
      }
   }
   return res;
}

static llvm::DenseMap<mlir::Value, RWFlags> analyzeStepStateRWWithNested(subop::ExecutionStepOp topStep);

void printStepStateRW(subop::ExecutionStepOp step, llvm::raw_ostream& os) {
   auto m = isTopLevelExecutionStep(step) ? analyzeStepStateRWWithNested(step) : analyzeStepStateRW(step);
   if (m.empty()) {
      return;
   }
   mlir::OpPrintingFlags flags;
   os << "// states:\n";
   for (auto it : m) {
      os << "//   ";
      if (it.second.read) os << "R";
      if (it.second.write) os << "W";
      os << " ";
      it.first.printAsOperand(os, flags);
      os << " : ";
      it.first.getType().print(os);
      os << "\n";
   }
}

mlir::Value getReturnedInternalValueForStepResult(subop::ExecutionStepOp step, unsigned resultIdx) {
   auto& block = step.getSubOps().front();
   auto ret = mlir::dyn_cast<subop::ExecutionStepReturnOp>(block.getTerminator());
   assert(ret && "execution_step body must terminate with execution_step_return");
   auto inputs = ret.getInputs();
   assert(resultIdx < inputs.size() && "resultIdx must be within execution_step_return inputs");
   assert(resultIdx < step.getNumResults() && "resultIdx must be within execution_step results");
   return inputs[resultIdx];
}

static void addConstructionStepsForSingleState(mlir::Value state,
                                               const llvm::DenseMap<mlir::Value, int>& createdAtByState,
                                               const llvm::DenseMap<mlir::Value, llvm::SmallSet<int, 16>>& writesByState,
                                               llvm::SmallVectorImpl<int>& outSteps) {
   mlir::Value stateCanon = canonicalizeStateValueDeep(state);
   auto itC = createdAtByState.find(stateCanon);
   assert(itC != createdAtByState.end() && "state must have a recorded createdAt");
   int createdAt = itC->second;
   assert(createdAt >= 0 && "createdAt must be non-negative when present");
   outSteps.push_back(createdAt);

   auto itW = writesByState.find(stateCanon);
   assert(itW != writesByState.end() && "state must have a recorded writes set");
   for (int w : itW->second) {
      assert(w >= createdAt && "write steps must not occur before creation");
      outSteps.push_back(w);
   }
}

static void forEachShadowChainPredecessorValue(
   mlir::Value state, const llvm::DenseMap<mlir::Value, mlir::Value>& mergedFromShadowState,
   llvm::function_ref<void(mlir::Value)> fn) {
   mlir::Value cur = canonicalizeStateValueDeep(state);
   llvm::DenseSet<void*> visited;
   for (;;) {
      auto itShadow = mergedFromShadowState.find(cur);
      if (itShadow == mergedFromShadowState.end()) break;
      mlir::Value shadow = itShadow->second;
      void* key = shadow.getAsOpaquePointer();
      if (!visited.insert(key).second) break;
      fn(shadow);
      cur = canonicalizeStateValueDeep(shadow);
   }
}

llvm::SmallVector<int, 8> getConstructionStepIndicesForState(
   mlir::Value state, const llvm::DenseMap<mlir::Value, int>& createdAtByState,
   const llvm::DenseMap<mlir::Value, llvm::SmallSet<int, 16>>& writesByState,
   const llvm::DenseMap<mlir::Value, mlir::Value>& mergedFromShadowState) {
   llvm::SmallVector<int, 8> steps;
   mlir::Value stateCanon = canonicalizeStateValueDeep(state);

   addConstructionStepsForSingleState(stateCanon, createdAtByState, writesByState, steps);
   forEachShadowChainPredecessorValue(stateCanon, mergedFromShadowState, [&](mlir::Value shadow) {
      addConstructionStepsForSingleState(canonicalizeStateValueDeep(shadow), createdAtByState, writesByState, steps);
   });

   llvm::sort(steps);
   steps.erase(std::unique(steps.begin(), steps.end()), steps.end());
   return steps;
}

/// Direct state deps from step RW: within one step, every read `r` and write `w` implies `w` depends on `r`.
static void appendSameStepRwDirectPrereqs(
   mlir::Value constructedState, llvm::ArrayRef<int> stepIndices,
   const llvm::DenseMap<int, llvm::DenseMap<mlir::Value, RWFlags>>& rwByStep, llvm::DenseSet<mlir::Value>& seen,
   llvm::SmallVectorImpl<mlir::Value>& out) {
   mlir::Value selfCanon = canonicalizeStateValueDeep(constructedState);
   for (int si : stepIndices) {
      auto itRw = rwByStep.find(si);
      assert(itRw != rwByStep.end());
      bool writesSelf = false;
      for (auto& kv : itRw->second) {
         if (kv.first == selfCanon && kv.second.write) writesSelf = true;
      }
      if (!writesSelf) continue;
      for (auto& kv : itRw->second) {
         if (!kv.second.read) continue;
         if (kv.first == selfCanon) continue;
         if (seen.insert(kv.first).second) out.push_back(kv.first);
      }
   }
}

llvm::SmallVector<mlir::Value, 8> getPrereqStatesForConstructionSteps(
   llvm::ArrayRef<int> stepIndices, mlir::Value constructedState,
   const llvm::DenseMap<int, llvm::DenseMap<mlir::Value, RWFlags>>& rwByStep) {
   llvm::SmallVector<mlir::Value, 8> prereqs;
   llvm::DenseSet<mlir::Value> seen;
   appendSameStepRwDirectPrereqs(constructedState, stepIndices, rwByStep, seen, prereqs);
   return prereqs;
}

static void mergeRwInto(llvm::DenseMap<mlir::Value, RWFlags>& dst,
                        const llvm::DenseMap<mlir::Value, RWFlags>& src) {
   for (auto& kv : src) {
      auto& f = dst[kv.first];
      f.read = f.read || kv.second.read;
      f.write = f.write || kv.second.write;
   }
}

static void augmentStepRwWithNestedScanListHivReads(subop::ExecutionStepOp step,
                                                    llvm::DenseMap<mlir::Value, RWFlags>& rw) {
   step.walk([&](subop::ScanListOp scan) {
      mlir::Value st = findUpstreamLookupHashIndexedView(scan.getList());
      rw[canonicalizeStateValueDeep(st)].read = true;
   });
}

/// RW for a top-level step: direct body plus all nested `execution_step` bodies (merged, no separate indices).
static llvm::DenseMap<mlir::Value, RWFlags> analyzeStepStateRWWithNested(subop::ExecutionStepOp topStep) {
   llvm::DenseMap<mlir::Value, RWFlags> res = analyzeStepStateRW(topStep);
   topStep.walk([&](subop::ExecutionStepOp nested) {
      if (nested == topStep) return;
      mergeRwInto(res, analyzeStepStateRW(nested));
   });
   augmentStepRwWithNestedScanListHivReads(topStep, res);
   return res;
}

/// Map an entry-region block argument produced by `execution_step` / `nested_execution_group` /
/// `execution_group` region boundaries to the corresponding parent operand (possibly another region
/// argument), so nested bodies can be traced back to outer SSA state values.
static mlir::Value mapRegionArgToParentOperand(mlir::BlockArgument ba) {
   mlir::Block* block = ba.getOwner();
   mlir::Region* region = block->getParent();
   if (!region) return ba;
   mlir::Operation* parent = region->getParentOp();
   unsigned idx = ba.getArgNumber();
   if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(parent)) {
      if (idx < step.getNumOperands()) return step.getOperand(idx);
      return ba;
   }
   if (auto nm = mlir::dyn_cast<subop::NestedMapOp>(parent)) {
      // Region arg 0 is the outer tuple stream; nested parameters (lists, keys, refs) are
      // bound from the same lookup/nested_map pipeline — trace via the stream operand.
      return nm.getStream();
   }
   if (auto lock = mlir::dyn_cast<subop::LockOp>(parent)) {
      return lock.getStream();
   }
   if (auto neg = mlir::dyn_cast<subop::NestedExecutionGroupOp>(parent)) {
      if (idx < neg.getNumOperands()) return neg.getOperand(idx);
      return ba;
   }
   if (auto eg = mlir::dyn_cast<subop::ExecutionGroupOp>(parent)) {
      if (idx < eg.getNumOperands()) return eg.getOperand(idx);
      return ba;
   }
   return ba;
}

static mlir::Value resolveStateOperandThroughNestedRegions(mlir::Value v) {
   mlir::Value cur = v;
   llvm::DenseSet<mlir::Value> seen;
   while (auto ba = mlir::dyn_cast<mlir::BlockArgument>(cur)) {
      if (!seen.insert(cur).second) break;
      mlir::Value next = mapRegionArgToParentOperand(ba);
      if (next == cur) break;
      cur = next;
   }
   return cur;
}

/// Any `!subop.state` / `thread_local<state>` used as an operand anywhere under a construction
/// `execution_step` (including nested_map / nested_execution_group bodies). This augments
/// `getPrereqStatesForConstructionSteps`, which only sees `analyzeStepStateRW` at the step's
/// top-level block — nested join steps often read `hash_indexed_view` only inside nested regions.
static void appendNestedStateOperandsAsPrereqs(subop::ExecutionStepOp step, mlir::Value constructedState,
                                               llvm::DenseSet<mlir::Value>& seen,
                                               llvm::SmallVectorImpl<mlir::Value>& out) {
   step.walk([&](mlir::Operation* op) {
      for (mlir::Value v : op->getOperands()) {
         mlir::Value outer = resolveStateOperandThroughNestedRegions(v);
         if (!isStateType(outer.getType()) && !isThreadLocalOfStateType(outer.getType())) continue;
         mlir::Value c = canonicalizeStateValueDeep(outer);
         if (c == constructedState) continue;
         if (seen.insert(c).second) out.push_back(c);
      }
   });
}

/// Trace SSA backward through nested regions / tuple pipelines to the `!subop.state` (or
/// `thread_local<state>`) that owns lookup entry refs / lists. Asserts if no state is found.
static mlir::Value findUpstreamLookupHashIndexedView(mlir::Value v) {
   assert(v && "findUpstreamLookupHashIndexedView requires a non-null seed");
   llvm::DenseSet<void*> visited;
   llvm::SmallVector<mlir::Value, 8> stack;
   stack.push_back(v);
   while (!stack.empty()) {
      mlir::Value cur = stack.pop_back_val();
      void* key = cur.getAsOpaquePointer();
      if (!visited.insert(key).second) continue;
      if (isStateType(cur.getType()) || isThreadLocalOfStateType(cur.getType())) {
         return canonicalizeStateValueDeep(cur);
      }
      if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(cur)) {
         stack.push_back(resolveStateOperandThroughNestedRegions(ba));
         continue;
      }
      if (auto* def = cur.getDefiningOp()) {
         if (auto lookup = mlir::dyn_cast<subop::LookupOp>(def)) {
            return canonicalizeStateValueDeep(lookup.getState());
         }
         if (auto lookupOrInsert = mlir::dyn_cast<subop::LookupOrInsertOp>(def)) {
            return canonicalizeStateValueDeep(lookupOrInsert.getState());
         }
         if (auto insert = mlir::dyn_cast<subop::InsertOp>(def)) {
            return canonicalizeStateValueDeep(insert.getState());
         }
         if (auto scanRefs = mlir::dyn_cast<subop::ScanRefsOp>(def)) {
            return canonicalizeStateValueDeep(scanRefs.getState());
         }
         if (auto scanList = mlir::dyn_cast<subop::ScanListOp>(def)) {
            stack.push_back(scanList.getList());
            continue;
         }
         if (auto nm = mlir::dyn_cast<subop::NestedMapOp>(def)) {
            stack.push_back(nm.getStream());
            continue;
         }
         if (mlir::isa<subop::UnwrapOptionalRefOp>(def)) {
            for (mlir::Value op : def->getOperands()) stack.push_back(op);
            continue;
         }
         for (mlir::Value op : def->getOperands()) stack.push_back(op);
      }
   }
   assert(false && "could not trace SSA back to a subop state (lookup/list/entry-ref pipeline)");
   return v;
}

static bool isGetExternalLeafState(mlir::Value v) {
   return mlir::isa<subop::TableType, subop::ExternalHashIndexType>(v.getType());
}

static std::optional<uint64_t> getCacheGetKeyForState(mlir::Value v) {
   if (!v) return std::nullopt;
   v = canonicalizeStateValueDeep(v);
   if (auto* def = v.getDefiningOp()) {
      if (auto get = mlir::dyn_cast<subop::CacheGetOp>(def)) {
         return static_cast<uint64_t>(get.getKey());
      }
      if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(def)) {
         auto& block = step.getSubOps().front();
         auto ret = mlir::dyn_cast<subop::ExecutionStepReturnOp>(block.getTerminator());
         if (!ret) return std::nullopt;
         for (unsigned i = 0; i < ret.getNumOperands() && i < step.getNumResults(); ++i) {
            if (step.getResult(i) != v) continue;
            mlir::Value returned = ret.getOperand(i);
            if (auto get = returned.getDefiningOp<subop::CacheGetOp>()) {
               return static_cast<uint64_t>(get.getKey());
            }
         }
      }
   }
   return std::nullopt;
}

// Keep small string helpers for type fingerprints / debug keys.
llvm::SmallVector<std::string, 4> sortedUniqueStrings(llvm::ArrayRef<std::string> in) {
   llvm::SmallVector<std::string, 4> xs(in.begin(), in.end());
   llvm::sort(xs);
   xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
   return xs;
}

std::string joinSortedStrings(llvm::ArrayRef<std::string> xs) {
   auto s = sortedUniqueStrings(xs);
   std::string out;
   for (auto& x : s) {
      if (!out.empty()) out.push_back('|');
      out.append(x);
   }
   return out;
}

static std::string sanitizeBaseName(llvm::StringRef name) {
   auto dollar = name.find('$');
   llvm::StringRef base = name;
   if (dollar != llvm::StringRef::npos) {
      base = name.substr(0, dollar);
   }
   // ColumnManager / other passes may uniquify names as "<base>_u_<n>".
   auto pos = base.rfind("_u_");
   if (pos == llvm::StringRef::npos) return base.str();
   llvm::StringRef tail = base.substr(pos + 3);
   if (tail.empty()) return base.str();
   for (char c : tail) {
      if (c < '0' || c > '9') return base.str();
   }
   return base.substr(0, pos).str();
}

static std::string normalizeGeneratedResultMemberName(llvm::StringRef name) {
   std::string base = sanitizeBaseName(name);
   size_t underscore = base.rfind('_');
   if (underscore == std::string::npos || underscore + 1 >= base.size()) return base;
   for (char c : llvm::StringRef(base).drop_front(underscore + 1)) {
      if (c < '0' || c > '9') return base;
   }
   return base.substr(0, underscore);
}

/// Like `sanitizeBaseName` for scopes/columns, but keeps compiler member slot suffixes (`member$0` vs `member$1`).
static std::string sanitizeMemberSlotName(llvm::StringRef name) {
   auto pos = name.rfind("_u_");
   if (pos == llvm::StringRef::npos) return name.str();
   llvm::StringRef tail = name.substr(pos + 3);
   if (tail.empty()) return name.str();
   for (char c : tail) {
      if (c < '0' || c > '9') return name.str();
   }
   return name.substr(0, pos).str();
}

static std::string sanitizeScopeName(llvm::StringRef scope) {
   // ColumnManager may uniquify scopes as "<base>_u_<n>" across runs/contexts.
   auto pos = scope.rfind("_u_");
   if (pos == llvm::StringRef::npos) return scope.str();
   llvm::StringRef tail = scope.substr(pos + 3);
   if (tail.empty()) return scope.str();
   for (char c : tail) {
      if (c < '0' || c > '9') return scope.str();
   }
   return scope.substr(0, pos).str();
}

static llvm::SmallVector<lingodb::runtime::ExternalDatasourceProperty::Mapping, 8>
filterExternalDatasourceMappingForJoinMatch(
   llvm::ArrayRef<lingodb::runtime::ExternalDatasourceProperty::Mapping> mapping,
   const llvm::SmallSet<std::string, 8>& joinKeyColumnIdentifiersSanitized) {
   llvm::SmallVector<lingodb::runtime::ExternalDatasourceProperty::Mapping, 8> kept;
   for (const auto& m : mapping) {
      if (joinKeyColumnIdentifiersSanitized.contains(sanitizeBaseName(m.identifier)) ||
          joinKeyColumnIdentifiersSanitized.contains(sanitizeBaseName(m.memberName))) {
         kept.push_back(m);
      }
   }
   return kept;
}

static uint64_t hashCombineU64(uint64_t a, uint64_t b) {
   return static_cast<uint64_t>(llvm::hash_combine(a, b));
}

static uint64_t hashType(mlir::Type t) {
   if (!t) return 0;
   // mlir::hash_value(Type) is pointer-like and not stable across MLIRContexts.
   // Use a printed form (with subop-specific normalization handled separately) for cross-module matching.
   std::string s;
   llvm::raw_string_ostream ss(s);
   t.print(ss);
   ss.flush();
   return static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(s)));
}

static uint64_t hashOpName(mlir::Operation& op) {
   return static_cast<uint64_t>(llvm::hash_value(op.getName().getStringRef()));
}

struct StateHashEnv {
   llvm::DenseMap<mlir::Value, uint64_t> hashByState;
   const llvm::DenseSet<mlir::Value>* transparentStates = nullptr;
   const StateDependencyGraph* depGraph = nullptr;
   const llvm::DenseMap<mlir::Value, std::string>* tableDescrByTableState = nullptr;

   uint64_t get(mlir::Value state) const {
      llvm::DenseSet<mlir::Value> visiting;
      return getRec(state, visiting);
   }

   uint64_t getRec(mlir::Value state, llvm::DenseSet<mlir::Value>& visiting) const {
      state = canonicalizeStateValueDeep(state);
      if (auto cacheKey = getCacheGetKeyForState(state)) return *cacheKey;
      auto it = hashByState.find(state);
      if (it != hashByState.end()) return it->second;

      if (transparentStates && depGraph && transparentStates->contains(state)) {
         auto inserted = visiting.insert(state);
         assert(inserted.second && "transparent state hash resolution must be acyclic");
         llvm::SmallVector<uint64_t, 4> predHashes;
         for (mlir::Value pred : directPredecessorsInDepGraph(state, *depGraph)) {
            pred = canonicalizeStateValueDeep(pred);
            if (pred == state) continue;
            if (isGetExternalLeafState(pred)) {
               uint64_t h = hashType(pred.getType());
               if (tableDescrByTableState) {
                  if (auto it = tableDescrByTableState->find(pred); it != tableDescrByTableState->end()) {
                     h = static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(it->second)));
                     h = hashCombineU64(h, hashType(pred.getType()));
                  }
               }
               predHashes.push_back(h);
               continue;
            }
            predHashes.push_back(getRec(pred, visiting));
         }
         llvm::sort(predHashes);
         uint64_t h = static_cast<uint64_t>(llvm::hash_value(llvm::StringRef("transparent_state_deps")));
         h = hashCombineU64(h, hashType(state.getType()));
         for (uint64_t predHash : predHashes) h = hashCombineU64(h, predHash);
         visiting.erase(state);
         return h;
      }

      assert(it != hashByState.end() && "state must have a hash value");
      llvm_unreachable("state must have a hash value");
   }
};

static uint64_t provisionalStateHash(mlir::Value state, unsigned ordinal) {
   uint64_t h = static_cast<uint64_t>(llvm::hash_value(llvm::StringRef("state_provisional")));
   h = hashCombineU64(h, hashType(state.getType()));
   h = hashCombineU64(h, ordinal);
   return h;
}

static std::string fingerprintSortedMemberPairs(subop::MemberManager& mm, llvm::ArrayRef<subop::Member> members);
static std::string fingerprintMemberTypesSequence(subop::MemberManager& mm, llvm::ArrayRef<subop::Member> members);

/// Join HIV cross-query matching: hash/link slots + first stored lookup key; payload columns tracked separately.
struct JoinHivMatchDetails {
   llvm::SmallSet<std::string, 8> indexMemberNamesSanitized;
   llvm::SmallSet<uint64_t, 8> joinKeyColumnAttrHashes;
   llvm::SmallSet<std::string, 8> joinKeyColumnIdentifiersSanitized;
   std::string storedValueMembersFingerprint;
};

static subop::CreateHashIndexedView findCreateHashIndexedViewForState(
   mlir::Value hiv, const llvm::DenseMap<mlir::Value, llvm::SmallVector<subop::ExecutionStepOp, 8>>& writerStepsByState) {
   auto itW = writerStepsByState.find(hiv);
   assert(itW != writerStepsByState.end() && "hash_indexed_view reuse target must have a writer step");
   for (subop::ExecutionStepOp ws : itW->second) {
      subop::CreateHashIndexedView found;
      ws.walk([&](subop::CreateHashIndexedView op) { found = op; });
      if (found) return found;
   }
   assert(false && "hash_indexed_view must be produced by create_hash_indexed_view");
}

static JoinHivMatchDetails computeJoinHivMatchDetails(
   mlir::Value hiv, subop::HashIndexedViewType hivTy, subop::MemberManager& mm,
   lingodb::compiler::dialect::tuples::ColumnManager& columnManager,
   const llvm::DenseMap<mlir::Value, llvm::SmallVector<subop::ExecutionStepOp, 8>>& writerStepsByState,
   llvm::ArrayRef<int> constructionStepIndices,
   const llvm::DenseMap<int, subop::ExecutionStepOp>& stepByIndex) {
   subop::CreateHashIndexedView chiv = findCreateHashIndexedViewForState(hiv, writerStepsByState);

   JoinHivMatchDetails details;
   details.storedValueMembersFingerprint =
      fingerprintSortedMemberPairs(mm, hivTy.getValueMembers().getMembers());

   details.indexMemberNamesSanitized.insert(sanitizeMemberSlotName(mm.getName(chiv.getHashMember().getMember())));
   details.indexMemberNamesSanitized.insert(sanitizeMemberSlotName(mm.getName(chiv.getLinkMember().getMember())));
   auto valueMembers = hivTy.getValueMembers().getMembers();
   subop::Member joinValueMember;
   if (!valueMembers.empty()) {
      joinValueMember = valueMembers.front();
      details.indexMemberNamesSanitized.insert(sanitizeMemberSlotName(mm.getName(joinValueMember)));
   }

   auto recordJoinKeyColumnHash = [&](llvm::StringRef scope, llvm::StringRef name, mlir::Type colTy) {
      std::string nameSan = sanitizeBaseName(name);
      std::string scopeSan = sanitizeScopeName(scope);
      details.joinKeyColumnIdentifiersSanitized.insert(nameSan);
      uint64_t h = static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(scopeSan)));
      h = hashCombineU64(h, static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(nameSan))));
      h = hashCombineU64(h, hashType(colTy));
      details.joinKeyColumnAttrHashes.insert(h);
   };

   if (joinValueMember) {
      auto joinValueSanitized = sanitizeMemberSlotName(mm.getName(joinValueMember));
      llvm::SmallSet<subop::ExecutionStepOp, 8> stepsVisited;
      auto recordJoinKeysFromStep = [&](subop::ExecutionStepOp ws) {
         if (!ws || !stepsVisited.insert(ws).second) return;
         ws.walk([&](subop::MaterializeOp mat) {
            if (!mlir::isa<subop::BufferType>(mat.getState().getType())) return;
            for (auto& [member, colRef] : mat.getMapping().getMapping()) {
               if (sanitizeMemberSlotName(mm.getName(member)) != joinValueSanitized) continue;
               auto [scope, name] = columnManager.getName(&colRef.getColumn());
               recordJoinKeyColumnHash(scope, name, colRef.getColumn().type);
            }
         });
         ws.walk([&](subop::GatherOp gather) {
            for (auto& [member, colDef] : gather.getMapping().getMapping()) {
               (void)member;
               auto [scope, name] = columnManager.getName(&colDef.getColumn());
               if (sanitizeBaseName(name) != sanitizeBaseName(mm.getName(joinValueMember))) continue;
               recordJoinKeyColumnHash(scope, name, colDef.getColumn().type);
            }
         });
      };
      if (auto itW = writerStepsByState.find(hiv); itW != writerStepsByState.end()) {
         for (subop::ExecutionStepOp ws : itW->second) recordJoinKeysFromStep(ws);
      }
      for (int si : constructionStepIndices) {
         auto itS = stepByIndex.find(si);
         if (itS != stepByIndex.end()) recordJoinKeysFromStep(itS->second);
      }
   }

   return details;
}

static void relaxJoinHivDepTokensInProfile(
   llvm::SmallVector<std::string, 8>& depTokensSorted, const JoinHivMatchDetails& joinDetails,
   const llvm::DenseMap<mlir::Value, lingodb::runtime::ExternalDatasourceProperty>& externalDatasourceByTableState,
   const llvm::DenseMap<mlir::Value, std::string>& tableDescrByTableState) {
   if (joinDetails.joinKeyColumnIdentifiersSanitized.empty()) return;
   for (std::string& tok : depTokensSorted) {
      llvm::StringRef tokRef(tok);
      if (!tokRef.consume_front("table:")) continue;
      llvm::StringRef descr = tokRef;
      for (const auto& [tableVal, ds] : externalDatasourceByTableState) {
         mlir::Value tableCanon = canonicalizeStateValueDeep(tableVal);
         auto itFull = tableDescrByTableState.find(tableCanon);
         if (itFull == tableDescrByTableState.end()) itFull = tableDescrByTableState.find(tableVal);
         if (itFull == tableDescrByTableState.end() || itFull->second != descr) continue;
         auto kept = filterExternalDatasourceMappingForJoinMatch(ds.mapping,
                                                                 joinDetails.joinKeyColumnIdentifiersSanitized);
         tok = std::string("table:") + renderExternalDataSourceDescrMatchString(ds, kept, /*includeFilters=*/false);
         break;
      }
   }
   llvm::sort(depTokensSorted);
   depTokensSorted.erase(std::unique(depTokensSorted.begin(), depTokensSorted.end()), depTokensSorted.end());
}

static std::string normalizedHashIndexedViewTypeFingerprintForJoinMatch(subop::MemberManager& mm,
                                                                        subop::HashIndexedViewType hivTy,
                                                                        const JoinHivMatchDetails& details) {
   llvm::SmallVector<subop::Member, 4> indexValueMembers;
   for (auto m : hivTy.getValueMembers().getMembers()) {
      if (details.indexMemberNamesSanitized.contains(sanitizeMemberSlotName(mm.getName(m)))) {
         indexValueMembers.push_back(m);
      }
   }
   return std::string("hash_indexed_view{key_members=") +
          fingerprintSortedMemberPairs(mm, hivTy.getKeyMembers().getMembers()) +
          ",index_value_members=" + fingerprintSortedMemberPairs(mm, indexValueMembers) +
          ",compare=" + (hivTy.getCompareHashForLookup() ? "1" : "0") + "}";
}

static bool residualPredicateValueSupportedForRelaxedHiv(mlir::Value v,
                                                         llvm::DenseSet<mlir::Value>& seen) {
   if (!seen.insert(v).second) return true;
   if (mlir::isa<mlir::BlockArgument>(v)) return true;
   mlir::Operation* op = v.getDefiningOp();
   assert(op && "residual predicate value must have a defining op or be a block argument");
   if (mlir::isa<db::ConstantOp>(op)) return true;
   if (auto cast = mlir::dyn_cast<db::CastOp>(op))
      return residualPredicateValueSupportedForRelaxedHiv(cast.getVal(), seen);
   if (auto cmp = mlir::dyn_cast<db::CmpOp>(op))
      return residualPredicateValueSupportedForRelaxedHiv(cmp.getLeft(), seen) &&
             residualPredicateValueSupportedForRelaxedHiv(cmp.getRight(), seen);
   if (auto between = mlir::dyn_cast<db::BetweenOp>(op))
      return residualPredicateValueSupportedForRelaxedHiv(between.getVal(), seen) &&
             residualPredicateValueSupportedForRelaxedHiv(between.getLower(), seen) &&
             residualPredicateValueSupportedForRelaxedHiv(between.getUpper(), seen);
   if (auto oneOf = mlir::dyn_cast<db::OneOfOp>(op)) {
      if (!residualPredicateValueSupportedForRelaxedHiv(oneOf.getVal(), seen)) return false;
      for (mlir::Value arg : oneOf.getVals())
         if (!residualPredicateValueSupportedForRelaxedHiv(arg, seen)) return false;
      return true;
   }
   if (auto rt = mlir::dyn_cast<db::RuntimeCall>(op)) {
      for (mlir::Value arg : rt.getArgs())
         if (!residualPredicateValueSupportedForRelaxedHiv(arg, seen)) return false;
      return true;
   }
   if (auto andOp = mlir::dyn_cast<db::AndOp>(op)) {
      for (mlir::Value arg : andOp->getOperands())
         if (!residualPredicateValueSupportedForRelaxedHiv(arg, seen)) return false;
      return true;
   }
   if (auto orOp = mlir::dyn_cast<db::OrOp>(op)) {
      for (mlir::Value arg : orOp->getOperands())
         if (!residualPredicateValueSupportedForRelaxedHiv(arg, seen)) return false;
      return true;
   }
   if (auto notOp = mlir::dyn_cast<db::NotOp>(op))
      return residualPredicateValueSupportedForRelaxedHiv(notOp.getVal(), seen);
   if (auto derive = mlir::dyn_cast<db::DeriveTruth>(op))
      return residualPredicateValueSupportedForRelaxedHiv(derive.getVal(), seen);
   return false;
}

static bool residualPredicateValueIsComplexForRelaxedHiv(mlir::Value v,
                                                         llvm::DenseSet<mlir::Value>& seen) {
   if (!seen.insert(v).second) return false;
   if (mlir::isa<mlir::BlockArgument>(v)) return false;
   mlir::Operation* op = v.getDefiningOp();
   assert(op && "residual predicate value must have a defining op or be a block argument");
   if (mlir::isa<db::CastOp, db::CmpOp, db::BetweenOp, db::OneOfOp>(op)) return true;
   for (mlir::Value operand : op->getOperands())
      if (residualPredicateValueIsComplexForRelaxedHiv(operand, seen)) return true;
   return false;
}

static bool residualPredicateMapSupportedForRelaxedHiv(subop::MapOp map,
                                                       subop::FilterOp filter) {
   auto ret = mlir::cast<tuples::ReturnOp>(map.getFn().front().getTerminator());
   for (auto attr : filter.getConditions()) {
      auto cond = mlir::cast<tuples::ColumnRefAttr>(attr);
      bool found = false;
      for (unsigned i = 0; i < map.getComputedCols().size(); ++i) {
         auto def = mlir::cast<tuples::ColumnDefAttr>(map.getComputedCols()[i]);
         if (&def.getColumn() != &cond.getColumn()) continue;
         llvm::DenseSet<mlir::Value> seen;
         if (!residualPredicateValueSupportedForRelaxedHiv(ret.getOperand(i), seen)) return false;
         found = true;
         break;
      }
      assert(found && "residual predicate condition must be produced by predicate map");
   }
   return true;
}

static bool residualPredicateMapComplexForRelaxedHiv(subop::MapOp map,
                                                     subop::FilterOp filter) {
   auto ret = mlir::cast<tuples::ReturnOp>(map.getFn().front().getTerminator());
   for (auto attr : filter.getConditions()) {
      auto cond = mlir::cast<tuples::ColumnRefAttr>(attr);
      for (unsigned i = 0; i < map.getComputedCols().size(); ++i) {
         auto def = mlir::cast<tuples::ColumnDefAttr>(map.getComputedCols()[i]);
         if (&def.getColumn() != &cond.getColumn()) continue;
         llvm::DenseSet<mlir::Value> seen;
         if (residualPredicateValueIsComplexForRelaxedHiv(ret.getOperand(i), seen)) return true;
         break;
      }
   }
   return false;
}

std::string normalizedSubopStateTypeFingerprint(subop::MemberManager& mm, mlir::Type t);

struct StepDagHasher {
   subop::ExecutionStepOp step;
   mlir::Value targetState;
   mlir::Type targetStateType;
   const llvm::DenseMap<mlir::Value, std::string>* tableDescrByTableState = nullptr;
   subop::MemberManager* memberManager = nullptr;
   lingodb::compiler::dialect::tuples::ColumnManager* columnManager = nullptr;
   const llvm::SmallSet<std::string, 8>* joinIndexMemberNamesSanitized = nullptr;
   const llvm::SmallSet<uint64_t, 8>* joinKeyColumnAttrHashes = nullptr;
   const llvm::SmallSet<std::string, 8>* joinKeyColumnIdentifiersSanitized = nullptr;
   const llvm::DenseMap<mlir::Value, lingodb::runtime::ExternalDatasourceProperty>* externalDatasourceByTableState =
      nullptr;
   const StateHashEnv* stateHashEnv = nullptr;
   const llvm::DenseSet<mlir::Value>* targetConstructionStates = nullptr;
   llvm::DenseMap<mlir::Value, uint64_t> memo;
   llvm::DenseMap<const lingodb::compiler::dialect::tuples::Column*, uint64_t> columnHashByColumn;
   llvm::DenseMap<const lingodb::compiler::dialect::tuples::Column*, mlir::Value> refStateByColumn;
   llvm::DenseMap<mlir::Value, uint64_t> regionValueHashByValue;
   llvm::DenseMap<const void*, uint64_t>* outColumnHashByColumn = nullptr;
   using SelectedColumnSet = llvm::SmallSet<uint64_t, 8>;

   bool isWithinStep(mlir::Operation* op) {
      for (auto* p = op; p; p = p->getParentOp()) {
         if (p == step.getOperation()) return true;
      }
      return false;
   }

   bool targetAllowsJoinPayloadRelaxation() const {
      return mlir::isa_and_nonnull<subop::HashIndexedViewType>(targetStateType);
   }

   bool targetAllowsResidualTableFilterSkip() const {
      return mlir::isa_and_nonnull<subop::HashIndexedViewType>(targetStateType);
   }

   subop::MemberManager& memberManagerForType(mlir::Type t) const {
      assert(t && "memberManagerForType requires a type");
      auto* dialect = t.getContext()->getLoadedDialect<subop::SubOperatorDialect>();
      assert(dialect && "subop state type must belong to a context with SubOperatorDialect loaded");
      return dialect->getMemberManager();
   }

   uint64_t hashMlirType(mlir::Type t) const {
      if (!t) return 0;
      if (auto tl = mlir::dyn_cast<subop::ThreadLocalType>(t)) {
         return hashMlirType(tl.getWrapped());
      }
      if (targetAllowsJoinPayloadRelaxation() && joinIndexMemberNamesSanitized && memberManager) {
         if (auto buf = mlir::dyn_cast<subop::BufferType>(t)) {
            auto& typeMm = memberManagerForType(t);
            llvm::SmallVector<subop::Member, 8> kept;
            for (auto m : buf.getMembers().getMembers()) {
               if (joinIndexMemberNamesSanitized->contains(sanitizeMemberSlotName(typeMm.getName(m)))) {
                  kept.push_back(m);
               }
            }
            std::string fp = std::string("buffer{members=") + fingerprintSortedMemberPairs(typeMm, kept) + "}";
            return static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(fp)));
         }
         if (auto hiv = mlir::dyn_cast<subop::HashIndexedViewType>(t)) {
            auto& typeMm = memberManagerForType(t);
            llvm::SmallVector<subop::Member, 4> indexValueMembers;
            for (auto m : hiv.getValueMembers().getMembers()) {
               if (joinIndexMemberNamesSanitized->contains(sanitizeMemberSlotName(typeMm.getName(m)))) {
                  indexValueMembers.push_back(m);
               }
            }
            std::string fp = std::string("hash_indexed_view{key_members=") +
                             fingerprintSortedMemberPairs(typeMm, hiv.getKeyMembers().getMembers()) +
                             ",index_value_members=" + fingerprintSortedMemberPairs(typeMm, indexValueMembers) +
                             ",compare=" + (hiv.getCompareHashForLookup() ? "1" : "0") + "}";
            return static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(fp)));
         }
      }
      if (!targetAllowsJoinPayloadRelaxation()) {
         if (auto list = mlir::dyn_cast<subop::ListType>(t)) {
            uint64_t h = hashString("list");
            return hashCombineU64(h, hashMlirType(list.getT()));
         }
         if (auto tableEntry = mlir::dyn_cast<subop::TableEntryRefType>(t)) {
            auto& typeMm = memberManagerForType(t);
            uint64_t h = hashString("table_entry_ref");
            h = hashCombineU64(h, hashString(fingerprintSortedMemberPairs(
                                    typeMm, tableEntry.getTableColumns().getMembers())));
            return h;
         }
         if (auto lookupEntry = mlir::dyn_cast<subop::LookupEntryRefType>(t)) {
            uint64_t h = hashString("lookup_entry_ref");
            return hashCombineU64(h, hashMlirType(lookupEntry.getState()));
         }
         if (auto hashMapEntry = mlir::dyn_cast<subop::HashMapEntryRefType>(t)) {
            uint64_t h = hashString("hash_map_entry_ref");
            return hashCombineU64(h, hashMlirType(hashMapEntry.getHashMap()));
         }
         if (auto preAggrEntry = mlir::dyn_cast<subop::PreAggrHTEntryRefType>(t)) {
            uint64_t h = hashString("optimistic_ht_entry_ref");
            return hashCombineU64(h, hashMlirType(preAggrEntry.getHashMap()));
         }
         if (auto externalEntry = mlir::dyn_cast<subop::ExternalHashIndexEntryRefType>(t)) {
            uint64_t h = hashString("external_hash_index_entry_ref");
            return hashCombineU64(h, hashMlirType(externalEntry.getExternalHashIndex()));
         }
         if (auto hashMultiEntry = mlir::dyn_cast<subop::HashMultiMapEntryRefType>(t)) {
            uint64_t h = hashString("hash_multimap_entry_ref");
            return hashCombineU64(h, hashMlirType(hashMultiEntry.getHashMultimap()));
         }
      }
      if (memberManager) {
         if (auto hiv = mlir::dyn_cast<subop::HashIndexedViewType>(t)) {
            auto& typeMm = memberManagerForType(t);
            uint64_t h = hashString("hash_indexed_view_type");
            h = hashCombineU64(h, hashString(fingerprintMemberTypesSequence(
                                    typeMm, hiv.getKeyMembers().getMembers())));
            h = hashCombineU64(h, hashString(fingerprintMemberTypesSequence(
                                    typeMm, hiv.getValueMembers().getMembers())));
            h = hashCombineU64(h, hiv.getCompareHashForLookup() ? 1 : 0);
            return h;
         }
      }
      if (memberManager && isStateType(t) && !mlir::isa<subop::HashIndexedViewType>(t)) {
         return hashString(normalizedSubopStateTypeFingerprint(memberManagerForType(t), t));
      }
      return hashType(t);
   }

   uint64_t hashExternalLeaf(mlir::Value v) {
      assert(tableDescrByTableState);
      if (targetAllowsJoinPayloadRelaxation() && externalDatasourceByTableState && joinKeyColumnIdentifiersSanitized) {
         if (auto itDs = externalDatasourceByTableState->find(v); itDs != externalDatasourceByTableState->end()) {
            auto kept =
               filterExternalDatasourceMappingForJoinMatch(itDs->second.mapping, *joinKeyColumnIdentifiersSanitized);
            std::string descr = renderExternalDataSourceDescrMatchString(
               itDs->second, kept, /*includeFilters=*/false);
            uint64_t h = static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(descr)));
            if (auto tableTy = mlir::dyn_cast<subop::TableType>(v.getType())) {
               llvm::SmallVector<subop::Member, 8> keptMembers;
               for (subop::Member m : tableTy.getMembers().getMembers()) {
                  llvm::StringRef memberName = memberManager->getName(m);
                  if (joinKeyColumnIdentifiersSanitized->contains(sanitizeBaseName(memberName)) ||
                      joinKeyColumnIdentifiersSanitized->contains(sanitizeMemberSlotName(memberName))) {
                     keptMembers.push_back(m);
                  }
               }
               std::string fp = std::string("table{members=") +
                                fingerprintSortedMemberPairs(*memberManager, keptMembers) + "}";
               h = hashCombineU64(h, static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(fp))));
            } else {
               h = hashCombineU64(h, hashMlirType(v.getType()));
            }
            return h;
         }
      }
      if (auto it = tableDescrByTableState->find(v); it != tableDescrByTableState->end()) {
         uint64_t h = static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(it->second)));
         h = hashCombineU64(h, hashMlirType(v.getType()));
         return h;
      }
      assert(!isGetExternalLeafState(v) && "get_external leaf state must have GetExternal descr");
      return hashMlirType(v.getType());
   }

   uint64_t hashString(llvm::StringRef s) const {
      return static_cast<uint64_t>(llvm::hash_value(s));
   }

   bool isTargetConstructionState(mlir::Value state) const {
      if (!state || !targetConstructionStates) return false;
      return targetConstructionStates->contains(canonicalizeStateValueDeep(state));
   }

   uint64_t hashTargetConstructionState(mlir::Type t) const {
      uint64_t h = hashString("construction_target_state");
      if (auto tl = mlir::dyn_cast<subop::ThreadLocalType>(t)) t = tl.getWrapped();
      if (targetAllowsJoinPayloadRelaxation()) return hashCombineU64(h, hashMlirType(t));
      if (memberManager && t && !mlir::isa<subop::HashIndexedViewType>(t)) {
         h = hashCombineU64(h, hashString(normalizedSubopStateTypeFingerprint(*memberManager, t)));
      } else {
         h = hashCombineU64(h, hashMlirType(t));
      }
      return h;
   }

   uint64_t hashMemberSemantic(subop::Member member) const {
      assert(memberManager);
      std::string name = sanitizeMemberSlotName(memberManager->getName(member));
      uint64_t h = hashString(name);
      h = hashCombineU64(h, hashMlirType(memberManager->getType(member)));
      return h;
   }

   uint64_t hashStateMemberSemantic(mlir::Value state, subop::Member member) {
      state = canonicalizeStateValueDeep(state);
      if (!isTargetConstructionState(state)) return hashMemberSemantic(member);
      auto members = getMembersForStateValue(state);
      auto itMember = llvm::find(members, member);
      assert(itMember != members.end() && "target member must belong to state payload layout");
      uint64_t h = hashString("target_member_pos");
      h = hashCombineU64(h, static_cast<uint64_t>(std::distance(members.begin(), itMember)));
      h = hashCombineU64(h, hashMlirType(memberManager->getType(member)));
      return h;
   }

   uint64_t hashStateMemberColumn(mlir::Value state, subop::Member member, mlir::Type colTy) {
      assert(memberManager);
      state = canonicalizeStateValueDeep(state);
      uint64_t h = hashString("state_column");
      if (externalDatasourceByTableState) {
         auto itDs = externalDatasourceByTableState->find(state);
         if (itDs == externalDatasourceByTableState->end()) {
            for (auto& kv : *externalDatasourceByTableState) {
               if (canonicalizeStateValueDeep(kv.first) == state) {
                  itDs = externalDatasourceByTableState->find(kv.first);
                  break;
               }
            }
         }
         if (itDs != externalDatasourceByTableState->end() && mlir::isa<subop::TableType>(state.getType())) {
            h = hashCombineU64(h, hashExternalLeaf(state));
            h = hashCombineU64(h, hashString(itDs->second.tableName));
            llvm::StringRef memberName = memberManager->getName(member);
            std::string memberNameSan = sanitizeBaseName(memberName);
            bool found = false;
            for (const auto& m : itDs->second.mapping) {
               if (m.memberName == memberName || sanitizeBaseName(m.memberName) == memberNameSan) {
                  h = hashCombineU64(h, hashString(m.identifier));
                  h = hashCombineU64(h, hashString(normalizeMappingMemberName(m.memberName)));
                  found = true;
                  break;
               }
            }
            if (!found) h = hashCombineU64(h, hashMemberSemantic(member));
            return hashCombineU64(h, hashMlirType(colTy));
         }
      }
      assert(stateHashEnv && "non-table state column hashing requires state hash environment");
      h = hashCombineU64(h, isTargetConstructionState(state) ? hashTargetConstructionState(state.getType())
                                                             : stateHashEnv->get(state));
      auto members = getMembersForStateValue(state);
      auto itMember = llvm::find(members, member);
      assert(itMember != members.end() && "scanned member must belong to state payload layout");
      uint64_t memberPos = static_cast<uint64_t>(std::distance(members.begin(), itMember));
      h = hashCombineU64(h, hashString("member_pos"));
      h = hashCombineU64(h, memberPos);
      return hashCombineU64(h, hashMlirType(colTy));
   }

   void defineColumn(lingodb::compiler::dialect::tuples::ColumnDefAttr def, uint64_t h) {
      columnHashByColumn[&def.getColumn()] = h;
      if (outColumnHashByColumn) (*outColumnHashByColumn)[&def.getColumn()] = h;
   }

   void defineRefColumn(lingodb::compiler::dialect::tuples::ColumnDefAttr def, uint64_t h, mlir::Value state) {
      defineColumn(def, h);
      if (state) refStateByColumn[&def.getColumn()] = canonicalizeStateValueDeep(state);
   }

   uint64_t hashColumnRef(lingodb::compiler::dialect::tuples::ColumnRefAttr ref) {
      auto it = columnHashByColumn.find(&ref.getColumn());
      assert(it != columnHashByColumn.end() && "ColumnRefAttr must be bound by the stream column hash environment");
      return it->second;
   }

   uint64_t hashColumnDef(lingodb::compiler::dialect::tuples::ColumnDefAttr def) {
      auto it = columnHashByColumn.find(&def.getColumn());
      assert(it != columnHashByColumn.end() && "ColumnDefAttr must be bound by the stream column hash environment");
      return it->second;
   }

   mlir::Value refStateFor(lingodb::compiler::dialect::tuples::ColumnRefAttr ref) {
      auto it = refStateByColumn.find(&ref.getColumn());
      assert(it != refStateByColumn.end() && "reference ColumnRefAttr must be bound to its source state");
      return it->second;
   }

   uint64_t hashRegion(mlir::Region& r) {
      // Order-insensitive: hash only the SSA graph that feeds terminators.
      uint64_t h = 0;
      for (auto& b : r) {
         if (b.empty()) continue;
         auto* term = b.getTerminator();
         h = hashCombineU64(h, static_cast<uint64_t>(b.getNumArguments()));
         for (auto v : term->getOperands()) {
            h = hashCombineU64(h, hashValue(v));
         }
      }
      return h;
   }

   uint64_t hashAttrNormalized(mlir::Attribute a) {
      if (!a) return 0;
      // Stable hashing for column/member attributes across MLIRContexts.
      if (auto mr = mlir::dyn_cast<subop::MemberAttr>(a)) {
         assert(memberManager);
         auto m = mr.getMember();
         auto name = sanitizeBaseName(memberManager->getName(m));
         uint64_t h = static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(name)));
         h = hashCombineU64(h, hashMlirType(memberManager->getType(m)));
         return h;
      }
      if (auto cr = mlir::dyn_cast<lingodb::compiler::dialect::tuples::ColumnRefAttr>(a)) {
         return hashColumnRef(cr);
      }
      if (auto cd = mlir::dyn_cast<lingodb::compiler::dialect::tuples::ColumnDefAttr>(a)) {
         return hashColumnDef(cd);
      }
      if (auto fsym = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(a)) {
         auto v = fsym.getValue();
         std::string norm = sanitizeScopeName(v);
         return static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(norm)));
      }
      if (auto sym = mlir::dyn_cast<mlir::SymbolRefAttr>(a)) {
         uint64_t h = 0;
         std::string root = sanitizeScopeName(sym.getRootReference().getValue());
         h = hashCombineU64(h, static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(root))));
         for (auto n : sym.getNestedReferences()) {
            // Nested names are usually stable, but normalize scopes too for safety.
            std::string nn = sanitizeScopeName(n.getValue());
            h = hashCombineU64(h, static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(nn))));
         }
         return h;
      }
      if (auto arr = mlir::dyn_cast<mlir::ArrayAttr>(a)) {
         uint64_t h = 0;
         for (auto x : arr) h = hashCombineU64(h, hashAttrNormalized(x));
         return h;
      }
      if (auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(a)) {
         llvm::SmallVector<mlir::NamedAttribute, 16> xs(dict.getValue().begin(), dict.getValue().end());
         llvm::sort(xs, [](auto a, auto b) { return a.getName().strref() < b.getName().strref(); });
         uint64_t h = 0;
         for (auto na : xs) {
            h = hashCombineU64(h, static_cast<uint64_t>(llvm::hash_value(na.getName().strref())));
            h = hashCombineU64(h, hashAttrNormalized(na.getValue()));
         }
         return h;
      }
      std::string s;
      llvm::raw_string_ostream ss(s);
      a.print(ss);
      ss.flush();
      return static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(s)));
   }

   uint64_t hashSelectedStreamColumnAttr(mlir::Attribute a) {
      if (auto cr = mlir::dyn_cast<lingodb::compiler::dialect::tuples::ColumnRefAttr>(a)) {
         return hashColumnRef(cr);
      }
      if (auto cd = mlir::dyn_cast<lingodb::compiler::dialect::tuples::ColumnDefAttr>(a)) {
         return hashColumnDef(cd);
      }
      return hashAttrNormalized(a);
   }

   bool includeMemberForJoinIndexHash(llvm::StringRef memberNameSanitized) const {
      if (!targetAllowsJoinPayloadRelaxation() || !joinIndexMemberNamesSanitized) return true;
      return joinIndexMemberNamesSanitized->contains(memberNameSanitized.str());
   }

   uint64_t hashAttrDictSorted(mlir::Operation& op) {
      llvm::SmallVector<mlir::NamedAttribute, 16> attrs(op.getAttrs().begin(), op.getAttrs().end());
      llvm::sort(attrs, [](auto a, auto b) { return a.getName().strref() < b.getName().strref(); });
      uint64_t h = 0;
      for (auto na : attrs) {
         h = hashCombineU64(h, static_cast<uint64_t>(llvm::hash_value(na.getName().strref())));
         h = hashCombineU64(h, hashAttrNormalized(na.getValue()));
      }
      return h;
   }

   uint64_t hashGatherOpRelaxed(subop::GatherOp gather) {
      uint64_t h = hashValue(gather.getStream());
      mlir::Value refState = refStateFor(gather.getRef());
      if (joinKeyColumnAttrHashes) {
         for (auto& [member, colDef] : gather.getMapping().getMapping()) {
            uint64_t colH = hashStateMemberColumn(refState, member, colDef.getColumn().type);
            defineColumn(colDef, colH);
            bool isJoinKey = joinKeyColumnAttrHashes->contains(colH);
            if (!isJoinKey && joinKeyColumnIdentifiersSanitized && columnManager) {
               auto [scope, name] = columnManager->getName(&colDef.getColumn());
               (void)scope;
               isJoinKey = joinKeyColumnIdentifiersSanitized->contains(sanitizeBaseName(name));
            }
            if (!isJoinKey) continue;
            (void)member;
            h = hashCombineU64(h, colH);
         }
      }
      return h;
   }

   static std::string opOneLine(mlir::Operation* op) {
      std::string s;
      llvm::raw_string_ostream os(s);
      op->print(os);
      os.flush();
      for (char& c : s) {
         if (c == '\n' || c == '\r' || c == '\t') c = ' ';
      }
      if (s.size() > 500) s.resize(500);
      return s;
   }

   bool valueHasOnlyUseBy(mlir::Value v, mlir::Operation* expectedUser) {
      mlir::Operation* onlyUser = nullptr;
      for (mlir::OpOperand& use : v.getUses()) {
         mlir::Operation* user = use.getOwner();
         if (!isWithinStep(user)) continue;
         if (onlyUser && onlyUser != user) return false;
         onlyUser = user;
      }
      return onlyUser == expectedUser;
   }

   subop::ScanRefsOp traceSingleUseStreamToTableScan(mlir::Operation* user, mlir::Value stream) {
      for (;;) {
         mlir::Operation* def = stream.getDefiningOp();
         if (!def || !isWithinStep(def)) return {};
         if (!valueHasOnlyUseBy(stream, user)) return {};
         if (auto scan = mlir::dyn_cast<subop::ScanRefsOp>(def)) {
            if (mlir::isa<subop::TableType>(scan.getState().getType())) return scan;
            return {};
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
         if (auto filter = mlir::dyn_cast<subop::FilterOp>(def)) {
            user = def;
            stream = filter.getStream();
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

   subop::MapOp predicateMapForRelaxedTableFilter(subop::FilterOp filter) {
      auto map = mlir::dyn_cast_or_null<subop::MapOp>(filter.getStream().getDefiningOp());
      if (!map) return {};
      if (!valueHasOnlyUseBy(map.getResult(), filter.getOperation())) return {};

      llvm::DenseSet<const void*> computedCols;
      for (auto attr : map.getComputedCols()) {
         auto colDef = mlir::cast<lingodb::compiler::dialect::tuples::ColumnDefAttr>(attr);
         computedCols.insert(&colDef.getColumn());
      }
      for (auto attr : filter.getConditions()) {
         auto cond = mlir::cast<lingodb::compiler::dialect::tuples::ColumnRefAttr>(attr);
         if (!computedCols.contains(&cond.getColumn())) return {};
      }
      if (!residualPredicateMapSupportedForRelaxedHiv(map, filter)) return {};
      if (!traceSingleUseStreamToTableScan(map.getOperation(), map.getStream())) return {};
      return map;
   }

   bool isRelaxedTableFilter(subop::FilterOp filter, subop::MapOp& predMap, subop::ScanRefsOp& scan) {
      predMap = predicateMapForRelaxedTableFilter(filter);
      if (!predMap) return false;
      scan = traceSingleUseStreamToTableScan(predMap.getOperation(), predMap.getStream());
      return static_cast<bool>(scan);
   }

   void bindRegionArgs(mlir::Region& r, llvm::ArrayRef<uint64_t> argHashes) {
      if (r.empty()) return;
      auto& block = r.front();
      assert(block.getNumArguments() >= argHashes.size() && "region has fewer block args than mapped column hashes");
      for (size_t i = 0; i < argHashes.size(); ++i) {
         regionValueHashByValue[block.getArgument(i)] = argHashes[i];
      }
   }

   uint64_t hashMapOp(subop::MapOp map) {
      uint64_t upstream = hashValue(map.getStream());
      llvm::SmallVector<uint64_t, 8> inputHashes;
      inputHashes.reserve(map.getInputCols().size());
      for (auto attr : map.getInputCols()) {
         inputHashes.push_back(hashColumnRef(mlir::cast<lingodb::compiler::dialect::tuples::ColumnRefAttr>(attr)));
      }
      bindRegionArgs(map.getFn(), inputHashes);

      auto ret = mlir::cast<tuples::ReturnOp>(map.getFn().front().getTerminator());
      assert(ret.getNumOperands() == map.getComputedCols().size() &&
             "map return values must align with computed columns");
      uint64_t regionH = hashRegion(map.getFn());
      llvm::SmallVector<uint64_t, 8> sortedInputHashes(inputHashes.begin(), inputHashes.end());
      llvm::sort(sortedInputHashes);
      for (size_t i = 0; i < map.getComputedCols().size(); ++i) {
         auto def = mlir::cast<lingodb::compiler::dialect::tuples::ColumnDefAttr>(map.getComputedCols()[i]);
         uint64_t colH = hashString("map_column");
         colH = hashCombineU64(colH, hashOpName(*map.getOperation()));
         colH = hashCombineU64(colH, upstream);
         colH = hashCombineU64(colH, regionH);
         colH = hashCombineU64(colH, hashValue(ret.getOperand(i)));
         for (uint64_t inH : sortedInputHashes) colH = hashCombineU64(colH, inH);
         colH = hashCombineU64(colH, hashMlirType(def.getColumn().type));
         defineColumn(def, colH);
      }
      return upstream;
   }

   uint64_t hashFilterOp(subop::FilterOp filter) {
      uint64_t h = hashValue(filter.getStream());
      uint64_t local = hashOpName(*filter.getOperation());
      local = hashCombineU64(local, hashAttrNormalized(filter.getFilterSemanticAttr()));
      llvm::SmallVector<uint64_t, 8> conditionHashes;
      for (auto condAttr : filter.getConditions()) {
         auto cond = mlir::cast<lingodb::compiler::dialect::tuples::ColumnRefAttr>(condAttr);
         conditionHashes.push_back(hashColumnRef(cond));
      }
      llvm::sort(conditionHashes);
      for (uint64_t condH : conditionHashes) local = hashCombineU64(local, condH);
      return hashCombineU64(local, h);
   }

   uint64_t hashScanOp(subop::ScanOp scan) {
      uint64_t stateH = hashValue(scan.getState());
      uint64_t h = hashCombineU64(hashOpName(*scan.getOperation()), stateH);
      mlir::Value state = canonicalizeStateValueDeep(scan.getState());
      for (auto& [member, colDef] : scan.getMapping().getMapping()) {
         uint64_t colH = hashStateMemberColumn(state, member, colDef.getColumn().type);
         defineColumn(colDef, colH);
         h = hashCombineU64(h, colH);
      }
      return h;
   }

   uint64_t hashScanRefsOp(subop::ScanRefsOp scan) {
      uint64_t stateH = hashValue(scan.getState());
      uint64_t refH = hashString("scan_ref");
      refH = hashCombineU64(refH, stateH);
      refH = hashCombineU64(refH, hashMlirType(scan.getRef().getColumn().type));
      defineRefColumn(scan.getRef(), refH, scan.getState());
      return hashCombineU64(hashOpName(*scan.getOperation()), refH);
   }

   uint64_t hashScanListOp(subop::ScanListOp scan) {
      uint64_t listH = hashValue(scan.getList());
      uint64_t refH = hashString("scan_list_ref");
      refH = hashCombineU64(refH, listH);
      defineRefColumn(scan.getElem(), refH, findUpstreamLookupHashIndexedView(scan.getList()));
      return hashCombineU64(hashOpName(*scan.getOperation()), refH);
   }

   uint64_t hashLookupLikeOp(mlir::Operation& op, mlir::Value stream, mlir::Value state,
                             mlir::ArrayAttr keys,
                             lingodb::compiler::dialect::tuples::ColumnDefAttr ref) {
      uint64_t h = hashCombineU64(hashOpName(op), hashValue(stream));
      h = hashCombineU64(h, hashValue(state));
      for (auto key : keys) {
         h = hashCombineU64(h, hashColumnRef(mlir::cast<lingodb::compiler::dialect::tuples::ColumnRefAttr>(key)));
      }
      uint64_t refH = hashString("lookup_ref");
      refH = hashCombineU64(refH, h);
      defineRefColumn(ref, refH, state);
      return hashCombineU64(h, refH);
   }

   uint64_t hashGatherMappingOnly(subop::GatherOp gather) {
      uint64_t h = hashValue(gather.getStream());
      mlir::Value state = refStateFor(gather.getRef());
      for (auto& [member, colDef] : gather.getMapping().getMapping()) {
         uint64_t colH = hashStateMemberColumn(state, member, colDef.getColumn().type);
         defineColumn(colDef, colH);
      }
      return h;
   }

   uint64_t hashRenamingMappingOnly(subop::RenamingOp rename) {
      uint64_t h = hashValue(rename.getStream());
      for (auto attr : rename.getColumns()) {
         auto def = mlir::cast<lingodb::compiler::dialect::tuples::ColumnDefAttr>(attr);
         assert(def.getFromExisting() && "renaming columns must carry fromExisting");
         auto arr = mlir::dyn_cast<mlir::ArrayAttr>(def.getFromExisting());
         assert(arr && arr.size() == 1 && "renaming fromExisting must be a single column ref");
         uint64_t fromH = hashColumnRef(mlir::cast<lingodb::compiler::dialect::tuples::ColumnRefAttr>(arr[0]));
         defineColumn(def, fromH);
      }
      return h;
   }

   uint64_t hashSelectedStreamProducer(mlir::Value stream, const SelectedColumnSet& selected) {
      auto* def = stream.getDefiningOp();
      assert(def && "relaxed join hash expects tuple stream producer op");
      assert(isWithinStep(def) && "relaxed join hash only traces producers within one execution_step");

      if (auto gather = mlir::dyn_cast<subop::GatherOp>(def)) {
         hashGatherMappingOnly(gather);
         for (auto& [member, colDef] : gather.getMapping().getMapping()) {
            (void)member;
            uint64_t outH = hashColumnDef(colDef);
            if (!selected.contains(outH)) continue;
         }
         return hashSelectedStreamProducer(gather.getStream(), selected);
      }

      if (auto map = mlir::dyn_cast<subop::MapOp>(def)) {
         hashMapOp(map);
         SelectedColumnSet nextSelected(selected.begin(), selected.end());
         bool matched = false;
         for (auto attr : map.getComputedCols()) {
            auto colDef = mlir::cast<lingodb::compiler::dialect::tuples::ColumnDefAttr>(attr);
            uint64_t outH = hashColumnDef(colDef);
            if (!selected.contains(outH)) continue;
            matched = true;
            nextSelected.erase(outH);
            for (auto inAttr : map.getInputCols()) {
               auto inCol = mlir::cast<lingodb::compiler::dialect::tuples::ColumnRefAttr>(inAttr);
               nextSelected.insert(hashColumnRef(inCol));
            }
         }
         (void)matched;
         return hashSelectedStreamProducer(map.getStream(), nextSelected);
      }

      if (auto filter = mlir::dyn_cast<subop::FilterOp>(def)) {
         subop::MapOp predMap;
         subop::ScanRefsOp scan;
         if (isRelaxedTableFilter(filter, predMap, scan)) {
            return hashSelectedStreamProducer(predMap.getStream(), selected);
         }
         if (targetAllowsResidualTableFilterSkip() &&
             traceSingleUseStreamToTableScan(filter.getOperation(), filter.getStream())) {
            return hashSelectedStreamProducer(filter.getStream(), selected);
         }
         SelectedColumnSet nextSelected(selected.begin(), selected.end());
         uint64_t local = hashOpName(*def);
         local = hashCombineU64(local, hashAttrNormalized(filter.getFilterSemanticAttr()));
         for (auto condAttr : filter.getConditions()) {
            uint64_t condH = hashColumnRef(mlir::cast<lingodb::compiler::dialect::tuples::ColumnRefAttr>(condAttr));
            local = hashCombineU64(local, condH);
            nextSelected.insert(condH);
         }
         return hashCombineU64(local, hashSelectedStreamProducer(filter.getStream(), nextSelected));
      }

      if (auto rename = mlir::dyn_cast<subop::RenamingOp>(def)) {
         hashRenamingMappingOnly(rename);
         SelectedColumnSet nextSelected;
         for (uint64_t key : selected) nextSelected.insert(key);
         for (auto attr : rename.getColumns()) {
            auto colDef = mlir::cast<lingodb::compiler::dialect::tuples::ColumnDefAttr>(attr);
            uint64_t outH = hashColumnDef(colDef);
            if (!selected.contains(outH) || !colDef.getFromExisting()) continue;
            nextSelected.erase(outH);
            nextSelected.insert(outH);
         }
         return hashSelectedStreamProducer(rename.getStream(), nextSelected);
      }

      if (auto scan = mlir::dyn_cast<subop::ScanRefsOp>(def)) {
         return hashScanRefsOp(scan);
      }

      uint64_t h = hashCombineU64(hashOpName(*def), hashOp(*def));
      llvm::SmallVector<uint64_t, 8> selectedHashes(selected.begin(), selected.end());
      llvm::sort(selectedHashes);
      for (uint64_t selectedHash : selectedHashes) h = hashCombineU64(h, selectedHash);
      return h;
   }

   uint64_t hashMaterializeOpRelaxed(subop::MaterializeOp mat) {
      uint64_t streamH = hashValue(mat.getStream());
      uint64_t h = hashOpName(*mat.getOperation());
      h = hashCombineU64(h, streamH);
      for (auto t : mat->getResultTypes()) h = hashCombineU64(h, hashMlirType(t));
      SelectedColumnSet selected;
      for (auto& [member, colRef] : mat.getMapping().getMapping()) {
         if (!includeMemberForJoinIndexHash(sanitizeMemberSlotName(memberManager->getName(member)))) continue;
         uint64_t colH = hashSelectedStreamColumnAttr(colRef);
         selected.insert(colH);
         h = hashCombineU64(h, colH);
         h = hashCombineU64(h, hashAttrNormalized(subop::MemberAttr::get(mat->getContext(), member)));
      }
      h = hashCombineU64(h, hashSelectedStreamProducer(mat.getStream(), selected));
      return h;
   }

   uint64_t hashMaterializeOp(subop::MaterializeOp mat) {
      if (mlir::isa<subop::ResultTableType>(mat.getState().getType())) {
         auto rt = mlir::cast<subop::ResultTableType>(mat.getState().getType());
         llvm::ArrayRef<subop::Member> resultMembers = rt.getMembers().getMembers();
         uint64_t h = hashCombineU64(hashOpName(*mat.getOperation()), hashString("result_table_materialize"));
         std::string typeFp = normalizedSubopStateTypeFingerprint(*memberManager, rt);
         h = hashCombineU64(h, hashString(typeFp));
         llvm::SmallVector<std::tuple<unsigned, std::string, uint64_t>, 16> fields;
         for (auto& [member, colRef] : mat.getMapping().getMapping()) {
            auto itMember = llvm::find(resultMembers, member);
            assert(itMember != resultMembers.end() && "result_table materialize member must belong to result layout");
            unsigned ordinal = static_cast<unsigned>(std::distance(resultMembers.begin(), itMember));
            auto [scope, leaf] = columnManager->getName(&colRef.getColumn());
            std::string colSem = sanitizeScopeName(scope);
            colSem.push_back('.');
            colSem.append(normalizeGeneratedResultMemberName(leaf));
            uint64_t colH = hashString(colSem);
            colH = hashCombineU64(colH, hashMlirType(colRef.getColumn().type));
            fields.push_back({ordinal, normalizeGeneratedResultMemberName(memberManager->getName(member)), colH});
         }
         llvm::sort(fields, [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
         for (const auto& [ordinal, memberName, colH] : fields) {
            h = hashCombineU64(h, ordinal);
            h = hashCombineU64(h, hashString(memberName));
            h = hashCombineU64(h, colH);
         }
         return h;
      }
      uint64_t h = hashCombineU64(hashOpName(*mat.getOperation()), hashValue(mat.getStream()));
      h = hashCombineU64(h, hashValue(mat.getState()));
      for (auto& [member, colRef] : mat.getMapping().getMapping()) {
         h = hashCombineU64(h, hashStateMemberSemantic(mat.getState(), member));
         h = hashCombineU64(h, hashColumnRef(colRef));
      }
      return h;
   }

   uint64_t hashColumnRefMemberMappingWrite(mlir::Operation& op, mlir::Value stream,
                                            lingodb::compiler::dialect::tuples::ColumnRefAttr ref,
                                            subop::ColumnRefMemberMappingAttr mapping) {
      uint64_t h = hashCombineU64(hashOpName(op), hashValue(stream));
      if (ref) h = hashCombineU64(h, hashColumnRef(ref));
      mlir::Value state = ref ? refStateFor(ref) : mlir::Value{};
      for (auto& [member, colRef] : mapping.getMapping()) {
         h = hashCombineU64(h, state ? hashStateMemberSemantic(state, member) : hashMemberSemantic(member));
         h = hashCombineU64(h, hashColumnRef(colRef));
      }
      return h;
   }

   uint64_t hashReduceOp(subop::ReduceOp reduce) {
      uint64_t h = hashCombineU64(hashOpName(*reduce.getOperation()), hashValue(reduce.getStream()));
      h = hashCombineU64(h, hashColumnRef(reduce.getRef()));
      llvm::SmallVector<uint64_t, 8> regionArgs;
      for (auto attr : reduce.getColumns()) {
         uint64_t colH = hashColumnRef(mlir::cast<lingodb::compiler::dialect::tuples::ColumnRefAttr>(attr));
         regionArgs.push_back(colH);
         h = hashCombineU64(h, colH);
      }
      mlir::Value state = refStateFor(reduce.getRef());
      for (auto attr : reduce.getMembers()) {
         auto member = mlir::cast<subop::MemberAttr>(attr).getMember();
         uint64_t memberH = hashStateMemberColumn(state, member, memberManager->getType(member));
         regionArgs.push_back(memberH);
         h = hashCombineU64(h, hashStateMemberSemantic(state, member));
      }
      bindRegionArgs(reduce.getRegion(), regionArgs);
      h = hashCombineU64(h, hashRegion(reduce.getRegion()));
      if (!reduce.getCombine().empty()) h = hashCombineU64(h, hashRegion(reduce.getCombine()));
      return h;
   }

   uint64_t hashOp(mlir::Operation& op) {
      if (auto cmp = mlir::dyn_cast<db::CmpOp>(&op)) {
         using P = db::DBCmpPredicate;
         P predicate = cmp.getPredicate();
         uint64_t leftH = hashValue(cmp.getLeft());
         uint64_t rightH = hashValue(cmp.getRight());
         if (predicate == P::eq || predicate == P::neq || predicate == P::isa) {
            if (rightH < leftH) std::swap(leftH, rightH);
         } else if (predicate == P::gt || predicate == P::gte) {
            predicate = predicate == P::gt ? P::lt : P::lte;
            std::swap(leftH, rightH);
         }
         uint64_t h = 0;
         h = hashCombineU64(h, hashOpName(op));
         h = hashCombineU64(h, hashAttrNormalized(op.getPropertiesAsAttribute()));
         for (auto t : op.getResultTypes()) h = hashCombineU64(h, hashMlirType(t));
         h = hashCombineU64(h, static_cast<uint64_t>(predicate));
         h = hashCombineU64(h, leftH);
         h = hashCombineU64(h, rightH);
         for (auto& r : op.getRegions()) h = hashCombineU64(h, hashRegion(r));
         return h;
      }
      if (auto oneOf = mlir::dyn_cast<db::OneOfOp>(&op)) {
         uint64_t h = 0;
         h = hashCombineU64(h, hashOpName(op));
         h = hashCombineU64(h, hashAttrDictSorted(op));
         h = hashCombineU64(h, hashAttrNormalized(op.getPropertiesAsAttribute()));
         for (auto t : op.getResultTypes()) h = hashCombineU64(h, hashMlirType(t));
         h = hashCombineU64(h, hashValue(oneOf.getVal()));
         llvm::SmallVector<uint64_t, 8> valueHashes;
         for (auto v : oneOf.getVals()) valueHashes.push_back(hashValue(v));
         llvm::sort(valueHashes);
         for (uint64_t valueH : valueHashes) h = hashCombineU64(h, valueH);
         return h;
      }
      if (op.hasTrait<mlir::OpTrait::IsCommutative>() || mlir::isa<db::AndOp, db::OrOp>(&op)) {
         uint64_t h = 0;
         h = hashCombineU64(h, hashOpName(op));
         h = hashCombineU64(h, hashAttrDictSorted(op));
         h = hashCombineU64(h, hashAttrNormalized(op.getPropertiesAsAttribute()));
         for (auto t : op.getResultTypes()) h = hashCombineU64(h, hashMlirType(t));
         llvm::SmallVector<uint64_t, 8> operandHashes;
         for (auto v : op.getOperands()) operandHashes.push_back(hashValue(v));
         llvm::sort(operandHashes);
         for (uint64_t operandH : operandHashes) h = hashCombineU64(h, operandH);
         for (auto& r : op.getRegions()) h = hashCombineU64(h, hashRegion(r));
         return h;
      }
      if (targetAllowsJoinPayloadRelaxation()) {
         if (auto gather = mlir::dyn_cast<subop::GatherOp>(&op)) return hashGatherOpRelaxed(gather);
         if (auto mat = mlir::dyn_cast<subop::MaterializeOp>(&op)) return hashMaterializeOpRelaxed(mat);
         if (isCreateLike(op)) {
            uint64_t h = hashOpName(op);
            for (auto t : op.getResultTypes()) h = hashCombineU64(h, hashMlirType(t));
            for (auto v : op.getOperands()) h = hashCombineU64(h, hashValue(v));
            return h;
         }
      }
      if (auto scan = mlir::dyn_cast<subop::ScanOp>(&op)) return hashScanOp(scan);
      if (auto scanRefs = mlir::dyn_cast<subop::ScanRefsOp>(&op)) return hashScanRefsOp(scanRefs);
      if (auto scanList = mlir::dyn_cast<subop::ScanListOp>(&op)) return hashScanListOp(scanList);
      if (auto gather = mlir::dyn_cast<subop::GatherOp>(&op)) return hashGatherMappingOnly(gather);
      if (auto rename = mlir::dyn_cast<subop::RenamingOp>(&op)) return hashRenamingMappingOnly(rename);
      if (auto map = mlir::dyn_cast<subop::MapOp>(&op)) return hashMapOp(map);
      if (auto filter = mlir::dyn_cast<subop::FilterOp>(&op)) return hashFilterOp(filter);
      if (auto mat = mlir::dyn_cast<subop::MaterializeOp>(&op)) return hashMaterializeOp(mat);
      if (auto lookup = mlir::dyn_cast<subop::LookupOp>(&op))
         return hashLookupLikeOp(op, lookup.getStream(), lookup.getState(), lookup.getKeys(), lookup.getRef());
      if (auto lookupOrInsert = mlir::dyn_cast<subop::LookupOrInsertOp>(&op))
         return hashLookupLikeOp(op, lookupOrInsert.getStream(), lookupOrInsert.getState(),
                                 lookupOrInsert.getKeys(), lookupOrInsert.getRef());
      if (auto insert = mlir::dyn_cast<subop::InsertOp>(&op))
         return hashColumnRefMemberMappingWrite(op, insert.getStream(), {}, insert.getMapping());
      if (auto scatter = mlir::dyn_cast<subop::ScatterOp>(&op))
         return hashColumnRefMemberMappingWrite(op, scatter.getStream(), scatter.getRef(), scatter.getMapping());
      if (auto reduce = mlir::dyn_cast<subop::ReduceOp>(&op)) return hashReduceOp(reduce);
      if (auto begin = mlir::dyn_cast<subop::GetBeginReferenceOp>(&op)) {
         uint64_t streamH = hashValue(begin.getStream());
         uint64_t refH = hashCombineU64(hashOpName(op), streamH);
         refH = hashCombineU64(refH, hashValue(begin.getState()));
         refH = hashCombineU64(refH, hashMlirType(begin.getRef().getColumn().type));
         defineRefColumn(begin.getRef(), refH, begin.getState());
         return streamH;
      }
      if (auto end = mlir::dyn_cast<subop::GetEndReferenceOp>(&op)) {
         uint64_t streamH = hashValue(end.getStream());
         uint64_t refH = hashCombineU64(hashOpName(op), streamH);
         refH = hashCombineU64(refH, hashValue(end.getState()));
         refH = hashCombineU64(refH, hashMlirType(end.getRef().getColumn().type));
         defineRefColumn(end.getRef(), refH, end.getState());
         return streamH;
      }
      if (auto unwrap = mlir::dyn_cast<subop::UnwrapOptionalRefOp>(&op)) {
         uint64_t streamH = hashValue(unwrap.getStream());
         uint64_t refH = hashColumnRef(unwrap.getOptionalRef());
         uint64_t outRefH = hashCombineU64(hashOpName(op), streamH);
         outRefH = hashCombineU64(outRefH, refH);
         outRefH = hashCombineU64(outRefH, hashMlirType(unwrap.getRef().getColumn().type));
         defineRefColumn(unwrap.getRef(), outRefH, refStateFor(unwrap.getOptionalRef()));
         return streamH;
      }
      if (auto entries = mlir::dyn_cast<subop::EntriesBetweenOp>(&op)) {
         uint64_t streamH = hashValue(entries.getStream());
         uint64_t refH = hashCombineU64(hashOpName(op), streamH);
         refH = hashCombineU64(refH, hashColumnRef(entries.getLeftRef()));
         refH = hashCombineU64(refH, hashColumnRef(entries.getRightRef()));
         refH = hashCombineU64(refH, hashMlirType(entries.getBetween().getColumn().type));
         defineRefColumn(entries.getBetween(), refH, refStateFor(entries.getLeftRef()));
         return streamH;
      }
      if (auto offset = mlir::dyn_cast<subop::OffsetReferenceBy>(&op)) {
         uint64_t streamH = hashValue(offset.getStream());
         uint64_t refH = hashCombineU64(hashOpName(op), streamH);
         refH = hashCombineU64(refH, hashColumnRef(offset.getRef()));
         refH = hashCombineU64(refH, hashColumnRef(offset.getIdx()));
         refH = hashCombineU64(refH, hashMlirType(offset.getNewRef().getColumn().type));
         defineRefColumn(offset.getNewRef(), refH, refStateFor(offset.getRef()));
         return streamH;
      }
      uint64_t h = 0;
      h = hashCombineU64(h, hashOpName(op));
      h = hashCombineU64(h, hashAttrDictSorted(op));
      h = hashCombineU64(h, hashAttrNormalized(op.getPropertiesAsAttribute()));
      for (auto t : op.getResultTypes()) h = hashCombineU64(h, hashMlirType(t));
      // Operand order is semantic.
      for (auto v : op.getOperands()) h = hashCombineU64(h, hashValue(v));
      // Include nested regions (map/reduce/combine etc.)
      for (auto& r : op.getRegions()) h = hashCombineU64(h, hashRegion(r));
      return h;
   }

   uint64_t hashValue(mlir::Value v) {
      if (auto itRegion = regionValueHashByValue.find(v); itRegion != regionValueHashByValue.end())
         return itRegion->second;
      v = canonicalizeStateValueDeep(v);
      if (auto itRegion = regionValueHashByValue.find(v); itRegion != regionValueHashByValue.end())
         return itRegion->second;
      if (auto it = memo.find(v); it != memo.end()) return it->second;
      // Cycle guard (shouldn't happen in SSA, but ExecutionStep canonicalization can create
      // self-references if we accidentally try to hash the step op itself).
      memo[v] = 0;
      uint64_t h = 0;
      if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(v)) {
         mlir::Value state = canonicalizeStateValueDeep(ba);
         if (isTargetConstructionState(state)) {
            h = hashTargetConstructionState(state.getType());
         } else if (stateHashEnv && (isStateType(state.getType()) || isThreadLocalOfStateType(state.getType())) &&
             !isGetExternalLeafState(state)) {
            h = stateHashEnv->get(state);
         } else {
            h = hashExternalLeaf(state);
         }
      } else if (auto* def = v.getDefiningOp()) {
         if (mlir::isa<subop::ExecutionStepOp>(def)) {
            if (isTargetConstructionState(v)) {
               h = hashTargetConstructionState(v.getType());
            } else if (stateHashEnv && (isStateType(v.getType()) || isThreadLocalOfStateType(v.getType())) &&
                !isGetExternalLeafState(v)) {
               h = stateHashEnv->get(v);
            } else {
               h = hashExternalLeaf(v);
            }
            memo[v] = h;
            return h;
         }
         // Only hash ops inside this execution_step; everything else is treated as an external leaf.
         assert(isWithinStep(def));
         h = hashOp(*def);
      } else {
         assert(0);
      }
      memo[v] = h;
      return h;
   }

   bool operationWritesTargetState(mlir::Operation& op, mlir::Value targetState) {
      mlir::Value targetCanon = canonicalizeStateValueDeep(targetState);
      auto matchesState = [&](mlir::Value state) {
         return state && canonicalizeStateValueDeep(state) == targetCanon;
      };

      if (auto mat = mlir::dyn_cast<subop::MaterializeOp>(&op)) return matchesState(mat.getState());
      if (auto hiv = mlir::dyn_cast<subop::CreateHashIndexedView>(&op)) return matchesState(hiv.getResult());
      if (auto lock = mlir::dyn_cast<subop::LockOp>(&op)) {
         return matchesState(findUpstreamLookupHashIndexedView(lock.getStream()));
      }
      if (auto scatter = mlir::dyn_cast<subop::ScatterOp>(&op)) {
         return matchesState(findUpstreamLookupHashIndexedView(scatter.getStream()));
      }
      if (auto reduce = mlir::dyn_cast<subop::ReduceOp>(&op)) {
         return matchesState(findUpstreamLookupHashIndexedView(reduce.getStream()));
      }
      if (auto from = mlir::dyn_cast<subop::CreateFrom>(&op)) return matchesState(from.getResult());
      if (auto merge = mlir::dyn_cast<subop::MergeOp>(&op)) return matchesState(merge.getResult());

      auto sub = mlir::dyn_cast<subop::SubOperator>(&op);
      if (!sub) return false;
      auto writtenMembers = sub.getWrittenMembers();
      if (writtenMembers.empty()) return false;
      for (mlir::Value operand : op.getOperands()) {
         if (!isStateType(operand.getType()) && !isThreadLocalOfStateType(operand.getType())) continue;
         mlir::Value operandCanon = canonicalizeStateValueDeep(operand);
         if (operandCanon != targetCanon) continue;
         if (intersectsMembers(writtenMembers, getMembersForStateValue(operandCanon))) return true;
      }
      return false;
   }

   uint64_t hashStepWriterRoots() {
      llvm::DenseMap<mlir::Value, RWFlags> rw = analyzeStepStateRWWithNested(step);
      mlir::Value writtenState;
      for (auto& kv : rw) {
         if (!kv.second.write) continue;
         assert(!writtenState && "construction step hash expects at most one written state");
         writtenState = kv.first;
      }
      if (!writtenState) return 0;

      uint64_t h = 0;
      step.walk([&](mlir::Operation* op) {
         if (mlir::isa<subop::ExecutionStepOp>(op)) return;
         if (!operationWritesTargetState(*op, writtenState)) return;
         h = hashCombineU64(h, hashOp(*op));
      });
      return h;
   }

   uint64_t hashStepReturnGraph() {
      auto& block = step.getSubOps().front();
      auto ret = mlir::cast<subop::ExecutionStepReturnOp>(block.getTerminator());
      if (targetState && mlir::isa<subop::ResultTableType>(targetStateType) &&
          ret.getNumOperands() > 0 && isCreateOnlyExecutionStep(step)) {
         bool returnsOnlyTarget = true;
         for (auto v : ret.getInputs()) {
            if (canonicalizeStateValueDeep(v) != canonicalizeStateValueDeep(targetState)) {
               returnsOnlyTarget = false;
               break;
            }
         }
         if (returnsOnlyTarget) {
            uint64_t h = hashString("result_table_create_return");
            std::string typeFp = normalizedSubopStateTypeFingerprint(*memberManager, targetStateType);
            return hashCombineU64(h, hashString(typeFp));
         }
      }
      uint64_t h = 0;
      for (auto v : ret.getInputs()) h = hashCombineU64(h, hashValue(v));
      if (h != 0) return h;

      // Build steps like table scan -> filter/map -> materialize often return no SSA state value;
      // hash only the single-write chain root so upstream filter/map semantics contribute without
      // pulling unrelated step-local ops into construction matching.
      return hashStepWriterRoots();
   }
};

std::string sanitizeMemberBaseName(llvm::StringRef name) {
   return sanitizeBaseName(name);
}

std::string typeFingerprint(mlir::Type t) {
   std::string s;
   llvm::raw_string_ostream ss(s);
   t.print(ss);
   ss.flush();
   return s;
}

std::string fingerprintSortedMemberPairs(subop::MemberManager& mm, llvm::ArrayRef<subop::Member> members) {
   llvm::SmallVector<std::string, 16> parts;
   parts.reserve(members.size());
   for (auto m : members) {
      std::string p = sanitizeMemberBaseName(mm.getName(m));
      p.push_back(':');
      p.append(typeFingerprint(mm.getType(m)));
      parts.push_back(std::move(p));
   }
   llvm::sort(parts);
   return joinSortedStrings(parts);
}

std::string fingerprintMemberTypesMultiset(subop::MemberManager& mm, llvm::ArrayRef<subop::Member> members) {
   llvm::SmallVector<std::string, 16> types;
   types.reserve(members.size());
   for (auto m : members) {
      types.push_back(typeFingerprint(mm.getType(m)));
   }
   llvm::sort(types);
   std::string out;
   for (auto& t : types) {
      if (!out.empty()) out.push_back('|');
      out.append(t);
   }
   return out;
}

std::string fingerprintMemberTypesSequence(subop::MemberManager& mm, llvm::ArrayRef<subop::Member> members) {
   std::string out;
   for (auto m : members) {
      if (!out.empty()) out.push_back('|');
      out.append(typeFingerprint(mm.getType(m)));
   }
   return out;
}

static subop::PreAggrHtFragmentType fragmentTypeForAggregateHt(subop::PreAggrHtType ht) {
   return subop::PreAggrHtFragmentType::get(ht.getContext(), ht.getKeyMembers(), ht.getValueMembers(),
                                            ht.getWithLock());
}

static bool lookupOrInsertTargetsAggregateFragment(subop::LookupOrInsertOp lookup,
                                                   subop::PreAggrHtFragmentType fragTy) {
   auto refTy = mlir::dyn_cast<subop::LookupEntryRefType>(lookup.getRef().getColumn().type);
   return refTy && refTy.getState() == fragTy;
}

static std::string columnRefSemanticFingerprint(
   lingodb::compiler::dialect::tuples::ColumnRefAttr col,
   lingodb::compiler::dialect::tuples::ColumnManager& columnManager) {
   auto [scope, leaf] = columnManager.getName(&col.getColumn());
   std::string out = sanitizeScopeName(scope);
   out.push_back('.');
   out.append(normalizeGeneratedResultMemberName(leaf));
   out.push_back(':');
   out.append(typeFingerprint(col.getColumn().type));
   return out;
}

static std::string aggregateGroupKeyFingerprint(
   mlir::Value aggregateState, const ModuleReuseInfo& reuse, subop::MemberManager& memberManager,
   lingodb::compiler::dialect::tuples::ColumnManager& columnManager) {
   auto ht = mlir::cast<subop::PreAggrHtType>(aggregateState.getType());
   auto fragTy = fragmentTypeForAggregateHt(ht);

   llvm::SmallVector<mlir::Value, 4> buildStates;
   buildStates.push_back(canonicalizeStateValueDeep(aggregateState));
   for (mlir::Value cur = canonicalizeStateValueDeep(aggregateState);;) {
      auto it = reuse.mergedFromShadowState.find(cur);
      if (it == reuse.mergedFromShadowState.end()) break;
      cur = canonicalizeStateValueDeep(it->second);
      buildStates.push_back(cur);
   }

   subop::LookupOrInsertOp lookupOp;
   for (mlir::Value s : buildStates) {
      auto it = reuse.writerStepsByState.find(canonicalizeStateValueDeep(s));
      if (it == reuse.writerStepsByState.end()) continue;
      for (subop::ExecutionStepOp step : it->second) {
         step.walk([&](subop::LookupOrInsertOp lookup) {
            if (!lookupOrInsertTargetsAggregateFragment(lookup, fragTy)) return;
            assert(!lookupOp && "aggregate state must have a single lookup_or_insert construction site");
            lookupOp = lookup;
         });
      }
   }
   assert(lookupOp && "aggregate match requires lookup_or_insert keys");

   std::string out = "optimistic_ht_key{keys=[";
   for (size_t i = 0; i < lookupOp.getKeys().size(); ++i) {
      if (i) out.push_back(',');
      auto col = mlir::cast<lingodb::compiler::dialect::tuples::ColumnRefAttr>(lookupOp.getKeys()[i]);
      out.append(columnRefSemanticFingerprint(col, columnManager));
   }
   out.append("],key_types=");
   out.append(fingerprintMemberTypesMultiset(memberManager, ht.getKeyMembers().getMembers()));
   out.append(",lock=");
   out.append(ht.getWithLock() ? "1" : "0");
   out.push_back('}');
   return out;
}

static std::string aggregateDependencyFingerprint(llvm::ArrayRef<std::string> depTokensSorted) {
   llvm::SmallVector<std::string, 4> deps;
   for (llvm::StringRef d : depTokensSorted) {
      std::string s = d.str();
      size_t mapStart = s.find(";mapping=[");
      if (mapStart != std::string::npos) {
         size_t mapEnd = s.find("];filters=", mapStart);
         assert(mapEnd != std::string::npos && "table dependency token must contain filters after mapping");
         s.erase(mapStart, mapEnd + 1 - mapStart);
      }
      deps.push_back(std::move(s));
   }
   llvm::sort(deps);
   return joinSortedStrings(deps);
}

// Fingerprint SubOp "state-like" types in a way that is stable across separate compilations of the same SQL.
// (MemberManager assigns unique `$<id>` suffixes that differ per MLIRContext/module.)
std::string normalizedSubopStateTypeFingerprint(subop::MemberManager& mm, mlir::Type t) {
   assert(!mlir::isa<subop::HashIndexedViewType>(t) &&
          "hash_indexed_view must use normalizedHashIndexedViewTypeFingerprintForJoinMatch");
   if (auto rt = mlir::dyn_cast<subop::ResultTableType>(t)) {
      // Ignore compiler-chosen member names: only the multiset of member types matters for cross-module matching.
      return std::string("result_table{types=") + fingerprintMemberTypesMultiset(mm, rt.getMembers().getMembers()) + "}";
   }
   if (auto lt = mlir::dyn_cast<subop::LocalTableType>(t)) {
      // Prefer explicit output column names + positional member types (stable across MLIRContexts).
      auto cols = lt.getColumnNames();
      auto mems = lt.getMembers().getMembers();
      assert(cols.size() == mems.size() && "local_table columnNames must align with member list");
      llvm::SmallVector<std::string, 16> pairs;
      pairs.reserve(mems.size());
      for (size_t i = 0; i < mems.size(); i++) {
         auto sattr = mlir::dyn_cast<mlir::StringAttr>(cols[i]);
         assert(sattr && "local_table columnNames must be string attrs");
         std::string p = sattr.getValue().str();
         p.push_back(':');
         p.append(typeFingerprint(mm.getType(mems[i])));
         pairs.push_back(std::move(p));
      }
      llvm::sort(pairs);
      return std::string("local_table{cols_types=") + joinSortedStrings(pairs) + "}";
   }
   if (auto st = mlir::dyn_cast<subop::SimpleStateType>(t)) {
      return std::string("simple_state{members=") + fingerprintSortedMemberPairs(mm, st.getValueMembers().getMembers()) + "}";
   }
   if (auto hm = mlir::dyn_cast<subop::HashMapType>(t)) {
      // Hash tables: normalize by key/value member *types* + lock flag.
      // Member names are compiler-generated and unstable across MLIRContexts.
      return std::string("hashmap{key_types=") + fingerprintMemberTypesMultiset(mm, hm.getKeyMembers().getMembers()) +
             ",val_types=" + fingerprintMemberTypesMultiset(mm, hm.getValueMembers().getMembers()) +
             ",lock=" + (hm.getWithLock() ? "1" : "0") + "}";
   }
   if (auto ht = mlir::dyn_cast<subop::PreAggrHtType>(t)) {
      // Aggregate hash table: normalize by key/value member *types* + lock flag.
      return std::string("optimistic_ht{key_types=") + fingerprintMemberTypesMultiset(mm, ht.getKeyMembers().getMembers()) +
             ",val_types=" + fingerprintMemberTypesMultiset(mm, ht.getValueMembers().getMembers()) +
             ",lock=" + (ht.getWithLock() ? "1" : "0") + "}";
   }
   if (auto frag = mlir::dyn_cast<subop::PreAggrHtFragmentType>(t)) {
      // Aggregate table fragment: same normalization as the global table.
      return std::string("optimistic_ht_fragment{key_types=") + fingerprintMemberTypesMultiset(mm, frag.getKeyMembers().getMembers()) +
             ",val_types=" + fingerprintMemberTypesMultiset(mm, frag.getValueMembers().getMembers()) +
             ",lock=" + (frag.getWithLock() ? "1" : "0") + "}";
   }
   if (auto tbl = mlir::dyn_cast<subop::TableType>(t)) {
      return std::string("table{members=") + fingerprintSortedMemberPairs(mm, tbl.getMembers().getMembers()) +
             ",filtered=" + (tbl.getFiltered() ? "1" : "0") + "}";
   }
   if (auto ehi = mlir::dyn_cast<subop::ExternalHashIndexType>(t)) {
      return std::string("externalhashindex{key_types=") +
             fingerprintMemberTypesMultiset(mm, ehi.getKeyMembers().getMembers()) +
             ",val_types=" + fingerprintMemberTypesMultiset(mm, ehi.getValueMembers().getMembers()) + "}";
   }
   if (auto heap = mlir::dyn_cast<subop::HeapType>(t)) {
      return std::string("heap{members=") + fingerprintSortedMemberPairs(mm, heap.getMembers().getMembers()) +
             ",max=" + std::to_string(heap.getMaxElements()) + "}";
   }
   std::string ty = typeFingerprint(t);
   llvm_unreachable((std::string("unsupported state type for cross-query reuse matching: ") + ty).c_str());
}

llvm::DenseMap<mlir::Value, std::string> buildTableDescrByTableState(mlir::ModuleOp moduleOp) {
   // Maps every `get_external` execution_step result (table, externalhashindex, …) to normalized descr hex.
   llvm::DenseMap<mlir::Value, std::string> tableDescr;
   moduleOp.walk([&](subop::ExecutionStepOp step) {
      if (!isExternalTableRefStep(step)) return;
      auto& block = step.getSubOps().front();
      subop::GetExternalOp geOp;
      bool found = false;
      for (auto& op : block.without_terminator()) {
         if (auto g = mlir::dyn_cast<subop::GetExternalOp>(&op)) {
            geOp = g;
            found = true;
            break;
         }
      }
      assert(found && "table_ref step must contain get_external");
      assert(step.getNumResults() == 1 && "table_ref step must return one value");
      mlir::Value t = canonicalizeStateValueDeep(step.getResult(0));
      tableDescr[t] = normalizeExternalDataSourceDescrHex(geOp.getDescr());
   });
   return tableDescr;
}

struct StateMatchProfile {
   int queryId = -1;
   mlir::Value value;
   bool eligible = false;
   uint64_t stateHash = 0;
   llvm::SmallVector<std::string, 8> depTokensSorted;
   llvm::SmallVector<uint64_t, 8> constructionStepHashes;
   llvm::SmallVector<std::pair<int, uint64_t>, 8> constructionStepHashByIndex;
   uint64_t constructionHash = 0;
   std::string typeFingerprintStr;
   /// Full HIV/buffer stored-value column layout (excluded from `constructionHash` / match type key).
   std::string storedValueMembersFingerprint;
   /// Aggregate hash-table group-key identity. Payloads are intentionally excluded from aggregate matching.
   std::string aggregateGroupKeyFingerprint;
   /// A scan_refs(table) -> ... -> map(predicate) -> filter residual table filter was observed
   /// in the HIV construction closure. Used to keep relaxed HIV fallback matching filter-specific.
   bool hasResidualTableFilter = false;
   bool hasComplexResidualTableFilter = false;
   bool hasUnsupportedResidualTableFilter = false;
};

/// Phase 1 output: per top-level-step state read/write (nested bodies merged into parent).
struct ModuleStepRwAnalysis {
   llvm::DenseMap<int, llvm::DenseMap<mlir::Value, RWFlags>> rwByStep;
};

/// Phase 4 output: construction / type fingerprints (only computed for eligible states).
struct StateConstructionMatchHashes {
   llvm::SmallVector<uint64_t, 8> constructionStepHashes;
   llvm::SmallVector<std::pair<int, uint64_t>, 8> constructionStepHashByIndex;
   uint64_t constructionHash = 0;
   std::string typeFingerprintStr;
   std::string storedValueMembersFingerprint;
};

static bool streamValueHasOnlyUseByInStep(mlir::Value v, mlir::Operation* expectedUser,
                                          subop::ExecutionStepOp step) {
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

static subop::ScanRefsOp traceSingleUseStreamToTableScanForProfile(subop::ExecutionStepOp step,
                                                                   mlir::Operation* user,
                                                                   mlir::Value stream) {
   for (;;) {
      mlir::Operation* def = stream.getDefiningOp();
      if (!def) return {};
      if (!streamValueHasOnlyUseByInStep(stream, user, step)) return {};
      if (auto scan = mlir::dyn_cast<subop::ScanRefsOp>(def)) {
         if (mlir::isa<subop::TableType>(scan.getState().getType())) return scan;
         return {};
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

static bool stepHasResidualTableFilterForProfile(subop::ExecutionStepOp step, bool requireSupported) {
   bool found = false;
   step.walk([&](subop::FilterOp filter) {
      if (found) return;
      auto map = mlir::dyn_cast_or_null<subop::MapOp>(filter.getStream().getDefiningOp());
      if (!map) return;
      if (!streamValueHasOnlyUseByInStep(map.getResult(), filter.getOperation(), step)) return;
      llvm::DenseSet<const void*> computedCols;
      for (auto attr : map.getComputedCols()) {
         auto def = mlir::cast<tuples::ColumnDefAttr>(attr);
         computedCols.insert(&def.getColumn());
      }
      for (auto attr : filter.getConditions()) {
         auto ref = mlir::cast<tuples::ColumnRefAttr>(attr);
         if (!computedCols.contains(&ref.getColumn())) return;
      }
      if (requireSupported && !residualPredicateMapSupportedForRelaxedHiv(map, filter)) return;
      found = static_cast<bool>(traceSingleUseStreamToTableScanForProfile(step, map.getOperation(), map.getStream()));
   });
   return found;
}

static bool constructionHasResidualTableFilter(llvm::ArrayRef<int> constructionStepIndices,
                                               const llvm::DenseMap<int, subop::ExecutionStepOp>& stepByIndex,
                                               bool requireSupported) {
   for (int si : constructionStepIndices) {
      auto itS = stepByIndex.find(si);
      assert(itS != stepByIndex.end());
      if (stepHasResidualTableFilterForProfile(itS->second, requireSupported)) return true;
   }
   return false;
}

static bool constructionHasComplexResidualTableFilter(
   llvm::ArrayRef<int> constructionStepIndices,
   const llvm::DenseMap<int, subop::ExecutionStepOp>& stepByIndex) {
   for (int si : constructionStepIndices) {
      auto itS = stepByIndex.find(si);
      assert(itS != stepByIndex.end());
      subop::ExecutionStepOp step = itS->second;
      bool found = false;
      step.walk([&](subop::FilterOp filter) {
         if (found) return;
         auto map = mlir::dyn_cast_or_null<subop::MapOp>(filter.getStream().getDefiningOp());
         if (!map) return;
         if (!streamValueHasOnlyUseByInStep(map.getResult(), filter.getOperation(), step)) return;
         llvm::DenseSet<const void*> computedCols;
         for (auto attr : map.getComputedCols()) {
            auto def = mlir::cast<tuples::ColumnDefAttr>(attr);
            computedCols.insert(&def.getColumn());
         }
         for (auto attr : filter.getConditions()) {
            auto ref = mlir::cast<tuples::ColumnRefAttr>(attr);
            if (!computedCols.contains(&ref.getColumn())) return;
         }
         if (!residualPredicateMapSupportedForRelaxedHiv(map, filter)) return;
         if (!residualPredicateMapComplexForRelaxedHiv(map, filter)) return;
         found = static_cast<bool>(
            traceSingleUseStreamToTableScanForProfile(step, map.getOperation(), map.getStream()));
      });
      if (found) return true;
   }
   return false;
}

struct ModuleMatchAndReuseAnalysis {
   struct StateInfo {
      int createdAt = -1;
      llvm::SmallSet<int, 16> reads;
      llvm::SmallSet<int, 16> writes;
   };

   llvm::DenseMap<mlir::Value, StateInfo> stateInfo;
   llvm::DenseMap<int, subop::ExecutionStepOp> stepByIndex;
   llvm::DenseMap<int, llvm::DenseMap<mlir::Value, RWFlags>> rwByStep;
   llvm::DenseMap<mlir::Value, int> createdAtByState;
   llvm::DenseMap<mlir::Value, mlir::Value> mergedFromShadowState;
   llvm::DenseSet<mlir::Value> transparentStates;
   llvm::DenseMap<mlir::Value, llvm::SmallSet<int, 16>> writesByState;
   llvm::SmallVector<mlir::Value, 64> statesSorted;
   ModuleReuseInfo reuse;
};

/// Register execution steps and state metadata (creation site, merge shadow, create-only) before RW analysis.
static void registerModuleExecutionSteps(mlir::ModuleOp moduleOp, ModuleMatchAndReuseAnalysis& a) {
   auto recordState = [&](mlir::Value v) -> ModuleMatchAndReuseAnalysis::StateInfo& { return a.stateInfo[v]; };

   size_t idx = 0;
   walkTopLevelExecutionSteps(moduleOp, [&](subop::ExecutionStepOp step) {
      int stepIdx = static_cast<int>(idx++);
      a.stepByIndex[stepIdx] = step;

      bool tableRef = isExternalTableRefStep(step);

      if (tableRef) {
         subop::GetExternalOp geOp;
         step.walk([&](subop::GetExternalOp g) {
            geOp = g;
         });
         assert(geOp && "table_ref step must contain get_external");
         assert(step.getNumResults() == 1 && "table_ref step must return one value");
         mlir::Value t = canonicalizeStateValueDeep(step.getResult(0));
         a.reuse.externalDatasourceByTableState[t] =
            lingodb::utility::deserializeFromHexString<lingodb::runtime::ExternalDatasourceProperty>(geOp.getDescr());
      }

      for (auto r : step.getResults()) {
         if (!isStateType(r.getType()) && !isThreadLocalOfStateType(r.getType())) continue;
         mlir::Value key = canonicalizeStateValueDeep(r);
         auto& info = recordState(key);
         if (info.createdAt < 0) info.createdAt = stepIdx;
         a.createdAtByState[key] = info.createdAt;
         if (tableRef || executionStepReturnsStateValue(step)) info.writes.insert(stepIdx);
      }

      step.walk([&](subop::MergeOp merge) {
         mlir::Value in = canonicalizeStateValueDeep(merge.getThreadLocal());
         mlir::Value out = canonicalizeStateValueDeep(merge.getResult());
         a.mergedFromShadowState[out] = in;
         a.reuse.mergedFromShadowState[out] = in;
      });

      step.walk([&](mlir::Operation* op) {
         for (auto r : op->getResults()) {
            auto t = r.getType();
            if (!isStateType(t) && !isThreadLocalOfStateType(t)) continue;
            auto& info = recordState(r);
            if (info.createdAt < 0) info.createdAt = stepIdx;
            a.createdAtByState[r] = info.createdAt;
         }
      });

      if (isCreateOnlyExecutionStep(step)) {
         llvm::SmallVector<mlir::Value, 4> keys;
         for (mlir::Value r : step.getResults()) if (r) keys.push_back(r);
         auto& block = step.getSubOps().front();
         if (auto ret = mlir::dyn_cast<subop::ExecutionStepReturnOp>(block.getTerminator())) {
            for (mlir::Value o : ret.getOperands()) if (o) keys.push_back(o);
         }
         for (mlir::Value r : keys) {
            mlir::Value key = canonicalizeStateValueDeep(r);
            auto [it, inserted] = a.reuse.createOnlyStepForState.try_emplace(key, step);
            assert(inserted || it->second == step);
         }
      }
   });
}

/// Phase 1: analyze each execution_step body RW; record per-state reads/writes and `reuse.steps`.
static ModuleStepRwAnalysis analyzeModuleExecutionStepRw(ModuleMatchAndReuseAnalysis& a) {
   ModuleStepRwAnalysis rw;
   auto recordState = [&](mlir::Value v) -> ModuleMatchAndReuseAnalysis::StateInfo& { return a.stateInfo[v]; };

   llvm::SmallVector<int, 128> stepOrder;
   stepOrder.reserve(a.stepByIndex.size());
   for (auto& it : a.stepByIndex) stepOrder.push_back(it.first);
   llvm::sort(stepOrder);

   for (int stepIdx : stepOrder) {
      subop::ExecutionStepOp step = a.stepByIndex[stepIdx];
      llvm::DenseMap<mlir::Value, RWFlags> stepRw = analyzeStepStateRWWithNested(step);
      rw.rwByStep[stepIdx] = stepRw;
      a.rwByStep[stepIdx] = stepRw;

      ModuleReuseInfo::StepRW re;
      re.step = step;
      for (auto& it : stepRw) {
         auto& info = recordState(it.first);
         if (info.createdAt < 0 && isFuncBlockArgument(it.first)) {
            info.createdAt = -2;
            a.createdAtByState[it.first] = info.createdAt;
         }
         if (it.second.read) {
            info.reads.insert(stepIdx);
            re.reads.push_back(it.first);
         }
         if (it.second.write) {
            info.writes.insert(stepIdx);
            re.writes.push_back(it.first);
            a.reuse.writerStepsByState[it.first].push_back(step);
         }
      }
      a.reuse.steps.push_back(std::move(re));
   }

   a.rwByStep = rw.rwByStep;
   a.transparentStates = computeTransparentStatesFromRw(a.rwByStep);
   return rw;
}

static void finalizeModuleMatchAnalysis(mlir::ModuleOp moduleOp, ModuleMatchAndReuseAnalysis& a) {
   llvm::DenseMap<mlir::Value, ModuleMatchAndReuseAnalysis::StateInfo> collapsed;
   for (auto& it : a.stateInfo) {
      mlir::Value key = canonicalizeStateValueDeep(it.first);
      auto& dst = collapsed[key];
      if (dst.createdAt < 0 || (it.second.createdAt >= 0 && it.second.createdAt < dst.createdAt)) {
         dst.createdAt = it.second.createdAt;
      }
      dst.reads.insert(it.second.reads.begin(), it.second.reads.end());
      dst.writes.insert(it.second.writes.begin(), it.second.writes.end());
   }
   a.stateInfo = std::move(collapsed);
   a.createdAtByState.clear();
   for (auto& it : a.stateInfo) {
      if (it.second.createdAt >= 0) a.createdAtByState[it.first] = it.second.createdAt;
   }

   llvm::DenseMap<mlir::Value, llvm::SmallVector<subop::ExecutionStepOp, 8>> collapsedWriters;
   for (auto& kv : a.reuse.writerStepsByState) {
      auto& dst = collapsedWriters[canonicalizeStateValueDeep(kv.first)];
      for (subop::ExecutionStepOp s : kv.second) dst.push_back(s);
   }
   a.reuse.writerStepsByState = std::move(collapsedWriters);

   llvm::DenseMap<mlir::Value, subop::ExecutionStepOp> collapsedCreateOnly;
   for (auto& kv : a.reuse.createOnlyStepForState) {
      collapsedCreateOnly[canonicalizeStateValueDeep(kv.first)] = kv.second;
   }
   a.reuse.createOnlyStepForState = std::move(collapsedCreateOnly);

   llvm::DenseMap<mlir::Value, mlir::Value> collapsedShadow;
   for (auto& kv : a.mergedFromShadowState) {
      collapsedShadow[canonicalizeStateValueDeep(kv.first)] = canonicalizeStateValueDeep(kv.second);
   }
   a.mergedFromShadowState = std::move(collapsedShadow);
   a.reuse.mergedFromShadowState = a.mergedFromShadowState;

   for (auto& it : a.stateInfo) {
      a.writesByState[it.first] = it.second.writes;
      a.statesSorted.push_back(it.first);
   }
   llvm::sort(a.statesSorted, [&](mlir::Value x, mlir::Value y) {
      int cx = a.stateInfo[x].createdAt;
      int cy = a.stateInfo[y].createdAt;
      if (cx != cy) return cx < cy;
      return x.getAsOpaquePointer() < y.getAsOpaquePointer();
   });

   moduleOp.walk([&](subop::CreateHashIndexedView chiv) {
      mlir::Value globalBuf = canonicalizeStateValueDeep(chiv.getSource());
      mlir::Value hiv = canonicalizeStateValueDeep(chiv.getResult());
      assert(mlir::isa<subop::BufferType>(globalBuf.getType()));
      assert((mlir::isa<subop::HashIndexedViewType, subop::MixedHashIndexedViewType>(hiv.getType())));
      if (mlir::Value existing = hashIndexedViewShadowingBuffer(globalBuf, a.reuse)) {
         assert(existing == hiv && "each join buffer must feed at most one hash_indexed_view in a serial chain");
      }
      a.mergedFromShadowState[hiv] = globalBuf;
      a.reuse.mergedFromShadowState[hiv] = globalBuf;
   });
}

static ModuleMatchAndReuseAnalysis analyzeModuleForMatchAndReuse(mlir::ModuleOp moduleOp) {
   ModuleMatchAndReuseAnalysis a;
   registerModuleExecutionSteps(moduleOp, a);
   analyzeModuleExecutionStepRw(a);
   finalizeModuleMatchAnalysis(moduleOp, a);
   return a;
}

static StateHashEnv buildStateHashEnv(const ModuleMatchAndReuseAnalysis& module, const StateDependencyGraph& depGraph,
                                      const llvm::DenseMap<mlir::Value, std::string>& tableDescrByTableState) {
   StateHashEnv env;
   env.transparentStates = &module.transparentStates;
   env.depGraph = &depGraph;
   env.tableDescrByTableState = &tableDescrByTableState;
   unsigned ordinal = 0;
   for (mlir::Value state : module.statesSorted) {
      state = canonicalizeStateValueDeep(state);
      if (module.transparentStates.contains(state)) continue;
      if (auto cacheKey = getCacheGetKeyForState(state)) {
         env.hashByState[state] = *cacheKey;
         continue;
      }
      if (auto it = tableDescrByTableState.find(state); it != tableDescrByTableState.end()) {
         uint64_t h = static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(it->second)));
         h = hashCombineU64(h, hashType(state.getType()));
         env.hashByState[state] = h;
         continue;
      }
      env.hashByState[state] = provisionalStateHash(state, ordinal++);
   }
   return env;
}

static void assertTableStatesHaveNoStateDependencies(const ModuleMatchAndReuseAnalysis& module,
                                                     const StateDependencyGraph& depGraph) {
   for (mlir::Value state : module.statesSorted) {
      if (!mlir::isa<subop::TableType>(state.getType())) continue;
      assert(directPredecessorsInDepGraph(state, depGraph).empty() &&
             "table states are special match-only leaves and must not depend on other states");
   }
}

static llvm::SmallVector<mlir::Value, 64> collectReuseCandidateStates(mlir::ModuleOp moduleOp,
                                                                    const ModuleMatchAndReuseAnalysis& a) {
   llvm::SmallVector<mlir::Value, 64> candidates;
   subop::ExecutionGroupOp group = getMainExecutionGroup(moduleOp);
   for (mlir::Operation& op : group.getSubOps().front()) {
      auto step = mlir::dyn_cast<subop::ExecutionStepOp>(&op);
      if (!step) continue;
      assert(isTopLevelExecutionStep(step) && "execution_group body must contain only top-level steps");
      for (mlir::Value r : step.getResults()) {
         if (!isStateType(r.getType()) && !isThreadLocalOfStateType(r.getType())) continue;
         mlir::Value key = canonicalizeStateValueDeep(r);
         auto itCreated = a.createdAtByState.find(key);
         assert(itCreated != a.createdAtByState.end() && "execution_step state result must be tracked");
         if (itCreated->second < 0) continue;
         // Special states are only folded into dependency equality; they are never reused directly.
         if (isSpecialNonReuseStateValue(key, a.transparentStates)) continue;
         if (getCacheGetKeyForState(key)) continue;
         // Carrier states that are not transparent are also not concrete reuse targets.
         if (isTransparentDepCarrierType(key.getType())) continue;
         candidates.push_back(key);
      }
   }
   llvm::sort(candidates, [&](mlir::Value x, mlir::Value y) {
      int cx = a.createdAtByState.lookup(x);
      int cy = a.createdAtByState.lookup(y);
      if (cx != cy) return cx < cy;
      return x.getAsOpaquePointer() < y.getAsOpaquePointer();
   });
   return candidates;
}

static bool constructionStepHasMultipleWrites(int stepIdx, const ModuleStepRwAnalysis& rw) {
   auto itRW = rw.rwByStep.find(stepIdx);
   assert(itRW != rw.rwByStep.end());
   unsigned writeCount = 0;
   for (auto& kv : itRW->second) {
      if (kv.second.write) writeCount++;
      if (writeCount > 1) return true;
   }
   return false;
}

/// Phase 3: dependency-graph eligibility plus single-write construction constraint.
static bool determineStateReuseEligibility(
   mlir::Value state, const StateDependencyGraph& depGraph,
   llvm::ArrayRef<int> constructionStepIndices, const ModuleStepRwAnalysis& stepRw,
   const ModuleMatchAndReuseAnalysis& module,
   const llvm::DenseMap<mlir::Value, std::string>& tableDescrByTableState,
   StateDepEligibility& outDep) {
   mlir::Value stateCanon = canonicalizeStateValueDeep(state);
   assert(!isSpecialNonReuseStateValue(stateCanon, module.transparentStates) && "special states are never reuse targets");
   assert(!isTransparentDepCarrierType(stateCanon.getType()) && "carrier states are never reuse match targets");
   assert(!module.writesByState.lookup(stateCanon).empty() &&
          "reuse candidate must be constructed in at least one execution_step");

   outDep = evaluateStateDepEligibility(stateCanon, depGraph, module.transparentStates, tableDescrByTableState);
   if (!outDep.eligible) return false;

   for (int si : constructionStepIndices) {
      if (constructionStepHasMultipleWrites(si, stepRw)) return false;
   }
   return true;
}

/// Phase 4: construction hash and type fingerprint (eligible states only).
static StateConstructionMatchHashes computeEligibleStateMatchHashes(
   mlir::Value state, llvm::ArrayRef<int> constructionStepIndices, const ModuleMatchAndReuseAnalysis& module,
   const llvm::DenseMap<mlir::Value, std::string>& tableDescrByTableState, const StateHashEnv& stateHashEnv,
   subop::MemberManager& memberManager,
   lingodb::compiler::dialect::tuples::ColumnManager& columnManager) {
   const bool isHiv = mlir::isa<subop::HashIndexedViewType>(state.getType());
   std::optional<JoinHivMatchDetails> joinHivDetails;
   if (isHiv) {
      joinHivDetails.emplace(computeJoinHivMatchDetails(
         state, mlir::cast<subop::HashIndexedViewType>(state.getType()), memberManager, columnManager,
         module.reuse.writerStepsByState, constructionStepIndices, module.stepByIndex));
   }

   StateConstructionMatchHashes hashes;
   llvm::SmallVector<uint64_t, 16> stepHashes;
   stepHashes.reserve(constructionStepIndices.size());
   llvm::DenseSet<mlir::Value> targetConstructionStates;
   targetConstructionStates.insert(canonicalizeStateValueDeep(state));
   forEachShadowChainPredecessorValue(canonicalizeStateValueDeep(state), module.mergedFromShadowState,
                                      [&](mlir::Value shadow) {
                                         targetConstructionStates.insert(canonicalizeStateValueDeep(shadow));
                                      });
   for (int si : constructionStepIndices) {
      auto itS = module.stepByIndex.find(si);
      assert(itS != module.stepByIndex.end());
      StepDagHasher hasher;
      hasher.step = itS->second;
      hasher.targetState = state;
      hasher.targetStateType = state.getType();
      hasher.tableDescrByTableState = &tableDescrByTableState;
      hasher.memberManager = &memberManager;
      hasher.columnManager = &columnManager;
      hasher.externalDatasourceByTableState = &module.reuse.externalDatasourceByTableState;
      hasher.stateHashEnv = &stateHashEnv;
      hasher.targetConstructionStates = &targetConstructionStates;
      if (joinHivDetails) {
         hasher.joinIndexMemberNamesSanitized = &joinHivDetails->indexMemberNamesSanitized;
         hasher.joinKeyColumnAttrHashes = &joinHivDetails->joinKeyColumnAttrHashes;
         hasher.joinKeyColumnIdentifiersSanitized = &joinHivDetails->joinKeyColumnIdentifiersSanitized;
      }
      uint64_t stepHash = hasher.hashStepReturnGraph();
      stepHashes.push_back(stepHash);
      hashes.constructionStepHashByIndex.push_back({si, stepHash});
   }
   llvm::sort(stepHashes);

   hashes.constructionStepHashes.assign(stepHashes.begin(), stepHashes.end());
   for (auto h : stepHashes) hashes.constructionHash = hashCombineU64(hashes.constructionHash, h);

   if (joinHivDetails) {
      hashes.typeFingerprintStr = normalizedHashIndexedViewTypeFingerprintForJoinMatch(
         memberManager, mlir::cast<subop::HashIndexedViewType>(state.getType()), *joinHivDetails);
      hashes.storedValueMembersFingerprint = joinHivDetails->storedValueMembersFingerprint;
   } else {
      assert(!isHiv);
      hashes.typeFingerprintStr = normalizedSubopStateTypeFingerprint(memberManager, state.getType());
   }
   return hashes;
}

llvm::SmallVector<StateMatchProfile, 128> buildStateMatchProfiles(
   int queryId,
   mlir::ModuleOp moduleOp,
   const llvm::DenseMap<mlir::Value, std::string>& tableDescrByTableState) {
   auto* dialect = moduleOp.getContext()->getLoadedDialect<subop::SubOperatorDialect>();
   assert(dialect && "subop dialect must be loaded to fingerprint members");
   subop::MemberManager& memberManager = dialect->getMemberManager();
   auto* tupDialect = moduleOp.getContext()->getLoadedDialect<lingodb::compiler::dialect::tuples::TupleStreamDialect>();
   assert(tupDialect && "tuples dialect must be loaded");

   ModuleMatchAndReuseAnalysis module = analyzeModuleForMatchAndReuse(moduleOp);

   ModuleStepRwAnalysis stepRw{module.rwByStep};

   // Phase 2: same-step RW → state dependency DAG.
   StateDependencyGraph depGraph = buildStateDependencyGraphFromStepRw(stepRw.rwByStep);
   StateHashEnv stateHashEnv = buildStateHashEnv(module, depGraph, tableDescrByTableState);
   assertTableStatesHaveNoStateDependencies(module, depGraph);

   llvm::SmallVector<mlir::Value, 64> candidates = collectReuseCandidateStates(moduleOp, module);

   llvm::SmallVector<StateMatchProfile, 128> profiles;
   profiles.reserve(candidates.size());

   for (mlir::Value state : candidates) {
      auto constructionStepIndices = getConstructionStepIndicesForState(
         canonicalizeStateValueDeep(state), module.createdAtByState, module.writesByState,
         module.mergedFromShadowState);

      StateDepEligibility depElig;
      bool eligible = determineStateReuseEligibility(
         state, depGraph, constructionStepIndices, stepRw, module, tableDescrByTableState, depElig);

      StateMatchProfile prof;
      prof.queryId = queryId;
      prof.value = state;
      prof.eligible = eligible;
      prof.stateHash = stateHashEnv.get(state);
      prof.depTokensSorted.assign(depElig.depTokensSorted.begin(), depElig.depTokensSorted.end());

      if (prof.eligible) {
         if (mlir::isa<subop::HashIndexedViewType>(state.getType())) {
            JoinHivMatchDetails joinHivDetails = computeJoinHivMatchDetails(
               state, mlir::cast<subop::HashIndexedViewType>(state.getType()), memberManager,
               tupDialect->getColumnManager(), module.reuse.writerStepsByState, constructionStepIndices,
               module.stepByIndex);
            relaxJoinHivDepTokensInProfile(prof.depTokensSorted, joinHivDetails,
                                           module.reuse.externalDatasourceByTableState, tableDescrByTableState);
            bool hasAnyResidualFilter =
               constructionHasResidualTableFilter(constructionStepIndices, module.stepByIndex,
                                                  /*requireSupported*/ false);
            prof.hasResidualTableFilter =
               constructionHasResidualTableFilter(constructionStepIndices, module.stepByIndex,
                                                  /*requireSupported*/ true);
            prof.hasComplexResidualTableFilter =
               constructionHasComplexResidualTableFilter(constructionStepIndices, module.stepByIndex);
            prof.hasUnsupportedResidualTableFilter = hasAnyResidualFilter && !prof.hasResidualTableFilter;
         }
         StateConstructionMatchHashes hashes = computeEligibleStateMatchHashes(
            state, constructionStepIndices, module, tableDescrByTableState, stateHashEnv, memberManager,
            tupDialect->getColumnManager());
         prof.constructionStepHashes.assign(hashes.constructionStepHashes.begin(), hashes.constructionStepHashes.end());
         prof.constructionStepHashByIndex.assign(hashes.constructionStepHashByIndex.begin(),
                                                 hashes.constructionStepHashByIndex.end());
         prof.constructionHash = hashes.constructionHash;
         prof.typeFingerprintStr = std::move(hashes.typeFingerprintStr);
         prof.storedValueMembersFingerprint = std::move(hashes.storedValueMembersFingerprint);
         if (mlir::isa<subop::HashIndexedViewType>(state.getType())) {
            assert(!prof.storedValueMembersFingerprint.empty());
            module.reuse.joinBuildStoredValueMembersByState[state] = prof.storedValueMembersFingerprint;
         } else if (mlir::isa<subop::PreAggrHtType>(state.getType())) {
            prof.aggregateGroupKeyFingerprint = aggregateGroupKeyFingerprint(
               state, module.reuse, memberManager, tupDialect->getColumnManager());
         }
      }

      profiles.push_back(std::move(prof));
   }

   return profiles;
}

static llvm::DenseMap<mlir::Operation*, const ModuleReuseInfo::StepRW*>
buildReuseRwByStepOpMap(const ModuleReuseInfo& reuse) {
   llvm::DenseMap<mlir::Operation*, const ModuleReuseInfo::StepRW*> byOp;
   for (const auto& e : reuse.steps) {
      byOp[const_cast<subop::ExecutionStepOp&>(e.step).getOperation()] = &e;
   }
   return byOp;
}

static llvm::SmallVector<lingodb::runtime::FilterDescription, 8>
decodeSimpleMatchFiltersForState(
   mlir::Value state, const ModuleReuseInfo& reuse,
   const llvm::DenseMap<mlir::Operation*, const ModuleReuseInfo::StepRW*>& rwByStepOp) {
   llvm::SmallVector<lingodb::runtime::FilterDescription, 8> decoded;
   auto itW = reuse.writerStepsByState.find(canonicalizeStateValueForReuse(state));
   if (itW == reuse.writerStepsByState.end()) itW = reuse.writerStepsByState.find(state);
   if (itW == reuse.writerStepsByState.end()) return decoded;
   for (ExecutionStepOp ws : itW->second) {
      const ModuleReuseInfo::StepRW* rw = rwByStepOp.lookup(ws.getOperation());
      if (!rw) continue;
      for (mlir::Value r : rw->reads) {
         if (!mlir::isa<subop::TableType>(r.getType())) continue;
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
         for (const auto& f : it->second.filterDescriptions) decoded.push_back(f);
      }
   }
   return decoded;
}

static llvm::SmallVector<lingodb::runtime::FilterDescription, 8>
decodeSimpleMatchFiltersAlongShadowChain(mlir::Value state, const ModuleReuseInfo& reuse) {
   llvm::DenseMap<mlir::Operation*, const ModuleReuseInfo::StepRW*> rwByStepOp =
      buildReuseRwByStepOpMap(reuse);
   llvm::SmallVector<lingodb::runtime::FilterDescription, 8> filters =
      decodeSimpleMatchFiltersForState(state, reuse, rwByStepOp);
   if (!filters.empty()) return filters;
   forEachShadowChainPredecessorValue(canonicalizeStateValueForReuse(state), reuse.mergedFromShadowState,
                                      [&](mlir::Value shadow) {
                                         if (!filters.empty()) return;
                                         filters = decodeSimpleMatchFiltersForState(shadow, reuse, rwByStepOp);
                                      });
   return filters;
}

struct MatchNumericFilterRange {
   bool hasLower = false;
   double lower = 0.0;
   bool lowerInclusive = true;
   bool hasUpper = false;
   double upper = 0.0;
   bool upperInclusive = true;
   std::optional<double> eq;
};

static std::optional<double> getMatchNumericFilterValue(const lingodb::runtime::FilterDescription& f) {
   if (const auto* v = std::get_if<int64_t>(&f.value)) return static_cast<double>(*v);
   if (const auto* v = std::get_if<double>(&f.value)) return *v;
   return std::nullopt;
}

static bool isMatchRangeFilterOp(lingodb::runtime::FilterOp op) {
   switch (op) {
      case lingodb::runtime::FilterOp::EQ:
      case lingodb::runtime::FilterOp::LT:
      case lingodb::runtime::FilterOp::LTE:
      case lingodb::runtime::FilterOp::GT:
      case lingodb::runtime::FilterOp::GTE:
         return true;
      default:
         return false;
   }
}

static void addMatchRangeConstraint(MatchNumericFilterRange& range, lingodb::runtime::FilterOp op, double value) {
   switch (op) {
      case lingodb::runtime::FilterOp::EQ:
         range.eq = value;
         break;
      case lingodb::runtime::FilterOp::GT:
      case lingodb::runtime::FilterOp::GTE: {
         bool inclusive = op == lingodb::runtime::FilterOp::GTE;
         if (!range.hasLower || value > range.lower) {
            range.hasLower = true;
            range.lower = value;
            range.lowerInclusive = inclusive;
         } else if (value == range.lower) {
            range.lowerInclusive = range.lowerInclusive && inclusive;
         }
         break;
      }
      case lingodb::runtime::FilterOp::LT:
      case lingodb::runtime::FilterOp::LTE: {
         bool inclusive = op == lingodb::runtime::FilterOp::LTE;
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

static bool matchRangeAllowsValue(const MatchNumericFilterRange& range, double value) {
   if (range.hasLower && (value < range.lower || (value == range.lower && !range.lowerInclusive))) return false;
   if (range.hasUpper && (value > range.upper || (value == range.upper && !range.upperInclusive))) return false;
   return true;
}

static bool matchRangeIsEmpty(const MatchNumericFilterRange& range) {
   if (range.eq && !matchRangeAllowsValue(range, *range.eq)) return true;
   if (!range.hasLower || !range.hasUpper) return false;
   if (range.lower > range.upper) return true;
   return range.lower == range.upper && !(range.lowerInclusive && range.upperInclusive);
}

static bool matchRangesAreDisjoint(const MatchNumericFilterRange& a, const MatchNumericFilterRange& b) {
   if (matchRangeIsEmpty(a) || matchRangeIsEmpty(b)) return true;
   if (a.eq && b.eq) return *a.eq != *b.eq;
   if (a.eq) return !matchRangeAllowsValue(b, *a.eq);
   if (b.eq) return !matchRangeAllowsValue(a, *b.eq);
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

static bool matchFiltersDefinitelyDisjoint(
   llvm::ArrayRef<lingodb::runtime::FilterDescription> filtersA,
   llvm::ArrayRef<lingodb::runtime::FilterDescription> filtersB) {
   for (const auto& a : filtersA) {
      if (a.op != lingodb::runtime::FilterOp::EQ) continue;
      const auto* strA = std::get_if<std::string>(&a.value);
      if (!strA) continue;
      for (const auto& b : filtersB) {
         if (b.op != lingodb::runtime::FilterOp::EQ) continue;
         if (a.columnName != b.columnName) continue;
         const auto* strB = std::get_if<std::string>(&b.value);
         if (!strB) continue;
         if (*strA != *strB) return true;
      }
   }

   llvm::StringMap<MatchNumericFilterRange> rangesA;
   llvm::StringMap<MatchNumericFilterRange> rangesB;
   auto addAll = [](llvm::StringMap<MatchNumericFilterRange>& ranges,
                    llvm::ArrayRef<lingodb::runtime::FilterDescription> filters) {
      for (const auto& f : filters) {
         if (!isMatchRangeFilterOp(f.op)) continue;
         std::optional<double> value = getMatchNumericFilterValue(f);
         if (!value) continue;
         addMatchRangeConstraint(ranges[f.columnName], f.op, *value);
      }
   };
   addAll(rangesA, filtersA);
   addAll(rangesB, filtersB);
   for (auto& a : rangesA) {
      auto b = rangesB.find(a.getKey());
      if (b == rangesB.end()) continue;
      if (matchRangesAreDisjoint(a.getValue(), b->getValue())) return true;
   }
   return false;
}

static bool profilesDefinitelyDisjointByFilters(
   const StateMatchProfile& a, const ModuleReuseInfo& reuseA,
   const StateMatchProfile& b, const ModuleReuseInfo& reuseB) {
   llvm::SmallVector<lingodb::runtime::FilterDescription, 8> filtersA =
      decodeSimpleMatchFiltersAlongShadowChain(a.value, reuseA);
   llvm::SmallVector<lingodb::runtime::FilterDescription, 8> filtersB =
      decodeSimpleMatchFiltersAlongShadowChain(b.value, reuseB);
   if (filtersA.empty() || filtersB.empty()) return false;
   return matchFiltersDefinitelyDisjoint(filtersA, filtersB);
}

} // anonymous

mlir::Value canonicalizeStateValueForReuse(mlir::Value v) {
   return canonicalizeStateValueDeep(v);
}

void printExecutionSteps(mlir::ModuleOp moduleOp, llvm::raw_ostream& os) {
   struct StateInfo {
      int createdAt = -1;
      llvm::SmallSet<int, 16> reads;
      llvm::SmallSet<int, 16> writes;
   };

   llvm::DenseMap<mlir::Value, StateInfo> stateInfo;
   llvm::DenseMap<int, subop::ExecutionStepOp> stepByIndex;
   llvm::DenseMap<int, llvm::DenseMap<mlir::Value, RWFlags>> rwByStep;
   llvm::DenseMap<mlir::Value, int> createdAtByState; // convenience (state value -> createdAt)
   llvm::DenseMap<mlir::Value, mlir::Value> mergedFromShadowState;

   auto recordState = [&](mlir::Value v) -> StateInfo& {
      return stateInfo[v];
   };

   // Treat external state inputs (func arguments) as "created" outside of execution steps.
   moduleOp.walk([&](mlir::func::FuncOp func) {
      for (auto a : func.getArguments()) {
         auto t = a.getType();
         if (!isStateType(t) && !isThreadLocalOfStateType(t)) continue;
         auto& info = recordState(a);
         createdAtByState[a] = info.createdAt;
      }
   });

   size_t idx = 0;
   walkTopLevelExecutionSteps(moduleOp, [&](subop::ExecutionStepOp step) {
      int stepIdx = static_cast<int>(idx++);
      stepByIndex[stepIdx] = step;

      bool tableRef = isExternalTableRefStep(step);
      bool createOnly = isCreateOnlyExecutionStep(step);

      // Record creation of step results (state values).
      for (auto r : step.getResults()) {
         auto t = r.getType();
         if (!isStateType(t) && !isThreadLocalOfStateType(t)) {
            continue;
         }
         auto& info = recordState(r);
         if (info.createdAt < 0) {
            info.createdAt = stepIdx;
         }
         createdAtByState[r] = info.createdAt;
         if (!createOnly && !tableRef) {
            info.writes.insert(stepIdx); // creation in non-create-only step counts as write
         }
      }

      // Record creation for state-typed SSA anywhere under this top-level step (incl. nested steps).
      step.walk([&](mlir::Operation* op) {
         for (auto r : op->getResults()) {
            auto t = r.getType();
            if (!isStateType(t) && !isThreadLocalOfStateType(t)) continue;
            auto& info = recordState(r);
            if (info.createdAt < 0) info.createdAt = stepIdx;
            createdAtByState[r] = info.createdAt;
         }
      });

      // RW: top-level body + all nested execution_step bodies merged here.
      auto rw = analyzeStepStateRWWithNested(step);
      rwByStep[stepIdx] = rw;
      for (auto it : rw) {
         auto& info = recordState(it.first);
         if (it.second.read) info.reads.insert(stepIdx);
         if (it.second.write) info.writes.insert(stepIdx);
      }

      os << "\n// ---- execution_step[" << stepIdx << "]";
      if (tableRef) os << " table_ref";
      if (createOnly) os << " create_only";
      os << " ----\n";
      printStepStateRW(step, os);
      step->print(os);
      os << "\n";
   });

   mlir::OpPrintingFlags flags;
   os << "\n// ==== state debug summary ====\n";
   // After deep canonicalization, any remaining block-argument state should be a real external
   // (func argument). Otherwise it's a missing mapping and should crash loudly.
   for (auto& it : stateInfo) {
      if (it.second.createdAt >= 0) continue;
      if (!mlir::isa<mlir::BlockArgument>(it.first)) continue;
      assert(isFuncBlockArgument(it.first) && "unresolved external block-argument state: missing canonicalization mapping");
      it.second.createdAt = -2;
      createdAtByState[it.first] = it.second.createdAt;
   }
   for (auto& it : stateInfo) {
      os << "// ";
      it.first.printAsOperand(os, flags);
      os << " : ";
      it.first.getType().print(os);
      os << "\n";
      os << "//   created_at: ";
      os << it.second.createdAt << "\n";
      os << "//   read_at: ";
      for (int s : it.second.reads) os << s << " ";
      os << "\n";
      os << "//   write_at: ";
      for (int s : it.second.writes) os << s << " ";
      os << "\n";
   }

   os << "\n// ==== merged_from_shadow_state (global <- thread_local / hiv <- buffer) ====\n";
   moduleOp.walk([&](subop::CreateHashIndexedView chiv) {
      mlir::Value globalBuf = canonicalizeStateValueDeep(chiv.getSource());
      mlir::Value hiv = canonicalizeStateValueDeep(chiv.getResult());
      if (!mlir::isa<subop::BufferType>(globalBuf.getType())) return;
      if (!mlir::isa<subop::HashIndexedViewType>(hiv.getType())) return;
      mergedFromShadowState[hiv] = globalBuf;
      os << "// ";
      globalBuf.printAsOperand(os, flags);
      os << " <- ";
      hiv.printAsOperand(os, flags);
      os << " (hiv<-buf)\n";
   });
   moduleOp.walk([&](subop::ExecutionStepOp step) {
      auto& block = step.getSubOps().front();
      for (auto& op : block.without_terminator()) {
         auto merge = mlir::dyn_cast<subop::MergeOp>(op);
         if (!merge) continue;
         assert(merge.getThreadLocal());
         mlir::Value in = canonicalizeStateValueDeep(merge.getThreadLocal());
         mlir::Value out = canonicalizeStateValueDeep(merge.getResult());
         mergedFromShadowState[out] = in;
         os << "// ";
         in.printAsOperand(os, flags);
         os << " <- ";
         out.printAsOperand(os, flags);
         os << " (merge)\n";
      }
   });

   // Convenience: state -> writes set.
   llvm::DenseMap<mlir::Value, llvm::SmallSet<int, 16>> writesByState;
   for (auto& it : stateInfo) {
      // Always materialize an entry so downstream code can assert presence.
      writesByState[it.first] = it.second.writes;
   }

   os << "\n// ==== state construction ====\n";
   llvm::SmallVector<mlir::Value, 32> states;
   states.reserve(stateInfo.size());
   for (auto& it : stateInfo) {
      states.push_back(it.first);
   }
   llvm::sort(states, [&](mlir::Value a, mlir::Value b) {
      int ca = stateInfo[a].createdAt;
      int cb = stateInfo[b].createdAt;
      if (ca != cb) return ca < cb;
      return a.getAsOpaquePointer() < b.getAsOpaquePointer();
   });

   llvm::DenseSet<mlir::Value> shadowChainPartners;
   for (auto& kv : mergedFromShadowState) shadowChainPartners.insert(kv.second);

   for (auto s : states) {
      if (shadowChainPartners.contains(s)) continue;

      auto stepIdxs = getConstructionStepIndicesForState(s, createdAtByState, writesByState, mergedFromShadowState);
      auto prereqStepIdxs = sortedUniqueStepIndices(stepIdxs);
      os << "\n// -- state ";
      s.printAsOperand(os, flags);
      os << " : ";
      s.getType().print(os);
      os << " --\n";
      os << "//   construction_steps: ";
      for (int si : stepIdxs) os << si << " ";
      os << "\n";
      if (auto it = mergedFromShadowState.find(s); it != mergedFromShadowState.end()) {
         os << "//   merged_from_shadow_state: ";
         it->second.printAsOperand(os, flags);
         os << "\n";
      }

      llvm::SmallVector<mlir::Value, 8> prereqs = getPrereqStatesForConstructionSteps(prereqStepIdxs, s, rwByStep);
      llvm::DenseSet<mlir::Value> prereqSeen;
      for (auto pv : prereqs) prereqSeen.insert(pv);
      for (int si : prereqStepIdxs) {
         auto itSt = stepByIndex.find(si);
         assert(itSt != stepByIndex.end());
         appendNestedStateOperandsAsPrereqs(itSt->second, s, prereqSeen, prereqs);
      }
      if (!prereqs.empty()) {
         os << "//   prereq_states(read): ";
         for (auto p : prereqs) {
            p.printAsOperand(os, flags);
            os << " ";
         }
         os << "\n";
      }

      for (int si : stepIdxs) {
         auto itStep = stepByIndex.find(si);
         assert(itStep != stepByIndex.end() && "construction step index must exist");
         os << "\n// ---- construction execution_step[" << si << "] ----\n";
         // Also show which internal SSA value is returned for the state, if any.
         // (Useful when the constructed state is a step result).
         // Find if s is a result of this step.
         subop::ExecutionStepOp step = itStep->second;
         for (unsigned r = 0; r < step.getNumResults(); r++) {
            if (step.getResult(r) != s) continue;
            mlir::Value internal = getReturnedInternalValueForStepResult(step, r);
            assert(internal && "constructed state result must map to a returned internal value");
            os << "//   returned_internal_value: ";
            internal.printAsOperand(os, flags);
            os << "\n";
            if (auto* def = internal.getDefiningOp()) {
               if (mlir::isa<subop::MergeOp>(def)) {
                  os << "//   returned_internal_def: subop.merge\n";
               } else if (isCreateLike(*def)) {
                  os << "//   returned_internal_def: create_like\n";
               }
            }
            break;
         }
         // Print full step (includes all internal operators).
         step->print(os);
         os << "\n";
      }
   }

   // Intra-module equivalence classes for "table-dependent" states.
   {
      auto tableDescr = buildTableDescrByTableState(moduleOp);
      auto profiles = buildStateMatchProfiles(/*queryId*/ 0, moduleOp, tableDescr);
      llvm::StringMap<llvm::SmallVector<const StateMatchProfile*, 8>> groups;
      for (auto& p : profiles) {
         if (!p.eligible) continue;
         std::string deps;
         for (auto& d : p.depTokensSorted) {
            if (!deps.empty()) deps.push_back('|');
            deps.append(d);
         }
         std::string key = deps + "@@type=" + p.typeFingerprintStr + "@@h=" + std::to_string(p.constructionHash);
         groups[key].push_back(&p);
      }

      os << "\n// ==== intra-query equivalent table-dependent states ====\n";
      size_t printed = 0;
      mlir::OpPrintingFlags flags;
      for (auto& it : groups) {
         if (it.second.size() < 2) continue;
         os << "\n// -- eq_group key=" << it.getKey() << " --\n";
         for (auto* p : it.second) {
            os << "//   ";
            p->value.printAsOperand(os, flags);
            os << "\n";
         }
         if (++printed >= 50) {
            os << "\n// ... truncated eq_groups (max 50) ...\n";
            break;
         }
      }
      if (printed == 0) os << "// (no groups)\n";
   }
}

namespace {
template <typename GroupOp>
static void printExecutionGroupStepLines(GroupOp group, llvm::raw_ostream& os, llvm::StringRef whereLabel) {
   if (group.getSubOps().empty()) return;
   mlir::Block& body = group.getSubOps().front();
   os << "// --- " << whereLabel << " ---\n";
   unsigned ord = 0;
   for (mlir::Operation& op : body.without_terminator()) {
      if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(&op)) {
         os << "//   [" << ord << "] execution_step inner_ops:";
         for (mlir::Operation& inner : step.getSubOps().front().without_terminator()) {
            os << " " << inner.getName().getStringRef();
         }
         os << " | step_inputs:";
         for (mlir::Value in : step.getInputs()) {
            os << " ";
            if (auto* def = in.getDefiningOp()) {
               os << def->getName().getStringRef();
               if (auto defStep = mlir::dyn_cast<subop::ExecutionStepOp>(def)) {
                  if (auto* defGroupOp = defStep->getParentOp()) {
                     if (auto encEg = mlir::dyn_cast<subop::ExecutionGroupOp>(defGroupOp)) {
                        unsigned srcOrd = 0;
                        for (mlir::Operation& o : encEg.getSubOps().front().without_terminator()) {
                           if (&o == def) {
                              os << "[step " << srcOrd << "]";
                              break;
                           }
                           ++srcOrd;
                        }
                     } else if (auto encNeg = mlir::dyn_cast<subop::NestedExecutionGroupOp>(defGroupOp)) {
                        unsigned srcOrd = 0;
                        for (mlir::Operation& o : encNeg.getSubOps().front().without_terminator()) {
                           if (&o == def) {
                              os << "[nested_step " << srcOrd << "]";
                              break;
                           }
                           ++srcOrd;
                        }
                     }
                  }
               }
            } else if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(in)) {
               if (mlir::Operation* regParent = ba.getOwner()->getParentOp()) {
                  os << "arg@" << regParent->getName() << "#" << ba.getArgNumber();
               } else {
                  os << "block_arg";
               }
            } else {
               os << "?";
            }
         }
         os << "\n";
         // Nested lowering uses a fresh ordinal space per nested_execution_group.
         step->walk([&](subop::NestedExecutionGroupOp nested) {
            printExecutionGroupStepLines(nested, os, "nested_execution_group(under outer step)");
            return mlir::WalkResult::skip();
         });
      } else {
         os << "//   [" << ord << "] " << op.getName() << " (non-execution_step)\n";
      }
      ++ord;
   }
}
} // namespace

bool isExternalTableRefStep(subop::ExecutionStepOp step) {
   auto& block = step.getSubOps().front();
   auto ops = block.without_terminator();
   if (std::distance(ops.begin(), ops.end()) != 1) {
      return false;
   }
   auto* onlyOp = &*ops.begin();
   if (!mlir::isa<subop::GetExternalOp>(onlyOp)) {
      return false;
   }
   auto ret = mlir::cast<subop::ExecutionStepReturnOp>(block.getTerminator());
   if (ret.getInputs().size() != 1) {
      return false;
   }
   return ret.getInputs()[0].getDefiningOp() == onlyOp;
}

void printTopLevelExecutionStepLayout(mlir::ModuleOp moduleOp, llvm::raw_ostream& os) {
   os << "\n// ==== execution_step layout (block order == SubOp lowering ordinal) ====\n";
   os << "// Note: use the same indices as stderr from SubOpToControlFlow (per execution_group / nested group).\n";
   for (mlir::func::FuncOp func : moduleOp.getOps<mlir::func::FuncOp>()) {
      if (func.isDeclaration()) continue;
      os << "// func @" << func.getName() << "\n";
      for (mlir::Operation& top : func.front()) {
         if (auto group = mlir::dyn_cast<subop::ExecutionGroupOp>(&top)) {
            printExecutionGroupStepLines(group, os, "execution_group @main pipeline");
         }
      }
   }
}

void printCrossQueryStateMatches(llvm::ArrayRef<std::pair<int, mlir::ModuleOp>> queries, llvm::raw_ostream& os) {
   assert(queries.size() >= 2 && "need at least two modules to compare");

   struct QueryModel {
      int id = -1;
      mlir::ModuleOp module;
      llvm::DenseMap<mlir::Value, std::string> tableDescr;
      llvm::SmallVector<StateMatchProfile, 128> profiles;
      ModuleReuseInfo reuse;
   };

   llvm::SmallVector<QueryModel, 4> models;
   models.reserve(queries.size());
   for (auto& q : queries) {
      QueryModel m;
      m.id = q.first;
      m.module = q.second;
      m.tableDescr = buildTableDescrByTableState(m.module);
      m.profiles = buildStateMatchProfiles(m.id, m.module, m.tableDescr);
      m.reuse = collectModuleReuseInfo(m.module);
      models.push_back(std::move(m));
   }

   os << "\n// ==== cross-query state matches (experimental) ====\n";
   os << "// Eligible states: deps resolve to get_external leaves (optionally via one transparent carrier chain), constructionHash + type_fp match.\n";
   os << "// Special cases: transparent buffer/thread_local (folded, not reuse targets); hash_indexed_view (join match details).\n";

   // Debug helper: print eligible join hash_indexed_view profiles per query (capped).
   {
      os << "\n// ==== debug: eligible hash_indexed_view (join build) profiles (capped) ====\n";
      size_t cap = 20;
      for (auto& m : models) {
         size_t printedDbg = 0;
         for (auto& p : m.profiles) {
            if (!mlir::isa<subop::HashIndexedViewType>(p.value.getType())) continue;
            std::string deps = joinSortedStrings(p.depTokensSorted);
            os << "//   query[" << m.id << "] ";
            mlir::OpPrintingFlags dbgFlags;
            p.value.printAsOperand(os, dbgFlags);
            os << " eligible=" << (p.eligible ? "true" : "false");
            os << " state_hash=" << p.stateHash;
            os << " h=" << p.constructionHash;
            os << " step_hashes=[";
            for (size_t i = 0; i < p.constructionStepHashes.size(); i++) {
               if (i) os << ",";
               os << p.constructionStepHashes[i];
            }
            os << "]";
            if (!p.constructionStepHashByIndex.empty()) {
               os << " step_hashes_by_index=[";
               for (size_t i = 0; i < p.constructionStepHashByIndex.size(); i++) {
                  if (i) os << ",";
                  os << p.constructionStepHashByIndex[i].first << ":" << p.constructionStepHashByIndex[i].second;
               }
               os << "]";
            }
            os << " type_fp=" << p.typeFingerprintStr;
            if (!p.storedValueMembersFingerprint.empty()) {
               os << " stored_cols=" << p.storedValueMembersFingerprint;
            }
            os << " deps=" << deps << "\n";
            if (++printedDbg >= cap) break;
         }
         if (printedDbg == 0) os << "//   query[" << m.id << "] (none)\n";
      }
   }

   // Debug helper: print eligible optimistic_ht-related profiles per query (capped).
   {
      os << "\n// ==== debug: eligible optimistic_ht profiles (capped) ====\n";
      size_t cap = 20;
      for (auto& m : models) {
         size_t printedDbg = 0;
         for (auto& p : m.profiles) {
            std::string ty;
            llvm::raw_string_ostream tss(ty);
            p.value.getType().print(tss);
            tss.flush();
            if (ty.find("optimistic_ht") == std::string::npos) continue;

            std::string deps = joinSortedStrings(p.depTokensSorted);
            os << "//   query[" << m.id << "] ";
            mlir::OpPrintingFlags dbgFlags;
            p.value.printAsOperand(os, dbgFlags);
            os << " eligible=" << (p.eligible ? "true" : "false");
            os << " state_hash=" << p.stateHash;
            os << " type=" << ty;
            os << " deps=" << deps;
            if (!p.aggregateGroupKeyFingerprint.empty()) {
               os << " agg_key=" << p.aggregateGroupKeyFingerprint;
            }
            os << " h=" << p.constructionHash;
            os << "\n";
            if (++printedDbg >= cap) {
               os << "//   ... truncated (max " << cap << ") ...\n";
               break;
            }
         }
         if (printedDbg == 0) {
            os << "//   query[" << m.id << "] (none)\n";
         }
      }
   }

   {
      os << "\n// ==== debug: eligible result_table profiles (capped) ====\n";
      size_t cap = 20;
      for (auto& m : models) {
         size_t printedDbg = 0;
         for (auto& p : m.profiles) {
            if (!mlir::isa<subop::ResultTableType>(p.value.getType())) continue;
            std::string deps = joinSortedStrings(p.depTokensSorted);
            os << "//   query[" << m.id << "] ";
            mlir::OpPrintingFlags dbgFlags;
            p.value.printAsOperand(os, dbgFlags);
            os << " eligible=" << (p.eligible ? "true" : "false");
            os << " state_hash=" << p.stateHash;
            os << " h=" << p.constructionHash;
            os << " step_hashes=[";
            for (size_t i = 0; i < p.constructionStepHashes.size(); i++) {
               if (i) os << ",";
               os << p.constructionStepHashes[i];
            }
            os << "]";
            if (!p.constructionStepHashByIndex.empty()) {
               os << " step_hashes_by_index=[";
               for (size_t i = 0; i < p.constructionStepHashByIndex.size(); i++) {
                  if (i) os << ",";
                  os << p.constructionStepHashByIndex[i].first << ":" << p.constructionStepHashByIndex[i].second;
               }
               os << "]";
            }
            os << " type_fp=" << p.typeFingerprintStr;
            os << " deps=" << deps << "\n";
            if (++printedDbg >= cap) break;
         }
         if (printedDbg == 0) os << "//   query[" << m.id << "] (none)\n";
      }
   }

   {
      os << "\n// ==== debug: all other state profiles (capped) ====\n";
      size_t cap = 40;
      for (auto& m : models) {
         size_t printedDbg = 0;
         for (auto& p : m.profiles) {
            if (mlir::isa<subop::HashIndexedViewType, subop::ResultTableType, subop::PreAggrHtType>(
                   p.value.getType()))
               continue;
            std::string deps = joinSortedStrings(p.depTokensSorted);
            std::string ty;
            llvm::raw_string_ostream tss(ty);
            p.value.getType().print(tss);
            tss.flush();
            os << "//   query[" << m.id << "] ";
            mlir::OpPrintingFlags dbgFlags;
            p.value.printAsOperand(os, dbgFlags);
            os << " eligible=" << (p.eligible ? "true" : "false");
            os << " state_hash=" << p.stateHash;
            os << " h=" << p.constructionHash;
            os << " step_hashes=[";
            for (size_t i = 0; i < p.constructionStepHashes.size(); i++) {
               if (i) os << ",";
               os << p.constructionStepHashes[i];
            }
            os << "]";
            if (!p.constructionStepHashByIndex.empty()) {
               os << " step_hashes_by_index=[";
               for (size_t i = 0; i < p.constructionStepHashByIndex.size(); i++) {
                  if (i) os << ",";
                  os << p.constructionStepHashByIndex[i].first << ":" << p.constructionStepHashByIndex[i].second;
               }
               os << "]";
            }
            os << " type=" << ty;
            os << " type_fp=" << p.typeFingerprintStr;
            os << " deps=" << deps << "\n";
            if (++printedDbg >= cap) {
               os << "//   ... truncated (max " << cap << ") ...\n";
               break;
            }
         }
         if (printedDbg == 0) os << "//   query[" << m.id << "] (none)\n";
      }
   }

   mlir::OpPrintingFlags flags;
   llvm::SmallVector<CrossQueryStateMatchGroup, 64> groups = collectCrossQueryStateMatchGroups(queries);
   size_t printed = 0;
   for (const CrossQueryStateMatchGroup& g : groups) {
      os << "\n// -- match_group cache_key=" << g.cacheKey
         << " filter_pred_reuse=" << (g.enableFilterPredReuse ? "true" : "false")
         << " requires_join_layout_union=" << (g.requiresJoinLayoutUnion ? "true" : "false") << " --\n";
      for (const CrossQueryStateMatchEntry& e : g.entries) {
         os << "//   query[" << e.query << "] ";
         e.state.printAsOperand(os, flags);
         for (auto& m : models) {
            if (m.id != e.query) continue;
            for (auto& p : m.profiles) {
               if (p.value != e.state) continue;
               os << " state_hash=" << p.stateHash;
               break;
            }
            break;
         }
         os << "\n";
      }
      if (++printed >= 100) {
         os << "\n// ... truncated match_groups (max 100) ...\n";
         return;
      }
   }
   if (printed == 0) os << "// (no matches)\n";
}

ModuleReuseInfo collectModuleReuseInfo(mlir::ModuleOp moduleOp) {
   return analyzeModuleForMatchAndReuse(moduleOp).reuse;
}

llvm::DenseMap<const void*, uint64_t> collectStateConstructionColumnHashes(mlir::ModuleOp moduleOp,
                                                                           mlir::Value state) {
   llvm::DenseMap<const void*, uint64_t> out;
   if (!state) return out;

   auto* dialect = moduleOp.getContext()->getLoadedDialect<subop::SubOperatorDialect>();
   assert(dialect && "subop dialect must be loaded to fingerprint members");
   subop::MemberManager& memberManager = dialect->getMemberManager();
   auto* tupDialect =
      moduleOp.getContext()->getLoadedDialect<lingodb::compiler::dialect::tuples::TupleStreamDialect>();
   assert(tupDialect && "tuples dialect must be loaded");

   ModuleMatchAndReuseAnalysis module = analyzeModuleForMatchAndReuse(moduleOp);
   StateDependencyGraph depGraph = buildStateDependencyGraphFromStepRw(module.rwByStep);
   llvm::DenseMap<mlir::Value, std::string> tableDescrByTableState = buildTableDescrByTableState(moduleOp);
   StateHashEnv stateHashEnv = buildStateHashEnv(module, depGraph, tableDescrByTableState);

   mlir::Value stateCanon = canonicalizeStateValueDeep(state);
   auto constructionStepIndices = getConstructionStepIndicesForState(
      stateCanon, module.createdAtByState, module.writesByState, module.mergedFromShadowState);
   if (constructionStepIndices.empty()) return out;

   const bool isHiv = mlir::isa<subop::HashIndexedViewType>(stateCanon.getType());
   std::optional<JoinHivMatchDetails> joinHivDetails;
   if (isHiv) {
      joinHivDetails.emplace(computeJoinHivMatchDetails(
         stateCanon, mlir::cast<subop::HashIndexedViewType>(stateCanon.getType()), memberManager,
         tupDialect->getColumnManager(), module.reuse.writerStepsByState, constructionStepIndices,
         module.stepByIndex));
   }

   llvm::DenseSet<mlir::Value> targetConstructionStates;
   targetConstructionStates.insert(stateCanon);
   forEachShadowChainPredecessorValue(stateCanon, module.mergedFromShadowState, [&](mlir::Value shadow) {
      targetConstructionStates.insert(canonicalizeStateValueDeep(shadow));
   });

   for (int si : constructionStepIndices) {
      auto itS = module.stepByIndex.find(si);
      assert(itS != module.stepByIndex.end());
      StepDagHasher hasher;
      hasher.step = itS->second;
      hasher.targetState = stateCanon;
      hasher.targetStateType = stateCanon.getType();
      hasher.tableDescrByTableState = &tableDescrByTableState;
      hasher.memberManager = &memberManager;
      hasher.columnManager = &tupDialect->getColumnManager();
      hasher.externalDatasourceByTableState = &module.reuse.externalDatasourceByTableState;
      hasher.stateHashEnv = &stateHashEnv;
      hasher.targetConstructionStates = &targetConstructionStates;
      hasher.outColumnHashByColumn = &out;
      if (joinHivDetails) {
         hasher.joinIndexMemberNamesSanitized = &joinHivDetails->indexMemberNamesSanitized;
         hasher.joinKeyColumnAttrHashes = &joinHivDetails->joinKeyColumnAttrHashes;
         hasher.joinKeyColumnIdentifiersSanitized = &joinHivDetails->joinKeyColumnIdentifiersSanitized;
      }
      (void)hasher.hashStepReturnGraph();
   }
   return out;
}

llvm::SmallVector<CrossQueryStateMatchGroup, 64>
collectCrossQueryStateMatchGroups(llvm::ArrayRef<std::pair<int, mlir::ModuleOp>> queries) {
   assert(queries.size() >= 2 && "need at least two modules to compare");

   struct QueryModel {
      int id = -1;
      mlir::ModuleOp module;
      llvm::DenseMap<mlir::Value, std::string> tableDescr;
      llvm::SmallVector<StateMatchProfile, 128> profiles;
      ModuleReuseInfo reuse;
   };

   llvm::SmallVector<QueryModel, 4> models;
   models.reserve(queries.size());
   for (auto& q : queries) {
      QueryModel m;
      m.id = q.first;
      m.module = q.second;
      m.tableDescr = buildTableDescrByTableState(m.module);
      m.profiles = buildStateMatchProfiles(m.id, m.module, m.tableDescr);
      m.reuse = collectModuleReuseInfo(m.module);
      models.push_back(std::move(m));
   }

   llvm::SmallVector<const StateMatchProfile*, 256> all;
   llvm::DenseMap<int, const QueryModel*> modelByQueryId;
   for (auto& m : models) {
      modelByQueryId[m.id] = &m;
      for (auto& p : m.profiles) {
         if (!p.eligible) continue;
         all.push_back(&p);
      }
   }

   auto makeKeyStr = [](const StateMatchProfile& p) -> std::string {
      std::string deps;
      for (auto& d : p.depTokensSorted) {
         if (!deps.empty()) deps.push_back('|');
         deps.append(d);
      }
      return deps + "@@type=" + p.typeFingerprintStr + "@@h=" + std::to_string(p.constructionHash);
   };
   auto makeRelaxedHivKeyStr = [](const StateMatchProfile& p) -> std::string {
      std::string deps;
      for (auto& d : p.depTokensSorted) {
         if (!deps.empty()) deps.push_back('|');
         deps.append(d);
      }
      return deps + "@@type=" + p.typeFingerprintStr + "@@hiv_relaxed";
   };
   auto complexResidualMatchAllowed = [&](const StateMatchProfile& a, const QueryModel* modelA,
                                          const StateMatchProfile& b, const QueryModel* modelB) {
      if (!a.hasComplexResidualTableFilter && !b.hasComplexResidualTableFilter) return true;
      if (!a.hasResidualTableFilter || !b.hasResidualTableFilter) return true;
      if (a.hasComplexResidualTableFilter && b.hasComplexResidualTableFilter) return true;
      return !decodeSimpleMatchFiltersAlongShadowChain(a.value, modelA->reuse).empty() ||
             !decodeSimpleMatchFiltersAlongShadowChain(b.value, modelB->reuse).empty();
   };
   using ProfileBucketMap = llvm::DenseMap<uint64_t, llvm::SmallVector<const StateMatchProfile*, 8>>;
   auto appendToBucket = [](ProfileBucketMap& buckets,
                            llvm::SmallVectorImpl<uint64_t>& bucketOrder,
                            uint64_t hash,
                            const StateMatchProfile* profile) {
      auto it = buckets.find(hash);
      if (it == buckets.end()) {
         bucketOrder.push_back(hash);
         it = buckets.try_emplace(hash).first;
      }
      it->second.push_back(profile);
   };

   ProfileBucketMap profilesByConstructionHash;
   llvm::SmallVector<uint64_t, 64> constructionHashOrder;
   for (auto* p : all) {
      appendToBucket(profilesByConstructionHash, constructionHashOrder, p->constructionHash, p);
   }

   ProfileBucketMap hivProfilesByRelaxedHash;
   llvm::SmallVector<uint64_t, 32> hivRelaxedHashOrder;
   for (auto* p : all) {
      if (!mlir::isa<subop::HashIndexedViewType>(p->value.getType())) continue;
      std::string k = makeRelaxedHivKeyStr(*p);
      uint64_t hash = static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(k)));
      appendToBucket(hivProfilesByRelaxedHash, hivRelaxedHashOrder, hash, p);
   }

   llvm::SmallVector<CrossQueryStateMatchGroup, 64> out;
   llvm::DenseSet<mlir::Value> matchedStates;

   auto appendGroupFromBucket = [&](llvm::ArrayRef<const StateMatchProfile*> bucket,
                                    llvm::function_ref<bool(const StateMatchProfile&)> seedOk,
                                    llvm::function_ref<bool(const StateMatchProfile&, const StateMatchProfile&)> peerOk,
                                    llvm::function_ref<std::string(const StateMatchProfile&)> keyForSeed,
                                    bool enableFilterPredReuse) {
      for (size_t i = 0; i < bucket.size(); i++) {
         auto* a = bucket[i];
         if (matchedStates.contains(a->value)) continue;
         if (!seedOk(*a)) continue;

         llvm::SmallVector<const StateMatchProfile*, 8> members;
         members.push_back(a);
         llvm::DenseSet<int> seenQueries;
         seenQueries.insert(a->queryId);
         for (size_t j = i + 1; j < bucket.size(); j++) {
            auto* b = bucket[j];
            if (matchedStates.contains(b->value)) continue;
            if (seenQueries.contains(b->queryId)) continue;
            if (!peerOk(*a, *b)) continue;
            members.push_back(b);
            seenQueries.insert(b->queryId);
         }
         if (members.size() < 2) continue;

         std::string k = keyForSeed(*a);
         uint64_t cacheKey = static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(k)));

         CrossQueryStateMatchGroup g;
         g.cacheKey = cacheKey;
         bool allHiv = true;
         bool sameStoredHivLayout = true;
         std::optional<std::string> firstStoredHivLayout;
         for (const StateMatchProfile* p : members) {
            if (!mlir::isa<subop::HashIndexedViewType>(p->value.getType())) {
               allHiv = false;
               break;
            }
            if (!firstStoredHivLayout) {
               firstStoredHivLayout = p->storedValueMembersFingerprint;
            } else if (*firstStoredHivLayout != p->storedValueMembersFingerprint) {
               sameStoredHivLayout = false;
            }
         }
         if (allHiv && !sameStoredHivLayout) g.requiresJoinLayoutUnion = true;
         bool needsFilterPredReuse = enableFilterPredReuse;
         if (needsFilterPredReuse && allHiv) {
            needsFilterPredReuse = false;
            for (const StateMatchProfile* p : members) {
               if (p->hasResidualTableFilter || p->hasComplexResidualTableFilter) {
                  needsFilterPredReuse = true;
                  break;
               }
               const QueryModel* model = modelByQueryId.lookup(p->queryId);
               assert(model && "missing query model for profile");
               if (!decodeSimpleMatchFiltersAlongShadowChain(p->value, model->reuse).empty()) {
                  needsFilterPredReuse = true;
                  break;
               }
               if (llvm::any_of(p->depTokensSorted, [](llvm::StringRef dep) {
                      return dep.starts_with("cache:");
                   })) {
                  needsFilterPredReuse = true;
                  break;
               }
            }
         }
         g.enableFilterPredReuse = needsFilterPredReuse;
         if (allHiv && needsFilterPredReuse) g.requiresJoinLayoutUnion = true;
         for (const StateMatchProfile* p : members) {
            g.entries.push_back(CrossQueryStateMatchEntry{p->queryId, p->value});
            matchedStates.insert(p->value);
         }
         out.push_back(std::move(g));
      }
   };

   for (uint64_t hash : constructionHashOrder) {
      auto bucketIt = profilesByConstructionHash.find(hash);
      assert(bucketIt != profilesByConstructionHash.end());
      auto& bucket = bucketIt->second;
      appendGroupFromBucket(
         bucket,
         [](const StateMatchProfile& p) {
            return !mlir::isa<subop::PreAggrHtType>(p.value.getType());
         },
         [&](const StateMatchProfile& a, const StateMatchProfile& b) {
            if (mlir::isa<subop::PreAggrHtType>(b.value.getType())) return false;
            if (a.constructionHash != b.constructionHash) return false;
            if (a.typeFingerprintStr != b.typeFingerprintStr) return false;
            if (a.depTokensSorted != b.depTokensSorted) return false;
            const QueryModel* modelA = modelByQueryId.lookup(a.queryId);
            const QueryModel* modelB = modelByQueryId.lookup(b.queryId);
            assert(modelA && modelB && "missing query model for profile");
            if (mlir::isa<subop::HashIndexedViewType>(a.value.getType()) &&
                mlir::isa<subop::HashIndexedViewType>(b.value.getType())) {
               if (a.hasUnsupportedResidualTableFilter || b.hasUnsupportedResidualTableFilter) return false;
               if (!complexResidualMatchAllowed(a, modelA, b, modelB)) return false;
            }
            // TODO(batch-reuse): Restore filter-disjoint pruning as part of clustering / reuse cost modeling.
            (void)profilesDefinitelyDisjointByFilters;
            return true;
         },
         makeKeyStr, /*enableFilterPredReuse=*/true);
   }

   for (uint64_t hash : hivRelaxedHashOrder) {
      auto bucketIt = hivProfilesByRelaxedHash.find(hash);
      assert(bucketIt != hivProfilesByRelaxedHash.end());
      auto& bucket = bucketIt->second;
      appendGroupFromBucket(
         bucket,
         [](const StateMatchProfile& p) {
            return mlir::isa<subop::HashIndexedViewType>(p.value.getType());
         },
         [&](const StateMatchProfile& a, const StateMatchProfile& b) {
            if (!mlir::isa<subop::HashIndexedViewType>(b.value.getType())) return false;
            if (a.hasUnsupportedResidualTableFilter || b.hasUnsupportedResidualTableFilter) return false;
            if (a.typeFingerprintStr != b.typeFingerprintStr) return false;
            if (a.depTokensSorted != b.depTokensSorted) return false;
            const QueryModel* modelA = modelByQueryId.lookup(a.queryId);
            const QueryModel* modelB = modelByQueryId.lookup(b.queryId);
            assert(modelA && modelB && "missing query model for HIV relaxed profile");
            if (!complexResidualMatchAllowed(a, modelA, b, modelB)) return false;
            // TODO(batch-reuse): Restore filter-disjoint pruning as part of clustering / reuse cost modeling.
            return true;
         },
         makeRelaxedHivKeyStr, /*enableFilterPredReuse=*/true);
   }

   // Aggregate hash tables match on table/filter dependencies plus group-key identity. Payload value
   // layout is intentionally ignored here; the reuse rewrite computes the payload union later.
   auto aggregateMatchKeyStr = [](const StateMatchProfile& p) -> std::string {
      return aggregateDependencyFingerprint(p.depTokensSorted) + "@@agg=" + p.aggregateGroupKeyFingerprint;
   };
   ProfileBucketMap aggregateProfilesByMatchHash;
   llvm::SmallVector<uint64_t, 32> aggregateMatchHashOrder;
   for (auto* p : all) {
      if (!mlir::isa<subop::PreAggrHtType>(p->value.getType())) continue;
      assert(!p->aggregateGroupKeyFingerprint.empty() && "eligible aggregate profile must carry group-key fingerprint");
      std::string k = aggregateMatchKeyStr(*p);
      uint64_t hash = static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(k)));
      appendToBucket(aggregateProfilesByMatchHash, aggregateMatchHashOrder, hash, p);
   }
   for (uint64_t hash : aggregateMatchHashOrder) {
      auto bucketIt = aggregateProfilesByMatchHash.find(hash);
      assert(bucketIt != aggregateProfilesByMatchHash.end());
      auto& bucket = bucketIt->second;
      appendGroupFromBucket(
         bucket,
         [](const StateMatchProfile& p) {
            return mlir::isa<subop::PreAggrHtType>(p.value.getType());
         },
         [&](const StateMatchProfile& a, const StateMatchProfile& b) {
            if (!mlir::isa<subop::PreAggrHtType>(b.value.getType())) return false;
            if (aggregateDependencyFingerprint(a.depTokensSorted) !=
                aggregateDependencyFingerprint(b.depTokensSorted))
               return false;
            return a.aggregateGroupKeyFingerprint == b.aggregateGroupKeyFingerprint;
         },
         aggregateMatchKeyStr, /*enableFilterPredReuse=*/false);
   }
   return out;
}

llvm::SmallVector<CrossQueryStateMatchPair, 64>
collectCrossQueryStateMatchPairs(llvm::ArrayRef<std::pair<int, mlir::ModuleOp>> queries) {
   llvm::SmallVector<CrossQueryStateMatchPair, 64> out;
   for (const CrossQueryStateMatchGroup& g : collectCrossQueryStateMatchGroups(queries)) {
      if (g.entries.size() < 2) continue;
      for (size_t i = 1; i < g.entries.size(); ++i) {
         CrossQueryStateMatchPair p;
         p.queryA = g.entries[0].query;
         p.queryB = g.entries[i].query;
         p.stateA = g.entries[0].state;
         p.stateB = g.entries[i].state;
         p.cacheKey = g.cacheKey;
         p.enableFilterPredReuse = g.enableFilterPredReuse;
         out.push_back(p);
      }
   }
   return out;
}

void forEachShadowChainPredecessor(mlir::Value v, const ModuleReuseInfo& reuse,
                                   llvm::function_ref<void(mlir::Value)> fn) {
   forEachShadowChainPredecessorValue(canonicalizeStateValueForReuse(v), reuse.mergedFromShadowState, fn);
}

mlir::Value hashIndexedViewShadowingBuffer(mlir::Value mergedGlobalBuffer, const ModuleReuseInfo& reuse) {
   mergedGlobalBuffer = canonicalizeStateValueForReuse(mergedGlobalBuffer);
   for (auto& kv : reuse.mergedFromShadowState) {
      if (kv.second != mergedGlobalBuffer) continue;
      if (mlir::isa<subop::HashIndexedViewType>(kv.first.getType())) return kv.first;
   }
   return {};
}

mlir::Value bufferJoinChainRootForReuse(mlir::Value v, const ModuleReuseInfo& reuse) {
   v = canonicalizeStateValueForReuse(v);
   if (mlir::isa<subop::HashIndexedViewType>(v.getType())) return v;
   if (mlir::Value hiv = hashIndexedViewShadowingBuffer(v, reuse)) return hiv;
   return v;
}

mlir::Value resolveCacheTargetStateForReuse(mlir::Value v, const ModuleReuseInfo& reuse) {
   return bufferJoinChainRootForReuse(v, reuse);
}

bool isBufferJoinChainNonRootPartner(mlir::Value v, const ModuleReuseInfo& reuse) {
   v = canonicalizeStateValueForReuse(v);
   for (auto& kv : reuse.mergedFromShadowState) {
      if (kv.second == v) return true;
   }
   return false;
}

void forEachBufferJoinChainPartner(mlir::Value chainRootBuffer, const ModuleReuseInfo& reuse,
                                   llvm::function_ref<void(mlir::Value)> fn) {
   mlir::Value root = bufferJoinChainRootForReuse(chainRootBuffer, reuse);
   fn(root);
   forEachShadowChainPredecessor(root, reuse, fn);
}

} // namespace lingodb::compiler::dialect::subop
