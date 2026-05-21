#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateExtraction.h"

#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/runtime/ExternalDataSourceProperty.h"
#include "lingodb/utility/Serialization.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <algorithm>
#include <cassert>
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

static std::string normalizeExternalDataSourceDescrHex(llvm::StringRef hexDescr) {
   using lingodb::runtime::ExternalDatasourceProperty;
   using lingodb::runtime::FilterDescription;
   using lingodb::runtime::FilterOp;

   ExternalDatasourceProperty ds = lingodb::utility::deserializeFromHexString<ExternalDatasourceProperty>(hexDescr);

   // Normalize mapping order.
   llvm::SmallVector<ExternalDatasourceProperty::Mapping, 8> mapping(ds.mapping.begin(), ds.mapping.end());
   llvm::sort(mapping, [](const auto& a, const auto& b) {
      return a.memberName < b.memberName;
   });

   // Normalize filters: de-duplicate then sort by a stable key.
   std::unordered_set<FilterDescription> uniq;
   llvm::SmallVector<FilterDescription, 8> filters;
   for (auto& f : ds.filterDescriptions) {
      if (!uniq.insert(f).second) continue;
      filters.push_back(f);
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
   llvm::sort(filters, [&](const FilterDescription& a, const FilterDescription& b) {
      if (a.columnName != b.columnName) return a.columnName < b.columnName;
      if (a.columnId != b.columnId) return a.columnId < b.columnId;
      if (a.op != b.op) return static_cast<uint8_t>(a.op) < static_cast<uint8_t>(b.op);
      auto av = filterValueToStr(a);
      auto bv = filterValueToStr(b);
      return av < bv;
   });

   // Render to a stable string that we still use for matching (for now).
   std::string s;
   llvm::raw_string_ostream ss(s);
   ss << "table=" << ds.tableName;
   ss << ";index=" << ds.index;
   ss << ";indexType=" << ds.indexType;
   ss << ";mapping=[";
   for (size_t i = 0; i < mapping.size(); i++) {
      if (i) ss << ",";
      ss << mapping[i].memberName << "->" << mapping[i].identifier;
   }
   ss << "]";
   ss << ";filters=[";
   for (size_t i = 0; i < filters.size(); i++) {
      if (i) ss << ",";
      ss << filters[i].columnName << "#" << filters[i].columnId << ":" << filterOpToStr(filters[i].op) << "{"
         << filterValueToStr(filters[i]) << "}";
   }
   ss << "]";
   ss.flush();
   return s;
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

// Canonicalize across nested regions/steps until we reach a step input/result or an external value.
static mlir::Value canonicalizeStateValueDeep(mlir::Value v) {
   llvm::DenseSet<void*> seen;
   while (true) {
      assert(v && "canonicalizeStateValueDeep called with null value");
      void* key = v.getAsOpaquePointer();
      assert(seen.insert(key).second && "canonicalizeStateValueDeep hit a cycle");
      auto step = findEnclosingExecutionStep(v);
      if (!step) {
         // Outside any execution_step: must be a func argument (external), otherwise missing mapping.
         assert(isFuncBlockArgument(v) && "state value outside any execution_step must be a func argument");
      }
      auto c = canonicalizeStateValue(step, v);
      if (c == v) return v;
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

struct RWFlags {
   bool read = false;
   bool write = false;
};

llvm::DenseMap<mlir::Value, RWFlags> analyzeStepStateRW(subop::ExecutionStepOp step) {
   llvm::DenseMap<mlir::Value, RWFlags> res;
   auto& block = step.getSubOps().front();

   if (isExternalTableRefStep(step)) {
      assert(step.getNumResults() == 1 && "external table ref step should have exactly one result");
      res[step.getResult(0)].write = true;
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
         if (isStateType(stv.getType()) || isThreadLocalOfStateType(stv.getType())) {
            res[canonicalizeStateValueDeep(stv)].write = true;
         }
         continue;
      }
      // `create_hash_indexed_view`: reads source buffer; produces a new hash_indexed_view state.
      if (auto hiv = mlir::dyn_cast<subop::CreateHashIndexedView>(&op)) {
         mlir::Value src = hiv.getSource();
         if (isStateType(src.getType()) || isThreadLocalOfStateType(src.getType())) {
            res[canonicalizeStateValueDeep(src)].read = true;
         }
         mlir::Value out = hiv.getResult();
         if (isStateType(out.getType()) || isThreadLocalOfStateType(out.getType())) {
            res[canonicalizeStateValueDeep(out)].write = true;
         }
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

void printStepStateRW(subop::ExecutionStepOp step, llvm::raw_ostream& os) {
   auto m = analyzeStepStateRW(step);
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

void addConstructionStepsForSingleState(mlir::Value state,
                                               const llvm::DenseMap<mlir::Value, int>& createdAtByState,
                                               const llvm::DenseMap<mlir::Value, llvm::SmallSet<int, 16>>& writesByState,
                                               llvm::SmallVectorImpl<int>& outSteps) {
   auto itC = createdAtByState.find(state);
   assert(itC != createdAtByState.end() && "state must have a recorded createdAt");
   int createdAt = itC->second;
   assert(createdAt >= 0 && "createdAt must be non-negative when present");
   outSteps.push_back(createdAt);

   auto itW = writesByState.find(state);
   assert(itW != writesByState.end() && "state must have a recorded writes set");
   // Include ALL write steps as part of construction (rare, but requested).
   for (int w : itW->second) {
      assert(w >= createdAt && "write steps must not occur before creation");
      outSteps.push_back(w);
   }
}

llvm::SmallVector<int, 8> getConstructionStepIndicesForState(
   mlir::Value state,
   const llvm::DenseMap<mlir::Value, int>& createdAtByState,
   const llvm::DenseMap<mlir::Value, llvm::SmallSet<int, 16>>& writesByState,
   const llvm::DenseMap<mlir::Value, mlir::Value>& mergedFromThreadLocal,
   const llvm::DenseMap<mlir::Value, mlir::Value>* hashIndexedViewFromMergedBuffer) {
   llvm::SmallVector<int, 8> steps;

   // Normal state: createdAt (+ all write steps).
   addConstructionStepsForSingleState(state, createdAtByState, writesByState, steps);

   // Merge-produced global state: include thread_local side construction steps as well,
   // but do NOT treat the thread_local as shareable output by itself.
   if (auto it = mergedFromThreadLocal.find(state); it != mergedFromThreadLocal.end()) {
      mlir::Value tl = it->second;
      addConstructionStepsForSingleState(tl, createdAtByState, writesByState, steps);
   }

   // TL buffer -> merged global buffer -> hash_indexed_view: profile/match on HIV, include buffer+TL steps.
   if (hashIndexedViewFromMergedBuffer) {
      if (auto itG = hashIndexedViewFromMergedBuffer->find(state); itG != hashIndexedViewFromMergedBuffer->end()) {
         mlir::Value globalBuf = itG->second;
         addConstructionStepsForSingleState(globalBuf, createdAtByState, writesByState, steps);
         if (auto itTL = mergedFromThreadLocal.find(globalBuf); itTL != mergedFromThreadLocal.end()) {
            addConstructionStepsForSingleState(itTL->second, createdAtByState, writesByState, steps);
         }
      }
   }

   llvm::sort(steps);
   steps.erase(std::unique(steps.begin(), steps.end()), steps.end());
   return steps;
}

llvm::SmallVector<mlir::Value, 8> getPrereqStatesForConstructionSteps(
   llvm::ArrayRef<int> stepIndices,
   mlir::Value constructedState,
   const llvm::DenseMap<int, llvm::DenseMap<mlir::Value, RWFlags>>& rwByStep) {
   llvm::SmallVector<mlir::Value, 8> prereqs;
   llvm::DenseSet<mlir::Value> seen;
   for (int s : stepIndices) {
      auto it = rwByStep.find(s);
      assert(it != rwByStep.end() && "construction step must exist in rwByStep");
      for (auto kv : it->second) {
         mlir::Value v = kv.first;
         if (v == constructedState) continue;
         if (!kv.second.read) continue;
         if (seen.insert(v).second) {
            prereqs.push_back(v);
         }
      }
   }
   return prereqs;
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
                                               llvm::SmallVector<mlir::Value, 8>& out) {
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

bool isTableStateValue(mlir::Value v) {
   return mlir::isa<subop::TableType>(v.getType());
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

static std::string fingerprintSortedMemberPairs(subop::MemberManager& mm, llvm::ArrayRef<subop::Member> members);

/// Join HIV cross-query matching: hash/link slots + first stored lookup key; payload columns tracked separately.
struct JoinHivMatchDetails {
   llvm::SmallSet<std::string, 8> indexMemberNamesSanitized;
   llvm::SmallSet<uint64_t, 8> joinKeyColumnAttrHashes;
   std::string storedValueMembersFingerprint;
};

static std::string tableNameDepToken(mlir::Value tableState,
                                    const llvm::DenseMap<mlir::Value, lingodb::runtime::ExternalDatasourceProperty>&
                                       externalDatasourceByTableState) {
   if (auto it = externalDatasourceByTableState.find(tableState); it != externalDatasourceByTableState.end()) {
      return std::string("table_name:") + it->second.tableName;
   }
   return {};
}

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

static std::optional<JoinHivMatchDetails> computeJoinHivMatchDetails(
   mlir::Value hiv, subop::HashIndexedViewType hivTy, subop::MemberManager& mm,
   lingodb::compiler::dialect::tuples::ColumnManager& columnManager,
   const llvm::DenseMap<mlir::Value, llvm::SmallVector<subop::ExecutionStepOp, 8>>& writerStepsByState) {
   subop::CreateHashIndexedView chiv = findCreateHashIndexedViewForState(hiv, writerStepsByState);
   if (!chiv) return std::nullopt;

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
      uint64_t h = static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(scopeSan)));
      h = hashCombineU64(h, static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(nameSan))));
      h = hashCombineU64(h, hashType(colTy));
      details.joinKeyColumnAttrHashes.insert(h);
   };

   if (joinValueMember) {
      auto joinValueSanitized = sanitizeMemberSlotName(mm.getName(joinValueMember));
      auto itW = writerStepsByState.find(hiv);
      if (itW != writerStepsByState.end()) {
         for (subop::ExecutionStepOp ws : itW->second) {
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
         }
      }
   }

   return details;
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

static bool executionStepBuildsJoinBuffer(subop::ExecutionStepOp step) {
   bool found = false;
   step.walk([&](subop::MaterializeOp mat) {
      if (mlir::isa<subop::BufferType>(mat.getState().getType())) found = true;
   });
   return found;
}

struct StepDagHasher {
   subop::ExecutionStepOp step;
   const llvm::DenseMap<mlir::Value, std::string>* tableDescrByTableState = nullptr;
   subop::MemberManager* memberManager = nullptr;
   lingodb::compiler::dialect::tuples::ColumnManager* columnManager = nullptr;
   bool relaxJoinPayloadColumns = false;
   const llvm::SmallSet<std::string, 8>* joinIndexMemberNamesSanitized = nullptr;
   const llvm::SmallSet<uint64_t, 8>* joinKeyColumnAttrHashes = nullptr;
   const llvm::DenseMap<mlir::Value, lingodb::runtime::ExternalDatasourceProperty>* externalDatasourceByTableState =
      nullptr;
   llvm::DenseMap<mlir::Value, uint64_t> memo;

   bool isWithinStep(mlir::Operation* op) {
      for (auto* p = op; p; p = p->getParentOp()) {
         if (p == step.getOperation()) return true;
      }
      return false;
   }

   uint64_t hashMlirType(mlir::Type t) const {
      if (!t) return 0;
      if (auto tl = mlir::dyn_cast<subop::ThreadLocalType>(t)) {
         return hashMlirType(tl.getWrapped());
      }
      if (relaxJoinPayloadColumns && joinIndexMemberNamesSanitized && memberManager) {
         if (auto buf = mlir::dyn_cast<subop::BufferType>(t)) {
            llvm::SmallVector<subop::Member, 8> kept;
            for (auto m : buf.getMembers().getMembers()) {
               if (joinIndexMemberNamesSanitized->contains(sanitizeMemberSlotName(memberManager->getName(m)))) {
                  kept.push_back(m);
               }
            }
            std::string fp = std::string("buffer{members=") + fingerprintSortedMemberPairs(*memberManager, kept) + "}";
            return static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(fp)));
         }
         if (auto hiv = mlir::dyn_cast<subop::HashIndexedViewType>(t)) {
            llvm::SmallVector<subop::Member, 4> indexValueMembers;
            for (auto m : hiv.getValueMembers().getMembers()) {
               if (joinIndexMemberNamesSanitized->contains(sanitizeMemberSlotName(memberManager->getName(m)))) {
                  indexValueMembers.push_back(m);
               }
            }
            std::string fp = std::string("hash_indexed_view{key_members=") +
                             fingerprintSortedMemberPairs(*memberManager, hiv.getKeyMembers().getMembers()) +
                             ",index_value_members=" + fingerprintSortedMemberPairs(*memberManager, indexValueMembers) +
                             ",compare=" + (hiv.getCompareHashForLookup() ? "1" : "0") + "}";
            return static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(fp)));
         }
      }
      return hashType(t);
   }

   uint64_t hashExternalLeaf(mlir::Value v) {
      if (relaxJoinPayloadColumns && externalDatasourceByTableState) {
         if (auto it = externalDatasourceByTableState->find(v); it != externalDatasourceByTableState->end()) {
            return static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(it->second.tableName)));
         }
      }
      // Treat external tables as stable leaves keyed by GetExternal descr + type.
      if (tableDescrByTableState) {
         if (auto it = tableDescrByTableState->find(v); it != tableDescrByTableState->end()) {
            uint64_t h = static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(it->second)));
            h = hashCombineU64(h, hashMlirType(v.getType()));
            return h;
         }
      }
      // Fallback: type-only leaf.
      return hashMlirType(v.getType());
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
         assert(columnManager);
         auto [scope, name] = columnManager->getName(&cr.getColumn());
         name = sanitizeBaseName(name);
         scope = sanitizeScopeName(scope);
         uint64_t h = static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(scope)));
         h = hashCombineU64(h, static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(name))));
         h = hashCombineU64(h, hashMlirType(cr.getColumn().type));
         return h;
      }
      if (auto cd = mlir::dyn_cast<lingodb::compiler::dialect::tuples::ColumnDefAttr>(a)) {
         assert(columnManager);
         auto [scope, name] = columnManager->getName(&cd.getColumn());
         name = sanitizeBaseName(name);
         scope = sanitizeScopeName(scope);
         uint64_t h = static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(scope)));
         h = hashCombineU64(h, static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(name))));
         h = hashCombineU64(h, hashMlirType(cd.getColumn().type));
         if (cd.getFromExisting()) h = hashCombineU64(h, hashAttrNormalized(cd.getFromExisting()));
         return h;
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
      assert(0);
   }

   bool includeMemberForJoinIndexHash(llvm::StringRef memberNameSanitized) const {
      if (!relaxJoinPayloadColumns || !joinIndexMemberNamesSanitized) return true;
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
      uint64_t h = hashOpName(*gather.getOperation());
      for (auto t : gather->getResultTypes()) h = hashCombineU64(h, hashMlirType(t));
      for (auto v : gather->getOperands()) h = hashCombineU64(h, hashValue(v));
      if (joinKeyColumnAttrHashes) {
         for (auto& [member, colDef] : gather.getMapping().getMapping()) {
            uint64_t colH = hashAttrNormalized(colDef);
            if (!joinKeyColumnAttrHashes->contains(colH)) continue;
            (void)member;
            h = hashCombineU64(h, colH);
         }
      }
      return h;
   }

   uint64_t hashMaterializeOpRelaxed(subop::MaterializeOp mat) {
      uint64_t h = hashOpName(*mat.getOperation());
      for (auto t : mat->getResultTypes()) h = hashCombineU64(h, hashMlirType(t));
      for (auto v : mat->getOperands()) h = hashCombineU64(h, hashValue(v));
      for (auto& [member, colRef] : mat.getMapping().getMapping()) {
         if (!includeMemberForJoinIndexHash(sanitizeMemberSlotName(memberManager->getName(member)))) continue;
         h = hashCombineU64(h, hashAttrNormalized(colRef));
         h = hashCombineU64(h, hashAttrNormalized(subop::MemberAttr::get(mat->getContext(), member)));
      }
      return h;
   }

   uint64_t hashOp(mlir::Operation& op) {
      if (relaxJoinPayloadColumns) {
         if (auto gather = mlir::dyn_cast<subop::GatherOp>(&op)) return hashGatherOpRelaxed(gather);
         if (auto mat = mlir::dyn_cast<subop::MaterializeOp>(&op)) return hashMaterializeOpRelaxed(mat);
      }
      uint64_t h = 0;
      h = hashCombineU64(h, hashOpName(op));
      h = hashCombineU64(h, hashAttrDictSorted(op));
      for (auto t : op.getResultTypes()) h = hashCombineU64(h, hashMlirType(t));
      // Operand order is semantic.
      for (auto v : op.getOperands()) h = hashCombineU64(h, hashValue(v));
      // Include nested regions (map/reduce/combine etc.)
      for (auto& r : op.getRegions()) h = hashCombineU64(h, hashRegion(r));
      return h;
   }

   uint64_t hashValue(mlir::Value v) {
      v = canonicalizeStateValueDeep(v);
      if (auto it = memo.find(v); it != memo.end()) return it->second;
      // Cycle guard (shouldn't happen in SSA, but ExecutionStep canonicalization can create
      // self-references if we accidentally try to hash the step op itself).
      memo[v] = 0;
      uint64_t h = 0;
      if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(v)) {
         h = hashExternalLeaf(canonicalizeStateValueDeep(ba));
      } else if (auto* def = v.getDefiningOp()) {
         if (mlir::isa<subop::ExecutionStepOp>(def)) {
            h = hashExternalLeaf(v);
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

   uint64_t hashStepReturnGraph() {
      auto& block = step.getSubOps().front();
      auto ret = mlir::cast<subop::ExecutionStepReturnOp>(block.getTerminator());
      uint64_t h = 0;
      for (auto v : ret.getInputs()) h = hashCombineU64(h, hashValue(v));
      return h;
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

// Fingerprint SubOp "state-like" types in a way that is stable across separate compilations of the same SQL.
// (MemberManager assigns unique `$<id>` suffixes that differ per MLIRContext/module.)
std::string normalizedSubopStateTypeFingerprint(subop::MemberManager& mm, mlir::Type t) {
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
   if (auto tl = mlir::dyn_cast<subop::ThreadLocalType>(t)) {
      return std::string("thread_local{") + normalizedSubopStateTypeFingerprint(mm, tl.getWrapped()) + "}";
   }
   // Generic fallback for other subop state types: use the printed type form but normalize
   // compiler-generated suffixes (e.g. "$<id>" and "_u_<id>") so that identical SQL compiled
   // in different MLIRContexts can still match.
   std::string s = typeFingerprint(t);
   std::string out;
   out.reserve(s.size());
   for (size_t i = 0; i < s.size();) {
      if (s[i] == '$') {
         out.push_back('$');
         i++;
         while (i < s.size() && s[i] >= '0' && s[i] <= '9') i++;
         continue;
      }
      if (i + 3 < s.size() && s[i] == '_' && s[i + 1] == 'u' && s[i + 2] == '_') {
         out.append("_u_");
         i += 3;
         while (i < s.size() && s[i] >= '0' && s[i] <= '9') i++;
         continue;
      }
      out.push_back(s[i]);
      i++;
   }
   return out;
}

llvm::DenseMap<mlir::Value, std::string> buildTableDescrByTableState(mlir::ModuleOp moduleOp) {
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
      mlir::Value t = step.getResult(0);
      tableDescr[t] = normalizeExternalDataSourceDescrHex(geOp.getDescr());
   });
   return tableDescr;
}

struct StateMatchProfile {
   int queryId = -1;
   mlir::Value value;
   bool eligible = false;
   llvm::SmallVector<std::string, 8> depTokensSorted;
   llvm::SmallVector<uint64_t, 8> constructionStepHashes;
   uint64_t constructionHash = 0;
   std::string typeFingerprintStr;
   /// Full HIV/buffer stored-value column layout (excluded from `constructionHash` / match type key).
   std::string storedValueMembersFingerprint;
};

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
   llvm::DenseMap<mlir::Value, mlir::Value> mergedFromThreadLocal;
   llvm::DenseMap<mlir::Value, llvm::SmallSet<int, 16>> writesByState;
   llvm::SmallVector<mlir::Value, 64> statesSorted;
   ModuleReuseInfo reuse;
};

static ModuleMatchAndReuseAnalysis analyzeModuleForMatchAndReuse(mlir::ModuleOp moduleOp) {
   ModuleMatchAndReuseAnalysis a;
   auto recordState = [&](mlir::Value v) -> ModuleMatchAndReuseAnalysis::StateInfo& { return a.stateInfo[v]; };

   size_t idx = 0;
   moduleOp.walk([&](subop::ExecutionStepOp step) {
      int stepIdx = static_cast<int>(idx++);
      a.stepByIndex[stepIdx] = step;

      bool tableRef = isExternalTableRefStep(step);
      bool createOnly = isCreateOnlyExecutionStep(step);

      // 0) Record decoded external datasource property for table_ref steps.
      if (tableRef) {
         auto& block = step.getSubOps().front();
         subop::GetExternalOp geOp;
         for (auto& op : block.without_terminator()) {
            if (auto g = mlir::dyn_cast<subop::GetExternalOp>(&op)) {
               geOp = g;
               break;
            }
         }
         assert(geOp && "table_ref step must contain get_external");
         assert(step.getNumResults() == 1 && "table_ref step must return one value");
         mlir::Value t = canonicalizeStateValueDeep(step.getResult(0));
         // Decode once and keep it for downstream rewrites (filter delaying etc.).
         a.reuse.externalDatasourceByTableState[t] =
            lingodb::utility::deserializeFromHexString<lingodb::runtime::ExternalDatasourceProperty>(geOp.getDescr());
      }

      // 1) Record creation/writes for step results (state values).
      for (auto r : step.getResults()) {
         if (!isStateType(r.getType()) && !isThreadLocalOfStateType(r.getType())) continue;
         auto& info = recordState(r);
         if (info.createdAt < 0) info.createdAt = stepIdx;
         a.createdAtByState[r] = info.createdAt;
         if (!createOnly && !tableRef) info.writes.insert(stepIdx);
      }

      // 2) Analyze read/write usage inside the step body.
      auto rw = analyzeStepStateRW(step);
      a.rwByStep[stepIdx] = rw;

      ModuleReuseInfo::StepRW re;
      re.step = step;

      for (auto it : rw) {
         auto& info = recordState(it.first);
         if (info.createdAt < 0 && isFuncBlockArgument(it.first)) {
            info.createdAt = -2; // external (func argument)
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

      // 3) Collect merge pairing in the same pass.
      auto& block = step.getSubOps().front();
      for (auto& op : block.without_terminator()) {
         auto merge = mlir::dyn_cast<subop::MergeOp>(op);
         if (!merge) continue;
         mlir::Value in = canonicalizeStateValueDeep(merge.getThreadLocal());
         mlir::Value out = canonicalizeStateValueDeep(merge.getResult());
         a.mergedFromThreadLocal[out] = in;
         a.reuse.mergedFromThreadLocal[out] = in;
      }

      // 4) Register create-only steps for any canonical state key they return.
      if (isCreateOnlyExecutionStep(step)) {
         llvm::SmallVector<mlir::Value, 4> keys;
         for (mlir::Value r : step.getResults()) if (r) keys.push_back(r);
         if (auto ret = mlir::dyn_cast<subop::ExecutionStepReturnOp>(block.getTerminator())) {
            for (mlir::Value o : ret.getOperands()) if (o) keys.push_back(o);
         }
         for (mlir::Value r : keys) {
            mlir::Value key = canonicalizeStateValueDeep(r);
            if (!a.reuse.createOnlyStepForState.contains(key)) a.reuse.createOnlyStepForState.insert({key, step});
            if (key != r && !a.reuse.createOnlyStepForState.contains(r)) a.reuse.createOnlyStepForState.insert({r, step});
         }
      }

      a.reuse.steps.push_back(std::move(re));
   });

   // Materialize writesByState and a stable sorted state list for matching.
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

   // Global buffer -> create_hash_indexed_view (at most one HIV per buffer). Optional TL merge on buffer.
   moduleOp.walk([&](subop::CreateHashIndexedView chiv) {
      mlir::Value globalBuf = canonicalizeStateValueDeep(chiv.getSource());
      mlir::Value hiv = canonicalizeStateValueDeep(chiv.getResult());
      if (!mlir::isa<subop::BufferType>(globalBuf.getType())) return;
      if (!mlir::isa<subop::HashIndexedViewType>(hiv.getType())) return;
      assert(!a.reuse.mergedBufferToHashIndexedView.contains(globalBuf) &&
             "each join buffer must feed at most one hash_indexed_view in a serial chain");
      a.reuse.hashIndexedViewFromMergedBuffer[hiv] = globalBuf;
      a.reuse.mergedBufferToHashIndexedView[globalBuf] = hiv;
   });

   return a;
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

   auto a = analyzeModuleForMatchAndReuse(moduleOp);

   llvm::DenseSet<mlir::Value> threadLocalMergePartners;
   for (auto& kv : a.mergedFromThreadLocal) threadLocalMergePartners.insert(kv.second);

   llvm::DenseSet<mlir::Value> bufferJoinChainBufferPartners;
   for (auto& kv : a.reuse.mergedBufferToHashIndexedView) bufferJoinChainBufferPartners.insert(kv.first);

   llvm::SmallVector<StateMatchProfile, 128> profiles;

   for (auto s : a.statesSorted) {
      // Only consider states that are constructed within this module (i.e., appear as execution_step results).
      // External states (func arguments) and any other untracked values must never be treated as "constructed".
      auto itCreated = a.createdAtByState.find(s);
      if (itCreated == a.createdAtByState.end() || itCreated->second < 0) {
         continue;
      }

      if (threadLocalMergePartners.contains(s)) continue;
      // Join chain is represented by the HIV; the merged global buffer is not a separate reuse target.
      if (bufferJoinChainBufferPartners.contains(s)) continue;

      auto stepIdxs = getConstructionStepIndicesForState(s, a.createdAtByState, a.writesByState, a.mergedFromThreadLocal,
                                                         &a.reuse.hashIndexedViewFromMergedBuffer);
      auto prereqStepIdxs = sortedUniqueStepIndices(stepIdxs);

      const bool isJoinHivRoot = a.reuse.hashIndexedViewFromMergedBuffer.contains(s);
      std::optional<JoinHivMatchDetails> joinHivDetails;
      if (auto hivTy = mlir::dyn_cast<subop::HashIndexedViewType>(s.getType())) {
         joinHivDetails =
            computeJoinHivMatchDetails(s, hivTy, memberManager, tupDialect->getColumnManager(), a.reuse.writerStepsByState);
         if (joinHivDetails) {
            a.reuse.joinBuildStoredValueMembersByState[s] = joinHivDetails->storedValueMembersFingerprint;
         }
      }

      llvm::SmallVector<uint64_t, 16> stepHashes;
      stepHashes.reserve(stepIdxs.size());
      for (int si : stepIdxs) {
         auto itS = a.stepByIndex.find(si);
         assert(itS != a.stepByIndex.end());
         StepDagHasher hasher;
         hasher.step = itS->second;
         hasher.tableDescrByTableState = &tableDescrByTableState;
         hasher.memberManager = &memberManager;
         hasher.columnManager = &tupDialect->getColumnManager();
         if (joinHivDetails) {
            hasher.relaxJoinPayloadColumns = true;
            hasher.joinIndexMemberNamesSanitized = &joinHivDetails->indexMemberNamesSanitized;
            hasher.joinKeyColumnAttrHashes = &joinHivDetails->joinKeyColumnAttrHashes;
            hasher.externalDatasourceByTableState = &a.reuse.externalDatasourceByTableState;
         }
         stepHashes.push_back(hasher.hashStepReturnGraph());
      }
      llvm::sort(stepHashes);
      uint64_t constructionHash = 0;
      for (auto h : stepHashes) constructionHash = hashCombineU64(constructionHash, h);

      auto prereqs = getPrereqStatesForConstructionSteps(prereqStepIdxs, s, a.rwByStep);
      llvm::DenseSet<mlir::Value> prereqSeen;
      for (auto pv : prereqs) prereqSeen.insert(pv);
      for (int si : prereqStepIdxs) {
         auto itSt = a.stepByIndex.find(si);
         assert(itSt != a.stepByIndex.end());
         appendNestedStateOperandsAsPrereqs(itSt->second, s, prereqSeen, prereqs);
      }

      llvm::SmallVector<std::string, 8> depTokens;
      llvm::DenseMap<mlir::Value, llvm::SmallVector<std::string, 8>> depMemo;
      llvm::DenseSet<mlir::Value> visiting;
      std::function<bool(mlir::Value, llvm::SmallVector<std::string, 8>&)> collectDeps =
         [&](mlir::Value st, llvm::SmallVector<std::string, 8>& out) -> bool {
            if (auto itM = depMemo.find(st); itM != depMemo.end()) {
               out.append(itM->second.begin(), itM->second.end());
               return true;
            }
            assert(visiting.insert(st).second && "state dep graph must be acyclic");

            llvm::SmallVector<std::string, 8> local;
            auto itC = a.createdAtByState.find(st);
            if (itC == a.createdAtByState.end()) return false;
            auto stHashSteps = getConstructionStepIndicesForState(st, a.createdAtByState, a.writesByState, a.mergedFromThreadLocal,
                                                                &a.reuse.hashIndexedViewFromMergedBuffer);
            auto stRwSteps = sortedUniqueStepIndices(stHashSteps);
            llvm::SmallVector<mlir::Value, 8> stPrereqs =
               getPrereqStatesForConstructionSteps(stRwSteps, st, a.rwByStep);
            llvm::DenseSet<mlir::Value> nestedSeen;
            for (auto pv : stPrereqs) nestedSeen.insert(pv);
            for (int si : stRwSteps) {
               auto itSt = a.stepByIndex.find(si);
               assert(itSt != a.stepByIndex.end());
               appendNestedStateOperandsAsPrereqs(itSt->second, st, nestedSeen, stPrereqs);
            }
            for (auto p : stPrereqs) {
               // Ignore pure create-only prerequisite states.
               if (auto itPC = a.createdAtByState.find(p); itPC != a.createdAtByState.end()) {
                  auto itPS = a.stepByIndex.find(itPC->second);
                  if (itPS != a.stepByIndex.end() && isCreateOnlyExecutionStep(itPS->second)) {
                     continue;
                  }
               }
               if (isTableStateValue(p)) {
                  if (isJoinHivRoot) {
                     std::string tableNameToken = tableNameDepToken(p, a.reuse.externalDatasourceByTableState);
                     if (!tableNameToken.empty()) {
                        local.push_back(std::move(tableNameToken));
                        continue;
                     }
                  }
                  auto itD = tableDescrByTableState.find(p);
                  if (itD != tableDescrByTableState.end()) {
                     local.push_back(std::string("table:") + itD->second);
                  } else {
                     // Some pipelines may introduce table-typed values that are not direct get_external results.
                     // Fall back to a stable, type-based token so identical shapes can still match cross-query.
                     local.push_back(std::string("table_type:") + normalizedSubopStateTypeFingerprint(memberManager, p.getType()));
                  }
                  continue;
               }
               if (mlir::isa<subop::ResultTableType>(p.getType())) {
                  // Always record the RT shape token for cross-query matching, but also expand
                  // transitive construction deps: a `result_table` may be created empty and later
                  // filled from a heap/buffer/etc.; ignoring that edge lets consumers (e.g. another
                  // RT or `create_from`) look "table-only eligible" while still depending on heap.
                  local.push_back(std::string("rt:") + normalizedSubopStateTypeFingerprint(memberManager, p.getType()));
                  if (!collectDeps(p, local)) return false;
                  continue;
               }
               if (threadLocalMergePartners.contains(p)) continue;
               // Merge **result** (global side): same logical construction unit as the TL operand. Downstream
               // states (e.g. `hash_indexed_view` built only from the merged buffer) must not treat this
               // global `State` as an opaque internal edge — fold to that value's own construction deps.
               if (a.mergedFromThreadLocal.contains(p)) {
                  if (!collectDeps(p, local)) return false;
                  continue;
               }
               // `hash_indexed_view` from a merged join buffer is the same construction unit as that buffer.
               if (auto itHiv = a.reuse.hashIndexedViewFromMergedBuffer.find(p);
                   itHiv != a.reuse.hashIndexedViewFromMergedBuffer.end()) {
                  mlir::Value mergedBuf = itHiv->second;
                  // When profiling the merged buffer itself, HIV is downstream — do not recurse back.
                  if (mergedBuf == st) continue;
                  if (visiting.contains(mergedBuf)) continue;
                  if (!collectDeps(mergedBuf, local)) return false;
                  continue;
               }
               // Cross-query reuse may only *match* on states whose construction deps are external tables
               // (descr / type token) and/or `result_table` fingerprints. Any other `!subop.*` state edge
               // (heap/buffer/hashmap/hash_indexed_view, …) is pipeline-internal and disqualifies reuse.
               if (isThreadLocalOfStateType(p.getType())) {
                  if (!collectDeps(p, local)) return false;
                  continue;
               }
               // Profiling `hash_indexed_view`: merged global buffer (+ TL) are the same reuse unit.
               if (auto itChain = a.reuse.hashIndexedViewFromMergedBuffer.find(st);
                   itChain != a.reuse.hashIndexedViewFromMergedBuffer.end()) {
                  if (p == itChain->second) continue;
                  if (auto itTL = a.mergedFromThreadLocal.find(itChain->second);
                      itTL != a.mergedFromThreadLocal.end() && p == itTL->second) {
                     continue;
                  }
               }
               if (mlir::isa<subop::State>(p.getType())) return false;
               llvm_unreachable("collectDeps: unexpected prereq value type");
            }

            visiting.erase(st);
            llvm::sort(local);
            local.erase(std::unique(local.begin(), local.end()), local.end());
            depMemo[st] = local;
            out.append(local.begin(), local.end());
            return true;
         };

      bool ok = true;
      for (auto p : prereqs) {
         if (!collectDeps(p, depTokens)) {
            ok = false;
            break;
         }
      }
      llvm::sort(depTokens);
      depTokens.erase(std::unique(depTokens.begin(), depTokens.end()), depTokens.end());

      // Hard constraint: if any construction step writes multiple states, this state must not participate in matching.
      // (We cannot safely delete only the "write side" without affecting other written states.)
      bool multiWrite = false;
      for (int si : stepIdxs) {
         auto itRW = a.rwByStep.find(si);
         assert(itRW != a.rwByStep.end());
         unsigned writeCount = 0;
         for (auto kv : itRW->second) {
            if (kv.second.write) writeCount++;
            if (writeCount > 1) break;
         }
         if (writeCount > 1) {
            multiWrite = true;
            break;
         }
      }

      // New constraint: if a state has no writer steps and is not a merge result (global <- thread_local),
      // do not reuse it. (These are typically pure create-only or view-like values that would require
      // special handling; this experimental project prefers to skip them.)
      bool hasWriter = a.reuse.writerStepsByState.contains(s);
      bool isMergeResult = a.mergedFromThreadLocal.contains(s);
      bool isJoinBufferHashViewRoot = a.reuse.hashIndexedViewFromMergedBuffer.contains(s);
      bool hasWriterOrIsMergeResult = hasWriter || isMergeResult || isJoinBufferHashViewRoot;

      StateMatchProfile prof;
      prof.queryId = queryId;
      prof.value = s;
      // If a state truly depends only on external tables, it may have an empty dep token set
      // when all intermediate prereqs are internal create-only states.
      // We still want to match it.
      prof.eligible = ok && !multiWrite && hasWriterOrIsMergeResult;
      prof.depTokensSorted.assign(depTokens.begin(), depTokens.end());
      prof.constructionStepHashes.assign(stepHashes.begin(), stepHashes.end());
      prof.constructionHash = constructionHash;
      if (joinHivDetails) {
         prof.typeFingerprintStr = normalizedHashIndexedViewTypeFingerprintForJoinMatch(
            memberManager, mlir::cast<subop::HashIndexedViewType>(s.getType()), *joinHivDetails);
         prof.storedValueMembersFingerprint = joinHivDetails->storedValueMembersFingerprint;
      } else {
         prof.typeFingerprintStr = normalizedSubopStateTypeFingerprint(memberManager, s.getType());
      }
      profiles.push_back(std::move(prof));
   }

   return profiles;
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
   llvm::DenseMap<mlir::Value, mlir::Value> mergedFromThreadLocal; // global -> thread_local

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
   moduleOp.walk([&](subop::ExecutionStepOp step) {
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

      // Record creation for internal state-typed SSA values inside step body.
      {
         auto& block = step.getSubOps().front();
         for (auto& op : block.without_terminator()) {
            for (auto r : op.getResults()) {
               auto t = r.getType();
               if (!isStateType(t) && !isThreadLocalOfStateType(t)) continue;
               auto& info = recordState(r);
               if (info.createdAt < 0) info.createdAt = stepIdx;
               createdAtByState[r] = info.createdAt;
            }
         }
      }

      // Record read/write usage of state operands inside this step.
      auto rw = analyzeStepStateRW(step);
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

   os << "\n// ==== thread_local -> merge -> global pairs ====\n";
   moduleOp.walk([&](subop::ExecutionStepOp step) {
      auto& block = step.getSubOps().front();
      for (auto& op : block.without_terminator()) {
         auto merge = mlir::dyn_cast<subop::MergeOp>(op);
         if (!merge) continue;
         assert(merge.getThreadLocal());
         mlir::Value in = canonicalizeStateValueDeep(merge.getThreadLocal());
         mlir::Value out = canonicalizeStateValueDeep(merge.getResult());
         mergedFromThreadLocal[out] = in;
         os << "// ";
         in.printAsOperand(os, flags);
         os << " -> ";
         out.printAsOperand(os, flags);
         os << "\n";
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

   llvm::DenseSet<mlir::Value> threadLocalMergePartners;
   for (auto& kv : mergedFromThreadLocal) threadLocalMergePartners.insert(kv.second);

   for (auto s : states) {
      if (threadLocalMergePartners.contains(s)) continue;

      auto stepIdxs = getConstructionStepIndicesForState(s, createdAtByState, writesByState, mergedFromThreadLocal, nullptr);
      auto prereqStepIdxs = sortedUniqueStepIndices(stepIdxs);
      os << "\n// -- state ";
      s.printAsOperand(os, flags);
      os << " : ";
      s.getType().print(os);
      os << " --\n";
      os << "//   construction_steps: ";
      for (int si : stepIdxs) os << si << " ";
      os << "\n";
      if (auto it = mergedFromThreadLocal.find(s); it != mergedFromThreadLocal.end()) {
         os << "//   merged_from_thread_local: ";
         it->second.printAsOperand(os, flags);
         os << "\n";
      }

      llvm::SmallVector<mlir::Value, 8> prereqs =
         getPrereqStatesForConstructionSteps(prereqStepIdxs, s, rwByStep);
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
   };

   llvm::SmallVector<QueryModel, 4> models;
   models.reserve(queries.size());
   for (auto& q : queries) {
      QueryModel m;
      m.id = q.first;
      m.module = q.second;
      m.tableDescr = buildTableDescrByTableState(m.module);
      m.profiles = buildStateMatchProfiles(m.id, m.module, m.tableDescr);
      models.push_back(std::move(m));
   }

   os << "\n// ==== cross-query state matches (experimental) ====\n";
   os << "// rules: construction-step read deps may only be external tables (matched by GetExternal descr)\n";
   os << "// and/or result_table values (matched by a member/type shape fingerprint that ignores `$id` suffixes).\n";
   os << "// States with any other prereq state are skipped.\n";
   os << "// Match key: sorted dep tokens + construction fingerprints + result type fingerprint.\n";
   os << "// Match key: sorted dep tokens + constructionHash + result type fingerprint.\n";

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
            os << " h=" << p.constructionHash;
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

   // Debug helper: print eligible optimistic_ht_fragment-like profiles per query (capped).
   {
      os << "\n// ==== debug: eligible optimistic_ht_fragment profiles (capped) ====\n";
      size_t cap = 20;
      for (auto& m : models) {
         size_t printedDbg = 0;
         for (auto& p : m.profiles) {
            std::string ty;
            llvm::raw_string_ostream tss(ty);
            p.value.getType().print(tss);
            tss.flush();
            if (ty.find("optimistic_ht_fragment") == std::string::npos) continue;

            std::string deps = joinSortedStrings(p.depTokensSorted);
            os << "//   query[" << m.id << "] ";
            mlir::OpPrintingFlags dbgFlags;
            p.value.printAsOperand(os, dbgFlags);
            os << " eligible=" << (p.eligible ? "true" : "false");
            os << " type=" << ty;
            os << " deps=" << deps;
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

   // O(n^2) pair enumeration. Time is not important; avoids building giant string keys.
   llvm::SmallVector<const StateMatchProfile*, 256> all;
   for (auto& m : models) {
      for (auto& p : m.profiles) {
         if (!p.eligible) continue;
         all.push_back(&p);
      }
   }

   mlir::OpPrintingFlags flags;
   size_t printed = 0;
   for (size_t i = 0; i < all.size(); i++) {
      for (size_t j = i + 1; j < all.size(); j++) {
         auto* a = all[i];
         auto* b = all[j];
         if (a->queryId == b->queryId) continue;
         if (a->constructionHash != b->constructionHash) continue;
         if (a->typeFingerprintStr != b->typeFingerprintStr) continue;
         if (a->depTokensSorted != b->depTokensSorted) continue;

         os << "\n// -- match_pair --\n";
         os << "//   query[" << a->queryId << "] ";
         a->value.printAsOperand(os, flags);
         os << "\n";
         os << "//   query[" << b->queryId << "] ";
         b->value.printAsOperand(os, flags);
         os << "\n";
         printed++;
         if (printed >= 100) {
            os << "\n// ... truncated match_pairs (max 100) ...\n";
            return;
         }
      }
   }
   if (printed == 0) os << "// (no matches)\n";
}

ModuleReuseInfo collectModuleReuseInfo(mlir::ModuleOp moduleOp) {
   return analyzeModuleForMatchAndReuse(moduleOp).reuse;
}

llvm::SmallVector<CrossQueryStateMatchPair, 64>
collectCrossQueryStateMatchPairs(llvm::ArrayRef<std::pair<int, mlir::ModuleOp>> queries) {
   assert(queries.size() >= 2 && "need at least two modules to compare");

   struct QueryModel {
      int id = -1;
      mlir::ModuleOp module;
      llvm::DenseMap<mlir::Value, std::string> tableDescr;
      llvm::SmallVector<StateMatchProfile, 128> profiles;
   };

   llvm::SmallVector<QueryModel, 4> models;
   models.reserve(queries.size());
   for (auto& q : queries) {
      QueryModel m;
      m.id = q.first;
      m.module = q.second;
      m.tableDescr = buildTableDescrByTableState(m.module);
      m.profiles = buildStateMatchProfiles(m.id, m.module, m.tableDescr);
      models.push_back(std::move(m));
   }

   llvm::SmallVector<const StateMatchProfile*, 256> all;
   for (auto& m : models) {
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

   llvm::SmallVector<CrossQueryStateMatchPair, 64> out;
   for (size_t i = 0; i < all.size(); i++) {
      for (size_t j = i + 1; j < all.size(); j++) {
         auto* a = all[i];
         auto* b = all[j];
         if (a->queryId == b->queryId) continue;
         if (a->constructionHash != b->constructionHash) continue;
         if (a->typeFingerprintStr != b->typeFingerprintStr) continue;
         if (a->depTokensSorted != b->depTokensSorted) continue;

         std::string k = makeKeyStr(*a);
         uint64_t cacheKey = static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(k)));

         CrossQueryStateMatchPair p;
         p.queryA = a->queryId;
         p.queryB = b->queryId;
         p.stateA = a->value;
         p.stateB = b->value;
         p.cacheKey = cacheKey;
         out.push_back(p);
      }
   }
   return out;
}

mlir::Value bufferJoinChainRootForReuse(mlir::Value v, const ModuleReuseInfo& reuse) {
   v = canonicalizeStateValueForReuse(v);
   if (auto it = reuse.mergedBufferToHashIndexedView.find(v); it != reuse.mergedBufferToHashIndexedView.end()) {
      return it->second;
   }
   return v;
}

mlir::Value resolveCacheTargetStateForReuse(mlir::Value v, const ModuleReuseInfo& reuse) {
   return bufferJoinChainRootForReuse(v, reuse);
}

bool isBufferJoinChainNonRootPartner(mlir::Value v, const ModuleReuseInfo& reuse) {
   v = canonicalizeStateValueForReuse(v);
   if (reuse.mergedBufferToHashIndexedView.contains(v)) return true;
   if (auto itG = reuse.hashIndexedViewFromMergedBuffer.find(v); itG != reuse.hashIndexedViewFromMergedBuffer.end()) {
      if (auto itTL = reuse.mergedFromThreadLocal.find(itG->second); itTL != reuse.mergedFromThreadLocal.end()) {
         if (itTL->second == v) return true;
      }
   }
   return false;
}

void forEachBufferJoinChainPartner(mlir::Value chainRootBuffer, const ModuleReuseInfo& reuse,
                                   llvm::function_ref<void(mlir::Value)> fn) {
   mlir::Value root = bufferJoinChainRootForReuse(chainRootBuffer, reuse);
   if (!mlir::isa<subop::BufferType>(root.getType())) return;
   if (mlir::isa<subop::HashIndexedViewType>(root.getType())) {
      fn(root);
      if (auto itG = reuse.hashIndexedViewFromMergedBuffer.find(root); itG != reuse.hashIndexedViewFromMergedBuffer.end()) {
         fn(itG->second);
         if (auto itTL = reuse.mergedFromThreadLocal.find(itG->second); itTL != reuse.mergedFromThreadLocal.end()) {
            fn(itTL->second);
         }
      }
      return;
   }
   if (!mlir::isa<subop::BufferType>(root.getType())) return;
   if (auto itH = reuse.mergedBufferToHashIndexedView.find(root); itH != reuse.mergedBufferToHashIndexedView.end()) {
      fn(itH->second);
      fn(root);
      if (auto itTL = reuse.mergedFromThreadLocal.find(root); itTL != reuse.mergedFromThreadLocal.end()) {
         fn(itTL->second);
      }
      return;
   }
   fn(root);
}

} // namespace lingodb::compiler::dialect::subop

