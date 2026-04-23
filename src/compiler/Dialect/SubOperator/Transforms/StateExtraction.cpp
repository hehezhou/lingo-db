#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateExtraction.h"

#include "mlir/IR/BuiltinOps.h"

#include <cassert>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/raw_ostream.h>

namespace lingodb::compiler::dialect::subop {

namespace {

bool isCreateLike(mlir::Operation& op) {
   return mlir::isa<
      subop::CreateThreadLocalOp,
      subop::CreateHeapOp,
      subop::GenericCreateOp>(op);
}

bool isStateType(mlir::Type t) {
   return mlir::isa<subop::State>(t);
}

bool isThreadLocalOfStateType(mlir::Type t) {
   auto tl = mlir::dyn_cast_or_null<subop::ThreadLocalType>(t);
   return tl && isStateType(tl.getWrapped());
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
      assert(0);
   }
   auto* owner = ba.getOwner();
   assert(owner && "block argument must have an owner block");
   assert(owner == &step.getSubOps().front() && "block argument must belong to execution_step body block");
   auto inputs = step.getInputs();
   auto idx = ba.getArgNumber();
   assert(idx < inputs.size() && "block argument index must be within execution_step inputs");
   return inputs[idx];
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

void addStateOperands(mlir::Operation& op, llvm::SmallVectorImpl<mlir::Value>& out) {
   for (auto v : op.getOperands()) {
      auto t = v.getType();
      if (isStateType(t) || isThreadLocalOfStateType(t)) {
         out.push_back(v);
      }
   }
}

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

      auto readMembers = sub.getReadMembers();
      auto writtenMembers = sub.getWrittenMembers();

      llvm::SmallVector<mlir::Value, 4> stateOperands;
      addStateOperands(op, stateOperands);
      if (stateOperands.empty() && readMembers.empty() && writtenMembers.empty()) {
         continue;
      }

      llvm::SmallVector<mlir::Value, 4> stateResults;
      for (auto r : op.getResults()) {
         auto t = r.getType();
         if (isStateType(t) || isThreadLocalOfStateType(t)) {
            stateResults.push_back(r);
         }
      }

      for (auto s : stateOperands) {
         auto key = canonicalizeStateValue(step, s);
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
   const llvm::DenseMap<mlir::Value, mlir::Value>& mergedFromThreadLocal) {
   llvm::SmallVector<int, 8> steps;

   // Normal state: createdAt (+ first real write).
   addConstructionStepsForSingleState(state, createdAtByState, writesByState, steps);

   // Merge-produced global state: include thread_local side construction steps as well,
   // but do NOT treat the thread_local as shareable output by itself.
   if (auto it = mergedFromThreadLocal.find(state); it != mergedFromThreadLocal.end()) {
      mlir::Value tl = it->second;
      addConstructionStepsForSingleState(tl, createdAtByState, writesByState, steps);
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
   for (auto& it : stateInfo) {
      os << "// ";
      it.first.printAsOperand(os, flags);
      os << " : ";
      it.first.getType().print(os);
      os << "\n";
      os << "//   created_at: " << it.second.createdAt << "\n";
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
         mlir::Value in = canonicalizeStateValue(step, merge.getThreadLocal());
         mlir::Value out = canonicalizeStateValue(step, merge.getResult());
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

   for (auto s : states) {
      // If this is a thread_local that is merged into some global state, don't print it separately.
      // The merged global state's construction will include the thread_local side steps.
      bool isThreadLocalMergedAway = false;
      for (auto& kv : mergedFromThreadLocal) {
         if (kv.second == s) {
            isThreadLocalMergedAway = true;
            break;
         }
      }
      if (isThreadLocalMergedAway) {
         continue;
      }

      auto stepIdxs = getConstructionStepIndicesForState(s, createdAtByState, writesByState, mergedFromThreadLocal);
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

      auto prereqs = getPrereqStatesForConstructionSteps(stepIdxs, s, rwByStep);
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
}

} // namespace lingodb::compiler::dialect::subop

