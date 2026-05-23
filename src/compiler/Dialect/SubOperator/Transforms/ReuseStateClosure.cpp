#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseRewriteCommon.h"
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

bool opaqueClosureContains(const llvm::DenseSet<void*>& closure, mlir::Value v) {
   return v && closure.contains(v.getAsOpaquePointer());
}

/// Link `execution_step` operands, body block arguments, and `execution_step_return` operands
/// with step results whenever either side is already in \p closure (fixpoint over the join
/// buffer → HIV SSA region).
void expandClosureThroughExecutionStepPorts(mlir::ModuleOp module, llvm::DenseSet<void*>& closure) {
   for (;;) {
      size_t before = closure.size();
      module.walk([&](subop::ExecutionStepOp step) {
         mlir::Block& body = step.getSubOps().front();
         for (unsigned i = 0; i < step.getNumOperands() && i < body.getNumArguments(); ++i) {
            mlir::Value opnd = step.getOperand(i);
            mlir::Value barg = body.getArgument(i);
            if (opaqueClosureContains(closure, opnd)) closure.insert(barg.getAsOpaquePointer());
            if (opaqueClosureContains(closure, barg)) closure.insert(opnd.getAsOpaquePointer());
         }
      });
      module.walk([&](subop::ExecutionStepReturnOp ret) {
         auto step = mlir::dyn_cast<subop::ExecutionStepOp>(ret->getParentOp());
         if (!step) return;
         for (unsigned i = 0; i < ret.getNumOperands() && i < step.getNumResults(); ++i) {
            mlir::Value rv = ret.getOperand(i);
            mlir::Value sr = step.getResult(i);
            if (opaqueClosureContains(closure, rv)) closure.insert(sr.getAsOpaquePointer());
            if (opaqueClosureContains(closure, sr)) closure.insert(rv.getAsOpaquePointer());
         }
      });
      if (closure.size() == before) break;
   }
}

void expandClosureThroughNestedExecutionGroupPorts(mlir::ModuleOp module, llvm::DenseSet<void*>& closure) {
   for (;;) {
      size_t before = closure.size();
      module.walk([&](subop::NestedExecutionGroupOp neg) {
         mlir::Block& body = neg.getSubOps().front();
         for (unsigned i = 0; i < neg.getNumOperands() && i < body.getNumArguments(); ++i) {
            mlir::Value opnd = neg.getOperand(i);
            mlir::Value barg = body.getArgument(i);
            if (opaqueClosureContains(closure, opnd)) closure.insert(barg.getAsOpaquePointer());
            if (opaqueClosureContains(closure, barg)) closure.insert(opnd.getAsOpaquePointer());
         }
      });
      if (closure.size() == before) break;
   }
}

bool executionStepTouchesClosure(subop::ExecutionStepOp step, const llvm::DenseSet<void*>& closure) {
   for (mlir::Value v : step.getOperands()) {
      if (opaqueClosureContains(closure, v)) return true;
   }
   for (mlir::OpResult r : step.getResults()) {
      if (opaqueClosureContains(closure, r)) return true;
   }
   for (mlir::BlockArgument a : step.getSubOps().front().getArguments()) {
      if (opaqueClosureContains(closure, a)) return true;
   }
   return false;
}

bool opOperandsOrNestedBlockArgsTouchClosure(mlir::Operation* op, const llvm::DenseSet<void*>& closure) {
   for (mlir::Value v : op->getOperands()) {
      if (opaqueClosureContains(closure, v)) return true;
   }
   for (mlir::Region& reg : op->getRegions()) {
      for (mlir::Block& b : reg) {
         for (mlir::BlockArgument a : b.getArguments()) {
            if (opaqueClosureContains(closure, a)) return true;
         }
      }
   }
   return false;
}

/// Block argument type for an execution_step operand when `is_thread_local` is false but the
/// operand is still `!subop.thread_local<state>` (nested probe steps keep the unwrapped state type).
static mlir::Type executionStepBodyArgTypeForOperand(mlir::Value operand, mlir::Attribute isThreadLocalAttr) {
   if (mlir::cast<mlir::BoolAttr>(isThreadLocalAttr).getValue()) return operand.getType();
   if (auto tl = mlir::dyn_cast<subop::ThreadLocalType>(operand.getType())) return tl.getWrapped();
   return operand.getType();
}

/// Keep `execution_step` result types and body entry types aligned with `execution_step_return`
/// and step operands. When \p closureFilter is non-null, only touch steps that reach the join
/// buffer / HIV closure (derived from `computeJoinBufferHivSsaClosure`).
void synchronizeExecutionStepPortTypes(mlir::ModuleOp module, const llvm::DenseSet<void*>* closureFilter) {
   for (;;) {
      bool changed = false;
      module.walk([&](subop::ExecutionStepReturnOp ret) {
         auto* parent = ret->getParentOp();
         auto step = mlir::dyn_cast<subop::ExecutionStepOp>(parent);
         if (!step || step.getNumResults() != ret.getNumOperands()) return;
         if (closureFilter && !executionStepTouchesClosure(step, *closureFilter)) return;
         for (unsigned i = 0; i < step.getNumResults(); ++i) {
            mlir::Type t = ret.getOperand(i).getType();
            if (t != step.getResult(i).getType()) {
               step.getResult(i).setType(t);
               changed = true;
            }
         }
      });
      module.walk([&](subop::ExecutionStepOp step) {
         if (closureFilter && !executionStepTouchesClosure(step, *closureFilter)) return;
         mlir::Block& body = step.getSubOps().front();
         bool nestedProbeStep = step->getParentOfType<subop::NestedExecutionGroupOp>() != nullptr;
         auto tlsFlags = step.getIsThreadLocal();
         for (unsigned i = 0; i < step.getNumOperands() && i < body.getNumArguments(); ++i) {
            mlir::Type wt = step.getOperand(i).getType();
            if (nestedProbeStep) {
               mlir::Attribute tlsAttr =
                  i < tlsFlags.size() ? tlsFlags[i] : mlir::BoolAttr::get(step.getContext(), false);
               wt = executionStepBodyArgTypeForOperand(step.getOperand(i), tlsAttr);
            }
            if (body.getArgument(i).getType() != wt) {
               body.getArgument(i).setType(wt);
               changed = true;
            }
         }
      });
      if (!changed) break;
   }
}

static bool isJoinPredCarrierType(mlir::Type t) {
   if (mlir::isa<subop::BufferType>(t)) return true;
   if (mlir::isa<subop::HashIndexedViewType>(t)) return true;
   if (auto tl = mlir::dyn_cast<subop::ThreadLocalType>(t))
      return mlir::isa<subop::BufferType>(tl.getWrapped());
   return false;
}

/// SSA closure for one or more reuse join buffers and their derived `hash_indexed_view` values:
/// forward uses (incl. `merge`, `execution_step` region entry), backward across step boundaries and
/// `merge` results, and limited "any state-like result of an op that uses v".
JoinBufferHivSsaClosure computeJoinBufferHivSsaClosure(llvm::ArrayRef<mlir::Value> canonicalBuffers,
                                                       const ModuleReuseInfo& reuse) {
   JoinBufferHivSsaClosure result;
   llvm::DenseSet<void*>& outClosure = result.opaque;
   llvm::SmallVector<mlir::Value, 64>& outList = result.values;
   auto push = [&](mlir::Value x) {
      if (!x || !isJoinPredCarrierType(x.getType())) return;
      void* k = x.getAsOpaquePointer();
      if (!outClosure.insert(k).second) return;
      outList.push_back(x);
   };

   for (mlir::Value root : canonicalBuffers) {
      llvm::SmallVector<mlir::Value, 8> seeds;
      collectJoinBufferReachabilitySeeds(root, reuse, seeds);
      for (mlir::Value s : seeds) push(s);
   }

   for (size_t qi = 0; qi < outList.size(); ++qi) {
      mlir::Value v = outList[qi];

      if (auto* def = v.getDefiningOp()) {
         if (auto merge = mlir::dyn_cast<subop::MergeOp>(def)) {
            push(merge.getThreadLocal());
         }
      }

      if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(v)) {
         mlir::Block* owner = ba.getOwner();
         if (owner) {
            if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(owner->getParentOp())) {
               if (&step.getSubOps().front() == owner) {
                  unsigned idx = ba.getArgNumber();
                  if (idx < step.getNumOperands()) push(step.getOperand(idx));
               }
            }
         }
      }

      if (auto* def = v.getDefiningOp()) {
         if (auto ret = mlir::dyn_cast<subop::ExecutionStepReturnOp>(def)) {
            if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(ret->getParentOp())) {
               for (unsigned i = 0; i < ret.getNumOperands(); ++i) {
                  if (ret.getOperand(i) == v && i < step.getNumResults()) push(step.getResult(i));
               }
            }
         }
      }

      for (mlir::OpOperand& use : v.getUses()) {
         mlir::Operation* op = use.getOwner();
         if (mlir::isa<subop::BufferType>(v.getType())) {
            if (auto chiv = mlir::dyn_cast<subop::CreateHashIndexedView>(op)) {
               if (chiv.getSource() == v) push(chiv.getResult());
            }
         }
         if (auto merge = mlir::dyn_cast<subop::MergeOp>(op)) {
            if (merge.getThreadLocal() == v) push(merge.getResult());
         }
         if (mlir::Value inner = mapStateThroughExecutionStepOperands(v, op)) push(inner);

         bool usesV = false;
         for (mlir::Value ov : op->getOperands()) {
            if (ov == v) {
               usesV = true;
               break;
            }
         }
         if (usesV) {
            for (mlir::OpResult res : op->getResults()) {
               if (isJoinPredCarrierType(res.getType())) push(res);
            }
         }
      }
   }
   return result;
}

mlir::Value mapStateThroughExecutionStepOperands(mlir::Value v, mlir::Operation* user) {
   for (mlir::Operation* p = user; p; p = p->getParentOp()) {
      auto step = mlir::dyn_cast<subop::ExecutionStepOp>(p);
      if (!step) continue;
      for (unsigned i = 0; i < step.getNumOperands(); ++i) {
         if (step.getOperand(i) != v) continue;
         mlir::Region& r = step.getSubOps();
         if (r.empty()) continue;
         mlir::Block& body = r.front();
         if (i < body.getNumArguments()) return body.getArgument(i);
      }
   }
   return {};
}

/// SSA seeds for tracing a join build buffer to `create_hash_indexed_view`: the merge-result buffer,
/// its paired `thread_local` writer state, and any buffer/thread_local values recorded as **written**
/// on writer steps for those keys (`ModuleReuseInfo` / `analyzeStepStateRW`).
void collectJoinBufferReachabilitySeeds(mlir::Value canonicalMergedBuffer, const ModuleReuseInfo& reuse,
                                               llvm::SmallVectorImpl<mlir::Value>& seeds) {
   llvm::DenseSet<void*> seen;
   auto push = [&](mlir::Value v) {
      if (!v) return;
      if (!mlir::isa<subop::BufferType, subop::ThreadLocalType>(v.getType())) return;
      if (!seen.insert(v.getAsOpaquePointer()).second) return;
      seeds.push_back(v);
   };

   push(canonicalMergedBuffer);
   if (auto it = findReuseMap(reuse.mergedFromShadowState, canonicalMergedBuffer);
       it != reuse.mergedFromShadowState.end()) {
      push(it->second);
   }

   llvm::DenseMap<mlir::Operation*, const ModuleReuseInfo::StepRW*> rwByStepOp;
   for (const auto& e : reuse.steps) {
      // `reuse` is const; `ExecutionStepOp::getOperation()` is non-const in this MLIR build.
      rwByStepOp[const_cast<subop::ExecutionStepOp&>(e.step).getOperation()] = &e;
   }
   auto pushWriterBuffers = [&](mlir::Value key) {
      auto itW = findReuseMap(reuse.writerStepsByState, key);
      if (itW == reuse.writerStepsByState.end()) return;
      for (ExecutionStepOp ws : itW->second) {
         const ModuleReuseInfo::StepRW* rw = rwByStepOp.lookup(ws.getOperation());
         if (!rw) continue;
         for (mlir::Value w : rw->writes) {
            push(w);
         }
      }
   };
   pushWriterBuffers(canonicalMergedBuffer);
   if (auto it = findReuseMap(reuse.mergedFromShadowState, canonicalMergedBuffer);
       it != reuse.mergedFromShadowState.end()) {
      pushWriterBuffers(it->second);
   }
}

llvm::DenseSet<mlir::Value> expandNeededStatesFromTargets(llvm::ArrayRef<CacheTarget> targets0,
                                                                 const ModuleReuseInfo& reuse) {
   auto rwByStepOp = buildRwByStepOpMap(reuse);

   llvm::DenseSet<mlir::Value> needed;
   llvm::SmallVector<mlir::Value, 32> queue;
   auto enqueue = [&](mlir::Value v) {
      if (!isPipelineStateValue(v)) return;
      if (needed.insert(v).second) queue.push_back(v);
   };

   for (auto& t : targets0) {
      enqueue(t.state);
      auto it = findReuseMap(reuse.mergedFromShadowState, t.state);
      if (it != reuse.mergedFromShadowState.end()) {
         enqueue(it->second);
      }
   }

   for (size_t qi = 0; qi < queue.size(); ++qi) {
      mlir::Value s = queue[qi];

      if (auto itShadow = findReuseMap(reuse.mergedFromShadowState, s);
          itShadow != reuse.mergedFromShadowState.end()) {
         enqueue(itShadow->second);
      }
      if (mlir::Value hiv = hashIndexedViewShadowingBuffer(s, reuse)) {
         enqueue(hiv);
      }

      auto itW = findReuseMap(reuse.writerStepsByState, s);
      if (itW == reuse.writerStepsByState.end()) continue;
      for (ExecutionStepOp w : itW->second) {
         const ModuleReuseInfo::StepRW* rw = rwByStepOp.lookup(w.getOperation());
         for (mlir::Value r : rw->reads) {
            enqueue(r);
         }
      }
   }
   return needed;
}


llvm::DenseSet<void*> closureLiveStatesFromReuseReturnSeeds(
   llvm::ArrayRef<mlir::Value> returnPipelineSeeds, const ModuleReuseInfo& reuse,
   const llvm::DenseMap<mlir::Operation*, const ModuleReuseInfo::StepRW*>& rwByStepOp) {
   llvm::DenseSet<void*> live;
   llvm::SmallVector<mlir::Value, 64> queue;

   auto enqueue = [&](mlir::Value v) {
      v = peelBlockArgsToEnclosingOperands(v);
      if (!isPipelineStateValue(v)) return;
      mlir::Value c = canonicalizeStateValueForReuse(v);
      void* k = c.getAsOpaquePointer();
      if (!live.insert(k).second) return;
      queue.push_back(c);
   };

   for (mlir::Value s : returnPipelineSeeds) enqueue(s);

   auto walkProducerStep = [&](ExecutionStepOp step) {
      if (const ModuleReuseInfo::StepRW* rw = rwByStepOp.lookup(step.getOperation())) {
         for (mlir::Value r : rw->reads) enqueue(r);
      }
      for (mlir::Value opnd : step.getOperands()) enqueue(opnd);
   };

   for (size_t qi = 0; qi < queue.size(); ++qi) {
      mlir::Value v = queue[qi];
      if (auto itW = findReuseMap(reuse.writerStepsByState, v); itW != reuse.writerStepsByState.end()) {
         for (ExecutionStepOp w : itW->second) walkProducerStep(w);
      }
      if (auto itC = findReuseMap(reuse.createOnlyStepForState, v); itC != reuse.createOnlyStepForState.end()) {
         walkProducerStep(itC->second);
      }
   }
   return live;
}

} // namespace lingodb::compiler::dialect::subop
