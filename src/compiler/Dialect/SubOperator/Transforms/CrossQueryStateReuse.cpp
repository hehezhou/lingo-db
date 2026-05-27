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

namespace lingodb::compiler::dialect::subop {
namespace {

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

   // HIV probe `filter_pred` filters are inserted after consumer layout align via
   // `applyProbePredFiltersForConsumerClosures` (see `rewritePlansWithSyntheticQuery0`).
   (void)isJoinBuf;
   (void)isJoinHiv;
   (void)joinBufPredProbeInjectedGroups;
   (void)predMember;
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
      mlir::Operation* insertAfter = st.getDefiningOp();
      if (auto itW = reuse.writerStepsByState.find(st); itW != reuse.writerStepsByState.end()) {
         for (auto s : itW->second) {
            auto* op = s.getOperation();
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
      mlir::OpBuilder builder(stateTy.getContext());
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
   llvm::ArrayRef<CrossQueryStateMatchPair> matches) {
   ReusePlanRewriteResult res;

   llvm::SmallVector<CrossQueryStateMatchPair, 64> matchesLocal(matches.begin(), matches.end());

   llvm::SmallVector<CacheTarget, 64> targets0;
   llvm::SmallVector<CacheTarget, 64> targets1;
   targets0.reserve(matchesLocal.size());
   targets1.reserve(matchesLocal.size());

   auto reuse0Early = collectModuleReuseInfo(query0);
   auto reuse1Early = collectModuleReuseInfo(query1);

   for (auto& m : matchesLocal) {
      bool enableFilterPredReuse = true;
      if (m.stateA && m.stateB) {
         mlir::Value hivA = resolveCacheTargetStateForReuse(m.stateA, reuse0Early);
         mlir::Value hivB = resolveCacheTargetStateForReuse(m.stateB, reuse1Early);
         if (mlir::isa<subop::HashIndexedViewType>(hivA.getType()) &&
             mlir::isa<subop::HashIndexedViewType>(hivB.getType()) &&
             joinMatchPeerExternalFiltersIdentical(query0, query1, m.stateA, m.stateB, reuse0Early, reuse1Early)) {
            enableFilterPredReuse = false;
         }
      }
      m.enableFilterPredReuse = enableFilterPredReuse;

      if (m.stateA) {
         mlir::Value ta = resolveCacheTargetStateForReuse(m.stateA, reuse0Early);
         assert(!mlir::isa<ThreadLocalType>(ta.getType()) &&
                "match pairs must never target thread_local-wrapped states");
         targets0.push_back(CacheTarget{ta, m.cacheKey, enableFilterPredReuse});
      }
      if (m.stateB) {
         mlir::Value tb = resolveCacheTargetStateForReuse(m.stateB, reuse1Early);
         assert(!mlir::isa<ThreadLocalType>(tb.getType()) &&
                "match pairs must never target thread_local-wrapped states");
         targets1.push_back(CacheTarget{tb, m.cacheKey, enableFilterPredReuse});
      }
   }

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

   auto reuseSynthetic = collectModuleReuseInfo(*res.query0);
   ClonedJoinBufferBuildSitesByKey joinBuildSites =
      recordClonedJoinBufferBuildSites(*res.query0, targetsQ0, reuseSynthetic);

   CachedJoinBufferLayoutsByKey producerLayoutsByKey;
   extendSyntheticJoinBuffersToColumnUnion(*res.query0, query0, query1, matchesLocal, targetsQ0, mapping,
                                           &producerLayoutsByKey);
   insertSyntheticFilterPredsAfterColumnUnion(*res.query0, query0, query1, matchesLocal, targetsQ0,
                                              producerLayoutsByKey, joinBuildSites);

   // Producer: cache_puts (cloned synthetic IR — needs its own reuse snapshot).
   {
      auto reuseSynthetic = collectModuleReuseInfo(*res.query0);
      insertCachePutsForTargets(*res.query0, targetsQ0, &reuseSynthetic);
      refreshCachedJoinLayoutsFromSyntheticCachePuts(*res.query0, targetsQ0, producerLayoutsByKey);
   }

   injectCacheGetsAndDeleteConstructionSteps(query0, targets0, &reuse0, /*joinBufferHashmapLayoutAlreadyApplied=*/true,
                                           /*joinBufferWritePredAlreadyApplied=*/true);
   injectCacheGetsAndDeleteConstructionSteps(query1, targets1, &reuse1, /*joinBufferHashmapLayoutAlreadyApplied=*/true,
                                           /*joinBufferWritePredAlreadyApplied=*/true);

   llvm::SmallVector<ConsumerCacheGetProbeClosure, 4> probeClosuresQ0;
   llvm::SmallVector<ConsumerCacheGetProbeClosure, 4> probeClosuresQ1;

   for (auto& t : targets0) {
      if (auto it = producerLayoutsByKey.find(t.cacheKey); it != producerLayoutsByKey.end()) {
         std::optional<unsigned> consumerQ =
            t.enableFilterPredReuse ? std::optional<unsigned>(0u) : std::nullopt;
         alignConsumerModulesToCachedJoinLayout(query0, it->second, t.cacheKey, consumerQ, &probeClosuresQ0);
      }
   }
   for (auto& t : targets1) {
      if (auto it = producerLayoutsByKey.find(t.cacheKey); it != producerLayoutsByKey.end()) {
         std::optional<unsigned> consumerQ =
            t.enableFilterPredReuse ? std::optional<unsigned>(1u) : std::nullopt;
         alignConsumerModulesToCachedJoinLayout(query1, it->second, t.cacheKey, consumerQ, &probeClosuresQ1);
      }
   }

   applyProbePredFiltersForConsumerClosures(query0, probeClosuresQ0);
   applyProbePredFiltersForConsumerClosures(query1, probeClosuresQ1);
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

   return res;
}

} // namespace lingodb::compiler::dialect::subop

