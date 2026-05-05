#include "lingodb/compiler/Dialect/SubOperator/Transforms/CrossQueryStateReuse.h"

#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <string>

namespace lingodb::compiler::dialect::subop {
namespace {

struct CacheTarget {
   mlir::Value state;
   uint64_t cacheKey;
};

static bool isPipelineStateValue(mlir::Value v) {
   mlir::Type t = v.getType();
   if (mlir::isa<subop::State>(t)) return true;
   auto tl = mlir::dyn_cast<subop::ThreadLocalType>(t);
   return tl && mlir::isa<subop::State>(tl.getWrapped());
}

/// `ModuleReuseInfo` maps are keyed by values produced inside `collectModuleReuseInfo`, which uses
/// `canonicalizeStateValueForReuse` for merge pairing and writer/create registration. Match that
/// here when `neededStates` / targets carry a different SSA view of the same state.
template <typename MapT>
static auto findReuseMap(const MapT& m, mlir::Value st) -> decltype(m.find(st)) {
   mlir::Value c = canonicalizeStateValueForReuse(st);
   auto it = m.find(c);
   if (it != m.end()) return it;
   if (c != st) return m.find(st);
   return m.end();
}

ExecutionGroupOp getSingleExecutionGroup(mlir::ModuleOp module) {
   llvm::SmallVector<ExecutionGroupOp, 4> groups;
   module.walk([&](ExecutionGroupOp g) { groups.push_back(g); });
   assert(groups.size() == 1 && "expected exactly one execution_group per module");
   return groups.front();
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

/// BFS over states: start from matched `targets0` states (plus paired thread_local from
/// `mergedFromThreadLocal`), then for every state S pull in any state read by a writer of S
/// (`ModuleReuseInfo::StepRW::reads`). This matches how construction depends on prerequisite
/// states without relying on SSA `getDefiningOp` alone.
static llvm::DenseSet<mlir::Value> expandNeededStatesFromTargets(llvm::ArrayRef<CacheTarget> targets0,
                                                                 const ModuleReuseInfo& reuse) {
   llvm::DenseMap<mlir::Operation*, const ModuleReuseInfo::StepRW*> rwByStepOp;
   for (const auto& e : reuse.steps) {
      ExecutionStepOp stepOp = e.step;
      rwByStepOp[stepOp.getOperation()] = &e;
   }

   llvm::DenseSet<mlir::Value> needed;
   llvm::SmallVector<mlir::Value, 32> queue;
   auto enqueue = [&](mlir::Value v) {
      if (!isPipelineStateValue(v)) return;
      if (needed.insert(v).second) queue.push_back(v);
   };

   for (auto& t : targets0) {
      enqueue(t.state);
      auto it = findReuseMap(reuse.mergedFromThreadLocal, t.state);
      if (it != reuse.mergedFromThreadLocal.end()) {
         enqueue(it->second);
      }
   }

   for (size_t qi = 0; qi < queue.size(); ++qi) {
      mlir::Value s = queue[qi];
      auto itW = findReuseMap(reuse.writerStepsByState, s);
      if (itW == reuse.writerStepsByState.end()) continue;
      for (ExecutionStepOp w : itW->second) {
         const ModuleReuseInfo::StepRW* rw = rwByStepOp.lookup(w.getOperation());
         for (mlir::Value r : rw->reads) {
            enqueue(r);
         }
      }
      if (auto itM = findReuseMap(reuse.mergedFromThreadLocal, s);
          itM != reuse.mergedFromThreadLocal.end()) {
         enqueue(itM->second);
      }
   }
   return needed;
}

/// For every needed state, include:
/// - each top-level `execution_step` listed in `writerStepsByState` (all writes / updates), and
/// - the top-level `execution_step` that **defines** the state SSA result, if any (create path).
/// - the create-only `execution_step` registered in `ModuleReuseInfo::createOnlyStepForState`, if any.
static llvm::SmallVector<ExecutionStepOp, 32>
collectCreateAndWriteStepsForStates(ExecutionGroupOp donor, const ModuleReuseInfo& reuse,
                                    const llvm::DenseSet<mlir::Value>& neededStates) {
   llvm::DenseSet<mlir::Operation*> seen;
   llvm::SmallVector<ExecutionStepOp, 32> out;
   for (mlir::Value s : neededStates) {
      assert(canonicalizeStateValueForReuse(s) == s);
      if (auto itC = reuse.createOnlyStepForState.find(s); itC != reuse.createOnlyStepForState.end()) {
         ExecutionStepOp top = liftToTopLevelStepInDonor(donor, itC->second);
         if (seen.insert(top.getOperation()).second) out.push_back(top);
      }
      auto itW = findReuseMap(reuse.writerStepsByState, s);
      if (itW == reuse.writerStepsByState.end()) continue;
      for (ExecutionStepOp w : itW->second) {
         ExecutionStepOp top = liftToTopLevelStepInDonor(donor, w);
         if (seen.insert(top.getOperation()).second) out.push_back(top);
      }
   }
   llvm::sort(out, [](ExecutionStepOp a, ExecutionStepOp b) { return a->isBeforeInBlock(b); });
   return out;
}

static int executionStepDonorIndex(ExecutionGroupOp donor, ExecutionStepOp target) {
   int i = 0;
   for (mlir::Operation& op : donor.getSubOps().front()) {
      if (auto es = mlir::dyn_cast<ExecutionStepOp>(&op)) {
         if (es == target) return i;
         ++i;
      }
   }
   return -1;
}

static const ModuleReuseInfo::StepRW* lookupStepRw(const ModuleReuseInfo& reuse, ExecutionStepOp step) {
   for (const auto& e : reuse.steps) {
      if (e.step == step) return &e;
   }
   return nullptr;
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

void insertCachePutsForTargets(mlir::ModuleOp producerModule, llvm::ArrayRef<CacheTarget> targets) {
   if (targets.empty()) return;
   auto reuse = collectModuleReuseInfo(producerModule);

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
}

void injectCacheGetsAndDeleteConstructionSteps(mlir::ModuleOp consumerModule,
                                               llvm::ArrayRef<CacheTarget> targets) {
   if (targets.empty()) return;

   auto reuse = collectModuleReuseInfo(consumerModule);

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
   llvm::DenseSet<mlir::Operation*> uniqOpsToErase;
   llvm::SmallVector<mlir::Operation*, 128> opsToErase;

   auto rewriteOne = [&](mlir::Value state, uint64_t cacheKey) {
      assert(!mlir::isa<ThreadLocalType>(state.getType()) &&
             "rewrite must never target thread_local-wrapped states");

      llvm::SmallVector<mlir::Value, 2> statesToDelete;
      statesToDelete.push_back(state);
      if (auto itTL = reuse.mergedFromThreadLocal.find(state); itTL != reuse.mergedFromThreadLocal.end()) {
         // If the match is on a merge result, we still must delete the thread_local side steps,
         // but we must never cache that thread_local value.
         statesToDelete.push_back(itTL->second);
      }

      llvm::SmallVector<ExecutionStepOp, 32> writerSteps;
      for (auto st : statesToDelete) {
         auto itW = reuse.writerStepsByState.find(st);
         if (itW == reuse.writerStepsByState.end()) continue;
         for (auto s : itW->second) writerSteps.push_back(s);
      }
      if (writerSteps.empty()) return;

      auto group = findEnclosingExecutionGroup(state);
      mlir::Value cached;
      if (auto it = keyToCached.find(cacheKey); it != keyToCached.end()) {
         cached = it->second;
      } else {
         cached = insertCacheGetAtExecutionGroupStart(group, state.getType(), cacheKey);
         keyToCached[cacheKey] = cached;
      }
      state.replaceAllUsesWith(cached);

      // Delete create steps too (create_only steps are part of construction but may not be marked as writes).
      for (auto st : statesToDelete) {
         if (auto defStep = mlir::dyn_cast_or_null<ExecutionStepOp>(st.getDefiningOp())) {
            auto* op = defStep.getOperation();
            if (uniqOpsToErase.insert(op).second) opsToErase.push_back(op);
         }
      }
      for (auto s : writerSteps) {
         auto* op = s.getOperation();
         if (uniqOpsToErase.insert(op).second) opsToErase.push_back(op);
      }
   };

   for (auto& t : targets) {
      if (!t.state) continue;
      rewriteOne(t.state, t.cacheKey);
   }
}

} // namespace

ReusePlanRewriteResult rewritePlansWithSyntheticQuery0(
   mlir::ModuleOp query0,
   mlir::ModuleOp query1,
   llvm::ArrayRef<CrossQueryStateMatchPair> matches) {
   ReusePlanRewriteResult res;

   llvm::SmallVector<CacheTarget, 64> targets0;
   llvm::SmallVector<CacheTarget, 64> targets1;
   targets0.reserve(matches.size());
   targets1.reserve(matches.size());

   for (auto& m : matches) {
      if (m.stateA) {
         assert(!mlir::isa<ThreadLocalType>(m.stateA.getType()) &&
                "match pairs must never target thread_local-wrapped states; match should use merge result");
         targets0.push_back(CacheTarget{m.stateA, m.cacheKey});
      }
      if (m.stateB) {
         assert(!mlir::isa<ThreadLocalType>(m.stateB.getType()) &&
                "match pairs must never target thread_local-wrapped states; match should use merge result");
         targets1.push_back(CacheTarget{m.stateB, m.cacheKey});
      }
   }

   res.numTargetsQuery0 = targets0.size();
   res.numTargetsQuery1 = targets1.size();

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
   auto reuse0 = collectModuleReuseInfo(query0);
   llvm::DenseSet<mlir::Value> neededStates = expandNeededStatesFromTargets(targets0, reuse0);
   auto stepsToClone = collectCreateAndWriteStepsForStates(donorGroup, reuse0, neededStates);
   auto mapping = cloneExecutionStepsToQuery0(q0Group, stepsToClone);

   // Map targets into query0.
   llvm::SmallVector<CacheTarget, 64> targetsQ0;
   targetsQ0.reserve(targets0.size());
   for (auto& t : targets0) {
      if (!t.state) continue;
      mlir::Value mapped = mapping.lookupOrNull(t.state);
      if (!mapped) {
         // If the matched state value is a canonical step input (BlockArgument), cache the value produced
         // by the last writer step for this state.
         if (!t.state.getDefiningOp()) {
            if (auto itW = reuse0.writerStepsByState.find(t.state); itW != reuse0.writerStepsByState.end()) {
               if (!itW->second.empty()) {
                  auto lastWriter = itW->second.back();
                  if (lastWriter.getNumResults() == 1) {
                     mapped = mapping.lookupOrNull(lastWriter.getResult(0));
                  }
               }
            }
         }
      }
      if (!mapped) continue;
      targetsQ0.push_back(CacheTarget{mapped, t.cacheKey});
   }
   res.numTargetsQuery0Mapped = targetsQ0.size();
   if (targetsQ0.empty()) {
      return res;
   }

   // Producer: cache_puts.
   insertCachePutsForTargets(*res.query0, targetsQ0);

   // Consumers: cache_get + delete construction steps.
   injectCacheGetsAndDeleteConstructionSteps(query0, targets0);
   injectCacheGetsAndDeleteConstructionSteps(query1, targets1);

   return res;
}

} // namespace lingodb::compiler::dialect::subop

