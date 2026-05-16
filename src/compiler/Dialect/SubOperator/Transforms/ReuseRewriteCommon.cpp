#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseRewriteCommon.h"

#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"

#include "mlir/IR/BuiltinOps.h"

namespace lingodb::compiler::dialect::subop {

bool isPipelineStateValue(mlir::Value v) {
   mlir::Type t = v.getType();
   if (mlir::isa<subop::State>(t)) return true;
   auto tl = mlir::dyn_cast<subop::ThreadLocalType>(t);
   return tl && mlir::isa<subop::State>(tl.getWrapped());
}

mlir::Value peelBlockArgsToEnclosingOperands(mlir::Value v) {
   for (;;) {
      auto ba = mlir::dyn_cast<mlir::BlockArgument>(v);
      if (!ba) break;
      mlir::Block* owner = ba.getOwner();
      mlir::Operation* parent = owner->getParentOp();
      if (!parent) break;

      if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(parent)) {
         if (&step.getSubOps().front() == owner && ba.getArgNumber() < step.getNumOperands()) {
            v = step.getOperand(ba.getArgNumber());
            continue;
         }
      }
      if (auto neg = mlir::dyn_cast<subop::NestedExecutionGroupOp>(parent)) {
         if (&neg.getSubOps().front() == owner && ba.getArgNumber() < neg.getNumOperands()) {
            v = neg.getOperand(ba.getArgNumber());
            continue;
         }
      }
      break;
   }
   return v;
}

llvm::DenseMap<mlir::Operation*, const ModuleReuseInfo::StepRW*> buildRwByStepOpMap(const ModuleReuseInfo& reuse) {
   llvm::DenseMap<mlir::Operation*, const ModuleReuseInfo::StepRW*> rwByStepOp;
   for (const auto& e : reuse.steps) {
      ExecutionStepOp stepOp = e.step;
      rwByStepOp[stepOp.getOperation()] = &e;
   }
   return rwByStepOp;
}

ExecutionGroupOp getSingleExecutionGroup(mlir::ModuleOp module) {
   llvm::SmallVector<ExecutionGroupOp, 4> groups;
   module.walk([&](ExecutionGroupOp g) { groups.push_back(g); });
   assert(groups.size() == 1 && "expected exactly one execution_group per module");
   return groups.front();
}

} // namespace lingodb::compiler::dialect::subop
