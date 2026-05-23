#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_REUSESTATECLOSURE_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_REUSESTATECLOSURE_H

#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseRewriteCommon.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace lingodb::compiler::dialect::subop {

struct JoinBufferHivSsaClosure {
   llvm::DenseSet<void*> opaque;
   llvm::SmallVector<mlir::Value, 64> values;
};

bool opaqueClosureContains(const llvm::DenseSet<void*>& closure, mlir::Value v);

void expandClosureThroughExecutionStepPorts(mlir::ModuleOp module, llvm::DenseSet<void*>& closure);

/// Link \c nested_execution_group operands with body block arguments (probe pipeline ports).
void expandClosureThroughNestedExecutionGroupPorts(mlir::ModuleOp module, llvm::DenseSet<void*>& closure);

bool executionStepTouchesClosure(ExecutionStepOp step, const llvm::DenseSet<void*>& closure);

bool opOperandsOrNestedBlockArgsTouchClosure(mlir::Operation* op, const llvm::DenseSet<void*>& closure);

void synchronizeExecutionStepPortTypes(mlir::ModuleOp module, const llvm::DenseSet<void*>* closureFilter);

mlir::Value mapStateThroughExecutionStepOperands(mlir::Value v, mlir::Operation* user);

void collectJoinBufferReachabilitySeeds(mlir::Value canonicalMergedBuffer, const ModuleReuseInfo& reuse,
                                        llvm::SmallVectorImpl<mlir::Value>& seeds);

JoinBufferHivSsaClosure computeJoinBufferHivSsaClosure(llvm::ArrayRef<mlir::Value> canonicalBuffers,
                                                       const ModuleReuseInfo& reuse);

llvm::DenseSet<mlir::Value> expandNeededStatesFromTargets(llvm::ArrayRef<CacheTarget> targets0,
                                                          const ModuleReuseInfo& reuse);

llvm::DenseSet<void*> closureLiveStatesFromReuseReturnSeeds(
   llvm::ArrayRef<mlir::Value> returnPipelineSeeds, const ModuleReuseInfo& reuse,
   const llvm::DenseMap<mlir::Operation*, const ModuleReuseInfo::StepRW*>& rwByStepOp);

} // namespace lingodb::compiler::dialect::subop

#endif
