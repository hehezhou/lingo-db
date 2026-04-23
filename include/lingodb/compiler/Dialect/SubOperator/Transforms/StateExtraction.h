#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_STATEEXTRACTION_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_STATEEXTRACTION_H

#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"

#include <llvm/Support/raw_ostream.h>

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace lingodb::compiler::dialect::subop {

void printExecutionSteps(mlir::ModuleOp moduleOp, llvm::raw_ostream& os);

} // namespace lingodb::compiler::dialect::subop

#endif

