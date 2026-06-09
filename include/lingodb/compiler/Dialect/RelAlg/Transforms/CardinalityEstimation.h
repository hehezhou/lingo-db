#ifndef LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_CARDINALITYESTIMATION_H
#define LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_CARDINALITYESTIMATION_H

#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"

#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace lingodb::catalog {
class Catalog;
} // namespace lingodb::catalog

namespace lingodb::runtime {
struct ExternalDatasourceProperty;
} // namespace lingodb::runtime

namespace lingodb::compiler::dialect::relalg {

/// Estimate rows for external table descriptors by evaluating their pushdown filters on the table sample.
/// Multiple descriptors are interpreted as OR-reuse alternatives. Each descriptor may also carry OR clauses.
/// Optional map/filter predicate op pairs are aligned with filterSources and interpreted as additional
/// per-source predicates.
double estimateExternalDatasourceOrRowsFromSample(
   llvm::ArrayRef<lingodb::runtime::ExternalDatasourceProperty> filterSources,
   lingodb::catalog::Catalog& catalog,
   llvm::ArrayRef<mlir::Operation*> mapOps = {},
   llvm::ArrayRef<mlir::Operation*> filterOps = {});

} // namespace lingodb::compiler::dialect::relalg

#endif // LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_CARDINALITYESTIMATION_H
