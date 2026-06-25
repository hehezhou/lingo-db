#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_CROSSQUERYSTATEREUSE_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_CROSSQUERYSTATEREUSE_H

#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseJoinSuperset.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateExtraction.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseRewriteCommon.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <string>

namespace lingodb::catalog {
class Catalog;
} // namespace lingodb::catalog
namespace lingodb::compiler::dialect::subop {

struct ReusePlanRewriteResult {
   mlir::OwningOpRef<mlir::ModuleOp> query0; // synthetic producer that cache_puts reusable states
   size_t numTargetsQuery0 = 0;
   size_t numTargetsQuery1 = 0;
   size_t numTargetsQuery0Mapped = 0;
   // Convenience counters for reporting: exclude `!subop.table<...>` targets (external tables).
   size_t numTargetsQuery0NoTable = 0;
   size_t numTargetsQuery1NoTable = 0;
   size_t numTargetsQuery0MappedNoTable = 0;
   // Union reuse means the consumer gets a shared identical/mixed state. Build-step reuse means
   // construction is shared, but each consumer still gets a separate final state.
   size_t numUnionTargetsQuery0 = 0;
   size_t numUnionTargetsQuery1 = 0;
   size_t numUnionTargetsQuery0Mapped = 0;
   size_t numUnionTargetsQuery0NoTable = 0;
   size_t numUnionTargetsQuery1NoTable = 0;
   size_t numUnionTargetsQuery0MappedNoTable = 0;
   size_t numBuildStepTargetsQuery0 = 0;
   size_t numBuildStepTargetsQuery1 = 0;
   size_t numBuildStepTargetsQuery0Mapped = 0;
   size_t numBuildStepTargetsQuery0NoTable = 0;
   size_t numBuildStepTargetsQuery1NoTable = 0;
   size_t numBuildStepTargetsQuery0MappedNoTable = 0;
};

struct BatchReusePlanRewriteResult {
   mlir::OwningOpRef<mlir::ModuleOp> synthetic;
   llvm::SmallVector<size_t, 8> numTargetsPerQuery;
   llvm::SmallVector<size_t, 8> numTargetsNoTablePerQuery;
   size_t numTargetsSyntheticMapped = 0;
   size_t numTargetsSyntheticMappedNoTable = 0;
   llvm::SmallVector<size_t, 8> numUnionTargetsPerQuery;
   llvm::SmallVector<size_t, 8> numUnionTargetsNoTablePerQuery;
   llvm::SmallVector<size_t, 8> numBuildStepTargetsPerQuery;
   llvm::SmallVector<size_t, 8> numBuildStepTargetsNoTablePerQuery;
   size_t numUnionTargetsSyntheticMapped = 0;
   size_t numUnionTargetsSyntheticMappedNoTable = 0;
   size_t numBuildStepTargetsSyntheticMapped = 0;
   size_t numBuildStepTargetsSyntheticMappedNoTable = 0;
   CachedJoinBufferLayoutsByKey cachedJoinLayouts;
};

// Three-stage pipeline:
// 1) analysis: provided by StateExtraction (collectModuleReuseInfo, buildStateMatchProfiles)
// 2) match: provided by StateExtraction (collectCrossQueryStateMatchPairs)
// 3) rewrite: implemented here
//
// Rewrites both queries to use cache_get, and returns a synthetic query0 module that
// computes/cache_puts the reusable states (cloned from query0's construction steps).
ReusePlanRewriteResult rewritePlansWithSyntheticQuery0(
   mlir::ModuleOp query0,
   mlir::ModuleOp query1,
   llvm::ArrayRef<CrossQueryStateMatchPair> matches,
   lingodb::catalog::Catalog* catalog = nullptr);

BatchReusePlanRewriteResult rewritePlansWithSyntheticQueryBatch(
   llvm::ArrayRef<mlir::ModuleOp> queries,
   llvm::ArrayRef<CrossQueryStateMatchGroup> groups,
   lingodb::catalog::Catalog* catalog = nullptr);

/// \p reuseBeforeMutation must reflect the module **before** `rewriteHashmapTypesInModule` runs on
/// \p producerModule (when null, it is collected here). Used to skip a redundant full-module scan.
void insertCachePutsForTargets(mlir::ModuleOp producerModule, llvm::ArrayRef<CacheTarget> targets,
                               const ModuleReuseInfo* reuseBeforeMutation = nullptr);

/// \p reuseBeforeMutation same contract as for `insertCachePutsForTargets`. When
/// \p joinBufferHashmapLayoutAlreadyApplied is true, join-buffer / HIV layout extension was already
/// applied on \p consumerModule and is not run again.
void injectCacheGetsAndDeleteConstructionSteps(mlir::ModuleOp consumerModule, llvm::ArrayRef<CacheTarget> targets,
                                               const ModuleReuseInfo* reuseBeforeMutation = nullptr,
                                               bool joinBufferHashmapLayoutAlreadyApplied = false,
                                               bool joinBufferWritePredAlreadyApplied = false);

} // namespace lingodb::compiler::dialect::subop

#endif
