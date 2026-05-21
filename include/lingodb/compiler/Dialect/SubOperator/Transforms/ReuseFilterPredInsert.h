#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_REUSEFILTERPREDINSERT_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_REUSEFILTERPREDINSERT_H

#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseRewriteCommon.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseStateClosure.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/runtime/ExternalDataSourceProperty.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace lingodb::compiler::dialect::subop {

subop::Member makeOrGetPredMember(mlir::MLIRContext* ctx);

/// Payload member for per-query reuse predicates (`filter_pred$N`, \p slot from semantic key).
subop::Member makeOrGetPredMemberForSlot(mlir::MLIRContext* ctx, unsigned slot);

bool valueMembersContainMemberNamed(mlir::MLIRContext* ctx, subop::StateMembersAttr members, llvm::StringRef name);

llvm::SmallVector<mlir::Value, 8> collectJoinBuffersFeedingHashIndexedView(llvm::ArrayRef<mlir::Value> candidates,
                                                                             const ModuleReuseInfo& reuse);

void rewriteHashmapTypesInModule(mlir::ModuleOp module, llvm::ArrayRef<mlir::Value> extendJoinBufferStates,
                                 const ModuleReuseInfo* reuseForJoinBuffers);

void collectJoinBufferStatesFromTargets(llvm::ArrayRef<CacheTarget> targets,
                                        llvm::SmallVector<mlir::Value, 8>& out,
                                        const ModuleReuseInfo* reuse = nullptr);

llvm::SmallVector<runtime::FilterDescription, 8> decodeFiltersForStateFromWriterSteps(
   mlir::Value state, const ModuleReuseInfo& reuse,
   const llvm::DenseMap<mlir::Operation*, const ModuleReuseInfo::StepRW*>* rwByStepOp = nullptr);

llvm::SmallVector<runtime::FilterDescription, 8> decodeFiltersFromTableScanInExecutionStep(
   ExecutionStepOp step);

void insertWriteSidePredIntoHashMapConstructionStep(ExecutionStepOp step,
                                                    llvm::ArrayRef<runtime::FilterDescription> filters);

void insertWriteSidePredIntoBufferConstructionStep(ExecutionStepOp step,
                                                   llvm::ArrayRef<runtime::FilterDescription> filters);

/// Like \c insertWriteSidePredIntoBufferConstructionStep but targets \p predMemberName (e.g. \c filter_pred$1).
void insertWriteSidePredIntoBufferConstructionStepForPredMember(
   ExecutionStepOp step, llvm::ArrayRef<runtime::FilterDescription> filters, llvm::StringRef predMemberName);

void materializeConstantTruePredMemberOnBufferMaterialize(subop::MaterializeOp matOp, llvm::StringRef predMemberName,
                                                           bool updateStreamOperand);

/// First \c filter_pred$N member on a hash-indexed view (consumer probe after layout align).
std::optional<subop::Member> findFilterPredMemberOnHashIndexedView(subop::HashIndexedViewType hiv);

void insertScanRefsPredFilter(ExecutionStepOp step, subop::Member predMember);

void insertHashIndexedViewGatherPredFilters(ExecutionStepOp step, subop::Member predMember);

/// Fill missing `filter_pred$0` materialize mappings from table-scan pushdown filters (or constant true).
void ensureJoinBufferFilterPredMaterializeMappings(mlir::ModuleOp module);

void propagateSubOpColumnAttrsFromSsaStateLayout(mlir::ModuleOp module,
                                                 const llvm::DenseSet<void*>* closureFilter);

void alignBufferMergeThreadLocalsWithExtendedMergeResult(mlir::ModuleOp module);

mlir::Operation* drillToStateConsumerSkippingNestedScopes(mlir::Block& startBlock, mlir::Value startState);

mlir::Value materializeRuntimeFiltersAsSubopFilter(
   mlir::OpBuilder& b, mlir::Location loc, mlir::Value stream,
   const llvm::DenseMap<llvm::StringRef, tuples::ColumnRefAttr>& colByName,
   llvm::ArrayRef<runtime::FilterDescription> filters);

/// Decode external-table filters for each cache target (including paired `thread_local` writers).
llvm::DenseMap<mlir::Value, llvm::SmallVector<runtime::FilterDescription, 8>>
decodeFiltersByCacheTargets(llvm::ArrayRef<CacheTarget> targets, const ModuleReuseInfo& reuse);

/// Re-apply stream-level construction filters at `cache_get` use sites (non-join hashmap/buffer states).
void materializeRuntimeFiltersAtCacheGetUses(mlir::Value cached,
                                             llvm::ArrayRef<runtime::FilterDescription> decodedFilters);

/// After join-buffer/HIV layout + `propagateSubOpColumnAttrsFromSsaStateLayout`, insert
/// `filter(all_true [filter_pred])` on HIV `gather`/`lookup` probe paths (must run last).
void applyJoinBufferProbePredFiltersAfterLayout(mlir::ModuleOp module);

} // namespace lingodb::compiler::dialect::subop

#endif
