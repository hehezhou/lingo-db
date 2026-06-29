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

/// Keep only filters whose \c columnName maps to a column on the step's table \c scan_refs.
llvm::SmallVector<runtime::FilterDescription, 8> restrictFiltersToTableScanInExecutionStep(
   ExecutionStepOp step, llvm::ArrayRef<runtime::FilterDescription> filters);

void insertWriteSidePredIntoHashMapConstructionStep(ExecutionStepOp step,
                                                    llvm::ArrayRef<runtime::FilterDescription> filters);

void insertWriteSidePredIntoBufferConstructionStep(ExecutionStepOp step,
                                                   llvm::ArrayRef<runtime::FilterDescription> filters);

/// Like \c insertWriteSidePredIntoBufferConstructionStep but targets \p predMemberName (e.g. \c filter_pred$1).
void insertWriteSidePredIntoBufferConstructionStepForPredMember(
   ExecutionStepOp step, llvm::ArrayRef<runtime::FilterDescription> filters, llvm::StringRef predMemberName,
   bool allowSharedScanPredicate = true, bool clearScanPushdown = false);

void materializeConstantTruePredMemberOnBufferMaterialize(subop::MaterializeOp matOp, llvm::StringRef predMemberName,
                                                           bool updateStreamOperand);

/// First \c filter_pred$N member on a hash-indexed view (consumer probe after layout align).
std::optional<subop::Member> findFilterPredMemberOnHashIndexedView(subop::HashIndexedViewType hiv);

void insertScanRefsPredFilter(ExecutionStepOp step, subop::Member predMember);

/// Retag a hash-indexed-view probe to MixedHIV and make scan_list apply stored filter_pred$N.
void insertHashIndexedViewGatherPredFilters(ExecutionStepOp step, subop::Member predMember,
                                            const llvm::DenseSet<void*>* closureFilter = nullptr);

/// After a \c scan_list / \c scan_refs producer, \c gather filter_pred + \c filter(all_true).
void insertProbePredFilterImmediatelyAfterScanProducer(mlir::Operation* anchorOp, mlir::Value scanStream,
                                                       tuples::ColumnRefAttr entryRef, subop::Member predMember);

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

/// Build a boolean predicate column after \c scanOp from runtime simple filters. The scan table ref is widened
/// for filter columns, needed columns are gathered immediately after the scan, and downstream users can be
/// rewired to consume the predicate stream.
std::pair<mlir::Value, tuples::ColumnRefAttr> materializeRuntimeFiltersAsPredicateColumnAfterScanRefs(
   subop::ScanRefsOp scanOp, llvm::ArrayRef<runtime::FilterDescription> filters,
   llvm::StringRef predLeafName, bool rewireDownstreamUses);

struct RuntimeFilterIdClause {
   unsigned id = 0;
   llvm::SmallVector<runtime::FilterDescription, 8> filters;
};

/// Build an i64 id column after \c scanOp. Each clause is evaluated as an AND of runtime simple filters;
/// because callers use this for disjoint clauses, the first matching id is the row source id.
std::pair<mlir::Value, tuples::ColumnRefAttr> materializeRuntimeFilterClausesAsIdColumnAfterScanRefs(
   subop::ScanRefsOp scanOp, llvm::ArrayRef<RuntimeFilterIdClause> clauses,
   unsigned defaultId, llvm::StringRef idLeafName, bool rewireDownstreamUses);

/// Decode external-table filters for each cache target (including paired `thread_local` writers).
llvm::DenseMap<mlir::Value, llvm::SmallVector<runtime::FilterDescription, 8>>
decodeFiltersByCacheTargets(llvm::ArrayRef<CacheTarget> targets, const ModuleReuseInfo& reuse);

/// Re-apply stream-level construction filters at `cache_get` use sites (non-join hashmap/buffer states).
void materializeRuntimeFiltersAtCacheGetUses(mlir::Value cached,
                                             llvm::ArrayRef<runtime::FilterDescription> decodedFilters);

} // namespace lingodb::compiler::dialect::subop

#endif
