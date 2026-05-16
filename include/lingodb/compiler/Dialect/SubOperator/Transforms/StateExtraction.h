#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_STATEEXTRACTION_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_STATEEXTRACTION_H

#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/runtime/ExternalDataSourceProperty.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include <llvm/Support/raw_ostream.h>

#include <cstdint>
#include <utility>

namespace mlir {
class ModuleOp;
class Value;
} // namespace mlir

namespace lingodb::compiler::dialect::subop {

/// Canonicalize a state SSA value to the same form used as keys in `ModuleReuseInfo` maps
/// (`writerStepsByState`, `mergedFromThreadLocal`, `createOnlyStepForState`).
mlir::Value canonicalizeStateValueForReuse(mlir::Value v);

void printExecutionSteps(mlir::ModuleOp moduleOp, llvm::raw_ostream& os);

/// For each `execution_group`, print top-level ops in block order with indices that match
/// `SubOpToControlFlow` lowering (`handleExecutionStepCPU` stderr `execution_step ordinal`).
void printTopLevelExecutionStepLayout(mlir::ModuleOp moduleOp, llvm::raw_ostream& os);

// Experimental: compare states across multiple already-lowered SubOp modules.
// Each entry is (query_id, module).
void printCrossQueryStateMatches(llvm::ArrayRef<std::pair<int, mlir::ModuleOp>> queries, llvm::raw_ostream& os);

struct CrossQueryStateMatchPair {
   int queryA = -1;
   int queryB = -1;
   mlir::Value stateA;
   mlir::Value stateB;
   uint64_t cacheKey = 0;
};

struct ModuleReuseInfo {
   // Per execution_step: which canonicalized states are read/written.
   struct StepRW {
      subop::ExecutionStepOp step;
      llvm::SmallVector<mlir::Value, 8> reads;
      llvm::SmallVector<mlir::Value, 8> writes;
   };
   llvm::SmallVector<StepRW, 128> steps;

   // Merge pairing: global_state -> thread_local_state (both canonicalized to step inputs/results).
   llvm::DenseMap<mlir::Value, mlir::Value> mergedFromThreadLocal;

   /// Join build chain: `thread_local` buffer -> merged global `!subop.buffer` -> `create_hash_indexed_view`.
   /// Maps hash_indexed_view SSA -> merged global buffer (cache/match representative stays the buffer).
   llvm::DenseMap<mlir::Value, mlir::Value> hashIndexedViewFromMergedBuffer;
   /// Inverse of the above (merged global buffer -> its paired hash_indexed_view), when present.
   llvm::DenseMap<mlir::Value, mlir::Value> mergedBufferToHashIndexedView;

   // Convenience: state -> steps that write it (subset of steps[]).
   llvm::DenseMap<mlir::Value, llvm::SmallVector<subop::ExecutionStepOp, 8>> writerStepsByState;

   /// Canonical state SSA value (same as `canonicalizeStateValueForReuse`) -> the **create-only**
   /// `execution_step` that returns it. Used when `writerStepsByState` has no entry for pure
   /// `subop.create` / `create_thread_local` producers.
   llvm::DenseMap<mlir::Value, subop::ExecutionStepOp> createOnlyStepForState;

   /// Canonical external table state (`!subop.table`) -> decoded external datasource property
   /// (parsed from `subop.get_external` `descr` hex string).
   llvm::DenseMap<mlir::Value, lingodb::runtime::ExternalDatasourceProperty> externalDatasourceByTableState;
};

ModuleReuseInfo collectModuleReuseInfo(mlir::ModuleOp moduleOp);

/// Canonical cache/match target for a join buffer chain: the `hash_indexed_view` when a serial
/// `buffer -> create_hash_indexed_view` link exists; otherwise `canonicalizeStateValueForReuse(v)`.
mlir::Value bufferJoinChainRootForReuse(mlir::Value v, const ModuleReuseInfo& reuse);

/// Map a matched state SSA to the value that should receive `cache_put` / `cache_get` (HIV root).
mlir::Value resolveCacheTargetStateForReuse(mlir::Value v, const ModuleReuseInfo& reuse);

/// True for thread_local merge partners and hash_indexed_view partners of a buffer join chain.
bool isBufferJoinChainNonRootPartner(mlir::Value v, const ModuleReuseInfo& reuse);

/// Invoke \p fn for merged global buffer, its thread_local (if any), and hash_indexed_view (if any).
void forEachBufferJoinChainPartner(mlir::Value chainRootBuffer, const ModuleReuseInfo& reuse,
                                   llvm::function_ref<void(mlir::Value)> fn);

llvm::SmallVector<CrossQueryStateMatchPair, 64>
collectCrossQueryStateMatchPairs(llvm::ArrayRef<std::pair<int, mlir::ModuleOp>> queries);

} // namespace lingodb::compiler::dialect::subop

#endif

