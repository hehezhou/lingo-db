#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseJoinSuperset.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseRewriteCommon.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseFilterPredInsert.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/DB/IR/DBTypes.h"
#include "lingodb/compiler/Dialect/util/UtilTypes.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseStateClosure.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsAttributes.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateExtraction.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateUsageTransformer.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/Transforms/CardinalityEstimation.h"
#include "lingodb/runtime/ExternalDataSourceProperty.h"
#include "lingodb/utility/Serialization.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <unordered_set>

namespace lingodb::compiler::dialect::subop {
namespace {

using lingodb::runtime::ExternalDatasourceProperty;

[[noreturn]] static void abortAggregateUnionUnsupported(llvm::StringRef message) {
   llvm::errs() << "aggregate union: " << message << "\n";
   std::abort();
}

static llvm::StringRef stripMemberSuffix(llvm::StringRef name) {
   size_t pos = name.find('$');
   if (pos == llvm::StringRef::npos) return name;
   return name.take_front(pos);
}

static std::string columnSemanticKey(llvm::StringRef scope, llvm::StringRef leaf) {
   return (scope + "\x1f" + leaf).str();
}

static std::string aggregatePayloadScopeSemantic(llvm::StringRef scope) {
   if (!scope.starts_with("oj")) return scope.str();
   llvm::StringRef tail = scope.drop_front(2);
   if (tail.empty()) return scope.str();
   for (char c : tail) {
      if (c < '0' || c > '9') return scope.str();
   }
   return "oj";
}

static constexpr llvm::StringRef kReuseFilterPredScope = "reuse_filter_pred";
static constexpr llvm::StringRef kReuseFilterPredUnionScope = "reuse_filter_pred_union";

static bool isFilterPredPayloadColumn(llvm::StringRef memberName, llvm::StringRef leaf) {
   return memberName.starts_with("filter_pred") || leaf == "filter_pred";
}

static std::optional<unsigned> parseFilterPredMemberSlot(llvm::StringRef memberName) {
   if (!memberName.consume_front("filter_pred$")) return std::nullopt;
   unsigned slot = 0;
   if (memberName.getAsInteger(10, slot)) return std::nullopt;
   return slot;
}

/// Union payload key for per-query write-side predicates on a shared table→HIV chain.
static std::string reuseFilterPredSemanticKey(unsigned reuseQueryIndex) {
   return columnSemanticKey(kReuseFilterPredScope, llvm::Twine(reuseQueryIndex).str());
}

static std::string reuseFilterPredUnionSemanticKey(unsigned unionSlot) {
   return columnSemanticKey(kReuseFilterPredUnionScope, llvm::Twine(unionSlot).str());
}

static bool parseReuseFilterPredUnionSemanticKey(llvm::StringRef semanticKey, unsigned& unionSlot) {
   size_t sep = semanticKey.find('\x1f');
   if (sep == llvm::StringRef::npos) return false;
   if (semanticKey.take_front(sep) != kReuseFilterPredUnionScope) return false;
   return !semanticKey.drop_front(sep + 1).getAsInteger(10, unionSlot);
}

static bool parseFilterPredLayoutSemanticKey(llvm::StringRef semanticKey, unsigned& predIndex) {
   if (parseReuseFilterPredSemanticKey(semanticKey, predIndex)) return true;
   if (parseReuseFilterPredUnionSemanticKey(semanticKey, predIndex)) return true;
   return static_cast<bool>(parseFilterPredMemberSlot(semanticKey));
}

static llvm::StringRef normalizeColumnIdentifier(llvm::StringRef identifier) {
   return stripMemberSuffix(identifier);
}

static uint64_t combinePayloadHash(uint64_t a, uint64_t b) {
   return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
}

static uint64_t hashPayloadString(llvm::StringRef s) {
   return static_cast<uint64_t>(llvm::hash_value(s));
}

static uint64_t hashPayloadType(mlir::Type type) {
   if (!type) return 0;
   std::string typeStr;
   llvm::raw_string_ostream os(typeStr);
   type.print(os);
   return hashPayloadString(os.str());
}

static uint64_t payloadSyntheticColumnHash(llvm::StringRef scope, llvm::StringRef leaf, mlir::Type colType) {
   uint64_t h = hashPayloadString("payload_synthetic_column");
   h = combinePayloadHash(h, hashPayloadString(scope));
   h = combinePayloadHash(h, hashPayloadString(leaf));
   return combinePayloadHash(h, hashPayloadType(colType));
}

static uint64_t payloadColumnIdentityHash(tuples::ColumnRefAttr ref,
                                          const llvm::DenseMap<const void*, uint64_t>& columnHashes) {
   auto it = columnHashes.find(&ref.getColumn());
   assert(it != columnHashes.end() && "payload column ref must have StateExtraction column identity hash");
   return it->second;
}

static bool isPayloadMemberSlotName(llvm::StringRef name) { return name.starts_with("member$"); }
static bool isJoinBufferInternalMemberName(llvm::StringRef name) {
   return name.starts_with("link$") || name.starts_with("hash$");
}

struct PayloadColumnSpec {
   std::string semanticKey;
   uint64_t semanticHash = 0;
   std::string scope;
   std::string leaf;
   mlir::Type colType;
   bool isJoinKey = false;
   bool inQuery0 = false;
   bool inQuery1 = false;
   bool fromExternalTable = false;
};

struct JoinBufferUnionPlan {
   subop::Member linkMember;
   subop::Member hashMember;
   llvm::SmallVector<PayloadColumnSpec, 8> payloadColumns;
   llvm::SmallVector<mlir::Type, 8> payloadMemberTypes;
   /// Buffer/HIV payload members aligned with \p payloadColumns (semantic union, not slot `member$N`).
   llvm::SmallVector<subop::Member, 8> payloadMembers;
};

static unsigned filterPredUnionSlotForQueryIndices(llvm::ArrayRef<unsigned> queryIndices) {
   unsigned maxIdx = 0;
   for (unsigned qIdx : queryIndices) maxIdx = std::max(maxIdx, qIdx);
   return maxIdx + 1;
}

static void ensureFilterPredUnionColumn(JoinBufferUnionPlan& plan, unsigned unionSlot,
                                        mlir::MLIRContext* ctx) {
   std::string semKey = reuseFilterPredUnionSemanticKey(unionSlot);
   for (const PayloadColumnSpec& spec : plan.payloadColumns) {
      if (spec.semanticKey == semKey) return;
   }
   PayloadColumnSpec predSpec;
   predSpec.scope = kReuseFilterPredUnionScope.str();
   predSpec.leaf = llvm::Twine(unionSlot).str();
   predSpec.colType = mlir::IntegerType::get(ctx, 1);
   predSpec.semanticKey = semKey;
   predSpec.semanticHash = payloadSyntheticColumnHash(predSpec.scope, predSpec.leaf, predSpec.colType);
   plan.payloadColumns.push_back(std::move(predSpec));
   llvm::sort(plan.payloadColumns, [](const PayloadColumnSpec& a, const PayloadColumnSpec& b) {
      if (a.isJoinKey != b.isJoinKey) return a.isJoinKey > b.isJoinKey;
      if (a.semanticHash != b.semanticHash) return a.semanticHash < b.semanticHash;
      return a.semanticKey < b.semanticKey;
   });
   plan.payloadMemberTypes.clear();
   plan.payloadMemberTypes.reserve(plan.payloadColumns.size());
   for (const PayloadColumnSpec& spec : plan.payloadColumns) plan.payloadMemberTypes.push_back(spec.colType);
}

static subop::CreateHashIndexedView findCreateHashIndexedViewForState(
   mlir::Value hiv, const llvm::DenseMap<mlir::Value, llvm::SmallVector<subop::ExecutionStepOp, 8>>& writerStepsByState) {
   auto itW = writerStepsByState.find(hiv);
   if (itW == writerStepsByState.end()) return {};
   for (subop::ExecutionStepOp ws : itW->second) {
      subop::CreateHashIndexedView found;
      ws.walk([&](subop::CreateHashIndexedView op) { found = op; });
      if (found) return found;
   }
   return {};
}

static subop::BufferType getInnerBufferTypeForMaterializeState(mlir::Type stateTy) {
   if (auto b = mlir::dyn_cast<subop::BufferType>(stateTy)) return b;
   if (auto tl = mlir::dyn_cast<subop::ThreadLocalType>(stateTy))
      return mlir::dyn_cast<subop::BufferType>(tl.getWrapped());
   return nullptr;
}

static subop::BufferType bufferTypeForJoinTarget(mlir::Value mergedBuffer, const ModuleReuseInfo& reuse) {
   if (auto b = mlir::dyn_cast<subop::BufferType>(mergedBuffer.getType())) return b;
   if (auto tl = mlir::dyn_cast<subop::ThreadLocalType>(mergedBuffer.getType()))
      return mlir::dyn_cast<subop::BufferType>(tl.getWrapped());
   mlir::Value canon = canonicalizeStateValueForReuse(mergedBuffer);
   if (auto it = reuse.mergedFromShadowState.find(canon); it != reuse.mergedFromShadowState.end())
      return getInnerBufferTypeForMaterializeState(it->second.getType());
   return nullptr;
}

static bool materializeTargetsJoinBuffer(subop::MaterializeOp mat, mlir::Value mergedBuffer,
                                        const ModuleReuseInfo& reuse) {
   subop::BufferType targetBuf = bufferTypeForJoinTarget(mergedBuffer, reuse);
   if (!targetBuf) return false;
   subop::BufferType matBuf = getInnerBufferTypeForMaterializeState(mat.getState().getType());
   if (!matBuf) return false;
   if (matBuf == targetBuf) return true;
   if (matBuf.getMembers() == targetBuf.getMembers()) return true;
   mlir::Value canonBuf = canonicalizeStateValueForReuse(mergedBuffer);
   if (auto itTL = reuse.mergedFromShadowState.find(canonBuf); itTL != reuse.mergedFromShadowState.end()) {
      if (mat.getState() == itTL->second) return true;
      if (subop::BufferType tlBuf = getInnerBufferTypeForMaterializeState(itTL->second.getType());
          tlBuf && tlBuf.getMembers() == matBuf.getMembers()) {
         return true;
      }
   }
   return false;
}

static subop::MaterializeOp findJoinBufferMaterializeInStep(subop::ExecutionStepOp buildStep) {
   subop::MaterializeOp matOp;
   buildStep.walk([&](subop::MaterializeOp m) {
      if (!getInnerBufferTypeForMaterializeState(m.getState().getType())) return;
      matOp = m;
   });
   return matOp;
}

static subop::MapOp findJoinHashMapBeforeMaterialize(mlir::Block& body, subop::MaterializeOp matOp) {
   subop::MapOp mapOp;
   for (mlir::Operation& op : body.without_terminator()) {
      if (!op.isBeforeInBlock(matOp.getOperation())) continue;
      if (auto m = mlir::dyn_cast<subop::MapOp>(&op)) mapOp = m;
   }
   return mapOp;
}

/// Join-build hash map key column → buffer \c materialize member (not HIV value-member slot order).
static std::string joinKeyMemberNameFromBuildStep(subop::ExecutionStepOp buildStep, subop::Member linkMember,
                                                  subop::Member hashMember, subop::MemberManager& mm,
                                                  tuples::ColumnManager& cm) {
   subop::MaterializeOp mat = findJoinBufferMaterializeInStep(buildStep);
   assert(mat && "join superset: build step must materialize join buffer");
   mlir::Block& body = buildStep.getSubOps().front();
   subop::MapOp hashMap = findJoinHashMapBeforeMaterialize(body, mat);
   if (!hashMap || hashMap.getInputCols().empty()) return "";
   auto keyRef = mlir::cast<tuples::ColumnRefAttr>(hashMap.getInputCols()[0]);
   auto [keyScope, keyLeaf] = cm.getName(&keyRef.getColumn());
   for (auto& [member, colRef] : mat.getMapping().getMapping()) {
      if (member == linkMember || member == hashMember) continue;
      auto [scope, leaf] = cm.getName(&colRef.getColumn());
      if (scope == keyScope && leaf == keyLeaf) return mm.getName(member);
   }
   return "";
}

static mlir::Value resolveJoinMergedBuffer(mlir::Value hivOrBuf, mlir::ModuleOp module, const ModuleReuseInfo& reuse) {
   mlir::Value canon = resolveCacheTargetStateForReuse(hivOrBuf, reuse);
   if (auto it = findReuseMap(reuse.mergedFromShadowState, canon);
       it != reuse.mergedFromShadowState.end()) {
      return it->second;
   }
   if (bufferTypeForJoinTarget(canon, reuse)) return canon;
   mlir::Value fromChiv;
   module.walk([&](subop::CreateHashIndexedView chiv) {
      if (fromChiv) return;
      if (canonicalizeStateValueForReuse(chiv.getResult()) != canon) return;
      fromChiv = chiv.getSource();
   });
   return fromChiv ? fromChiv : canon;
}

static subop::ExecutionStepOp findBufferBuildStepWithTableMaterialize(mlir::Value mergedBuffer,
                                                                      const ModuleReuseInfo& reuse);
static subop::ExecutionStepOp findBufferBuildStepWithTableScan(mlir::ModuleOp module);

static bool stepHasTableScan(subop::ExecutionStepOp step) {
   bool hasTableScan = false;
   step.walk([&](subop::ScanRefsOp scan) {
      if (mlir::isa<subop::TableType>(scan.getState().getType())) hasTableScan = true;
   });
   return hasTableScan;
}

static bool materializeWritesExactState(subop::MaterializeOp mat, mlir::Value targetState) {
   mlir::Value matState = peelBlockArgsToEnclosingOperands(mat.getState());
   mlir::Value target = peelBlockArgsToEnclosingOperands(targetState);
   return canonicalizeStateValueForReuse(matState) == canonicalizeStateValueForReuse(target);
}

static subop::ExecutionStepOp findExactJoinBuildStepForState(mlir::Value buildState,
                                                             const ModuleReuseInfo& reuse) {
   auto tryWriters = [&](mlir::Value key) -> subop::ExecutionStepOp {
      auto itW = reuse.writerStepsByState.find(key);
      if (itW == reuse.writerStepsByState.end()) return {};
      for (subop::ExecutionStepOp step : itW->second) {
         if (!stepHasTableScan(step)) continue;
         subop::MaterializeOp matOp;
         step.walk([&](subop::MaterializeOp mat) {
            if (matOp) return;
            if (!materializeWritesExactState(mat, buildState)) return;
            matOp = mat;
         });
         if (matOp) return step;
      }
      return {};
   };
   if (subop::ExecutionStepOp step = tryWriters(buildState)) return step;
   mlir::Value canon = canonicalizeStateValueForReuse(buildState);
   if (canon != buildState) {
      if (subop::ExecutionStepOp step = tryWriters(canon)) return step;
   }
   return {};
}

static subop::ExecutionStepOp findJoinBufferBuildStepFromWriterChain(mlir::Value mergedBuffer,
                                                                     const ModuleReuseInfo& reuse) {
   if (auto it = reuse.mergedFromShadowState.find(canonicalizeStateValueForReuse(mergedBuffer));
       it != reuse.mergedFromShadowState.end()) {
      if (subop::ExecutionStepOp step = findExactJoinBuildStepForState(it->second, reuse)) return step;
   }
   if (subop::ExecutionStepOp step = findExactJoinBuildStepForState(mergedBuffer, reuse)) return step;

   llvm::SmallVector<mlir::Value, 4> candidates;
   auto addCandidate = [&](mlir::Value state) {
      if (!state) return;
      candidates.push_back(state);
      mlir::Value canon = canonicalizeStateValueForReuse(state);
      if (canon != state) candidates.push_back(canon);
   };

   addCandidate(mergedBuffer);
   mlir::Value canonMerged = canonicalizeStateValueForReuse(mergedBuffer);
   if (auto it = reuse.mergedFromShadowState.find(canonMerged); it != reuse.mergedFromShadowState.end())
      addCandidate(it->second);

   llvm::DenseSet<void*> seenSteps;
   for (mlir::Value candidate : candidates) {
      auto tryWriters = [&](mlir::Value key) -> subop::ExecutionStepOp {
         auto itW = reuse.writerStepsByState.find(key);
         if (itW == reuse.writerStepsByState.end()) return {};
         for (subop::ExecutionStepOp step : itW->second) {
            if (!seenSteps.insert(step.getOperation()).second) continue;
            if (!stepHasTableScan(step)) continue;
            subop::MaterializeOp matOp;
            step.walk([&](subop::MaterializeOp mat) {
               if (matOp) return;
               if (!materializeTargetsJoinBuffer(mat, mergedBuffer, reuse)) return;
               matOp = mat;
            });
            if (matOp) return step;
         }
         return {};
      };
      if (subop::ExecutionStepOp step = tryWriters(candidate)) return step;
   }
   return {};
}

static subop::ExecutionStepOp findJoinBufferBuildStepForHiv(mlir::ModuleOp module, mlir::Value hiv,
                                                            const ModuleReuseInfo& reuse) {
   mlir::Value buf = resolveJoinMergedBuffer(hiv, module, reuse);
   if (subop::ExecutionStepOp step = findJoinBufferBuildStepFromWriterChain(buf, reuse)) return step;
   if (subop::ExecutionStepOp step = findBufferBuildStepWithTableMaterialize(buf, reuse)) return step;
   return findBufferBuildStepWithTableScan(module);
}

static subop::ExecutionStepOp findStrictJoinBufferBuildStepForHiv(mlir::ModuleOp module, mlir::Value hiv,
                                                                  const ModuleReuseInfo& reuse) {
   mlir::Value buf = resolveJoinMergedBuffer(hiv, module, reuse);
   if (subop::ExecutionStepOp step = findJoinBufferBuildStepFromWriterChain(buf, reuse)) return step;
   return findBufferBuildStepWithTableMaterialize(buf, reuse);
}

static subop::ExecutionStepOp findBufferBuildStepWithTableMaterialize(mlir::Value mergedBuffer,
                                                                        const ModuleReuseInfo& reuse) {
   for (const ModuleReuseInfo::StepRW& rw : reuse.steps) {
      subop::ExecutionStepOp step = rw.step;
      bool hasTableScan = stepHasTableScan(step);
      bool hasJoinMat = false;
      step->walk([&](subop::MaterializeOp mat) {
         if (materializeTargetsJoinBuffer(mat, mergedBuffer, reuse)) hasJoinMat = true;
      });
      if (hasTableScan && hasJoinMat) return step;
   }
   return {};
}

static subop::ExecutionStepOp findBufferBuildStepWithTableScan(mlir::ModuleOp module) {
   subop::ExecutionStepOp found;
   module.walk([&](subop::ExecutionStepOp step) {
      if (found) return;
      bool hasTableScan = false;
      bool hasBufferMat = false;
      step->walk([&](subop::ScanRefsOp scan) {
         if (mlir::isa<subop::TableType>(scan.getState().getType())) hasTableScan = true;
      });
      step->walk([&](subop::MaterializeOp mat) {
         if (getInnerBufferTypeForMaterializeState(mat.getState().getType())) hasBufferMat = true;
      });
      if (hasTableScan && hasBufferMat) found = step;
   });
   return found;
}

static void insertPayloadColumnSpec(llvm::DenseMap<uint64_t, PayloadColumnSpec>& unionCols,
                                    PayloadColumnSpec spec,
                                    std::optional<unsigned> layoutSideIndex = std::nullopt);

static void collectPayloadFromMaterialize(
   subop::MaterializeOp mat, llvm::StringRef linkMemberName, llvm::StringRef hashMemberName, subop::MemberManager& mm,
   lingodb::compiler::dialect::tuples::ColumnManager& cm, llvm::StringRef joinKeyMemberName,
   llvm::DenseMap<uint64_t, PayloadColumnSpec>& out, std::optional<unsigned> layoutSideIndex,
   std::optional<unsigned> reuseQueryIndex, const llvm::DenseMap<const void*, uint64_t>& columnHashes) {
   for (auto& [member, colRef] : mat.getMapping().getMapping()) {
      llvm::StringRef memName = mm.getName(member);
      if (memName == linkMemberName || memName == hashMemberName || isJoinBufferInternalMemberName(memName))
         continue;
      auto [scope, leaf] = cm.getName(&colRef.getColumn());
      PayloadColumnSpec spec;
      spec.colType = colRef.getColumn().type;
      spec.isJoinKey = !joinKeyMemberName.empty() && memName == joinKeyMemberName;
      if (reuseQueryIndex && isFilterPredPayloadColumn(memName, leaf)) {
         spec.scope = kReuseFilterPredScope.str();
         spec.leaf = llvm::Twine(*reuseQueryIndex).str();
         spec.semanticKey = reuseFilterPredSemanticKey(*reuseQueryIndex);
         spec.semanticHash = payloadSyntheticColumnHash(spec.scope, spec.leaf, spec.colType);
      } else {
         spec.scope = scope;
         spec.leaf = leaf;
         spec.semanticKey = columnSemanticKey(scope, leaf);
         spec.semanticHash = payloadColumnIdentityHash(colRef, columnHashes);
      }
      insertPayloadColumnSpec(out, std::move(spec), layoutSideIndex);
   }
}

static mlir::Type memberTypeForIdentifier(subop::TableType tableTy, subop::MemberManager& mm, llvm::StringRef identifier);
static subop::Member tableMemberForIdentifier(subop::TableType tableTy, subop::MemberManager& mm,
                                              llvm::StringRef identifier);

static ExternalDatasourceProperty mergeExternalDatasource(const ExternalDatasourceProperty& a,
                                                        const ExternalDatasourceProperty& b);

/// Unique \c get_external in an external-table construction step (\c isExternalTableRefStep).
static subop::GetExternalOp findUniqueGetExternalInTableRefStep(subop::ExecutionStepOp tableStep) {
   assert(isExternalTableRefStep(tableStep) && "expected external table_ref construction step");
   subop::GetExternalOp ge;
   for (mlir::Operation& op : tableStep.getSubOps().front().without_terminator()) {
      if (auto g = mlir::dyn_cast<subop::GetExternalOp>(&op)) {
         assert(!ge && "table_ref step must contain exactly one get_external");
         ge = g;
      }
   }
   assert(ge && "table_ref step must contain get_external");
   return ge;
}

/// Resolve the external table backing \c scan_refs: block-arg → step operand → reuse map → table_ref step.
static bool resolveScannedTableExternal(subop::ExecutionStepOp buildStep, mlir::Value tableStateInBody,
                                       const ModuleReuseInfo& reuse, llvm::StringRef& tableName,
                                       ExternalDatasourceProperty& ds, bool& haveDs, subop::TableType& tableTy) {
   mlir::Value external = tableStateInBody;
   if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(external)) {
      if (ba.getOwner() == &buildStep.getSubOps().front() && ba.getArgNumber() < buildStep.getNumOperands()) {
         external = buildStep.getOperand(ba.getArgNumber());
      }
   }
   external = peelBlockArgsToEnclosingOperands(external);

   if (auto ge = mlir::dyn_cast_or_null<subop::GetExternalOp>(external.getDefiningOp())) {
      ds = lingodb::utility::deserializeFromHexString<ExternalDatasourceProperty>(ge.getDescr());
      tableName = ds.tableName;
      haveDs = true;
      tableTy = mlir::cast<subop::TableType>(ge.getResult().getType());
      assert(!tableName.empty());
      return true;
   }

   mlir::Value canon = canonicalizeStateValueForReuse(external);
   if (auto it = findReuseMap(reuse.externalDatasourceByTableState, canon);
       it != reuse.externalDatasourceByTableState.end()) {
      ds = it->second;
      tableName = ds.tableName;
      haveDs = true;
      tableTy = mlir::cast<subop::TableType>(external.getType());
      assert(!tableName.empty());
      return true;
   }

   if (auto tableStep = mlir::dyn_cast_or_null<subop::ExecutionStepOp>(external.getDefiningOp())) {
      assert(isExternalTableRefStep(tableStep) &&
             "scan_refs table state must come from external table_ref construction step");
      subop::GetExternalOp ge = findUniqueGetExternalInTableRefStep(tableStep);
      ds = lingodb::utility::deserializeFromHexString<ExternalDatasourceProperty>(ge.getDescr());
      tableName = ds.tableName;
      haveDs = true;
      tableTy = mlir::cast<subop::TableType>(ge.getResult().getType());
      assert(!tableName.empty());
      return true;
   }
   return false;
}

static subop::ScanRefsOp findTableScanRefsForExternalTable(subop::ExecutionStepOp buildStep, llvm::StringRef tableName,
                                                         const ModuleReuseInfo& reuse) {
   mlir::Block& body = buildStep.getSubOps().front();
   for (mlir::Operation& op : body.without_terminator()) {
      auto scanOp = mlir::dyn_cast<subop::ScanRefsOp>(&op);
      if (!scanOp || !mlir::isa<subop::TableType>(scanOp.getState().getType())) continue;
      llvm::StringRef resolved;
      ExternalDatasourceProperty ds;
      bool haveDs = false;
      subop::TableType tableTy;
      if (!resolveScannedTableExternal(buildStep, scanOp.getState(), reuse, resolved, ds, haveDs, tableTy) ||
          !haveDs) {
         continue;
      }
      if (resolved == tableName) return scanOp;
   }
   return {};
}

struct ResolvedExternalTableScan {
   subop::ScanRefsOp scanOp;
   llvm::StringRef tableName;
   ExternalDatasourceProperty datasource;
   subop::TableType tableType;
};

static std::optional<ResolvedExternalTableScan> resolveExternalTableScanForDonor(subop::ExecutionStepOp buildStep,
                                                                                 const ModuleReuseInfo& reuse,
                                                                                 llvm::StringRef donorTableName) {
   subop::ScanRefsOp scanOp = findTableScanRefsForExternalTable(buildStep, donorTableName, reuse);
   if (!scanOp) return std::nullopt;
   ResolvedExternalTableScan resolved;
   resolved.scanOp = scanOp;
   bool haveDs = false;
   bool ok = resolveScannedTableExternal(buildStep, scanOp.getState(), reuse, resolved.tableName, resolved.datasource,
                                         haveDs, resolved.tableType);
   if (!ok || !haveDs || resolved.tableName != donorTableName)
      llvm_unreachable("join superset: peer table scan must resolve to donor table");
   return resolved;
}

static unsigned countUnionPayloadLeavesOnTable(llvm::ArrayRef<PayloadColumnSpec> payloadColumns,
                                               subop::TableType tableTy, subop::MemberManager& mm) {
   unsigned overlap = 0;
   for (const PayloadColumnSpec& spec : payloadColumns) {
      unsigned qIdx = 0;
      if (parseFilterPredLayoutSemanticKey(spec.semanticKey, qIdx)) continue;
      if (tableMemberForIdentifier(tableTy, mm, spec.leaf)) ++overlap;
   }
   return overlap;
}

static void insertPayloadColumnSpec(llvm::DenseMap<uint64_t, PayloadColumnSpec>& unionCols,
                                    PayloadColumnSpec spec,
                                    std::optional<unsigned> layoutSideIndex) {
   assert(spec.semanticHash && "payload spec must carry a hash identity");
   auto markSide = [&](PayloadColumnSpec& existing) {
      if (layoutSideIndex && *layoutSideIndex == 0) existing.inQuery0 = true;
      if (layoutSideIndex && *layoutSideIndex == 1) existing.inQuery1 = true;
   };

   auto it = unionCols.find(spec.semanticHash);
   if (it != unionCols.end()) {
      markSide(it->second);
      it->second.isJoinKey |= spec.isJoinKey;
      it->second.fromExternalTable |= spec.fromExternalTable;
      return;
   }

   llvm::StringRef specLeaf = normalizeColumnIdentifier(spec.leaf);
   for (auto existingIt = unionCols.begin(), e = unionCols.end(); existingIt != e; ++existingIt) {
      PayloadColumnSpec& existing = existingIt->second;
      unsigned predIdx = 0;
      if (parseFilterPredLayoutSemanticKey(existing.semanticKey, predIdx)) continue;
      if (normalizeColumnIdentifier(existing.leaf) != specLeaf) continue;
      if (existing.colType != spec.colType) continue;
      if (!spec.fromExternalTable && !existing.fromExternalTable) continue;

      if (existing.fromExternalTable && !spec.fromExternalTable) {
         markSide(existing);
         existing.isJoinKey |= spec.isJoinKey;
         return;
      }

      spec.inQuery0 |= existing.inQuery0;
      spec.inQuery1 |= existing.inQuery1;
      spec.isJoinKey |= existing.isJoinKey;
      markSide(spec);
      unionCols.erase(existingIt);
      unionCols.try_emplace(spec.semanticHash, std::move(spec));
      return;
   }

   markSide(spec);
   unionCols.try_emplace(spec.semanticHash, std::move(spec));
}

static subop::ScanRefsOp findTableScanForPayloadLeaf(subop::ExecutionStepOp buildStep, llvm::StringRef leaf,
                                                   subop::MemberManager& mm) {
   subop::ScanRefsOp found;
   buildStep.walk([&](subop::ScanRefsOp scanOp) {
      if (found) return;
      auto tableTy = mlir::dyn_cast<subop::TableType>(scanOp.getState().getType());
      if (!tableTy) return;
      if (tableMemberForIdentifier(tableTy, mm, leaf)) found = scanOp;
   });
   return found;
}

static subop::ScanRefsOp findDonorTableScanInUnionPlan(subop::ExecutionStepOp buildStep,
                                                       const JoinBufferUnionPlan& plan,
                                                       const ModuleReuseInfo& reuse) {
   (void)reuse;
   auto& mm = buildStep.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   subop::ScanRefsOp bestScan;
   unsigned bestOverlap = 0;
   buildStep.walk([&](subop::ScanRefsOp scanOp) {
      auto tableTy = mlir::dyn_cast<subop::TableType>(scanOp.getState().getType());
      if (!tableTy) return;
      unsigned overlap = countUnionPayloadLeavesOnTable(plan.payloadColumns, tableTy, mm);
      if (overlap > bestOverlap) {
         bestOverlap = overlap;
         bestScan = scanOp;
      }
   });
   return bestScan;
}

static void ingestExternalTableColumnsFromBuildStepScan(subop::ExecutionStepOp buildStep,
                                                        const ModuleReuseInfo& reuse,
                                                        llvm::DenseMap<uint64_t, PayloadColumnSpec>& unionCols) {
   llvm::SmallVector<PayloadColumnSpec, 16> unionSpecs;
   unionSpecs.reserve(unionCols.size());
   for (auto& it : unionCols) unionSpecs.push_back(it.second);

   auto& mm = buildStep.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   bool sawTableScan = false;
   buildStep.walk([&](subop::ScanRefsOp scanOp) {
      if (!mlir::isa<subop::TableType>(scanOp.getState().getType())) return;
      sawTableScan = true;

      llvm::StringRef tableName;
      ExternalDatasourceProperty ds;
      bool haveDs = false;
      subop::TableType tableTy;
      if (!resolveScannedTableExternal(buildStep, scanOp.getState(), reuse, tableName, ds, haveDs, tableTy) ||
          !haveDs) {
         return;
      }
      if (countUnionPayloadLeavesOnTable(unionSpecs, tableTy, mm) == 0) return;

      for (const PayloadColumnSpec& existing : unionSpecs) {
         unsigned predIdx = 0;
         if (parseFilterPredLayoutSemanticKey(existing.semanticKey, predIdx)) continue;
         llvm::StringRef leaf = normalizeColumnIdentifier(existing.leaf);
         if (!tableMemberForIdentifier(tableTy, mm, leaf)) continue;
         PayloadColumnSpec spec;
         spec.scope = tableName.str();
         spec.leaf = leaf.str();
         spec.colType = memberTypeForIdentifier(tableTy, mm, leaf);
         assert(spec.colType && "external table mapping column must exist on scanned table type");
         spec.semanticKey = columnSemanticKey(spec.scope, spec.leaf);
         spec.semanticHash = existing.semanticHash;
         spec.isJoinKey = existing.isJoinKey;
         spec.inQuery0 = existing.inQuery0;
         spec.inQuery1 = existing.inQuery1;
         spec.fromExternalTable = true;
         insertPayloadColumnSpec(unionCols, std::move(spec));
	      }
	   });
   if (!sawTableScan) llvm_unreachable("join union ingest: build step must contain scan_refs on a table state");
}

static void mergePeerExternalFromBuildStepScan(ExternalDatasourceProperty& merged, bool& haveMerged,
                                               subop::ExecutionStepOp peerBuild, const ModuleReuseInfo& reusePeer,
                                               llvm::StringRef donorTableName) {
   auto resolved = resolveExternalTableScanForDonor(peerBuild, reusePeer, donorTableName);
   if (!resolved) return;
   if (!haveMerged) {
      merged = resolved->datasource;
      haveMerged = true;
   } else {
      merged = mergeExternalDatasource(merged, resolved->datasource);
   }
}

static mlir::Type columnTypeForIdentifierFromPeerBuildScan(subop::ExecutionStepOp peerBuild,
                                                           const ModuleReuseInfo& reusePeer,
                                                           llvm::StringRef donorTableName, llvm::StringRef identifier) {
   auto resolved = resolveExternalTableScanForDonor(peerBuild, reusePeer, donorTableName);
   if (!resolved) return {};
   auto& mm = peerBuild.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   mlir::Type ty = memberTypeForIdentifier(resolved->tableType, mm, identifier);
   assert(ty && "peer scanned table must contain union payload column");
   return ty;
}

static bool tryGetHivDonorExternalDatasource(mlir::ModuleOp module, mlir::Value hiv, const ModuleReuseInfo& reuse,
                                             ExternalDatasourceProperty& outDs) {
   mlir::Value canon = resolveCacheTargetStateForReuse(hiv, reuse);
   mlir::Value buf = canon;
   if (auto it = reuse.mergedFromShadowState.find(canon); it != reuse.mergedFromShadowState.end()) {
      buf = it->second;
   }
   subop::ExecutionStepOp buildStep = findBufferBuildStepWithTableMaterialize(buf, reuse);
   if (!buildStep) buildStep = findBufferBuildStepWithTableScan(module);
   if (!buildStep) return false;

   bool found = false;
   buildStep.walk([&](subop::ScanRefsOp scanOp) {
      if (found) return;
      if (!mlir::isa<subop::TableType>(scanOp.getState().getType())) return;
      llvm::StringRef tableName;
      ExternalDatasourceProperty ds;
      bool haveDs = false;
      subop::TableType tableTy;
      if (!resolveScannedTableExternal(buildStep, scanOp.getState(), reuse, tableName, ds, haveDs, tableTy) ||
          !haveDs) {
         return;
      }
      outDs = ds;
      found = true;
   });
   return found;
}

static JoinBufferUnionPlan buildUnionPlan(mlir::Value hivA, mlir::Value hivB, const ModuleReuseInfo& reuseA,
                                        const ModuleReuseInfo& reuseB, mlir::ModuleOp modA, mlir::ModuleOp modB,
                                        unsigned queryIndexA, unsigned queryIndexB,
                                        bool enableFilterPredReuse) {
   JoinBufferUnionPlan plan;
   mlir::MLIRContext* ctxA = hivA.getContext();
   auto* subDialectA = ctxA->getLoadedDialect<subop::SubOperatorDialect>();
   auto* tupleDialectA = ctxA->getLoadedDialect<tuples::TupleStreamDialect>();
   assert(subDialectA && tupleDialectA);
   auto& mm = subDialectA->getMemberManager();
   auto& cm = tupleDialectA->getColumnManager();

   mlir::Value hiv = resolveCacheTargetStateForReuse(hivA, reuseA);
   subop::CreateHashIndexedView chiv = findCreateHashIndexedViewForState(hiv, reuseA.writerStepsByState);
   assert(chiv && "join superset: HIV must have create_hash_indexed_view writer");
   plan.linkMember = chiv.getLinkMember().getMember();
   plan.hashMember = chiv.getHashMember().getMember();
   std::string joinKeyMemberName;
   subop::ExecutionStepOp buildStep = findJoinBufferBuildStepForHiv(modA, hivA, reuseA);
   assert(buildStep && "join superset: HIV build step required for union plan");
   joinKeyMemberName = joinKeyMemberNameFromBuildStep(buildStep, plan.linkMember, plan.hashMember, mm, cm);
   llvm::StringRef linkMemberName = mm.getName(plan.linkMember);
   llvm::StringRef hashMemberName = mm.getName(plan.hashMember);

   llvm::DenseMap<uint64_t, PayloadColumnSpec> unionCols;
   auto ingestHiv = [&](mlir::Value h, const ModuleReuseInfo& reuse, mlir::ModuleOp mod,
                        unsigned layoutSideIndex, unsigned reuseQueryIndex) {
      llvm::DenseMap<const void*, uint64_t> columnHashes = collectStateConstructionColumnHashes(mod, h);
      auto* hCtx = h.getContext();
      auto& hMm = hCtx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
      auto& hCm = hCtx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      mlir::Value mergedBuf = resolveJoinMergedBuffer(h, mod, reuse);
      subop::ExecutionStepOp buildStep = findJoinBufferBuildStepForHiv(mod, h, reuse);
      assert(buildStep && "join superset: build step required for ingestHiv");
      buildStep.walk([&](subop::MaterializeOp mat) {
         if (!getInnerBufferTypeForMaterializeState(mat.getState().getType()) &&
             !materializeTargetsJoinBuffer(mat, mergedBuf, reuse)) {
            return;
         }
         collectPayloadFromMaterialize(mat, linkMemberName, hashMemberName, hMm, hCm, joinKeyMemberName, unionCols,
                                       layoutSideIndex, reuseQueryIndex, columnHashes);
      });
      if (enableFilterPredReuse) {
         PayloadColumnSpec predSpec;
         predSpec.scope = kReuseFilterPredScope.str();
         predSpec.leaf = llvm::Twine(reuseQueryIndex).str();
         predSpec.colType = mlir::IntegerType::get(h.getContext(), 1);
         predSpec.semanticKey = reuseFilterPredSemanticKey(reuseQueryIndex);
         predSpec.semanticHash = payloadSyntheticColumnHash(predSpec.scope, predSpec.leaf, predSpec.colType);
         unionCols.try_emplace(predSpec.semanticHash, predSpec);
      }
      ingestExternalTableColumnsFromBuildStepScan(buildStep, reuse, unionCols);
   };
   ingestHiv(hivA, reuseA, modA, /*layoutSideIndex=*/0, queryIndexA);
   ingestHiv(hivB, reuseB, modB, /*layoutSideIndex=*/1, queryIndexB);
   if (enableFilterPredReuse) {
      unsigned unionSlot = filterPredUnionSlotForQueryIndices({queryIndexA, queryIndexB});
      if (unionSlot < 8) {
         PayloadColumnSpec predSpec;
         predSpec.scope = kReuseFilterPredUnionScope.str();
         predSpec.leaf = llvm::Twine(unionSlot).str();
         predSpec.colType = mlir::IntegerType::get(hivA.getContext(), 1);
         predSpec.semanticKey = reuseFilterPredUnionSemanticKey(unionSlot);
         predSpec.semanticHash = payloadSyntheticColumnHash(predSpec.scope, predSpec.leaf, predSpec.colType);
         unionCols.try_emplace(predSpec.semanticHash, predSpec);
      }
   }

   llvm::SmallVector<PayloadColumnSpec*, 8> ordered;
   ordered.reserve(unionCols.size());
   for (auto& it : unionCols) ordered.push_back(&it.second);
   llvm::sort(ordered, [](const PayloadColumnSpec* a, const PayloadColumnSpec* b) {
      if (a->isJoinKey != b->isJoinKey) return a->isJoinKey > b->isJoinKey;
      if (a->semanticHash != b->semanticHash) return a->semanticHash < b->semanticHash;
      return a->semanticKey < b->semanticKey;
   });
   for (PayloadColumnSpec* p : ordered) {
      plan.payloadColumns.push_back(*p);
      plan.payloadMemberTypes.push_back(p->colType);
   }
   return plan;
}

static std::optional<unsigned> parseMemberSlot(llvm::StringRef name) {
   if (!name.consume_front("member$")) return std::nullopt;
   unsigned slot = 0;
   if (name.getAsInteger(10, slot)) return std::nullopt;
   return slot;
}

/// Next `member$N` slot after pre-existing and already-assigned payload members.
static unsigned nextPayloadMemberSlot(subop::MemberManager& mm, const llvm::StringMap<subop::Member>& existingByName,
                                      llvm::ArrayRef<subop::Member> assigned) {
   unsigned maxSlot = 0;
   auto bump = [&](subop::Member m) {
      if (auto slot = parseMemberSlot(mm.getName(m))) maxSlot = std::max(maxSlot, *slot + 1);
      if (auto predSlot = parseFilterPredMemberSlot(mm.getName(m))) maxSlot = std::max(maxSlot, *predSlot + 1);
   };
   for (const auto& it : existingByName) bump(it.second);
   for (subop::Member m : assigned) bump(m);
   return maxSlot;
}

static subop::Member allocUnusedPayloadMemberSlot(subop::MemberManager& mm, mlir::Type colType, unsigned& nextSlot) {
   for (;; ++nextSlot) {
      std::string name = "member$" + std::to_string(nextSlot);
      if (!mm.hasMemberDirect(name)) return mm.createMemberDirect(name, colType);
   }
}

static mlir::Type cloneTypeToContext(mlir::Type ty, mlir::MLIRContext* ctx) {
   if (!ty) return ty;
   if (auto nullable = mlir::dyn_cast<db::NullableType>(ty))
      return db::NullableType::get(cloneTypeToContext(nullable.getType(), ctx));
   if (auto tuple = mlir::dyn_cast<mlir::TupleType>(ty)) {
      llvm::SmallVector<mlir::Type> types;
      for (mlir::Type elem : tuple.getTypes()) types.push_back(cloneTypeToContext(elem, ctx));
      return mlir::TupleType::get(ctx, types);
   }
   if (auto i = mlir::dyn_cast<mlir::IntegerType>(ty)) {
      return mlir::IntegerType::get(ctx, i.getWidth(), i.getSignedness());
   }
   if (mlir::isa<mlir::IndexType>(ty)) return mlir::IndexType::get(ctx);
   if (auto c = mlir::dyn_cast<db::CharType>(ty)) return db::CharType::get(ctx, c.getLen());
   if (mlir::isa<db::StringType>(ty)) return db::StringType::get(ctx);
   if (auto d = mlir::dyn_cast<db::DateType>(ty)) return db::DateType::get(ctx, d.getUnit());
   if (auto t = mlir::dyn_cast<db::TimestampType>(ty)) return db::TimestampType::get(ctx, t.getUnit());
   if (auto dec = mlir::dyn_cast<db::DecimalType>(ty)) return db::DecimalType::get(ctx, dec.getP(), dec.getS());
   if (auto f = mlir::dyn_cast<mlir::FloatType>(ty)) {
      if (f.isF64()) return mlir::Float64Type::get(ctx);
      if (f.isF32()) return mlir::Float32Type::get(ctx);
      if (f.isF16()) return mlir::Float16Type::get(ctx);
   }
   if (auto iv = mlir::dyn_cast<db::IntervalType>(ty))
      return db::IntervalType::get(ctx, iv.getUnit());
   if (auto ref = mlir::dyn_cast<util::RefType>(ty))
      return util::RefType::get(ctx, cloneTypeToContext(ref.getElementType(), ctx));
   if (auto buf = mlir::dyn_cast<util::BufferType>(ty))
      return util::BufferType::get(ctx, cloneTypeToContext(buf.getT(), ctx));
   if (mlir::isa<util::VarLen32Type>(ty)) return util::VarLen32Type::get(ctx);
   llvm_unreachable("cloneTypeToContext: unsupported type for cross-context layout clone");
}

struct ResidualTableFilter {
   subop::ScanRefsOp scan;
   subop::MapOp predMap;
   subop::FilterOp filter;
};

static bool streamValueHasOnlyUseByWithinStep(mlir::Value v, mlir::Operation* expectedUser,
                                              subop::ExecutionStepOp step) {
   mlir::Operation* onlyUser = nullptr;
   auto isWithinStep = [&](mlir::Operation* op) {
      for (mlir::Operation* cur = op; cur; cur = cur->getParentOp())
         if (cur == step.getOperation()) return true;
      return false;
   };
   for (mlir::OpOperand& use : v.getUses()) {
      mlir::Operation* user = use.getOwner();
      if (!isWithinStep(user)) continue;
      if (onlyUser && onlyUser != user) return false;
      onlyUser = user;
   }
   return onlyUser == expectedUser;
}

static subop::ScanRefsOp traceSingleUseStreamToTableScanInStep(subop::ExecutionStepOp step,
                                                               mlir::Operation* user, mlir::Value stream) {
   for (;;) {
      mlir::Operation* def = stream.getDefiningOp();
      if (!def) return {};
      if (!streamValueHasOnlyUseByWithinStep(stream, user, step)) return {};
      if (auto scan = mlir::dyn_cast<subop::ScanRefsOp>(def)) {
         if (mlir::isa<subop::TableType>(scan.getState().getType())) return scan;
         return {};
      }
      if (auto gather = mlir::dyn_cast<subop::GatherOp>(def)) {
         user = def;
         stream = gather.getStream();
         continue;
      }
      if (auto map = mlir::dyn_cast<subop::MapOp>(def)) {
         user = def;
         stream = map.getStream();
         continue;
      }
      if (auto rename = mlir::dyn_cast<subop::RenamingOp>(def)) {
         user = def;
         stream = rename.getStream();
         continue;
      }
      return {};
   }
}

static std::optional<ResidualTableFilter> findResidualTableFilterInBuildStep(subop::ExecutionStepOp step) {
   std::optional<ResidualTableFilter> found;
   step.walk([&](subop::FilterOp filter) {
      if (found) return;
      auto map = mlir::dyn_cast_or_null<subop::MapOp>(filter.getStream().getDefiningOp());
      if (!map) return;
      if (!streamValueHasOnlyUseByWithinStep(map.getResult(), filter.getOperation(), step)) return;
      llvm::DenseSet<const void*> computedCols;
      for (auto attr : map.getComputedCols()) {
         auto def = mlir::cast<tuples::ColumnDefAttr>(attr);
         computedCols.insert(&def.getColumn());
      }
      for (auto attr : filter.getConditions()) {
         auto ref = mlir::cast<tuples::ColumnRefAttr>(attr);
         if (!computedCols.contains(&ref.getColumn())) return;
      }
      subop::ScanRefsOp scan = traceSingleUseStreamToTableScanInStep(step, map.getOperation(), map.getStream());
      if (!scan) return;
      found = ResidualTableFilter{scan, map, filter};
   });
   return found;
}

static std::string residualFilterFingerprint(ResidualTableFilter f) {
   std::string s;
   llvm::raw_string_ostream os(s);
   f.predMap.getOperation()->print(os);
   f.filter.getFilterSemanticAttr().print(os);
   f.filter.getConditions().print(os);
   os.flush();
   return s;
}

static bool residualTableFiltersIdentical(subop::ExecutionStepOp a, subop::ExecutionStepOp b) {
   auto fa = findResidualTableFilterInBuildStep(a);
   auto fb = findResidualTableFilterInBuildStep(b);
   if (!fa && !fb) return true;
   if (!fa || !fb) return false;
   return residualFilterFingerprint(*fa) == residualFilterFingerprint(*fb);
}

static unsigned residualFilterConditionResultIndex(subop::MapOp map, subop::FilterOp filter) {
   assert(filter.getConditions().size() == 1 && "residual filter: expected a single predicate condition");
   auto cond = mlir::cast<tuples::ColumnRefAttr>(filter.getConditions()[0]);
   for (unsigned i = 0; i < map.getComputedCols().size(); ++i) {
      auto def = mlir::cast<tuples::ColumnDefAttr>(map.getComputedCols()[i]);
      if (&def.getColumn() == &cond.getColumn()) return i;
   }
   llvm_unreachable("residual filter: filter condition must be produced by predicate map");
}

static tuples::ColumnDefAttr makeResidualFilterPredDef(mlir::MLIRContext* ctx, unsigned qIdx) {
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   tuples::ColumnDefAttr def = cm.createDef(cm.getUniqueScope("residual_filter_pred$" + llvm::Twine(qIdx).str()),
                                            "filter_pred");
   def.getColumn().type = mlir::IntegerType::get(ctx, 1);
   return def;
}

static void setMaterializeMapping(subop::MaterializeOp mat, subop::Member member,
                                  tuples::ColumnRefAttr ref) {
   llvm::SmallVector<subop::RefMappingPairT> pairs;
   bool replaced = false;
   for (auto pr : mat.getMapping().getMapping()) {
      if (pr.first == member) {
         pairs.push_back({member, ref});
         replaced = true;
      } else {
         pairs.push_back(pr);
      }
   }
   if (!replaced) pairs.push_back({member, ref});
   mat.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(mat.getContext(), pairs));
}

static tuples::ColumnRefAttr materializedColumnForMember(subop::MaterializeOp mat, subop::Member member) {
   for (auto& pr : mat.getMapping().getMapping()) {
      if (pr.first == member) return pr.second;
   }
   return {};
}

static mlir::BlockArgument appendMapInputColumn(subop::MapOp map, tuples::ColumnRefAttr ref) {
   llvm::SmallVector<mlir::Attribute> inputs(map.getInputCols().begin(), map.getInputCols().end());
   inputs.push_back(ref);
   map.setInputColsAttr(mlir::ArrayAttr::get(map.getContext(), inputs));
   return map.getFn().front().addArgument(ref.getColumn().type, map.getLoc());
}

static subop::ScanRefsOp findTableScanRefsInBuildStep(subop::ExecutionStepOp step);

static bool mapHasInputSemantic(subop::MapOp map, llvm::StringRef semantic,
                                tuples::ColumnManager& cm) {
   for (auto attr : map.getInputCols()) {
      auto ref = mlir::cast<tuples::ColumnRefAttr>(attr);
      auto [scope, leaf] = cm.getName(&ref.getColumn());
      if (columnSemanticKey(scope, leaf) == semantic) return true;
   }
   return false;
}

static void ensureSyntheticMapHasPeerInputs(subop::ExecutionStepOp syntheticBuild,
                                            subop::MapOp syntheticMap,
                                            subop::MapOp peerMap) {
   auto* ctx = syntheticMap.getContext();
   auto& synthCm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& peerCm = peerMap.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   subop::ScanRefsOp scan = findTableScanRefsInBuildStep(syntheticBuild);
   assert(scan && "residual filter rewrite: synthetic build must scan a table");
   auto tableTy = mlir::cast<subop::TableType>(scan.getState().getType());

   llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> gatherPairs;
   llvm::SmallVector<tuples::ColumnRefAttr, 4> newInputs;
   for (auto attr : peerMap.getInputCols()) {
      auto peerRef = mlir::cast<tuples::ColumnRefAttr>(attr);
      auto [scope, leaf] = peerCm.getName(&peerRef.getColumn());
      std::string semantic = columnSemanticKey(scope, leaf);
      if (mapHasInputSemantic(syntheticMap, semantic, synthCm)) continue;
      subop::Member member = tableMemberForIdentifier(tableTy, mm, leaf);
      assert(member && "residual filter rewrite: synthetic table scan must contain peer predicate input");
      tuples::ColumnDefAttr def = synthCm.createDef(scope, leaf);
      def.getColumn().type = cloneTypeToContext(mm.getType(member), ctx);
      gatherPairs.push_back({member, def});
      newInputs.push_back(synthCm.createRef(&def.getColumn()));
   }
   if (gatherPairs.empty()) return;

   mlir::OpBuilder b(syntheticMap);
   auto gather = b.create<subop::GatherOp>(
      syntheticMap.getLoc(), syntheticMap.getStream(), synthCm.createRef(&scan.getRef().getColumn()),
      subop::ColumnDefMemberMappingAttr::get(ctx, gatherPairs));
   syntheticMap->setOperand(0, gather.getRes());
   for (tuples::ColumnRefAttr ref : newInputs)
      appendMapInputColumn(syntheticMap, ref);
}

static void andResidualPredicateWithSimpleFilter(subop::MapOp map, unsigned resultIdx,
                                                 tuples::ColumnRefAttr simplePredRef) {
   mlir::Block& block = map.getFn().front();
   auto ret = mlir::cast<tuples::ReturnOp>(block.getTerminator());
   mlir::BlockArgument simplePredArg = appendMapInputColumn(map, simplePredRef);
   mlir::OpBuilder b(ret);
   mlir::Value combined = b.create<db::AndOp>(ret.getLoc(), mlir::ValueRange{simplePredArg, ret.getOperand(resultIdx)});
   llvm::SmallVector<mlir::Value> retVals(ret->getOperands().begin(), ret->getOperands().end());
   retVals[resultIdx] = combined;
   auto newRet = b.create<tuples::ReturnOp>(ret.getLoc(), retVals);
   ret.erase();
   (void)newRet;
}

static void renameResidualPredicateMapResult(subop::MapOp map, subop::FilterOp filter,
                                             tuples::ColumnDefAttr predDef) {
   unsigned idx = residualFilterConditionResultIndex(map, filter);
   llvm::SmallVector<mlir::Attribute> computed(map.getComputedCols().begin(), map.getComputedCols().end());
   computed[idx] = predDef;
   map.setComputedColsAttr(mlir::ArrayAttr::get(map.getContext(), computed));
}

static mlir::BlockArgument mapBlockArgForInputSemanticLocal(subop::MapOp map, const std::string& semantic,
                                                            tuples::ColumnManager& cm) {
   mlir::Block& block = map.getFn().front();
   for (unsigned i = 0; i < map.getInputCols().size(); ++i) {
      auto ref = mlir::cast<tuples::ColumnRefAttr>(map.getInputCols()[i]);
      auto [scope, leaf] = cm.getName(&ref.getColumn());
      if (columnSemanticKey(scope, leaf) == semantic) return block.getArgument(i);
   }
   llvm_unreachable("residual filter: peer predicate input must exist in synthetic predicate map");
}

static mlir::Value cloneResidualPredicateExprToSynthetic(mlir::Value v, mlir::IRMapping& mapping,
                                                        mlir::OpBuilder& b, mlir::MLIRContext* ctx);

static mlir::Attribute cloneResidualAttrToContext(mlir::Attribute attr, mlir::MLIRContext* ctx) {
   if (!attr || attr.getContext() == ctx) return attr;
   mlir::Builder b(ctx);
   if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(attr))
      return b.getIntegerAttr(cloneTypeToContext(i.getType(), ctx), i.getValue());
   if (auto f = mlir::dyn_cast<mlir::FloatAttr>(attr))
      return b.getFloatAttr(cloneTypeToContext(f.getType(), ctx), f.getValue());
   if (auto s = mlir::dyn_cast<mlir::StringAttr>(attr)) return b.getStringAttr(s.getValue());
   llvm_unreachable("residual filter: unsupported cloned attribute");
}

static mlir::Value cloneResidualPredicateExprToSynthetic(mlir::Value v, mlir::IRMapping& mapping,
                                                        mlir::OpBuilder& b, mlir::MLIRContext* ctx) {
   if (mapping.contains(v)) return mapping.lookup(v);
   mlir::Operation* op = v.getDefiningOp();
   assert(op && "residual filter: unmapped peer predicate block argument");
   mlir::Location loc = mlir::UnknownLoc::get(ctx);
   auto finish = [&](mlir::Value cloned) {
      assert(cloned);
      assert(cloned.getType() == cloneTypeToContext(v.getType(), ctx) &&
             "residual filter: cloned expression type mismatch");
      if (mlir::Operation* def = cloned.getDefiningOp()) def->setLoc(loc);
      mapping.map(v, cloned);
      return cloned;
   };
   mlir::Value out;
   if (auto c = mlir::dyn_cast<db::ConstantOp>(op)) {
      out = b.create<db::ConstantOp>(loc, cloneTypeToContext(c.getType(), ctx),
                                     cloneResidualAttrToContext(c.getValue(), ctx));
   } else if (auto c = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
      out = b.create<mlir::arith::ConstantOp>(
         loc, mlir::cast<mlir::TypedAttr>(cloneResidualAttrToContext(c.getValue(), ctx)));
   } else if (auto cast = mlir::dyn_cast<db::CastOp>(op)) {
      out = b.create<db::CastOp>(loc, cloneTypeToContext(cast.getType(), ctx),
                                 cloneResidualPredicateExprToSynthetic(cast.getVal(), mapping, b, ctx));
   } else if (auto cmpi = mlir::dyn_cast<mlir::arith::CmpIOp>(op)) {
      out = b.create<mlir::arith::CmpIOp>(
         loc, cmpi.getPredicate(),
         cloneResidualPredicateExprToSynthetic(cmpi.getLhs(), mapping, b, ctx),
         cloneResidualPredicateExprToSynthetic(cmpi.getRhs(), mapping, b, ctx));
   } else if (auto ori = mlir::dyn_cast<mlir::arith::OrIOp>(op)) {
      out = b.create<mlir::arith::OrIOp>(
         loc,
         cloneResidualPredicateExprToSynthetic(ori.getLhs(), mapping, b, ctx),
         cloneResidualPredicateExprToSynthetic(ori.getRhs(), mapping, b, ctx));
   } else if (auto cmp = mlir::dyn_cast<db::CmpOp>(op)) {
      out = b.create<db::CmpOp>(
         loc, cmp.getPredicate(),
         cloneResidualPredicateExprToSynthetic(cmp.getLeft(), mapping, b, ctx),
         cloneResidualPredicateExprToSynthetic(cmp.getRight(), mapping, b, ctx));
   } else if (auto between = mlir::dyn_cast<db::BetweenOp>(op)) {
      out = b.create<db::BetweenOp>(
         loc,
         cloneResidualPredicateExprToSynthetic(between.getVal(), mapping, b, ctx),
         cloneResidualPredicateExprToSynthetic(between.getLower(), mapping, b, ctx),
         cloneResidualPredicateExprToSynthetic(between.getUpper(), mapping, b, ctx),
         between.getLowerInclusive(), between.getUpperInclusive());
   } else if (auto oneOf = mlir::dyn_cast<db::OneOfOp>(op)) {
      llvm::SmallVector<mlir::Value, 8> vals;
      for (mlir::Value arg : oneOf.getVals())
         vals.push_back(cloneResidualPredicateExprToSynthetic(arg, mapping, b, ctx));
      out = b.create<db::OneOfOp>(
         loc,
         cloneResidualPredicateExprToSynthetic(oneOf.getVal(), mapping, b, ctx), vals);
   } else if (auto rt = mlir::dyn_cast<db::RuntimeCall>(op)) {
      llvm::SmallVector<mlir::Value, 4> args;
      for (mlir::Value arg : rt.getArgs())
         args.push_back(cloneResidualPredicateExprToSynthetic(arg, mapping, b, ctx));
      out = b.create<db::RuntimeCall>(loc, cloneTypeToContext(rt.getRes().getType(), ctx), rt.getFn(), args)
               .getRes();
   } else if (auto andOp = mlir::dyn_cast<db::AndOp>(op)) {
      llvm::SmallVector<mlir::Value, 4> args;
      for (mlir::Value arg : andOp->getOperands())
         args.push_back(cloneResidualPredicateExprToSynthetic(arg, mapping, b, ctx));
      out = b.create<db::AndOp>(loc, args);
   } else if (auto orOp = mlir::dyn_cast<db::OrOp>(op)) {
      llvm::SmallVector<mlir::Value, 4> args;
      for (mlir::Value arg : orOp->getOperands())
         args.push_back(cloneResidualPredicateExprToSynthetic(arg, mapping, b, ctx));
      out = b.create<db::OrOp>(loc, args);
   } else if (auto notOp = mlir::dyn_cast<db::NotOp>(op)) {
      out = b.create<db::NotOp>(loc,
                                cloneResidualPredicateExprToSynthetic(notOp.getVal(), mapping, b, ctx));
   } else if (auto derive = mlir::dyn_cast<db::DeriveTruth>(op)) {
      out = b.create<db::DeriveTruth>(loc,
                                      cloneResidualPredicateExprToSynthetic(derive.getVal(), mapping, b, ctx));
   } else {
      llvm_unreachable("residual filter: unsupported predicate expression op");
   }
   return finish(out);
}

static tuples::ColumnRefAttr appendPeerResidualPredicateToSyntheticMap(subop::MapOp syntheticMap,
                                                                       subop::MapOp peerMap,
                                                                       subop::FilterOp peerFilter,
                                                                       tuples::ColumnDefAttr predDef) {
   auto* ctx = syntheticMap.getContext();
   auto& synthCm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& peerCm = peerMap.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

   mlir::IRMapping mapping;
   for (unsigned i = 0; i < peerMap.getInputCols().size(); ++i) {
      auto peerInput = mlir::cast<tuples::ColumnRefAttr>(peerMap.getInputCols()[i]);
      auto [scope, leaf] = peerCm.getName(&peerInput.getColumn());
      mapping.map(peerMap.getFn().front().getArgument(i),
                  mapBlockArgForInputSemanticLocal(syntheticMap, columnSemanticKey(scope, leaf), synthCm));
   }

   mlir::Block& peerBlock = peerMap.getFn().front();
   auto peerRet = mlir::cast<tuples::ReturnOp>(peerBlock.getTerminator());
   unsigned peerIdx = residualFilterConditionResultIndex(peerMap, peerFilter);

   mlir::Block& synthBlock = syntheticMap.getFn().front();
   auto synthRet = mlir::cast<tuples::ReturnOp>(synthBlock.getTerminator());
   mlir::OpBuilder b(synthRet);
   mlir::Value cloned = cloneResidualPredicateExprToSynthetic(peerRet.getOperand(peerIdx), mapping, b, ctx);

   llvm::SmallVector<mlir::Value> retVals(synthRet->getOperands().begin(), synthRet->getOperands().end());
   retVals.push_back(cloned);
   b.setInsertionPoint(synthRet);
   auto newRet = b.create<tuples::ReturnOp>(synthRet.getLoc(), retVals);
   synthRet.erase();
   (void)newRet;

   llvm::SmallVector<mlir::Attribute> computed(syntheticMap.getComputedCols().begin(),
                                               syntheticMap.getComputedCols().end());
   computed.push_back(predDef);
   syntheticMap.setComputedColsAttr(mlir::ArrayAttr::get(ctx, computed));
   return synthCm.createRef(&predDef.getColumn());
}

static tuples::ColumnRefAttr appendTrueResidualPredicateToSyntheticMap(subop::MapOp syntheticMap,
                                                                       tuples::ColumnDefAttr predDef) {
   auto* ctx = syntheticMap.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   mlir::Block& block = syntheticMap.getFn().front();
   auto ret = mlir::cast<tuples::ReturnOp>(block.getTerminator());
   mlir::OpBuilder b(ret);
   mlir::Value trueVal = b.create<db::ConstantOp>(ret.getLoc(), mlir::IntegerType::get(ctx, 1),
                                                  b.getI64IntegerAttr(1));
   llvm::SmallVector<mlir::Value> retVals(ret->getOperands().begin(), ret->getOperands().end());
   retVals.push_back(trueVal);
   auto newRet = b.create<tuples::ReturnOp>(ret.getLoc(), retVals);
   ret.erase();
   (void)newRet;

   llvm::SmallVector<mlir::Attribute> computed(syntheticMap.getComputedCols().begin(),
                                               syntheticMap.getComputedCols().end());
   computed.push_back(predDef);
   syntheticMap.setComputedColsAttr(mlir::ArrayAttr::get(ctx, computed));
   return cm.createRef(&predDef.getColumn());
}

static bool mapComputesColumn(subop::MapOp map, tuples::ColumnRefAttr ref) {
   for (auto attr : map.getComputedCols()) {
      auto def = mlir::cast<tuples::ColumnDefAttr>(attr);
      if (&def.getColumn() == &ref.getColumn()) return true;
   }
   return false;
}

static subop::MapOp findMapComputingColumn(subop::ExecutionStepOp step, tuples::ColumnRefAttr ref) {
   subop::MapOp found;
   step.walk([&](subop::MapOp map) {
      if (!found && mapComputesColumn(map, ref)) found = map;
   });
   assert(found && "residual filter rewrite: simple predicate producer map must exist");
   return found;
}

static mlir::Value streamInputOf(mlir::Operation* op) {
   if (auto gather = mlir::dyn_cast<subop::GatherOp>(op)) return gather.getStream();
   if (auto map = mlir::dyn_cast<subop::MapOp>(op)) return map.getStream();
   if (auto rename = mlir::dyn_cast<subop::RenamingOp>(op)) return rename.getStream();
   llvm_unreachable("residual filter rewrite: unexpected stream op in predicate producer chain");
}

static void setStreamInputOf(mlir::Operation* op, mlir::Value stream) {
   if (mlir::isa<subop::GatherOp, subop::MapOp, subop::RenamingOp>(op)) {
      op->setOperand(0, stream);
      return;
   }
   llvm_unreachable("residual filter rewrite: unexpected stream op in predicate producer chain");
}

static bool streamChainContains(mlir::Value stream, mlir::Value target) {
   for (;;) {
      if (stream == target) return true;
      mlir::Operation* def = stream.getDefiningOp();
      if (!def) return false;
      if (!mlir::isa<subop::GatherOp, subop::MapOp, subop::RenamingOp>(def)) return false;
      stream = streamInputOf(def);
   }
}

static void moveSimplePredicateProducerBeforeResidualMap(subop::ExecutionStepOp step,
                                                        subop::MapOp residualMap,
                                                        tuples::ColumnRefAttr predRef) {
   subop::MapOp predMap = findMapComputingColumn(step, predRef);
   if (predMap == residualMap) return;
   if (predMap->isBeforeInBlock(residualMap.getOperation())) {
      if (streamChainContains(residualMap.getStream(), predMap.getResult())) return;
      if (residualMap.getStream() != predMap.getResult()) residualMap->setOperand(0, predMap.getResult());
      return;
   }

   llvm::SmallVector<mlir::Operation*, 4> chain;
   mlir::Value stream = predMap.getStream();
   while (stream != residualMap.getResult()) {
      mlir::Operation* def = stream.getDefiningOp();
      assert(def && "residual filter rewrite: predicate producer must be downstream of residual map");
      chain.push_back(def);
      stream = streamInputOf(def);
   }

   mlir::Value newStream = residualMap.getStream();
   for (mlir::Operation* op : llvm::reverse(chain)) {
      setStreamInputOf(op, newStream);
      op->moveBefore(residualMap.getOperation());
      newStream = op->getResult(0);
   }
   predMap->setOperand(0, newStream);
   predMap->moveBefore(residualMap.getOperation());
   residualMap->setOperand(0, predMap.getResult());
}

static void replaceRefIfSameColumn(tuples::ColumnRefAttr& ref,
                                   tuples::ColumnRefAttr oldRef,
                                   tuples::ColumnRefAttr newRef) {
   if (ref && &ref.getColumn() == &oldRef.getColumn()) ref = newRef;
}

static mlir::Value splitResidualPredicateMapResults(subop::MapOp map,
                                                    tuples::ColumnRefAttr& pred0Ref,
                                                    tuples::ColumnRefAttr& pred1Ref) {
   if (map.getComputedCols().size() <= 1) return map.getResult();
   auto* ctx = map.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   llvm::SmallVector<mlir::Attribute> originalComputed(map.getComputedCols().begin(), map.getComputedCols().end());

   mlir::Value stream = map.getResult();
   mlir::Operation* insertAfter = map.getOperation();
   llvm::SmallVector<mlir::Operation*, 4> newMaps;
   for (size_t i = 1; i < originalComputed.size(); ++i) {
      auto originalDef = mlir::cast<tuples::ColumnDefAttr>(originalComputed[i]);
      tuples::ColumnRefAttr originalRef = cm.createRef(&originalDef.getColumn());
      auto [scope, leaf] = cm.getName(&originalDef.getColumn());
      tuples::ColumnDefAttr splitDef = cm.createDef(cm.getUniqueScope(scope + "$split"), leaf);
      splitDef.getColumn().type = originalDef.getColumn().type;
      tuples::ColumnRefAttr splitRef = cm.createRef(&splitDef.getColumn());

      mlir::OpBuilder b(insertAfter);
      b.setInsertionPointAfter(insertAfter);
      auto nextMap = b.create<subop::MapOp>(map.getLoc(), tuples::TupleStreamType::get(ctx), stream,
                                            b.getArrayAttr({splitDef}),
                                            b.getArrayAttr({originalRef}));
      mlir::Block* block = new mlir::Block();
      block->addArgument(originalRef.getColumn().type, map.getLoc());
      nextMap.getFn().push_back(block);
      mlir::OpBuilder rb(ctx);
      rb.setInsertionPointToStart(block);
      rb.create<tuples::ReturnOp>(map.getLoc(), mlir::ValueRange{block->getArgument(0)});

      replaceRefIfSameColumn(pred0Ref, originalRef, splitRef);
      replaceRefIfSameColumn(pred1Ref, originalRef, splitRef);

      stream = nextMap.getResult();
      insertAfter = nextMap.getOperation();
      newMaps.push_back(nextMap.getOperation());
   }

   map.getResult().replaceUsesWithIf(stream, [&](mlir::OpOperand& use) {
      mlir::Operation* owner = use.getOwner();
      for (mlir::Operation* newMap : newMaps)
         if (owner == newMap) return false;
      return true;
   });
   return stream;
}

static mlir::Value insertResidualFilterUnionAfterPredicates(mlir::Value stream,
                                                            tuples::ColumnRefAttr pred0Ref,
                                                            tuples::ColumnRefAttr pred1Ref) {
   auto* ctx = pred0Ref.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   tuples::ColumnDefAttr unionPred = cm.createDef(cm.getUniqueScope("residual_filter_union"), "pred");
   unionPred.getColumn().type = mlir::IntegerType::get(ctx, 1);
   tuples::ColumnRefAttr unionRef = cm.createRef(&unionPred.getColumn());

   mlir::Operation* anchor = stream.getDefiningOp();
   assert(anchor && "residual filter union must be inserted after a stream producer");
   mlir::OpBuilder b(anchor);
   b.setInsertionPointAfter(anchor);
   auto map = b.create<subop::MapOp>(anchor->getLoc(), tuples::TupleStreamType::get(ctx), stream,
                                     b.getArrayAttr({unionPred}),
                                     b.getArrayAttr({pred0Ref, pred1Ref}));
   mlir::Block* block = new mlir::Block();
   block->addArgument(pred0Ref.getColumn().type, anchor->getLoc());
   block->addArgument(pred1Ref.getColumn().type, anchor->getLoc());
   map.getFn().push_back(block);
   mlir::OpBuilder rb(ctx);
   rb.setInsertionPointToStart(block);
   mlir::Value unionValue = rb.create<db::OrOp>(anchor->getLoc(),
                                                mlir::ValueRange{block->getArgument(0), block->getArgument(1)});
   rb.create<tuples::ReturnOp>(anchor->getLoc(), mlir::ValueRange{unionValue});

   b.setInsertionPointAfter(map);
   auto filter = b.create<subop::FilterOp>(anchor->getLoc(), map.getResult(),
                                           subop::FilterSemantic::all_true,
                                           b.getArrayAttr({unionRef}));
   stream.replaceUsesWithIf(filter.getRes(), [&](mlir::OpOperand& use) {
      mlir::Operation* owner = use.getOwner();
      if (owner->getBlock() != filter->getBlock()) return false;
      if (!filter->isBeforeInBlock(owner)) return false;
      return owner != map.getOperation() && owner != filter.getOperation();
   });
   return filter.getRes();
}

static void rewireStreamUsesAfterAnchorInStep(mlir::Value oldStream, mlir::Value newStream,
                                              mlir::Operation* anchorOp,
                                              llvm::ArrayRef<mlir::Operation*> excludeOps) {
   oldStream.replaceUsesWithIf(newStream, [&](mlir::OpOperand& use) {
      mlir::Operation* owner = use.getOwner();
      if (owner->getBlock() != anchorOp->getBlock()) return false;
      if (!anchorOp->isBeforeInBlock(owner)) return false;
      for (mlir::Operation* ex : excludeOps)
         if (owner == ex) return false;
      return true;
   });
}

static subop::ScanRefsOp findTableScanRefsInBuildStep(subop::ExecutionStepOp step) {
   subop::ScanRefsOp found;
   step.walk([&](subop::ScanRefsOp scan) {
      if (found) return;
      if (mlir::isa<subop::TableType>(scan.getState().getType())) found = scan;
   });
   return found;
}

static subop::MapOp createResidualPredicateMapAfterTableScan(subop::ExecutionStepOp syntheticBuild,
                                                             subop::MapOp peerMap) {
   auto* ctx = syntheticBuild.getContext();
   auto& synthCm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& peerCm = peerMap.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   subop::ScanRefsOp scan = findTableScanRefsInBuildStep(syntheticBuild);
   assert(scan && "residual filter rewrite: synthetic build must scan a table");
   auto tableTy = mlir::cast<subop::TableType>(scan.getState().getType());

   llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> gatherPairs;
   llvm::SmallVector<mlir::Attribute> mapInputs;
   for (auto attr : peerMap.getInputCols()) {
      auto peerRef = mlir::cast<tuples::ColumnRefAttr>(attr);
      auto [scope, leaf] = peerCm.getName(&peerRef.getColumn());
      subop::Member member = tableMemberForIdentifier(tableTy, mm, leaf);
      assert(member && "residual filter rewrite: synthetic table scan must contain peer predicate input");
      tuples::ColumnDefAttr def = synthCm.createDef(scope, leaf);
      def.getColumn().type = cloneTypeToContext(mm.getType(member), ctx);
      gatherPairs.push_back({member, def});
      mapInputs.push_back(synthCm.createRef(&def.getColumn()));
   }

   mlir::OpBuilder b(scan);
   b.setInsertionPointAfter(scan);
   mlir::Value stream = scan.getRes();
   llvm::SmallVector<mlir::Operation*> excludeOps{scan.getOperation()};
   if (!gatherPairs.empty()) {
      auto scanRef = synthCm.createRef(&scan.getRef().getColumn());
      auto gather = b.create<subop::GatherOp>(scan.getLoc(), stream, scanRef,
                                              subop::ColumnDefMemberMappingAttr::get(ctx, gatherPairs));
      stream = gather.getRes();
      b.setInsertionPointAfter(gather);
      excludeOps.push_back(gather.getOperation());
   }

   auto map = b.create<subop::MapOp>(scan.getLoc(), tuples::TupleStreamType::get(ctx), stream,
                                     b.getArrayAttr({}), b.getArrayAttr(mapInputs));
   mlir::Block* block = new mlir::Block();
   for (auto input : mapInputs) {
      auto ref = mlir::cast<tuples::ColumnRefAttr>(input);
      block->addArgument(ref.getColumn().type, scan.getLoc());
   }
   map.getFn().push_back(block);
   mlir::OpBuilder rb(ctx);
   rb.setInsertionPointToStart(block);
   rb.create<tuples::ReturnOp>(scan.getLoc(), mlir::ValueRange{});
   excludeOps.push_back(map.getOperation());
   rewireStreamUsesAfterAnchorInStep(scan.getRes(), map.getResult(), scan.getOperation(), excludeOps);
   return map;
}

static bool rewriteSyntheticResidualFiltersAsFilterPreds(subop::ExecutionStepOp syntheticBuild,
                                                         subop::ExecutionStepOp peerBuild0,
                                                         subop::ExecutionStepOp peerBuild1,
                                                         subop::MaterializeOp mat,
                                                         unsigned qIdx0,
                                                         unsigned qIdx1,
                                                         subop::Member predMember0,
                                                         subop::Member predMember1,
                                                         llvm::ArrayRef<runtime::FilterDescription> simpleFilters0,
                                                         llvm::ArrayRef<runtime::FilterDescription> simpleFilters1) {
   auto synthResidual = findResidualTableFilterInBuildStep(syntheticBuild);
   auto peerResidual0 = findResidualTableFilterInBuildStep(peerBuild0);
   auto peerResidual1 = findResidualTableFilterInBuildStep(peerBuild1);
   if (!peerResidual0 && !peerResidual1) return false;
   if (peerResidual0 && peerResidual1 &&
       residualFilterFingerprint(*peerResidual0) == residualFilterFingerprint(*peerResidual1))
      return false;

   auto* ctx = syntheticBuild.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   tuples::ColumnDefAttr pred0 = makeResidualFilterPredDef(ctx, qIdx0);
   tuples::ColumnDefAttr pred1 = makeResidualFilterPredDef(ctx, qIdx1);

   tuples::ColumnRefAttr simplePred0Ref;
   tuples::ColumnRefAttr simplePred1Ref;
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   if (!simpleFilters0.empty()) {
      insertWriteSidePredIntoBufferConstructionStepForPredMember(syntheticBuild, simpleFilters0,
                                                                 mm.getName(predMember0));
      simplePred0Ref = materializedColumnForMember(mat, predMember0);
      assert(simplePred0Ref && "residual filter rewrite: simple predicate materialize missing for q0");
   }
   if (!simpleFilters1.empty()) {
      insertWriteSidePredIntoBufferConstructionStepForPredMember(syntheticBuild, simpleFilters1,
                                                                 mm.getName(predMember1));
      simplePred1Ref = materializedColumnForMember(mat, predMember1);
      assert(simplePred1Ref && "residual filter rewrite: simple predicate materialize missing for q1");
   }
   subop::MapOp syntheticMap = synthResidual ? synthResidual->predMap
                                             : createResidualPredicateMapAfterTableScan(
                                                  syntheticBuild, peerResidual0 ? peerResidual0->predMap
                                                                                : peerResidual1->predMap);
   if (simplePred0Ref) moveSimplePredicateProducerBeforeResidualMap(syntheticBuild, syntheticMap, simplePred0Ref);
   if (simplePred1Ref) moveSimplePredicateProducerBeforeResidualMap(syntheticBuild, syntheticMap, simplePred1Ref);
   if (peerResidual0) ensureSyntheticMapHasPeerInputs(syntheticBuild, syntheticMap, peerResidual0->predMap);
   if (peerResidual1) ensureSyntheticMapHasPeerInputs(syntheticBuild, syntheticMap, peerResidual1->predMap);

   tuples::ColumnRefAttr pred0Ref;
   if (peerResidual0) {
      if (synthResidual && synthResidual->predMap == syntheticMap) {
         unsigned pred0Idx = residualFilterConditionResultIndex(synthResidual->predMap, synthResidual->filter);
         renameResidualPredicateMapResult(synthResidual->predMap, synthResidual->filter, pred0);
         pred0Ref = cm.createRef(&pred0.getColumn());
         if (simplePred0Ref) andResidualPredicateWithSimpleFilter(syntheticMap, pred0Idx, simplePred0Ref);
      } else {
         pred0Ref = appendPeerResidualPredicateToSyntheticMap(syntheticMap, peerResidual0->predMap,
                                                              peerResidual0->filter, pred0);
         if (simplePred0Ref) {
            unsigned pred0Idx = syntheticMap.getComputedCols().size() - 1;
            andResidualPredicateWithSimpleFilter(syntheticMap, pred0Idx, simplePred0Ref);
         }
      }
   } else if (simplePred0Ref) {
      pred0Ref = simplePred0Ref;
   } else {
      pred0Ref = appendTrueResidualPredicateToSyntheticMap(syntheticMap, pred0);
   }
   tuples::ColumnRefAttr pred1Ref;
   if (peerResidual1) {
      if (synthResidual && peerResidual0 && residualFilterFingerprint(*peerResidual0) ==
                              residualFilterFingerprint(*peerResidual1)) {
         pred1Ref = appendTrueResidualPredicateToSyntheticMap(syntheticMap, pred1);
      } else {
         pred1Ref = appendPeerResidualPredicateToSyntheticMap(syntheticMap, peerResidual1->predMap,
                                                              peerResidual1->filter, pred1);
      }
      if (simplePred1Ref) {
         unsigned pred1Idx = syntheticMap.getComputedCols().size() - 1;
         andResidualPredicateWithSimpleFilter(syntheticMap, pred1Idx, simplePred1Ref);
      }
   } else if (simplePred1Ref) {
      pred1Ref = simplePred1Ref;
   } else {
      pred1Ref = appendTrueResidualPredicateToSyntheticMap(syntheticMap, pred1);
   }

   mlir::Value residualInputStream = syntheticMap.getStream();
   mlir::Value finalPredicateStream = splitResidualPredicateMapResults(syntheticMap, pred0Ref, pred1Ref);
   setMaterializeMapping(mat, predMember0, pred0Ref);
   setMaterializeMapping(mat, predMember1, pred1Ref);
   if (mlir::Operation* finalPredDef = finalPredicateStream.getDefiningOp()) {
      rewireStreamUsesAfterAnchorInStep(residualInputStream, finalPredicateStream, finalPredDef,
                                        llvm::ArrayRef<mlir::Operation*>{finalPredDef});
   }
   if (peerResidual0 && peerResidual1) {
      finalPredicateStream = insertResidualFilterUnionAfterPredicates(finalPredicateStream, pred0Ref, pred1Ref);
   }
   if (synthResidual) {
      synthResidual->filter.getRes().replaceAllUsesWith(finalPredicateStream);
      synthResidual->filter.erase();
   }

   return true;
}

static void collectSemanticHashToMemberFromMaterialize(
   subop::MaterializeOp mat, subop::Member linkM, subop::Member hashM, subop::MemberManager& mm,
   const llvm::DenseMap<const void*, uint64_t>& columnHashes, llvm::DenseMap<uint64_t, subop::Member>& out) {
   if (!mat) return;
   for (auto& [member, colRef] : mat.getMapping().getMapping()) {
      llvm::StringRef memName = mm.getName(member);
      if (member == linkM || member == hashM || isJoinBufferInternalMemberName(memName)) continue;
      if (mm.getType(member) != colRef.getColumn().type) continue;
      out.try_emplace(payloadColumnIdentityHash(colRef, columnHashes), member);
   }
}

/// Assign \p plan.payloadMembers from cloned build-step materialize mappings + fresh members for union-only columns.
/// Reuses existing buffer members (e.g. \c member$0, \c member$1) from materialize; new union columns get the next
/// \c member$N slot via \c createMemberDirect (not semantic leaf names like \c s_comment$0).
static void assignPayloadMembersForPlan(subop::MemberManager& mm, lingodb::compiler::dialect::tuples::ColumnManager& cm,
                                        subop::MaterializeOp matOp, JoinBufferUnionPlan& plan,
                                        const llvm::DenseMap<const void*, uint64_t>& columnHashes) {
   llvm::DenseMap<uint64_t, subop::Member> bySemanticHash;
   collectSemanticHashToMemberFromMaterialize(matOp, plan.linkMember, plan.hashMember, mm, columnHashes,
                                              bySemanticHash);
   plan.payloadMembers.clear();
   plan.payloadMembers.reserve(plan.payloadColumns.size());
   llvm::SmallVector<subop::Member, 8> existingPayloadMembers;
   for (auto& it : bySemanticHash) existingPayloadMembers.push_back(it.second);
   llvm::StringMap<subop::Member> noSemanticMembers;
   unsigned nextSlot = nextPayloadMemberSlot(mm, noSemanticMembers, existingPayloadMembers);
   if (subop::BufferType bufTy = getInnerBufferTypeForMaterializeState(matOp.getState().getType())) {
      for (subop::Member m : bufTy.getMembers().getMembers()) {
         if (m == plan.linkMember || m == plan.hashMember) continue;
         if (auto slot = parseMemberSlot(mm.getName(m))) nextSlot = std::max(nextSlot, *slot + 1);
         if (auto predSlot = parseFilterPredMemberSlot(mm.getName(m))) nextSlot = std::max(nextSlot, *predSlot + 1);
      }
   }
   mlir::MLIRContext* synthCtx = matOp.getContext();
   subop::Member canonicalPredMember = makeOrGetPredMemberForSlot(matOp.getContext(), 0);
   mlir::Type canonicalPredTy = mm.getType(canonicalPredMember);
   for (const PayloadColumnSpec& spec : plan.payloadColumns) {
      unsigned predIdx = 0;
      if (parseFilterPredLayoutSemanticKey(spec.semanticKey, predIdx)) {
         plan.payloadMembers.push_back(mm.getOrCreateMemberDirect(
            "filter_pred$" + llvm::Twine(predIdx).str(), canonicalPredTy, /*allowTypeUpdate=*/false));
         continue;
      }
      auto it = bySemanticHash.find(spec.semanticHash);
      if (it != bySemanticHash.end()) {
         mlir::Type wantTy = cloneTypeToContext(spec.colType, synthCtx);
         if (mm.getType(it->second) != wantTy)
            llvm_unreachable("join superset: reused member type must match union column");
         plan.payloadMembers.push_back(it->second);
         continue;
      }
      mlir::Type slotTy = cloneTypeToContext(spec.colType, synthCtx);
      plan.payloadMembers.push_back(allocUnusedPayloadMemberSlot(mm, slotTy, nextSlot));
   }
}

static subop::StateMembersAttr bufferMembersForPlan(mlir::MLIRContext* ctx, const JoinBufferUnionPlan& plan) {
   assert(plan.payloadMembers.size() == plan.payloadColumns.size() &&
          "join superset: payload member list must match union columns");
   llvm::SmallVector<subop::Member> members;
   members.push_back(plan.linkMember);
   members.push_back(plan.hashMember);
   members.append(plan.payloadMembers.begin(), plan.payloadMembers.end());
   return subop::StateMembersAttr::get(ctx, members);
}

static subop::Member findBufferMemberByName(subop::BufferType bufTy, subop::MemberManager& mm, llvm::StringRef name) {
   if (name.empty()) return {};
   for (subop::Member m : bufTy.getMembers().getMembers()) {
      if (mm.getName(m) == name) return m;
   }
   return {};
}

static void resolveHashLinkMembersForChiv(subop::CreateHashIndexedView chiv, subop::BufferType bufTy,
                                          subop::MemberManager& mm, subop::Member& hashM, subop::Member& linkM) {
   auto oldHiv = mlir::dyn_cast<subop::HashIndexedViewType>(chiv.getType());
   llvm::StringRef preferHashName = chiv.getHashMember() ? mm.getName(chiv.getHashMember().getMember()) : llvm::StringRef{};
   llvm::StringRef preferLinkName = chiv.getLinkMember() ? mm.getName(chiv.getLinkMember().getMember()) : llvm::StringRef{};
   hashM = findBufferMemberByName(bufTy, mm, preferHashName);
   linkM = findBufferMemberByName(bufTy, mm, preferLinkName);
   if (!hashM && oldHiv && oldHiv.getKeyMembers().getMembers().size() == 1) {
      hashM = findBufferMemberByName(bufTy, mm, mm.getName(oldHiv.getKeyMembers().getMembers()[0]));
   }
   if (!hashM) {
      llvm::SmallVector<subop::Member, 4> hashes;
      for (subop::Member m : bufTy.getMembers().getMembers()) {
         if (mm.getName(m).starts_with("hash$")) hashes.push_back(m);
      }
      if (hashes.size() == 1) hashM = hashes.front();
   }
   if (!linkM && hashM) {
      if (auto slot = parseMemberSlot(mm.getName(hashM))) {
         linkM = findBufferMemberByName(bufTy, mm, "link$" + llvm::Twine(*slot).str());
      }
   }
   if (!linkM) {
      llvm::SmallVector<subop::Member, 4> links;
      for (subop::Member m : bufTy.getMembers().getMembers()) {
         if (mm.getName(m).starts_with("link$")) links.push_back(m);
      }
      if (links.size() == 1) linkM = links.front();
   }
   // Lowering expects link$ then hash$ as the first two buffer members.
   if (!hashM || !linkM) {
      auto members = bufTy.getMembers().getMembers();
      if (members.size() >= 2 && mm.getName(members[0]).starts_with("link$") &&
          mm.getName(members[1]).starts_with("hash$")) {
         linkM = members[0];
         hashM = members[1];
      }
   }
}

static subop::HashIndexedViewType asHashIndexedViewLayoutType(mlir::Type type) {
   if (auto hiv = mlir::dyn_cast<subop::HashIndexedViewType>(type)) return hiv;
   if (auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(type)) {
      return subop::HashIndexedViewType::get(mixed.getContext(), mixed.getKeyMembers(), mixed.getValueMembers(),
                                             mixed.getCompareHashForLookup());
   }
   return nullptr;
}

static bool lookupEntryRefStateHasLayout(mlir::Type type, subop::HashIndexedViewType layoutHiv) {
   auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(type);
   return ler && asHashIndexedViewLayoutType(ler.getState()) == layoutHiv;
}

static bool sameMemberTypeSequence(subop::MemberManager& mmA, subop::StateMembersAttr a,
                                   subop::MemberManager& mmB, subop::StateMembersAttr b) {
   auto as = a.getMembers();
   auto bs = b.getMembers();
   if (as.size() != bs.size()) return false;
   for (size_t i = 0; i < as.size(); ++i) {
      std::string typeA;
      std::string typeB;
      llvm::raw_string_ostream osA(typeA);
      llvm::raw_string_ostream osB(typeB);
      mmA.getType(as[i]).print(osA);
      mmB.getType(bs[i]).print(osB);
      osA.flush();
      osB.flush();
      if (typeA != typeB) return false;
   }
   return true;
}

static bool hashIndexedViewLayoutsArePhysicallyCompatible(mlir::Value a, mlir::Value b) {
   auto hivA = asHashIndexedViewLayoutType(a.getType());
   auto hivB = asHashIndexedViewLayoutType(b.getType());
   if (!hivA || !hivB) return false;
   if (hivA.getCompareHashForLookup() != hivB.getCompareHashForLookup()) return false;
   auto* dialectA = a.getContext()->getLoadedDialect<subop::SubOperatorDialect>();
   auto* dialectB = b.getContext()->getLoadedDialect<subop::SubOperatorDialect>();
   assert(dialectA && dialectB);
   auto& mmA = dialectA->getMemberManager();
   auto& mmB = dialectB->getMemberManager();
   return sameMemberTypeSequence(mmA, hivA.getKeyMembers(), mmB, hivB.getKeyMembers()) &&
          sameMemberTypeSequence(mmA, hivA.getValueMembers(), mmB, hivB.getValueMembers());
}

static std::optional<std::string> firstFilterPredMemberName(subop::StateMembersAttr members,
                                                            subop::MemberManager& mm) {
   std::optional<std::string> best;
   std::optional<unsigned> bestSlot;
   for (subop::Member m : members.getMembers()) {
      llvm::StringRef name = mm.getName(m);
      auto slot = parseFilterPredMemberSlot(name);
      if (!slot) continue;
      if (!best || *slot < *bestSlot) {
         best = name.str();
         bestSlot = *slot;
      }
   }
   return best;
}

static void syncCreateHashIndexedViewFromBuffer(subop::CreateHashIndexedView chiv) {
   auto* ctx = chiv.getContext();
   mlir::Value src = chiv.getSource();
   auto bufTy = mlir::dyn_cast<subop::BufferType>(src.getType());
   if (!bufTy) return;
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   subop::Member linkM;
   subop::Member hashM;
   resolveHashLinkMembersForChiv(chiv, bufTy, mm, hashM, linkM);
   if (!hashM || !linkM) return;
   chiv.setLinkMemberAttr(subop::MemberAttr::get(ctx, linkM));
   chiv.setHashMemberAttr(subop::MemberAttr::get(ctx, hashM));
   llvm::SmallVector<subop::Member> vals;
   for (subop::Member m : bufTy.getMembers().getMembers()) {
      llvm::StringRef n = mm.getName(m);
      if (n.starts_with("link$") || n.starts_with("hash$")) continue;
      vals.push_back(m);
   }
   auto oldHiv = asHashIndexedViewLayoutType(chiv.getType());
   assert(oldHiv && "create_hash_indexed_view must produce HIV-like type");
   auto keyMs = subop::StateMembersAttr::get(ctx, llvm::SmallVector<subop::Member>{hashM});
   auto valMs = subop::StateMembersAttr::get(ctx, vals);
   if (std::optional<std::string> predName = firstFilterPredMemberName(valMs, mm)) {
      auto newMixed = subop::MixedHashIndexedViewType::get(ctx, keyMs, valMs, oldHiv.getCompareHashForLookup(),
                                                           mlir::StringAttr::get(ctx, *predName));
      chiv.getResult().setType(newMixed);
      return;
   }
   chiv.getResult().setType(subop::HashIndexedViewType::get(ctx, keyMs, valMs, oldHiv.getCompareHashForLookup()));
}

static void syncMaterializeMappingsToBufferMembers(mlir::ModuleOp module, subop::StateMembersAttr targetMembers,
                                                   const llvm::DenseSet<void*>& closure,
                                                   const llvm::DenseMap<const void*, uint64_t>* columnHashes);

static void applyBufferLayoutToSsaClosure(mlir::ModuleOp module, llvm::ArrayRef<mlir::Value> canonicalBuffers,
                                          subop::StateMembersAttr targetMembers, const ModuleReuseInfo& reuse,
                                          const llvm::DenseMap<const void*, uint64_t>* columnHashes = nullptr) {
   JoinBufferHivSsaClosure joinClosure = computeJoinBufferHivSsaClosure(canonicalBuffers, reuse);
   llvm::DenseSet<void*>& closure = joinClosure.opaque;
   expandClosureThroughExecutionStepPorts(module, closure);

   auto* ctx = module.getContext();
   auto targetBufTy = subop::BufferType::get(ctx, targetMembers);
   auto targetTlTy = subop::ThreadLocalType::get(ctx, mlir::cast<subop::State>(targetBufTy));

   auto widenCarrierValue = [&](mlir::Value v) {
      if (!opaqueClosureContains(closure, v)) return;
      if (mlir::isa<subop::BufferType>(v.getType())) {
         v.setType(targetBufTy);
      } else if (auto tl = mlir::dyn_cast<subop::ThreadLocalType>(v.getType())) {
         if (mlir::isa<subop::BufferType>(tl.getWrapped())) v.setType(targetTlTy);
      }
   };

   module.walk([&](mlir::Operation* op) {
      for (mlir::Value v : op->getOperands()) widenCarrierValue(v);
      for (mlir::Value v : op->getResults()) widenCarrierValue(v);
   });
   module.walk([&](mlir::Operation* op) {
      for (mlir::Region& reg : op->getRegions()) {
         for (mlir::Block& block : reg) {
            for (mlir::BlockArgument a : block.getArguments()) widenCarrierValue(a);
         }
      }
   });

   module.walk([&](subop::MergeOp merge) {
      if (!opaqueClosureContains(closure, merge.getRes()) && !opaqueClosureContains(closure, merge.getThreadLocal()))
         return;
      merge.getRes().setType(targetBufTy);
      merge.getThreadLocal().setType(targetTlTy);
   });
   module.walk([&](subop::GenericCreateOp create) {
      if (!opaqueClosureContains(closure, create.getRes())) return;
      if (mlir::isa<subop::ThreadLocalType>(create.getType()) &&
          mlir::isa<subop::BufferType>(mlir::cast<subop::ThreadLocalType>(create.getType()).getWrapped())) {
         create.getResult().setType(targetTlTy);
      }
   });
   module.walk([&](subop::MaterializeOp mat) {
      if (!opaqueClosureContains(closure, mat.getState())) return;
      if (mlir::isa<subop::BufferType>(mat.getState().getType())) {
         mat.getState().setType(targetBufTy);
      } else if (mlir::isa<subop::ThreadLocalType>(mat.getState().getType()) &&
                 mlir::isa<subop::BufferType>(mlir::cast<subop::ThreadLocalType>(mat.getState().getType()).getWrapped())) {
         mat.getState().setType(targetTlTy);
      }
   });
   module.walk([&](subop::CreateHashIndexedView chiv) {
      if (!opaqueClosureContains(closure, chiv.getSource())) return;
      syncCreateHashIndexedViewFromBuffer(chiv);
   });
   syncMaterializeMappingsToBufferMembers(module, targetMembers, closure, columnHashes);
   synchronizeExecutionStepPortTypes(module, &closure);
}

/// After buffer layout is applied, canonicalize \c materialize member slots to \p targetMembers (by name / payload hash).
static void syncMaterializeMappingsToBufferMembers(mlir::ModuleOp module, subop::StateMembersAttr targetMembers,
                                                   const llvm::DenseSet<void*>& closure,
                                                   const llvm::DenseMap<const void*, uint64_t>* columnHashes) {
   auto* ctx = module.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   llvm::StringMap<subop::Member> validByName;
   for (subop::Member m : targetMembers.getMembers()) validByName[mm.getName(m)] = m;

   module.walk([&](subop::MaterializeOp mat) {
      if (!opaqueClosureContains(closure, mat.getState())) return;
      subop::BufferType bufTy = getInnerBufferTypeForMaterializeState(mat.getState().getType());
      if (!bufTy || bufTy.getMembers() != targetMembers) return;

      llvm::DenseMap<uint64_t, subop::Member> byPayloadHash;
      if (columnHashes) {
         for (auto& pr : mat.getMapping().getMapping()) {
            auto itName = validByName.find(mm.getName(pr.first));
            if (itName == validByName.end()) continue;
            auto itHash = columnHashes->find(&pr.second.getColumn());
            if (itHash == columnHashes->end()) continue;
            byPayloadHash[itHash->second] = itName->second;
         }
      }

      llvm::SmallVector<subop::RefMappingPairT> pairs;
      for (auto& pr : mat.getMapping().getMapping()) {
         llvm::StringRef memName = mm.getName(pr.first);
         subop::Member canon;
         if (auto it = validByName.find(memName); it != validByName.end()) canon = it->second;
         if (!canon && (memName.starts_with("link$") || memName.starts_with("hash$"))) continue;
         if (!canon) {
            if (auto slot = parseFilterPredMemberSlot(memName)) {
               std::string predName = "filter_pred$" + std::to_string(*slot);
               if (auto it = validByName.find(predName); it != validByName.end()) canon = it->second;
            }
         }
         if (!canon && columnHashes) {
            auto itHash = columnHashes->find(&pr.second.getColumn());
            if (itHash != columnHashes->end()) {
               if (auto it = byPayloadHash.find(itHash->second); it != byPayloadHash.end())
                  canon = it->second;
            }
         }
         if (!canon) continue;
         pairs.push_back({canon, pr.second});
      }
      mat.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, pairs));
   });
}

static void alignBufferMergeThreadLocalsWithMergeResult(mlir::ModuleOp module,
                                                        const llvm::DenseSet<void*>* closureFilter) {
   auto* ctx = module.getContext();
   module.walk([&](subop::MergeOp merge) {
      if (closureFilter && !opaqueClosureContains(*closureFilter, merge.getRes()) &&
          !opaqueClosureContains(*closureFilter, merge.getThreadLocal())) {
         return;
      }
      auto resBuf = mlir::dyn_cast<subop::BufferType>(merge.getRes().getType());
      if (!resBuf) return;
      auto targetTl = subop::ThreadLocalType::get(ctx, mlir::cast<subop::State>(resBuf));
      llvm::SmallDenseSet<void*> seen;
      llvm::SmallVector<mlir::Value, 16> worklist;
      worklist.push_back(merge.getThreadLocal());
      while (!worklist.empty()) {
         mlir::Value v = worklist.back();
         worklist.pop_back();
         void* k = v.getAsOpaquePointer();
         if (!seen.insert(k).second) continue;
         v.setType(targetTl);
         if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(v)) {
            mlir::Block* owner = ba.getOwner();
            mlir::Operation* parentOp = owner->getParentOp();
            if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(parentOp)) {
               unsigned idx = ba.getArgNumber();
               if (idx < step.getNumOperands()) worklist.push_back(step.getOperand(idx));
            }
         }
      }
   });
}

static ExternalDatasourceProperty mergeExternalDatasource(const ExternalDatasourceProperty& a,
                                                        const ExternalDatasourceProperty& b) {
   assert(a.tableName == b.tableName && "join superset: table name must match");
   ExternalDatasourceProperty out = a;
   std::unordered_set<std::string> have;
   for (const auto& m : out.mapping) have.insert(m.identifier);
   for (const auto& m : b.mapping) {
      if (have.insert(m.identifier).second) out.mapping.push_back(m);
   }
   llvm::sort(out.mapping, [](const auto& x, const auto& y) { return x.memberName < y.memberName; });
   return out;
}

static bool filterDescrLess(const lingodb::runtime::FilterDescription& a,
                            const lingodb::runtime::FilterDescription& b) {
   if (a.columnName != b.columnName) return a.columnName < b.columnName;
   if (a.columnId != b.columnId) return a.columnId < b.columnId;
   return static_cast<uint8_t>(a.op) < static_cast<uint8_t>(b.op);
}

static bool filterClauseEquals(llvm::ArrayRef<lingodb::runtime::FilterDescription> a,
                               llvm::ArrayRef<lingodb::runtime::FilterDescription> b) {
   if (a.size() != b.size()) return false;
   llvm::SmallVector<lingodb::runtime::FilterDescription, 8> sa(a.begin(), a.end());
   llvm::SmallVector<lingodb::runtime::FilterDescription, 8> sb(b.begin(), b.end());
   llvm::sort(sa, filterDescrLess);
   llvm::sort(sb, filterDescrLess);
   for (size_t i = 0; i < sa.size(); ++i) {
      if (!(sa[i] == sb[i])) return false;
   }
   return true;
}

/// Compare pushdown filter descr (`filterDescriptions` + `orFilterClauses`) independent of mapping.
static bool externalDatasourceFiltersEqual(const ExternalDatasourceProperty& a,
                                           const ExternalDatasourceProperty& b) {
   if (!filterClauseEquals(a.filterDescriptions, b.filterDescriptions)) return false;
   if (a.orFilterClauses.size() != b.orFilterClauses.size()) return false;
   llvm::SmallVector<bool, 4> matched(b.orFilterClauses.size(), false);
   for (const auto& clauseA : a.orFilterClauses) {
      bool found = false;
      for (size_t i = 0; i < b.orFilterClauses.size(); ++i) {
         if (matched[i]) continue;
         if (!filterClauseEquals(clauseA, b.orFilterClauses[i])) continue;
         matched[i] = true;
         found = true;
         break;
      }
      if (!found) return false;
   }
   return true;
}

static bool allExternalFilterSourcesIdentical(llvm::ArrayRef<ExternalDatasourceProperty> filterSources) {
   if (filterSources.size() <= 1) return true;
   for (size_t i = 1; i < filterSources.size(); ++i) {
      if (!externalDatasourceFiltersEqual(filterSources.front(), filterSources[i])) return false;
   }
   return true;
}

static bool externalDatasourceHasValueFilter(const ExternalDatasourceProperty& ds) {
   auto isValueFilter = [](const lingodb::runtime::FilterDescription& f) {
      return f.op != lingodb::runtime::FilterOp::NOTNULL;
   };
   if (llvm::any_of(ds.filterDescriptions, isValueFilter)) return true;
   for (const auto& clause : ds.orFilterClauses)
      if (llvm::any_of(clause, isValueFilter)) return true;
   return false;
}

/// Merge pushdown filters from matched queries into `(filterDescriptions AND ...) OR (orFilterClauses[i] AND ...)`.
static void mergeExternalFiltersForOrReuse(ExternalDatasourceProperty& merged,
                                           llvm::ArrayRef<ExternalDatasourceProperty> filterSources) {
   if (llvm::any_of(filterSources, [](const ExternalDatasourceProperty& ds) {
          return !externalDatasourceHasValueFilter(ds);
       })) {
      merged.filterDescriptions.clear();
      merged.orFilterClauses.clear();
      return;
   }

   if (allExternalFilterSourcesIdentical(filterSources)) {
      merged.filterDescriptions = filterSources.front().filterDescriptions;
      merged.orFilterClauses.clear();
      return;
   }

   llvm::SmallVector<llvm::SmallVector<lingodb::runtime::FilterDescription, 8>, 4> uniqueClauses;
   auto tryAddClause = [&](llvm::ArrayRef<lingodb::runtime::FilterDescription> clause) {
      if (clause.empty()) return;
      for (const auto& existing : uniqueClauses) {
         if (filterClauseEquals(existing, clause)) return;
      }
      uniqueClauses.emplace_back(clause.begin(), clause.end());
   };
   for (const ExternalDatasourceProperty& src : filterSources) {
      tryAddClause(src.filterDescriptions);
      for (const auto& clause : src.orFilterClauses) tryAddClause(clause);
   }
   merged.filterDescriptions.clear();
   merged.orFilterClauses.clear();
   if (uniqueClauses.empty()) return;
   merged.filterDescriptions.assign(uniqueClauses.front().begin(), uniqueClauses.front().end());
   if (uniqueClauses.size() == 1) return;
   for (size_t i = 1; i < uniqueClauses.size(); ++i) {
      merged.orFilterClauses.emplace_back(uniqueClauses[i].begin(), uniqueClauses[i].end());
   }
}

static subop::ExecutionStepOp createMergedExternalTableRefStep(mlir::OpBuilder& gb, mlir::Location loc,
                                                               subop::TableType tableTy, llvm::StringRef descrHex) {
   auto step = gb.create<subop::ExecutionStepOp>(loc, mlir::TypeRange{tableTy}, mlir::ValueRange{},
                                                gb.getArrayAttr({gb.getBoolAttr(false)}));
   auto& block = step.getSubOps().emplaceBlock();
   mlir::OpBuilder ib = mlir::OpBuilder::atBlockBegin(&block);
   auto ge = ib.create<subop::GetExternalOp>(loc, tableTy, mlir::StringAttr::get(gb.getContext(), descrHex));
   ib.create<subop::ExecutionStepReturnOp>(loc, ge.getRes());
   return step;
}

static void rewireBuildStepScannedTable(subop::ExecutionStepOp buildStep, subop::ScanRefsOp scanOp,
                                        mlir::Value oldTableState, mlir::Value newTableState, subop::TableType newTy) {
   mlir::Value oldCanon = canonicalizeStateValueForReuse(oldTableState);
   mlir::Block& body = buildStep.getSubOps().front();
   for (unsigned i = 0; i < buildStep.getNumOperands(); ++i) {
      if (canonicalizeStateValueForReuse(buildStep.getOperand(i)) != oldCanon) continue;
      buildStep.setOperand(i, newTableState);
      if (i < body.getNumArguments()) body.getArgument(i).setType(newTy);
   }
   mlir::Value scanSt = scanOp.getState();
   if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(scanSt)) {
      assert(ba.getOwner() == &body && "scan_refs table state must be a build-step block argument");
      ba.setType(newTy);
   } else {
      assert(canonicalizeStateValueForReuse(scanSt) == oldCanon && "scan_refs table state must match rewired operand");
      scanOp.getStateMutable().assign(newTableState);
   }
}

static mlir::Type memberTypeForIdentifier(subop::TableType tableTy, subop::MemberManager& mm,
                                          llvm::StringRef identifier) {
   if (!tableTy) return {};
   llvm::StringRef id = normalizeColumnIdentifier(identifier);
   for (subop::Member m : tableTy.getMembers().getMembers()) {
      if (stripMemberSuffix(mm.getName(m)) == id) return mm.getType(m);
   }
   return {};
}

static subop::Member tableMemberForIdentifier(subop::TableType tableTy, subop::MemberManager& mm,
                                              llvm::StringRef identifier);

static subop::TableType tableTypeFromMergedExternal(mlir::MLIRContext* ctx, subop::MemberManager& mm,
                                                    const ExternalDatasourceProperty& ds, subop::TableType hintTy,
                                                    llvm::function_ref<mlir::Type(llvm::StringRef)> lookupPeerType) {
   llvm::SmallVector<subop::Member> members;
   for (const auto& map : ds.mapping) {
      mlir::Type ty = memberTypeForIdentifier(hintTy, mm, map.identifier);
      if (!ty) ty = lookupPeerType(map.identifier);
      assert(ty && "join superset: missing column type in merged external layout");
      if (subop::Member existing = tableMemberForIdentifier(hintTy, mm, map.identifier)) {
         assert(mm.getType(existing) == ty && "join superset: column type mismatch for reused table member");
         members.push_back(existing);
      } else {
         members.push_back(mm.createMember(map.identifier, ty));
      }
   }
   return subop::TableType::get(ctx, subop::StateMembersAttr::get(ctx, members), hintTy.getFiltered());
}

static subop::Member tableMemberForIdentifier(subop::TableType tableTy, subop::MemberManager& mm,
                                              llvm::StringRef identifier) {
   if (!tableTy) return {};
   llvm::StringRef id = normalizeColumnIdentifier(identifier);
   for (subop::Member m : tableTy.getMembers().getMembers()) {
      if (stripMemberSuffix(mm.getName(m)) == id) return m;
   }
   return {};
}

static llvm::StringRef semanticKeyLeaf(llvm::StringRef semanticKey) {
   return semanticKey.split('\x1f').second;
}

/// Propagate a widened \c !subop.table type along SSA values and \c execution_step operand/block-arg ports.
static void refreshTableStateTypesInModule(mlir::ModuleOp module, mlir::Value tableStateRoot, subop::TableType newTy) {
   if (!tableStateRoot || !newTy) return;

   llvm::DenseSet<void*> closure;
   llvm::SmallVector<mlir::Value, 16> worklist;
   auto seed = [&](mlir::Value v) {
      if (!v) return;
      closure.insert(v.getAsOpaquePointer());
      worklist.push_back(v);
   };
   seed(tableStateRoot);
   seed(peelBlockArgsToEnclosingOperands(tableStateRoot));

   expandClosureThroughExecutionStepPorts(module, closure);

   llvm::DenseSet<void*> visited;
   while (!worklist.empty()) {
      mlir::Value v = worklist.pop_back_val();
      if (!visited.insert(v.getAsOpaquePointer()).second) continue;
      v.setType(newTy);

      if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(v)) {
         if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(ba.getOwner()->getParentOp())) {
            unsigned i = ba.getArgNumber();
            if (i < step.getNumOperands()) seed(step.getOperand(i));
         }
      } else if (mlir::Operation* def = v.getDefiningOp()) {
         if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(def)) {
            if (auto ret = mlir::dyn_cast<subop::ExecutionStepReturnOp>(step.getSubOps().front().getTerminator())) {
               for (unsigned i = 0; i < ret.getNumOperands() && i < step.getNumResults(); ++i) {
                  if (ret.getOperand(i) == v) seed(step.getResult(i));
               }
            }
         }
      }

      for (mlir::OpOperand& use : v.getUses()) {
         mlir::Operation* user = use.getOwner();
         if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(user)) {
            unsigned i = use.getOperandNumber();
            mlir::Block& body = step.getSubOps().front();
            if (i < body.getNumArguments()) seed(body.getArgument(i));
         }
      }
   }

   synchronizeExecutionStepPortTypes(module, &closure);

   auto* ctx = module.getContext();
   llvm::SmallVector<subop::Member> refCols;
   for (subop::Member m : newTy.getMembers().getMembers()) refCols.push_back(m);
   auto entryRefTy = subop::TableEntryRefType::get(ctx, subop::StateMembersAttr::get(ctx, refCols));

   // Table scan refs use ColumnManager's global (scope,leaf) objects. Widening must not mutate
   // ref.getColumn().type in place — other modules may still reference the same Column*.
   subop::ColumnUsageAnalysis columnUsage(module);
   subop::SubOpStateUsageTransformer transformer(columnUsage, ctx,
                                                 [&](mlir::Operation*, mlir::Type) { return entryRefTy; });
   llvm::DenseMap<tuples::Column*, tuples::ColumnDefAttr> widenedScanRefByColumn;
   module.walk([&](subop::ScanRefsOp scan) {
      if (!opaqueClosureContains(closure, scan.getState())) return;
      auto oldRef = scan.getRef();
      tuples::Column* oldCol = &oldRef.getColumn();
      auto [it, inserted] = widenedScanRefByColumn.try_emplace(oldCol);
      if (inserted) it->second = transformer.createReplacementColumn(oldRef, entryRefTy);
      scan.setRefAttr(it->second);
   });
   module.walk([&](subop::MapOp map) {
      llvm::SmallVector<mlir::Attribute> computed;
      bool changed = false;
      computed.reserve(map.getComputedCols().size());
      for (auto attr : map.getComputedCols()) {
         auto def = mlir::cast<tuples::ColumnDefAttr>(attr);
         auto it = widenedScanRefByColumn.find(&def.getColumn());
         if (it != widenedScanRefByColumn.end()) {
            computed.push_back(it->second);
            changed = true;
         } else {
            computed.push_back(attr);
         }
      }
      if (changed) map.setComputedColsAttr(mlir::ArrayAttr::get(ctx, computed));
   });
}

static void collectMapOpsOnStreamChain(mlir::Value stream, llvm::SmallVector<subop::MapOp, 8>& out) {
   llvm::DenseSet<void*> seen;
   llvm::SmallVector<mlir::Value, 8> worklist = {stream};
   while (!worklist.empty()) {
      mlir::Value v = worklist.pop_back_val();
      if (!seen.insert(v.getAsOpaquePointer()).second) continue;
      for (mlir::Operation* user : v.getUsers()) {
         if (auto mapOp = mlir::dyn_cast<subop::MapOp>(user)) {
            if (mapOp.getStream() == v) out.push_back(mapOp);
         } else if (auto nested = mlir::dyn_cast<subop::NestedMapOp>(user)) {
            if (nested.getStream() == v) worklist.push_back(nested.getRes());
         }
      }
   }
}

static void syncMapInputColsFromGather(subop::GatherOp gather, tuples::ColumnManager& cm,
                                       const llvm::StringMap<tuples::ColumnRefAttr>* extraRefsByKey = nullptr) {
   auto* ctx = gather.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   llvm::StringMap<tuples::ColumnRefAttr> outRefByKey;
   llvm::StringMap<tuples::ColumnRefAttr> outRefByNormLeaf;
   for (auto& pr : gather.getMapping().getMapping()) {
      auto [scope, leaf] = cm.getName(&pr.second.getColumn());
      tuples::ColumnRefAttr outRef = cm.createRef(&pr.second.getColumn());
      outRefByKey[columnSemanticKey(scope, leaf)] = outRef;
      outRefByNormLeaf[leaf] = outRef;
      outRefByNormLeaf[mm.getName(pr.first)] = outRef;
   }
   if (extraRefsByKey) {
      for (const auto& entry : *extraRefsByKey) {
         outRefByKey[entry.getKey()] = entry.getValue();
      }
   }
   if (outRefByKey.empty()) return;
   llvm::SmallVector<subop::MapOp, 8> mapOps;
   collectMapOpsOnStreamChain(gather.getRes(), mapOps);
   for (subop::MapOp mapOp : mapOps) {
      bool changed = false;
      llvm::SmallVector<mlir::Attribute> newInputs;
      for (auto attr : mapOp.getInputCols()) {
         auto cref = mlir::cast<tuples::ColumnRefAttr>(attr);
         auto [scope, leaf] = cm.getName(&cref.getColumn());
         tuples::ColumnRefAttr replacement;
         if (auto it = outRefByKey.find(columnSemanticKey(scope, leaf)); it != outRefByKey.end()) {
            replacement = it->second;
         } else if (isPayloadMemberSlotName(leaf) || scope.starts_with("lookup_u_")) {
            if (auto it = outRefByNormLeaf.find(normalizeColumnIdentifier(leaf)); it != outRefByNormLeaf.end()) {
               replacement = it->second;
            } else if (auto it = outRefByNormLeaf.find(leaf); it != outRefByNormLeaf.end()) {
               replacement = it->second;
            }
         }
         if (replacement && &cref.getColumn() != &replacement.getColumn()) {
            newInputs.push_back(replacement);
            changed = true;
         } else {
            newInputs.push_back(cref);
         }
      }
      if (changed) mapOp.setInputColsAttr(mlir::ArrayAttr::get(ctx, newInputs));
   }
}

static void propagateJoinSupersetColumnAttrs(mlir::ModuleOp module,
                                             const llvm::DenseSet<void*>* closureFilter,
                                             subop::HashIndexedViewType producerHiv, bool syncGatherOps,
                                             llvm::StringRef lookupListScope = {}) {
   auto* ctx = module.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto expectedLer = subop::LookupEntryRefType::get(ctx, producerHiv);
   auto expectedListTy = subop::ListType::get(ctx, expectedLer);
   auto shouldUpdateOp = [&](mlir::Operation* op) {
      if (!closureFilter) return true;
      return opOperandsOrNestedBlockArgsTouchClosure(op, *closureFilter);
   };
   auto syncLookupEntryRefToHiv = [&](tuples::ColumnRefAttr& cref) -> bool {
      bool changed = false;
      if (mlir::isa<subop::LookupEntryRefType>(cref.getColumn().type)) {
         if (!lookupEntryRefStateHasLayout(cref.getColumn().type, producerHiv)) return false;
         if (cref.getColumn().type != expectedLer) {
            cref.getColumn().type = expectedLer;
            changed = true;
         }
      } else if (auto list = mlir::dyn_cast<subop::ListType>(cref.getColumn().type)) {
         if (!lookupEntryRefStateHasLayout(list.getT(), producerHiv)) return false;
         if (cref.getColumn().type != expectedListTy) {
            cref.getColumn().type = expectedListTy;
            changed = true;
         }
      }
      return changed;
   };
   auto syncLookupEntryDefToHiv = [&](tuples::ColumnDefAttr& def) -> bool {
      if (!lookupEntryRefStateHasLayout(def.getColumn().type, producerHiv)) return false;
      if (def.getColumn().type == expectedLer) return false;
      def.getColumn().type = expectedLer;
      return true;
   };
   if (syncGatherOps) {
      module.walk([&](subop::GatherOp op) {
         if (!shouldUpdateOp(op.getOperation())) return;
         auto r = op.getRef();
         bool changed = syncLookupEntryRefToHiv(r);
         auto m = op.getMapping();
         llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> out;
         for (auto [mem, def] : m.getMapping()) {
            tuples::ColumnDefAttr d = def;
            if (syncLookupEntryDefToHiv(d)) changed = true;
            out.push_back({mem, d});
         }
         if (changed) {
            op.setRefAttr(r);
            op.setMappingAttr(subop::ColumnDefMemberMappingAttr::get(ctx, out));
         }
         syncMapInputColsFromGather(op, cm);
      });
   }
   module.walk([&](subop::MaterializeOp op) {
      if (!shouldUpdateOp(op.getOperation())) return;
      auto m = op.getMapping();
      llvm::SmallVector<std::pair<subop::Member, tuples::ColumnRefAttr>> out;
      bool changed = false;
      for (auto [mem, cref] : m.getMapping()) {
         tuples::ColumnRefAttr c = cref;
         if (syncLookupEntryRefToHiv(c)) changed = true;
         out.push_back({mem, c});
      }
      if (changed) op.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, out));
   });
   module.walk([&](subop::LookupOp op) {
      if (!shouldUpdateOp(op.getOperation())) return;
      if (closureFilter && !opaqueClosureContains(*closureFilter, op.getState())) return;
      auto hivTy = asHashIndexedViewLayoutType(op.getState().getType());
      if (!hivTy || hivTy != producerHiv) return;
      auto r = op.getRef();
      if (r.getColumn().type != expectedListTy) {
         r.getColumn().type = expectedListTy;
         op.setRefAttr(r);
      }
      for (mlir::Operation* user : op.getResult().getUsers()) {
         auto nm = mlir::dyn_cast<subop::NestedMapOp>(user);
         if (!nm) continue;
         mlir::Region& reg = nm.getRegion();
         if (reg.empty()) continue;
         for (mlir::BlockArgument barg : reg.front().getArguments()) {
            if (!mlir::isa<subop::ListType>(barg.getType())) continue;
            if (barg.getType() != expectedListTy) barg.setType(expectedListTy);
         }
      }
   });
   module.walk([&](subop::ScanListOp op) {
      if (!shouldUpdateOp(op.getOperation())) return;
      auto elem = op.getElem();
      if (!lookupListScope.empty()) {
         auto [elemScope, elemLeaf] = cm.getName(&elem.getColumn());
         (void)elemLeaf;
         if (elemScope != lookupListScope) return;
      }
      if (syncLookupEntryDefToHiv(elem)) op.setElemAttr(elem);
   });
}

static void syncProbeListCarriersInClosure(mlir::ModuleOp module, const llvm::DenseSet<void*>* closureFilter) {
   auto* ctx = module.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   module.walk([&](subop::ExecutionStepOp step) {
      if (closureFilter && !opOperandsOrNestedBlockArgsTouchClosure(step.getOperation(), *closureFilter)) return;
      llvm::StringMap<const tuples::Column*> entryColByKey;
      step.walk([&](subop::ScanListOp scan) {
         auto listTy = mlir::dyn_cast<subop::ListType>(scan.getList().getType());
         if (!listTy) return;
         auto elem = scan.getElem();
         if (elem.getColumn().type != listTy.getT()) {
            elem.getColumn().type = listTy.getT();
         }
         auto [scope, leaf] = cm.getName(&elem.getColumn());
         entryColByKey[columnSemanticKey(scope, leaf)] = &elem.getColumn();
      });
      step.walk([&](subop::GatherOp gather) {
         auto ref = gather.getRef();
         if (!mlir::isa<subop::LookupEntryRefType>(ref.getColumn().type)) return;
         auto [scope, leaf] = cm.getName(&ref.getColumn());
         if (auto it = entryColByKey.find(columnSemanticKey(scope, leaf)); it != entryColByKey.end()) {
            ref.getColumn().type = it->second->type;
            gather.setRefAttr(ref);
         }
      });
   });
}

static void propagateJoinSupersetColumnAttrsForClosure(mlir::ModuleOp module,
                                                       const llvm::DenseSet<void*>& closureFilter) {
   llvm::DenseSet<const void*> seenHiv;
   auto& cm = module.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   module.walk([&](subop::LookupOp op) {
      if (!opOperandsOrNestedBlockArgsTouchClosure(op.getOperation(), closureFilter)) return;
      auto hivTy = asHashIndexedViewLayoutType(op.getState().getType());
      if (!hivTy) return;
      if (!seenHiv.insert(hivTy.getAsOpaquePointer()).second) return;
      auto listRef = op.getRef();
      auto [listScope, listLeaf] = cm.getName(&listRef.getColumn());
      propagateJoinSupersetColumnAttrs(module, &closureFilter, hivTy, true, listScope);
   });
   syncProbeListCarriersInClosure(module, &closureFilter);
}

static void patchBufferBuildStepForUnion(mlir::ModuleOp synthetic, subop::ExecutionStepOp buildStep,
                                         const JoinBufferUnionPlan& plan,
                                         const ModuleReuseInfo& reuseSynthetic,
                                         llvm::ArrayRef<std::pair<mlir::ModuleOp, mlir::Value>> peerHivs,
                                         llvm::ArrayRef<const ModuleReuseInfo*> peerReuses,
                                         const llvm::DenseMap<const void*, uint64_t>& columnHashes) {
   assert(peerHivs.size() == peerReuses.size() && "join superset: peer HIVs and reuse metadata must align");
   mlir::Block& body = buildStep.getSubOps().front();
   auto* ctx = synthetic.getContext();
   auto* subDialect = ctx->getLoadedDialect<subop::SubOperatorDialect>();
   auto* tupleDialect = ctx->getLoadedDialect<tuples::TupleStreamDialect>();
   auto& mm = subDialect->getMemberManager();
   auto& cm = tupleDialect->getColumnManager();

   subop::ScanRefsOp scanOp = findDonorTableScanInUnionPlan(buildStep, plan, reuseSynthetic);
   if (!scanOp) {
      for (const PayloadColumnSpec& spec : plan.payloadColumns) {
         unsigned predIdx = 0;
         if (parseFilterPredLayoutSemanticKey(spec.semanticKey, predIdx)) continue;
         scanOp = findTableScanForPayloadLeaf(buildStep, spec.leaf, mm);
         if (scanOp) break;
      }
   }
   if (!scanOp) return;

   auto rewriteDonorExternalTableForUnion = [&]() {
      mlir::Value tableState = scanOp.getState();
      ExternalDatasourceProperty mergedDs;
      bool haveDs = false;
      llvm::StringRef donorTableName;
      subop::TableType donorTableTy;
      bool haveDonorExternal = resolveScannedTableExternal(buildStep, tableState, reuseSynthetic, donorTableName,
                                                           mergedDs, haveDs, donorTableTy) &&
                               haveDs &&
                               countUnionPayloadLeavesOnTable(plan.payloadColumns, donorTableTy, mm) > 0;
      if (!haveDonorExternal) return;

      llvm::SmallVector<ExternalDatasourceProperty, 4> filterSources;
      filterSources.push_back(mergedDs);
      for (size_t pi = 0; pi < peerHivs.size(); ++pi) {
         auto [peerMod, peerHiv] = peerHivs[pi];
         if (!peerMod || !peerHiv) continue;
         const ModuleReuseInfo& reusePeer = *peerReuses[pi];
         subop::ExecutionStepOp peerBuild = findJoinBufferBuildStepForHiv(peerMod, peerHiv, reusePeer);
         if (!peerBuild) continue;
         if (auto resolved = resolveExternalTableScanForDonor(peerBuild, reusePeer, donorTableName))
            filterSources.push_back(std::move(resolved->datasource));
         mergePeerExternalFromBuildStepScan(mergedDs, haveDs, peerBuild, reusePeer, donorTableName);
      }
      assert(haveDs && "join superset: merged external datasource required for donor table");
      mergeExternalFiltersForOrReuse(mergedDs, filterSources);

      auto lookupPeerColumnType = [&](llvm::StringRef identifier) -> mlir::Type {
         for (size_t pi = 0; pi < peerHivs.size(); ++pi) {
            auto [peerMod, peerHiv] = peerHivs[pi];
            if (!peerMod || !peerHiv) continue;
            const ModuleReuseInfo& reusePeer = *peerReuses[pi];
            subop::ExecutionStepOp peerBuild = findJoinBufferBuildStepForHiv(peerMod, peerHiv, reusePeer);
            if (!peerBuild) continue;
            if (mlir::Type ty =
                   columnTypeForIdentifierFromPeerBuildScan(peerBuild, reusePeer, donorTableName, identifier)) {
               return ty;
            }
         }
         return {};
      };

      assert(donorTableTy && "join superset: resolveScannedTableExternal must provide donor table type");
      auto newTableTy = tableTypeFromMergedExternal(ctx, mm, mergedDs, donorTableTy, lookupPeerColumnType);
      for (auto& map : mergedDs.mapping) {
         if (subop::Member m = tableMemberForIdentifier(newTableTy, mm, map.identifier))
            map.memberName = mm.getName(m);
      }
      llvm::sort(mergedDs.mapping, [](const auto& x, const auto& y) { return x.memberName < y.memberName; });
      std::string hex = lingodb::utility::serializeToHexString(mergedDs);
      subop::ExecutionGroupOp eg = buildStep->getParentOfType<subop::ExecutionGroupOp>();
      assert(eg && "join superset: buffer build step must live in an execution_group");
      mlir::OpBuilder gb = mlir::OpBuilder::atBlockBegin(&eg.getSubOps().front());
      gb.setInsertionPoint(buildStep);
      subop::ExecutionStepOp mergedTableRefStep =
         createMergedExternalTableRefStep(gb, buildStep.getLoc(), newTableTy, hex);
      mlir::Value mergedTableState = mergedTableRefStep.getResult(0);
      rewireBuildStepScannedTable(buildStep, scanOp, tableState, mergedTableState, newTableTy);
      refreshTableStateTypesInModule(synthetic, mergedTableState, newTableTy);
   };
   rewriteDonorExternalTableForUnion();

   subop::MaterializeOp matOp = findJoinBufferMaterializeInStep(buildStep);
   assert(matOp && "join superset: buffer build step must materialize into join buffer");

   auto collectMaterializedPayloadHashes = [&]() {
      llvm::DenseSet<uint64_t> keys;
      for (auto& [member, colRef] : matOp.getMapping().getMapping()) {
         if (member == plan.linkMember || member == plan.hashMember) continue;
         keys.insert(payloadColumnIdentityHash(colRef, columnHashes));
      }
      return keys;
   };
   llvm::DenseSet<uint64_t> materializedHashes = collectMaterializedPayloadHashes();

   subop::MapOp hashMapOp = findJoinHashMapBeforeMaterialize(body, matOp);
   assert(hashMapOp);

   auto appendMaterializeMapping = [&](subop::Member bufMem, tuples::ColumnDefAttr colDef) {
      llvm::SmallVector<subop::RefMappingPairT> matPairs;
      for (auto pr : matOp.getMapping().getMapping()) matPairs.push_back(pr);
      matPairs.push_back({bufMem, cm.createRef(&colDef.getColumn())});
      matOp.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, matPairs));
   };

   auto isPayloadMaterialized = [&](const PayloadColumnSpec& spec) {
      if (materializedHashes.contains(spec.semanticHash)) return true;
      return false;
   };

   auto appendUnionPayloadGather = [&](const PayloadColumnSpec& spec, subop::Member bufMem) {
      assert(!spec.scope.empty() && "join superset: payload column scope required");

      subop::ScanRefsOp specScan = findTableScanForPayloadLeaf(buildStep, spec.leaf, mm);
      if (!specScan) specScan = scanOp;
      subop::TableType tableTy = mlir::dyn_cast<subop::TableType>(specScan.getState().getType());
      assert(tableTy && "join superset: table type required for payload column gather");
      subop::Member tableMem = tableMemberForIdentifier(tableTy, mm, spec.leaf);
      if (!tableMem) return false;

      tuples::ColumnDefAttr colDef = cm.createDef(spec.scope, spec.leaf);
      colDef.getColumn().type = cloneTypeToContext(spec.colType, ctx);

      // Chain union-only gathers on the current materialize stream (clone build chain + prior
      // union-only gathers), not from hashMapOp each time.
      mlir::Value mapStream = matOp.getStream();
      mlir::Operation* streamAnchor = mapStream.getDefiningOp();
      if (!streamAnchor) streamAnchor = hashMapOp;
      mlir::OpBuilder gb(streamAnchor);
      gb.setInsertionPointAfter(streamAnchor);
      auto mapping = subop::ColumnDefMemberMappingAttr::get(
         ctx, llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>>{{tableMem, colDef}});
      auto gatherTy = mapStream.getType();
      auto refDef = specScan.getRef();
      auto gatherRef = cm.createRef(&refDef.getColumn());
      auto newGather =
         gb.create<subop::GatherOp>(mlir::UnknownLoc::get(ctx), gatherTy, mapStream, gatherRef, mapping);
      matOp->setOperand(0, newGather.getRes());
      appendMaterializeMapping(bufMem, colDef);
      materializedHashes.insert(spec.semanticHash);
      return true;
   };

   for (size_t i = 0; i < plan.payloadColumns.size(); ++i) {
      const PayloadColumnSpec& spec = plan.payloadColumns[i];
      unsigned reusePredQueryIdx = 0;
      if (parseFilterPredLayoutSemanticKey(spec.semanticKey, reusePredQueryIdx)) {
         continue;
      }
      if (isPayloadMaterialized(spec)) continue;

      assert(i < plan.payloadMembers.size() && "join superset: payload slot out of range");
      (void)appendUnionPayloadGather(spec, plan.payloadMembers[i]);
   }

   buildStep.walk([&](subop::GatherOp gather) { syncMapInputColsFromGather(gather, cm); });
}

static CachedJoinBufferLayout layoutFromUnionPlan(subop::HashIndexedViewType producerHiv,
                                                  const JoinBufferUnionPlan& plan);
struct ProbeAlignDebugCtx;
static void remapClosureGathersToAlignedConsumerHiv(mlir::ModuleOp consumer, const llvm::DenseSet<void*>* ssaClosure,
                                                    subop::HashIndexedViewType alignedHiv,
                                                    const CachedJoinBufferLayout& consumerLayout,
                                                    const llvm::StringSet<>* probeLookupScopes,
                                                    const ProbeAlignDebugCtx* dbg);
static void remapAlignedHivClosureGathers(mlir::ModuleOp module, const llvm::DenseSet<void*>* ssaClosure,
                                          subop::HashIndexedViewType alignedHiv,
                                          const CachedJoinBufferLayout& layout,
                                          const llvm::StringSet<>* probeLookupScopes,
                                          std::optional<uint64_t> cacheKey,
                                          llvm::StringRef passName);

struct SyntheticJoinBuildSite {
   mlir::Value mergedBuffer;
   subop::ExecutionStepOp buildStep;
   subop::MaterializeOp materialize;
};

static SyntheticJoinBuildSite findSyntheticJoinBuildSite(mlir::ModuleOp synthetic, mlir::Value syntheticHiv,
                                                         const ModuleReuseInfo& reuseSynthetic) {
   SyntheticJoinBuildSite site;
   site.mergedBuffer = resolveJoinMergedBuffer(syntheticHiv, synthetic, reuseSynthetic);
   site.buildStep = findBufferBuildStepWithTableMaterialize(site.mergedBuffer, reuseSynthetic);
   if (!site.buildStep) site.buildStep = findBufferBuildStepWithTableScan(synthetic);
   if (!site.buildStep) return site;

   site.buildStep.walk([&](subop::MaterializeOp mat) {
      if (site.materialize) return;
      if (!materializeTargetsJoinBuffer(mat, site.mergedBuffer, reuseSynthetic)) return;
      site.materialize = mat;
   });
   if (!site.materialize) site.materialize = findJoinBufferMaterializeInStep(site.buildStep);
   return site;
}

static subop::HashIndexedViewType findSyntheticProducerHivForMergedBuffer(mlir::ModuleOp synthetic,
                                                                          mlir::Value mergedBuffer) {
   subop::HashIndexedViewType producerHiv;
   mlir::Value canonMergedBuf = canonicalizeStateValueForReuse(mergedBuffer);
   synthetic.walk([&](subop::CreateHashIndexedView chiv) {
      if (producerHiv) return;
      if (canonicalizeStateValueForReuse(chiv.getSource()) != canonMergedBuf) return;
      producerHiv = asHashIndexedViewLayoutType(chiv.getResult().getType());
   });
   return producerHiv;
}

static void finalizeSyntheticJoinProducerClosure(mlir::ModuleOp synthetic, llvm::ArrayRef<mlir::Value> roots,
                                                 const JoinBufferUnionPlan& plan,
                                                 const ModuleReuseInfo& reuseSynthetic) {
   JoinBufferHivSsaClosure joinClosure = computeJoinBufferHivSsaClosure(roots, reuseSynthetic);
   alignBufferMergeThreadLocalsWithMergeResult(synthetic, &joinClosure.opaque);

   subop::HashIndexedViewType producerHiv = findSyntheticProducerHivForMergedBuffer(synthetic, roots.front());
   assert(producerHiv && "join superset: synthetic must create hash_indexed_view on merged buffer");

   expandClosureThroughExecutionStepPorts(synthetic, joinClosure.opaque);
   remapAlignedHivClosureGathers(synthetic, &joinClosure.opaque, producerHiv, layoutFromUnionPlan(producerHiv, plan),
                                 /*probeLookupScopes=*/nullptr, /*cacheKey=*/std::nullopt,
                                 /*passName=*/"");
   propagateJoinSupersetColumnAttrsForClosure(synthetic, joinClosure.opaque);
   synchronizeExecutionStepPortTypes(synthetic, nullptr);
}

static void applyUnionPlanToSyntheticHiv(mlir::ModuleOp synthetic, mlir::Value syntheticHiv, JoinBufferUnionPlan& plan,
                                         const ModuleReuseInfo& reuseSynthetic,
                                         mlir::ModuleOp query0Module, mlir::Value hivA, const ModuleReuseInfo& reuseA,
                                         mlir::ModuleOp query1Module, mlir::Value hivB,
                                         const ModuleReuseInfo& reuseB,
                                         llvm::ArrayRef<std::pair<mlir::ModuleOp, mlir::Value>> extraPeerHivs = {},
                                         llvm::ArrayRef<const ModuleReuseInfo*> extraPeerReuses = {}) {
   assert(extraPeerHivs.size() == extraPeerReuses.size() &&
          "join superset: extra peer HIVs and reuse metadata must align");
   auto* ctx = synthetic.getContext();
   auto* subDialect = ctx->getLoadedDialect<subop::SubOperatorDialect>();
   auto& mm = subDialect->getMemberManager();
   SyntheticJoinBuildSite site = findSyntheticJoinBuildSite(synthetic, syntheticHiv, reuseSynthetic);
   assert(site.materialize && "join superset: synthetic build step must materialize into merged join buffer");
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   llvm::DenseMap<const void*, uint64_t> syntheticColumnHashes =
      collectStateConstructionColumnHashes(synthetic, syntheticHiv);
   assignPayloadMembersForPlan(mm, cm, site.materialize, plan, syntheticColumnHashes);
   auto targetMembers = bufferMembersForPlan(ctx, plan);
   llvm::SmallVector<mlir::Value, 4> roots = {site.mergedBuffer};

   applyBufferLayoutToSsaClosure(synthetic, roots, targetMembers, reuseSynthetic, &syntheticColumnHashes);

   if (site.buildStep) {
      llvm::SmallVector<std::pair<mlir::ModuleOp, mlir::Value>, 8> peerHivs = {
         {query1Module, hivB},
         {query0Module, hivA},
      };
      llvm::SmallVector<const ModuleReuseInfo*, 8> peerReuses = {&reuseB, &reuseA};
      peerHivs.append(extraPeerHivs.begin(), extraPeerHivs.end());
      peerReuses.append(extraPeerReuses.begin(), extraPeerReuses.end());
      patchBufferBuildStepForUnion(synthetic, site.buildStep, plan, reuseSynthetic, peerHivs, peerReuses,
                                   syntheticColumnHashes);
   }

   applyBufferLayoutToSsaClosure(synthetic, roots, targetMembers, reuseSynthetic, &syntheticColumnHashes);
   finalizeSyntheticJoinProducerClosure(synthetic, roots, plan, reuseSynthetic);
}

static bool typeEmbedsHashIndexedView(mlir::Type t) {
   if (mlir::isa<subop::HashIndexedViewType>(t)) return true;
   if (auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(t)) return !!ler.getState();
   if (auto list = mlir::dyn_cast<subop::ListType>(t)) return typeEmbedsHashIndexedView(list.getT());
   return false;
}

static bool lookupEntryRefEmbedsHashIndexedView(subop::LookupEntryRefType ler) {
   return mlir::isa<subop::HashIndexedViewType>(ler.getState());
}

static bool sameHashIndexedViewLayout(subop::HashIndexedViewType a, subop::HashIndexedViewType b) {
   if (!a || !b) return false;
   if (a == b) return true;
   return a.getKeyMembers().getMembers() == b.getKeyMembers().getMembers() &&
          a.getValueMembers().getMembers() == b.getValueMembers().getMembers() &&
          a.getCompareHashForLookup() == b.getCompareHashForLookup();
}

static bool sameHashIndexedViewJoinKey(subop::HashIndexedViewType a, subop::HashIndexedViewType b) {
   if (!a || !b) return false;
   return a.getKeyMembers().getMembers() == b.getKeyMembers().getMembers() &&
          a.getCompareHashForLookup() == b.getCompareHashForLookup();
}

static bool typeEmbedsHashIndexedViewState(mlir::Type t, subop::HashIndexedViewType hiv) {
   if (!t || !hiv) return false;
   if (auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(t)) {
      if (!lookupEntryRefEmbedsHashIndexedView(ler)) return false;
      auto st = mlir::dyn_cast<subop::HashIndexedViewType>(ler.getState());
      return st && sameHashIndexedViewLayout(st, hiv);
   }
   if (auto list = mlir::dyn_cast<subop::ListType>(t)) {
      return typeEmbedsHashIndexedViewState(list.getT(), hiv);
   }
   return false;
}

static mlir::Type replaceEmbeddedHivInType(mlir::MLIRContext* ctx, mlir::Type t,
                                           subop::HashIndexedViewType producerHiv,
                                           subop::HashIndexedViewType consumerHivBeforeAlign) {
   if (!t) return t;
   if (mlir::isa<subop::HashIndexedViewType>(t)) {
      if (t == producerHiv) return t;
      if (consumerHivBeforeAlign && t != consumerHivBeforeAlign) return t;
      return producerHiv;
   }
   if (auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(t)) {
      if (!lookupEntryRefEmbedsHashIndexedView(ler)) return t;
      if (ler.getState() == producerHiv) return t;
      if (consumerHivBeforeAlign && ler.getState() != consumerHivBeforeAlign) return t;
      return subop::LookupEntryRefType::get(ctx, producerHiv);
   }
   if (auto list = mlir::dyn_cast<subop::ListType>(t)) {
      mlir::Type nt = replaceEmbeddedHivInType(ctx, list.getT(), producerHiv, consumerHivBeforeAlign);
      if (nt == list.getT()) return t;
      return subop::ListType::get(ctx, mlir::cast<subop::StateEntryReference>(nt));
   }
   return t;
}

struct ProbeAlignDebugCtx;

struct ConsumerCachedHivSites {
   llvm::DenseSet<void*> ssaClosure;
   subop::HashIndexedViewType consumerHivBeforeAlign = nullptr;
   /// \c lookup_u_* list ref scopes for \c LookupOp nodes that consume this \c cache_get HIV.
   llvm::StringSet<> probeLookupScopes;
   /// \c scan_list ops reached while walking HIV/list carriers from the \c cache_get root.
   llvm::SmallVector<subop::ScanListOp, 16> scanListsFromTraverse;
};

static void traverseConsumerHivUsesFromRoot(mlir::Value root, subop::HashIndexedViewType consumerHiv,
                                            subop::HashIndexedViewType consumerHivBeforeAlign,
                                            const CachedJoinBufferLayout& consumerLayout,
                                            ConsumerCachedHivSites& sites, const ProbeAlignDebugCtx* dbg);

static void addToSsaClosure(mlir::Value v, ConsumerCachedHivSites& sites) {
   if (!v) return;
   sites.ssaClosure.insert(v.getAsOpaquePointer());
}

static bool isJoinProbeCompilerScope(llvm::StringRef scope) { return scope.starts_with("lookup_u_"); }

static bool reuseProbeAlignDebugEnabled() {
   static int cached = -1;
   if (cached < 0) cached = std::getenv("LINGODB_REUSE_PROBE_ALIGN_DEBUG") ? 1 : 0;
   return cached != 0;
}

struct ProbeAlignDebugCtx {
   std::optional<uint64_t> cacheKey;
   llvm::StringRef passName;
};

static void debugProbeAlign(const ProbeAlignDebugCtx* dbg, llvm::function_ref<void(llvm::raw_ostream&)> fn) {
   if (!reuseProbeAlignDebugEnabled()) return;
   llvm::errs() << "[reuse-probe-align]";
   if (dbg) {
      if (dbg->cacheKey) llvm::errs() << " cache_key=" << *dbg->cacheKey;
      if (!dbg->passName.empty()) llvm::errs() << " pass=" << dbg->passName;
   }
   llvm::errs() << ' ';
   fn(llvm::errs());
   llvm::errs() << '\n';
}

static void remapAlignedHivClosureGathers(mlir::ModuleOp module, const llvm::DenseSet<void*>* ssaClosure,
                                          subop::HashIndexedViewType alignedHiv,
                                          const CachedJoinBufferLayout& layout,
                                          const llvm::StringSet<>* probeLookupScopes,
                                          std::optional<uint64_t> cacheKey,
                                          llvm::StringRef passName) {
   std::optional<ProbeAlignDebugCtx> dbg;
   if (!passName.empty()) {
      dbg.emplace();
      dbg->cacheKey = cacheKey;
      dbg->passName = passName;
   }
   remapClosureGathersToAlignedConsumerHiv(module, ssaClosure, alignedHiv, layout, probeLookupScopes,
                                           dbg ? &*dbg : nullptr);
}

static std::string mlirTypeToString(mlir::Type t) {
   if (!t) return "<null>";
   std::string s;
   llvm::raw_string_ostream os(s);
   t.print(os);
   return os.str();
}

static void setValueCarrierType(mlir::Value v, subop::HashIndexedViewType producerHiv,
                                subop::HashIndexedViewType consumerHivBeforeAlign,
                                const ProbeAlignDebugCtx* dbg = nullptr) {
   mlir::MLIRContext* ctx = v.getContext();
   mlir::Type oldTy = v.getType();
   if (mlir::Type nt = replaceEmbeddedHivInType(ctx, oldTy, producerHiv, consumerHivBeforeAlign); nt != oldTy) {
      debugProbeAlign(dbg, [&](llvm::raw_ostream& os) {
         os << "setValueCarrierType value=" << v << " old_type=" << mlirTypeToString(oldTy)
            << " new_type=" << mlirTypeToString(nt) << " aligned_hiv=" << mlirTypeToString(producerHiv);
      });
      v.setType(nt);
   }
}

/// \c scan_list elem and gather \c ref must share the same compiler \c lookup_u_* scope name.
static bool lookupProbeRefScopesMatch(llvm::StringRef scanEntryScope, llvm::StringRef gatherRefScope) {
   if (!isJoinProbeCompilerScope(scanEntryScope) || !isJoinProbeCompilerScope(gatherRefScope)) return false;
   return scanEntryScope == gatherRefScope;
}

static void printSsaValueOrigin(llvm::raw_ostream& os, mlir::Value v) {
   os << v;
   if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(v)) {
      os << " (block_arg#" << ba.getArgNumber() << " in ";
      if (mlir::Operation* parent = ba.getOwner()->getParentOp()) os << parent->getName();
      else os << "block";
      os << ')';
      return;
   }
   if (mlir::Operation* def = v.getDefiningOp()) {
      os << " (def=" << def->getName() << " op=" << def << ')';
   }
}

static void debugScanListCarrierMismatch(subop::ScanListOp scan, const ProbeAlignDebugCtx* dbg) {
   auto listTy = mlir::dyn_cast<subop::ListType>(scan.getList().getType());
   if (!listTy) return;
   auto elem = scan.getElem();
   if (elem.getColumn().type == listTy.getT()) return;
   auto* ctx = scan.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto [elemScope, elemLeaf] = cm.getName(&elem.getColumn());
   debugProbeAlign(dbg, [&](llvm::raw_ostream& os) {
      os << "INCONSISTENT scan_list op=" << scan.getOperation();
      os << " elem_scope=" << elemScope << " leaf=" << elemLeaf;
      os << " list_operand=";
      printSsaValueOrigin(os, scan.getList());
      os << " list_elem_type=" << mlirTypeToString(listTy.getT());
      os << " elem_attr_type=" << mlirTypeToString(elem.getColumn().type);
   });
}

static void debugScanListSkippedFromCacheGetProbe(subop::ScanListOp scan, subop::HashIndexedViewType alignedHiv,
                                                  const ConsumerCacheGetProbeClosure& probe, llvm::StringRef reason,
                                                  const ProbeAlignDebugCtx* dbg) {
   auto* ctx = scan.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto [elemScope, elemLeaf] = cm.getName(&scan.getElem().getColumn());
   const bool inClosure = opaqueClosureContains(probe.ssaClosure, scan.getList());
   const bool seenOnTraverse = llvm::is_contained(probe.scanListsFromTraverse, scan);
   debugProbeAlign(dbg, [&](llvm::raw_ostream& os) {
      os << "SKIP scan_list align reason=" << reason << " op=" << scan.getOperation();
      os << " elem_scope=" << elemScope << " leaf=" << elemLeaf;
      os << " list_operand=";
      printSsaValueOrigin(os, scan.getList());
      os << " list_in_closure=" << (inClosure ? "yes" : "no");
      os << " seen_on_cache_get_traverse=" << (seenOnTraverse ? "yes" : "no");
      os << " list_type=" << mlirTypeToString(scan.getList().getType());
      os << " aligned_hiv=" << mlirTypeToString(alignedHiv);
      if (!probe.probeLookupScopes.empty()) {
         os << " cache_get_lookup_scopes={";
         bool first = true;
         for (const auto& s : probe.probeLookupScopes) {
            if (!first) os << ',';
            os << s.first();
            first = false;
         }
         os << '}';
      }
   });
}

static void debugScanListsMissedByTraverse(mlir::ModuleOp module, const ConsumerCacheGetProbeClosure& probe,
                                           const ProbeAlignDebugCtx* dbg) {
   llvm::DenseSet<subop::ScanListOp> aligned;
   for (subop::ScanListOp scan : probe.scanListsFromTraverse) aligned.insert(scan);
   module.walk([&](subop::ScanListOp scan) {
      if (!opaqueClosureContains(probe.ssaClosure, scan.getList())) return;
      if (!typeEmbedsHashIndexedViewState(scan.getList().getType(), probe.alignedHiv)) return;
      if (aligned.contains(scan)) return;
      debugScanListSkippedFromCacheGetProbe(scan, probe.alignedHiv, probe, "in_closure_but_not_reached_by_traverse",
                                            dbg);
   });
}

static CachedJoinBufferLayout layoutFromUnionPlan(subop::HashIndexedViewType producerHiv,
                                                  const JoinBufferUnionPlan& plan) {
   CachedJoinBufferLayout out;
   out.producerHiv = producerHiv;
   out.payloadMembers.assign(plan.payloadMembers.begin(), plan.payloadMembers.end());
   out.payloadColumnTypes.assign(plan.payloadMemberTypes.begin(), plan.payloadMemberTypes.end());
   out.payloadSemanticKeys.reserve(plan.payloadColumns.size());
   for (size_t unionIdx = 0; unionIdx < plan.payloadColumns.size(); ++unionIdx) {
      const PayloadColumnSpec& spec = plan.payloadColumns[unionIdx];
      out.payloadSemanticKeys.push_back(spec.semanticKey);
      if (spec.inQuery0) {
         out.query0SemanticKeys.push_back(spec.semanticKey);
         out.query0SlotInUnion.push_back(static_cast<unsigned>(unionIdx));
      }
      if (spec.inQuery1) {
         out.query1SemanticKeys.push_back(spec.semanticKey);
         out.query1SlotInUnion.push_back(static_cast<unsigned>(unionIdx));
      }
   }
   return out;
}

static unsigned operandIndexOf(mlir::Operation* op, mlir::Value v) {
   for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      if (op->getOperand(i) == v) return i;
   }
   llvm_unreachable("operand not found");
}

static subop::Member cloneMemberToContext(subop::Member srcMember, mlir::MLIRContext* srcCtx, mlir::MLIRContext* dstCtx,
                                         bool allowMemberTypeUpdate) {
   if (srcCtx == dstCtx) return srcMember;
   auto& srcMm = srcCtx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto& dstMm = dstCtx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   return dstMm.getOrCreateMemberDirect(srcMm.getName(srcMember), cloneTypeToContext(srcMm.getType(srcMember), dstCtx),
                                        allowMemberTypeUpdate);
}

static llvm::StringMap<subop::Member> semKeyToMemberMap(const CachedJoinBufferLayout& layout) {
   llvm::StringMap<subop::Member> semKeyToMember;
   assert(layout.payloadSemanticKeys.size() == layout.payloadMembers.size());
   for (size_t i = 0; i < layout.payloadSemanticKeys.size(); ++i) {
      semKeyToMember[layout.payloadSemanticKeys[i]] = layout.payloadMembers[i];
   }
   return semKeyToMember;
}

static subop::Member cloneMemberToContext(subop::Member srcMember, mlir::MLIRContext* srcCtx,
                                          mlir::MLIRContext* dstCtx, bool allowMemberTypeUpdate);

static std::string aggregateColumnSemanticKey(tuples::ColumnRefAttr col,
                                              tuples::ColumnManager& cm) {
   auto [scope, leaf] = cm.getName(&col.getColumn());
   return columnSemanticKey(aggregatePayloadScopeSemantic(scope), leaf);
}

static mlir::Value stripCastLikeForAggregateSemantic(mlir::Value v) {
   for (;;) {
      mlir::Operation* def = v.getDefiningOp();
      if (!def) return v;
      llvm::StringRef name = def->getName().getStringRef();
      if ((name == "db.cast" || name == "arith.extsi" || name == "arith.extui") &&
          def->getNumOperands() == 1 && def->getNumResults() == 1) {
         v = def->getOperand(0);
         continue;
      }
      return v;
   }
}

static bool aggregateExprContainsValue(mlir::Value root, mlir::Value needle, llvm::DenseSet<void*>& seen) {
   root = stripCastLikeForAggregateSemantic(root);
   if (root == needle) return true;
   if (!seen.insert(root.getAsOpaquePointer()).second) return false;
   mlir::Operation* def = root.getDefiningOp();
   if (!def) return false;
   for (mlir::Value operand : def->getOperands()) {
      if (aggregateExprContainsValue(operand, needle, seen)) return true;
   }
   return false;
}

static bool aggregateExprContainsValue(mlir::Value root, mlir::Value needle) {
   llvm::DenseSet<void*> seen;
   return aggregateExprContainsValue(root, needle, seen);
}

static std::optional<unsigned> findAggregateInputArgIndex(mlir::Value root, mlir::Block& block, unsigned numCols,
                                                         llvm::DenseSet<void*>& seen) {
   root = stripCastLikeForAggregateSemantic(root);
   if (!seen.insert(root.getAsOpaquePointer()).second) return std::nullopt;
   if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(root)) {
      if (barg.getOwner() == &block && barg.getArgNumber() < numCols) return barg.getArgNumber();
      return std::nullopt;
   }
   mlir::Operation* def = root.getDefiningOp();
   if (!def) return std::nullopt;
   for (mlir::Value operand : def->getOperands()) {
      if (auto idx = findAggregateInputArgIndex(operand, block, numCols, seen)) return idx;
   }
   return std::nullopt;
}

static std::optional<unsigned> findAggregateInputArgIndex(mlir::Value root, mlir::Block& block, unsigned numCols) {
   llvm::DenseSet<void*> seen;
   return findAggregateInputArgIndex(root, block, numCols, seen);
}

static std::string aggregatePayloadSemanticKeyForReturn(subop::ReduceOp reduce, unsigned memberIdx,
                                                        tuples::ColumnManager& cm) {
   assert(!reduce.getRegion().empty() && "aggregate reduce must have update region");
   mlir::Block& block = reduce.getRegion().front();
   auto ret = mlir::cast<tuples::ReturnOp>(block.getTerminator());
   assert(memberIdx < ret.getNumOperands() && "reduce return must align with member list");
   const unsigned numCols = reduce.getColumns().size();
   mlir::Value current = block.getArgument(numCols + memberIdx);
   mlir::Value returned = stripCastLikeForAggregateSemantic(ret.getOperand(memberIdx));
   if (returned == current) return "identity";

   if (auto* def = returned.getDefiningOp()) {
      if (def->getName().getStringRef() == "db.add" && def->getNumOperands() == 2) {
         mlir::Value lhs = stripCastLikeForAggregateSemantic(def->getOperand(0));
         mlir::Value rhs = stripCastLikeForAggregateSemantic(def->getOperand(1));
         mlir::Value payload = {};
         if (lhs == current) payload = rhs;
         if (rhs == current) payload = lhs;
         if (payload) {
            if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(payload)) {
               if (barg.getOwner() == &block && barg.getArgNumber() < numCols) {
                  auto col = mlir::cast<tuples::ColumnRefAttr>(reduce.getColumns()[barg.getArgNumber()]);
                  return "sum:" + aggregateColumnSemanticKey(col, cm);
               }
            }
            if (payload.getDefiningOp() && payload.getDefiningOp()->getName().getStringRef() == "db.constant") {
               return "count:*";
            }
         }
      }
   }
   if (auto inputIdx = findAggregateInputArgIndex(returned, block, numCols)) {
      auto col = mlir::cast<tuples::ColumnRefAttr>(reduce.getColumns()[*inputIdx]);
      if (aggregateExprContainsValue(returned, current)) return "sum:" + aggregateColumnSemanticKey(col, cm);
      return aggregateColumnSemanticKey(col, cm);
   }
   llvm_unreachable("aggregate union: unsupported reduce payload update expression");
}

static subop::PreAggrHtFragmentType fragmentTypeForAggregateHt(subop::PreAggrHtType ht) {
   return subop::PreAggrHtFragmentType::get(ht.getContext(), ht.getKeyMembers(), ht.getValueMembers(),
                                            ht.getWithLock());
}

static subop::PreAggrHtType aggregateHtTypeFromState(mlir::Value state) {
   if (auto ht = mlir::dyn_cast<subop::PreAggrHtType>(state.getType())) return ht;
   if (auto tl = mlir::dyn_cast<subop::ThreadLocalType>(state.getType())) {
      if (auto frag = mlir::dyn_cast<subop::PreAggrHtFragmentType>(tl.getWrapped())) {
         return subop::PreAggrHtType::get(state.getContext(), frag.getKeyMembers(), frag.getValueMembers(),
                                          frag.getWithLock());
      }
   }
   if (auto frag = mlir::dyn_cast<subop::PreAggrHtFragmentType>(state.getType())) {
      return subop::PreAggrHtType::get(state.getContext(), frag.getKeyMembers(), frag.getValueMembers(),
                                       frag.getWithLock());
   }
   return nullptr;
}

struct AggregatePayloadMemberInfo {
   std::string semanticKey;
   uint64_t semanticHash = 0;
   subop::Member member;
   mlir::Type type;
   tuples::ColumnRefAttr sourceColumn;
};

static uint64_t aggregatePayloadHashForInfo(llvm::StringRef semanticKey,
                                            tuples::ColumnRefAttr sourceColumn,
                                            mlir::Type payloadType,
                                            const llvm::DenseMap<const void*, uint64_t>& columnHashes) {
   if (semanticKey == "count:*") return payloadSyntheticColumnHash("aggregate", "count", payloadType);
   if (semanticKey == "identity") return payloadSyntheticColumnHash("aggregate", "identity", payloadType);

   llvm::StringRef kind = "payload";
   if (semanticKey.consume_front("sum:")) kind = "sum";
   assert(sourceColumn && "aggregate payload must have a source column identity hash");
   uint64_t h = hashPayloadString("aggregate_payload");
   h = combinePayloadHash(h, hashPayloadString(kind));
   return combinePayloadHash(h, payloadColumnIdentityHash(sourceColumn, columnHashes));
}

static tuples::ColumnRefAttr resolveAggregatePayloadSourceColumn(subop::ReduceOp reduceOp, unsigned payloadIdx) {
   mlir::Block& block = reduceOp.getRegion().front();
   auto ret = mlir::cast<tuples::ReturnOp>(block.getTerminator());
   const unsigned numCols = reduceOp.getColumns().size();
   mlir::Value current = block.getArgument(numCols + payloadIdx);

   mlir::Value returned = stripCastLikeForAggregateSemantic(ret.getOperand(payloadIdx));
   if (auto inputIdx = findAggregateInputArgIndex(returned, block, numCols))
      return mlir::cast<tuples::ColumnRefAttr>(reduceOp.getColumns()[*inputIdx]);

   auto* def = returned.getDefiningOp();
   if (!def || def->getName().getStringRef() != "db.add")
      abortAggregateUnionUnsupported("aggregate payload is neither passthrough nor additive");
   mlir::Value lhs = stripCastLikeForAggregateSemantic(def->getOperand(0));
   mlir::Value rhs = stripCastLikeForAggregateSemantic(def->getOperand(1));
   mlir::Value payload = lhs == current ? rhs : lhs;
   if (auto inputIdx = findAggregateInputArgIndex(payload, block, numCols))
      return mlir::cast<tuples::ColumnRefAttr>(reduceOp.getColumns()[*inputIdx]);
   abortAggregateUnionUnsupported("aggregate additive payload source is not an input column");
}

static llvm::SmallVector<AggregatePayloadMemberInfo, 16>
collectAggregatePayloadMembers(mlir::ModuleOp module, mlir::Value aggregateState) {
   auto* ctx = module.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   llvm::DenseMap<const void*, uint64_t> columnHashes =
      collectStateConstructionColumnHashes(module, aggregateState);
   subop::PreAggrHtType ht = aggregateHtTypeFromState(aggregateState);
   assert(ht && "aggregate payload collection requires optimistic_ht-like state");
   subop::PreAggrHtFragmentType fragTy = fragmentTypeForAggregateHt(ht);

   llvm::SmallVector<subop::ReduceOp, 4> reduceOps;
   module.walk([&](subop::ReduceOp reduce) {
      auto refTy = mlir::dyn_cast<subop::LookupEntryRefType>(reduce.getRef().getColumn().type);
      if (!refTy || refTy.getState() != fragTy) return mlir::WalkResult::advance();
      reduceOps.push_back(reduce);
      return mlir::WalkResult::advance();
   });
   module.walk([&](subop::ReduceOp reduce) {
      auto refTy = mlir::dyn_cast<subop::LookupEntryRefType>(reduce.getRef().getColumn().type);
      if (!refTy || refTy.getState() != ht) return mlir::WalkResult::advance();
      reduceOps.push_back(reduce);
      return mlir::WalkResult::advance();
   });
   if (reduceOps.empty()) llvm_unreachable("aggregate union: expected reduce op for aggregate hash table");

   llvm::DenseMap<subop::Member, AggregatePayloadMemberInfo> byMember;
   auto recordMember = [&](subop::Member member, std::string semanticKey, tuples::ColumnRefAttr sourceColumn) {
      mlir::Type memberTy = mm.getType(member);
      uint64_t semanticHash = aggregatePayloadHashForInfo(semanticKey, sourceColumn, memberTy, columnHashes);
      AggregatePayloadMemberInfo info{std::move(semanticKey), semanticHash, member, memberTy, sourceColumn};
      auto it = byMember.find(member);
      if (it == byMember.end() || (it->second.semanticKey == "identity" && info.semanticKey != "identity")) {
         byMember[member] = std::move(info);
      }
   };
   for (subop::ReduceOp reduceOp : reduceOps) {
      for (unsigned i = 0; i < reduceOp.getMembers().size(); ++i) {
         auto member = mlir::cast<subop::MemberAttr>(reduceOp.getMembers()[i]).getMember();
         tuples::ColumnRefAttr sourceColumn;
         std::string semanticKey = aggregatePayloadSemanticKeyForReturn(reduceOp, i, cm);
         if (semanticKey != "count:*" && semanticKey != "identity") {
            sourceColumn = resolveAggregatePayloadSourceColumn(reduceOp, i);
         }
         recordMember(member, std::move(semanticKey), sourceColumn);
      }
   }
   llvm::SmallVector<AggregatePayloadMemberInfo, 16> out;
   for (subop::Member member : ht.getValueMembers().getMembers()) {
      auto it = byMember.find(member);
      if (it != byMember.end()) {
         out.push_back(std::move(it->second));
      } else {
         mlir::Type memberTy = mm.getType(member);
         out.push_back({"identity", aggregatePayloadHashForInfo("identity", {}, memberTy, columnHashes),
                        member, memberTy, {}});
      }
   }
   return out;
}

static CachedAggregateLayout buildAggregateUnionLayout(mlir::Value producerState, mlir::Value peerState,
                                                       mlir::ModuleOp producerModule, mlir::ModuleOp peerModule) {
   CachedAggregateLayout layout;
   subop::PreAggrHtType producerHt = mlir::cast<subop::PreAggrHtType>(producerState.getType());
   layout.producerHt = producerHt;

   llvm::SmallVector<AggregatePayloadMemberInfo, 16> producer = collectAggregatePayloadMembers(producerModule, producerState);
   llvm::SmallVector<AggregatePayloadMemberInfo, 16> peer = collectAggregatePayloadMembers(peerModule, peerState);
   for (const auto& p : producer) {
      layout.payloadSemanticKeys.push_back(p.semanticKey);
      layout.payloadMembers.push_back(p.member);
      layout.payloadColumnTypes.push_back(p.type);
      layout.query0SemanticKeys.push_back(p.semanticKey);
      layout.query0Members.push_back(p.member);
   }
   for (const auto& p : peer) {
      layout.query1SemanticKeys.push_back(p.semanticKey);
      layout.query1Members.push_back(p.member);
   }
   return layout;
}

static subop::ExecutionStepOp findAggregateBuildStepForHt(mlir::ModuleOp module, mlir::Value aggregateState) {
   subop::PreAggrHtType ht = mlir::cast<subop::PreAggrHtType>(aggregateState.getType());
   subop::PreAggrHtFragmentType fragTy = fragmentTypeForAggregateHt(ht);
   subop::ExecutionStepOp found;
   module.walk([&](subop::ReduceOp reduce) {
      auto refTy = mlir::dyn_cast<subop::LookupEntryRefType>(reduce.getRef().getColumn().type);
      if (!refTy || refTy.getState() != fragTy) return mlir::WalkResult::advance();
      found = reduce->getParentOfType<subop::ExecutionStepOp>();
      return mlir::WalkResult::interrupt();
   });
   assert(found && "aggregate union: expected build step for aggregate hash table");
   return found;
}

static subop::ReduceOp findAggregateReduceInStep(subop::ExecutionStepOp step) {
   subop::ReduceOp found;
   step.walk([&](subop::ReduceOp reduce) {
      assert(!found && "aggregate union: expected a single reduce in aggregate build step");
      found = reduce;
   });
   assert(found && "aggregate union: aggregate build step must contain reduce");
   return found;
}

static subop::LookupOrInsertOp findAggregateLookupInStep(subop::ExecutionStepOp step) {
   subop::LookupOrInsertOp found;
   step.walk([&](subop::LookupOrInsertOp lookup) {
      assert(!found && "aggregate union: expected a single lookup_or_insert in aggregate build step");
      found = lookup;
   });
   assert(found && "aggregate union: aggregate build step must contain lookup_or_insert");
   return found;
}

static subop::ScanRefsOp findAggregateTableScanInStep(subop::ExecutionStepOp step) {
   subop::ScanRefsOp found;
   step.walk([&](subop::ScanRefsOp scan) {
      if (!mlir::isa<subop::TableType>(scan.getState().getType())) return;
      assert(!found && "aggregate union: expected a single table scan in aggregate build step");
      found = scan;
   });
   assert(found && "aggregate union: aggregate build step must scan a table");
   return found;
}

static std::string columnSemanticKey(tuples::ColumnRefAttr col, tuples::ColumnManager& cm) {
   auto [scope, leaf] = cm.getName(&col.getColumn());
   return columnSemanticKey(scope, leaf);
}

static std::string columnSemanticKey(tuples::ColumnDefAttr col, tuples::ColumnManager& cm) {
   auto [scope, leaf] = cm.getName(&col.getColumn());
   return columnSemanticKey(scope, leaf);
}

static subop::MapOp findMapProducingColumn(subop::ExecutionStepOp step, tuples::ColumnRefAttr col,
                                           tuples::ColumnManager& cm) {
   std::string want = columnSemanticKey(col, cm);
   subop::MapOp found;
   step.walk([&](subop::MapOp map) {
      for (auto attr : map.getComputedCols()) {
         if (columnSemanticKey(mlir::cast<tuples::ColumnDefAttr>(attr), cm) == want) {
            found = map;
            return mlir::WalkResult::interrupt();
         }
      }
      return mlir::WalkResult::advance();
   });
   return found;
}

static subop::MapOp findProducerAggregateMap(subop::ExecutionStepOp step) {
   subop::MapOp found;
   step.walk([&](subop::MapOp map) {
      for (auto attr : map.getComputedCols()) {
         auto def = mlir::cast<tuples::ColumnDefAttr>(attr);
         auto& cm = step.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
         auto [scope, leaf] = cm.getName(&def.getColumn());
         (void)leaf;
         if (scope == "aggMap") {
            found = map;
            return mlir::WalkResult::interrupt();
         }
      }
      return mlir::WalkResult::advance();
   });
   assert(found && "aggregate union: expected producer aggregate map");
   return found;
}

static bool mapHasInputSemantic(subop::MapOp map, const std::string& semantic, tuples::ColumnManager& cm) {
   for (auto attr : map.getInputCols()) {
      if (columnSemanticKey(mlir::cast<tuples::ColumnRefAttr>(attr), cm) == semantic) return true;
   }
   return false;
}

static mlir::BlockArgument mapBlockArgForInputSemantic(subop::MapOp map, const std::string& semantic,
                                                       tuples::ColumnManager& cm) {
   mlir::Block& block = map.getFn().front();
   for (unsigned i = 0; i < map.getInputCols().size(); ++i) {
      if (columnSemanticKey(mlir::cast<tuples::ColumnRefAttr>(map.getInputCols()[i]), cm) == semantic)
         return block.getArgument(i);
   }
   llvm_unreachable("aggregate union: missing map input");
}

static mlir::Value mapReturnValueForComputedSemantic(subop::MapOp map, const std::string& semantic,
                                                     tuples::ColumnManager& cm) {
   mlir::Block& block = map.getFn().front();
   auto ret = mlir::cast<tuples::ReturnOp>(block.getTerminator());
   for (unsigned i = 0; i < map.getComputedCols().size(); ++i) {
      if (columnSemanticKey(mlir::cast<tuples::ColumnDefAttr>(map.getComputedCols()[i]), cm) == semantic)
         return ret.getOperand(i);
   }
   llvm_unreachable("aggregate union: missing map computed column");
}

static tuples::ColumnRefAttr mapComputedRefForSemantic(subop::MapOp map, const std::string& semantic,
                                                       tuples::ColumnManager& cm) {
   for (auto attr : map.getComputedCols()) {
      auto def = mlir::cast<tuples::ColumnDefAttr>(attr);
      if (columnSemanticKey(def, cm) == semantic) return cm.createRef(&def.getColumn());
   }
   llvm_unreachable("aggregate union: missing map computed column");
}

static void widenAggregateExternalTableForPeer(mlir::ModuleOp synthetic, subop::ExecutionStepOp buildStep,
                                               subop::ScanRefsOp scanOp, subop::ExecutionStepOp peerBuild,
                                               const ModuleReuseInfo& reuseSynthetic,
                                               const ModuleReuseInfo& reusePeer) {
   auto* ctx = synthetic.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   mlir::Value tableState = scanOp.getState();
   ExternalDatasourceProperty mergedDs;
   bool haveDs = false;
   llvm::StringRef donorTableName;
   subop::TableType donorTableTy;
   bool ok = resolveScannedTableExternal(buildStep, tableState, reuseSynthetic, donorTableName,
                                         mergedDs, haveDs, donorTableTy);
   (void)ok;
   assert(ok && haveDs && donorTableTy && "aggregate union: expected external donor table");
   mergePeerExternalFromBuildStepScan(mergedDs, haveDs, peerBuild, reusePeer, donorTableName);
   auto lookupPeerColumnType = [&](llvm::StringRef identifier) -> mlir::Type {
      return columnTypeForIdentifierFromPeerBuildScan(peerBuild, reusePeer, donorTableName, identifier);
   };
   auto newTableTy = tableTypeFromMergedExternal(ctx, mm, mergedDs, donorTableTy, lookupPeerColumnType);
   for (auto& map : mergedDs.mapping) {
      if (subop::Member m = tableMemberForIdentifier(newTableTy, mm, map.identifier)) map.memberName = mm.getName(m);
   }
   llvm::sort(mergedDs.mapping, [](const auto& x, const auto& y) { return x.memberName < y.memberName; });
   std::string hex = lingodb::utility::serializeToHexString(mergedDs);
   subop::ExecutionGroupOp eg = buildStep->getParentOfType<subop::ExecutionGroupOp>();
   assert(eg && "aggregate union: build step must live in execution_group");
   mlir::OpBuilder gb = mlir::OpBuilder::atBlockBegin(&eg.getSubOps().front());
   gb.setInsertionPoint(buildStep);
   subop::ExecutionStepOp mergedTableRefStep =
      createMergedExternalTableRefStep(gb, buildStep.getLoc(), newTableTy, hex);
   rewireBuildStepScannedTable(buildStep, scanOp, tableState, mergedTableRefStep.getResult(0), newTableTy);
   refreshTableStateTypesInModule(synthetic, mergedTableRefStep.getResult(0), newTableTy);
}

static void ensureMapInputGatheredFromTable(subop::ExecutionStepOp buildStep, subop::ScanRefsOp scanOp,
                                            subop::MapOp map, tuples::ColumnRefAttr inputCol,
                                            tuples::ColumnManager& cm) {
   auto* ctx = buildStep.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   std::string semantic = columnSemanticKey(inputCol, cm);
   if (mapHasInputSemantic(map, semantic, cm)) return;

   auto [scope, leaf] = cm.getName(&inputCol.getColumn());
   auto tableTy = mlir::cast<subop::TableType>(scanOp.getState().getType());
   subop::Member tableMember = tableMemberForIdentifier(tableTy, mm, leaf);
   assert(tableMember && "aggregate union: widened external table must contain map input column");
   tuples::ColumnDefAttr def = cm.createDef(scope, leaf);
   def.getColumn().type = cloneTypeToContext(inputCol.getColumn().type, ctx);

   mlir::Value stream = map.getStream();
   mlir::Operation* anchor = stream.getDefiningOp();
   assert(anchor && "aggregate union: map stream must be defined by preceding subop");
   mlir::OpBuilder b(anchor);
   b.setInsertionPointAfter(anchor);
   auto mapping = subop::ColumnDefMemberMappingAttr::get(
      ctx, llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>>{{tableMember, def}});
   auto gatherRef = cm.createRef(&scanOp.getRef().getColumn());
   auto gather = b.create<subop::GatherOp>(map.getLoc(), stream.getType(), stream, gatherRef, mapping);
   map.getStreamMutable().assign(gather.getRes());

   llvm::SmallVector<mlir::Attribute> inputs(map.getInputCols().begin(), map.getInputCols().end());
   inputs.push_back(cm.createRef(&def.getColumn()));
   map.setInputColsAttr(mlir::ArrayAttr::get(ctx, inputs));
   map.getFn().front().addArgument(def.getColumn().type, mlir::UnknownLoc::get(ctx));
}

static tuples::ColumnRefAttr cloneColumnRefToContext(tuples::ColumnRefAttr col, mlir::MLIRContext* ctx,
                                                     mlir::Type overrideType = {}) {
   auto& srcCm = col.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& dstCm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto [scope, leaf] = srcCm.getName(&col.getColumn());
   auto out = dstCm.createRef(scope, leaf);
   out.getColumn().type = overrideType ? overrideType : cloneTypeToContext(col.getColumn().type, ctx);
   return out;
}

static mlir::Attribute cloneAggregateMapAttrToContext(mlir::Attribute attr, mlir::MLIRContext* ctx) {
   if (!attr || attr.getContext() == ctx) return attr;
   mlir::Builder b(ctx);
   if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(attr))
      return b.getIntegerAttr(cloneTypeToContext(i.getType(), ctx), i.getValue());
   if (auto f = mlir::dyn_cast<mlir::FloatAttr>(attr))
      return b.getFloatAttr(cloneTypeToContext(f.getType(), ctx), f.getValue());
   if (auto s = mlir::dyn_cast<mlir::StringAttr>(attr)) return b.getStringAttr(s.getValue());
   llvm_unreachable("aggregate union: unsupported cloned map attribute");
}

static mlir::Value cloneAggregateMapExprToProducer(mlir::Value v, mlir::IRMapping& mapping,
                                                   mlir::OpBuilder& b, mlir::MLIRContext* ctx) {
   if (mapping.contains(v)) return mapping.lookup(v);
   mlir::Operation* op = v.getDefiningOp();
   assert(op && "aggregate union: unmapped peer map block argument");
   mlir::Location loc = mlir::UnknownLoc::get(ctx);
   auto finish = [&](mlir::Value cloned) -> mlir::Value {
      assert(cloned && "aggregate union: expression clone must produce a value");
      cloned.setType(cloneTypeToContext(v.getType(), ctx));
      if (mlir::Operation* def = cloned.getDefiningOp()) def->setLoc(loc);
      mapping.map(v, cloned);
      return cloned;
   };
   mlir::Value out;
   if (auto c = mlir::dyn_cast<db::ConstantOp>(op)) {
      out = b.create<db::ConstantOp>(loc, cloneTypeToContext(c.getType(), ctx),
                                     cloneAggregateMapAttrToContext(c.getValue(), ctx));
   } else if (auto cast = mlir::dyn_cast<db::CastOp>(op)) {
      out = b.create<db::CastOp>(loc, cloneTypeToContext(cast.getType(), ctx),
                                 cloneAggregateMapExprToProducer(cast.getVal(), mapping, b, ctx));
   } else if (auto add = mlir::dyn_cast<db::AddOp>(op)) {
      out = b.create<db::AddOp>(loc, cloneAggregateMapExprToProducer(add.getLeft(), mapping, b, ctx),
                                cloneAggregateMapExprToProducer(add.getRight(), mapping, b, ctx));
   } else if (auto sub = mlir::dyn_cast<db::SubOp>(op)) {
      out = b.create<db::SubOp>(loc, cloneAggregateMapExprToProducer(sub.getLeft(), mapping, b, ctx),
                                cloneAggregateMapExprToProducer(sub.getRight(), mapping, b, ctx));
   } else if (auto mul = mlir::dyn_cast<db::MulOp>(op)) {
      out = b.create<db::MulOp>(loc, cloneAggregateMapExprToProducer(mul.getLeft(), mapping, b, ctx),
                                cloneAggregateMapExprToProducer(mul.getRight(), mapping, b, ctx));
   } else if (auto div = mlir::dyn_cast<db::DivOp>(op)) {
      out = b.create<db::DivOp>(loc, cloneAggregateMapExprToProducer(div.getLeft(), mapping, b, ctx),
                                cloneAggregateMapExprToProducer(div.getRight(), mapping, b, ctx));
   } else {
      llvm_unreachable("aggregate union: unsupported peer map expression op");
   }
   return finish(out);
}

static tuples::ColumnRefAttr clonePeerMapResultIntoProducer(subop::ExecutionStepOp buildStep,
                                                            subop::MapOp producerMap,
                                                            subop::MapOp peerMap,
                                                            tuples::ColumnRefAttr peerResultCol,
                                                            const ModuleReuseInfo& reuseSynthetic,
                                                            const ModuleReuseInfo& reusePeer,
                                                            mlir::ModuleOp synthetic) {
   auto* ctx = synthetic.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   subop::ScanRefsOp scanOp = findAggregateTableScanInStep(buildStep);
   widenAggregateExternalTableForPeer(synthetic, buildStep, scanOp, peerMap->getParentOfType<subop::ExecutionStepOp>(),
                                      reuseSynthetic, reusePeer);
   scanOp = findAggregateTableScanInStep(buildStep);

   for (auto attr : peerMap.getInputCols()) {
      ensureMapInputGatheredFromTable(buildStep, scanOp, producerMap,
                                      cloneColumnRefToContext(mlir::cast<tuples::ColumnRefAttr>(attr), ctx),
                                      cm);
   }

   std::string missingSemantic = columnSemanticKey(cloneColumnRefToContext(peerResultCol, ctx), cm);
   for (auto attr : producerMap.getComputedCols()) {
      if (columnSemanticKey(mlir::cast<tuples::ColumnDefAttr>(attr), cm) == missingSemantic)
         return mapComputedRefForSemantic(producerMap, missingSemantic, cm);
   }

   mlir::IRMapping mapping;
   for (unsigned i = 0; i < peerMap.getInputCols().size(); ++i) {
      auto peerInput = mlir::cast<tuples::ColumnRefAttr>(peerMap.getInputCols()[i]);
      mapping.map(peerMap.getFn().front().getArgument(i),
                  mapBlockArgForInputSemantic(producerMap, columnSemanticKey(cloneColumnRefToContext(peerInput, ctx), cm), cm));
   }
   auto& peerCm = peerMap.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   for (unsigned i = 0; i < peerMap.getComputedCols().size(); ++i) {
      auto peerComputed = mlir::cast<tuples::ColumnDefAttr>(peerMap.getComputedCols()[i]);
      std::string semantic = columnSemanticKey(peerCm.createRef(&peerComputed.getColumn()), peerCm);
      for (auto prodAttr : producerMap.getComputedCols()) {
         if (columnSemanticKey(mlir::cast<tuples::ColumnDefAttr>(prodAttr), cm) == semantic) {
            mapping.map(peerMap.getFn().front().getTerminator()->getOperand(i),
                        mapReturnValueForComputedSemantic(producerMap, semantic, cm));
         }
      }
   }

   mlir::Block& peerBlock = peerMap.getFn().front();
   mlir::Block& prodBlock = producerMap.getFn().front();
   auto prodRet = mlir::cast<tuples::ReturnOp>(prodBlock.getTerminator());
   mlir::OpBuilder b(prodRet);
   auto peerRet = mlir::cast<tuples::ReturnOp>(peerBlock.getTerminator());
   unsigned peerResultIdx = 0;
   bool foundPeerResult = false;
   for (unsigned i = 0; i < peerMap.getComputedCols().size(); ++i) {
      auto def = mlir::cast<tuples::ColumnDefAttr>(peerMap.getComputedCols()[i]);
      if (columnSemanticKey(peerCm.createRef(&def.getColumn()), peerCm) == columnSemanticKey(peerResultCol, peerCm)) {
         peerResultIdx = i;
         foundPeerResult = true;
         break;
      }
   }
   assert(foundPeerResult && "aggregate union: peer map result not found");
   (void)foundPeerResult;
   mlir::Value newResult = cloneAggregateMapExprToProducer(peerRet.getOperand(peerResultIdx), mapping, b, ctx);
   llvm::SmallVector<mlir::Value> retVals(prodRet->getOperands().begin(), prodRet->getOperands().end());
   retVals.push_back(newResult);
   b.setInsertionPoint(prodRet);
   auto newRet = b.create<tuples::ReturnOp>(prodRet.getLoc(), retVals);
   prodRet.erase();

   llvm::SmallVector<mlir::Attribute> computed(producerMap.getComputedCols().begin(), producerMap.getComputedCols().end());
   auto [scope, leaf] = cm.getName(&cloneColumnRefToContext(peerResultCol, ctx).getColumn());
   auto def = cm.createDef(scope, leaf);
   def.getColumn().type = cloneTypeToContext(peerResultCol.getColumn().type, ctx);
   computed.push_back(def);
   producerMap.setComputedColsAttr(mlir::ArrayAttr::get(ctx, computed));
   (void)newRet;
   return cm.createRef(&def.getColumn());
}

static std::optional<unsigned> parseAggregateValueSlot(llvm::StringRef name) {
   if (!name.consume_front("aggrVal$")) return std::nullopt;
   unsigned slot = 0;
   if (name.getAsInteger(10, slot)) return std::nullopt;
   return slot;
}

static subop::Member allocUnusedAggregateValueSlot(subop::MemberManager& mm, mlir::Type colType, unsigned& nextSlot) {
   for (;; ++nextSlot) {
      std::string name = "aggrVal$" + std::to_string(nextSlot);
      if (!mm.hasMemberDirect(name)) return mm.createMemberDirect(name, colType);
   }
}

static mlir::Value createZeroForType(mlir::OpBuilder& b, mlir::Location loc, mlir::Type ty) {
   return b.create<db::ConstantOp>(loc, ty, b.getI64IntegerAttr(0));
}

static mlir::Value createOneForType(mlir::OpBuilder& b, mlir::Location loc, mlir::Type ty) {
   return b.create<db::ConstantOp>(loc, ty, b.getI64IntegerAttr(1));
}

static void insertReturnOperand(tuples::ReturnOp ret, unsigned idx, mlir::Value v) {
   mlir::OpBuilder b(ret);
   llvm::SmallVector<mlir::Value> operands(ret->getOperands().begin(), ret->getOperands().end());
   operands.insert(operands.begin() + idx, v);
   auto newRet = b.create<tuples::ReturnOp>(ret.getLoc(), operands);
   ret.erase();
   (void)newRet;
}

static void insertAggregateLookupInitial(subop::LookupOrInsertOp lookup, unsigned idx, mlir::Type ty) {
   mlir::Block& block = lookup.getInitFn().front();
   auto ret = mlir::cast<tuples::ReturnOp>(block.getTerminator());
   mlir::OpBuilder b(ret);
   insertReturnOperand(ret, idx, createZeroForType(b, lookup.getLoc(), ty));
}

static void insertAggregateReduceUpdate(subop::ReduceOp reduce, unsigned idx, tuples::ColumnRefAttr sourceCol,
                                        subop::Member member, mlir::Type ty) {
   auto* ctx = reduce.getContext();
   llvm::SmallVector<mlir::Attribute> cols(reduce.getColumns().begin(), reduce.getColumns().end());
   llvm::SmallVector<mlir::Attribute> members(reduce.getMembers().begin(), reduce.getMembers().end());
   const unsigned oldNumCols = cols.size();
   cols.push_back(sourceCol);
   members.insert(members.begin() + idx, subop::MemberAttr::get(ctx, member));
   reduce.setColumnsAttr(mlir::ArrayAttr::get(ctx, cols));
   reduce.setMembersAttr(mlir::ArrayAttr::get(ctx, members));

   mlir::Block& block = reduce.getRegion().front();
   mlir::Location loc = reduce.getLoc();
   mlir::BlockArgument input = block.insertArgument(oldNumCols, sourceCol.getColumn().type, loc);
   mlir::BlockArgument current = block.insertArgument(oldNumCols + 1 + idx, ty, loc);
   auto ret = mlir::cast<tuples::ReturnOp>(block.getTerminator());
   mlir::OpBuilder b(ret);
   mlir::Value updated;
   if (columnSemanticKey(sourceCol, ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager()) == "count:*") {
      updated = b.create<db::AddOp>(loc, current, createOneForType(b, loc, ty));
   } else {
      updated = b.create<db::AddOp>(loc, current, input);
   }
   insertReturnOperand(ret, idx, updated);
}

static void insertPairwiseAddRegionResult(mlir::Region& region, unsigned idx, mlir::Type ty, mlir::Location loc) {
   mlir::Block& block = region.front();
   auto ret = mlir::cast<tuples::ReturnOp>(block.getTerminator());
   unsigned oldNumValues = ret.getNumOperands();
   mlir::BlockArgument left = block.insertArgument(idx, ty, loc);
   mlir::BlockArgument right = block.insertArgument(oldNumValues + 1 + idx, ty, loc);
   mlir::OpBuilder b(ret);
   insertReturnOperand(ret, idx, b.create<db::AddOp>(loc, left, right));
}

static void insertAggregateReduceCombine(subop::ReduceOp reduce, unsigned idx, mlir::Type ty) {
   insertPairwiseAddRegionResult(reduce.getCombine(), idx, ty, reduce.getLoc());
}

static void insertAggregateMergeCombine(subop::MergeOp merge, unsigned idx, mlir::Type ty) {
   insertPairwiseAddRegionResult(merge.getCombineFn(), idx, ty, merge.getLoc());
}

static void updateSyntheticAggregateStateTypes(mlir::ModuleOp synthetic, subop::PreAggrHtFragmentType oldFrag,
                                               subop::PreAggrHtFragmentType newFrag, subop::PreAggrHtType oldHt,
                                               subop::PreAggrHtType newHt) {
   auto oldTl = subop::ThreadLocalType::get(synthetic.getContext(), oldFrag);
   auto newTl = subop::ThreadLocalType::get(synthetic.getContext(), newFrag);
   auto update = [&](mlir::Value v) {
      if (v.getType() == oldFrag) v.setType(newFrag);
      if (v.getType() == oldTl) v.setType(newTl);
      if (v.getType() == oldHt) v.setType(newHt);
   };
   synthetic.walk([&](mlir::Operation* op) {
      for (mlir::Value operand : op->getOperands()) update(operand);
      for (mlir::Value result : op->getResults()) update(result);
      for (mlir::Region& region : op->getRegions())
         for (mlir::Block& block : region)
            for (mlir::BlockArgument arg : block.getArguments()) update(arg);
   });
   synchronizeExecutionStepPortTypes(synthetic, nullptr);
}

static void syncAggregateEntryRefAttrs(mlir::ModuleOp module, subop::PreAggrHtFragmentType oldFrag,
                                       subop::PreAggrHtFragmentType newFrag, subop::PreAggrHtType oldHt,
                                       subop::PreAggrHtType newHt) {
   auto* ctx = module.getContext();
   auto newLookupRef = subop::LookupEntryRefType::get(ctx, newFrag);
   auto oldPreAggrRef = subop::PreAggrHTEntryRefType::get(ctx, oldHt);
   auto newPreAggrRef = subop::PreAggrHTEntryRefType::get(ctx, newHt);
   module.walk([&](subop::LookupOrInsertOp op) {
      auto ref = op.getRef();
      if (ref.getColumn().type == subop::LookupEntryRefType::get(ctx, oldFrag) ||
          ref.getColumn().type == newLookupRef) {
         ref.getColumn().type = newLookupRef;
         op.setRefAttr(ref);
      }
   });
   module.walk([&](subop::ReduceOp op) {
      auto ref = op.getRef();
      if (ref.getColumn().type == subop::LookupEntryRefType::get(ctx, oldFrag) ||
          ref.getColumn().type == newLookupRef) {
         ref.getColumn().type = newLookupRef;
         op.setRefAttr(ref);
      }
   });
   module.walk([&](subop::ScanRefsOp op) {
      auto ref = op.getRef();
      if (ref.getColumn().type == oldPreAggrRef || ref.getColumn().type == newPreAggrRef) {
         ref.getColumn().type = newPreAggrRef;
         op.setRefAttr(ref);
      }
   });
}

static void applySyntheticAggregatePayloadUnion(mlir::ModuleOp synthetic, mlir::Value syntheticHtState,
                                                CachedAggregateLayout& layout, mlir::ModuleOp query1,
                                                mlir::Value peerHtState,
                                                const ModuleReuseInfo& reuseSynthetic,
                                                const ModuleReuseInfo& reusePeer) {
   auto* ctx = synthetic.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   subop::ExecutionStepOp buildStep = findAggregateBuildStepForHt(synthetic, syntheticHtState);
   subop::ExecutionStepOp peerBuild = findAggregateBuildStepForHt(query1, peerHtState);
   subop::ReduceOp reduce = findAggregateReduceInStep(buildStep);
   subop::LookupOrInsertOp lookup = findAggregateLookupInStep(buildStep);
   subop::PreAggrHtType oldHt = mlir::cast<subop::PreAggrHtType>(syntheticHtState.getType());
   subop::PreAggrHtFragmentType oldFrag = fragmentTypeForAggregateHt(oldHt);

   llvm::SmallVector<AggregatePayloadMemberInfo, 16> producerInfos =
      collectAggregatePayloadMembers(synthetic, syntheticHtState);
   llvm::DenseMap<uint64_t, AggregatePayloadMemberInfo> producerByHash;
   unsigned nextSlot = 0;
   for (const auto& p : producerInfos) {
      producerByHash[p.semanticHash] = p;
      if (auto slot = parseAggregateValueSlot(mm.getName(p.member))) nextSlot = std::max(nextSlot, *slot + 1);
   }

   llvm::SmallVector<subop::Member> newMembers(oldHt.getValueMembers().getMembers().begin(),
                                               oldHt.getValueMembers().getMembers().end());
   llvm::SmallVector<mlir::Type> newTypes;
   for (subop::Member m : newMembers) newTypes.push_back(mm.getType(m));

   llvm::SmallVector<AggregatePayloadMemberInfo, 16> peerInfos =
      collectAggregatePayloadMembers(query1, peerHtState);
   subop::MapOp producerMap = findProducerAggregateMap(buildStep);
   llvm::SmallVector<std::pair<unsigned, mlir::Type>, 4> insertedPayloads;
   for (unsigned peerIdx = 0; peerIdx < peerInfos.size(); ++peerIdx) {
      const auto& p = peerInfos[peerIdx];
      if (producerByHash.contains(p.semanticHash)) continue;
      if (p.semanticKey == "count:*") continue;
      tuples::ColumnRefAttr source = cloneColumnRefToContext(p.sourceColumn, ctx);
      auto sourceProducerMap = findMapProducingColumn(buildStep, source, cm);
      if (sourceProducerMap) {
         source = mapComputedRefForSemantic(sourceProducerMap, columnSemanticKey(source, cm), cm);
      } else {
         subop::MapOp peerMap = findMapProducingColumn(peerBuild, p.sourceColumn,
                                                       query1.getContext()
                                                          ->getLoadedDialect<tuples::TupleStreamDialect>()
                                                          ->getColumnManager());
         assert(peerMap && "aggregate union: missing computed aggregate payload requires peer map");
         source = clonePeerMapResultIntoProducer(buildStep, producerMap, peerMap, p.sourceColumn,
                                                reuseSynthetic, reusePeer, synthetic);
      }
      mlir::Type slotTy = cloneTypeToContext(p.type, ctx);
      subop::Member member = allocUnusedAggregateValueSlot(mm, slotTy, nextSlot);
      unsigned insertIdx = layout.payloadSemanticKeys.size();
      newMembers.insert(newMembers.begin() + insertIdx, member);
      newTypes.insert(newTypes.begin() + insertIdx, slotTy);
      layout.payloadSemanticKeys.insert(layout.payloadSemanticKeys.begin() + insertIdx, p.semanticKey);
      layout.payloadMembers.insert(layout.payloadMembers.begin() + insertIdx, member);
      layout.payloadColumnTypes.insert(layout.payloadColumnTypes.begin() + insertIdx, slotTy);
      insertAggregateLookupInitial(lookup, insertIdx, slotTy);
      insertAggregateReduceUpdate(reduce, insertIdx, source, member, slotTy);
      insertAggregateReduceCombine(reduce, insertIdx, slotTy);
      insertedPayloads.push_back({insertIdx, slotTy});
   }

   auto newFrag = subop::PreAggrHtFragmentType::get(ctx, oldHt.getKeyMembers(),
                                                    subop::StateMembersAttr::get(ctx, newMembers),
                                                    oldHt.getWithLock());
   auto newHt = subop::PreAggrHtType::get(ctx, oldHt.getKeyMembers(),
                                          subop::StateMembersAttr::get(ctx, newMembers),
                                          oldHt.getWithLock());
   synthetic.walk([&](subop::MergeOp merge) {
      if (merge.getRes().getType() == oldHt || merge.getRes().getType() == newHt) {
         for (auto [idx, ty] : insertedPayloads) insertAggregateMergeCombine(merge, idx, ty);
      }
   });
   updateSyntheticAggregateStateTypes(synthetic, oldFrag, newFrag, oldHt, newHt);
   syncAggregateEntryRefAttrs(synthetic, oldFrag, newFrag, oldHt, newHt);
   layout.producerHt = newHt;
}

static llvm::StringMap<llvm::SmallVector<subop::Member, 4>>
aggregateSemanticToConsumerMembers(const CachedAggregateLayout& layout, unsigned consumerIdx) {
   llvm::StringMap<llvm::SmallVector<subop::Member, 4>> out;
   llvm::ArrayRef<std::string> keys = consumerIdx == 0 ? layout.query0SemanticKeys : layout.query1SemanticKeys;
   llvm::ArrayRef<subop::Member> members = consumerIdx == 0 ? layout.query0Members : layout.query1Members;
   assert(keys.size() == members.size());
   for (size_t i = 0; i < keys.size(); ++i) out[keys[i]].push_back(members[i]);
   return out;
}

struct ConsumerAggregateAlignment {
   subop::PreAggrHtType ht;
   llvm::DenseMap<subop::Member, subop::Member> memberRemap;
   llvm::DenseMap<subop::Member, subop::Member> keyMemberRemap;
};

static llvm::DenseMap<subop::Member, subop::Member>
buildAggregateKeyMemberRemap(subop::PreAggrHtType oldHt, subop::PreAggrHtType alignedHt) {
   llvm::DenseMap<subop::Member, subop::Member> remap;
   auto& mm = oldHt.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   llvm::ArrayRef<subop::Member> oldKeys = oldHt.getKeyMembers().getMembers();
   llvm::ArrayRef<subop::Member> alignedKeys = alignedHt.getKeyMembers().getMembers();
   assert(oldKeys.size() == alignedKeys.size() && "aggregate cache_get key layout must preserve arity");
   for (size_t i = 0; i < oldKeys.size(); ++i) {
      if (mm.getType(oldKeys[i]) != mm.getType(alignedKeys[i]))
         llvm_unreachable("aggregate cache_get key layout must preserve key member types");
      if (oldKeys[i] != alignedKeys[i]) remap[oldKeys[i]] = alignedKeys[i];
   }
   return remap;
}

static ConsumerAggregateAlignment buildConsumerAggregateAlignment(const CachedAggregateLayout& layout,
                                                                  unsigned consumerIdx, mlir::MLIRContext* ctx) {
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   mlir::MLIRContext* producerCtx = layout.producerHt.getContext();

   llvm::SmallVector<subop::Member> keyMembers;
   for (subop::Member m : layout.producerHt.getKeyMembers().getMembers())
      keyMembers.push_back(cloneMemberToContext(m, producerCtx, ctx, /*allowMemberTypeUpdate=*/false));

   auto semanticToOld = aggregateSemanticToConsumerMembers(layout, consumerIdx);
   llvm::StringSet<> producerPayloadSemantics;
   for (const std::string& semantic : layout.payloadSemanticKeys) producerPayloadSemantics.insert(semantic);
   for (const auto& it : semanticToOld) {
      if (!producerPayloadSemantics.contains(it.getKey()))
         abortAggregateUnionUnsupported("consumer aggregate table has a payload semantic not present in producer");
   }
   llvm::StringMap<unsigned> nextOldIdx;
   llvm::StringMap<subop::Member> firstAlignedBySemantic;
   llvm::SmallVector<subop::Member> valueMembers;
   llvm::DenseMap<subop::Member, subop::Member> remap;
   unsigned nextSlot = 0;
   for (const auto& it : semanticToOld) {
      for (subop::Member m : it.second) {
         if (auto slot = parseAggregateValueSlot(mm.getName(m))) nextSlot = std::max(nextSlot, *slot + 1);
      }
   }
   for (subop::Member m : valueMembers) {
      if (auto slot = parseAggregateValueSlot(mm.getName(m))) nextSlot = std::max(nextSlot, *slot + 1);
   }

   assert(layout.payloadSemanticKeys.size() == layout.payloadMembers.size());
   for (size_t i = 0; i < layout.payloadSemanticKeys.size(); ++i) {
      llvm::StringRef semantic = layout.payloadSemanticKeys[i];
      mlir::Type slotTy = cloneTypeToContext(layout.payloadColumnTypes[i], ctx);
      subop::Member chosen;
      auto itOld = semanticToOld.find(semantic);
      if (itOld != semanticToOld.end()) {
         unsigned& oldIdx = nextOldIdx[semantic];
         while (oldIdx < itOld->second.size()) {
            subop::Member candidate = itOld->second[oldIdx++];
            if (mm.getType(candidate) == slotTy) {
               chosen = candidate;
               break;
            }
         }
      }
      if (!chosen) chosen = allocUnusedAggregateValueSlot(mm, slotTy, nextSlot);
      valueMembers.push_back(chosen);
      if (!firstAlignedBySemantic.contains(semantic)) firstAlignedBySemantic[semantic] = chosen;
      if (itOld != semanticToOld.end()) {
         for (subop::Member oldMember : itOld->second) {
            if (mm.getType(oldMember) == slotTy) remap[oldMember] = firstAlignedBySemantic[semantic];
         }
      }
   }

   ConsumerAggregateAlignment out;
   out.ht = subop::PreAggrHtType::get(ctx, subop::StateMembersAttr::get(ctx, keyMembers),
                                      subop::StateMembersAttr::get(ctx, valueMembers),
                                      layout.producerHt.getWithLock());
   out.memberRemap = std::move(remap);
   return out;
}

/// After \c cache_get HIV types are updated: fix \c scan_list list/elem carriers and every \c gather in the
/// same block whose \c lookup_u_* ref scope matches \c scan_list elem (same probe site name).
static void alignScanListAndProbeUsesInBlock(subop::ScanListOp scanList, subop::HashIndexedViewType alignedHiv,
                                             subop::HashIndexedViewType consumerHivBeforeAlign,
                                             const CachedJoinBufferLayout& consumerLayout,
                                             const ProbeAlignDebugCtx* dbg) {
   auto* ctx = scanList.getContext();
   auto listTy = mlir::dyn_cast<subop::ListType>(scanList.getList().getType());
   if (!listTy) return;
   auto listElemLer = mlir::dyn_cast<subop::LookupEntryRefType>(listTy.getT());
   if (!listElemLer) return;
   auto listHiv = mlir::dyn_cast<subop::HashIndexedViewType>(listElemLer.getState());
   if (!listHiv) return;

   setValueCarrierType(scanList.getList(), alignedHiv, consumerHivBeforeAlign);
   listTy = mlir::dyn_cast<subop::ListType>(scanList.getList().getType());
   if (!listTy) return;
   listElemLer = mlir::dyn_cast<subop::LookupEntryRefType>(listTy.getT());
   if (!listElemLer) return;
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   subop::LookupEntryRefType expectedLer = subop::LookupEntryRefType::get(ctx, alignedHiv);
   auto scanElem = scanList.getElem();
   auto [entryScope, entryLeaf] = cm.getName(&scanElem.getColumn());
   const std::string elemTyBefore = mlirTypeToString(scanElem.getColumn().type);

   debugProbeAlign(dbg, [&](llvm::raw_ostream& os) {
      os << "scan_list op=" << scanList.getOperation();
      os << " list=" << scanList.getList();
      os << " lookup_scope=" << entryScope << " leaf=" << entryLeaf;
      os << " aligned_hiv=" << mlirTypeToString(alignedHiv);
      os << " elem_type_before=" << elemTyBefore;
   });

   if (!isJoinProbeCompilerScope(entryScope)) return;

   scanElem = cm.createDef(entryScope, entryLeaf);
   scanElem.getColumn().type = expectedLer;
   scanList.setElemAttr(scanElem);

   subop::ListType expectedListTy = subop::ListType::get(ctx, expectedLer);
   if (scanList.getList().getType() != expectedListTy) scanList.getList().setType(expectedListTy);

   debugProbeAlign(dbg, [&](llvm::raw_ostream& os) {
      os << "  scan_list elem_type_after=" << mlirTypeToString(scanElem.getColumn().type);
   });

   llvm::StringMap<subop::Member> semKeyToMember = semKeyToMemberMap(consumerLayout);
   llvm::DenseSet<subop::Member> hivMemberSet(consumerLayout.payloadMembers.begin(),
                                                consumerLayout.payloadMembers.end());
   llvm::StringMap<subop::Member> hivMemberByName;
   for (subop::Member m : alignedHiv.getValueMembers().getMembers()) {
      hivMemberByName[mm.getName(m)] = m;
   }

   auto resolveGatherMember = [&](subop::Member useMem, tuples::ColumnDefAttr def) -> subop::Member {
      if (hivMemberSet.contains(useMem)) return useMem;
      if (auto it = hivMemberByName.find(mm.getName(useMem)); it != hivMemberByName.end()) return it->second;
      auto [scope, leaf] = cm.getName(&def.getColumn());
      std::string semKey = columnSemanticKey(scope, leaf);
      if (auto it = semKeyToMember.find(semKey); it != semKeyToMember.end()) return it->second;
      if (!isPayloadMemberSlotName(leaf)) {
         llvm::StringRef wantLeaf = normalizeColumnIdentifier(leaf);
         for (const auto& e : semKeyToMember) {
            if (normalizeColumnIdentifier(semanticKeyLeaf(e.getKey())) == wantLeaf) return e.getValue();
         }
      }
      return useMem;
   };

   mlir::Block* block = scanList->getBlock();
   for (mlir::Operation& op : *block) {
      auto gather = mlir::dyn_cast<subop::GatherOp>(&op);
      if (!gather) continue;
      auto gatherRef = gather.getRef();
      auto [refScope, refLeaf] = cm.getName(&gatherRef.getColumn());

      if (!lookupProbeRefScopesMatch(entryScope, refScope)) {
         if (isJoinProbeCompilerScope(refScope)) {
            debugProbeAlign(dbg, [&](llvm::raw_ostream& os) {
               os << "  skip gather op=" << gather.getOperation() << " ref_scope=" << refScope
                  << " (want " << entryScope << ")";
            });
         }
         continue;
      }

      const std::string refTyBefore = mlirTypeToString(gatherRef.getColumn().type);
      debugProbeAlign(dbg, [&](llvm::raw_ostream& os) {
         os << "  rewrite gather op=" << gather.getOperation() << " ref_scope=" << refScope
            << " leaf=" << refLeaf << " ref_type_before=" << refTyBefore;
      });

      gatherRef.getColumn().type = expectedLer;
      gather.setRefAttr(gatherRef);

      llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> newMapping;
      bool mappingChanged = false;
      for (auto& pr : gather.getMapping().getMapping()) {
         subop::Member nm = resolveGatherMember(pr.first, pr.second);
         if (nm != pr.first) mappingChanged = true;
         tuples::ColumnDefAttr def = pr.second;
         auto [defScope, defLeaf] = cm.getName(&def.getColumn());
         if (isJoinProbeCompilerScope(defScope) && !lookupProbeRefScopesMatch(entryScope, defScope)) {
            debugProbeAlign(dbg, [&](llvm::raw_ostream& os) {
               os << "    mapping payload@" << defScope << "::" << defLeaf << " -> @" << entryScope
                  << "::" << mm.getName(nm);
            });
            def = cm.createDef(entryScope, mm.getName(nm));
            def.getColumn().type = mm.getType(nm);
            mappingChanged = true;
         } else if (auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(def.getColumn().type)) {
            auto st = mlir::dyn_cast<subop::HashIndexedViewType>(ler.getState());
            if (st && lookupProbeRefScopesMatch(entryScope, defScope) &&
                sameHashIndexedViewLayout(st, alignedHiv) && def.getColumn().type != expectedLer) {
               def.getColumn().type = expectedLer;
               mappingChanged = true;
            }
         }
         newMapping.push_back({nm, def});
      }
      if (mappingChanged) gather.setMappingAttr(subop::ColumnDefMemberMappingAttr::get(ctx, newMapping));
      syncMapInputColsFromGather(gather, cm);

      debugProbeAlign(dbg, [&](llvm::raw_ostream& os) {
         os << "    gather ref_type_after=" << mlirTypeToString(gather.getRef().getColumn().type);
      });
   }
}

/// From \c cache_get HIV: recurse through port args; at \c scan_list update list/elem and same-block probe gathers.
static void traverseConsumerHivUsesFromRoot(mlir::Value root, subop::HashIndexedViewType consumerHiv,
                                            subop::HashIndexedViewType consumerHivBeforeAlign,
                                            const CachedJoinBufferLayout& consumerLayout,
                                            ConsumerCachedHivSites& sites, const ProbeAlignDebugCtx* dbg) {
   llvm::DenseSet<void*> visited;
   llvm::SmallVector<mlir::Value> worklist;
   auto enqueue = [&](mlir::Value val) {
      if (!val || !visited.insert(val.getAsOpaquePointer()).second) return;
      addToSsaClosure(val, sites);
      worklist.push_back(val);
   };

   setValueCarrierType(root, consumerHiv, consumerHivBeforeAlign, dbg);
   enqueue(root);

   while (!worklist.empty()) {
      mlir::Value v = worklist.pop_back_val();
      if (typeEmbedsHashIndexedView(v.getType()))
         setValueCarrierType(v, consumerHiv, consumerHivBeforeAlign, dbg);

      for (mlir::Operation* user : v.getUsers()) {
         if (auto ret = mlir::dyn_cast<subop::ExecutionStepReturnOp>(user)) {
            auto step = mlir::dyn_cast<subop::ExecutionStepOp>(ret->getParentOp());
            if (!step) continue;
            for (unsigned i = 0; i < ret.getNumOperands() && i < step.getNumResults(); ++i) {
               if (ret.getOperand(i) != v) continue;
               setValueCarrierType(step.getResult(i), consumerHiv, consumerHivBeforeAlign, dbg);
               enqueue(step.getResult(i));
            }
            continue;
         }

         if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(user)) {
            unsigned i = operandIndexOf(step.getOperation(), v);
            mlir::Block& body = step.getSubOps().front();
            assert(i < body.getNumArguments() && "execution_step operand without block argument");
            mlir::BlockArgument barg = body.getArgument(i);
            setValueCarrierType(barg, consumerHiv, consumerHivBeforeAlign, dbg);
            if (i < step.getNumResults())
               setValueCarrierType(step.getResult(i), consumerHiv, consumerHivBeforeAlign, dbg);
            enqueue(barg);
            if (i < step.getNumResults()) enqueue(step.getResult(i));
            continue;
         }

         if (auto neg = mlir::dyn_cast<subop::NestedExecutionGroupOp>(user)) {
            unsigned i = operandIndexOf(neg.getOperation(), v);
            mlir::Block& body = neg.getSubOps().front();
            assert(i < body.getNumArguments());
            mlir::BlockArgument barg = body.getArgument(i);
            setValueCarrierType(barg, consumerHiv, consumerHivBeforeAlign, dbg);
            enqueue(barg);
            continue;
         }

         if (auto lookup = mlir::dyn_cast<subop::LookupOp>(user)) {
            assert(lookup.getState() == v && "lookup state operand must be the HIV value");
            auto* ctx = v.getContext();
            auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
            auto listRef = lookup.getRef();
            auto [listScope, listLeaf] = cm.getName(&listRef.getColumn());
            if (isJoinProbeCompilerScope(listScope)) sites.probeLookupScopes.insert(listScope);
            mlir::Type expectedListTy =
               subop::ListType::get(ctx, subop::LookupEntryRefType::get(ctx, consumerHiv));
            if (listRef.getColumn().type != expectedListTy)
               listRef.getColumn().type = expectedListTy;
            debugProbeAlign(dbg, [&](llvm::raw_ostream& os) {
               os << "traverse lookup op=" << lookup.getOperation() << " list_scope=" << listScope
                  << " list_leaf=" << listLeaf;
            });
            enqueue(lookup.getResult());
            for (mlir::Operation* streamUser : lookup.getResult().getUsers()) {
               auto nm = mlir::dyn_cast<subop::NestedMapOp>(streamUser);
               if (!nm) continue;
               enqueue(nm.getRes());
               mlir::Region& reg = nm.getRegion();
               if (reg.empty()) continue;
               for (mlir::BlockArgument barg : reg.front().getArguments()) {
                  setValueCarrierType(barg, consumerHiv, consumerHivBeforeAlign, dbg);
                  enqueue(barg);
               }
            }
            continue;
         }

         if (auto nm = mlir::dyn_cast<subop::NestedMapOp>(user)) {
            if (nm.getStream() != v) continue;
            enqueue(nm.getRes());
            mlir::Region& reg = nm.getRegion();
            if (!reg.empty()) {
               for (mlir::BlockArgument barg : reg.front().getArguments()) {
                  setValueCarrierType(barg, consumerHiv, consumerHivBeforeAlign, dbg);
                  enqueue(barg);
               }
            }
            continue;
         }

         if (auto scanList = mlir::dyn_cast<subop::ScanListOp>(user)) {
            assert(scanList.getList() == v && "scan_list list operand must be the carrier value");
            sites.scanListsFromTraverse.push_back(scanList);
            auto& cm = v.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
            auto scanElem = scanList.getElem();
            auto [elemScope, elemLeaf] = cm.getName(&scanElem.getColumn());
            if (isJoinProbeCompilerScope(elemScope)) sites.probeLookupScopes.insert(elemScope);
            debugProbeAlign(dbg, [&](llvm::raw_ostream& os) {
               os << "traverse scan_list op=" << scanList.getOperation();
               os << " list_operand=";
               printSsaValueOrigin(os, v);
               os << " elem_scope=" << elemScope << " leaf=" << elemLeaf;
            });
            continue;
         }

         // Closure discovery only follows HIV/list carriers; ignore unrelated users (e.g. tuples.return).
      }
   }
}

static void expandConsumerProbeClosureThroughPorts(mlir::ModuleOp consumer, llvm::DenseSet<void*>& closure) {
   for (;;) {
      size_t before = closure.size();
      expandClosureThroughExecutionStepPorts(consumer, closure);
      expandClosureThroughNestedExecutionGroupPorts(consumer, closure);
      if (closure.size() == before) break;
   }
}

static void refreshConsumerCacheGetProbeClosure(mlir::ModuleOp consumer, ConsumerCacheGetProbeClosure& probe,
                                                llvm::StringRef passName) {
   assert(probe.cacheGetRoot && "consumer probe closure must have a cache_get root");
   assert(probe.alignedHiv && "consumer probe closure must have an aligned HIV layout");

   ConsumerCachedHivSites sites;
   sites.consumerHivBeforeAlign = probe.consumerHivBeforeAlign;
   ProbeAlignDebugCtx traverseDbg;
   traverseDbg.cacheKey = probe.cacheKey;
   traverseDbg.passName = passName;
   traverseConsumerHivUsesFromRoot(probe.cacheGetRoot, probe.alignedHiv, probe.consumerHivBeforeAlign,
                                   probe.consumerLayout, sites, &traverseDbg);
   probe.ssaClosure = std::move(sites.ssaClosure);
   probe.probeLookupScopes = std::move(sites.probeLookupScopes);
   probe.scanListsFromTraverse = std::move(sites.scanListsFromTraverse);
   expandConsumerProbeClosureThroughPorts(consumer, probe.ssaClosure);
}

static ConsumerCacheGetProbeClosure buildConsumerCacheGetProbeClosure(
   mlir::ModuleOp consumer, mlir::Value cacheGetResult, subop::HashIndexedViewType alignedHiv,
   subop::HashIndexedViewType consumerHivBeforeAlign, const CachedJoinBufferLayout& consumerLayout,
   std::optional<uint64_t> cacheKey, std::optional<unsigned> consumerReuseQueryIndex) {
   ConsumerCacheGetProbeClosure out;
   out.cacheGetRoot = cacheGetResult;
   out.alignedHiv = alignedHiv;
   out.consumerHivBeforeAlign = consumerHivBeforeAlign;
   out.consumerLayout = consumerLayout;
   out.cacheKey = cacheKey;
   out.consumerReuseQueryIndex = consumerReuseQueryIndex;
   refreshConsumerCacheGetProbeClosure(consumer, out, "traverse");
   return out;
}

static bool scanListListCarrierSharesAlignedHivKeyLayout(subop::ScanListOp scan,
                                                         subop::HashIndexedViewType alignedHiv) {
   auto listTy = mlir::dyn_cast<subop::ListType>(scan.getList().getType());
   if (!listTy) return false;
   auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(listTy.getT());
   if (!ler || !lookupEntryRefEmbedsHashIndexedView(ler)) return false;
   auto st = mlir::dyn_cast<subop::HashIndexedViewType>(ler.getState());
   if (!st) return false;
   return st.getCompareHashForLookup() == alignedHiv.getCompareHashForLookup() &&
          st.getKeyMembers().getMembers().size() == alignedHiv.getKeyMembers().getMembers().size();
}

static bool scanListReachedFromCacheGetHivTraverse(subop::ScanListOp scan, subop::HashIndexedViewType alignedHiv,
                                                   const ConsumerCacheGetProbeClosure& probe,
                                                   const ProbeAlignDebugCtx* dbg) {
   const bool listEmbeds = typeEmbedsHashIndexedViewState(scan.getList().getType(), alignedHiv);
   const bool inClosure = opaqueClosureContains(probe.ssaClosure, scan.getList());
   const bool seenOnTraverse = llvm::is_contained(probe.scanListsFromTraverse, scan);
   if (listEmbeds && inClosure && seenOnTraverse) return true;
   // Q1 may still carry pre-align embedded HIV (e.g. filter_pred$0) on the list operand while cache_get is $1.
   if (seenOnTraverse && inClosure && scanListListCarrierSharesAlignedHivKeyLayout(scan, alignedHiv)) return true;

   if (listEmbeds && (inClosure || seenOnTraverse)) {
      llvm::StringRef reason = !seenOnTraverse   ? "not_reached_from_cache_get_traverse"
                               : !inClosure      ? "list_operand_not_in_closure"
                               : /* !listEmbeds */ "list_type_does_not_embed_aligned_hiv";
      debugScanListSkippedFromCacheGetProbe(scan, alignedHiv, probe, reason, dbg);
   }
   return false;
}

static void alignScanListForProbeClosure(mlir::ModuleOp consumer, ConsumerCacheGetProbeClosure& probe,
                                         llvm::StringRef passName) {
   ProbeAlignDebugCtx dbg;
   dbg.cacheKey = probe.cacheKey;
   dbg.passName = passName;

   for (subop::ScanListOp scan : probe.scanListsFromTraverse) {
      if (!scanListReachedFromCacheGetHivTraverse(scan, probe.alignedHiv, probe, &dbg)) continue;
      alignScanListAndProbeUsesInBlock(scan, probe.alignedHiv, probe.consumerHivBeforeAlign, probe.consumerLayout,
                                       &dbg);
      debugScanListCarrierMismatch(scan, &dbg);
   }
   debugScanListsMissedByTraverse(consumer, probe, &dbg);

   consumer.walk([&](subop::ScanListOp scan) {
      if (!opaqueClosureContains(probe.ssaClosure, scan.getList())) return;
      if (!typeEmbedsHashIndexedViewState(scan.getList().getType(), probe.alignedHiv)) return;
      debugScanListCarrierMismatch(scan, &dbg);
   });
}

/// Refresh carriers outside the \c cache_get traversal closure only when they already embed \p replaceFromHiv
/// (e.g. stale producer-layout \c lookup_entry_ref). Does not retag unrelated local join HIVs (other \c hash$N).
static void refreshStaleEmbeddedHivCarriers(mlir::ModuleOp module, subop::HashIndexedViewType canonicalHiv,
                                            subop::HashIndexedViewType replaceFromHiv,
                                            const llvm::DenseSet<void*>& closure,
                                            const ProbeAlignDebugCtx* dbg) {
   if (!replaceFromHiv || replaceFromHiv == canonicalHiv) return;
   auto* ctx = module.getContext();
   for (;;) {
      llvm::SmallVector<std::pair<mlir::Value, mlir::Type>> updates;
      module.walk([&](mlir::Operation* op) {
         auto consider = [&](mlir::Value val) {
            if (!val) return;
            if (!opaqueClosureContains(closure, val)) return;
            if (!typeEmbedsHashIndexedViewState(val.getType(), replaceFromHiv)) return;
            mlir::Type oldTy = val.getType();
            if (mlir::Type nt = replaceEmbeddedHivInType(ctx, oldTy, canonicalHiv, replaceFromHiv); nt != oldTy) {
               debugProbeAlign(dbg, [&](llvm::raw_ostream& os) {
                  os << "refreshStaleEmbeddedHivCarriers value=" << val << " old_type=" << mlirTypeToString(oldTy)
                     << " new_type=" << mlirTypeToString(nt);
               });
               updates.push_back({val, nt});
            }
         };
         for (mlir::Value r : op->getResults()) consider(r);
         for (mlir::Region& reg : op->getRegions()) {
            for (mlir::Block& block : reg) {
               for (mlir::BlockArgument a : block.getArguments()) consider(a);
            }
         }
      });
      if (updates.empty()) break;
      for (auto [val, nt] : updates) val.setType(nt);
   }
}

struct ConsumerColumnBinding {
   subop::Member member;
   mlir::Type columnType;
};

/// Match union payload semantic key to a consumer probe/build member by normalized column leaf.
static std::optional<subop::Member> findConsumerMemberForPayloadSlotByLeaf(
   llvm::StringRef semanticKey, const llvm::StringMap<ConsumerColumnBinding>& semanticToConsumer) {
   llvm::StringRef wantLeaf = normalizeColumnIdentifier(semanticKeyLeaf(semanticKey));
   std::optional<subop::Member> found;
   unsigned matches = 0;
   for (const auto& entry : semanticToConsumer) {
      if (normalizeColumnIdentifier(semanticKeyLeaf(entry.getKey())) != wantLeaf) continue;
      found = entry.second.member;
      ++matches;
   }
   if (matches == 1) return found;
   return std::nullopt;
}

static std::optional<subop::Member> findConsumerMemberByName(llvm::StringRef name,
                                                             subop::HashIndexedViewType consumerHivBeforeAlign) {
   if (!consumerHivBeforeAlign) return std::nullopt;
   auto& mm = consumerHivBeforeAlign.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   for (subop::Member m : consumerHivBeforeAlign.getValueMembers().getMembers()) {
      if (mm.getName(m) == name) return m;
   }
   return std::nullopt;
}

/// Map semantic table columns to members used on the consumer's cached HIV probe path (pre-align gathers).
static void collectConsumerPayloadSemanticMembers(mlir::ModuleOp consumer,
                                                  llvm::StringMap<ConsumerColumnBinding>& out) {
   auto& cm = consumer.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto record = [&](subop::Member mem, llvm::StringRef scope, llvm::StringRef leaf, mlir::Type colTy) {
      if (isJoinProbeCompilerScope(scope)) return;
      out.try_emplace(columnSemanticKey(scope, leaf), ConsumerColumnBinding{mem, colTy});
   };

   auto recordOnHivState = [&](subop::Member mem, tuples::ColumnDefAttr def) {
      auto [scope, leaf] = cm.getName(&def.getColumn());
      record(mem, scope, leaf, def.getColumn().type);
   };

   consumer.walk([&](subop::GatherOp gather) {
      auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(gather.getRef().getColumn().type);
      if (!ler || !mlir::isa<subop::HashIndexedViewType>(ler.getState())) return;
      for (auto [mem, def] : gather.getMapping().getMapping()) {
         recordOnHivState(mem, def);
      }
   });
   consumer.walk([&](subop::ScatterOp scatter) {
      auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(scatter.getRef().getColumn().type);
      if (!ler || !mlir::isa<subop::HashIndexedViewType>(ler.getState())) return;
      for (auto [mem, colRef] : scatter.getMapping().getMapping()) {
         auto [scope, leaf] = cm.getName(&colRef.getColumn());
         record(mem, scope, leaf, colRef.getColumn().type);
      }
   });
   consumer.walk([&](subop::MaterializeOp mat) {
      if (!mlir::isa<subop::BufferType>(mat.getState().getType())) return;
      for (auto& [mem, colRef] : mat.getMapping().getMapping()) {
         auto [scope, leaf] = cm.getName(&colRef.getColumn());
         record(mem, scope, leaf, colRef.getColumn().type);
      }
   });
}

static bool consumerQueryHadPayloadColumn(const CachedJoinBufferLayout& layout, llvm::StringRef semKey,
                                          unsigned consumerQueryIndex) {
   if (consumerQueryIndex == 0) return llvm::is_contained(layout.query0SemanticKeys, semKey);
   return llvm::is_contained(layout.query1SemanticKeys, semKey);
}

/// Remap \c gather member keys to the aligned \c cache_get HIV layout (semantic keys, not slot guessing).
static void remapClosureGathersToAlignedConsumerHiv(mlir::ModuleOp consumer,
                                                    const llvm::DenseSet<void*>* ssaClosure,
                                                    subop::HashIndexedViewType alignedHiv,
                                                    const CachedJoinBufferLayout& consumerLayout,
                                                    const llvm::StringSet<>* probeLookupScopes,
                                                    const ProbeAlignDebugCtx* dbg) {
   auto* ctx = consumer.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   llvm::DenseSet<subop::Member> hivMemberSet;
   llvm::StringMap<subop::Member> hivMemberByName;
   for (subop::Member m : alignedHiv.getValueMembers().getMembers()) {
      hivMemberSet.insert(m);
      hivMemberByName[mm.getName(m)] = m;
   }

   llvm::StringMap<subop::Member> semKeyToMember;
   assert(consumerLayout.payloadSemanticKeys.size() == consumerLayout.payloadMembers.size());
   for (size_t i = 0; i < consumerLayout.payloadSemanticKeys.size(); ++i) {
      semKeyToMember[consumerLayout.payloadSemanticKeys[i]] = consumerLayout.payloadMembers[i];
   }

   llvm::StringMap<ConsumerColumnBinding> semanticToConsumer;
   collectConsumerPayloadSemanticMembers(consumer, semanticToConsumer);

   auto resolveGatherMember = [&](subop::Member useMem, tuples::ColumnDefAttr def) -> subop::Member {
      if (auto itRemap = consumerLayout.probeGatherMemberRemap.find(useMem);
          itRemap != consumerLayout.probeGatherMemberRemap.end()) {
         return itRemap->second;
      }
      if (hivMemberSet.contains(useMem)) return useMem;
      if (auto it = hivMemberByName.find(mm.getName(useMem)); it != hivMemberByName.end()) return it->second;
      auto [scope, leaf] = cm.getName(&def.getColumn());
      std::string semKey = columnSemanticKey(scope, leaf);
      if (auto it = semKeyToMember.find(semKey); it != semKeyToMember.end()) return it->second;
      if (auto it = semanticToConsumer.find(semKey); it != semanticToConsumer.end()) {
         if (hivMemberSet.contains(it->second.member)) return it->second.member;
         if (auto bn = hivMemberByName.find(mm.getName(it->second.member)); bn != hivMemberByName.end()) {
            return bn->second;
         }
      }
      if (!isPayloadMemberSlotName(leaf)) {
         llvm::StringRef wantLeaf = normalizeColumnIdentifier(leaf);
         for (const auto& e : semKeyToMember) {
            if (normalizeColumnIdentifier(semanticKeyLeaf(e.getKey())) == wantLeaf) return e.getValue();
         }
         for (const auto& e : semanticToConsumer) {
            if (normalizeColumnIdentifier(semanticKeyLeaf(e.getKey())) == wantLeaf) {
               if (hivMemberSet.contains(e.second.member)) return e.second.member;
               if (auto bn = hivMemberByName.find(mm.getName(e.second.member)); bn != hivMemberByName.end()) {
                  return bn->second;
               }
            }
         }
      } else if (isJoinProbeCompilerScope(scope)) {
         // Probe gathers name columns `@lookup_u_*::@member$N`; map pre-union consumer slots via semantics.
         for (const auto& e : semanticToConsumer) {
            if (e.second.member != useMem) continue;
            if (auto it = semKeyToMember.find(e.getKey()); it != semKeyToMember.end()) return it->second;
         }
      }
      if (isJoinProbeCompilerScope(scope) && isPayloadMemberSlotName(leaf) && !hivMemberSet.contains(useMem)) {
         mlir::Type wantTy = mm.getType(useMem);
         llvm::SmallVector<subop::Member, 4> sameTyInAligned;
         for (subop::Member m : alignedHiv.getValueMembers().getMembers()) {
            if (mm.getType(m) == wantTy) sameTyInAligned.push_back(m);
         }
         if (sameTyInAligned.size() == 1) return sameTyInAligned[0];
      }
      return useMem;
   };

   auto expectedEntryRef = subop::LookupEntryRefType::get(ctx, alignedHiv);
   consumer.walk([&](subop::GatherOp gather) {
      auto gatherRef = gather.getRef();
      auto [refScope, refLeaf] = cm.getName(&gatherRef.getColumn());
      if (!isJoinProbeCompilerScope(refScope)) return;
      if (probeLookupScopes && !probeLookupScopes->empty() && !probeLookupScopes->contains(refScope)) {
         debugProbeAlign(dbg, [&](llvm::raw_ostream& os) {
            os << "remapClosure skip gather op=" << gather.getOperation() << " ref_scope=" << refScope
               << " (not in cache_get lookup scopes)";
         });
         return;
      }
      if (ssaClosure && !opOperandsOrNestedBlockArgsTouchClosure(gather.getOperation(), *ssaClosure) &&
          (!probeLookupScopes || !probeLookupScopes->contains(refScope))) {
         return;
      }
      auto lerBefore = mlir::dyn_cast<subop::LookupEntryRefType>(gatherRef.getColumn().type);
      auto stBefore = lerBefore ? mlir::dyn_cast<subop::HashIndexedViewType>(lerBefore.getState()) : nullptr;
      if (!stBefore || !sameHashIndexedViewJoinKey(stBefore, alignedHiv)) return;

      if (gatherRef.getColumn().type != expectedEntryRef) {
         debugProbeAlign(dbg, [&](llvm::raw_ostream& os) {
            os << "remapClosure gather op=" << gather.getOperation() << " ref_scope=" << refScope
               << " ref_type_before=" << mlirTypeToString(gatherRef.getColumn().type)
               << " ref_type_after=" << mlirTypeToString(expectedEntryRef);
         });
         gatherRef.getColumn().type = expectedEntryRef;
         gather.setRefAttr(gatherRef);
      }
      auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(gatherRef.getColumn().type);
      if (!ler) return;
      auto st = mlir::dyn_cast<subop::HashIndexedViewType>(ler.getState());
      if (!st || !sameHashIndexedViewLayout(st, alignedHiv)) return;
      llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> out;
      llvm::StringMap<tuples::ColumnRefAttr> oldDefRefsToNewDefs;
      bool changed = false;
      for (auto& pr : gather.getMapping().getMapping()) {
         subop::Member nm = resolveGatherMember(pr.first, pr.second);
         if (auto bn = hivMemberByName.find(mm.getName(nm)); bn != hivMemberByName.end()) nm = bn->second;
         else if (!hivMemberSet.contains(nm)) nm = pr.first;
         if (nm != pr.first) changed = true;
         tuples::ColumnDefAttr def = pr.second;
         auto [defScope, defLeaf] = cm.getName(&def.getColumn());
         bool probePayloadLeafNeedsRename = isJoinProbeCompilerScope(defScope) &&
            isPayloadMemberSlotName(defLeaf) && defLeaf != mm.getName(nm);
         if (isJoinProbeCompilerScope(defScope) &&
             (!lookupProbeRefScopesMatch(refScope, defScope) || probePayloadLeafNeedsRename)) {
            std::string oldDefKey = columnSemanticKey(defScope, defLeaf);
            debugProbeAlign(dbg, [&](llvm::raw_ostream& os) {
               os << "  remapClosure mapping @" << defScope << "::" << defLeaf << " -> @" << refScope
                  << "::" << mm.getName(nm);
            });
            def = cm.createDef(refScope, mm.getName(nm));
            def.getColumn().type = mm.getType(nm);
            oldDefRefsToNewDefs[oldDefKey] = cm.createRef(&def.getColumn());
            changed = true;
         } else if (auto lerDef = mlir::dyn_cast<subop::LookupEntryRefType>(def.getColumn().type)) {
            if (sameHashIndexedViewLayout(mlir::cast<subop::HashIndexedViewType>(lerDef.getState()), alignedHiv) &&
                def.getColumn().type != gatherRef.getColumn().type) {
               def.getColumn().type = gatherRef.getColumn().type;
               changed = true;
            }
         }
         out.push_back({nm, def});
      }
      if (changed) gather.setMappingAttr(subop::ColumnDefMemberMappingAttr::get(ctx, out));
      syncMapInputColsFromGather(gather, cm, &oldDefRefsToNewDefs);
   });
}

/// Per-consumer HIV for \c cache_get: payload physical order follows the cached producer layout, but each slot
/// reuses the consumer's own \c member$N name and column type when that query already had the column; union-only
/// columns are inserted with producer types under fresh \c member$N slots (no type overwrite on existing members).
/// All union \c filter_pred$N slots are retained (same as synthetic); \p consumerReuseQueryIndex only selects which
/// pred is gathered on probe — not which preds appear in the aligned HIV type.
static subop::HashIndexedViewType buildConsumerAlignedHivType(mlir::ModuleOp consumer,
                                                                subop::HashIndexedViewType producerHiv,
                                                                const CachedJoinBufferLayout& layout,
                                                                subop::HashIndexedViewType consumerHivBeforeAlign,
                                                                std::optional<unsigned> consumerReuseQueryIndex,
                                                                CachedJoinBufferLayout& outConsumerLayout) {
   auto* ctx = consumer.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();

   llvm::StringMap<ConsumerColumnBinding> semanticToConsumer;
   collectConsumerPayloadSemanticMembers(consumer, semanticToConsumer);

   llvm::DenseMap<subop::Member, subop::Member> producerPayloadToConsumer;
   llvm::StringMap<subop::Member> assignedBySemantic;
   llvm::DenseSet<subop::Member> assignedMembers;
   llvm::DenseMap<subop::Member, subop::Member> probeGatherMemberRemap;
   assert(layout.payloadSemanticKeys.size() == layout.payloadMembers.size());

   llvm::SmallVector<subop::Member, 8> oldNonPredValueMembers;
   if (consumerHivBeforeAlign) {
      for (subop::Member m : consumerHivBeforeAlign.getValueMembers().getMembers()) {
         if (parseFilterPredMemberSlot(mm.getName(m))) continue;
         oldNonPredValueMembers.push_back(m);
      }
   }
   unsigned unionNonPredValueMembers = 0;
   for (llvm::StringRef semKey : layout.payloadSemanticKeys) {
      unsigned predIdx = 0;
      if (!parseFilterPredLayoutSemanticKey(semKey, predIdx)) ++unionNonPredValueMembers;
   }
   unsigned nonPredPayloadIndex = 0;

   auto ensureConsumerMember = [&](std::optional<subop::Member> reused, mlir::Type producerSlotTy) -> subop::Member {
      if (reused) {
         mlir::Type consumerTy = mm.getType(*reused);
         return mm.getOrCreateMemberDirect(mm.getName(*reused), consumerTy, /*allowTypeUpdate=*/false);
      }
      llvm::SmallVector<subop::Member, 8> bump;
      if (consumerHivBeforeAlign) {
         for (subop::Member m : consumerHivBeforeAlign.getValueMembers().getMembers()) bump.push_back(m);
      }
      for (const auto& assigned : assignedBySemantic) bump.push_back(assigned.second);
      unsigned nextSlot = nextPayloadMemberSlot(mm, assignedBySemantic, bump);
      mlir::Type slotTy = cloneTypeToContext(producerSlotTy, ctx);
      return allocUnusedPayloadMemberSlot(mm, slotTy, nextSlot);
   };

   for (size_t i = 0; i < layout.payloadSemanticKeys.size(); ++i) {
      mlir::Type producerSlotTy = layout.payloadColumnTypes[i];
      llvm::StringRef semKey = layout.payloadSemanticKeys[i];
      subop::Member producerMem = layout.payloadMembers[i];
      std::optional<subop::Member> probeGatherRemapFrom;

      unsigned predIdx = 0;
      if (parseFilterPredLayoutSemanticKey(semKey, predIdx)) {
         subop::Member consumerMem = makeOrGetPredMemberForSlot(ctx, predIdx);
         assignedBySemantic[semKey] = consumerMem;
         assignedMembers.insert(consumerMem);
         producerPayloadToConsumer[producerMem] = consumerMem;
         continue;
      }
      unsigned curNonPredPayloadIndex = nonPredPayloadIndex++;

      std::optional<subop::Member> reused;
      if (auto it = semanticToConsumer.find(semKey); it != semanticToConsumer.end()) {
         reused = it->second.member;
      } else {
         reused = findConsumerMemberForPayloadSlotByLeaf(semKey, semanticToConsumer);
      }
      // Probe gathers use @lookup_u_* column defs; reuse the consumer's pre-cache_get HIV slot name
      // when this query already had the column (assignPayloadMembersForPlan keeps member$N / flag$N).
      if (!reused && consumerHivBeforeAlign) {
         auto& producerMm = producerHiv.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
         reused = findConsumerMemberByName(producerMm.getName(producerMem), consumerHivBeforeAlign);
         if (!reused && oldNonPredValueMembers.size() == unionNonPredValueMembers &&
             curNonPredPayloadIndex < oldNonPredValueMembers.size()) {
            subop::Member oldMember = oldNonPredValueMembers[curNonPredPayloadIndex];
            if (!assignedMembers.contains(oldMember)) reused = oldMember;
         }
         if (!reused) {
            llvm::ArrayRef<subop::Member> oldVals = consumerHivBeforeAlign.getValueMembers().getMembers();
            for (unsigned qIdx : {0u, 1u}) {
               if (!consumerQueryHadPayloadColumn(layout, semKey, qIdx)) continue;
               const llvm::SmallVector<std::string>& qKeys =
                  qIdx == 0 ? layout.query0SemanticKeys : layout.query1SemanticKeys;
               if (qKeys.size() == 1 && qKeys[0] == semKey && oldVals.size() == 1) {
                  probeGatherRemapFrom = oldVals[0];
                  if (!assignedMembers.contains(oldVals[0])) reused = oldVals[0];
                  break;
               }
            }
         }
      }
      if (reused && assignedMembers.contains(*reused)) reused = std::nullopt;
      subop::Member consumerMem = ensureConsumerMember(reused, producerSlotTy);
      if (probeGatherRemapFrom) {
         probeGatherMemberRemap[*probeGatherRemapFrom] = consumerMem;
      } else if (reused && consumerHivBeforeAlign && *reused != consumerMem) {
         probeGatherMemberRemap[*reused] = consumerMem;
      }
      if (producerMem != consumerMem) {
         probeGatherMemberRemap[producerMem] = consumerMem;
      }
      assignedBySemantic[semKey] = consumerMem;
      assignedMembers.insert(consumerMem);
      producerPayloadToConsumer[layout.payloadMembers[i]] = consumerMem;
   }

   llvm::SmallVector<subop::Member> valueMembers;
   valueMembers.reserve(producerHiv.getValueMembers().getMembers().size());
   for (subop::Member m : producerHiv.getValueMembers().getMembers()) {
      if (auto it = producerPayloadToConsumer.find(m); it != producerPayloadToConsumer.end()) {
         valueMembers.push_back(it->second);
      } else if (consumerHivBeforeAlign) {
         if (auto reused = findConsumerMemberByName(mm.getName(m), consumerHivBeforeAlign)) {
            valueMembers.push_back(
               mm.getOrCreateMemberDirect(mm.getName(*reused), mm.getType(*reused), /*allowTypeUpdate=*/false));
         } else {
            valueMembers.push_back(cloneMemberToContext(m, producerHiv.getContext(), ctx, /*allowMemberTypeUpdate=*/false));
         }
      } else {
         valueMembers.push_back(cloneMemberToContext(m, producerHiv.getContext(), ctx, /*allowMemberTypeUpdate=*/false));
      }
   }

   mlir::MLIRContext* producerCtx = producerHiv.getContext();
   llvm::SmallVector<subop::Member> keyMembers;
   for (subop::Member m : producerHiv.getKeyMembers().getMembers()) {
      keyMembers.push_back(cloneMemberToContext(m, producerCtx, ctx, /*allowMemberTypeUpdate=*/false));
   }
   auto consumerHiv = subop::HashIndexedViewType::get(
      ctx, subop::StateMembersAttr::get(ctx, keyMembers), subop::StateMembersAttr::get(ctx, valueMembers),
      producerHiv.getCompareHashForLookup());
   if (consumerHivBeforeAlign) {
      llvm::ArrayRef<subop::Member> oldVals = consumerHivBeforeAlign.getValueMembers().getMembers();
      llvm::ArrayRef<subop::Member> newVals = consumerHiv.getValueMembers().getMembers();
      if (oldVals.size() == 1 && !newVals.empty() && oldVals[0] != newVals[0]) {
         probeGatherMemberRemap[oldVals[0]] = newVals[0];
      }
   }

   outConsumerLayout = layout;
   outConsumerLayout.probeGatherMemberRemap = std::move(probeGatherMemberRemap);
   outConsumerLayout.producerHiv = consumerHiv;
   outConsumerLayout.payloadMembers.clear();
   outConsumerLayout.payloadColumnTypes.clear();
   outConsumerLayout.payloadSemanticKeys.clear();
   outConsumerLayout.payloadMembers.reserve(layout.payloadMembers.size());
   outConsumerLayout.payloadColumnTypes.reserve(layout.payloadMembers.size());
   outConsumerLayout.payloadSemanticKeys.reserve(layout.payloadMembers.size());
   for (size_t i = 0; i < layout.payloadMembers.size(); ++i) {
      subop::Member m = layout.payloadMembers[i];
      subop::Member consumerMem;
      if (auto it = producerPayloadToConsumer.find(m); it != producerPayloadToConsumer.end()) {
         consumerMem = it->second;
      } else {
         consumerMem = cloneMemberToContext(m, producerCtx, ctx, /*allowMemberTypeUpdate=*/false);
      }
      outConsumerLayout.payloadMembers.push_back(consumerMem);
      outConsumerLayout.payloadColumnTypes.push_back(mm.getType(consumerMem));
      outConsumerLayout.payloadSemanticKeys.push_back(layout.payloadSemanticKeys[i]);
   }
   return consumerHiv;
}

static bool joinMatchPeerExternalFiltersIdenticalImpl(mlir::ModuleOp query0, mlir::ModuleOp query1, mlir::Value hivA,
                                                      mlir::Value hivB, const ModuleReuseInfo& reuse0,
                                                      const ModuleReuseInfo& reuse1) {
   ExternalDatasourceProperty dsA;
   ExternalDatasourceProperty dsB;
   if (!tryGetHivDonorExternalDatasource(query0, hivA, reuse0, dsA) ||
       !tryGetHivDonorExternalDatasource(query1, hivB, reuse1, dsB)) {
      return false;
   }
   if (dsA.tableName != dsB.tableName) return false;
   if (!externalDatasourceFiltersEqual(dsA, dsB)) return false;
   subop::ExecutionStepOp buildA = findJoinBufferBuildStepForHiv(query0, hivA, reuse0);
   subop::ExecutionStepOp buildB = findJoinBufferBuildStepForHiv(query1, hivB, reuse1);
   assert(buildA && buildB && "join external filter equality requires join build steps");
   return residualTableFiltersIdentical(buildA, buildB);
}

} // namespace

bool joinMatchPeerExternalFiltersIdentical(mlir::ModuleOp query0, mlir::ModuleOp query1, mlir::Value hivA,
                                           mlir::Value hivB, const ModuleReuseInfo& reuse0,
                                           const ModuleReuseInfo& reuse1) {
   return joinMatchPeerExternalFiltersIdenticalImpl(query0, query1, hivA, hivB, reuse0, reuse1);
}

double estimateMergedHivExternalFilterRows(mlir::ModuleOp query0, mlir::ModuleOp query1, mlir::Value hivA,
                                           mlir::Value hivB, const ModuleReuseInfo& reuse0,
                                           const ModuleReuseInfo& reuse1,
                                           lingodb::catalog::Catalog& catalog) {
   ExternalDatasourceProperty dsA;
   ExternalDatasourceProperty dsB;
   bool okA = tryGetHivDonorExternalDatasource(query0, hivA, reuse0, dsA);
   bool okB = tryGetHivDonorExternalDatasource(query1, hivB, reuse1, dsB);
   assert(okA && okB && "join superset CE: both HIVs must resolve to donor external datasources");
   (void)okA;
   (void)okB;
   assert(dsA.tableName == dsB.tableName && "join superset CE: matched HIVs must scan the same donor table");
   llvm::SmallVector<ExternalDatasourceProperty, 2> sources{dsA, dsB};
   subop::ExecutionStepOp buildA = findJoinBufferBuildStepForHiv(query0, hivA, reuse0);
   subop::ExecutionStepOp buildB = findJoinBufferBuildStepForHiv(query1, hivB, reuse1);
   assert(buildA && buildB && "join superset CE: both HIVs must have build steps");
   auto residualA = findResidualTableFilterInBuildStep(buildA);
   auto residualB = findResidualTableFilterInBuildStep(buildB);
   llvm::SmallVector<mlir::Operation*, 2> maps{
      residualA ? residualA->predMap.getOperation() : nullptr,
      residualB ? residualB->predMap.getOperation() : nullptr};
   llvm::SmallVector<mlir::Operation*, 2> filters{
      residualA ? residualA->filter.getOperation() : nullptr,
      residualB ? residualB->filter.getOperation() : nullptr};
   return relalg::estimateExternalDatasourceOrRowsFromSample(sources, catalog, maps, filters);
}

bool parseReuseFilterPredSemanticKey(llvm::StringRef semanticKey, unsigned& reuseQueryIndex) {
   size_t sep = semanticKey.find('\x1f');
   if (sep == llvm::StringRef::npos) return false;
   if (semanticKey.take_front(sep) != "reuse_filter_pred") return false;
   return !semanticKey.drop_front(sep + 1).getAsInteger(10, reuseQueryIndex);
}

static bool materializePredMemberFromUpstreamMixedScanList(subop::ExecutionStepOp buildStep,
                                                           llvm::StringRef predMemberName);

ClonedJoinBufferBuildSitesByKey recordClonedJoinBufferBuildSites(mlir::ModuleOp synthetic,
                                                                 llvm::ArrayRef<CacheTarget> targetsInSynthetic,
                                                                 const ModuleReuseInfo& reuseSynthetic) {
   ClonedJoinBufferBuildSitesByKey out;
   for (const CacheTarget& t : targetsInSynthetic) {
      if (!t.state || !mlir::isa<subop::HashIndexedViewType>(t.state.getType())) continue;
      mlir::Value mergedBuf = resolveJoinMergedBuffer(t.state, synthetic, reuseSynthetic);
      subop::ExecutionStepOp buildStep = findBufferBuildStepWithTableMaterialize(mergedBuf, reuseSynthetic);
      if (!buildStep) buildStep = findBufferBuildStepWithTableScan(synthetic);
      if (!buildStep) continue;
      ClonedJoinBufferBuildSite site;
      site.cacheKey = t.cacheKey;
      site.syntheticHiv = t.state;
      site.syntheticMergedBuffer = mergedBuf;
      site.syntheticBuildStep = buildStep;
      out[t.cacheKey] = site;
   }
   return out;
}

void insertSyntheticFilterPredsAfterColumnUnion(
   mlir::ModuleOp synthetic, mlir::ModuleOp query0, mlir::ModuleOp query1,
   llvm::ArrayRef<CrossQueryStateMatchPair> matches, llvm::ArrayRef<CacheTarget> targetsInSynthetic,
   const CachedJoinBufferLayoutsByKey& layoutsByKey,    const ClonedJoinBufferBuildSitesByKey& buildSites) {
   auto reuse0 = collectModuleReuseInfo(query0);
   auto reuse1 = collectModuleReuseInfo(query1);

   llvm::DenseMap<uint64_t, const CrossQueryStateMatchPair*> matchByKey;
   for (const auto& m : matches) {
      if (!m.stateA || !m.stateB) continue;
      matchByKey[m.cacheKey] = &m;
   }

   auto& mm = synthetic.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();

   for (const CacheTarget& t : targetsInSynthetic) {
      auto itL = layoutsByKey.find(t.cacheKey);
      auto itM = matchByKey.find(t.cacheKey);
      auto itB = buildSites.find(t.cacheKey);
      if (itL == layoutsByKey.end() || itM == matchByKey.end() || itB == buildSites.end()) continue;
      const CachedJoinBufferLayout& layout = itL->second;
      const CrossQueryStateMatchPair& match = *itM->second;
      if (!match.enableFilterPredReuse) continue;
      subop::ExecutionStepOp buildStep = itB->second.syntheticBuildStep;
      if (!buildStep) continue;

      llvm::DenseMap<unsigned, mlir::Value> hivByQuery;
      llvm::DenseMap<unsigned, ModuleReuseInfo*> reuseByQuery;
      llvm::DenseMap<unsigned, mlir::ModuleOp> modByQuery;
      llvm::DenseMap<unsigned, subop::ExecutionStepOp> peerBuilds;
      llvm::DenseMap<unsigned, subop::Member> predMembers;
      llvm::SmallVector<subop::Member, 2> unionPredMembers;
      hivByQuery[static_cast<unsigned>(match.queryA)] = resolveCacheTargetStateForReuse(match.stateA, reuse0);
      hivByQuery[static_cast<unsigned>(match.queryB)] = resolveCacheTargetStateForReuse(match.stateB, reuse1);
      reuseByQuery[static_cast<unsigned>(match.queryA)] = &reuse0;
      reuseByQuery[static_cast<unsigned>(match.queryB)] = &reuse1;
      modByQuery[static_cast<unsigned>(match.queryA)] = query0;
      modByQuery[static_cast<unsigned>(match.queryB)] = query1;

      for (size_t i = 0; i < layout.payloadSemanticKeys.size(); ++i) {
         unsigned qIdx = 0;
         if (parseReuseFilterPredUnionSemanticKey(layout.payloadSemanticKeys[i], qIdx)) {
            unionPredMembers.push_back(layout.payloadMembers[i]);
            continue;
         }
         if (!parseReuseFilterPredSemanticKey(layout.payloadSemanticKeys[i], qIdx)) continue;
         auto itH = hivByQuery.find(qIdx);
         auto itR = reuseByQuery.find(qIdx);
         auto itMod = modByQuery.find(qIdx);
         assert(itH != hivByQuery.end() && itR != reuseByQuery.end() && itMod != modByQuery.end() &&
                "insertSyntheticFilterPreds: reuse_query_index must belong to the match");
         predMembers[qIdx] = layout.payloadMembers[i];
         peerBuilds[qIdx] = findJoinBufferBuildStepForHiv(itMod->second, itH->second, *itR->second);
         assert(peerBuilds.lookup(qIdx) &&
                "insertSyntheticFilterPreds: peer join-buffer build step with table scan_refs required");
      }

      llvm::SmallVector<unsigned, 2> predQueryIndices;
      for (auto& kv : predMembers) predQueryIndices.push_back(kv.first);
      llvm::sort(predQueryIndices);

      if (predQueryIndices.size() == 2 && predMembers[predQueryIndices[0]] &&
          predMembers[predQueryIndices[1]] && peerBuilds[predQueryIndices[0]] &&
          peerBuilds[predQueryIndices[1]]) {
         subop::MaterializeOp mat = findJoinBufferMaterializeInStep(buildStep);
         assert(mat && "insertSyntheticFilterPreds: synthetic build step must materialize join buffer");
         llvm::DenseMap<unsigned, llvm::SmallVector<runtime::FilterDescription, 8>> simpleFilters;
         for (unsigned qIdx : predQueryIndices) {
            auto peerFilters = decodeFiltersFromTableScanInExecutionStep(peerBuilds[qIdx]);
            simpleFilters[qIdx] = restrictFiltersToTableScanInExecutionStep(buildStep, peerFilters);
         }
         unsigned q0 = predQueryIndices[0];
         unsigned q1 = predQueryIndices[1];
         if (rewriteSyntheticResidualFiltersAsFilterPreds(buildStep, peerBuilds[q0], peerBuilds[q1], mat,
                                                          q0, q1, predMembers[q0], predMembers[q1],
                                                          simpleFilters[q0], simpleFilters[q1])) {
            for (subop::Member unionPredMember : unionPredMembers) {
               materializeConstantTruePredMemberOnBufferMaterialize(
                  mat, mm.getName(unionPredMember), /*updateStreamOperand=*/true);
            }
            continue;
         }
      }

      for (size_t i = 0; i < layout.payloadSemanticKeys.size(); ++i) {
         unsigned qIdx = 0;
         if (!parseReuseFilterPredSemanticKey(layout.payloadSemanticKeys[i], qIdx)) continue;
         llvm::StringRef predName = mm.getName(layout.payloadMembers[i]);
         subop::ExecutionStepOp peerBuild = peerBuilds.lookup(qIdx);
         assert(peerBuild && "insertSyntheticFilterPreds: missing peer build for filter_pred slot");
         // Per-query predicates come from the peer query's donor table get_external descr, not from HIV
         // writer steps (cache_put / create_hash_indexed_view do not read !subop.table).
         auto peerFilters = decodeFiltersFromTableScanInExecutionStep(peerBuild);
         const bool peerHasValueFilters = llvm::any_of(
            peerFilters, [](const runtime::FilterDescription& f) { return f.op != runtime::FilterOp::NOTNULL; });
         auto filters = restrictFiltersToTableScanInExecutionStep(buildStep, peerFilters);
         if (peerHasValueFilters) {
            assert(!filters.empty() &&
                   "insertSyntheticFilterPreds: peer pushdown filters must map to synthetic donor table scan");
         }
         if (filters.empty() && materializePredMemberFromUpstreamMixedScanList(buildStep, predName)) {
            continue;
         }
         insertWriteSidePredIntoBufferConstructionStepForPredMember(buildStep, filters, predName);
      }
      if (!unionPredMembers.empty()) {
         subop::MaterializeOp mat = findJoinBufferMaterializeInStep(buildStep);
         assert(mat && "insertSyntheticFilterPreds: synthetic build step must materialize join buffer");
         for (subop::Member unionPredMember : unionPredMembers) {
            materializeConstantTruePredMemberOnBufferMaterialize(
               mat, mm.getName(unionPredMember), /*updateStreamOperand=*/true);
         }
      }
   }
}

void insertSyntheticFilterPredsAfterColumnUnionForGroups(
   mlir::ModuleOp synthetic, llvm::ArrayRef<mlir::ModuleOp> queries,
   llvm::ArrayRef<CrossQueryStateMatchGroup> groups,
   llvm::ArrayRef<CacheTarget> targetsInSynthetic,
   const CachedJoinBufferLayoutsByKey& layoutsByKey,
   const ClonedJoinBufferBuildSitesByKey& buildSites) {
   llvm::SmallVector<ModuleReuseInfo, 8> reuseByQuery;
   reuseByQuery.reserve(queries.size());
   for (mlir::ModuleOp query : queries) reuseByQuery.push_back(collectModuleReuseInfo(query));

   llvm::DenseMap<uint64_t, const CrossQueryStateMatchGroup*> groupByKey;
   for (const CrossQueryStateMatchGroup& group : groups) {
      if (group.entries.size() >= 2) groupByKey[group.cacheKey] = &group;
   }

   auto& mm = synthetic.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();

   for (const CacheTarget& t : targetsInSynthetic) {
      auto itL = layoutsByKey.find(t.cacheKey);
      auto itG = groupByKey.find(t.cacheKey);
      auto itB = buildSites.find(t.cacheKey);
      if (itL == layoutsByKey.end() || itG == groupByKey.end() || itB == buildSites.end()) continue;
      const CachedJoinBufferLayout& layout = itL->second;
      const CrossQueryStateMatchGroup& group = *itG->second;
      if (!group.enableFilterPredReuse) continue;
      subop::ExecutionStepOp buildStep = itB->second.syntheticBuildStep;
      if (!buildStep) continue;

      llvm::DenseMap<unsigned, mlir::Value> hivByQuery;
      llvm::DenseMap<unsigned, ModuleReuseInfo*> reuseInfoByQuery;
      llvm::DenseMap<unsigned, mlir::ModuleOp> moduleByQuery;
      for (const CrossQueryStateMatchEntry& entry : group.entries) {
         if (entry.query < 0 || static_cast<size_t>(entry.query) >= queries.size() || !entry.state) continue;
         auto qIdx = static_cast<unsigned>(entry.query);
         ModuleReuseInfo& reuse = reuseByQuery[entry.query];
         hivByQuery[qIdx] = resolveCacheTargetStateForReuse(entry.state, reuse);
         reuseInfoByQuery[qIdx] = &reuse;
         moduleByQuery[qIdx] = queries[entry.query];
      }

      llvm::DenseMap<unsigned, subop::ExecutionStepOp> peerBuilds;
      llvm::DenseMap<unsigned, subop::Member> predMembers;
      llvm::SmallVector<subop::Member, 2> unionPredMembers;
      for (size_t i = 0; i < layout.payloadSemanticKeys.size(); ++i) {
         unsigned qIdx = 0;
         if (parseReuseFilterPredUnionSemanticKey(layout.payloadSemanticKeys[i], qIdx)) {
            unionPredMembers.push_back(layout.payloadMembers[i]);
            continue;
         }
         if (!parseReuseFilterPredSemanticKey(layout.payloadSemanticKeys[i], qIdx)) continue;
         auto itH = hivByQuery.find(qIdx);
         auto itR = reuseInfoByQuery.find(qIdx);
         auto itM = moduleByQuery.find(qIdx);
         assert(itH != hivByQuery.end() && itR != reuseInfoByQuery.end() && itM != moduleByQuery.end() &&
                "insertSyntheticFilterPredsForGroups: filter_pred slot must belong to the match group");
         predMembers[qIdx] = layout.payloadMembers[i];
         peerBuilds[qIdx] = findJoinBufferBuildStepForHiv(itM->second, itH->second, *itR->second);
         assert(peerBuilds.lookup(qIdx) &&
                "insertSyntheticFilterPredsForGroups: peer join-buffer build step with table scan_refs required");
      }

      llvm::SmallVector<unsigned, 8> predQueryIndices;
      for (auto& kv : predMembers) predQueryIndices.push_back(kv.first);
      llvm::sort(predQueryIndices);

      if (predQueryIndices.size() == 2 && predMembers[predQueryIndices[0]] &&
          predMembers[predQueryIndices[1]] && peerBuilds[predQueryIndices[0]] &&
          peerBuilds[predQueryIndices[1]]) {
         subop::MaterializeOp mat = findJoinBufferMaterializeInStep(buildStep);
         assert(mat && "insertSyntheticFilterPredsForGroups: synthetic build step must materialize join buffer");
         llvm::DenseMap<unsigned, llvm::SmallVector<runtime::FilterDescription, 8>> simpleFilters;
         for (unsigned qIdx : predQueryIndices) {
            auto peerFilters = decodeFiltersFromTableScanInExecutionStep(peerBuilds[qIdx]);
            simpleFilters[qIdx] = restrictFiltersToTableScanInExecutionStep(buildStep, peerFilters);
         }
         unsigned q0 = predQueryIndices[0];
         unsigned q1 = predQueryIndices[1];
         if (rewriteSyntheticResidualFiltersAsFilterPreds(buildStep, peerBuilds[q0], peerBuilds[q1], mat,
                                                          q0, q1, predMembers[q0], predMembers[q1],
                                                          simpleFilters[q0], simpleFilters[q1])) {
            for (subop::Member unionPredMember : unionPredMembers) {
               materializeConstantTruePredMemberOnBufferMaterialize(
                  mat, mm.getName(unionPredMember), /*updateStreamOperand=*/true);
            }
            continue;
         }
      }

      for (unsigned qIdx : predQueryIndices) {
         llvm::StringRef predName = mm.getName(predMembers[qIdx]);
         subop::ExecutionStepOp peerBuild = peerBuilds.lookup(qIdx);
         assert(peerBuild && "insertSyntheticFilterPredsForGroups: missing peer build for filter_pred slot");
         auto peerFilters = decodeFiltersFromTableScanInExecutionStep(peerBuild);
         const bool peerHasValueFilters = llvm::any_of(
            peerFilters, [](const runtime::FilterDescription& f) { return f.op != runtime::FilterOp::NOTNULL; });
         auto filters = restrictFiltersToTableScanInExecutionStep(buildStep, peerFilters);
         if (peerHasValueFilters) {
            assert(!filters.empty() &&
                   "insertSyntheticFilterPredsForGroups: peer pushdown filters must map to synthetic donor table scan");
         }
         if (filters.empty() && materializePredMemberFromUpstreamMixedScanList(buildStep, predName)) {
            continue;
         }
         insertWriteSidePredIntoBufferConstructionStepForPredMember(buildStep, filters, predName);
      }
      if (!unionPredMembers.empty()) {
         subop::MaterializeOp mat = findJoinBufferMaterializeInStep(buildStep);
         assert(mat && "insertSyntheticFilterPredsForGroups: synthetic build step must materialize join buffer");
         for (subop::Member unionPredMember : unionPredMembers) {
            materializeConstantTruePredMemberOnBufferMaterialize(
               mat, mm.getName(unionPredMember), /*updateStreamOperand=*/true);
         }
      }
   }
}

static bool materializePredMemberFromUpstreamMixedScanList(subop::ExecutionStepOp buildStep,
                                                           llvm::StringRef predMemberName) {
   subop::MaterializeOp matOp = findJoinBufferMaterializeInStep(buildStep);
   if (!matOp) return false;
   auto* ctx = buildStep.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

   subop::BufferType bufTy = getInnerBufferTypeForMaterializeState(matOp.getState().getType());
   if (!bufTy) return false;
   subop::Member outPredMember;
   for (subop::Member m : bufTy.getMembers().getMembers()) {
      if (mm.getName(m) == predMemberName) {
         outPredMember = m;
         break;
      }
   }
   if (!outPredMember) return false;

   for (auto& [member, colRef] : matOp.getMapping().getMapping()) {
      if (member == outPredMember) return true;
   }

   subop::ScanListOp predScanList;
   subop::Member upstreamPredMember;
   buildStep.walk([&](subop::ScanListOp scanList) {
      if (predScanList) return;
      auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(scanList.getElem().getColumn().type);
      if (!ler) return;
      auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(ler.getState());
      if (!mixed) return;
      for (subop::Member m : mixed.getValueMembers().getMembers()) {
         if (mm.getName(m) == predMemberName) {
            predScanList = scanList;
            upstreamPredMember = m;
            return;
         }
      }
   });
   if (!predScanList || !upstreamPredMember) return false;

   mlir::Value stream = matOp.getStream();
   mlir::Operation* anchor = stream.getDefiningOp();
   mlir::OpBuilder b(matOp);
   if (anchor) b.setInsertionPointAfter(anchor);
   else b.setInsertionPoint(matOp);

   std::string scopeSeed = "reuse_upstream_pred";
   if (auto slot = parseFilterPredMemberSlot(predMemberName))
      scopeSeed = ("reuse_upstream_pred$" + llvm::Twine(*slot)).str();
   tuples::ColumnDefAttr predDef = cm.createDef(cm.getUniqueScope(scopeSeed), "filter_pred");
   predDef.getColumn().type = mlir::IntegerType::get(ctx, 1);
   auto mapping = subop::ColumnDefMemberMappingAttr::get(
      ctx, llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>>{{upstreamPredMember, predDef}});
   auto gather = b.create<subop::GatherOp>(matOp.getLoc(), stream.getType(), stream,
                                           cm.createRef(&predScanList.getElem().getColumn()), mapping);
   matOp->setOperand(0, gather.getRes());

   llvm::SmallVector<subop::RefMappingPairT> pairs;
   for (auto& pr : matOp.getMapping().getMapping()) {
      if (pr.first != outPredMember) pairs.push_back(pr);
   }
   pairs.push_back({outPredMember, cm.createRef(&predDef.getColumn())});
   matOp.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, pairs));
   return true;
}

void syncProbeGatherMappingsInModule(mlir::ModuleOp module) {
   auto& cm = module.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   llvm::DenseSet<const void*> seenHiv;
   module.walk([&](subop::LookupOp op) {
      auto hivTy = asHashIndexedViewLayoutType(op.getState().getType());
      if (!hivTy) return;
      if (!seenHiv.insert(hivTy.getAsOpaquePointer()).second) return;
      auto listRef = op.getRef();
      auto [listScope, listLeaf] = cm.getName(&listRef.getColumn());
      (void)listLeaf;
      propagateJoinSupersetColumnAttrs(module, nullptr, hivTy, true, listScope);
   });
   syncProbeListCarriersInClosure(module, nullptr);
}

void alignConsumerModulesToCachedJoinLayout(mlir::ModuleOp consumer, const CachedJoinBufferLayout& layout,
                                            std::optional<uint64_t> cacheKey,
                                            std::optional<unsigned> consumerReuseQueryIndex,
                                            llvm::SmallVectorImpl<ConsumerCacheGetProbeClosure>* outProbeClosures) {
   if (!layout.producerHiv) return;

   subop::HashIndexedViewType producerHiv = layout.producerHiv;
   llvm::SmallVector<ConsumerCacheGetProbeClosure, 4> perCacheGet;

   consumer.walk([&](subop::CacheGetOp get) {
      if (cacheKey && static_cast<uint64_t>(get.getKey()) != *cacheKey) return;

      auto consumerHivBeforeAlign = asHashIndexedViewLayoutType(get.getResult().getType());
      CachedJoinBufferLayout consumerLayout;
      subop::HashIndexedViewType alignedHiv = buildConsumerAlignedHivType(
         consumer, producerHiv, layout, consumerHivBeforeAlign, consumerReuseQueryIndex, consumerLayout);
      get.getResult().setType(alignedHiv);
      const uint64_t key = static_cast<uint64_t>(get.getKey());
      ConsumerCacheGetProbeClosure probe = buildConsumerCacheGetProbeClosure(
         consumer, get.getResult(), alignedHiv, consumerHivBeforeAlign, consumerLayout, key,
         consumerReuseQueryIndex);
      if (consumerHivBeforeAlign && consumerHivBeforeAlign != alignedHiv) {
         ProbeAlignDebugCtx refreshDbg;
         refreshDbg.cacheKey = probe.cacheKey;
         refreshDbg.passName = "refresh-stale";
         refreshStaleEmbeddedHivCarriers(consumer, alignedHiv, consumerHivBeforeAlign, probe.ssaClosure, &refreshDbg);
      }
      debugProbeAlign(nullptr, [&](llvm::raw_ostream& os) {
         os << "cache_get closure cache_key=" << probe.cacheKey.value_or(0)
            << " closure_size=" << probe.ssaClosure.size()
            << " aligned_hiv=" << mlirTypeToString(probe.alignedHiv) << " lookup_scopes={";
         bool first = true;
         for (const auto& s : probe.probeLookupScopes) {
            if (!first) os << ',';
            os << s.first();
            first = false;
         }
         os << '}';
      });
      perCacheGet.push_back(std::move(probe));
   });

   if (perCacheGet.empty()) return;

   llvm::DenseSet<void*> unionClosure;
   for (ConsumerCacheGetProbeClosure& probe : perCacheGet) {
      for (void* p : probe.ssaClosure) unionClosure.insert(p);

      alignScanListForProbeClosure(consumer, probe, "align");
      remapAlignedHivClosureGathers(consumer, &probe.ssaClosure, probe.alignedHiv, probe.consumerLayout,
                                    &probe.probeLookupScopes, probe.cacheKey, "remap");
   }

   synchronizeExecutionStepPortTypes(consumer, &unionClosure);

   for (ConsumerCacheGetProbeClosure& probe : perCacheGet) {
      alignScanListForProbeClosure(consumer, probe, "align-post-sync");
   }

   syncProbeListCarriersInClosure(consumer, &unionClosure);
   synchronizeExecutionStepPortTypes(consumer, nullptr);

   if (outProbeClosures) {
      const size_t appendStart = outProbeClosures->size();
      for (ConsumerCacheGetProbeClosure& probe : perCacheGet) {
         outProbeClosures->push_back(std::move(probe));
      }
      for (size_t i = appendStart; i < outProbeClosures->size(); ++i) {
         for (void* p : unionClosure) (*outProbeClosures)[i].ssaClosure.insert(p);
      }
   }
}

static void refreshProbeClosureFromCacheGetRoot(mlir::ModuleOp consumer, ConsumerCacheGetProbeClosure& probe) {
   if (!probe.cacheGetRoot || !probe.alignedHiv) return;
   refreshConsumerCacheGetProbeClosure(consumer, probe, "probe_pred_retraverse");
}

static std::optional<subop::HashIndexedViewType> hashIndexedViewFromScanListCarrier(
   subop::ScanListOp scanList) {
   if (auto listTy = mlir::dyn_cast<subop::ListType>(scanList.getList().getType())) {
      if (auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(listTy.getT())) {
         if (auto st = mlir::dyn_cast<subop::HashIndexedViewType>(ler.getState())) return st;
      }
   }
   if (auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(scanList.getElem().getColumn().type)) {
      if (auto st = mlir::dyn_cast<subop::HashIndexedViewType>(ler.getState())) return st;
   }
   return std::nullopt;
}

static bool scanListTargetsAlignedHivPredSlot(mlir::MLIRContext* ctx, subop::ScanListOp scanList,
                                               subop::HashIndexedViewType alignedHiv, subop::Member predMember,
                                               subop::HashIndexedViewType consumerHivBeforeAlign) {
   setValueCarrierType(scanList.getList(), alignedHiv, consumerHivBeforeAlign);
   std::optional<subop::HashIndexedViewType> stOpt = hashIndexedViewFromScanListCarrier(scanList);
   if (!stOpt) return false;
   subop::HashIndexedViewType st = *stOpt;
   if (st.getCompareHashForLookup() != alignedHiv.getCompareHashForLookup()) return false;
   if (st.getKeyMembers().getMembers().size() != alignedHiv.getKeyMembers().getMembers().size()) return false;
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   llvm::StringRef predName = mm.getName(predMember);
   return valueMembersContainMemberNamed(ctx, st.getValueMembers(), predName) &&
          valueMembersContainMemberNamed(ctx, alignedHiv.getValueMembers(), predName);
}

static void ensureReuseFilterPredColumn(JoinBufferUnionPlan& plan, unsigned queryIndex, mlir::MLIRContext* ctx) {
   std::string semKey = reuseFilterPredSemanticKey(queryIndex);
   for (const PayloadColumnSpec& spec : plan.payloadColumns) {
      if (spec.semanticKey == semKey) return;
   }
   PayloadColumnSpec predSpec;
   predSpec.scope = kReuseFilterPredScope.str();
   predSpec.leaf = llvm::Twine(queryIndex).str();
   predSpec.colType = mlir::IntegerType::get(ctx, 1);
   predSpec.semanticKey = semKey;
   predSpec.semanticHash = payloadSyntheticColumnHash(predSpec.scope, predSpec.leaf, predSpec.colType);
   plan.payloadColumns.push_back(std::move(predSpec));
   llvm::sort(plan.payloadColumns, [](const PayloadColumnSpec& a, const PayloadColumnSpec& b) {
      if (a.isJoinKey != b.isJoinKey) return a.isJoinKey > b.isJoinKey;
      if (a.semanticHash != b.semanticHash) return a.semanticHash < b.semanticHash;
      return a.semanticKey < b.semanticKey;
   });
   plan.payloadMemberTypes.clear();
   plan.payloadMemberTypes.reserve(plan.payloadColumns.size());
   for (const PayloadColumnSpec& spec : plan.payloadColumns) plan.payloadMemberTypes.push_back(spec.colType);
}

void extendSyntheticJoinBuffersToColumnUnionForGroups(
   mlir::ModuleOp synthetic, llvm::ArrayRef<mlir::ModuleOp> queries,
   llvm::ArrayRef<CrossQueryStateMatchGroup> groups,
   llvm::ArrayRef<CacheTarget> targetsInSynthetic,
   CachedJoinBufferLayoutsByKey* outLayouts) {
   if (groups.empty() || targetsInSynthetic.empty()) return;

   llvm::SmallVector<ModuleReuseInfo, 8> reuseByQuery;
   reuseByQuery.reserve(queries.size());
   for (mlir::ModuleOp query : queries) reuseByQuery.push_back(collectModuleReuseInfo(query));

   llvm::DenseMap<uint64_t, const CrossQueryStateMatchGroup*> groupByKey;
   for (const CrossQueryStateMatchGroup& group : groups) {
      if (group.entries.size() >= 2) groupByKey[group.cacheKey] = &group;
   }

   for (const CacheTarget& t : targetsInSynthetic) {
      auto itG = groupByKey.find(t.cacheKey);
      if (itG == groupByKey.end()) continue;
      const CrossQueryStateMatchGroup& group = *itG->second;
      if (!group.requiresJoinLayoutUnion) continue;
      if (!asHashIndexedViewLayoutType(t.state.getType())) continue;

      const CrossQueryStateMatchEntry* donor = nullptr;
      const CrossQueryStateMatchEntry* firstPeer = nullptr;
      for (const CrossQueryStateMatchEntry& entry : group.entries) {
         if (entry.query < 0 || static_cast<size_t>(entry.query) >= queries.size() || !entry.state) continue;
         if (!donor || entry.query < donor->query) donor = &entry;
      }
      if (!donor) continue;
      for (const CrossQueryStateMatchEntry& entry : group.entries) {
         if (&entry == donor) continue;
         if (entry.query < 0 || static_cast<size_t>(entry.query) >= queries.size() || !entry.state) continue;
         firstPeer = &entry;
         break;
      }
      if (!firstPeer) continue;

      const ModuleReuseInfo& reuseDonor = reuseByQuery[donor->query];
      const ModuleReuseInfo& reusePeer = reuseByQuery[firstPeer->query];
      mlir::Value hivDonor = resolveCacheTargetStateForReuse(donor->state, reuseDonor);
      mlir::Value hivPeer = resolveCacheTargetStateForReuse(firstPeer->state, reusePeer);
      if (!asHashIndexedViewLayoutType(hivDonor.getType()) ||
          !asHashIndexedViewLayoutType(hivPeer.getType())) {
         continue;
      }
      if (!group.enableFilterPredReuse && hashIndexedViewLayoutsArePhysicallyCompatible(hivDonor, hivPeer)) {
         continue;
      }
      if (!findStrictJoinBufferBuildStepForHiv(queries[donor->query], hivDonor, reuseDonor) ||
          !findStrictJoinBufferBuildStepForHiv(queries[firstPeer->query], hivPeer, reusePeer)) {
         continue;
      }

      JoinBufferUnionPlan plan =
         buildUnionPlan(hivDonor, hivPeer, reuseDonor, reusePeer, queries[donor->query],
                        queries[firstPeer->query], static_cast<unsigned>(donor->query),
                        static_cast<unsigned>(firstPeer->query), group.enableFilterPredReuse);
      if (group.enableFilterPredReuse) {
         llvm::SmallVector<unsigned, 8> queryIndices;
         for (const CrossQueryStateMatchEntry& entry : group.entries) {
            if (entry.query < 0 || static_cast<size_t>(entry.query) >= queries.size()) continue;
            auto qIdx = static_cast<unsigned>(entry.query);
            queryIndices.push_back(qIdx);
            ensureReuseFilterPredColumn(plan, qIdx, synthetic.getContext());
         }
         unsigned unionSlot = filterPredUnionSlotForQueryIndices(queryIndices);
         if (unionSlot < 8) {
            ensureFilterPredUnionColumn(plan, unionSlot, synthetic.getContext());
         }
      }
      if (plan.payloadColumns.empty()) continue;

      llvm::SmallVector<std::pair<mlir::ModuleOp, mlir::Value>, 8> extraPeerHivs;
      llvm::SmallVector<const ModuleReuseInfo*, 8> extraPeerReuses;
      for (const CrossQueryStateMatchEntry& entry : group.entries) {
         if (entry.query == donor->query || entry.query == firstPeer->query) continue;
         if (entry.query < 0 || static_cast<size_t>(entry.query) >= queries.size() || !entry.state) continue;
         const ModuleReuseInfo& reuseExtra = reuseByQuery[entry.query];
         mlir::Value hivExtra = resolveCacheTargetStateForReuse(entry.state, reuseExtra);
         if (!asHashIndexedViewLayoutType(hivExtra.getType())) continue;
         if (!findStrictJoinBufferBuildStepForHiv(queries[entry.query], hivExtra, reuseExtra)) continue;
         extraPeerHivs.push_back({queries[entry.query], hivExtra});
         extraPeerReuses.push_back(&reuseExtra);
      }

      auto reuseSynthetic = collectModuleReuseInfo(synthetic);
      applyUnionPlanToSyntheticHiv(synthetic, t.state, plan, reuseSynthetic, queries[donor->query],
                                   hivDonor, reuseDonor, queries[firstPeer->query], hivPeer, reusePeer,
                                   extraPeerHivs, extraPeerReuses);

      if (outLayouts) {
         mlir::Value canonHiv = t.state;
         synthetic.walk([&](subop::CachePutOp put) {
            if (put.getKey() != t.cacheKey) return;
            canonHiv = put.getState();
         });
         if (auto hivTy = asHashIndexedViewLayoutType(canonHiv.getType())) {
            (*outLayouts)[t.cacheKey] = layoutFromUnionPlan(hivTy, plan);
         }
      }
   }
}

void extendSyntheticJoinBuffersToColumnUnion(mlir::ModuleOp synthetic, mlir::ModuleOp query0, mlir::ModuleOp query1,
                                             llvm::ArrayRef<CrossQueryStateMatchPair> matches,
                                             llvm::ArrayRef<CacheTarget> targetsInSynthetic,
                                             const mlir::IRMapping& donorToSynthetic,
                                             CachedJoinBufferLayoutsByKey* outLayouts) {
   (void)donorToSynthetic;
   if (matches.empty() || targetsInSynthetic.empty()) return;

   auto reuse0 = collectModuleReuseInfo(query0);
   auto reuse1 = collectModuleReuseInfo(query1);

   llvm::DenseMap<uint64_t, const CrossQueryStateMatchPair*> matchByKey;
   for (const auto& m : matches) {
      if (!m.stateA || !m.stateB) continue;
      matchByKey[m.cacheKey] = &m;
   }

   for (const CacheTarget& t : targetsInSynthetic) {
      auto itM = matchByKey.find(t.cacheKey);
      if (itM == matchByKey.end()) continue;
      const CrossQueryStateMatchPair& m = *itM->second;
      const bool enableFilterPredReuse = m.enableFilterPredReuse;

      mlir::Value hivA = resolveCacheTargetStateForReuse(m.stateA, reuse0);
      mlir::Value hivB = resolveCacheTargetStateForReuse(m.stateB, reuse1);
      if (!asHashIndexedViewLayoutType(hivA.getType()) ||
          !asHashIndexedViewLayoutType(hivB.getType())) {
         continue;
      }
      if (!enableFilterPredReuse && hashIndexedViewLayoutsArePhysicallyCompatible(hivA, hivB)) {
         continue;
      }
      if (!findStrictJoinBufferBuildStepForHiv(query0, hivA, reuse0) ||
          !findStrictJoinBufferBuildStepForHiv(query1, hivB, reuse1)) {
         continue;
      }

      JoinBufferUnionPlan plan =
         buildUnionPlan(hivA, hivB, reuse0, reuse1, query0, query1,
                        static_cast<unsigned>(m.queryA), static_cast<unsigned>(m.queryB),
                        enableFilterPredReuse);
      if (plan.payloadColumns.empty()) continue;

      unsigned reuseFilterPredSlots = 0;
      for (const PayloadColumnSpec& spec : plan.payloadColumns) {
         unsigned qIdx = 0;
         if (parseReuseFilterPredSemanticKey(spec.semanticKey, qIdx)) ++reuseFilterPredSlots;
      }
      auto fpA = reuse0.joinBuildStoredValueMembersByState.find(hivA);
      auto fpB = reuse1.joinBuildStoredValueMembersByState.find(hivB);
      if (fpA != reuse0.joinBuildStoredValueMembersByState.end() &&
          fpB != reuse1.joinBuildStoredValueMembersByState.end() && fpA->second == fpB->second &&
          reuseFilterPredSlots < 2) {
         llvm::StringSet<> nonPredSemanticKeys;
         for (const PayloadColumnSpec& spec : plan.payloadColumns) {
            unsigned qIdx = 0;
            if (parseFilterPredLayoutSemanticKey(spec.semanticKey, qIdx)) continue;
            nonPredSemanticKeys.insert(spec.semanticKey);
         }
         // Same stored-value fingerprint can still hide cross-query payload (e.g. s_phone vs s_comment).
         if (nonPredSemanticKeys.size() <= 2) continue;
      }

      mlir::Value synthHiv = t.state;
      if (!asHashIndexedViewLayoutType(synthHiv.getType())) continue;

      auto reuseSynthetic = collectModuleReuseInfo(synthetic);
      applyUnionPlanToSyntheticHiv(synthetic, synthHiv, plan, reuseSynthetic, query0, hivA, reuse0, query1, hivB,
                                   reuse1);

      if (outLayouts) {
         mlir::Value canonHiv = synthHiv;
         synthetic.walk([&](subop::CachePutOp put) {
            if (put.getKey() != t.cacheKey) return;
            canonHiv = put.getState();
         });
         if (auto hivTy = asHashIndexedViewLayoutType(canonHiv.getType())) {
            (*outLayouts)[t.cacheKey] = layoutFromUnionPlan(hivTy, plan);
         }
      }
   }
}

void extendSyntheticAggregateHashTablesToPayloadUnion(mlir::ModuleOp synthetic, mlir::ModuleOp query0,
                                                      mlir::ModuleOp query1,
                                                      llvm::ArrayRef<CrossQueryStateMatchPair> matches,
                                                      llvm::ArrayRef<CacheTarget> targetsInSynthetic,
                                                      const mlir::IRMapping& donorToSynthetic,
                                                      CachedAggregateLayoutsByKey* outLayouts) {
   (void)donorToSynthetic;
   if (matches.empty() || targetsInSynthetic.empty()) return;

   llvm::DenseMap<uint64_t, const CrossQueryStateMatchPair*> matchByKey;
   for (const auto& m : matches) {
      if (!m.stateA || !m.stateB) continue;
      matchByKey[m.cacheKey] = &m;
   }

   for (const CacheTarget& t : targetsInSynthetic) {
      auto itM = matchByKey.find(t.cacheKey);
      if (itM == matchByKey.end()) continue;
      const CrossQueryStateMatchPair& m = *itM->second;
      if (!mlir::isa<subop::PreAggrHtType>(t.state.getType())) continue;
      assert(mlir::isa<subop::PreAggrHtType>(m.stateA.getType()) &&
             mlir::isa<subop::PreAggrHtType>(m.stateB.getType()) &&
             "aggregate union match must pair optimistic_ht states");

      CachedAggregateLayout layout = buildAggregateUnionLayout(m.stateA, m.stateB, query0, query1);
      layout.producerHt = mlir::cast<subop::PreAggrHtType>(t.state.getType());
      ModuleReuseInfo reuseSynthetic = collectModuleReuseInfo(synthetic);
      ModuleReuseInfo reusePeer = collectModuleReuseInfo(query1);
      applySyntheticAggregatePayloadUnion(synthetic, t.state, layout, query1, m.stateB, reuseSynthetic, reusePeer);
      if (outLayouts) (*outLayouts)[t.cacheKey] = std::move(layout);
   }
}

void alignConsumerModulesToCachedAggregateLayout(mlir::ModuleOp consumer, const CachedAggregateLayout& layout,
                                                 std::optional<uint64_t> cacheKey,
                                                 std::optional<unsigned> consumerReuseQueryIndex) {
   if (!consumerReuseQueryIndex) return;
   auto* ctx = consumer.getContext();
   ConsumerAggregateAlignment alignment =
      buildConsumerAggregateAlignment(layout, *consumerReuseQueryIndex, ctx);
   subop::PreAggrHtType alignedHt = alignment.ht;
   subop::PreAggrHTEntryRefType alignedEntryRef = subop::PreAggrHTEntryRefType::get(ctx, alignedHt);
   const llvm::DenseMap<subop::Member, subop::Member>& memberRemap = alignment.memberRemap;

   consumer.walk([&](subop::CacheGetOp get) {
      if (cacheKey && static_cast<uint64_t>(get.getKey()) != *cacheKey) return;
      auto oldHt = mlir::dyn_cast<subop::PreAggrHtType>(get.getResult().getType());
      if (!oldHt) return;
      alignment.keyMemberRemap = buildAggregateKeyMemberRemap(oldHt, alignedHt);
      get.getResult().setType(alignedHt);

      llvm::DenseSet<void*> closure;
      closure.insert(get.getResult().getAsOpaquePointer());
      expandClosureThroughExecutionStepPorts(consumer, closure);
      for (;;) {
         size_t before = closure.size();
         expandClosureThroughExecutionStepPorts(consumer, closure);
         if (closure.size() == before) break;
      }

      auto setAggStateType = [&](mlir::Value v) {
         if (!opaqueClosureContains(closure, v)) return;
         if (mlir::isa<subop::PreAggrHtType>(v.getType())) v.setType(alignedHt);
      };
      consumer.walk([&](mlir::Operation* op) {
         for (mlir::Value operand : op->getOperands()) setAggStateType(operand);
         for (mlir::Value result : op->getResults()) setAggStateType(result);
      });
      consumer.walk([&](mlir::Operation* op) {
         for (mlir::Region& region : op->getRegions())
            for (mlir::Block& block : region)
               for (mlir::BlockArgument arg : block.getArguments()) setAggStateType(arg);
      });

      consumer.walk([&](subop::ScanRefsOp scan) {
         if (!opaqueClosureContains(closure, scan.getState())) return;
         if (!mlir::isa<subop::PreAggrHtType>(scan.getState().getType())) return;
         auto ref = scan.getRef();
         ref.getColumn().type = alignedEntryRef;
         scan.setRefAttr(ref);
      });
      consumer.walk([&](subop::GatherOp gather) {
         auto refTy = mlir::dyn_cast<subop::PreAggrHTEntryRefType>(gather.getRef().getColumn().type);
         if (!refTy) return;
         if (refTy.getHashMap() != oldHt && refTy.getHashMap() != alignedHt) return;
         auto ref = gather.getRef();
         ref.getColumn().type = alignedEntryRef;
         gather.setRefAttr(ref);
         llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> pairs;
         bool changed = false;
         for (auto [member, col] : gather.getMapping().getMapping()) {
            subop::Member outMember = member;
            if (auto it = alignment.keyMemberRemap.find(member); it != alignment.keyMemberRemap.end()) {
               outMember = it->second;
               changed = true;
            } else if (auto it = memberRemap.find(member); it != memberRemap.end()) {
               outMember = it->second;
               changed = true;
            }
            pairs.push_back({outMember, col});
         }
         if (changed) gather.setMappingAttr(subop::ColumnDefMemberMappingAttr::get(ctx, pairs));
      });
      synchronizeExecutionStepPortTypes(consumer, &closure);
   });
}

void syncLookupCarrierAttrsFromState(mlir::ModuleOp module) {
   ModuleReuseInfo reuse = collectModuleReuseInfo(module);
   llvm::SmallVector<mlir::Value, 4> roots;
   module.walk([&](subop::CacheGetOp get) { roots.push_back(get.getResult()); });
   if (!roots.empty()) {
      JoinBufferHivSsaClosure joinClosure = computeJoinBufferHivSsaClosure(roots, reuse);
      propagateJoinSupersetColumnAttrsForClosure(module, joinClosure.opaque);
   }
}

void finalizeConsumerCachedJoinProbeColumnAttrs(mlir::ModuleOp consumer, ConsumerCacheGetProbeClosure& probe) {
   if (!probe.alignedHiv) return;

   alignScanListForProbeClosure(consumer, probe, "finalize");
   remapAlignedHivClosureGathers(consumer, &probe.ssaClosure, probe.alignedHiv, probe.consumerLayout,
                                 &probe.probeLookupScopes, probe.cacheKey, "finalize-remap");
   syncProbeListCarriersInClosure(consumer, &probe.ssaClosure);
}

void resyncConsumerCachedHivCarrierTypesFromCacheGet(mlir::ModuleOp consumer,
                                                     std::optional<uint64_t> cacheKey) {
   auto& mm = consumer.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   consumer.walk([&](subop::CacheGetOp get) {
      if (cacheKey && static_cast<uint64_t>(get.getKey()) != *cacheKey) return;
      auto canonicalHiv = asHashIndexedViewLayoutType(get.getResult().getType());
      if (!canonicalHiv) return;

      CachedJoinBufferLayout consumerLayout;
      consumerLayout.producerHiv = canonicalHiv;
      for (subop::Member m : canonicalHiv.getValueMembers().getMembers()) {
         consumerLayout.payloadMembers.push_back(m);
         consumerLayout.payloadColumnTypes.push_back(mm.getType(m));
         consumerLayout.payloadSemanticKeys.push_back(std::string(mm.getName(m)));
      }

      ConsumerCachedHivSites sites;
      // Only rewrite carriers that already embed this cache_get HIV (not other local join buffers).
      traverseConsumerHivUsesFromRoot(get.getResult(), canonicalHiv, canonicalHiv, consumerLayout, sites,
                                      /*dbg=*/nullptr);
      expandConsumerProbeClosureThroughPorts(consumer, sites.ssaClosure);
      synchronizeExecutionStepPortTypes(consumer, &sites.ssaClosure);
      propagateJoinSupersetColumnAttrsForClosure(consumer, sites.ssaClosure);
      synchronizeExecutionStepPortTypes(consumer, nullptr);
   });
}

void applyProbePredFiltersForConsumerClosures(mlir::ModuleOp consumer,
                                              llvm::MutableArrayRef<ConsumerCacheGetProbeClosure> probeClosures) {
   auto* ctx = consumer.getContext();
   for (ConsumerCacheGetProbeClosure& probe : probeClosures) {
      if (!probe.alignedHiv) continue;
      refreshProbeClosureFromCacheGetRoot(consumer, probe);
      alignScanListForProbeClosure(consumer, probe, "probe_pred_align");

      subop::Member predMember;
      if (probe.consumerReuseQueryIndex) {
         predMember = makeOrGetPredMemberForSlot(ctx, *probe.consumerReuseQueryIndex);
      } else if (auto found = findFilterPredMemberOnHashIndexedView(probe.alignedHiv)) {
         predMember = *found;
      } else {
         continue;
      }

      ProbeAlignDebugCtx dbg;
      dbg.cacheKey = probe.cacheKey;
      dbg.passName = "probe_pred_filter";
      llvm::DenseSet<subop::ScanListOp> seen;
      auto tryInsertOnScanList = [&](subop::ScanListOp scanList) {
         if (!seen.insert(scanList).second) return;
         const bool fromCacheGetTraverse = llvm::is_contained(probe.scanListsFromTraverse, scanList);
         if (!fromCacheGetTraverse && !opaqueClosureContains(probe.ssaClosure, scanList.getList())) return;
         if (!scanListTargetsAlignedHivPredSlot(ctx, scanList, probe.alignedHiv, predMember,
                                                probe.consumerHivBeforeAlign)) {
            return;
         }
         if (auto step = scanList->getParentOfType<subop::ExecutionStepOp>()) {
            insertHashIndexedViewGatherPredFilters(step, predMember, &probe.ssaClosure);
         }
      };
      for (subop::ScanListOp scanList : probe.scanListsFromTraverse) tryInsertOnScanList(scanList);
      consumer.walk([&](subop::ScanListOp scanList) { tryInsertOnScanList(scanList); });
   }
}

void refreshCachedJoinLayoutsFromSyntheticCachePuts(mlir::ModuleOp synthetic,
                                                    llvm::ArrayRef<CacheTarget> targetsInSynthetic,
                                                    CachedJoinBufferLayoutsByKey& layoutsByKey) {
   auto& mm = synthetic.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   for (const CacheTarget& t : targetsInSynthetic) {
      auto itPrevLayout = layoutsByKey.find(t.cacheKey);
      if (itPrevLayout == layoutsByKey.end()) continue;
      synthetic.walk([&](subop::CachePutOp put) {
         if (static_cast<uint64_t>(put.getKey()) != t.cacheKey) return;
         auto hivTy = asHashIndexedViewLayoutType(put.getState().getType());
         if (!hivTy) return;
         const CachedJoinBufferLayout* prev = &itPrevLayout->second;

         CachedJoinBufferLayout layout;
         layout.producerHiv = hivTy;
         layout.query0SemanticKeys = prev->query0SemanticKeys;
         layout.query1SemanticKeys = prev->query1SemanticKeys;
         layout.query0SlotInUnion = prev->query0SlotInUnion;
         layout.query1SlotInUnion = prev->query1SlotInUnion;
         size_t idx = 0;
         for (subop::Member m : hivTy.getValueMembers().getMembers()) {
            layout.payloadMembers.push_back(m);
            layout.payloadColumnTypes.push_back(mm.getType(m));
            if (idx < prev->payloadSemanticKeys.size()) {
               layout.payloadSemanticKeys.push_back(prev->payloadSemanticKeys[idx]);
            } else if (auto predSlot = parseFilterPredMemberSlot(mm.getName(m))) {
               layout.payloadSemanticKeys.push_back(reuseFilterPredSemanticKey(*predSlot));
            } else {
               assert(false && "cache_put layout refresh must preserve union semantic keys from producer plan");
            }
            ++idx;
         }
         layoutsByKey[t.cacheKey] = std::move(layout);
      });
   }
}

} // namespace lingodb::compiler::dialect::subop
