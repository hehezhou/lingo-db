#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseJoinSuperset.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseRewriteCommon.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseFilterPredInsert.h"
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
#include "lingodb/runtime/ExternalDataSourceProperty.h"
#include "lingodb/utility/Serialization.h"

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

static llvm::StringRef stripMemberSuffix(llvm::StringRef name) {
   size_t pos = name.find('$');
   if (pos == llvm::StringRef::npos) return name;
   return name.take_front(pos);
}

static std::string columnSemanticKey(llvm::StringRef scope, llvm::StringRef leaf) {
   return (scope + "\x1f" + leaf).str();
}

static constexpr llvm::StringRef kReuseFilterPredScope = "reuse_filter_pred";

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

static bool parseFilterPredLayoutSemanticKey(llvm::StringRef semanticKey, unsigned& predIndex) {
   if (parseReuseFilterPredSemanticKey(semanticKey, predIndex)) return true;
   return static_cast<bool>(parseFilterPredMemberSlot(semanticKey));
}

static llvm::StringRef normalizeColumnIdentifier(llvm::StringRef identifier) {
   return stripMemberSuffix(identifier);
}

static bool isPayloadMemberSlotName(llvm::StringRef name) { return name.starts_with("member$"); }

struct PayloadColumnSpec {
   std::string semanticKey;
   std::string scope;
   std::string leaf;
   mlir::Type colType;
   bool isJoinKey = false;
   bool inQuery0 = false;
   bool inQuery1 = false;
};

struct JoinBufferUnionPlan {
   subop::Member linkMember;
   subop::Member hashMember;
   llvm::SmallVector<PayloadColumnSpec, 8> payloadColumns;
   llvm::SmallVector<mlir::Type, 8> payloadMemberTypes;
   /// Buffer/HIV payload members aligned with \p payloadColumns (semantic union, not slot `member$N`).
   llvm::SmallVector<subop::Member, 8> payloadMembers;
};

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
static llvm::StringRef joinKeyMemberNameFromBuildStep(subop::ExecutionStepOp buildStep, subop::Member linkMember,
                                                      subop::Member hashMember, subop::MemberManager& mm,
                                                      tuples::ColumnManager& cm) {
   subop::MaterializeOp mat = findJoinBufferMaterializeInStep(buildStep);
   assert(mat && "join superset: build step must materialize join buffer");
   mlir::Block& body = buildStep.getSubOps().front();
   subop::MapOp hashMap = findJoinHashMapBeforeMaterialize(body, mat);
   assert(hashMap && "join superset: hash map must precede buffer materialize");
   assert(!hashMap.getInputCols().empty() && "join hash map must have key input columns");
   auto keyRef = mlir::cast<tuples::ColumnRefAttr>(hashMap.getInputCols()[0]);
   auto [keyScope, keyLeaf] = cm.getName(&keyRef.getColumn());
   for (auto& [member, colRef] : mat.getMapping().getMapping()) {
      if (member == linkMember || member == hashMember) continue;
      auto [scope, leaf] = cm.getName(&colRef.getColumn());
      if (scope == keyScope && leaf == keyLeaf) return mm.getName(member);
   }
   llvm_unreachable("join superset: materialize must map hash-map key column to a buffer member");
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
                                                                        const ModuleReuseInfo& reuse) {
   for (const ModuleReuseInfo::StepRW& rw : reuse.steps) {
      subop::ExecutionStepOp step = rw.step;
      bool hasTableScan = false;
      bool hasJoinMat = false;
      step->walk([&](subop::ScanRefsOp scan) {
         if (mlir::isa<subop::TableType>(scan.getState().getType())) hasTableScan = true;
      });
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

static void collectPayloadFromMaterialize(
   subop::MaterializeOp mat, llvm::StringRef linkMemberName, llvm::StringRef hashMemberName, subop::MemberManager& mm,
   lingodb::compiler::dialect::tuples::ColumnManager& cm, llvm::StringRef joinKeyMemberName,
   llvm::StringMap<PayloadColumnSpec>& out, std::optional<unsigned> reuseQueryIndex) {
   for (auto& [member, colRef] : mat.getMapping().getMapping()) {
      llvm::StringRef memName = mm.getName(member);
      if (memName == linkMemberName || memName == hashMemberName) continue;
      auto [scope, leaf] = cm.getName(&colRef.getColumn());
      PayloadColumnSpec spec;
      spec.colType = colRef.getColumn().type;
      spec.isJoinKey = !joinKeyMemberName.empty() && memName == joinKeyMemberName;
      if (reuseQueryIndex && isFilterPredPayloadColumn(memName, leaf)) {
         spec.scope = kReuseFilterPredScope.str();
         spec.leaf = llvm::Twine(*reuseQueryIndex).str();
         spec.semanticKey = reuseFilterPredSemanticKey(*reuseQueryIndex);
      } else {
         spec.scope = scope;
         spec.leaf = leaf;
         spec.semanticKey = columnSemanticKey(scope, leaf);
      }
      auto [it, inserted] = out.try_emplace(spec.semanticKey, spec);
      if (inserted) {
         it->second.inQuery0 = reuseQueryIndex && *reuseQueryIndex == 0;
         it->second.inQuery1 = reuseQueryIndex && *reuseQueryIndex == 1;
      } else {
         if (reuseQueryIndex && *reuseQueryIndex == 0) it->second.inQuery0 = true;
         if (reuseQueryIndex && *reuseQueryIndex == 1) it->second.inQuery1 = true;
      }
   }
}

static mlir::Type memberTypeForIdentifier(subop::TableType tableTy, subop::MemberManager& mm, llvm::StringRef identifier);
static subop::Member tableMemberForIdentifier(subop::TableType tableTy, subop::MemberManager& mm,
                                              llvm::StringRef identifier);

static ExternalDatasourceProperty mergeExternalDatasource(const ExternalDatasourceProperty& a,
                                                        const ExternalDatasourceProperty& b);

static subop::ScanRefsOp findTableScanRefsInBuildStep(subop::ExecutionStepOp buildStep) {
   mlir::Block& body = buildStep.getSubOps().front();
   for (mlir::Operation& op : body.without_terminator()) {
      auto s = mlir::dyn_cast<subop::ScanRefsOp>(&op);
      if (!s) continue;
      if (mlir::isa<subop::TableType>(s.getState().getType())) return s;
   }
   return {};
}

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

static std::optional<subop::GetExternalOp> resolveGetExternalOpForScannedTable(subop::ExecutionStepOp buildStep,
                                                                               mlir::Value tableStateInBody,
                                                                               const ModuleReuseInfo& reuse) {
   mlir::Value external = tableStateInBody;
   if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(external)) {
      if (ba.getOwner() == &buildStep.getSubOps().front() && ba.getArgNumber() < buildStep.getNumOperands()) {
         external = buildStep.getOperand(ba.getArgNumber());
      }
   }
   external = peelBlockArgsToEnclosingOperands(external);

   if (auto ge = mlir::dyn_cast_or_null<subop::GetExternalOp>(external.getDefiningOp())) return ge;

   if (auto tableStep = mlir::dyn_cast_or_null<subop::ExecutionStepOp>(external.getDefiningOp())) {
      assert(isExternalTableRefStep(tableStep) &&
             "scan_refs table state must come from external table_ref construction step");
      return findUniqueGetExternalInTableRefStep(tableStep);
   }

   mlir::Value canon = canonicalizeStateValueForReuse(external);
   if (findReuseMap(reuse.externalDatasourceByTableState, canon) !=
       reuse.externalDatasourceByTableState.end()) {
      subop::GetExternalOp ge;
      for (const ModuleReuseInfo::StepRW& rw : reuse.steps) {
         subop::ExecutionStepOp step = rw.step;
         if (!isExternalTableRefStep(step)) continue;
         if (canonicalizeStateValueForReuse(step.getResult(0)) != canon) continue;
         assert(!ge && "table state must map to exactly one external table_ref step");
         ge = findUniqueGetExternalInTableRefStep(step);
      }
      assert(ge && "scan_refs must resolve to get_external in table construction step");
      return ge;
   }
   return std::nullopt;
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

static unsigned countUnionPayloadLeavesOnTable(llvm::ArrayRef<PayloadColumnSpec> payloadColumns,
                                               subop::TableType tableTy, subop::MemberManager& mm) {
   unsigned overlap = 0;
   for (const PayloadColumnSpec& spec : payloadColumns) {
      unsigned qIdx = 0;
      if (parseReuseFilterPredSemanticKey(spec.semanticKey, qIdx)) continue;
      if (tableMemberForIdentifier(tableTy, mm, spec.leaf)) ++overlap;
   }
   return overlap;
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
                                                        llvm::StringMap<PayloadColumnSpec>& unionCols) {
   llvm::SmallVector<PayloadColumnSpec, 16> unionSpecs;
   unionSpecs.reserve(unionCols.size());
   for (auto& it : unionCols) unionSpecs.push_back(it.getValue());

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

      for (const auto& map : ds.mapping) {
         llvm::StringRef leaf = normalizeColumnIdentifier(map.identifier);
         PayloadColumnSpec spec;
         spec.scope = tableName.str();
         spec.leaf = leaf.str();
         spec.colType = memberTypeForIdentifier(tableTy, mm, leaf);
         assert(spec.colType && "external table mapping column must exist on scanned table type");
         spec.semanticKey = columnSemanticKey(spec.scope, spec.leaf);
         unionCols.try_emplace(spec.semanticKey, spec);
      }
   });
   assert(sawTableScan && "join union ingest: build step must contain scan_refs on a table state");
}

static void mergePeerExternalFromBuildStepScan(ExternalDatasourceProperty& merged, bool& haveMerged,
                                             subop::ExecutionStepOp peerBuild, const ModuleReuseInfo& reusePeer,
                                             llvm::StringRef donorTableName) {
   subop::ScanRefsOp scanOp = findTableScanRefsForExternalTable(peerBuild, donorTableName, reusePeer);
   if (!scanOp) return;
   llvm::StringRef peerTableName;
   ExternalDatasourceProperty peerDs;
   bool havePeer = false;
   subop::TableType peerTy;
   assert(resolveScannedTableExternal(peerBuild, scanOp.getState(), reusePeer, peerTableName, peerDs, havePeer,
                                      peerTy) &&
          havePeer && peerTableName == donorTableName);
   if (!haveMerged) {
      merged = peerDs;
      haveMerged = true;
   } else {
      merged = mergeExternalDatasource(merged, peerDs);
   }
}

static mlir::Type columnTypeForIdentifierFromPeerBuildScan(subop::ExecutionStepOp peerBuild,
                                                           const ModuleReuseInfo& reusePeer,
                                                           llvm::StringRef donorTableName, llvm::StringRef identifier) {
   subop::ScanRefsOp scanOp = findTableScanRefsForExternalTable(peerBuild, donorTableName, reusePeer);
   if (!scanOp) return {};
   llvm::StringRef peerTableName;
   ExternalDatasourceProperty peerDs;
   bool havePeer = false;
   subop::TableType peerTy;
   assert(resolveScannedTableExternal(peerBuild, scanOp.getState(), reusePeer, peerTableName, peerDs, havePeer,
                                      peerTy) &&
          havePeer && peerTableName == donorTableName);
   auto& mm = peerBuild.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   mlir::Type ty = memberTypeForIdentifier(peerTy, mm, identifier);
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
   llvm::StringRef joinKeyMemberName;
   mlir::Value canonBuf = hiv;
   if (auto it = reuseA.mergedFromShadowState.find(hiv);
       it != reuseA.mergedFromShadowState.end()) {
      canonBuf = it->second;
   }
   subop::ExecutionStepOp buildStep = findBufferBuildStepWithTableMaterialize(canonBuf, reuseA);
   if (!buildStep) buildStep = findBufferBuildStepWithTableScan(modA);
   assert(buildStep && "join superset: HIV build step required for union plan");
   joinKeyMemberName = joinKeyMemberNameFromBuildStep(buildStep, plan.linkMember, plan.hashMember, mm, cm);
   llvm::StringRef linkMemberName = mm.getName(plan.linkMember);
   llvm::StringRef hashMemberName = mm.getName(plan.hashMember);

   llvm::StringMap<PayloadColumnSpec> unionCols;
   auto ingestHiv = [&](mlir::Value h, const ModuleReuseInfo& reuse, unsigned reuseQueryIndex) {
      auto& cm = h.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      mlir::Value canon = resolveCacheTargetStateForReuse(h, reuse);
      mlir::Value buf = canon;
      if (auto it = reuse.mergedFromShadowState.find(canon);
          it != reuse.mergedFromShadowState.end()) {
         buf = it->second;
      }
      subop::ExecutionStepOp buildStep = findBufferBuildStepWithTableMaterialize(buf, reuse);
      assert(buildStep && "join superset: build step required for ingestHiv");
      buildStep.walk([&](subop::MaterializeOp mat) {
         if (!getInnerBufferTypeForMaterializeState(mat.getState().getType()) &&
             !materializeTargetsJoinBuffer(mat, buf, reuse)) {
            return;
         }
         collectPayloadFromMaterialize(mat, linkMemberName, hashMemberName, mm, cm, joinKeyMemberName, unionCols,
                                       reuseQueryIndex);
      });
      if (enableFilterPredReuse) {
         PayloadColumnSpec predSpec;
         predSpec.scope = kReuseFilterPredScope.str();
         predSpec.leaf = llvm::Twine(reuseQueryIndex).str();
         predSpec.colType = mlir::IntegerType::get(h.getContext(), 1);
         predSpec.semanticKey = reuseFilterPredSemanticKey(reuseQueryIndex);
         unionCols.try_emplace(predSpec.semanticKey, predSpec);
      }
      ingestExternalTableColumnsFromBuildStepScan(buildStep, reuse, unionCols);
   };
   ingestHiv(hivA, reuseA, 0);
   ingestHiv(hivB, reuseB, 1);

   llvm::SmallVector<PayloadColumnSpec*, 8> ordered;
   ordered.reserve(unionCols.size());
   for (auto& it : unionCols) ordered.push_back(&it.second);
   llvm::sort(ordered, [](const PayloadColumnSpec* a, const PayloadColumnSpec* b) {
      if (a->isJoinKey != b->isJoinKey) return a->isJoinKey > b->isJoinKey;
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

/// Next `member$N` slot after \p bySemanticKey reuse set and already-assigned payload members.
static unsigned nextPayloadMemberSlot(subop::MemberManager& mm, const llvm::StringMap<subop::Member>& bySemanticKey,
                                        llvm::ArrayRef<subop::Member> assigned) {
   unsigned maxSlot = 0;
   auto bump = [&](subop::Member m) {
      if (auto slot = parseMemberSlot(mm.getName(m))) maxSlot = std::max(maxSlot, *slot + 1);
      if (auto predSlot = parseFilterPredMemberSlot(mm.getName(m))) maxSlot = std::max(maxSlot, *predSlot + 1);
   };
   for (const auto& it : bySemanticKey) bump(it.second);
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
   if (!ty || ty.getContext() == ctx) return ty;
   if (auto nullable = mlir::dyn_cast<db::NullableType>(ty))
      return db::NullableType::get(cloneTypeToContext(nullable.getType(), ctx));
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
   llvm_unreachable("cloneTypeToContext: unsupported type for cross-context layout clone");
}

static void collectSemanticKeyToMemberFromMaterialize(
   subop::MaterializeOp mat, subop::Member linkM, subop::Member hashM, subop::MemberManager& mm,
   lingodb::compiler::dialect::tuples::ColumnManager& cm, llvm::StringMap<subop::Member>& out) {
   if (!mat) return;
   for (auto& [member, colRef] : mat.getMapping().getMapping()) {
      if (member == linkM || member == hashM) continue;
      auto [scope, leaf] = cm.getName(&colRef.getColumn());
      if (mm.getType(member) != colRef.getColumn().type) continue;
      out.try_emplace(columnSemanticKey(scope, leaf), member);
   }
}

/// Assign \p plan.payloadMembers from cloned build-step materialize mappings + fresh members for union-only columns.
/// Reuses existing buffer members (e.g. \c member$0, \c member$1) from materialize; new union columns get the next
/// \c member$N slot via \c createMemberDirect (not semantic leaf names like \c s_comment$0).
static void assignPayloadMembersForPlan(subop::MemberManager& mm, lingodb::compiler::dialect::tuples::ColumnManager& cm,
                                        subop::MaterializeOp matOp, JoinBufferUnionPlan& plan) {
   llvm::StringMap<subop::Member> bySemanticKey;
   collectSemanticKeyToMemberFromMaterialize(matOp, plan.linkMember, plan.hashMember, mm, cm, bySemanticKey);
   plan.payloadMembers.clear();
   plan.payloadMembers.reserve(plan.payloadColumns.size());
   unsigned nextSlot = nextPayloadMemberSlot(mm, bySemanticKey, plan.payloadMembers);
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
      if (parseReuseFilterPredSemanticKey(spec.semanticKey, predIdx)) {
         plan.payloadMembers.push_back(mm.getOrCreateMemberDirect(
            "filter_pred$" + llvm::Twine(predIdx).str(), canonicalPredTy, /*allowTypeUpdate=*/false));
         continue;
      }
      if (auto it = bySemanticKey.find(spec.semanticKey); it != bySemanticKey.end()) {
         mlir::Type wantTy = cloneTypeToContext(spec.colType, synthCtx);
         assert(mm.getType(it->second) == wantTy && "join superset: reused member type must match union column");
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
   auto oldHiv = mlir::cast<subop::HashIndexedViewType>(chiv.getType());
   auto keyMs = subop::StateMembersAttr::get(ctx, llvm::SmallVector<subop::Member>{hashM});
   auto valMs = subop::StateMembersAttr::get(ctx, vals);
   auto newHiv = subop::HashIndexedViewType::get(ctx, keyMs, valMs, oldHiv.getCompareHashForLookup());
   chiv.getResult().setType(newHiv);
}

static void syncMaterializeMappingsToBufferMembers(mlir::ModuleOp module, subop::StateMembersAttr targetMembers,
                                                   const llvm::DenseSet<void*>& closure);

static void applyBufferLayoutToSsaClosure(mlir::ModuleOp module, llvm::ArrayRef<mlir::Value> canonicalBuffers,
                                          subop::StateMembersAttr targetMembers, const ModuleReuseInfo& reuse) {
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
   syncMaterializeMappingsToBufferMembers(module, targetMembers, closure);
   synchronizeExecutionStepPortTypes(module, &closure);
}

/// After buffer layout is applied, canonicalize \c materialize member slots to \p targetMembers (by name / semantic key).
static void syncMaterializeMappingsToBufferMembers(mlir::ModuleOp module, subop::StateMembersAttr targetMembers,
                                                   const llvm::DenseSet<void*>& closure) {
   auto* ctx = module.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   llvm::StringMap<subop::Member> validByName;
   for (subop::Member m : targetMembers.getMembers()) validByName[mm.getName(m)] = m;

   module.walk([&](subop::MaterializeOp mat) {
      if (!opaqueClosureContains(closure, mat.getState())) return;
      subop::BufferType bufTy = getInnerBufferTypeForMaterializeState(mat.getState().getType());
      if (!bufTy || bufTy.getMembers() != targetMembers) return;

      llvm::StringMap<subop::Member> bySemanticKey;
      for (auto& pr : mat.getMapping().getMapping()) {
         auto itName = validByName.find(mm.getName(pr.first));
         if (itName == validByName.end()) continue;
         auto [scope, leaf] = cm.getName(&pr.second.getColumn());
         bySemanticKey[columnSemanticKey(scope, leaf)] = itName->second;
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
         if (!canon) {
            auto [scope, leaf] = cm.getName(&pr.second.getColumn());
            if (auto it = bySemanticKey.find(columnSemanticKey(scope, leaf)); it != bySemanticKey.end())
               canon = it->second;
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

/// Merge pushdown filters from matched queries into `(filterDescriptions AND ...) OR (orFilterClauses[i] AND ...)`.
static void mergeExternalFiltersForOrReuse(ExternalDatasourceProperty& merged,
                                           llvm::ArrayRef<ExternalDatasourceProperty> filterSources) {
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

static subop::GatherOp findPeerGatherForSemanticKey(subop::ExecutionStepOp peerBuild, llvm::StringRef semanticKey,
                                                    tuples::ColumnManager& peerCm) {
   llvm::StringRef wantLeaf = normalizeColumnIdentifier(semanticKeyLeaf(semanticKey));
   subop::GatherOp found;
   peerBuild->walk([&](subop::GatherOp g) {
      for (auto& [mem, def] : g.getMapping().getMapping()) {
         auto [scope, leaf] = peerCm.getName(&def.getColumn());
         (void)scope;
         if (normalizeColumnIdentifier(leaf) == wantLeaf) found = g;
      }
   });
   return found;
}

static subop::GatherOp findTableGatherOnStream(mlir::Block& body, mlir::Value stream, subop::MaterializeOp matOp) {
   subop::GatherOp found;
   for (mlir::Operation& op : body.without_terminator()) {
      if (!op.isBeforeInBlock(matOp.getOperation())) continue;
      if (auto g = mlir::dyn_cast<subop::GatherOp>(&op)) {
         if (g.getStream() == stream) found = g;
      }
   }
   return found;
}

static bool extendGatherMapping(subop::GatherOp gather, subop::Member tableMem, tuples::ColumnDefAttr colDef,
                                mlir::MLIRContext* ctx) {
   auto m = gather.getMapping();
   for (auto& [mem, def] : m.getMapping()) {
      if (mem == tableMem) return false;
      (void)def;
   }
   llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> pairs;
   for (auto& p : m.getMapping()) pairs.push_back(p);
   pairs.push_back({tableMem, colDef});
   gather.setMappingAttr(subop::ColumnDefMemberMappingAttr::get(ctx, pairs));
   return true;
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

static void syncMapInputColsFromGather(subop::GatherOp gather, tuples::ColumnManager& cm) {
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
         } else if (auto it = outRefByNormLeaf.find(leaf); it != outRefByNormLeaf.end()) {
            replacement = it->second;
         } else if (!isPayloadMemberSlotName(leaf)) {
            if (auto it = outRefByNormLeaf.find(normalizeColumnIdentifier(leaf)); it != outRefByNormLeaf.end()) {
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
   auto expectedLer = subop::LookupEntryRefType::get(ctx, producerHiv);
   auto expectedListTy = subop::ListType::get(ctx, expectedLer);
   auto shouldUpdateOp = [&](mlir::Operation* op) {
      if (!closureFilter) return true;
      return opOperandsOrNestedBlockArgsTouchClosure(op, *closureFilter);
   };
   auto syncLookupEntryRefToHiv = [&](tuples::ColumnRefAttr cref) -> bool {
      bool changed = false;
      if (auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(cref.getColumn().type)) {
         if (ler.getState() != producerHiv) return false;
         if (cref.getColumn().type != expectedLer) {
            cref.getColumn().type = expectedLer;
            changed = true;
         }
      } else if (auto list = mlir::dyn_cast<subop::ListType>(cref.getColumn().type)) {
         auto elemLer = mlir::dyn_cast<subop::LookupEntryRefType>(list.getT());
         if (!elemLer || elemLer.getState() != producerHiv) return false;
         if (cref.getColumn().type != expectedListTy) {
            cref.getColumn().type = expectedListTy;
            changed = true;
         }
      }
      return changed;
   };
   auto syncLookupEntryDefToHiv = [&](tuples::ColumnDefAttr def) -> bool {
      auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(def.getColumn().type);
      if (!ler || ler.getState() != producerHiv) return false;
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
         auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
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
      auto hivTy = mlir::dyn_cast<subop::HashIndexedViewType>(op.getState().getType());
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
         auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
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
         if (elem.getColumn().type != listTy.getT()) elem.getColumn().type = listTy.getT();
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
      auto hivTy = mlir::dyn_cast<subop::HashIndexedViewType>(op.getState().getType());
      if (!hivTy) return;
      if (!seenHiv.insert(hivTy.getAsOpaquePointer()).second) return;
      auto listRef = op.getRef();
      auto [listScope, listLeaf] = cm.getName(&listRef.getColumn());
      propagateJoinSupersetColumnAttrs(module, &closureFilter, hivTy, true, listScope);
   });
   syncProbeListCarriersInClosure(module, &closureFilter);
}

static void patchBufferBuildStepForUnion(mlir::ModuleOp synthetic, subop::ExecutionStepOp buildStep,
                                         const JoinBufferUnionPlan& plan, subop::BufferType targetBufTy,
                                         const ModuleReuseInfo& reuseSynthetic,
                                         llvm::ArrayRef<std::pair<mlir::ModuleOp, mlir::Value>> peerHivs,
                                         llvm::ArrayRef<const ModuleReuseInfo*> peerReuses) {
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
         if (parseReuseFilterPredSemanticKey(spec.semanticKey, predIdx)) continue;
         scanOp = findTableScanForPayloadLeaf(buildStep, spec.leaf, mm);
         if (scanOp) break;
      }
   }
   if (!scanOp) return;

   mlir::Value tableState = scanOp.getState();
   subop::TableType mergedSupplierTableTy;
   ExternalDatasourceProperty mergedDs;
   bool haveDs = false;
   llvm::StringRef donorTableName;
   subop::TableType donorTableTy;
   const bool haveDonorExternal = resolveScannedTableExternal(buildStep, tableState, reuseSynthetic, donorTableName,
                                                             mergedDs, haveDs, donorTableTy) &&
                                  haveDs &&
                                  countUnionPayloadLeavesOnTable(plan.payloadColumns, donorTableTy, mm) > 0;

   if (haveDonorExternal) {
   llvm::SmallVector<ExternalDatasourceProperty, 4> filterSources;
   filterSources.push_back(mergedDs);
   for (size_t pi = 0; pi < peerHivs.size(); ++pi) {
      auto [peerMod, peerHiv] = peerHivs[pi];
      if (!peerMod || !peerHiv) continue;
      const ModuleReuseInfo& reusePeer = *peerReuses[pi];
      mlir::Value peerCanon = resolveCacheTargetStateForReuse(peerHiv, reusePeer);
      mlir::Value peerBuf = peerCanon;
      if (auto it = reusePeer.mergedFromShadowState.find(peerCanon);
          it != reusePeer.mergedFromShadowState.end()) {
         peerBuf = it->second;
      }
      subop::ExecutionStepOp peerBuild = findBufferBuildStepWithTableMaterialize(peerBuf, reusePeer);
      if (!peerBuild) peerBuild = findBufferBuildStepWithTableScan(peerMod);
      if (peerBuild) {
         subop::ScanRefsOp peerScan = findTableScanRefsForExternalTable(peerBuild, donorTableName, reusePeer);
         if (peerScan) {
            llvm::StringRef peerTableName;
            ExternalDatasourceProperty peerDs;
            bool havePeerDs = false;
            subop::TableType peerTy;
            assert(resolveScannedTableExternal(peerBuild, peerScan.getState(), reusePeer, peerTableName, peerDs,
                                               havePeerDs, peerTy) &&
                   havePeerDs && peerTableName == donorTableName);
            filterSources.push_back(std::move(peerDs));
         }
         mergePeerExternalFromBuildStepScan(mergedDs, haveDs, peerBuild, reusePeer, donorTableName);
      }
   }
   assert(haveDs && "join superset: merged external datasource required for donor table");
   mergeExternalFiltersForOrReuse(mergedDs, filterSources);

   auto lookupPeerColumnType = [&](llvm::StringRef identifier) -> mlir::Type {
      for (size_t pi = 0; pi < peerHivs.size(); ++pi) {
         auto [peerMod, peerHiv] = peerHivs[pi];
         if (!peerMod || !peerHiv) continue;
         const ModuleReuseInfo& reusePeer = *peerReuses[pi];
         mlir::Value peerCanon = resolveCacheTargetStateForReuse(peerHiv, reusePeer);
         mlir::Value peerBuf = peerCanon;
         if (auto it = reusePeer.mergedFromShadowState.find(peerCanon);
             it != reusePeer.mergedFromShadowState.end()) {
            peerBuf = it->second;
         }
         subop::ExecutionStepOp peerBuild = findBufferBuildStepWithTableMaterialize(peerBuf, reusePeer);
         if (!peerBuild) peerBuild = findBufferBuildStepWithTableScan(peerMod);
         if (!peerBuild) continue;
         if (mlir::Type ty =
                columnTypeForIdentifierFromPeerBuildScan(peerBuild, reusePeer, donorTableName, identifier)) {
            return ty;
         }
      }
      return {};
   };
   subop::TableType hintTy = donorTableTy;
   assert(hintTy && "join superset: resolveScannedTableExternal must provide donor table type");
   auto newTableTy = tableTypeFromMergedExternal(ctx, mm, mergedDs, hintTy, lookupPeerColumnType);
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
   mergedSupplierTableTy = newTableTy;
   }

   subop::MaterializeOp matOp = findJoinBufferMaterializeInStep(buildStep);
   assert(matOp && "join superset: buffer build step must materialize into join buffer");

   auto collectMaterializedKeys = [&]() {
      std::unordered_set<std::string> keys;
      for (auto& [member, colRef] : matOp.getMapping().getMapping()) {
         if (member == plan.linkMember || member == plan.hashMember) continue;
         auto [scope, leaf] = cm.getName(&colRef.getColumn());
         keys.insert(columnSemanticKey(scope, leaf));
      }
      return keys;
   };
   std::unordered_set<std::string> materializedKeys = collectMaterializedKeys();

   subop::MapOp hashMapOp = findJoinHashMapBeforeMaterialize(body, matOp);
   assert(hashMapOp);

   auto appendMaterializeMember = [&](subop::Member bufMem, tuples::ColumnDefAttr colDef) {
      llvm::SmallVector<subop::RefMappingPairT> matPairs;
      for (auto pr : matOp.getMapping().getMapping()) matPairs.push_back(pr);
      matPairs.push_back({bufMem, cm.createRef(&colDef.getColumn())});
      matOp.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, matPairs));
   };

   auto isPayloadMaterialized = [&](const PayloadColumnSpec& spec) {
      if (materializedKeys.contains(spec.semanticKey)) return true;
      llvm::StringRef wantLeaf = normalizeColumnIdentifier(spec.leaf);
      for (const std::string& k : materializedKeys) {
         if (normalizeColumnIdentifier(semanticKeyLeaf(k)) == wantLeaf) return true;
      }
      return false;
   };

   int payloadSlotInPlan = 0;
   for (const PayloadColumnSpec& spec : plan.payloadColumns) {
      unsigned reusePredQueryIdx = 0;
      if (parseReuseFilterPredSemanticKey(spec.semanticKey, reusePredQueryIdx)) {
         ++payloadSlotInPlan;
         continue;
      }
      if (isPayloadMaterialized(spec)) {
         ++payloadSlotInPlan;
         continue;
      }

      subop::GatherOp templateGather;
      tuples::ColumnManager* templateCm = nullptr;
      for (size_t pi = 0; pi < peerHivs.size(); ++pi) {
         auto [peerMod, peerHiv] = peerHivs[pi];
         if (!peerMod || !peerHiv) continue;
         const ModuleReuseInfo& reusePeer = *peerReuses[pi];
         mlir::Value peerCanon = resolveCacheTargetStateForReuse(peerHiv, reusePeer);
         mlir::Value peerBuf = peerCanon;
         if (auto it = reusePeer.mergedFromShadowState.find(peerCanon);
             it != reusePeer.mergedFromShadowState.end()) {
            peerBuf = it->second;
         }
         subop::ExecutionStepOp peerBuild = findBufferBuildStepWithTableMaterialize(peerBuf, reusePeer);
         if (!peerBuild) peerBuild = findBufferBuildStepWithTableScan(peerMod);
         if (!peerBuild) continue;
         auto& peerCm = peerMod.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
         templateGather = findPeerGatherForSemanticKey(peerBuild, spec.semanticKey, peerCm);
         if (templateGather) {
            templateCm = &peerCm;
            break;
         }
      }
      subop::ScanRefsOp specScan = findTableScanForPayloadLeaf(buildStep, spec.leaf, mm);
      if (!specScan) specScan = scanOp;
      subop::TableType tableTy = mlir::dyn_cast<subop::TableType>(specScan.getState().getType());
      assert(tableTy && "join superset: table type required for payload column gather");
      subop::Member tableMem = tableMemberForIdentifier(tableTy, mm, spec.leaf);
      if (!tableMem) {
         ++payloadSlotInPlan;
         continue;
      }

      llvm::StringRef scope = spec.scope;
      if (scope.empty() && templateGather && templateCm) {
         auto def0 = templateGather.getMapping().getMapping().begin()->second;
         auto [scope0, leaf0] = templateCm->getName(&def0.getColumn());
         (void)leaf0;
         scope = scope0;
      }
      assert(!scope.empty() && "join superset: payload column scope required");
      tuples::ColumnDefAttr colDef = cm.createDef(spec.scope, spec.leaf);
      colDef.getColumn().type = spec.colType;
      assert(static_cast<size_t>(payloadSlotInPlan) < plan.payloadMembers.size() &&
             "join superset: payload slot out of range");
      subop::Member bufMem = plan.payloadMembers[payloadSlotInPlan];

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
      auto newGather = gb.create<subop::GatherOp>(mlir::UnknownLoc::get(ctx), gatherTy, mapStream,
                                                   gatherRef, mapping);
      matOp->setOperand(0, newGather.getRes());
      appendMaterializeMember(bufMem, colDef);

      materializedKeys.insert(spec.semanticKey);
      ++payloadSlotInPlan;
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

static void applyUnionPlanToSyntheticHiv(mlir::ModuleOp synthetic, mlir::Value syntheticHiv, JoinBufferUnionPlan& plan,
                                         const ModuleReuseInfo& reuseSynthetic,
                                         mlir::ModuleOp query0Module, mlir::Value hivA, const ModuleReuseInfo& reuseA,
                                         mlir::ModuleOp query1Module, mlir::Value hivB,
                                         const ModuleReuseInfo& reuseB) {
   auto* ctx = synthetic.getContext();
   auto* subDialect = ctx->getLoadedDialect<subop::SubOperatorDialect>();
   auto& mm = subDialect->getMemberManager();

   mlir::Value mergedBuf = resolveJoinMergedBuffer(syntheticHiv, synthetic, reuseSynthetic);

   subop::ExecutionStepOp buildStep = findBufferBuildStepWithTableMaterialize(mergedBuf, reuseSynthetic);
   if (!buildStep) buildStep = findBufferBuildStepWithTableScan(synthetic);
   subop::MaterializeOp matOp;
   if (buildStep) {
      buildStep.walk([&](subop::MaterializeOp m) {
         if (!materializeTargetsJoinBuffer(m, mergedBuf, reuseSynthetic)) return;
         matOp = m;
      });
      if (!matOp) matOp = findJoinBufferMaterializeInStep(buildStep);
   }
   assert(matOp && "join superset: synthetic build step must materialize into merged join buffer");
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   assignPayloadMembersForPlan(mm, cm, matOp, plan);
   auto targetMembers = bufferMembersForPlan(ctx, plan);
   auto targetBufTy = subop::BufferType::get(ctx, targetMembers);
   llvm::SmallVector<mlir::Value, 4> roots = {mergedBuf};

   applyBufferLayoutToSsaClosure(synthetic, roots, targetMembers, reuseSynthetic);

   if (buildStep) {
      llvm::SmallVector<std::pair<mlir::ModuleOp, mlir::Value>, 2> peerHivs = {
         {query1Module, hivB},
         {query0Module, hivA},
      };
      const ModuleReuseInfo* peerReuses[] = {&reuseB, &reuseA};
      patchBufferBuildStepForUnion(synthetic, buildStep, plan, targetBufTy, reuseSynthetic, peerHivs, peerReuses);
   }

   applyBufferLayoutToSsaClosure(synthetic, roots, targetMembers, reuseSynthetic);
   JoinBufferHivSsaClosure joinClosure = computeJoinBufferHivSsaClosure(roots, reuseSynthetic);
   alignBufferMergeThreadLocalsWithMergeResult(synthetic, &joinClosure.opaque);
   subop::HashIndexedViewType prodHiv;
   mlir::Value canonMergedBuf = canonicalizeStateValueForReuse(mergedBuf);
   synthetic.walk([&](subop::CreateHashIndexedView chiv) {
      if (prodHiv) return;
      if (canonicalizeStateValueForReuse(chiv.getSource()) != canonMergedBuf) return;
      prodHiv = mlir::dyn_cast<subop::HashIndexedViewType>(chiv.getResult().getType());
   });
   assert(prodHiv && "join superset: synthetic must create hash_indexed_view on merged buffer");
   expandClosureThroughExecutionStepPorts(synthetic, joinClosure.opaque);
   remapClosureGathersToAlignedConsumerHiv(synthetic, &joinClosure.opaque, prodHiv, layoutFromUnionPlan(prodHiv, plan),
                                           /*probeLookupScopes=*/nullptr, /*dbg=*/nullptr);
   propagateJoinSupersetColumnAttrsForClosure(synthetic, joinClosure.opaque);
   synchronizeExecutionStepPortTypes(synthetic, nullptr);
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
            if (listRef.getColumn().type != expectedListTy) listRef.getColumn().type = expectedListTy;
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
   ConsumerCachedHivSites sites;
   sites.consumerHivBeforeAlign = consumerHivBeforeAlign;
   ProbeAlignDebugCtx traverseDbg;
   traverseDbg.cacheKey = cacheKey;
   traverseDbg.passName = "traverse";
   traverseConsumerHivUsesFromRoot(cacheGetResult, alignedHiv, consumerHivBeforeAlign, consumerLayout, sites,
                                   &traverseDbg);
   out.ssaClosure = std::move(sites.ssaClosure);
   out.probeLookupScopes = std::move(sites.probeLookupScopes);
   out.scanListsFromTraverse = std::move(sites.scanListsFromTraverse);
   for (;;) {
      size_t before = out.ssaClosure.size();
      expandClosureThroughExecutionStepPorts(consumer, out.ssaClosure);
      expandClosureThroughNestedExecutionGroupPorts(consumer, out.ssaClosure);
      if (out.ssaClosure.size() == before) break;
   }
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

static void syncExecutionStepPortsForModule(mlir::ModuleOp module, const llvm::DenseSet<void*>* closureFilter) {
   synchronizeExecutionStepPortTypes(module, closureFilter);
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
      bool changed = false;
      for (auto& pr : gather.getMapping().getMapping()) {
         subop::Member nm = resolveGatherMember(pr.first, pr.second);
         if (auto bn = hivMemberByName.find(mm.getName(nm)); bn != hivMemberByName.end()) nm = bn->second;
         else if (!hivMemberSet.contains(nm)) nm = pr.first;
         if (nm != pr.first) changed = true;
         tuples::ColumnDefAttr def = pr.second;
         auto [defScope, defLeaf] = cm.getName(&def.getColumn());
         if (isJoinProbeCompilerScope(defScope) && !lookupProbeRefScopesMatch(refScope, defScope)) {
            debugProbeAlign(dbg, [&](llvm::raw_ostream& os) {
               os << "  remapClosure mapping @" << defScope << "::" << defLeaf << " -> @" << refScope
                  << "::" << mm.getName(nm);
            });
            def = cm.createDef(refScope, mm.getName(nm));
            def.getColumn().type = mm.getType(nm);
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
      syncMapInputColsFromGather(gather, cm);
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
   std::optional<subop::Member> pendingProbeGatherRemapFrom;
   assert(layout.payloadSemanticKeys.size() == layout.payloadMembers.size());

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

      unsigned predIdx = 0;
      if (parseFilterPredLayoutSemanticKey(semKey, predIdx)) {
         subop::Member consumerMem = makeOrGetPredMemberForSlot(ctx, predIdx);
         assignedBySemantic[semKey] = consumerMem;
         producerPayloadToConsumer[producerMem] = consumerMem;
         continue;
      }

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
         if (consumerReuseQueryIndex && consumerQueryHadPayloadColumn(layout, semKey, *consumerReuseQueryIndex)) {
            reused = findConsumerMemberByName(producerMm.getName(producerMem), consumerHivBeforeAlign);
         }
         if (!reused) {
            llvm::ArrayRef<subop::Member> oldVals = consumerHivBeforeAlign.getValueMembers().getMembers();
            for (unsigned qIdx : {0u, 1u}) {
               if (!consumerQueryHadPayloadColumn(layout, semKey, qIdx)) continue;
               const llvm::SmallVector<std::string>& qKeys =
                  qIdx == 0 ? layout.query0SemanticKeys : layout.query1SemanticKeys;
               if (qKeys.size() == 1 && qKeys[0] == semKey && oldVals.size() == 1) {
                  pendingProbeGatherRemapFrom = oldVals[0];
                  reused = std::nullopt;
                  break;
               }
            }
         }
      }
      subop::Member consumerMem = ensureConsumerMember(reused, producerSlotTy);
      if (pendingProbeGatherRemapFrom) {
         outConsumerLayout.probeGatherMemberRemap[*pendingProbeGatherRemapFrom] = consumerMem;
         pendingProbeGatherRemapFrom = std::nullopt;
      } else if (reused && consumerHivBeforeAlign && *reused != consumerMem) {
         outConsumerLayout.probeGatherMemberRemap[*reused] = consumerMem;
      }
      assignedBySemantic[semKey] = consumerMem;
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

   outConsumerLayout = layout;
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
   return externalDatasourceFiltersEqual(dsA, dsB);
}

} // namespace

bool joinMatchPeerExternalFiltersIdentical(mlir::ModuleOp query0, mlir::ModuleOp query1, mlir::Value hivA,
                                           mlir::Value hivB, const ModuleReuseInfo& reuse0,
                                           const ModuleReuseInfo& reuse1) {
   return joinMatchPeerExternalFiltersIdenticalImpl(query0, query1, hivA, hivB, reuse0, reuse1);
}

bool parseReuseFilterPredSemanticKey(llvm::StringRef semanticKey, unsigned& reuseQueryIndex) {
   size_t sep = semanticKey.find('\x1f');
   if (sep == llvm::StringRef::npos) return false;
   if (semanticKey.take_front(sep) != "reuse_filter_pred") return false;
   return !semanticKey.drop_front(sep + 1).getAsInteger(10, reuseQueryIndex);
}

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

static subop::ExecutionStepOp findJoinBufferBuildStepForHiv(mlir::ModuleOp module, mlir::Value hiv,
                                                            const ModuleReuseInfo& reuse) {
   mlir::Value canon = resolveCacheTargetStateForReuse(hiv, reuse);
   mlir::Value buf = canon;
   if (auto it = reuse.mergedFromShadowState.find(canon); it != reuse.mergedFromShadowState.end()) {
      buf = it->second;
   }
   if (subop::ExecutionStepOp step = findBufferBuildStepWithTableMaterialize(buf, reuse)) return step;
   return findBufferBuildStepWithTableScan(module);
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

      mlir::Value hivs[] = {resolveCacheTargetStateForReuse(match.stateA, reuse0),
                            resolveCacheTargetStateForReuse(match.stateB, reuse1)};
      ModuleReuseInfo* reuses[] = {&reuse0, &reuse1};
      mlir::ModuleOp peerMods[] = {query0, query1};

      for (size_t i = 0; i < layout.payloadSemanticKeys.size(); ++i) {
         unsigned qIdx = 0;
         if (!parseReuseFilterPredSemanticKey(layout.payloadSemanticKeys[i], qIdx)) continue;
         llvm::StringRef predName = mm.getName(layout.payloadMembers[i]);
         assert(qIdx < 2 && "insertSyntheticFilterPreds: reuse_query_index out of range");
         subop::ExecutionStepOp peerBuild = findJoinBufferBuildStepForHiv(peerMods[qIdx], hivs[qIdx], *reuses[qIdx]);
         assert(peerBuild &&
                "insertSyntheticFilterPreds: peer join-buffer build step with table scan_refs required");
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
         insertWriteSidePredIntoBufferConstructionStepForPredMember(buildStep, filters, predName);
      }
   }
}

void syncProbeGatherMappingsInModule(mlir::ModuleOp module) {
   auto& cm = module.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   llvm::DenseSet<const void*> seenHiv;
   module.walk([&](subop::LookupOp op) {
      auto hivTy = mlir::dyn_cast<subop::HashIndexedViewType>(op.getState().getType());
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

      auto consumerHivBeforeAlign = mlir::dyn_cast<subop::HashIndexedViewType>(get.getResult().getType());
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
      ProbeAlignDebugCtx remapDbg;
      remapDbg.cacheKey = probe.cacheKey;
      remapDbg.passName = "remap";
      remapClosureGathersToAlignedConsumerHiv(consumer, &probe.ssaClosure, probe.alignedHiv, probe.consumerLayout,
                                              &probe.probeLookupScopes, &remapDbg);
   }

   syncExecutionStepPortsForModule(consumer, &unionClosure);

   for (ConsumerCacheGetProbeClosure& probe : perCacheGet) {
      alignScanListForProbeClosure(consumer, probe, "align-post-sync");
   }

   syncProbeListCarriersInClosure(consumer, &unionClosure);
   syncExecutionStepPortsForModule(consumer, nullptr);

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
   ConsumerCachedHivSites sites;
   sites.consumerHivBeforeAlign = probe.consumerHivBeforeAlign;
   ProbeAlignDebugCtx traverseDbg;
   traverseDbg.cacheKey = probe.cacheKey;
   traverseDbg.passName = "probe_pred_retraverse";
   traverseConsumerHivUsesFromRoot(probe.cacheGetRoot, probe.alignedHiv, probe.consumerHivBeforeAlign,
                                   probe.consumerLayout, sites, &traverseDbg);
   probe.ssaClosure = std::move(sites.ssaClosure);
   probe.probeLookupScopes = std::move(sites.probeLookupScopes);
   probe.scanListsFromTraverse = std::move(sites.scanListsFromTraverse);
   for (;;) {
      size_t before = probe.ssaClosure.size();
      expandClosureThroughExecutionStepPorts(consumer, probe.ssaClosure);
      expandClosureThroughNestedExecutionGroupPorts(consumer, probe.ssaClosure);
      if (probe.ssaClosure.size() == before) break;
   }
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
      if (!mlir::isa<subop::HashIndexedViewType>(hivA.getType()) ||
          !mlir::isa<subop::HashIndexedViewType>(hivB.getType())) {
         continue;
      }

      JoinBufferUnionPlan plan =
         buildUnionPlan(hivA, hivB, reuse0, reuse1, query0, query1, enableFilterPredReuse);
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
            if (parseReuseFilterPredSemanticKey(spec.semanticKey, qIdx)) continue;
            nonPredSemanticKeys.insert(spec.semanticKey);
         }
         // Same stored-value fingerprint can still hide cross-query payload (e.g. s_phone vs s_comment).
         if (nonPredSemanticKeys.size() <= 2) continue;
      }

      mlir::Value synthHiv = t.state;
      if (!mlir::isa<subop::HashIndexedViewType>(synthHiv.getType())) continue;

      auto reuseSynthetic = collectModuleReuseInfo(synthetic);
      applyUnionPlanToSyntheticHiv(synthetic, synthHiv, plan, reuseSynthetic, query0, hivA, reuse0, query1, hivB,
                                   reuse1);

      if (outLayouts) {
         mlir::Value canonHiv = synthHiv;
         synthetic.walk([&](subop::CachePutOp put) {
            if (put.getKey() != t.cacheKey) return;
            canonHiv = put.getState();
         });
         if (auto hivTy = mlir::dyn_cast<subop::HashIndexedViewType>(canonHiv.getType())) {
            (*outLayouts)[t.cacheKey] = layoutFromUnionPlan(hivTy, plan);
         }
      }
   }
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
   ProbeAlignDebugCtx remapDbg;
   remapDbg.cacheKey = probe.cacheKey;
   remapDbg.passName = "finalize-remap";
   remapClosureGathersToAlignedConsumerHiv(consumer, &probe.ssaClosure, probe.alignedHiv, probe.consumerLayout,
                                           &probe.probeLookupScopes, &remapDbg);
   syncProbeListCarriersInClosure(consumer, &probe.ssaClosure);
}

void resyncConsumerCachedHivCarrierTypesFromCacheGet(mlir::ModuleOp consumer,
                                                     std::optional<uint64_t> cacheKey) {
   auto& mm = consumer.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   consumer.walk([&](subop::CacheGetOp get) {
      if (cacheKey && static_cast<uint64_t>(get.getKey()) != *cacheKey) return;
      auto canonicalHiv = mlir::dyn_cast<subop::HashIndexedViewType>(get.getResult().getType());
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
      expandClosureThroughExecutionStepPorts(consumer, sites.ssaClosure);
      syncExecutionStepPortsForModule(consumer, &sites.ssaClosure);
      propagateJoinSupersetColumnAttrsForClosure(consumer, sites.ssaClosure);
      syncExecutionStepPortsForModule(consumer, nullptr);
   });
}

void applyProbePredFiltersForConsumerClosures(mlir::ModuleOp consumer,
                                              llvm::MutableArrayRef<ConsumerCacheGetProbeClosure> probeClosures) {
   auto* ctx = consumer.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
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
         tuples::ColumnRefAttr entryRef = cm.createRef(&scanList.getElem().getColumn());
         insertProbePredFilterImmediatelyAfterScanProducer(scanList.getOperation(), scanList.getRes(), entryRef,
                                                           predMember);
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
      synthetic.walk([&](subop::CachePutOp put) {
         if (static_cast<uint64_t>(put.getKey()) != t.cacheKey) return;
         auto hivTy = mlir::dyn_cast<subop::HashIndexedViewType>(put.getState().getType());
         if (!hivTy) return;
         const CachedJoinBufferLayout* prev = nullptr;
         if (auto it = layoutsByKey.find(t.cacheKey); it != layoutsByKey.end()) prev = &it->second;

         CachedJoinBufferLayout layout;
         layout.producerHiv = hivTy;
         if (prev) {
            layout.query0SemanticKeys = prev->query0SemanticKeys;
            layout.query1SemanticKeys = prev->query1SemanticKeys;
            layout.query0SlotInUnion = prev->query0SlotInUnion;
            layout.query1SlotInUnion = prev->query1SlotInUnion;
         }
         size_t idx = 0;
         for (subop::Member m : hivTy.getValueMembers().getMembers()) {
            layout.payloadMembers.push_back(m);
            layout.payloadColumnTypes.push_back(mm.getType(m));
            assert(prev && idx < prev->payloadSemanticKeys.size() &&
                   "cache_put layout refresh must preserve union semantic keys from producer plan");
            layout.payloadSemanticKeys.push_back(prev->payloadSemanticKeys[idx]);
            ++idx;
         }
         layoutsByKey[t.cacheKey] = std::move(layout);
      });
   }
}

} // namespace lingodb::compiler::dialect::subop
