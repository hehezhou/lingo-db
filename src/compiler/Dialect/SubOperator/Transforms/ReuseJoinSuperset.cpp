#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseJoinSuperset.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/CrossQueryStateReuse.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseRewriteCommon.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseFilterPredInsert.h"
#include "lingodb/compiler/Dialect/DB/IR/DBTypes.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ReuseStateClosure.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsAttributes.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateExtraction.h"
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

static bool parseReuseFilterPredSemanticKey(llvm::StringRef semanticKey, unsigned& reuseQueryIndex) {
   size_t sep = semanticKey.find('\x1f');
   if (sep == llvm::StringRef::npos) return false;
   if (semanticKey.take_front(sep) != kReuseFilterPredScope) return false;
   return !semanticKey.drop_front(sep + 1).getAsInteger(10, reuseQueryIndex);
}

static bool parseFilterPredLayoutSemanticKey(llvm::StringRef semanticKey, unsigned& predIndex) {
   if (parseReuseFilterPredSemanticKey(semanticKey, predIndex)) return true;
   return static_cast<bool>(parseFilterPredMemberSlot(semanticKey));
}

static llvm::StringRef normalizeColumnIdentifier(llvm::StringRef identifier) {
   return stripMemberSuffix(identifier);
}

struct PayloadColumnSpec {
   std::string semanticKey;
   std::string scope;
   std::string leaf;
   mlir::Type colType;
   bool isJoinKey = false;
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
   if (auto it = reuse.mergedFromThreadLocal.find(canon); it != reuse.mergedFromThreadLocal.end())
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
   if (auto itTL = reuse.mergedFromThreadLocal.find(canonBuf); itTL != reuse.mergedFromThreadLocal.end()) {
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

static mlir::Value resolveJoinMergedBuffer(mlir::Value hivOrBuf, mlir::ModuleOp module, const ModuleReuseInfo& reuse) {
   mlir::Value canon = resolveCacheTargetStateForReuse(hivOrBuf, reuse);
   if (auto it = findReuseMap(reuse.hashIndexedViewFromMergedBuffer, canon);
       it != reuse.hashIndexedViewFromMergedBuffer.end()) {
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
      out.try_emplace(spec.semanticKey, spec);
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

static JoinBufferUnionPlan buildUnionPlan(mlir::Value hivA, mlir::Value hivB, const ModuleReuseInfo& reuseA,
                                        const ModuleReuseInfo& reuseB, mlir::ModuleOp modA, mlir::ModuleOp modB) {
   JoinBufferUnionPlan plan;
   mlir::MLIRContext* ctxA = hivA.getContext();
   auto* subDialectA = ctxA->getLoadedDialect<subop::SubOperatorDialect>();
   auto* tupleDialectA = ctxA->getLoadedDialect<tuples::TupleStreamDialect>();
   assert(subDialectA && tupleDialectA);
   auto& mm = subDialectA->getMemberManager();

   mlir::Value hiv = resolveCacheTargetStateForReuse(hivA, reuseA);
   auto hivTy = mlir::cast<subop::HashIndexedViewType>(hiv.getType());
   subop::CreateHashIndexedView chiv = findCreateHashIndexedViewForState(hiv, reuseA.writerStepsByState);
   assert(chiv && "join superset: HIV must have create_hash_indexed_view writer");
   plan.linkMember = chiv.getLinkMember().getMember();
   plan.hashMember = chiv.getHashMember().getMember();
   llvm::StringRef joinKeyMemberName;
   if (!hivTy.getValueMembers().getMembers().empty()) {
      joinKeyMemberName = mm.getName(hivTy.getValueMembers().getMembers().front());
   }
   llvm::StringRef linkMemberName = mm.getName(plan.linkMember);
   llvm::StringRef hashMemberName = mm.getName(plan.hashMember);

   llvm::StringMap<PayloadColumnSpec> unionCols;
   auto ingestHiv = [&](mlir::Value h, const ModuleReuseInfo& reuse, unsigned reuseQueryIndex) {
      auto& cm = h.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      mlir::Value canon = resolveCacheTargetStateForReuse(h, reuse);
      mlir::Value buf = canon;
      if (auto it = reuse.hashIndexedViewFromMergedBuffer.find(canon);
          it != reuse.hashIndexedViewFromMergedBuffer.end()) {
         buf = it->second;
      }
      subop::ExecutionStepOp buildStep = findBufferBuildStepWithTableMaterialize(buf, reuse);
      if (!buildStep) {
         if (mlir::Operation* def = h.getDefiningOp()) {
            if (auto mod = def->getParentOfType<mlir::ModuleOp>()) buildStep = findBufferBuildStepWithTableScan(mod);
         }
      }
      if (!buildStep) return;
      buildStep.walk([&](subop::MaterializeOp mat) {
         if (!getInnerBufferTypeForMaterializeState(mat.getState().getType()) &&
             !materializeTargetsJoinBuffer(mat, buf, reuse)) {
            return;
         }
         collectPayloadFromMaterialize(mat, linkMemberName, hashMemberName, mm, cm, joinKeyMemberName, unionCols,
                                       reuseQueryIndex);
      });
      if (kEnableReuseStateFilterPredReapply && buildStep) {
         PayloadColumnSpec predSpec;
         predSpec.scope = kReuseFilterPredScope.str();
         predSpec.leaf = llvm::Twine(reuseQueryIndex).str();
         predSpec.colType = mlir::IntegerType::get(h.getContext(), 1);
         predSpec.semanticKey = reuseFilterPredSemanticKey(reuseQueryIndex);
         unionCols.try_emplace(predSpec.semanticKey, predSpec);
      }
      if (buildStep) ingestExternalTableColumnsFromBuildStepScan(buildStep, reuse, unionCols);
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

static void syncCreateHashIndexedViewFromBuffer(subop::CreateHashIndexedView chiv) {
   auto* ctx = chiv.getContext();
   mlir::Value src = chiv.getSource();
   auto bufTy = mlir::dyn_cast<subop::BufferType>(src.getType());
   if (!bufTy) return;
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   subop::Member linkM;
   subop::Member hashM;
   for (subop::Member m : bufTy.getMembers().getMembers()) {
      llvm::StringRef n = mm.getName(m);
      if (n.starts_with("link$")) linkM = m;
      if (n.starts_with("hash$")) hashM = m;
   }
   if (linkM && hashM) {
      chiv.setLinkMemberAttr(subop::MemberAttr::get(ctx, linkM));
      chiv.setHashMemberAttr(subop::MemberAttr::get(ctx, hashM));
   } else {
      linkM = chiv.getLinkMember().getMember();
      hashM = chiv.getHashMember().getMember();
   }
   llvm::SmallVector<subop::Member> vals;
   for (subop::Member m : bufTy.getMembers().getMembers()) {
      if (m == linkM || m == hashM) continue;
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

static subop::MapOp findJoinHashMapBeforeMaterialize(mlir::Block& body, subop::MaterializeOp matOp) {
   subop::MapOp mapOp;
   for (mlir::Operation& op : body.without_terminator()) {
      if (!op.isBeforeInBlock(matOp.getOperation())) continue;
      if (auto m = mlir::dyn_cast<subop::MapOp>(&op)) mapOp = m;
   }
   return mapOp;
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
   module.walk([&](subop::ScanRefsOp scan) {
      if (!opaqueClosureContains(closure, scan.getState())) return;
      auto ref = scan.getRef();
      ref.getColumn().type = entryRefTy;
      scan.setRefAttr(ref);
   });
}

static void syncMapInputColsFromGather(subop::GatherOp gather, tuples::ColumnManager& cm) {
   auto* ctx = gather.getContext();
   llvm::StringMap<tuples::ColumnRefAttr> outRefByKey;
   for (auto& pr : gather.getMapping().getMapping()) {
      auto [scope, leaf] = cm.getName(&pr.second.getColumn());
      outRefByKey[columnSemanticKey(scope, leaf)] = cm.createRef(&pr.second.getColumn());
   }
   if (outRefByKey.empty()) return;
   for (mlir::Operation* user : gather.getRes().getUsers()) {
      auto mapOp = mlir::dyn_cast<subop::MapOp>(user);
      if (!mapOp || mapOp.getStream() != gather.getRes()) continue;
      bool changed = false;
      llvm::SmallVector<mlir::Attribute> newInputs;
      for (auto attr : mapOp.getInputCols()) {
         auto cref = mlir::cast<tuples::ColumnRefAttr>(attr);
         auto [scope, leaf] = cm.getName(&cref.getColumn());
         auto it = outRefByKey.find(columnSemanticKey(scope, leaf));
         if (it != outRefByKey.end() && &cref.getColumn() != &it->second.getColumn()) {
            newInputs.push_back(it->second);
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
                                             subop::HashIndexedViewType producerHiv, bool syncGatherOps) {
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
      if (syncLookupEntryDefToHiv(elem)) op.setElemAttr(elem);
   });
}

static void syncProbeListCarriersInClosure(mlir::ModuleOp module, const llvm::DenseSet<void*>& closureFilter) {
   auto* ctx = module.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   module.walk([&](subop::ExecutionStepOp step) {
      if (!opOperandsOrNestedBlockArgsTouchClosure(step.getOperation(), closureFilter)) return;
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
         auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(gather.getRef().getColumn().type);
         if (!ler) return;
         auto hiv = mlir::dyn_cast<subop::HashIndexedViewType>(ler.getState());
         if (!hiv) return;
         llvm::SmallVector<subop::Member, 8> hivMembers;
         for (subop::Member hm : hiv.getValueMembers().getMembers()) hivMembers.push_back(hm);
         llvm::DenseSet<subop::Member> hivMemberSet(hivMembers.begin(), hivMembers.end());
         llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> out;
         bool mappingChanged = false;
         for (auto& pr : gather.getMapping().getMapping()) {
            subop::Member useMem = pr.first;
            if (!hivMemberSet.contains(useMem)) {
               auto [defScope, defLeaf] = cm.getName(&pr.second.getColumn());
               llvm::StringRef wantLeaf = normalizeColumnIdentifier(defLeaf);
               for (subop::Member hm : hivMembers) {
                  if (normalizeColumnIdentifier(mm.getName(hm)) == wantLeaf) {
                     useMem = hm;
                     mappingChanged = true;
                     break;
                  }
               }
            }
            out.push_back({useMem, pr.second});
         }
         if (mappingChanged) gather.setMappingAttr(subop::ColumnDefMemberMappingAttr::get(ctx, out));
         syncMapInputColsFromGather(gather, cm);
      });
   });
}

static void propagateJoinSupersetColumnAttrsForClosure(mlir::ModuleOp module,
                                                       const llvm::DenseSet<void*>& closureFilter) {
   llvm::DenseSet<const void*> seenHiv;
   module.walk([&](subop::LookupOp op) {
      if (!opOperandsOrNestedBlockArgsTouchClosure(op.getOperation(), closureFilter)) return;
      auto hivTy = mlir::dyn_cast<subop::HashIndexedViewType>(op.getState().getType());
      if (!hivTy) return;
      if (!seenHiv.insert(hivTy.getAsOpaquePointer()).second) return;
      propagateJoinSupersetColumnAttrs(module, &closureFilter, hivTy, true);
   });
   syncProbeListCarriersInClosure(module, closureFilter);
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
   for (size_t pi = 0; pi < peerHivs.size(); ++pi) {
      auto [peerMod, peerHiv] = peerHivs[pi];
      if (!peerMod || !peerHiv) continue;
      const ModuleReuseInfo& reusePeer = *peerReuses[pi];
      mlir::Value peerCanon = resolveCacheTargetStateForReuse(peerHiv, reusePeer);
      mlir::Value peerBuf = peerCanon;
      if (auto it = reusePeer.hashIndexedViewFromMergedBuffer.find(peerCanon);
          it != reusePeer.hashIndexedViewFromMergedBuffer.end()) {
         peerBuf = it->second;
      }
      subop::ExecutionStepOp peerBuild = findBufferBuildStepWithTableMaterialize(peerBuf, reusePeer);
      if (!peerBuild) peerBuild = findBufferBuildStepWithTableScan(peerMod);
      if (peerBuild) mergePeerExternalFromBuildStepScan(mergedDs, haveDs, peerBuild, reusePeer, donorTableName);
   }
   assert(haveDs && "join superset: merged external datasource required for donor table");

   auto lookupPeerColumnType = [&](llvm::StringRef identifier) -> mlir::Type {
      for (size_t pi = 0; pi < peerHivs.size(); ++pi) {
         auto [peerMod, peerHiv] = peerHivs[pi];
         if (!peerMod || !peerHiv) continue;
         const ModuleReuseInfo& reusePeer = *peerReuses[pi];
         mlir::Value peerCanon = resolveCacheTargetStateForReuse(peerHiv, reusePeer);
         mlir::Value peerBuf = peerCanon;
         if (auto it = reusePeer.hashIndexedViewFromMergedBuffer.find(peerCanon);
             it != reusePeer.hashIndexedViewFromMergedBuffer.end()) {
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
   std::optional<subop::GetExternalOp> synthGe =
      resolveGetExternalOpForScannedTable(buildStep, tableState, reuseSynthetic);
   assert(synthGe && "join superset: scan_refs must resolve to get_external in table construction step");
   synthGe->setDescrAttr(mlir::StringAttr::get(ctx, hex));
   refreshTableStateTypesInModule(synthetic, synthGe->getResult(), newTableTy);

   refreshTableStateTypesInModule(synthetic, tableState, newTableTy);
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
         if (auto it = reusePeer.hashIndexedViewFromMergedBuffer.find(peerCanon);
             it != reusePeer.hashIndexedViewFromMergedBuffer.end()) {
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

      mlir::Value mapStream = hashMapOp.getResult();
      mlir::OpBuilder gb(hashMapOp);
      gb.setInsertionPointAfter(hashMapOp);
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
}

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

static void setValueCarrierType(mlir::Value v, subop::HashIndexedViewType producerHiv,
                                subop::HashIndexedViewType consumerHivBeforeAlign) {
   mlir::MLIRContext* ctx = v.getContext();
   if (mlir::Type nt = replaceEmbeddedHivInType(ctx, v.getType(), producerHiv, consumerHivBeforeAlign);
       nt != v.getType()) {
      v.setType(nt);
   }
}

struct ConsumerCachedHivSites {
   llvm::DenseSet<void*> ssaClosure;
   subop::HashIndexedViewType consumerHivBeforeAlign = nullptr;
};

static void addToSsaClosure(mlir::Value v, ConsumerCachedHivSites& sites) {
   if (!v) return;
   sites.ssaClosure.insert(v.getAsOpaquePointer());
}

static bool isJoinProbeCompilerScope(llvm::StringRef scope) { return scope.starts_with("lookup_u_"); }

static CachedJoinBufferLayout layoutFromUnionPlan(subop::HashIndexedViewType producerHiv,
                                                  const JoinBufferUnionPlan& plan) {
   CachedJoinBufferLayout out;
   out.producerHiv = producerHiv;
   out.payloadMembers.assign(plan.payloadMembers.begin(), plan.payloadMembers.end());
   out.payloadColumnTypes.assign(plan.payloadMemberTypes.begin(), plan.payloadMemberTypes.end());
   out.payloadSemanticKeys.reserve(plan.payloadColumns.size());
   for (const PayloadColumnSpec& spec : plan.payloadColumns) {
      out.payloadSemanticKeys.push_back(spec.semanticKey);
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

static void handleScanListSite(subop::ScanListOp scanList, subop::HashIndexedViewType consumerHiv,
                               subop::HashIndexedViewType consumerHivBeforeAlign) {
   setValueCarrierType(scanList.getList(), consumerHiv, consumerHivBeforeAlign);
   auto* ctx = scanList.getContext();
   auto expectedLer = subop::LookupEntryRefType::get(ctx, consumerHiv);
   auto& col = scanList.getElem().getColumn();
   if (mlir::isa<subop::LookupEntryRefType>(col.type) && col.type != expectedLer) col.type = expectedLer;
}

/// From \c cache_get HIV: recurse through port args; at \c scan_list update list/entry-ref types only.
static void traverseConsumerHivUsesFromRoot(mlir::Value root, subop::HashIndexedViewType consumerHiv,
                                            subop::HashIndexedViewType consumerHivBeforeAlign,
                                            ConsumerCachedHivSites& sites) {
   llvm::DenseSet<void*> visited;
   llvm::SmallVector<mlir::Value> worklist;
   auto enqueue = [&](mlir::Value val) {
      if (!val || !visited.insert(val.getAsOpaquePointer()).second) return;
      addToSsaClosure(val, sites);
      worklist.push_back(val);
   };

   setValueCarrierType(root, consumerHiv, consumerHivBeforeAlign);
   enqueue(root);

   while (!worklist.empty()) {
      mlir::Value v = worklist.pop_back_val();
      if (typeEmbedsHashIndexedView(v.getType())) setValueCarrierType(v, consumerHiv, consumerHivBeforeAlign);

      for (mlir::Operation* user : v.getUsers()) {
         if (auto ret = mlir::dyn_cast<subop::ExecutionStepReturnOp>(user)) {
            auto step = mlir::dyn_cast<subop::ExecutionStepOp>(ret->getParentOp());
            if (!step) continue;
            for (unsigned i = 0; i < ret.getNumOperands() && i < step.getNumResults(); ++i) {
               if (ret.getOperand(i) != v) continue;
               setValueCarrierType(step.getResult(i), consumerHiv, consumerHivBeforeAlign);
               enqueue(step.getResult(i));
            }
            continue;
         }

         if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(user)) {
            unsigned i = operandIndexOf(step.getOperation(), v);
            mlir::Block& body = step.getSubOps().front();
            assert(i < body.getNumArguments() && "execution_step operand without block argument");
            mlir::BlockArgument barg = body.getArgument(i);
            setValueCarrierType(barg, consumerHiv, consumerHivBeforeAlign);
            if (i < step.getNumResults()) setValueCarrierType(step.getResult(i), consumerHiv, consumerHivBeforeAlign);
            enqueue(barg);
            if (i < step.getNumResults()) enqueue(step.getResult(i));
            continue;
         }

         if (auto neg = mlir::dyn_cast<subop::NestedExecutionGroupOp>(user)) {
            unsigned i = operandIndexOf(neg.getOperation(), v);
            mlir::Block& body = neg.getSubOps().front();
            assert(i < body.getNumArguments());
            mlir::BlockArgument barg = body.getArgument(i);
            setValueCarrierType(barg, consumerHiv, consumerHivBeforeAlign);
            enqueue(barg);
            continue;
         }

         if (auto lookup = mlir::dyn_cast<subop::LookupOp>(user)) {
            assert(lookup.getState() == v && "lookup state operand must be the HIV value");
            auto* ctx = v.getContext();
            auto listRef = lookup.getRef();
            mlir::Type expectedListTy =
               subop::ListType::get(ctx, subop::LookupEntryRefType::get(ctx, consumerHiv));
            if (listRef.getColumn().type != expectedListTy) listRef.getColumn().type = expectedListTy;
            for (mlir::Operation* streamUser : lookup.getResult().getUsers()) {
               auto nm = mlir::dyn_cast<subop::NestedMapOp>(streamUser);
               assert(nm && "lookup HIV stream must feed nested_map");
               mlir::Region& reg = nm.getRegion();
               assert(!reg.empty());
               for (mlir::BlockArgument barg : reg.front().getArguments()) {
                  if (!typeEmbedsHashIndexedView(barg.getType())) continue;
                  setValueCarrierType(barg, consumerHiv, consumerHivBeforeAlign);
                  enqueue(barg);
               }
            }
            continue;
         }

         if (auto scanList = mlir::dyn_cast<subop::ScanListOp>(user)) {
            assert(scanList.getList() == v && "scan_list list operand must be the carrier value");
            handleScanListSite(scanList, consumerHiv, consumerHivBeforeAlign);
            continue;
         }

         llvm::errs() << "alignConsumer: unhandled HIV use: " << *user << "\n";
         assert(false && "alignConsumer: unhandled HIV/value use (extend traversal)");
      }
   }
}

static void refreshAllEmbeddedHivCarrierTypes(mlir::ModuleOp module, subop::HashIndexedViewType canonicalHiv) {
   auto* ctx = module.getContext();
   for (;;) {
      llvm::SmallVector<std::pair<mlir::Value, mlir::Type>> updates;
      module.walk([&](mlir::Operation* op) {
         auto consider = [&](mlir::Value val) {
            if (!val) return;
            if (mlir::Type nt = replaceEmbeddedHivInType(ctx, val.getType(), canonicalHiv, nullptr);
                nt != val.getType()) {
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

/// Resolve a consumer member for a union payload slot from probe-side gather/materialize mappings.
/// Matches by semantic column identity only (not producer union types).
static std::optional<subop::Member> findConsumerMemberForPayloadSlot(
   llvm::StringRef semanticKey, const llvm::StringMap<ConsumerColumnBinding>& semanticToConsumer,
   subop::MemberManager& mm) {
   if (!semanticKey.starts_with("member$")) {
      if (auto it = semanticToConsumer.find(semanticKey); it != semanticToConsumer.end()) {
         return it->second.member;
      }
      llvm::StringRef wantLeaf = normalizeColumnIdentifier(semanticKeyLeaf(semanticKey));
      llvm::SmallVector<subop::Member, 4> byLeaf;
      for (const auto& entry : semanticToConsumer) {
         if (normalizeColumnIdentifier(semanticKeyLeaf(entry.getKey())) != wantLeaf) continue;
         byLeaf.push_back(entry.second.member);
      }
      if (byLeaf.size() == 1) return byLeaf.front();
   }

   // Layout metadata may still carry producer `member$N` labels; fall back to unique probe column types.
   if (semanticKey == "member$0") {
      for (const auto& entry : semanticToConsumer) {
         if (mm.getName(entry.second.member) == "member$0") return entry.second.member;
      }
   }
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

   consumer.walk([&](subop::GatherOp gather) {
      auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(gather.getRef().getColumn().type);
      if (!ler || !mlir::isa<subop::HashIndexedViewType>(ler.getState())) return;
      for (auto [mem, def] : gather.getMapping().getMapping()) {
         auto [scope, leaf] = cm.getName(&def.getColumn());
         record(mem, scope, leaf, def.getColumn().type);
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

/// Per-consumer HIV for \c cache_get: payload physical order follows the cached producer layout, but each slot
/// reuses the consumer's own \c member$N name and column type when that query already had the column; union-only
/// columns are inserted with producer types under fresh \c member$N slots (no type overwrite on existing members).
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
   llvm::DenseSet<subop::Member> excludedProducerPredSlots;
   llvm::StringMap<subop::Member> assignedBySemantic;
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
         if (consumerReuseQueryIndex && predIdx != *consumerReuseQueryIndex) {
            excludedProducerPredSlots.insert(producerMem);
            continue;
         }
         subop::Member consumerMem = makeOrGetPredMemberForSlot(ctx, predIdx);
         assignedBySemantic[semKey] = consumerMem;
         producerPayloadToConsumer[producerMem] = consumerMem;
         continue;
      }

      std::optional<subop::Member> reused = findConsumerMemberForPayloadSlot(semKey, semanticToConsumer, mm);
      if (!reused && i == 0) {
         reused = findConsumerMemberByName("member$0", consumerHivBeforeAlign);
      }
      subop::Member consumerMem = ensureConsumerMember(reused, producerSlotTy);
      assignedBySemantic[semKey] = consumerMem;
      producerPayloadToConsumer[layout.payloadMembers[i]] = consumerMem;
   }

   llvm::SmallVector<subop::Member> valueMembers;
   valueMembers.reserve(producerHiv.getValueMembers().getMembers().size());
   for (subop::Member m : producerHiv.getValueMembers().getMembers()) {
      if (excludedProducerPredSlots.contains(m)) continue;
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
      if (excludedProducerPredSlots.contains(m)) continue;
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

static void patchSyntheticJoinBufferFilterPredsImpl(
   mlir::ModuleOp synthetic, mlir::ModuleOp query0, mlir::ModuleOp query1,
   llvm::ArrayRef<CrossQueryStateMatchPair> matches, llvm::ArrayRef<CacheTarget> targetsInSynthetic,
   const CachedJoinBufferLayoutsByKey& layoutsByKey) {
   if (!kEnableReuseStateFilterPredReapply) return;

   auto reuse0 = collectModuleReuseInfo(query0);
   auto reuse1 = collectModuleReuseInfo(query1);
   auto reuseSynthetic = collectModuleReuseInfo(synthetic);

   llvm::DenseMap<uint64_t, const CrossQueryStateMatchPair*> matchByKey;
   for (const auto& m : matches) {
      if (!m.stateA || !m.stateB) continue;
      matchByKey[m.cacheKey] = &m;
   }

   for (const CacheTarget& t : targetsInSynthetic) {
      auto itL = layoutsByKey.find(t.cacheKey);
      auto itM = matchByKey.find(t.cacheKey);
      if (itL == layoutsByKey.end() || itM == matchByKey.end()) continue;
      const CachedJoinBufferLayout& layout = itL->second;
      const CrossQueryStateMatchPair& match = *itM->second;

      mlir::Value synthHiv = t.state;
      if (!mlir::isa<subop::HashIndexedViewType>(synthHiv.getType())) continue;
      mlir::Value mergedBuf = synthHiv;
      if (auto it = reuseSynthetic.hashIndexedViewFromMergedBuffer.find(synthHiv);
          it != reuseSynthetic.hashIndexedViewFromMergedBuffer.end()) {
         mergedBuf = it->second;
      }
      subop::ExecutionStepOp buildStep = findBufferBuildStepWithTableMaterialize(mergedBuf, reuseSynthetic);
      if (!buildStep) continue;

      auto& mm = synthetic.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
      mlir::Value hivs[] = {resolveCacheTargetStateForReuse(match.stateA, reuse0),
                            resolveCacheTargetStateForReuse(match.stateB, reuse1)};
      ModuleReuseInfo* reuses[] = {&reuse0, &reuse1};

      for (size_t i = 0; i < layout.payloadSemanticKeys.size(); ++i) {
         unsigned qIdx = 0;
         if (!parseReuseFilterPredSemanticKey(layout.payloadSemanticKeys[i], qIdx)) continue;
         llvm::StringRef predName = mm.getName(layout.payloadMembers[i]);
         auto filters = decodeFiltersFromTableScanInExecutionStep(buildStep);
         if (filters.empty()) {
            filters = decodeFiltersForStateFromWriterSteps(hivs[qIdx], *reuses[qIdx]);
            filters = restrictFiltersToTableScanInExecutionStep(buildStep, filters);
         }
         insertWriteSidePredIntoBufferConstructionStepForPredMember(buildStep, filters, predName);
      }
   }
}

} // namespace

void patchSyntheticJoinBufferFilterPredsFromMatchedQueries(
   mlir::ModuleOp synthetic, mlir::ModuleOp query0, mlir::ModuleOp query1,
   llvm::ArrayRef<CrossQueryStateMatchPair> matches, llvm::ArrayRef<CacheTarget> targetsInSynthetic,
   const CachedJoinBufferLayoutsByKey& layoutsByKey) {
   patchSyntheticJoinBufferFilterPredsImpl(synthetic, query0, query1, matches, targetsInSynthetic, layoutsByKey);
}

void alignConsumerModulesToCachedJoinLayout(mlir::ModuleOp consumer, const CachedJoinBufferLayout& layout,
                                            std::optional<uint64_t> cacheKey,
                                            std::optional<unsigned> consumerReuseQueryIndex) {
   if (!layout.producerHiv) return;

   subop::HashIndexedViewType producerHiv = layout.producerHiv;
   ConsumerCachedHivSites sites;
   subop::HashIndexedViewType consumerHiv = nullptr;

   consumer.walk([&](subop::CacheGetOp get) {
      if (cacheKey && static_cast<uint64_t>(get.getKey()) != *cacheKey) return;

      sites.consumerHivBeforeAlign = mlir::dyn_cast<subop::HashIndexedViewType>(get.getResult().getType());
      CachedJoinBufferLayout consumerLayout;
      consumerHiv = buildConsumerAlignedHivType(consumer, producerHiv, layout, sites.consumerHivBeforeAlign,
                                                  consumerReuseQueryIndex, consumerLayout);
      get.getResult().setType(consumerHiv);
      traverseConsumerHivUsesFromRoot(get.getResult(), consumerHiv, sites.consumerHivBeforeAlign, sites);
   });

   if (sites.ssaClosure.empty() || !consumerHiv) return;

   expandClosureThroughExecutionStepPorts(consumer, sites.ssaClosure);
   syncExecutionStepPortsForModule(consumer, &sites.ssaClosure);
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

      mlir::Value hivA = resolveCacheTargetStateForReuse(m.stateA, reuse0);
      mlir::Value hivB = resolveCacheTargetStateForReuse(m.stateB, reuse1);
      if (!mlir::isa<subop::HashIndexedViewType>(hivA.getType()) ||
          !mlir::isa<subop::HashIndexedViewType>(hivB.getType())) {
         continue;
      }

      JoinBufferUnionPlan plan = buildUnionPlan(hivA, hivB, reuse0, reuse1, query0, query1);
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

   if (outLayouts) {
      for (const CacheTarget& t : targetsInSynthetic) {
         if (outLayouts->contains(t.cacheKey)) continue;
         auto hivTy = mlir::dyn_cast<subop::HashIndexedViewType>(t.state.getType());
         if (!hivTy) continue;
         CachedJoinBufferLayout layout;
         layout.producerHiv = hivTy;
         auto& mm = synthetic.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
         for (subop::Member m : hivTy.getValueMembers().getMembers()) {
            layout.payloadMembers.push_back(m);
            layout.payloadColumnTypes.push_back(mm.getType(m));
            layout.payloadSemanticKeys.push_back(std::string(mm.getName(m)));
         }
         (*outLayouts)[t.cacheKey] = std::move(layout);
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

void resyncConsumerCachedHivCarrierTypesFromCacheGet(mlir::ModuleOp consumer,
                                                     std::optional<uint64_t> cacheKey) {
   consumer.walk([&](subop::CacheGetOp get) {
      if (cacheKey && static_cast<uint64_t>(get.getKey()) != *cacheKey) return;
      auto canonicalHiv = mlir::dyn_cast<subop::HashIndexedViewType>(get.getResult().getType());
      if (!canonicalHiv) return;

      ConsumerCachedHivSites sites;
      traverseConsumerHivUsesFromRoot(get.getResult(), canonicalHiv, nullptr, sites);
      expandClosureThroughExecutionStepPorts(consumer, sites.ssaClosure);
      syncExecutionStepPortsForModule(consumer, &sites.ssaClosure);
      // `filter_pred` insertion can touch carriers outside the cache_get use closure.
      refreshAllEmbeddedHivCarrierTypes(consumer, canonicalHiv);
      syncExecutionStepPortsForModule(consumer, &sites.ssaClosure);
      propagateJoinSupersetColumnAttrsForClosure(consumer, sites.ssaClosure);
      syncExecutionStepPortsForModule(consumer, nullptr);
   });
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
         size_t idx = 0;
         for (subop::Member m : hivTy.getValueMembers().getMembers()) {
            layout.payloadMembers.push_back(m);
            layout.payloadColumnTypes.push_back(mm.getType(m));
            if (prev && idx < prev->payloadSemanticKeys.size()) {
               layout.payloadSemanticKeys.push_back(prev->payloadSemanticKeys[idx]);
            } else {
               layout.payloadSemanticKeys.push_back(std::string(mm.getName(m)));
            }
            ++idx;
         }
         layoutsByKey[t.cacheKey] = std::move(layout);
      });
   }
}

} // namespace lingodb::compiler::dialect::subop
