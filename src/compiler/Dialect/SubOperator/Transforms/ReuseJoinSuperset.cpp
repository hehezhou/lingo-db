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
#include "lingodb/compiler/Dialect/SubOperator/Utils.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateExtraction.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnCreationAnalysis.h"
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
#include <limits>
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

static llvm::StringRef columnSemanticKeyLeaf(llvm::StringRef semantic) {
   size_t pos = semantic.rfind('\x1f');
   if (pos == llvm::StringRef::npos) return semantic;
   return semantic.drop_front(pos + 1);
}

static llvm::StringRef columnSemanticKeyScope(llvm::StringRef semantic) {
   size_t pos = semantic.rfind('\x1f');
   if (pos == llvm::StringRef::npos) return {};
   return semantic.take_front(pos);
}

static std::optional<subop::Member> lookupMemberBySemanticOrUniqueLeaf(
   const llvm::StringMap<subop::Member>& bySemantic, llvm::StringRef semantic) {
   if (auto it = bySemantic.find(semantic); it != bySemantic.end()) return it->second;
   constexpr llvm::StringRef notNullSuffix = "__notnull";
   llvm::StringRef scope = columnSemanticKeyScope(semantic);
   llvm::StringRef leaf = columnSemanticKeyLeaf(semantic);
   if (!scope.empty()) {
      if (leaf.ends_with(notNullSuffix)) {
         std::string normalized = columnSemanticKey(scope, leaf.drop_back(notNullSuffix.size()));
         if (auto it = bySemantic.find(normalized); it != bySemantic.end()) return it->second;
      } else {
         std::string notNull = columnSemanticKey(scope, (leaf + notNullSuffix).str());
         if (auto it = bySemantic.find(notNull); it != bySemantic.end()) return it->second;
      }
   }
   std::optional<subop::Member> found;
   for (const auto& it : bySemantic) {
      if (columnSemanticKeyLeaf(it.getKey()) != leaf) continue;
      assert(!found && "probe payload semantic leaf must map to one aligned member");
      found = it.second;
   }
   return found;
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

static unsigned reuseSlotForEntry(const CrossQueryStateMatchEntry& entry) {
   if (entry.reuseSlot != std::numeric_limits<unsigned>::max()) return entry.reuseSlot;
   assert(entry.query >= 0 && "state reuse filter predicate slot must be backed by a query id");
   return static_cast<unsigned>(entry.query);
}

static unsigned aggregateReuseSlotForEntry(const CrossQueryStateMatchEntry& entry) {
   if (entry.reuseSlot != std::numeric_limits<unsigned>::max()) return entry.reuseSlot;
   assert(entry.query >= 0 && "aggregate reuse filter predicate slot must be backed by a query id");
   return static_cast<unsigned>(entry.query);
}

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

struct FilterPredLayoutSlot {
   bool isPred = false;
   bool isUnion = false;
   unsigned slot = 0;
};

static FilterPredLayoutSlot classifyFilterPredLayoutSlot(llvm::StringRef semanticKey,
                                                         llvm::StringRef memberName) {
   FilterPredLayoutSlot out;
   if (parseReuseFilterPredUnionSemanticKey(semanticKey, out.slot)) {
      out.isPred = true;
      out.isUnion = true;
      return out;
   }
   if (parseReuseFilterPredSemanticKey(semanticKey, out.slot)) {
      out.isPred = true;
      return out;
   }
   if (auto slot = parseFilterPredMemberSlot(memberName)) {
      out.isPred = true;
      out.slot = *slot;
   }
   return out;
}

static llvm::StringRef normalizeColumnIdentifier(llvm::StringRef identifier) {
   return stripMemberSuffix(identifier);
}

static llvm::StringRef sourceColumnIdentifierForPayloadLeaf(llvm::StringRef leaf) {
   constexpr llvm::StringRef notNullSuffix = "__notnull";
   if (leaf.ends_with(notNullSuffix)) return leaf.drop_back(notNullSuffix.size());
   return leaf;
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
   llvm::SmallVector<unsigned, 4> queryIndices;
   llvm::DenseMap<unsigned, subop::Member> sourceMemberByQueryIndex;
   bool fromExternalTable = false;
   unsigned stableOrder = 0;
};

struct JoinBufferUnionPlan {
   subop::Member linkMember;
   subop::Member hashMember;
   llvm::SmallVector<PayloadColumnSpec, 8> payloadColumns;
   llvm::SmallVector<mlir::Type, 8> payloadMemberTypes;
   /// Buffer/HIV payload members aligned with \p payloadColumns (semantic union, not slot `member$N`).
   llvm::SmallVector<subop::Member, 8> payloadMembers;
};

static void refreshPayloadMemberTypes(JoinBufferUnionPlan& plan) {
   plan.payloadMemberTypes.clear();
   plan.payloadMemberTypes.reserve(plan.payloadColumns.size());
   for (const PayloadColumnSpec& spec : plan.payloadColumns) plan.payloadMemberTypes.push_back(spec.colType);
}

static unsigned nextPayloadStableOrder(llvm::ArrayRef<PayloadColumnSpec> specs) {
   unsigned next = 0;
   for (const PayloadColumnSpec& spec : specs) next = std::max(next, spec.stableOrder + 1);
   return next;
}

static unsigned nextPayloadStableOrder(const llvm::DenseMap<uint64_t, PayloadColumnSpec>& specs) {
   unsigned next = 0;
   for (const auto& it : specs) next = std::max(next, it.second.stableOrder + 1);
   return next;
}

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
   predSpec.stableOrder = nextPayloadStableOrder(plan.payloadColumns);
   plan.payloadColumns.push_back(std::move(predSpec));
   refreshPayloadMemberTypes(plan);
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

static bool stepHasSourceScanRefs(subop::ExecutionStepOp step) {
   bool hasScanRefs = false;
   step.walk([&](subop::ScanRefsOp scan) {
      (void)scan;
      hasScanRefs = true;
   });
   return hasScanRefs;
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
         if (!stepHasSourceScanRefs(step)) continue;
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
            if (!stepHasSourceScanRefs(step)) continue;
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
      bool hasSourceScan = stepHasSourceScanRefs(step);
      bool hasJoinMat = false;
      step->walk([&](subop::MaterializeOp mat) {
         if (materializeTargetsJoinBuffer(mat, mergedBuffer, reuse)) hasJoinMat = true;
      });
      if (hasSourceScan && hasJoinMat) return step;
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
         if (mlir::isa<subop::TableType, subop::SharedTableType>(scan.getState().getType())) hasTableScan = true;
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
                                    std::optional<unsigned> ownerQueryIndex = std::nullopt);

static void collectPayloadFromMaterialize(
   subop::MaterializeOp mat, llvm::StringRef linkMemberName, llvm::StringRef hashMemberName, subop::MemberManager& mm,
   lingodb::compiler::dialect::tuples::ColumnManager& cm, llvm::StringRef joinKeyMemberName,
   llvm::DenseMap<uint64_t, PayloadColumnSpec>& out, std::optional<unsigned> ownerQueryIndex,
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
      if (ownerQueryIndex) spec.sourceMemberByQueryIndex[*ownerQueryIndex] = member;
      insertPayloadColumnSpec(out, std::move(spec), ownerQueryIndex);
   }
}

static mlir::Type memberTypeForIdentifier(subop::TableType tableTy, subop::MemberManager& mm, llvm::StringRef identifier);
static subop::Member tableMemberForIdentifier(subop::TableType tableTy, subop::MemberManager& mm,
                                              llvm::StringRef identifier);
static subop::Member tableMemberForIdentifier(mlir::Type tableLikeTy, subop::MemberManager& mm,
                                              llvm::StringRef identifier);

static ExternalDatasourceProperty mergeExternalDatasource(const ExternalDatasourceProperty& a,
                                                        const ExternalDatasourceProperty& b);

static mlir::Type reconcileExternalColumnTypes(mlir::Type a, mlir::Type b) {
   if (!a) return b;
   if (!b) return a;
   if (a == b) return a;
   if (auto nullableA = mlir::dyn_cast<db::NullableType>(a)) {
      if (nullableA.getType() == b) return a;
   }
   if (auto nullableB = mlir::dyn_cast<db::NullableType>(b)) {
      if (nullableB.getType() == a) return b;
   }
   llvm_unreachable("join superset: incompatible external column types for merged table");
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

static void insertPayloadColumnSpec(llvm::DenseMap<uint64_t, PayloadColumnSpec>& unionCols,
                                    PayloadColumnSpec spec,
                                    std::optional<unsigned> ownerQueryIndex) {
   assert(spec.semanticHash && "payload spec must carry a hash identity");
   auto markSide = [&](PayloadColumnSpec& existing) {
      if (!ownerQueryIndex) return;
      if (!llvm::is_contained(existing.queryIndices, *ownerQueryIndex))
         existing.queryIndices.push_back(*ownerQueryIndex);
   };
   auto mergeSpec = [&](PayloadColumnSpec& existing, const PayloadColumnSpec& incoming) {
      markSide(existing);
      existing.isJoinKey |= incoming.isJoinKey;
      existing.fromExternalTable |= incoming.fromExternalTable;
      for (const auto& entry : incoming.sourceMemberByQueryIndex)
         existing.sourceMemberByQueryIndex.try_emplace(entry.first, entry.second);
   };

   auto it = unionCols.find(spec.semanticHash);
   if (it != unionCols.end()) {
      mergeSpec(it->second, spec);
      return;
   }

   markSide(spec);
   spec.stableOrder = nextPayloadStableOrder(unionCols);
   unionCols.try_emplace(spec.semanticHash, std::move(spec));
}

static subop::ScanRefsOp findUniqueTableScanRefsInStep(subop::ExecutionStepOp buildStep) {
   subop::ScanRefsOp found;
	bool multiple = false;
	buildStep.walk([&](subop::ScanRefsOp scanOp) {
	   if (!mlir::isa<subop::TableType, subop::SharedTableType>(scanOp.getState().getType())) return;
	   if (found && found != scanOp) {
	      multiple = true;
	      return;
      }
      found = scanOp;
   });
   if (multiple) return {};
   return found;
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
   return memberTypeForIdentifier(resolved->tableType, mm, identifier);
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
                                          bool enableFilterPredReuse,
                                          llvm::ArrayRef<std::pair<mlir::ModuleOp, mlir::Value>> extraPeerHivs = {},
                                          llvm::ArrayRef<const ModuleReuseInfo*> extraPeerReuses = {},
                                          llvm::ArrayRef<unsigned> extraPeerReuseIndices = {}) {
   assert(extraPeerHivs.size() == extraPeerReuses.size() &&
          extraPeerHivs.size() == extraPeerReuseIndices.size() &&
          "join superset: extra peer HIVs/reuse metadata/reuse slots must align");
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
                        unsigned ownerQueryIndex) {
      llvm::DenseMap<const void*, uint64_t> columnHashes = collectStateConstructionColumnHashes(mod, h);
      auto* hCtx = h.getContext();
      auto& hMm = hCtx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
      auto& hCm = hCtx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      mlir::Value mergedBuf = resolveJoinMergedBuffer(h, mod, reuse);
      subop::ExecutionStepOp buildStep = findJoinBufferBuildStepForHiv(mod, h, reuse);
      assert(buildStep && "join superset: build step required for ingestHiv");
      buildStep.walk([&](subop::MaterializeOp mat) {
         if (!materializeTargetsJoinBuffer(mat, mergedBuf, reuse)) return;
         collectPayloadFromMaterialize(mat, linkMemberName, hashMemberName, hMm, hCm, joinKeyMemberName, unionCols,
                                       ownerQueryIndex, ownerQueryIndex, columnHashes);
      });
      if (enableFilterPredReuse) {
         PayloadColumnSpec predSpec;
         predSpec.scope = kReuseFilterPredScope.str();
         predSpec.leaf = llvm::Twine(ownerQueryIndex).str();
         predSpec.colType = mlir::IntegerType::get(h.getContext(), 1);
         predSpec.semanticKey = reuseFilterPredSemanticKey(ownerQueryIndex);
         predSpec.semanticHash = payloadSyntheticColumnHash(predSpec.scope, predSpec.leaf, predSpec.colType);
         predSpec.stableOrder = nextPayloadStableOrder(unionCols);
         unionCols.try_emplace(predSpec.semanticHash, predSpec);
      }
   };
   ingestHiv(hivA, reuseA, modA, queryIndexA);
   ingestHiv(hivB, reuseB, modB, queryIndexB);
   for (size_t i = 0; i < extraPeerHivs.size(); ++i) {
      auto [mod, hiv] = extraPeerHivs[i];
      assert(mod && hiv && extraPeerReuses[i] && "join superset: extra peer HIV must be resolved");
      ingestHiv(hiv, *extraPeerReuses[i], mod, extraPeerReuseIndices[i]);
   }
   if (enableFilterPredReuse) {
      llvm::SmallVector<unsigned, 8> queryIndices{queryIndexA, queryIndexB};
      queryIndices.append(extraPeerReuseIndices.begin(), extraPeerReuseIndices.end());
      unsigned unionSlot = filterPredUnionSlotForQueryIndices(queryIndices);
      PayloadColumnSpec predSpec;
      predSpec.scope = kReuseFilterPredUnionScope.str();
      predSpec.leaf = llvm::Twine(unionSlot).str();
      predSpec.colType = mlir::IntegerType::get(hivA.getContext(), 1);
      predSpec.semanticKey = reuseFilterPredUnionSemanticKey(unionSlot);
      predSpec.semanticHash = payloadSyntheticColumnHash(predSpec.scope, predSpec.leaf, predSpec.colType);
      predSpec.stableOrder = nextPayloadStableOrder(unionCols);
      unionCols.try_emplace(predSpec.semanticHash, predSpec);
   }

   llvm::SmallVector<PayloadColumnSpec*, 8> ordered;
   ordered.reserve(unionCols.size());
   for (auto& it : unionCols) ordered.push_back(&it.second);
   llvm::sort(ordered, [](const PayloadColumnSpec* a, const PayloadColumnSpec* b) {
      if (a->isJoinKey != b->isJoinKey) return a->isJoinKey > b->isJoinKey;
      if (a->stableOrder != b->stableOrder) return a->stableOrder < b->stableOrder;
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

static void sortUniqueSlots(llvm::SmallVectorImpl<unsigned>& slots) {
   llvm::sort(slots);
   slots.erase(std::unique(slots.begin(), slots.end()), slots.end());
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
   mlir::Operation* scanLike = nullptr;
   tuples::ColumnRefAttr sourceRef;
   mlir::Type sourceStateType;
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

static std::optional<ResidualTableFilter> traceSingleUseStreamToScanLikeInStep(
   subop::ExecutionStepOp step, mlir::Operation* user, mlir::Value stream,
   subop::MapOp predMap, subop::FilterOp filter) {
   for (;;) {
      mlir::Operation* def = stream.getDefiningOp();
      if (!def) return std::nullopt;
      if (!streamValueHasOnlyUseByWithinStep(stream, user, step)) return std::nullopt;
      if (auto scan = mlir::dyn_cast<subop::ScanRefsOp>(def)) {
         auto& cm = scan.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
         return ResidualTableFilter{scan.getOperation(), cm.createRef(&scan.getRef().getColumn()),
                                    scan.getState().getType(), predMap, filter};
      }
      if (auto scan = mlir::dyn_cast<subop::ScanListOp>(def)) {
         auto listTy = mlir::dyn_cast<subop::ListType>(scan.getList().getType());
         if (!listTy) return std::nullopt;
         auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(listTy.getT());
         if (!ler) return std::nullopt;
         auto& cm = scan.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
         return ResidualTableFilter{scan.getOperation(), cm.createRef(&scan.getElem().getColumn()),
                                    ler.getState(), predMap, filter};
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
      if (auto filterOp = mlir::dyn_cast<subop::FilterOp>(def)) {
         user = def;
         stream = filterOp.getStream();
         continue;
      }
      if (auto rename = mlir::dyn_cast<subop::RenamingOp>(def)) {
         user = def;
         stream = rename.getStream();
         continue;
      }
      return std::nullopt;
   }
}

static llvm::DenseSet<unsigned> residualPredicateInputIndices(subop::MapOp peerMap,
                                                              subop::FilterOp peerFilter);
static subop::Member sourceMemberForIdentifier(mlir::Type sourceStateType,
                                               subop::MemberManager& mm,
                                               llvm::StringRef identifier);
static std::optional<ResidualTableFilter> findScanLikeSourceInBuildStep(subop::ExecutionStepOp step);

static bool filterConditionsComeFromGather(subop::FilterOp filter, subop::GatherOp gather) {
   llvm::DenseSet<const void*> gatheredCols;
   for (auto& [member, def] : gather.getMapping().getMapping()) {
      (void)member;
      gatheredCols.insert(&def.getColumn());
   }
   for (auto attr : filter.getConditions()) {
      auto ref = mlir::cast<tuples::ColumnRefAttr>(attr);
      if (!gatheredCols.contains(&ref.getColumn())) return false;
   }
   return true;
}

static subop::MapOp normalizeGatherResidualFilterToMap(subop::FilterOp filter,
                                                       subop::GatherOp gather) {
   auto* ctx = filter.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();

   llvm::DenseMap<const tuples::Column*, subop::Member> memberByColumn;
   llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> gatherPairs;
   gatherPairs.reserve(gather.getMapping().getMapping().size());
   for (auto& [member, def] : gather.getMapping().getMapping()) {
      memberByColumn[&def.getColumn()] = member;
      gatherPairs.push_back({member, def});
   }

   llvm::DenseSet<const tuples::Column*> conditionColumns;
   for (auto attr : filter.getConditions()) {
      auto ref = mlir::cast<tuples::ColumnRefAttr>(attr);
      conditionColumns.insert(&ref.getColumn());
   }

   llvm::SmallVector<mlir::Attribute> mapInputs;
   llvm::SmallVector<mlir::Attribute> computedCols;
   llvm::DenseMap<const tuples::Column*, tuples::ColumnRefAttr> normalizedInputByOldColumn;
   llvm::DenseMap<const tuples::Column*, tuples::ColumnRefAttr> outputByOldColumn;
   for (auto& pair : gatherPairs) {
      if (!conditionColumns.contains(&pair.second.getColumn())) continue;
      const tuples::Column* oldColumn = &pair.second.getColumn();
      subop::Member member = pair.first;
      std::string memberName = mm.getName(member);
      tuples::ColumnDefAttr inputDef = cm.createDef(cm.getUniqueScope("residual_gather_pred_input"),
                                                    memberName);
      inputDef.getColumn().type = pair.second.getColumn().type;
      normalizedInputByOldColumn[oldColumn] = cm.createRef(&inputDef.getColumn());
      pair.second = inputDef;

      tuples::ColumnDefAttr outDef = cm.createDef(cm.getUniqueScope("residual_gather_pred"),
                                                  memberName);
      outDef.getColumn().type = inputDef.getColumn().type;
      computedCols.push_back(outDef);
      mapInputs.push_back(cm.createRef(&inputDef.getColumn()));
      outputByOldColumn[oldColumn] = cm.createRef(&outDef.getColumn());
   }

   assert(!computedCols.empty() && "gather residual filter must have at least one predicate column");
   gather.setMappingAttr(subop::ColumnDefMemberMappingAttr::get(ctx, gatherPairs));

   mlir::OpBuilder b(filter);
   b.setInsertionPointAfter(gather);
   auto map = b.create<subop::MapOp>(filter.getLoc(), tuples::TupleStreamType::get(ctx),
                                     gather.getResult(), b.getArrayAttr(computedCols),
                                     b.getArrayAttr(mapInputs));
   mlir::Block* block = new mlir::Block();
   for (auto attr : mapInputs) {
      auto ref = mlir::cast<tuples::ColumnRefAttr>(attr);
      block->addArgument(ref.getColumn().type, filter.getLoc());
   }
   map.getFn().push_back(block);
   mlir::OpBuilder rb(ctx);
   rb.setInsertionPointToStart(block);
   llvm::SmallVector<mlir::Value> retVals;
   for (mlir::BlockArgument arg : block->getArguments()) retVals.push_back(arg);
   rb.create<tuples::ReturnOp>(filter.getLoc(), retVals);

   llvm::SmallVector<mlir::Attribute> newConditions;
   newConditions.reserve(filter.getConditions().size());
   for (auto attr : filter.getConditions()) {
      auto ref = mlir::cast<tuples::ColumnRefAttr>(attr);
      auto itMember = memberByColumn.find(&ref.getColumn());
      assert(itMember != memberByColumn.end() && "gather residual condition must map to a source member");
      (void)itMember;
      auto it = normalizedInputByOldColumn.find(&ref.getColumn());
      assert(it != normalizedInputByOldColumn.end() && "gather residual condition was not normalized");
      (void)it;
      auto outIt = outputByOldColumn.find(&ref.getColumn());
      assert(outIt != outputByOldColumn.end() && "gather residual normalized input must have map output");
      newConditions.push_back(outIt->second);
   }
   filter.setConditionsAttr(b.getArrayAttr(newConditions));
   filter->setOperand(0, map.getResult());
   return map;
}

static std::optional<ResidualTableFilter> findResidualTableFilterInBuildStep(subop::ExecutionStepOp step) {
   std::optional<ResidualTableFilter> found;
   step.walk([&](subop::FilterOp filter) {
      if (found) return;
      auto map = mlir::dyn_cast_or_null<subop::MapOp>(filter.getStream().getDefiningOp());
      if (!map) {
         auto gather = mlir::dyn_cast_or_null<subop::GatherOp>(filter.getStream().getDefiningOp());
         if (!gather) return;
         if (!streamValueHasOnlyUseByWithinStep(gather.getResult(), filter.getOperation(), step)) return;
         if (!filterConditionsComeFromGather(filter, gather)) return;
         map = normalizeGatherResidualFilterToMap(filter, gather);
      }
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
      auto traced = traceSingleUseStreamToScanLikeInStep(step, map.getOperation(), map.getStream(), map, filter);
      if (!traced) return;
      auto& peerCm = map.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      auto& mm = map.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
      llvm::DenseSet<unsigned> usedInputs = residualPredicateInputIndices(map, filter);
      for (unsigned inputIdx : usedInputs) {
         auto peerRef = mlir::cast<tuples::ColumnRefAttr>(map.getInputCols()[inputIdx]);
         auto [scope, leaf] = peerCm.getName(&peerRef.getColumn());
         (void)scope;
         if (!sourceMemberForIdentifier(traced->sourceStateType, mm, leaf)) return;
      }
      found = *traced;
   });
   return found;
}

static llvm::SmallVector<ResidualTableFilter, 4>
collectResidualTableFiltersInBuildStep(subop::ExecutionStepOp step) {
   llvm::SmallVector<ResidualTableFilter, 4> found;
   step.walk([&](subop::FilterOp filter) {
      auto map = mlir::dyn_cast_or_null<subop::MapOp>(filter.getStream().getDefiningOp());
      if (!map) {
         auto gather = mlir::dyn_cast_or_null<subop::GatherOp>(filter.getStream().getDefiningOp());
         if (!gather) return;
         if (!streamValueHasOnlyUseByWithinStep(gather.getResult(), filter.getOperation(), step)) return;
         if (!filterConditionsComeFromGather(filter, gather)) return;
         map = normalizeGatherResidualFilterToMap(filter, gather);
      }
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
      auto traced = traceSingleUseStreamToScanLikeInStep(step, map.getOperation(), map.getStream(), map, filter);
      if (!traced) return;
      auto& peerCm = map.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      auto& mm = map.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
      llvm::DenseSet<unsigned> usedInputs = residualPredicateInputIndices(map, filter);
      for (unsigned inputIdx : usedInputs) {
         auto peerRef = mlir::cast<tuples::ColumnRefAttr>(map.getInputCols()[inputIdx]);
         auto [scope, leaf] = peerCm.getName(&peerRef.getColumn());
         (void)scope;
         if (!sourceMemberForIdentifier(traced->sourceStateType, mm, leaf)) return;
      }
      found.push_back(*traced);
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

static bool isReuseGeneratedUpstreamPredicateColumn(tuples::ColumnRefAttr ref,
                                                    tuples::ColumnManager& cm) {
   if (!ref) return false;
   auto [scope, name] = cm.getName(&ref.getColumn());
   llvm::StringRef scopeRef(scope);
   return scopeRef.starts_with("reuse_upstream_pred") ||
          scopeRef.starts_with("reuse_inherited_combined_pred");
}

static bool isReuseCombinedPredicateColumn(tuples::ColumnRefAttr ref,
                                           tuples::ColumnManager& cm) {
   if (!ref) return false;
   auto [scope, name] = cm.getName(&ref.getColumn());
   (void)name;
   return llvm::StringRef(scope).starts_with("reuse_combined_pred");
}

static mlir::BlockArgument appendMapInputColumn(subop::MapOp map, tuples::ColumnRefAttr ref) {
   llvm::SmallVector<mlir::Attribute> inputs(map.getInputCols().begin(), map.getInputCols().end());
   inputs.push_back(ref);
   map.setInputColsAttr(mlir::ArrayAttr::get(map.getContext(), inputs));
   return map.getFn().front().addArgument(ref.getColumn().type, map.getLoc());
}

static subop::StateMembersAttr sourceValueMembersForResidualGather(mlir::Type sourceStateType) {
   if (auto tableTy = mlir::dyn_cast<subop::TableType>(sourceStateType)) return tableTy.getMembers();
   if (auto tableTy = mlir::dyn_cast<subop::SharedTableType>(sourceStateType)) return tableTy.getTableMembers();
   if (auto bufferTy = mlir::dyn_cast<subop::BufferType>(sourceStateType)) return bufferTy.getMembers();
   if (auto threadLocalTy = mlir::dyn_cast<subop::ThreadLocalType>(sourceStateType))
      if (auto bufferTy = mlir::dyn_cast<subop::BufferType>(threadLocalTy.getWrapped()))
         return bufferTy.getMembers();
   if (auto hiv = mlir::dyn_cast<subop::HashIndexedViewType>(sourceStateType)) return hiv.getValueMembers();
   if (auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(sourceStateType)) return mixed.getValueMembers();
   return {};
}

static subop::Member sourceMemberForIdentifier(mlir::Type sourceStateType,
                                               subop::MemberManager& mm,
                                               llvm::StringRef identifier) {
   auto exactFilterPredMember = [&](subop::StateMembersAttr members) -> subop::Member {
      if (!members || !identifier.starts_with("filter_pred$")) return {};
      for (subop::Member member : members.getMembers()) {
         if (mm.getName(member) == identifier) return member;
      }
      return {};
   };

   if (auto tableTy = mlir::dyn_cast<subop::TableType>(sourceStateType))
      return tableMemberForIdentifier(tableTy, mm, identifier);
   if (auto tableTy = mlir::dyn_cast<subop::SharedTableType>(sourceStateType)) {
      if (subop::Member exactPred = exactFilterPredMember(tableTy.getPredicateMembers()))
         return exactPred;
      llvm::StringRef normalized = normalizeColumnIdentifier(identifier);
      for (subop::Member member : tableTy.getTableMembers().getMembers()) {
         llvm::StringRef name = mm.getName(member);
         if (name == identifier || normalizeColumnIdentifier(name) == normalized) return member;
      }
      return {};
   }
   subop::StateMembersAttr members = sourceValueMembersForResidualGather(sourceStateType);
   if (!members) return {};
   if (subop::Member exactPred = exactFilterPredMember(members)) return exactPred;
   llvm::StringRef normalized = normalizeColumnIdentifier(identifier);
   for (subop::Member member : members.getMembers()) {
      llvm::StringRef name = mm.getName(member);
      if (name == identifier || normalizeColumnIdentifier(name) == normalized) return member;
   }
   return {};
}

static std::optional<unsigned> findCompatibleMapInputIndex(subop::MapOp map,
                                                           llvm::StringRef semantic,
                                                           llvm::StringRef leaf,
                                                           mlir::Type type,
                                                           tuples::ColumnManager& cm) {
   std::optional<unsigned> byLeaf;
   for (unsigned i = 0; i < map.getInputCols().size(); ++i) {
      auto ref = mlir::cast<tuples::ColumnRefAttr>(map.getInputCols()[i]);
      auto [scope, existingLeaf] = cm.getName(&ref.getColumn());
      if (columnSemanticKey(scope, existingLeaf) == semantic && ref.getColumn().type == type) return i;
      if (existingLeaf == leaf && ref.getColumn().type == type) {
         assert(!byLeaf && "residual filter rewrite: ambiguous alias-compatible predicate input");
         byLeaf = i;
      }
   }
   return byLeaf;
}

static std::optional<unsigned> findMapInputSemanticIndex(subop::MapOp map,
                                                         llvm::StringRef semantic,
                                                         tuples::ColumnManager& cm) {
   for (unsigned i = 0; i < map.getInputCols().size(); ++i) {
      auto ref = mlir::cast<tuples::ColumnRefAttr>(map.getInputCols()[i]);
      auto [scope, leaf] = cm.getName(&ref.getColumn());
      if (columnSemanticKey(scope, leaf) == semantic) return i;
   }
   return std::nullopt;
}

static void deriveNonNullColumnsFromWidenedNullableGathers(subop::ExecutionStepOp buildStep);

static void collectBlockArgsUsedByValue(mlir::Value value,
                                        mlir::Block* block,
                                        llvm::DenseSet<unsigned>& out) {
   if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      if (arg.getOwner() == block) out.insert(arg.getArgNumber());
      return;
   }
   mlir::Operation* def = value.getDefiningOp();
   if (!def) return;
   for (mlir::Value operand : def->getOperands()) collectBlockArgsUsedByValue(operand, block, out);
}

static llvm::DenseSet<unsigned> residualPredicateInputIndices(subop::MapOp peerMap,
                                                              subop::FilterOp peerFilter) {
   llvm::DenseSet<unsigned> used;
   mlir::Block& peerBlock = peerMap.getFn().front();
   auto peerRet = mlir::cast<tuples::ReturnOp>(peerBlock.getTerminator());
   unsigned peerIdx = residualFilterConditionResultIndex(peerMap, peerFilter);
   collectBlockArgsUsedByValue(peerRet.getOperand(peerIdx), &peerBlock, used);
   return used;
}

static void ensureSyntheticMapHasPeerInputs(subop::ExecutionStepOp syntheticBuild,
                                            subop::MapOp syntheticMap,
                                            subop::MapOp peerMap,
                                            subop::FilterOp peerFilter) {
   auto* ctx = syntheticMap.getContext();
   auto& synthCm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& peerCm = peerMap.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto synthResidual = findResidualTableFilterInBuildStep(syntheticBuild);
   std::optional<ResidualTableFilter> source = synthResidual ? synthResidual : findScanLikeSourceInBuildStep(syntheticBuild);
   assert(source && "residual filter rewrite: synthetic build must have a scan-like source");
   assert(sourceValueMembersForResidualGather(source->sourceStateType) &&
          "residual filter rewrite: synthetic source must expose members");
   llvm::DenseSet<unsigned> usedInputs = residualPredicateInputIndices(peerMap, peerFilter);

   llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> gatherPairs;
   llvm::SmallVector<tuples::ColumnRefAttr, 4> newInputs;
   bool widenedSourceMemberType = false;
   for (unsigned inputIdx = 0; inputIdx < peerMap.getInputCols().size(); ++inputIdx) {
      if (!usedInputs.contains(inputIdx)) continue;
      auto attr = peerMap.getInputCols()[inputIdx];
      auto peerRef = mlir::cast<tuples::ColumnRefAttr>(attr);
      auto [scope, leaf] = peerCm.getName(&peerRef.getColumn());
      std::string semantic = columnSemanticKey(scope, leaf);
      mlir::Type inputType = cloneTypeToContext(peerRef.getColumn().type, ctx);
      subop::Member member = sourceMemberForIdentifier(source->sourceStateType, mm, leaf);
      assert(member && "residual filter rewrite: synthetic scan-like source must contain peer predicate input");
      if (auto existingIdx = findMapInputSemanticIndex(syntheticMap, semantic, synthCm)) {
         auto existingRef = mlir::cast<tuples::ColumnRefAttr>(syntheticMap.getInputCols()[*existingIdx]);
         if (existingRef.getColumn().type == inputType) continue;
         auto nullableInputTy = mlir::dyn_cast<db::NullableType>(inputType);
         if (!nullableInputTy || nullableInputTy.getType() != existingRef.getColumn().type)
            llvm_unreachable("residual filter rewrite: same semantic input must widen from non-null to nullable");
         member = mm.getOrCreateMemberDirect(mm.getName(member), inputType, /*allowTypeUpdate=*/true);
         existingRef.getColumn().type = inputType;
         syntheticMap.getFn().front().getArgument(*existingIdx).setType(inputType);
         widenedSourceMemberType = true;
         continue;
      }
      if (findCompatibleMapInputIndex(syntheticMap, semantic, leaf, inputType, synthCm)) continue;
      mlir::Type memberTy = cloneTypeToContext(mm.getType(member), ctx);
      if (memberTy != inputType) {
         auto nullableInputTy = mlir::dyn_cast<db::NullableType>(inputType);
         if (!nullableInputTy || nullableInputTy.getType() != memberTy)
            llvm_unreachable("residual filter rewrite: peer predicate input type must match source member type");
         member = mm.getOrCreateMemberDirect(mm.getName(member), inputType, /*allowTypeUpdate=*/true);
         widenedSourceMemberType = true;
      }
      tuples::ColumnDefAttr def = synthCm.createDef(scope, leaf);
      def.getColumn().type = cloneTypeToContext(mm.getType(member), ctx);
      gatherPairs.push_back({member, def});
      newInputs.push_back(synthCm.createRef(&def.getColumn()));
   }
   if (gatherPairs.empty()) return;

   mlir::OpBuilder b(syntheticMap);
   auto gather = b.create<subop::GatherOp>(
      syntheticMap.getLoc(), syntheticMap.getStream(), source->sourceRef,
      subop::ColumnDefMemberMappingAttr::get(ctx, gatherPairs));
   syntheticMap->setOperand(0, gather.getRes());
   for (tuples::ColumnRefAttr ref : newInputs)
      appendMapInputColumn(syntheticMap, ref);
   if (widenedSourceMemberType)
      deriveNonNullColumnsFromWidenedNullableGathers(syntheticBuild);
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
   } else if (auto isNull = mlir::dyn_cast<db::IsNullOp>(op)) {
      mlir::Value clonedVal = cloneResidualPredicateExprToSynthetic(isNull.getVal(), mapping, b, ctx);
      if (mlir::isa<db::NullableType>(clonedVal.getType())) {
         out = b.create<db::IsNullOp>(loc, clonedVal);
      } else {
         out = b.create<db::ConstantOp>(loc, mlir::IntegerType::get(ctx, 1),
                                        b.getI64IntegerAttr(0));
      }
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
   llvm::DenseSet<unsigned> mappedSyntheticArgs;
   llvm::DenseSet<unsigned> usedInputs = residualPredicateInputIndices(peerMap, peerFilter);
   for (unsigned i = 0; i < peerMap.getInputCols().size(); ++i) {
      if (!usedInputs.contains(i)) continue;
      auto peerInput = mlir::cast<tuples::ColumnRefAttr>(peerMap.getInputCols()[i]);
      auto [scope, leaf] = peerCm.getName(&peerInput.getColumn());
      std::string semantic = columnSemanticKey(scope, leaf);
      std::optional<unsigned> synthArgIdx = findCompatibleMapInputIndex(
         syntheticMap, semantic, leaf, cloneTypeToContext(peerInput.getColumn().type, ctx), synthCm);
      assert(synthArgIdx && "residual filter: peer predicate input must exist in synthetic predicate map");
      mlir::BlockArgument arg = syntheticMap.getFn().front().getArgument(*synthArgIdx);
      assert(mappedSyntheticArgs.insert(arg.getArgNumber()).second &&
             "residual filter rewrite: peer predicate is not local to one synthetic build-table row");
      mapping.map(peerMap.getFn().front().getArgument(i), arg);
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

static bool gatherComputesColumn(subop::GatherOp gather, tuples::ColumnRefAttr ref) {
   for (auto& [member, def] : gather.getMapping().getMapping()) {
      (void)member;
      if (&def.getColumn() == &ref.getColumn()) return true;
   }
   return false;
}

static mlir::Operation* findStreamOpComputingColumn(subop::ExecutionStepOp step,
                                                    tuples::ColumnRefAttr ref) {
   mlir::Operation* found = nullptr;
   step.walk([&](mlir::Operation* op) {
      if (found) return;
      if (auto map = mlir::dyn_cast<subop::MapOp>(op)) {
         if (mapComputesColumn(map, ref)) found = op;
         return;
      }
      if (auto gather = mlir::dyn_cast<subop::GatherOp>(op)) {
         if (gatherComputesColumn(gather, ref)) found = op;
         return;
      }
   });
   assert(found && "residual filter rewrite: simple predicate producer op must exist");
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

static bool streamIsAlreadyUnionFilteredBySharedTableScan(mlir::Value stream) {
   for (;;) {
      mlir::Operation* def = stream.getDefiningOp();
      if (!def) return false;
      if (auto scan = mlir::dyn_cast<subop::ScanRefsOp>(def)) {
         auto shared = mlir::dyn_cast<subop::SharedTableType>(scan.getState().getType());
         return shared && !shared.getPredicateMembers().getMembers().empty();
      }
      if (!mlir::isa<subop::GatherOp, subop::MapOp, subop::RenamingOp>(def)) return false;
      stream = streamInputOf(def);
   }
}

static bool moveSimplePredicateProducerBeforeResidualMap(subop::ExecutionStepOp step,
                                                         subop::MapOp residualMap,
                                                         tuples::ColumnRefAttr predRef) {
   mlir::Operation* predOp = findStreamOpComputingColumn(step, predRef);
   if (predOp == residualMap.getOperation()) return true;
   mlir::Value predStream = predOp->getResult(0);
   if (streamChainContains(residualMap.getStream(), predStream)) {
      return true;
   }
   if (predOp->getBlock() == residualMap->getBlock() &&
       predOp->isBeforeInBlock(residualMap.getOperation())) {
      if (residualMap.getStream() != predStream) residualMap->setOperand(0, predStream);
      return true;
   }
   if (!streamChainContains(streamInputOf(predOp), residualMap.getResult())) return false;

   llvm::SmallVector<mlir::Operation*, 4> chain;
   mlir::Value stream = streamInputOf(predOp);
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
   setStreamInputOf(predOp, newStream);
   predOp->moveBefore(residualMap.getOperation());
   residualMap->setOperand(0, predOp->getResult(0));
   return true;
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
                                                            llvm::ArrayRef<tuples::ColumnRefAttr> predRefs) {
   assert(!predRefs.empty() && "residual filter union requires predicate refs");
   auto* ctx = predRefs.front().getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   tuples::ColumnDefAttr unionPred = cm.createDef(cm.getUniqueScope("residual_filter_union"), "pred");
   unionPred.getColumn().type = mlir::IntegerType::get(ctx, 1);
   tuples::ColumnRefAttr unionRef = cm.createRef(&unionPred.getColumn());

   mlir::Operation* anchor = stream.getDefiningOp();
   assert(anchor && "residual filter union must be inserted after a stream producer");
   mlir::OpBuilder b(anchor);
   b.setInsertionPointAfter(anchor);
   llvm::SmallVector<mlir::Attribute, 8> predAttrs;
   for (tuples::ColumnRefAttr ref : predRefs) predAttrs.push_back(ref);
   auto map = b.create<subop::MapOp>(anchor->getLoc(), tuples::TupleStreamType::get(ctx), stream,
                                     b.getArrayAttr({unionPred}), b.getArrayAttr(predAttrs));
   mlir::Block* block = new mlir::Block();
   for (tuples::ColumnRefAttr ref : predRefs) block->addArgument(ref.getColumn().type, anchor->getLoc());
   map.getFn().push_back(block);
   mlir::OpBuilder rb(ctx);
   rb.setInsertionPointToStart(block);
   mlir::Value unionValue = block->getArgument(0);
   for (unsigned i = 1; i < block->getNumArguments(); ++i) {
      unionValue = rb.create<db::OrOp>(anchor->getLoc(), mlir::ValueRange{unionValue, block->getArgument(i)});
   }
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

static bool mapInputsAreOnlyGeneratedUpstreamPredicates(subop::MapOp map,
                                                        tuples::ColumnManager& cm) {
   if (!map || map.getInputCols().empty()) return false;
   for (mlir::Attribute attr : map.getInputCols()) {
      auto ref = mlir::dyn_cast<tuples::ColumnRefAttr>(attr);
      if (!isReuseGeneratedUpstreamPredicateColumn(ref, cm)) return false;
   }
   return true;
}

static void removeTrailingGeneratedUpstreamOnlyUnionFilter(subop::MaterializeOp mat) {
   auto& cm = mat.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   mlir::Value stream = mat.getStream();
   auto filter = mlir::dyn_cast_or_null<subop::FilterOp>(stream.getDefiningOp());
   if (!filter || filter.getConditions().empty()) return;
   auto map = mlir::dyn_cast_or_null<subop::MapOp>(filter.getStream().getDefiningOp());
   if (!map || !mapInputsAreOnlyGeneratedUpstreamPredicates(map, cm)) return;

   llvm::DenseSet<const void*> mapComputed;
   for (mlir::Attribute attr : map.getComputedCols()) {
      auto def = mlir::dyn_cast<tuples::ColumnDefAttr>(attr);
      if (def) mapComputed.insert(&def.getColumn());
   }
   if (mapComputed.empty()) return;
   for (mlir::Attribute attr : filter.getConditions()) {
      auto ref = mlir::dyn_cast<tuples::ColumnRefAttr>(attr);
      if (!ref || !mapComputed.contains(&ref.getColumn())) return;
   }
   mat->setOperand(0, map.getStream());
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

static std::optional<ResidualTableFilter> findScanLikeSourceInBuildStep(subop::ExecutionStepOp step) {
   mlir::Block& body = step.getSubOps().front();
   for (mlir::Operation& op : body.without_terminator()) {
      if (auto scan = mlir::dyn_cast<subop::ScanRefsOp>(&op)) {
         if (!sourceValueMembersForResidualGather(scan.getState().getType())) continue;
         auto& cm = scan.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
         return ResidualTableFilter{scan.getOperation(), cm.createRef(&scan.getRef().getColumn()),
                                    scan.getState().getType(), {}, {}};
      }
      if (auto scan = mlir::dyn_cast<subop::ScanListOp>(&op)) {
         auto listTy = mlir::dyn_cast<subop::ListType>(scan.getList().getType());
         if (!listTy) continue;
         auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(listTy.getT());
         if (!ler) continue;
         auto& cm = scan.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
         return ResidualTableFilter{scan.getOperation(), cm.createRef(&scan.getElem().getColumn()),
                                    ler.getState(), {}, {}};
      }
   }
   return std::nullopt;
}

static subop::MapOp createResidualPredicateMapAfterTableScan(subop::ExecutionStepOp syntheticBuild,
                                                             subop::MapOp peerMap,
                                                             subop::FilterOp peerFilter) {
   auto* ctx = syntheticBuild.getContext();
   auto& synthCm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& peerCm = peerMap.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto source = findScanLikeSourceInBuildStep(syntheticBuild);
   assert(source && "residual filter rewrite: synthetic build must have a scan-like source");
   assert(sourceValueMembersForResidualGather(source->sourceStateType) &&
          "residual filter rewrite: synthetic source must expose members");

   llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> gatherPairs;
   llvm::SmallVector<mlir::Attribute> mapInputs;
   llvm::DenseSet<unsigned> usedInputs = residualPredicateInputIndices(peerMap, peerFilter);
   for (unsigned inputIdx = 0; inputIdx < peerMap.getInputCols().size(); ++inputIdx) {
      if (!usedInputs.contains(inputIdx)) continue;
      auto attr = peerMap.getInputCols()[inputIdx];
      auto peerRef = mlir::cast<tuples::ColumnRefAttr>(attr);
      auto [scope, leaf] = peerCm.getName(&peerRef.getColumn());
      subop::Member member = sourceMemberForIdentifier(source->sourceStateType, mm, leaf);
      assert(member && "residual filter rewrite: synthetic scan-like source must contain peer predicate input");
      tuples::ColumnDefAttr def = synthCm.createDef(scope, leaf);
      def.getColumn().type = cloneTypeToContext(mm.getType(member), ctx);
      gatherPairs.push_back({member, def});
      mapInputs.push_back(synthCm.createRef(&def.getColumn()));
   }

   mlir::OpBuilder b(source->scanLike);
   b.setInsertionPointAfter(source->scanLike);
   mlir::Value stream = source->scanLike->getResult(0);
   llvm::SmallVector<mlir::Operation*> excludeOps{source->scanLike};
   if (!gatherPairs.empty()) {
      auto gather = b.create<subop::GatherOp>(source->scanLike->getLoc(), stream, source->sourceRef,
                                              subop::ColumnDefMemberMappingAttr::get(ctx, gatherPairs));
      stream = gather.getRes();
      b.setInsertionPointAfter(gather);
      excludeOps.push_back(gather.getOperation());
   }

   auto map = b.create<subop::MapOp>(source->scanLike->getLoc(), tuples::TupleStreamType::get(ctx), stream,
                                     b.getArrayAttr({}), b.getArrayAttr(mapInputs));
   mlir::Block* block = new mlir::Block();
   for (auto input : mapInputs) {
      auto ref = mlir::cast<tuples::ColumnRefAttr>(input);
      block->addArgument(ref.getColumn().type, source->scanLike->getLoc());
   }
   map.getFn().push_back(block);
   mlir::OpBuilder rb(ctx);
   rb.setInsertionPointToStart(block);
   rb.create<tuples::ReturnOp>(source->scanLike->getLoc(), mlir::ValueRange{});
   excludeOps.push_back(map.getOperation());
   rewireStreamUsesAfterAnchorInStep(source->scanLike->getResult(0), map.getResult(), source->scanLike, excludeOps);
   return map;
}

static tuples::ColumnRefAttr threadColumnRefFromSourceToMaterializeStream(subop::MaterializeOp matOp,
                                                                          mlir::Operation* sourceOp,
                                                                          tuples::ColumnDefAttr def,
                                                                          tuples::ColumnManager& cm);

static tuples::ColumnRefAttr insertAndPredsBeforeMaterialize(subop::MaterializeOp matOp,
                                                             llvm::ArrayRef<tuples::ColumnRefAttr> predRefs,
                                                             llvm::StringRef scopeSeed) {
   auto* ctx = matOp.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   llvm::SmallVector<tuples::ColumnRefAttr, 4> threadedPredRefs;
   threadedPredRefs.reserve(predRefs.size());
   for (tuples::ColumnRefAttr ref : predRefs) {
      tuples::ColumnDefAttr def = cm.createDef(&ref.getColumn());
      threadedPredRefs.push_back(threadColumnRefFromSourceToMaterializeStream(matOp, /*sourceOp=*/nullptr, def, cm));
   }
   predRefs = threadedPredRefs;
   if (predRefs.size() == 1) return predRefs.front();

   tuples::ColumnDefAttr outDef = cm.createDef(cm.getUniqueScope(scopeSeed), "filter_pred");
   outDef.getColumn().type = mlir::IntegerType::get(ctx, 1);
   tuples::ColumnRefAttr outRef = cm.createRef(&outDef.getColumn());

   mlir::OpBuilder b(matOp);
   llvm::SmallVector<mlir::Attribute, 4> inputs;
   for (tuples::ColumnRefAttr ref : predRefs) inputs.push_back(ref);
   auto map = b.create<subop::MapOp>(matOp.getLoc(), tuples::TupleStreamType::get(ctx),
                                     matOp.getStream(), b.getArrayAttr({outDef}),
                                     b.getArrayAttr(inputs));
   mlir::Block* block = new mlir::Block();
   for (tuples::ColumnRefAttr ref : predRefs)
      block->addArgument(ref.getColumn().type, matOp.getLoc());
   map.getFn().push_back(block);

   mlir::OpBuilder rb(ctx);
   rb.setInsertionPointToStart(block);
   mlir::Value combined;
   if (predRefs.empty()) {
      combined = rb.create<mlir::arith::ConstantIntOp>(matOp.getLoc(), 1, 1);
   } else {
      combined = block->getArgument(0);
      for (unsigned i = 1; i < block->getNumArguments(); ++i)
         combined = rb.create<db::AndOp>(matOp.getLoc(), mlir::ValueRange{combined, block->getArgument(i)});
   }
   rb.create<tuples::ReturnOp>(matOp.getLoc(), mlir::ValueRange{combined});

   matOp->setOperand(0, map.getResult());
   return outRef;
}

static void enqueueTupleStreamSourceForBlockArgument(mlir::BlockArgument arg,
                                                     llvm::SmallVectorImpl<mlir::Value>& worklist) {
   mlir::Operation* parent = arg.getOwner()->getParentOp();
   if (!parent) return;
   if (arg.getArgNumber() < parent->getNumOperands()) worklist.push_back(parent->getOperand(arg.getArgNumber()));
}

static bool lookupProducesColumn(subop::LookupOp lookup, tuples::ColumnRefAttr column) {
   return column && &lookup.getRef().getColumn() == &column.getColumn();
}

static void recordUniqueLookup(subop::LookupOp lookup, std::optional<subop::LookupOp>& found,
                               bool& multiple) {
   if (found && found->getOperation() != lookup.getOperation()) {
      multiple = true;
      found = std::nullopt;
      return;
   }
   found = lookup;
}

static std::optional<subop::LookupOp>
findUniqueLookupProducingColumnOnStream(mlir::Value rootStream, tuples::ColumnRefAttr column,
                                        bool& multiple) {
   multiple = false;
   std::optional<subop::LookupOp> found;
   llvm::DenseSet<void*> seen;
   llvm::SmallVector<mlir::Value, 8> worklist{rootStream};
   while (!worklist.empty()) {
      mlir::Value cur = worklist.pop_back_val();
      if (!cur || !seen.insert(cur.getAsOpaquePointer()).second) continue;
      if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(cur)) {
         enqueueTupleStreamSourceForBlockArgument(arg, worklist);
         continue;
      }
      mlir::Operation* def = cur.getDefiningOp();
      if (!def) continue;
      if (auto lookup = mlir::dyn_cast<subop::LookupOp>(def)) {
         if (lookupProducesColumn(lookup, column)) {
            recordUniqueLookup(lookup, found, multiple);
            if (multiple) return std::nullopt;
         }
         worklist.push_back(lookup.getStream());
         continue;
      }
      for (mlir::Value operand : def->getOperands()) {
         if (mlir::isa<tuples::TupleStreamType>(operand.getType())) worklist.push_back(operand);
      }
   }
   return found;
}

static std::optional<subop::LookupOp> findUniqueUpstreamLookupOpInReuse(mlir::Value value,
                                                                        bool& multiple) {
   multiple = false;
   std::optional<subop::LookupOp> found;
   llvm::DenseSet<void*> seen;
   llvm::SmallVector<mlir::Value, 8> worklist{value};
   while (!worklist.empty()) {
      mlir::Value cur = worklist.pop_back_val();
      if (!cur || !seen.insert(cur.getAsOpaquePointer()).second) continue;
      if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(cur)) {
         if (auto nested = mlir::dyn_cast<subop::NestedMapOp>(arg.getOwner()->getParentOp())) {
            if (arg.getArgNumber() == 0) {
               worklist.push_back(nested.getStream());
               continue;
            }
            unsigned paramIdx = arg.getArgNumber() - 1;
            if (paramIdx < nested.getParameters().size()) {
               auto param = mlir::dyn_cast<tuples::ColumnRefAttr>(nested.getParameters()[paramIdx]);
               bool nestedMultiple = false;
               std::optional<subop::LookupOp> lookup =
                  findUniqueLookupProducingColumnOnStream(nested.getStream(), param, nestedMultiple);
               if (nestedMultiple) {
                  multiple = true;
                  return std::nullopt;
               }
               if (lookup) {
                  recordUniqueLookup(*lookup, found, multiple);
                  if (multiple) return std::nullopt;
               }
               continue;
            }
         }
         enqueueTupleStreamSourceForBlockArgument(arg, worklist);
         continue;
      }
      mlir::Operation* def = cur.getDefiningOp();
      if (!def) continue;
      if (auto lookup = mlir::dyn_cast<subop::LookupOp>(def)) {
         recordUniqueLookup(lookup, found, multiple);
         if (multiple) return std::nullopt;
         continue;
      }
      if (auto scanList = mlir::dyn_cast<subop::ScanListOp>(def)) {
         worklist.push_back(scanList.getList());
         continue;
      }
      if (auto nested = mlir::dyn_cast<subop::NestedMapOp>(def)) {
         worklist.push_back(nested.getStream());
         continue;
      }
      for (mlir::Value operand : def->getOperands()) worklist.push_back(operand);
   }
   return found;
}

static void collectScanListsOnStreamChain(mlir::Value rootStream,
                                          llvm::SmallVectorImpl<subop::ScanListOp>& out,
                                          llvm::DenseSet<mlir::Operation*>& seenScanLists) {
   if (!rootStream) return;
   llvm::DenseSet<void*> seenStreams;
   llvm::SmallVector<mlir::Value, 4> worklist{rootStream};
   while (!worklist.empty()) {
      mlir::Value stream = worklist.pop_back_val();
      if (!stream || !seenStreams.insert(stream.getAsOpaquePointer()).second) continue;
      mlir::Operation* def = stream.getDefiningOp();
      if (!def) continue;
      if (auto scanList = mlir::dyn_cast<subop::ScanListOp>(def)) {
         if (seenScanLists.insert(scanList.getOperation()).second) out.push_back(scanList);
         continue;
      }
      if (auto gather = mlir::dyn_cast<subop::GatherOp>(def)) {
         worklist.push_back(gather.getStream());
         continue;
      }
      if (auto map = mlir::dyn_cast<subop::MapOp>(def)) {
         worklist.push_back(map.getStream());
         continue;
      }
      if (auto filter = mlir::dyn_cast<subop::FilterOp>(def)) {
         worklist.push_back(filter.getStream());
         continue;
      }
      if (auto rename = mlir::dyn_cast<subop::RenamingOp>(def)) {
         worklist.push_back(rename.getStream());
         continue;
      }
   }
}

static bool filterClauseEquals(llvm::ArrayRef<lingodb::runtime::FilterDescription> a,
                               llvm::ArrayRef<lingodb::runtime::FilterDescription> b);

static void collectScanRefsOnStreamChain(mlir::Value rootStream,
                                         llvm::SmallVectorImpl<subop::ScanRefsOp>& out,
                                         llvm::DenseSet<mlir::Operation*>& seenScanRefs) {
   if (!rootStream) return;
   llvm::DenseSet<void*> seenStreams;
   llvm::SmallVector<mlir::Value, 4> worklist{rootStream};
   while (!worklist.empty()) {
      mlir::Value stream = worklist.pop_back_val();
      if (!stream || !seenStreams.insert(stream.getAsOpaquePointer()).second) continue;
      mlir::Operation* def = stream.getDefiningOp();
      if (!def) continue;
      if (auto scanRefs = mlir::dyn_cast<subop::ScanRefsOp>(def)) {
         if (seenScanRefs.insert(scanRefs.getOperation()).second) out.push_back(scanRefs);
         continue;
      }
      if (auto gather = mlir::dyn_cast<subop::GatherOp>(def)) {
         worklist.push_back(gather.getStream());
         continue;
      }
      if (auto map = mlir::dyn_cast<subop::MapOp>(def)) {
         worklist.push_back(map.getStream());
         continue;
      }
      if (auto filter = mlir::dyn_cast<subop::FilterOp>(def)) {
         worklist.push_back(filter.getStream());
         continue;
      }
      if (auto rename = mlir::dyn_cast<subop::RenamingOp>(def)) {
         worklist.push_back(rename.getStream());
         continue;
      }
   }
}

static void collectScanListsBeforeOpInStep(subop::ExecutionStepOp step,
                                           mlir::Operation* beforeOp,
                                           llvm::SmallVectorImpl<subop::ScanListOp>& out,
                                           llvm::DenseSet<mlir::Operation*>& seenScanLists) {
   if (!step || !beforeOp) return;
   mlir::Block& body = step.getSubOps().front();
   if (beforeOp->getBlock() != &body) return;
   for (mlir::Operation& op : body.without_terminator()) {
      if (&op == beforeOp) break;
      auto scanList = mlir::dyn_cast<subop::ScanListOp>(&op);
      if (!scanList) continue;
      if (seenScanLists.insert(scanList.getOperation()).second) out.push_back(scanList);
   }
}

static void collectScanRefsBeforeOpInStep(subop::ExecutionStepOp step,
                                          mlir::Operation* beforeOp,
                                          llvm::SmallVectorImpl<subop::ScanRefsOp>& out,
                                          llvm::DenseSet<mlir::Operation*>& seenScanRefs) {
   if (!step || !beforeOp) return;
   mlir::Block& body = step.getSubOps().front();
   if (beforeOp->getBlock() != &body) return;
   for (mlir::Operation& op : body.without_terminator()) {
      if (&op == beforeOp) break;
      auto scanRefs = mlir::dyn_cast<subop::ScanRefsOp>(&op);
      if (!scanRefs) continue;
      if (seenScanRefs.insert(scanRefs.getOperation()).second) out.push_back(scanRefs);
   }
}

static llvm::SmallVector<subop::ScanListOp, 4>
scanListsOnMaterializeContextChains(subop::MaterializeOp matOp) {
   llvm::SmallVector<subop::ScanListOp, 4> out;
   llvm::DenseSet<mlir::Operation*> seenScanLists;

   collectScanListsOnStreamChain(matOp.getStream(), out, seenScanLists);
   mlir::Operation* child = matOp.getOperation();
   for (mlir::Operation* parent = matOp->getParentOp(); parent; child = parent, parent = parent->getParentOp()) {
      if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(parent))
         collectScanListsBeforeOpInStep(step, child, out, seenScanLists);
      auto nested = mlir::dyn_cast<subop::NestedMapOp>(parent);
      if (!nested) continue;
      collectScanListsOnStreamChain(nested.getStream(), out, seenScanLists);
   }
   return out;
}

static llvm::SmallVector<subop::ScanRefsOp, 4>
scanRefsOnMaterializeContextChains(subop::MaterializeOp matOp) {
   llvm::SmallVector<subop::ScanRefsOp, 4> out;
   llvm::DenseSet<mlir::Operation*> seenScanRefs;

   collectScanRefsOnStreamChain(matOp.getStream(), out, seenScanRefs);
   mlir::Operation* child = matOp.getOperation();
   for (mlir::Operation* parent = matOp->getParentOp(); parent; child = parent, parent = parent->getParentOp()) {
      if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(parent))
         collectScanRefsBeforeOpInStep(step, child, out, seenScanRefs);
      auto nested = mlir::dyn_cast<subop::NestedMapOp>(parent);
      if (!nested) continue;
      collectScanRefsOnStreamChain(nested.getStream(), out, seenScanRefs);
   }
   return out;
}

static subop::HashIndexedViewType asHashIndexedViewLayoutType(mlir::Type type);
static bool localHivLayoutHasFilterPredMember(mlir::MLIRContext* ctx,
                                              subop::HashIndexedViewType layout);

static llvm::SmallVector<tuples::ColumnRefAttr, 4>
threadUpstreamMixedPredColumnsToMaterializeStream(subop::ExecutionStepOp buildStep,
                                                  llvm::StringRef predMemberName) {
   subop::MaterializeOp matOp = findJoinBufferMaterializeInStep(buildStep);
   if (!matOp) return {};
   auto* ctx = buildStep.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

   llvm::SmallVector<subop::ScanListOp, 4> inputScanLists = scanListsOnMaterializeContextChains(matOp);
   llvm::SmallVector<std::pair<subop::ScanListOp, subop::Member>, 4> predSources;
   for (subop::ScanListOp scanList : inputScanLists) {
      auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(scanList.getElem().getColumn().type);
      if (!ler) continue;
      auto layout = asHashIndexedViewLayoutType(ler.getState());
      if (!layout) continue;
      for (subop::Member m : layout.getValueMembers().getMembers()) {
         if (mm.getName(m) != predMemberName) continue;
         predSources.push_back({scanList, m});
         break;
      }
   }

   llvm::SmallVector<tuples::ColumnRefAttr, 4> refs;
   unsigned sourceIdx = 0;
   for (auto [predScanList, upstreamPredMember] : predSources) {
      std::string scopeSeed = "reuse_upstream_pred";
      if (auto slot = parseFilterPredMemberSlot(predMemberName))
         scopeSeed = ("reuse_upstream_pred$" + llvm::Twine(*slot) + "_" +
                      llvm::Twine(sourceIdx)).str();
      tuples::ColumnDefAttr predDef = cm.createDef(cm.getUniqueScope(scopeSeed), "filter_pred");
      predDef.getColumn().type = mlir::IntegerType::get(ctx, 1);

      mlir::OpBuilder b(predScanList);
      b.setInsertionPointAfter(predScanList);
      auto mapping = subop::ColumnDefMemberMappingAttr::get(
         ctx, llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>>{{upstreamPredMember, predDef}});
      auto gather = b.create<subop::GatherOp>(predScanList.getLoc(), predScanList.getRes().getType(),
                                              predScanList.getRes(),
                                              cm.createRef(&predScanList.getElem().getColumn()), mapping);
      rewireStreamUsesAfterAnchorInStep(predScanList.getRes(), gather.getRes(), predScanList.getOperation(),
                                        {predScanList.getOperation(), gather.getOperation()});
      gather->setOperand(0, predScanList.getRes());

      refs.push_back(threadColumnRefFromSourceToMaterializeStream(matOp, predScanList.getOperation(),
                                                                  predDef, cm));
      ++sourceIdx;
   }
   for (subop::ScanListOp scanList : inputScanLists) {
      auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(scanList.getElem().getColumn().type);
      if (!ler) continue;
      auto layout = asHashIndexedViewLayoutType(ler.getState());
      if (!localHivLayoutHasFilterPredMember(ctx, layout)) continue;
      bool multipleLookups = false;
      std::optional<subop::LookupOp> lookup = findUniqueUpstreamLookupOpInReuse(scanList.getList(), multipleLookups);
      assert(!multipleLookups && "mixed HIV filter relax requires a unique lookup");
      if (!lookup || lookup->getKeys().size() <= 1) continue;
      for (mlir::Attribute keyAttr : llvm::drop_begin(lookup->getKeys(), 1)) {
         auto keyRef = mlir::dyn_cast<tuples::ColumnRefAttr>(keyAttr);
         assert(keyRef && keyRef.getColumn().type.isInteger(1) &&
                "mixed HIV lookup predicate key must be an i1 column");
         tuples::ColumnDefAttr predDef =
            cm.createDef(&keyRef.getColumn());
         refs.push_back(threadColumnRefFromSourceToMaterializeStream(matOp, lookup->getOperation(),
                                                                     predDef, cm));
      }
   }
   return refs;
}

static llvm::SmallVector<tuples::ColumnRefAttr, 4>
threadUpstreamBufferPredColumnsToMaterializeStream(subop::ExecutionStepOp buildStep,
                                                   llvm::StringRef predMemberName) {
   subop::MaterializeOp matOp = findJoinBufferMaterializeInStep(buildStep);
   if (!matOp) return {};
   auto* ctx = buildStep.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

   llvm::SmallVector<tuples::ColumnRefAttr, 4> refs;
   unsigned sourceIdx = 0;
   for (subop::ScanRefsOp scanRefs : scanRefsOnMaterializeContextChains(matOp)) {
      mlir::Value state = peelBlockArgsToEnclosingOperands(scanRefs.getState());
      if (!state) state = scanRefs.getState();
      subop::BufferType buffer = getInnerBufferTypeForMaterializeState(state.getType());
      if (!buffer) continue;
      subop::Member predMember;
      for (subop::Member m : buffer.getMembers().getMembers()) {
         if (mm.getName(m) == predMemberName) {
            predMember = m;
            break;
         }
      }
      if (!predMember) continue;

      std::string scopeSeed = "reuse_upstream_buffer_pred";
      if (auto slot = parseFilterPredMemberSlot(predMemberName))
         scopeSeed = ("reuse_upstream_buffer_pred$" + llvm::Twine(*slot) + "_" +
                      llvm::Twine(sourceIdx)).str();
      tuples::ColumnDefAttr predDef = cm.createDef(cm.getUniqueScope(scopeSeed), "filter_pred");
      predDef.getColumn().type = mlir::IntegerType::get(ctx, 1);

      mlir::OpBuilder b(scanRefs);
      b.setInsertionPointAfter(scanRefs);
      auto mapping = subop::ColumnDefMemberMappingAttr::get(
         ctx, llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>>{{predMember, predDef}});
      auto gather = b.create<subop::GatherOp>(scanRefs.getLoc(), scanRefs.getRes().getType(),
                                              scanRefs.getRes(),
                                              cm.createRef(&scanRefs.getRef().getColumn()), mapping);
      rewireStreamUsesAfterAnchorInStep(scanRefs.getRes(), gather.getRes(), scanRefs.getOperation(),
                                        {scanRefs.getOperation(), gather.getOperation()});
      gather->setOperand(0, scanRefs.getRes());
      refs.push_back(threadColumnRefFromSourceToMaterializeStream(matOp, scanRefs.getOperation(),
                                                                  predDef, cm));
      ++sourceIdx;
   }
   return refs;
}

static llvm::SmallVector<tuples::ColumnRefAttr, 4>
threadUpstreamSharedTablePredColumnsToMaterializeStream(subop::ExecutionStepOp buildStep,
                                                        llvm::StringRef predMemberName) {
   subop::MaterializeOp matOp = findJoinBufferMaterializeInStep(buildStep);
   if (!matOp) return {};
   auto* ctx = buildStep.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

   llvm::SmallVector<tuples::ColumnRefAttr, 4> refs;
   unsigned sourceIdx = 0;
   for (subop::ScanRefsOp scanRefs : scanRefsOnMaterializeContextChains(matOp)) {
      auto shared = mlir::dyn_cast<subop::SharedTableType>(scanRefs.getState().getType());
      if (!shared) continue;
      subop::Member predMember;
      for (subop::Member m : shared.getPredicateMembers().getMembers()) {
         if (mm.getName(m) == predMemberName) {
            predMember = m;
            break;
         }
      }
      if (!predMember) continue;

      std::string scopeSeed = "reuse_upstream_shared_pred";
      if (auto slot = parseFilterPredMemberSlot(predMemberName))
         scopeSeed = ("reuse_upstream_shared_pred$" + llvm::Twine(*slot) + "_" +
                      llvm::Twine(sourceIdx)).str();
      tuples::ColumnDefAttr predDef = cm.createDef(cm.getUniqueScope(scopeSeed), "filter_pred");
      predDef.getColumn().type = mlir::IntegerType::get(ctx, 1);

      mlir::OpBuilder b(scanRefs);
      b.setInsertionPointAfter(scanRefs);
      auto mapping = subop::ColumnDefMemberMappingAttr::get(
         ctx, llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>>{{predMember, predDef}});
      auto gather = b.create<subop::GatherOp>(scanRefs.getLoc(), scanRefs.getRes().getType(),
                                              scanRefs.getRes(),
                                              cm.createRef(&scanRefs.getRef().getColumn()), mapping);
      rewireStreamUsesAfterAnchorInStep(scanRefs.getRes(), gather.getRes(), scanRefs.getOperation(),
                                        {scanRefs.getOperation(), gather.getOperation()});
      gather->setOperand(0, scanRefs.getRes());
      refs.push_back(threadColumnRefFromSourceToMaterializeStream(matOp, scanRefs.getOperation(),
                                                                  predDef, cm));
      ++sourceIdx;
   }
   return refs;
}

static llvm::SmallVector<tuples::ColumnRefAttr, 4>
threadUpstreamPredColumnsToMaterializeStream(subop::ExecutionStepOp buildStep,
                                             llvm::StringRef predMemberName,
                                             bool includeBufferScanRefs) {
   llvm::SmallVector<tuples::ColumnRefAttr, 4> refs =
      threadUpstreamMixedPredColumnsToMaterializeStream(buildStep, predMemberName);
   llvm::SmallVector<tuples::ColumnRefAttr, 4> sharedRefs =
      threadUpstreamSharedTablePredColumnsToMaterializeStream(buildStep, predMemberName);
   refs.append(sharedRefs.begin(), sharedRefs.end());
   if (includeBufferScanRefs) {
      llvm::SmallVector<tuples::ColumnRefAttr, 4> bufferRefs =
         threadUpstreamBufferPredColumnsToMaterializeStream(buildStep, predMemberName);
      refs.append(bufferRefs.begin(), bufferRefs.end());
   }
   return refs;
}

static std::optional<uint64_t> cacheGetKeyForStepResult(mlir::Value value) {
   auto step = mlir::dyn_cast_or_null<subop::ExecutionStepOp>(value.getDefiningOp());
   if (!step) return std::nullopt;
   unsigned resultIdx = mlir::cast<mlir::OpResult>(value).getResultNumber();
   auto ret = mlir::dyn_cast<subop::ExecutionStepReturnOp>(step.getSubOps().front().getTerminator());
   if (!ret || resultIdx >= ret.getNumOperands()) return std::nullopt;
   mlir::Value returned = ret.getOperand(resultIdx);
   auto get = returned.getDefiningOp<subop::CacheGetOp>();
   if (!get) return std::nullopt;
   return static_cast<uint64_t>(get.getKey());
}

static subop::HashIndexedViewType asLocalHivLayoutType(mlir::Type type) {
      if (auto hiv = mlir::dyn_cast<subop::HashIndexedViewType>(type)) return hiv;
      if (auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(type)) {
         return subop::HashIndexedViewType::get(mixed.getContext(), mixed.getKeyMembers(),
                                                mixed.getValueMembers(), mixed.getCompareHashForLookup());
      }
      return nullptr;
}

static bool localHivLayoutHasFilterPredMember(mlir::MLIRContext* ctx,
                                              subop::HashIndexedViewType layout) {
   if (!layout) return false;
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   for (subop::Member member : layout.getValueMembers().getMembers()) {
      if (parseFilterPredMemberSlot(mm.getName(member))) return true;
   }
   return false;
}

struct DirectMixedDep {
   uint64_t cacheKey;
   mlir::Value state;
   subop::HashIndexedViewType layout;
};

static llvm::SmallVector<DirectMixedDep, 4> directMixedDepsForBuildStep(subop::ExecutionStepOp buildStep) {
   llvm::SmallVector<DirectMixedDep, 4> deps;
   llvm::DenseSet<uint64_t> seenKeys;
   for (mlir::Value input : buildStep.getInputs()) {
      auto key = cacheGetKeyForStepResult(input);
      if (!key || !seenKeys.insert(*key).second) continue;
      auto layout = asLocalHivLayoutType(input.getType());
      if (!localHivLayoutHasFilterPredMember(buildStep.getContext(), layout)) continue;
      deps.push_back(DirectMixedDep{*key, canonicalizeStateValueForReuse(input), layout});
   }
   return deps;
}

static std::optional<uint64_t> directDepKeyForScanList(subop::ScanListOp scanList,
                                                       llvm::ArrayRef<DirectMixedDep> directDeps) {
   bool multipleLookups = false;
   std::optional<subop::LookupOp> lookup = findUniqueUpstreamLookupOpInReuse(scanList.getList(), multipleLookups);
   assert(!multipleLookups && "mixed inherited pred scan_list must have one upstream lookup");
   if (!lookup) return std::nullopt;
   mlir::Value scanState = canonicalizeStateValueForReuse(
      peelBlockArgsToEnclosingOperands(lookup->getState()));
   auto scanKey = cacheGetKeyForStepResult(scanState);
   if (!scanKey) return std::nullopt;
   std::optional<uint64_t> found;
   for (const DirectMixedDep& dep : directDeps) {
      if (dep.cacheKey != *scanKey) continue;
      assert(canonicalizeStateValueForReuse(dep.state) == scanState &&
             "mixed inherited pred scan_list cache dep must map to the same state value");
      assert(!found && "mixed inherited pred scan_list must match one direct cache dep");
      found = dep.cacheKey;
   }
   return found;
}

static llvm::SmallVector<tuples::ColumnRefAttr, 4>
threadMappedUpstreamMixedPredColumnsToMaterializeStream(
   subop::ExecutionStepOp buildStep, unsigned targetSlot,
   uint64_t targetCacheKey,
   const llvm::DenseMap<uint64_t, llvm::DenseMap<unsigned, unsigned>>& consumerSlotByCacheKeyAndQuery) {
   auto itTargetSlots = consumerSlotByCacheKeyAndQuery.find(targetCacheKey);
   if (itTargetSlots == consumerSlotByCacheKeyAndQuery.end()) return {};
   llvm::SmallVector<DirectMixedDep, 4> directDeps = directMixedDepsForBuildStep(buildStep);
   if (directDeps.empty()) return {};
   llvm::DenseSet<uint64_t> directDepKeys;
   for (const DirectMixedDep& dep : directDeps) directDepKeys.insert(dep.cacheKey);

   llvm::DenseMap<uint64_t, llvm::SmallVector<unsigned, 4>> wantedSlotsByDep;
   for (const auto& [queryIdx, slot] : itTargetSlots->second) {
      if (slot != targetSlot) continue;
      for (uint64_t depKey : directDepKeys) {
         auto itSlotsByQuery = consumerSlotByCacheKeyAndQuery.find(depKey);
         if (itSlotsByQuery == consumerSlotByCacheKeyAndQuery.end()) continue;
         auto itDepSlot = itSlotsByQuery->second.find(queryIdx);
         if (itDepSlot == itSlotsByQuery->second.end()) continue;
         wantedSlotsByDep[depKey].push_back(itDepSlot->second);
      }
   }
   if (wantedSlotsByDep.empty()) return {};
   for (auto& [depKey, slots] : wantedSlotsByDep) {
      llvm::sort(slots);
      slots.erase(std::unique(slots.begin(), slots.end()), slots.end());
   }

   subop::MaterializeOp matOp = findJoinBufferMaterializeInStep(buildStep);
   if (!matOp) return {};
   auto* ctx = buildStep.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   llvm::SmallVector<subop::ScanListOp, 4> inputScanLists = scanListsOnMaterializeContextChains(matOp);

   llvm::SmallVector<tuples::ColumnRefAttr, 4> refs;
   llvm::StringSet<> emitted;
   unsigned sourceIdx = 0;
   for (subop::ScanListOp scanList : inputScanLists) {
      auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(scanList.getElem().getColumn().type);
      if (!ler) continue;
      auto scannedLayout = asLocalHivLayoutType(ler.getState());
      if (!localHivLayoutHasFilterPredMember(ctx, scannedLayout)) continue;
      std::optional<uint64_t> depKey = directDepKeyForScanList(scanList, directDeps);
      if (!depKey) continue;
      auto itWantedSlots = wantedSlotsByDep.find(*depKey);
      if (itWantedSlots == wantedSlotsByDep.end()) continue;
      for (unsigned depSlot : itWantedSlots->second) {
         std::string key = (llvm::Twine(*depKey) + ":" + llvm::Twine(depSlot)).str();
         if (!emitted.insert(key).second) continue;
         std::string predName = ("filter_pred$" + llvm::Twine(depSlot)).str();
         subop::Member predMember;
         for (subop::Member m : scannedLayout.getValueMembers().getMembers()) {
            if (mm.getName(m) == predName) {
               predMember = m;
               break;
            }
         }
         if (!predMember) continue;
         std::string scopeSeed = ("reuse_upstream_pred$" + llvm::Twine(targetSlot) + "_" +
                                  llvm::Twine(sourceIdx)).str();
         tuples::ColumnDefAttr predDef = cm.createDef(cm.getUniqueScope(scopeSeed), "filter_pred");
         predDef.getColumn().type = mlir::IntegerType::get(ctx, 1);

         mlir::OpBuilder b(scanList);
         b.setInsertionPointAfter(scanList);
         auto mapping = subop::ColumnDefMemberMappingAttr::get(
            ctx, llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>>{{predMember, predDef}});
         auto gather = b.create<subop::GatherOp>(scanList.getLoc(), scanList.getRes().getType(),
                                                 scanList.getRes(),
                                                 cm.createRef(&scanList.getElem().getColumn()), mapping);
         rewireStreamUsesAfterAnchorInStep(scanList.getRes(), gather.getRes(), scanList.getOperation(),
                                           {scanList.getOperation(), gather.getOperation()});
         gather->setOperand(0, scanList.getRes());
         refs.push_back(threadColumnRefFromSourceToMaterializeStream(matOp, scanList.getOperation(),
                                                                     predDef, cm));
         ++sourceIdx;
      }
   }
   return refs;
}

static llvm::SmallVector<unsigned, 8>
sourceMixedPredSlotsForMappedTargetSlots(
   const CrossQueryStateMatchGroup& group,
   const llvm::DenseMap<uint64_t, llvm::DenseMap<unsigned, unsigned>>& consumerSlotByCacheKeyAndQuery) {
   llvm::SmallVector<unsigned, 8> slots;
   auto itTargetSlots = consumerSlotByCacheKeyAndQuery.find(group.cacheKey);
   if (itTargetSlots == consumerSlotByCacheKeyAndQuery.end()) return slots;
   for (const auto& [queryIdx, targetSlot] : itTargetSlots->second) {
      (void)targetSlot;
      for (uint64_t depKey : group.cacheDeps) {
         auto itDepSlots = consumerSlotByCacheKeyAndQuery.find(depKey);
         if (itDepSlots == consumerSlotByCacheKeyAndQuery.end()) continue;
         auto itDepSlot = itDepSlots->second.find(queryIdx);
         if (itDepSlot == itDepSlots->second.end()) continue;
         slots.push_back(itDepSlot->second);
      }
   }
   sortUniqueSlots(slots);
   return slots;
}

static llvm::SmallVector<tuples::ColumnRefAttr, 4>
threadSourceMixedPredColumnToMaterializeStream(subop::ExecutionStepOp buildStep, unsigned sourceSlot) {
   llvm::SmallVector<DirectMixedDep, 4> directDeps = directMixedDepsForBuildStep(buildStep);
   if (directDeps.empty()) return {};
   subop::MaterializeOp matOp = findJoinBufferMaterializeInStep(buildStep);
   if (!matOp) return {};
   auto* ctx = buildStep.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   llvm::SmallVector<subop::ScanListOp, 4> inputScanLists = scanListsOnMaterializeContextChains(matOp);

   llvm::SmallVector<tuples::ColumnRefAttr, 4> refs;
   llvm::StringSet<> emitted;
   unsigned sourceIdx = 0;
   std::string predName = ("filter_pred$" + llvm::Twine(sourceSlot)).str();
   for (subop::ScanListOp scanList : inputScanLists) {
      auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(scanList.getElem().getColumn().type);
      if (!ler) continue;
      auto scannedLayout = asLocalHivLayoutType(ler.getState());
      if (!localHivLayoutHasFilterPredMember(ctx, scannedLayout)) continue;
      std::optional<uint64_t> depKey = directDepKeyForScanList(scanList, directDeps);
      if (!depKey) continue;
      std::string emittedKey = (llvm::Twine(*depKey) + ":" + llvm::Twine(sourceSlot)).str();
      if (!emitted.insert(emittedKey).second) continue;
      subop::Member predMember;
      for (subop::Member m : scannedLayout.getValueMembers().getMembers()) {
         if (mm.getName(m) == predName) {
            predMember = m;
            break;
         }
      }
      if (!predMember) continue;

      std::string scopeSeed = ("reuse_source_pred$" + llvm::Twine(sourceSlot) + "_" +
                               llvm::Twine(sourceIdx)).str();
      tuples::ColumnDefAttr predDef = cm.createDef(cm.getUniqueScope(scopeSeed), "filter_pred");
      predDef.getColumn().type = mlir::IntegerType::get(ctx, 1);

      mlir::OpBuilder b(scanList);
      b.setInsertionPointAfter(scanList);
      auto mapping = subop::ColumnDefMemberMappingAttr::get(
         ctx, llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>>{{predMember, predDef}});
      auto gather = b.create<subop::GatherOp>(scanList.getLoc(), scanList.getRes().getType(),
                                              scanList.getRes(),
                                              cm.createRef(&scanList.getElem().getColumn()), mapping);
      rewireStreamUsesAfterAnchorInStep(scanList.getRes(), gather.getRes(), scanList.getOperation(),
                                        {scanList.getOperation(), gather.getOperation()});
      gather->setOperand(0, scanList.getRes());
      refs.push_back(threadColumnRefFromSourceToMaterializeStream(matOp, scanList.getOperation(),
                                                                  predDef, cm));
      ++sourceIdx;
   }
   return refs;
}

static void inheritSlotMapForSyntheticTarget(
   uint64_t targetKey, llvm::ArrayRef<uint64_t> deps,
   llvm::DenseMap<uint64_t, llvm::DenseMap<unsigned, unsigned>>& consumerSlotByCacheKeyAndQuery) {
   if (deps.empty()) return;
   llvm::SmallVector<unsigned, 8> queries;
   llvm::DenseSet<unsigned> seenQueries;
   auto addQueries = [&](const llvm::DenseMap<unsigned, unsigned>& slots) {
      for (const auto& [queryIdx, slot] : slots) {
         (void)slot;
         if (seenQueries.insert(queryIdx).second) queries.push_back(queryIdx);
      }
   };
   auto itTarget = consumerSlotByCacheKeyAndQuery.find(targetKey);
   if (itTarget != consumerSlotByCacheKeyAndQuery.end()) addQueries(itTarget->second);
   if (queries.empty()) {
      for (uint64_t depKey : deps) {
         auto itDep = consumerSlotByCacheKeyAndQuery.find(depKey);
         if (itDep != consumerSlotByCacheKeyAndQuery.end()) addQueries(itDep->second);
      }
   }
   if (queries.empty()) return;
   llvm::sort(queries);

   llvm::StringMap<unsigned> slotBySignature;
   llvm::DenseMap<unsigned, unsigned> mergedSlots;
   llvm::DenseSet<unsigned> usedSlots;
   if (itTarget != consumerSlotByCacheKeyAndQuery.end()) {
      for (const auto& [queryIdx, slot] : itTarget->second) {
         mergedSlots[queryIdx] = slot;
         usedSlots.insert(slot);
      }
   }
   unsigned nextSlot = 0;
   for (unsigned queryIdx : queries) {
      std::string sig;
      llvm::raw_string_ostream os(sig);
      std::optional<unsigned> preferredSlot;
      if (itTarget != consumerSlotByCacheKeyAndQuery.end()) {
         auto itSlot = itTarget->second.find(queryIdx);
         if (itSlot != itTarget->second.end()) preferredSlot = itSlot->second;
      }
      for (uint64_t depKey : deps) {
         auto itDep = consumerSlotByCacheKeyAndQuery.find(depKey);
         if (itDep == consumerSlotByCacheKeyAndQuery.end()) continue;
         auto itSlot = itDep->second.find(queryIdx);
         if (itSlot == itDep->second.end()) continue;
         os << "dep:" << depKey << ':' << itSlot->second << '|';
      }
      os.flush();
      if (sig.empty()) continue;
      if (preferredSlot && mergedSlots.lookup(queryIdx) == *preferredSlot) {
         continue;
      }
      auto itSlot = slotBySignature.find(sig);
      if (itSlot == slotBySignature.end()) {
         unsigned assignedSlot = 0;
         bool preferredAlreadyBelongsToQuery =
            preferredSlot && mergedSlots.lookup(queryIdx) == *preferredSlot;
         if (preferredSlot && (preferredAlreadyBelongsToQuery || !usedSlots.contains(*preferredSlot))) {
            assignedSlot = *preferredSlot;
         } else {
            while (usedSlots.contains(nextSlot)) ++nextSlot;
            assignedSlot = nextSlot++;
         }
         usedSlots.insert(assignedSlot);
         itSlot = slotBySignature.try_emplace(sig, assignedSlot).first;
      }
      mergedSlots[queryIdx] = itSlot->second;
   }
   if (!mergedSlots.empty()) consumerSlotByCacheKeyAndQuery[targetKey] = std::move(mergedSlots);
}

static bool materializePredMemberFromUpstreamMixedScanList(subop::ExecutionStepOp buildStep,
                                                           llvm::StringRef predMemberName) {
   subop::MaterializeOp matOp = findJoinBufferMaterializeInStep(buildStep);
   if (!matOp) return false;
   auto* ctx = buildStep.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();

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

   std::string scopeSeed = "reuse_upstream_pred";
   if (auto slot = parseFilterPredMemberSlot(predMemberName))
      scopeSeed = ("reuse_upstream_pred$" + llvm::Twine(*slot)).str();
	   llvm::SmallVector<tuples::ColumnRefAttr, 4> upstreamRefs =
	      threadUpstreamPredColumnsToMaterializeStream(
	         buildStep, predMemberName, /*includeBufferScanRefs=*/!findUniqueTableScanRefsInStep(buildStep));
   if (upstreamRefs.empty()) return false;
   tuples::ColumnRefAttr predRef = insertAndPredsBeforeMaterialize(matOp, upstreamRefs, scopeSeed);

   llvm::SmallVector<subop::RefMappingPairT> pairs;
   for (auto& pr : matOp.getMapping().getMapping()) {
      if (pr.first != outPredMember) pairs.push_back(pr);
   }
   pairs.push_back({outPredMember, predRef});
   matOp.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, pairs));
   return true;
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
   auto combineWithUpstreamPreds = [&](subop::Member predMember,
                                       tuples::ColumnRefAttr localRef,
                                       unsigned qIdx) -> tuples::ColumnRefAttr {
      llvm::SmallVector<tuples::ColumnRefAttr, 4> refs;
      if (localRef) refs.push_back(localRef);
	      llvm::SmallVector<tuples::ColumnRefAttr, 4> upstreamRefs =
	         threadUpstreamPredColumnsToMaterializeStream(
	            syntheticBuild, mm.getName(predMember),
	            /*includeBufferScanRefs=*/!findUniqueTableScanRefsInStep(syntheticBuild));
      refs.append(upstreamRefs.begin(), upstreamRefs.end());
      if (refs.empty()) return {};
      std::string scopeSeed = ("reuse_combined_pred$" + llvm::Twine(qIdx)).str();
      tuples::ColumnRefAttr combined = insertAndPredsBeforeMaterialize(mat, refs, scopeSeed);
      setMaterializeMapping(mat, predMember, combined);
      return combined;
   };
   if (!simpleFilters0.empty()) {
      insertWriteSidePredIntoBufferConstructionStepForPredMember(syntheticBuild, simpleFilters0,
                                                                 mm.getName(predMember0));
      simplePred0Ref = materializedColumnForMember(mat, predMember0);
      assert(simplePred0Ref && "residual filter rewrite: simple predicate materialize missing for q0");
   }
   simplePred0Ref = combineWithUpstreamPreds(predMember0, simplePred0Ref, qIdx0);
   if (!simpleFilters1.empty()) {
      insertWriteSidePredIntoBufferConstructionStepForPredMember(syntheticBuild, simpleFilters1,
                                                                 mm.getName(predMember1));
      simplePred1Ref = materializedColumnForMember(mat, predMember1);
      assert(simplePred1Ref && "residual filter rewrite: simple predicate materialize missing for q1");
   }
   simplePred1Ref = combineWithUpstreamPreds(predMember1, simplePred1Ref, qIdx1);
   subop::MapOp syntheticMap = synthResidual ? synthResidual->predMap
                                             : createResidualPredicateMapAfterTableScan(
                                                  syntheticBuild, peerResidual0 ? peerResidual0->predMap
                                                                                : peerResidual1->predMap,
                                                  peerResidual0 ? peerResidual0->filter
                                                                : peerResidual1->filter);
   if (simplePred0Ref) moveSimplePredicateProducerBeforeResidualMap(syntheticBuild, syntheticMap, simplePred0Ref);
   if (simplePred1Ref) moveSimplePredicateProducerBeforeResidualMap(syntheticBuild, syntheticMap, simplePred1Ref);
   if (peerResidual0) ensureSyntheticMapHasPeerInputs(syntheticBuild, syntheticMap, peerResidual0->predMap,
                                                      peerResidual0->filter);
   if (peerResidual1) ensureSyntheticMapHasPeerInputs(syntheticBuild, syntheticMap, peerResidual1->predMap,
                                                      peerResidual1->filter);

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
      llvm::SmallVector<tuples::ColumnRefAttr, 2> predRefs{pred0Ref, pred1Ref};
      finalPredicateStream = insertResidualFilterUnionAfterPredicates(finalPredicateStream, predRefs);
   }
   if (synthResidual) {
      synthResidual->filter.getRes().replaceAllUsesWith(finalPredicateStream);
      synthResidual->filter.erase();
   }

   return true;
}

static bool rewriteSyntheticResidualFiltersAsFilterPredsForSlots(
   subop::ExecutionStepOp syntheticBuild,
   llvm::ArrayRef<unsigned> predQueryIndices,
   const llvm::DenseMap<unsigned, subop::ExecutionStepOp>& peerBuilds,
   subop::MaterializeOp mat,
   const llvm::DenseMap<unsigned, subop::Member>& predMembers,
   const llvm::DenseMap<unsigned, llvm::SmallVector<runtime::FilterDescription, 8>>& simpleFilters,
   uint64_t targetCacheKey,
   const llvm::DenseMap<uint64_t, llvm::DenseMap<unsigned, unsigned>>* consumerSlotByCacheKeyAndQuery) {
   if (predQueryIndices.size() < 2) return false;

   llvm::DenseMap<unsigned, llvm::SmallVector<ResidualTableFilter, 4>> residualsBySlot;
   llvm::StringSet<> residualFingerprints;
   for (unsigned qIdx : predQueryIndices) {
      subop::ExecutionStepOp peerBuild = peerBuilds.lookup(qIdx);
      assert(peerBuild && "residual filter rewrite: missing peer build");
      llvm::SmallVector<ResidualTableFilter, 4> residuals = collectResidualTableFiltersInBuildStep(peerBuild);
      if (residuals.empty()) continue;
      auto& slotResiduals = residualsBySlot[qIdx];
      for (ResidualTableFilter residual : residuals) {
         residualFingerprints.insert(residualFilterFingerprint(residual));
         slotResiduals.push_back(residual);
      }
   }
   if (residualsBySlot.empty()) return false;

   llvm::SmallVector<ResidualTableFilter, 4> synthResiduals =
      collectResidualTableFiltersInBuildStep(syntheticBuild);
   auto* ctx = syntheticBuild.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

   auto seedIt = residualsBySlot.begin();
   assert(!seedIt->second.empty() && "residual rewrite seed slot must have a residual");
   subop::MapOp seedMap = !synthResiduals.empty() ? synthResiduals.front().predMap
                                                  : seedIt->second.front().predMap;
   subop::MapOp syntheticMap = !synthResiduals.empty()
                                  ? synthResiduals.front().predMap
                                  : createResidualPredicateMapAfterTableScan(syntheticBuild, seedMap,
                                                                            seedIt->second.front().filter);

	   llvm::DenseMap<unsigned, tuples::ColumnRefAttr> simplePredBySlot;
	   llvm::DenseMap<unsigned, llvm::SmallVector<tuples::ColumnRefAttr, 4>> materializePredRefsBySlot;
	   bool hasTableScanSource = !!findUniqueTableScanRefsInStep(syntheticBuild);
	   for (unsigned qIdx : predQueryIndices) {
	      subop::ExecutionStepOp peerBuild = peerBuilds.lookup(qIdx);
	      assert(peerBuild && "residual filter rewrite: missing peer build");
	      llvm::SmallVector<runtime::FilterDescription, 8> peerFilters =
	         decodeFiltersFromTableScanInExecutionStep(peerBuild);
	      if (peerFilters.empty()) continue;
	      auto itSimple = simpleFilters.find(qIdx);
	      llvm::ArrayRef<runtime::FilterDescription> localFilters =
	         itSimple == simpleFilters.end() ? llvm::ArrayRef<runtime::FilterDescription>{}
	                                        : llvm::ArrayRef<runtime::FilterDescription>{itSimple->second};
	      bool localFiltersCoverPeerFilters = filterClauseEquals(localFilters, peerFilters);
	      if (!hasTableScanSource) continue;
	      subop::Member predMember = predMembers.lookup(qIdx);
	      assert(predMember && "residual filter rewrite: missing predicate member");
	      llvm::SmallVector<tuples::ColumnRefAttr, 4> sharedPredRefs =
	         threadUpstreamSharedTablePredColumnsToMaterializeStream(syntheticBuild, mm.getName(predMember));
	      if (!sharedPredRefs.empty()) {
	         materializePredRefsBySlot[qIdx].append(sharedPredRefs.begin(), sharedPredRefs.end());
	         continue;
	      }
	      insertWriteSidePredIntoBufferConstructionStepForPredMember(
	         syntheticBuild, localFiltersCoverPeerFilters ? localFilters : llvm::ArrayRef<runtime::FilterDescription>{peerFilters},
	         mm.getName(predMember),
	         /*allowSharedScanPredicate=*/!localFiltersCoverPeerFilters);
      tuples::ColumnRefAttr simpleRef = materializedColumnForMember(mat, predMember);
      assert(simpleRef && "residual filter rewrite: simple predicate materialize missing");
      if (moveSimplePredicateProducerBeforeResidualMap(syntheticBuild, syntheticMap, simpleRef)) {
         simplePredBySlot[qIdx] = simpleRef;
      } else {
         materializePredRefsBySlot[qIdx].push_back(simpleRef);
      }
   }
   for (unsigned qIdx : predQueryIndices) {
      subop::Member predMember = predMembers.lookup(qIdx);
      assert(predMember && "residual filter rewrite: missing predicate member");
      llvm::SmallVector<tuples::ColumnRefAttr, 4> upstreamRefs;
      if (consumerSlotByCacheKeyAndQuery) {
         upstreamRefs = threadMappedUpstreamMixedPredColumnsToMaterializeStream(
            syntheticBuild, qIdx, targetCacheKey, *consumerSlotByCacheKeyAndQuery);
      }
      if (upstreamRefs.empty()) {
         upstreamRefs =
            threadUpstreamPredColumnsToMaterializeStream(
               syntheticBuild, mm.getName(predMember),
               /*includeBufferScanRefs=*/!findUniqueTableScanRefsInStep(syntheticBuild));
      }
      materializePredRefsBySlot[qIdx].append(upstreamRefs.begin(), upstreamRefs.end());
   }

   for (auto& it : residualsBySlot) {
      for (ResidualTableFilter residual : it.second)
         ensureSyntheticMapHasPeerInputs(syntheticBuild, syntheticMap, residual.predMap, residual.filter);
   }

   llvm::SmallVector<tuples::ColumnRefAttr, 8> predRefs;
   predRefs.reserve(predQueryIndices.size());
   for (unsigned qIdx : predQueryIndices) {
      subop::Member predMember = predMembers.lookup(qIdx);
      assert(predMember && "residual filter rewrite: missing predicate member");
      llvm::SmallVector<tuples::ColumnRefAttr, 4> slotPredRefs;
      auto itResidual = residualsBySlot.find(qIdx);
      if (itResidual != residualsBySlot.end()) {
         for (size_t idx = 0; idx < itResidual->second.size(); ++idx) {
            tuples::ColumnDefAttr predDef = makeResidualFilterPredDef(ctx, qIdx);
            if (idx != 0) {
               auto [scope, leaf] = cm.getName(&predDef.getColumn());
               (void)leaf;
               predDef = cm.createDef(cm.getUniqueScope(scope), "filter_pred");
               predDef.getColumn().type = mlir::IntegerType::get(ctx, 1);
            }
            ResidualTableFilter residual = itResidual->second[idx];
            slotPredRefs.push_back(appendPeerResidualPredicateToSyntheticMap(syntheticMap, residual.predMap,
                                                                             residual.filter, predDef));
         }
      } else {
         tuples::ColumnDefAttr predDef = makeResidualFilterPredDef(ctx, qIdx);
         slotPredRefs.push_back(appendTrueResidualPredicateToSyntheticMap(syntheticMap, predDef));
      }
      if (auto itSimple = simplePredBySlot.find(qIdx); itSimple != simplePredBySlot.end()) {
         unsigned predIdx = syntheticMap.getComputedCols().size() - 1;
         andResidualPredicateWithSimpleFilter(syntheticMap, predIdx, itSimple->second);
      }
      llvm::SmallVector<tuples::ColumnRefAttr, 4> materializeRefs;
      for (tuples::ColumnRefAttr predRef : slotPredRefs) {
         tuples::ColumnDefAttr materializedPredDef = cm.createDef(&predRef.getColumn());
         materializeRefs.push_back(threadColumnRefFromSourceToMaterializeStream(
            mat, syntheticMap.getOperation(), materializedPredDef, cm));
      }
      if (auto itExtra = materializePredRefsBySlot.find(qIdx);
          itExtra != materializePredRefsBySlot.end() && !itExtra->second.empty()) {
         materializeRefs.append(itExtra->second.begin(), itExtra->second.end());
      }
      tuples::ColumnRefAttr materializedPredRef = insertAndPredsBeforeMaterialize(
         mat, materializeRefs, ("reuse_combined_pred$" + llvm::Twine(qIdx)).str());
      setMaterializeMapping(mat, predMember, materializedPredRef);
      predRefs.append(slotPredRefs.begin(), slotPredRefs.end());
   }

   mlir::Value finalPredicateStream = syntheticMap.getResult();
   finalPredicateStream = insertResidualFilterUnionAfterPredicates(finalPredicateStream, predRefs);
   for (size_t idx = 0; idx < synthResiduals.size(); ++idx) {
      ResidualTableFilter residual = synthResiduals[idx];
      if (idx == 0) {
         residual.filter.getRes().replaceAllUsesWith(finalPredicateStream);
      } else {
         residual.filter.getRes().replaceAllUsesWith(residual.filter.getStream());
      }
      residual.filter.erase();
   }
   deriveNonNullColumnsFromWidenedNullableGathers(syntheticBuild);
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

static void alignMaterializedPayloadMembersToUnionPlan(
   subop::MaterializeOp mat, const JoinBufferUnionPlan& plan, subop::MemberManager& mm,
   const llvm::DenseMap<const void*, uint64_t>& columnHashes) {
   llvm::DenseMap<uint64_t, subop::Member> targetMemberBySemanticHash;
   llvm::StringMap<subop::Member> targetMemberByName;
   assert(plan.payloadColumns.size() == plan.payloadMembers.size());
   for (size_t i = 0; i < plan.payloadColumns.size(); ++i) {
      const PayloadColumnSpec& spec = plan.payloadColumns[i];
      targetMemberBySemanticHash[spec.semanticHash] = plan.payloadMembers[i];
      targetMemberByName[mm.getName(plan.payloadMembers[i])] = plan.payloadMembers[i];
   }

   bool changed = false;
   llvm::SmallVector<subop::RefMappingPairT> pairs;
   pairs.reserve(mat.getMapping().getMapping().size());
   llvm::DenseSet<subop::Member> usedPlanMembers;
   for (auto& [member, colRef] : mat.getMapping().getMapping()) {
      subop::Member outMember = member;
      llvm::StringRef memName = mm.getName(member);
         if (member != plan.linkMember && member != plan.hashMember && !isJoinBufferInternalMemberName(memName)) {
         bool resolved = false;
         auto tryUseTarget = [&](subop::Member candidate) -> bool {
            if (!candidate) return false;
            if (mm.getType(candidate) != colRef.getColumn().type) return false;
            if (usedPlanMembers.contains(candidate)) return false;
            outMember = candidate;
            resolved = true;
            return true;
         };
         if (auto predSlot = parseFilterPredMemberSlot(memName)) {
            std::string predName = "filter_pred$" + std::to_string(*predSlot);
            if (auto itPred = targetMemberByName.find(predName); itPred != targetMemberByName.end())
               (void)tryUseTarget(itPred->second);
         }
         if (outMember == member) {
            auto hash = payloadColumnIdentityHash(colRef, columnHashes);
            auto it = targetMemberBySemanticHash.find(hash);
            if (it != targetMemberBySemanticHash.end()) {
               (void)tryUseTarget(it->second);
            }
         }
         assert(resolved &&
                "join superset: materialized payload must resolve to a union member by semantic identity");
      }
      if (outMember != member) changed = true;
      usedPlanMembers.insert(outMember);
      pairs.push_back({outMember, colRef});
   }
   if (changed) mat.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(mat.getContext(), pairs));
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
   llvm::StringSet<> usedPayloadMemberNames;
   for (const PayloadColumnSpec& spec : plan.payloadColumns) {
      unsigned predIdx = 0;
      if (parseFilterPredLayoutSemanticKey(spec.semanticKey, predIdx)) {
         subop::Member predMember = mm.getOrCreateMemberDirect(
            "filter_pred$" + llvm::Twine(predIdx).str(), canonicalPredTy, /*allowTypeUpdate=*/false);
         assert(usedPayloadMemberNames.insert(mm.getName(predMember)).second &&
                "join superset: duplicate filter_pred payload member in union plan");
         plan.payloadMembers.push_back(predMember);
         continue;
      }
      mlir::Type slotTy = cloneTypeToContext(spec.colType, synthCtx);
      auto tryUseMember = [&](subop::Member candidate) -> bool {
         if (!candidate) return false;
         if (mm.getType(candidate) != slotTy) return false;
         if (!usedPayloadMemberNames.insert(mm.getName(candidate)).second) return false;
         plan.payloadMembers.push_back(candidate);
         return true;
      };
      auto it = bySemanticHash.find(spec.semanticHash);
      if (it != bySemanticHash.end()) {
         if (tryUseMember(it->second)) continue;
      }
      subop::Member freshMember = allocUnusedPayloadMemberSlot(mm, slotTy, nextSlot);
      assert(usedPayloadMemberNames.insert(mm.getName(freshMember)).second &&
             "join superset: fresh payload member must be unique in union plan");
      plan.payloadMembers.push_back(freshMember);
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
      // The highest slot is reserved for the group/union predicate when a mixed HIV is produced
      // from several per-query predicate slots. Using it as the default scan slot keeps downstream
      // synthetic builds from seeing only the donor query's rows; consumer probes are retagged to
      // their specific per-query slot later.
      if (!best || *slot > *bestSlot) {
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
                                                   const llvm::DenseSet<void*>* closure,
                                                   const llvm::DenseMap<const void*, uint64_t>* columnHashes,
                                                   const llvm::DenseMap<uint64_t, subop::Member>* targetMemberByColumnHash);

static void applyBufferLayoutToSsaClosure(mlir::ModuleOp module, llvm::ArrayRef<mlir::Value> canonicalBuffers,
                                          subop::StateMembersAttr targetMembers, const ModuleReuseInfo& reuse,
                                          const llvm::DenseMap<const void*, uint64_t>* columnHashes = nullptr,
                                          const llvm::DenseMap<uint64_t, subop::Member>* targetMemberByColumnHash = nullptr) {
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
   syncMaterializeMappingsToBufferMembers(module, targetMembers, &closure, columnHashes, targetMemberByColumnHash);
   synchronizeExecutionStepPortTypes(module, &closure);
}

/// After buffer layout is applied, canonicalize \c materialize member slots to \p targetMembers (by name / payload hash).
static void syncMaterializeMappingsToBufferMembers(mlir::ModuleOp module, subop::StateMembersAttr targetMembers,
                                                   const llvm::DenseSet<void*>* closure,
                                                   const llvm::DenseMap<const void*, uint64_t>* columnHashes,
                                                   const llvm::DenseMap<uint64_t, subop::Member>* targetMemberByColumnHash) {
   auto* ctx = module.getContext();
	   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
	   llvm::StringMap<subop::Member> validByName;
	   subop::Member targetLinkMember;
	   subop::Member targetHashMember;
	   for (subop::Member m : targetMembers.getMembers()) {
	      llvm::StringRef name = mm.getName(m);
	      validByName[name] = m;
	      if (name.starts_with("link$")) targetLinkMember = m;
	      if (name.starts_with("hash$")) targetHashMember = m;
	   }

   module.walk([&](subop::MaterializeOp mat) {
      if (closure && !opaqueClosureContains(*closure, mat.getState())) return;
      subop::BufferType bufTy = getInnerBufferTypeForMaterializeState(mat.getState().getType());
      if (!bufTy || bufTy.getMembers() != targetMembers) return;

      llvm::DenseMap<uint64_t, subop::Member> byPayloadHash;
      llvm::DenseSet<subop::Member> usedTargetMembers;
      for (auto& pr : mat.getMapping().getMapping()) {
         if (auto itName = validByName.find(mm.getName(pr.first)); itName != validByName.end())
            usedTargetMembers.insert(itName->second);
      }
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
         bool payloadLike = !(memName.starts_with("link$") || memName.starts_with("hash$"));
         if (payloadLike && columnHashes) {
            auto itHash = columnHashes->find(&pr.second.getColumn());
            if (itHash != columnHashes->end()) {
               if (targetMemberByColumnHash) {
                  if (auto it = targetMemberByColumnHash->find(itHash->second);
                      it != targetMemberByColumnHash->end()) {
                     canon = it->second;
                  }
               }
               if (!canon) {
                  if (auto it = byPayloadHash.find(itHash->second); it != byPayloadHash.end())
                     canon = it->second;
               }
            }
         }
         if (!canon) {
            if (auto it = validByName.find(memName); it != validByName.end()) canon = it->second;
         }
	         if (!canon && memName.starts_with("link$")) canon = targetLinkMember;
	         if (!canon && memName.starts_with("hash$")) canon = targetHashMember;
	         if (!canon) {
	            if (auto slot = parseFilterPredMemberSlot(memName)) {
	               std::string predName = "filter_pred$" + std::to_string(*slot);
               if (auto it = validByName.find(predName); it != validByName.end()) canon = it->second;
            }
         }
         assert(canon && "materialize mapping must resolve to a member of the target buffer layout");
         usedTargetMembers.insert(canon);
         pairs.push_back({canon, pr.second});
      }
      mat.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, pairs));
   });
}

static void syncMaterializeMappingsToActualBufferTypes(mlir::ModuleOp module) {
   module.walk([&](subop::MaterializeOp mat) {
      mlir::Value actualState = peelBlockArgsToEnclosingOperands(mat.getState());
      subop::BufferType bufTy = getInnerBufferTypeForMaterializeState(actualState.getType());
      if (!bufTy) bufTy = getInnerBufferTypeForMaterializeState(mat.getState().getType());
      if (!bufTy) return;
      llvm::DenseSet<subop::Member> bufferMembers(bufTy.getMembers().getMembers().begin(),
                                                  bufTy.getMembers().getMembers().end());
      llvm::DenseSet<subop::Member> usedMembers;
      auto& mm = mat.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
      subop::Member actualLinkMember;
      subop::Member actualHashMember;
      for (subop::Member m : bufTy.getMembers().getMembers()) {
         llvm::StringRef name = mm.getName(m);
         if (name.starts_with("link$")) actualLinkMember = m;
         if (name.starts_with("hash$")) actualHashMember = m;
      }

      llvm::SmallVector<subop::RefMappingPairT> pairs;
      pairs.reserve(mat.getMapping().getMapping().size());
      for (auto& pr : mat.getMapping().getMapping()) {
         subop::Member member = pr.first;
         if (!bufferMembers.contains(member)) {
            llvm::StringRef name = mm.getName(member);
            if (name.starts_with("link$") && actualLinkMember) member = actualLinkMember;
            else if (name.starts_with("hash$") && actualHashMember) member = actualHashMember;
         }
         assert(bufferMembers.contains(member) && "materialize mapping must target the buffer layout");
         assert(!usedMembers.contains(member) && "materialize mapping must not write the same member twice");
	         usedMembers.insert(member);
	         pairs.push_back({member, pr.second});
	      }
	      mat.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(mat.getContext(), pairs));
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
      merged.sharedPredicateClauses.clear();
      merged.sharedPredicateSlots.clear();
      return;
   }

   if (allExternalFilterSourcesIdentical(filterSources)) {
      merged.filterDescriptions = filterSources.front().filterDescriptions;
      merged.orFilterClauses.clear();
      merged.sharedPredicateClauses.clear();
      merged.sharedPredicateSlots.clear();
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
   merged.sharedPredicateClauses.clear();
   merged.sharedPredicateSlots.clear();
   if (uniqueClauses.empty()) return;
   merged.filterDescriptions.assign(uniqueClauses.front().begin(), uniqueClauses.front().end());
   for (const auto& clause : llvm::drop_begin(uniqueClauses)) {
      merged.orFilterClauses.emplace_back(clause.begin(), clause.end());
   }
   for (const auto& clause : uniqueClauses) {
      merged.sharedPredicateClauses.emplace_back(clause.begin(), clause.end());
   }
   merged.sharedPredicateSlots.reserve(merged.sharedPredicateClauses.size());
   for (size_t i = 0; i < merged.sharedPredicateClauses.size(); ++i)
      merged.sharedPredicateSlots.push_back(i);
}

static void mergeExternalFiltersForOrReuseBySlot(
   ExternalDatasourceProperty& merged,
   llvm::SmallVector<std::pair<unsigned, ExternalDatasourceProperty>, 4> filterSourcesBySlot) {
   llvm::sort(filterSourcesBySlot, [](const auto& a, const auto& b) {
      return a.first < b.first;
   });

   llvm::SmallVector<std::pair<unsigned, ExternalDatasourceProperty>, 4> uniqueBySlot;
   for (auto& source : filterSourcesBySlot) {
      if (!uniqueBySlot.empty() && uniqueBySlot.back().first == source.first) {
         assert(externalDatasourceFiltersEqual(uniqueBySlot.back().second, source.second) &&
                "one reuse slot must not map to different shared-table predicates");
         continue;
      }
      uniqueBySlot.push_back(std::move(source));
   }

   llvm::SmallVector<ExternalDatasourceProperty, 4> orderedSources;
   orderedSources.reserve(uniqueBySlot.size());
   for (const auto& source : uniqueBySlot) orderedSources.push_back(source.second);
   mergeExternalFiltersForOrReuse(merged, orderedSources);
   if (merged.sharedPredicateClauses.empty()) return;

   merged.sharedPredicateClauses.clear();
   merged.sharedPredicateSlots.clear();
   merged.sharedPredicateClauses.reserve(uniqueBySlot.size());
   merged.sharedPredicateSlots.reserve(uniqueBySlot.size());
   for (const auto& source : uniqueBySlot) {
      assert(source.second.orFilterClauses.empty() &&
             "shared-table predicate slots currently require one conjunctive source clause per slot");
      merged.sharedPredicateClauses.emplace_back(source.second.filterDescriptions.begin(),
                                                source.second.filterDescriptions.end());
      merged.sharedPredicateSlots.push_back(source.first);
   }
}

static void collectFilterColumns(llvm::SmallVectorImpl<lingodb::runtime::FilterDescription>& out,
                                 const ExternalDatasourceProperty& ds) {
   out.append(ds.filterDescriptions.begin(), ds.filterDescriptions.end());
   for (const auto& clause : ds.orFilterClauses) out.append(clause.begin(), clause.end());
   for (const auto& clause : ds.sharedPredicateClauses) out.append(clause.begin(), clause.end());
}

static mlir::Type nullableTypeFromRuntimeFilterLiteral(mlir::MLIRContext* ctx,
                                                       const lingodb::runtime::FilterDescription& f) {
   if (f.op == lingodb::runtime::FilterOp::NOTNULL) return {};
   auto nullable = [&](mlir::Type ty) -> mlir::Type {
      if (!ty || mlir::isa<db::NullableType>(ty)) return ty;
      return db::NullableType::get(ty);
   };
   auto intTypeForValue = [&](int64_t v) -> mlir::Type {
      if (v >= std::numeric_limits<int32_t>::min() && v <= std::numeric_limits<int32_t>::max())
         return mlir::IntegerType::get(ctx, 32);
      return mlir::IntegerType::get(ctx, 64);
   };
   auto intTypeForValues = [&](const std::vector<int64_t>& values) -> mlir::Type {
      for (int64_t v : values) {
         if (v < std::numeric_limits<int32_t>::min() || v > std::numeric_limits<int32_t>::max())
            return mlir::IntegerType::get(ctx, 64);
      }
      return mlir::IntegerType::get(ctx, 32);
   };
   if (f.op == lingodb::runtime::FilterOp::IN) {
      if (std::holds_alternative<std::vector<std::string>>(f.values))
         return nullable(db::StringType::get(ctx));
      if (std::holds_alternative<std::vector<int64_t>>(f.values))
         return nullable(intTypeForValues(std::get<std::vector<int64_t>>(f.values)));
      if (std::holds_alternative<std::vector<double>>(f.values))
         return nullable(mlir::Float64Type::get(ctx));
      return {};
   }
   if (std::holds_alternative<std::string>(f.value)) return nullable(db::StringType::get(ctx));
   if (std::holds_alternative<int64_t>(f.value)) return nullable(intTypeForValue(std::get<int64_t>(f.value)));
   if (std::holds_alternative<double>(f.value)) return nullable(mlir::Float64Type::get(ctx));
   return {};
}

static void addFilterOnlyColumnsToExternalMapping(
   ExternalDatasourceProperty& merged, mlir::MLIRContext* ctx,
   llvm::function_ref<mlir::Type(llvm::StringRef)> lookupKnownType,
   llvm::StringMap<mlir::Type>& typeByIdentifier,
   llvm::ArrayRef<ExternalDatasourceProperty> filterSources = {}) {
   llvm::StringSet<> have;
   for (const auto& map : merged.mapping) {
      have.insert(normalizeColumnIdentifier(map.identifier));
      have.insert(normalizeColumnIdentifier(map.memberName));
   }

   llvm::SmallVector<lingodb::runtime::FilterDescription, 8> filters;
   collectFilterColumns(filters, merged);
   for (const ExternalDatasourceProperty& src : filterSources) collectFilterColumns(filters, src);
   for (const auto& f : filters) {
      llvm::StringRef id = normalizeColumnIdentifier(f.columnName);
      if (id.empty() || have.contains(id)) continue;

      mlir::Type ty = lookupKnownType(id);
      if (!ty) ty = nullableTypeFromRuntimeFilterLiteral(ctx, f);
      if (!ty && f.op == lingodb::runtime::FilterOp::NOTNULL) continue;
      assert(ty && "join superset: filter-only external column needs a known type");
      typeByIdentifier[id] = ty;
      merged.mapping.push_back({id.str(), id.str()});
      have.insert(id);
   }
   llvm::sort(merged.mapping, [](const auto& x, const auto& y) { return x.memberName < y.memberName; });
}

static subop::ExecutionStepOp createMergedExternalTableRefStep(mlir::OpBuilder& gb, mlir::Location loc,
                                                               mlir::Type tableTy, llvm::StringRef descrHex) {
   auto step = gb.create<subop::ExecutionStepOp>(loc, mlir::TypeRange{tableTy}, mlir::ValueRange{},
                                                gb.getArrayAttr({gb.getBoolAttr(false)}));
   auto& block = step.getSubOps().emplaceBlock();
   mlir::OpBuilder ib = mlir::OpBuilder::atBlockBegin(&block);
   auto ge = ib.create<subop::GetExternalOp>(loc, tableTy, mlir::StringAttr::get(gb.getContext(), descrHex));
   ib.create<subop::ExecutionStepReturnOp>(loc, ge.getRes());
   return step;
}

static void rewireBuildStepScannedTable(subop::ExecutionStepOp buildStep, subop::ScanRefsOp scanOp,
                                        mlir::Value oldTableState, mlir::Value newTableState, mlir::Type newTy) {
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
   auto refDef = scanOp.getRef();
   if (auto shared = mlir::dyn_cast<subop::SharedTableType>(newTy)) {
      refDef.getColumn().type = subop::SharedTableEntryRefType::get(
         scanOp.getContext(), shared.getTableMembers(), shared.getPredicateMembers());
   } else if (auto table = mlir::dyn_cast<subop::TableType>(newTy)) {
      refDef.getColumn().type = subop::TableEntryRefType::get(scanOp.getContext(), table.getMembers());
   } else {
      llvm_unreachable("rewired scan_refs state must be table-like");
   }
   scanOp.setRefAttr(refDef);
}

static void deriveNonNullColumnsFromWidenedNullableGathers(subop::ExecutionStepOp buildStep) {
   auto* ctx = buildStep.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

   llvm::SmallVector<subop::GatherOp, 4> gathers;
   buildStep.walk([&](subop::GatherOp gather) { gathers.push_back(gather); });
   for (subop::GatherOp gather : gathers) {
      struct DerivedColumn {
         tuples::ColumnDefAttr nullableDef;
         tuples::ColumnDefAttr nonNullDef;
      };
      llvm::SmallVector<subop::DefMappingPairT> mappingPairs;
      llvm::SmallVector<DerivedColumn, 2> derivedColumns;
      bool changed = false;
      for (auto& pr : gather.getMapping().getMapping()) {
         mlir::Type memberTy = mm.getType(pr.first);
         mlir::Type defTy = pr.second.getColumn().type;
         auto nullableTy = mlir::dyn_cast<db::NullableType>(memberTy);
         if (!nullableTy || nullableTy.getType() != defTy) {
            mappingPairs.push_back(pr);
            continue;
         }

         auto [scope, leaf] = cm.getName(&pr.second.getColumn());
         tuples::ColumnDefAttr nullableDef = cm.createDef(scope, sourceColumnIdentifierForPayloadLeaf(leaf));
         nullableDef.getColumn().type = memberTy;
         mappingPairs.push_back({pr.first, nullableDef});
         derivedColumns.push_back({nullableDef, pr.second});
         changed = true;
      }
      if (!changed) continue;

      gather.setMappingAttr(subop::ColumnDefMemberMappingAttr::get(ctx, mappingPairs));
      MapCreationHelper helper(ctx);
      mlir::OpBuilder b(gather);
      helper.buildBlock(b, [&](mlir::OpBuilder& rb) {
         llvm::SmallVector<mlir::Value, 2> returns;
         for (const DerivedColumn& col : derivedColumns) {
            tuples::ColumnRefAttr nullableRef = cm.createRef(&col.nullableDef.getColumn());
            mlir::Value nullableValue = helper.access(nullableRef, gather.getLoc());
            returns.push_back(rb.create<db::NullableGetVal>(gather.getLoc(), col.nonNullDef.getColumn().type,
                                                            nullableValue));
         }
         rb.create<tuples::ReturnOp>(gather.getLoc(), returns);
      });
      llvm::SmallVector<mlir::Attribute, 2> computed;
      for (const DerivedColumn& col : derivedColumns) computed.push_back(col.nonNullDef);
      b.setInsertionPointAfter(gather);
      auto map = b.create<subop::MapOp>(gather.getLoc(), tuples::TupleStreamType::get(ctx),
                                        gather.getRes(), b.getArrayAttr(computed), helper.getColRefs());
      map.getFn().push_back(helper.getMapBlock());
      rewireStreamUsesAfterAnchorInStep(gather.getRes(), map.getResult(), map.getOperation(),
                                        {gather.getOperation(), map.getOperation()});
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

static subop::StateMembersAttr tableDataMembersForType(mlir::Type tableLikeTy) {
   if (auto tableTy = mlir::dyn_cast<subop::TableType>(tableLikeTy)) return tableTy.getMembers();
   if (auto sharedTy = mlir::dyn_cast<subop::SharedTableType>(tableLikeTy)) return sharedTy.getTableMembers();
   return {};
}

static subop::Member tableMemberForIdentifier(subop::TableType tableTy, subop::MemberManager& mm,
                                              llvm::StringRef identifier);

static mlir::Type tableTypeFromMergedExternal(mlir::MLIRContext* ctx, subop::MemberManager& mm,
                                              const ExternalDatasourceProperty& ds, subop::TableType hintTy,
                                              llvm::function_ref<mlir::Type(llvm::StringRef)> lookupPeerType) {
   llvm::SmallVector<subop::Member> members;
   for (const auto& map : ds.mapping) {
      mlir::Type ty = memberTypeForIdentifier(hintTy, mm, map.identifier);
      mlir::Type peerTy = lookupPeerType(map.identifier);
      if (peerTy) peerTy = cloneTypeToContext(peerTy, ctx);
      ty = reconcileExternalColumnTypes(ty, peerTy);
      assert(ty && "join superset: missing column type in merged external layout");
      if (subop::Member existing = tableMemberForIdentifier(hintTy, mm, map.identifier)) {
         if (mm.getType(existing) != ty)
            existing = mm.getOrCreateMemberDirect(mm.getName(existing), ty, /*allowTypeUpdate=*/true);
         members.push_back(existing);
      } else {
         members.push_back(mm.createMember(map.identifier, ty));
      }
   }
   auto tableMembers = subop::StateMembersAttr::get(ctx, members);
   if (ds.sharedPredicateClauses.empty())
      return subop::TableType::get(ctx, tableMembers, hintTy.getFiltered());

   llvm::SmallVector<subop::Member> predMembers;
   assert((ds.sharedPredicateSlots.empty() ||
           ds.sharedPredicateSlots.size() == ds.sharedPredicateClauses.size()) &&
          "shared predicate slot metadata must align with physical predicate clauses");
   for (size_t i = 0; i < ds.sharedPredicateClauses.size(); ++i) {
      uint64_t logicalSlot = ds.sharedPredicateSlots.empty() ? i : ds.sharedPredicateSlots[i];
      predMembers.push_back(mm.getOrCreateMemberDirect((llvm::Twine("filter_pred$") + llvm::Twine(logicalSlot)).str(),
                                                       mlir::IntegerType::get(ctx, 1),
                                                       /*allowTypeUpdate=*/false));
   }
   return subop::SharedTableType::get(ctx, tableMembers,
                                      subop::StateMembersAttr::get(ctx, predMembers),
                                      hintTy.getFiltered());
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

static subop::Member tableMemberForIdentifier(mlir::Type tableLikeTy, subop::MemberManager& mm,
                                              llvm::StringRef identifier) {
   subop::StateMembersAttr members = tableDataMembersForType(tableLikeTy);
   if (!members) return {};
   llvm::StringRef id = normalizeColumnIdentifier(identifier);
   for (subop::Member m : members.getMembers()) {
      if (stripMemberSuffix(mm.getName(m)) == id) return m;
   }
   return {};
}

/// Propagate a widened \c !subop.table type along SSA values and \c execution_step operand/block-arg ports.
static void refreshTableStateTypesInModule(mlir::ModuleOp module, mlir::Value tableStateRoot, mlir::Type newTy) {
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
   mlir::Type entryRefTy;
   if (auto tableTy = mlir::dyn_cast<subop::TableType>(newTy)) {
      entryRefTy = subop::TableEntryRefType::get(ctx, tableTy.getMembers());
   } else if (auto sharedTy = mlir::dyn_cast<subop::SharedTableType>(newTy)) {
      entryRefTy = subop::SharedTableEntryRefType::get(ctx, sharedTy.getTableMembers(), sharedTy.getPredicateMembers());
   } else {
      llvm_unreachable("refreshed table state must be table-like");
   }

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
      auto defPtr = pr.second.getColumnPtr();
      if (!defPtr) continue;
      auto [scope, leaf] = cm.getName(defPtr.get());
      tuples::ColumnRefAttr outRef = cm.createRef(defPtr.get());
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
         auto cref = mlir::dyn_cast<tuples::ColumnRefAttr>(attr);
         if (!cref) {
            newInputs.push_back(attr);
            continue;
         }
         auto crefPtr = cref.getColumnPtr();
         if (!crefPtr) {
            newInputs.push_back(cref);
            continue;
         }
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
         auto replacementPtr = replacement ? replacement.getColumnPtr() : decltype(crefPtr){};
         if (replacementPtr && crefPtr != replacementPtr) {
            newInputs.push_back(replacement);
            changed = true;
         } else {
            newInputs.push_back(cref);
         }
      }
      if (changed) mapOp.setInputColsAttr(mlir::ArrayAttr::get(ctx, newInputs));
   }
}

static bool sameColumnIdentity(tuples::ColumnRefAttr a, tuples::ColumnRefAttr b,
                               tuples::ColumnManager& cm) {
   auto [aScope, aLeaf] = cm.getName(&a.getColumn());
   auto [bScope, bLeaf] = cm.getName(&b.getColumn());
   return aScope == bScope && aLeaf == bLeaf && a.getColumn().type == b.getColumn().type;
}

static std::optional<unsigned> findNestedMapParameter(subop::NestedMapOp nested,
                                                      tuples::ColumnRefAttr ref,
                                                      tuples::ColumnManager& cm) {
   for (unsigned i = 0; i < nested.getParameters().size(); ++i) {
      auto existing = mlir::cast<tuples::ColumnRefAttr>(nested.getParameters()[i]);
      if (sameColumnIdentity(existing, ref, cm)) return i;
   }
   return std::nullopt;
}

static mlir::BlockArgument ensureNestedMapParameter(subop::NestedMapOp nested,
                                                    tuples::ColumnRefAttr ref,
                                                    tuples::ColumnManager& cm) {
   if (auto idx = findNestedMapParameter(nested, ref, cm)) {
      assert(nested.getRegion().front().getNumArguments() > *idx + 1 &&
             "nested_map parameter must have a body argument");
      return nested.getRegion().front().getArgument(*idx + 1);
   }
   llvm::SmallVector<mlir::Attribute, 8> params(nested.getParameters().begin(),
                                                nested.getParameters().end());
   params.push_back(ref);
   nested.setParametersAttr(mlir::ArrayAttr::get(nested.getContext(), params));
   return nested.getRegion().front().addArgument(ref.getColumn().type, nested.getLoc());
}

static subop::NestedExecutionGroupOp firstNestedExecutionGroup(subop::NestedMapOp nested) {
   assert(!nested.getRegion().empty());
   for (mlir::Operation& op : nested.getRegion().front()) {
      if (auto neg = mlir::dyn_cast<subop::NestedExecutionGroupOp>(&op)) return neg;
   }
   llvm_unreachable("nested_map used by join build must contain nested_execution_group");
}

static mlir::BlockArgument ensureNestedExecutionGroupInput(subop::NestedExecutionGroupOp neg,
                                                           mlir::Value value) {
   for (unsigned i = 0; i < neg.getNumOperands(); ++i) {
      if (neg.getOperand(i) == value) {
         assert(i < neg.getSubOps().front().getNumArguments());
         return neg.getSubOps().front().getArgument(i);
      }
   }
   neg.getInputsMutable().append(value);
   return neg.getSubOps().front().addArgument(value.getType(), neg.getLoc());
}

static subop::ExecutionStepOp firstExecutionStep(subop::NestedExecutionGroupOp neg) {
   assert(!neg.getSubOps().empty());
   for (mlir::Operation& op : neg.getSubOps().front()) {
      if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(&op)) return step;
   }
   llvm_unreachable("nested_execution_group used by join build must contain execution_step");
}

static mlir::BlockArgument ensureExecutionStepInput(subop::ExecutionStepOp step,
                                                    mlir::Value value) {
   for (unsigned i = 0; i < step.getNumOperands(); ++i) {
      if (step.getOperand(i) == value) {
         assert(i < step.getSubOps().front().getNumArguments());
         return step.getSubOps().front().getArgument(i);
      }
   }
   step.getInputsMutable().append(value);
   llvm::SmallVector<mlir::Attribute, 8> threadLocal(step.getIsThreadLocal().begin(),
                                                     step.getIsThreadLocal().end());
   threadLocal.push_back(mlir::BoolAttr::get(step.getContext(), false));
   step.setIsThreadLocalAttr(mlir::ArrayAttr::get(step.getContext(), threadLocal));
   return step.getSubOps().front().addArgument(value.getType(), step.getLoc());
}

static void materializeColumnOnStreamBefore(mlir::Operation* anchor,
                                            mlir::Value& stream,
                                            tuples::ColumnDefAttr def,
                                            mlir::Value value) {
   assert(anchor && "join payload threading requires an anchor op");
   mlir::OpBuilder b(anchor);
   auto map = b.create<subop::MapOp>(anchor->getLoc(), tuples::TupleStreamType::get(anchor->getContext()),
                                     stream, b.getArrayAttr({def}), b.getArrayAttr({}));
   mlir::Block* block = new mlir::Block();
   map.getFn().push_back(block);
   mlir::OpBuilder rb(anchor->getContext());
   rb.setInsertionPointToStart(block);
   rb.create<tuples::ReturnOp>(anchor->getLoc(), mlir::ValueRange{value});
   stream = map.getResult();
}

static void threadTableRefToMaterializeStream(subop::MaterializeOp matOp,
                                              subop::ScanRefsOp scanOp,
                                              tuples::ColumnManager& cm) {
   subop::ExecutionStepOp targetStep = matOp->getParentOfType<subop::ExecutionStepOp>();
   assert(targetStep && "join payload materialize must live in an execution_step");
   (void)targetStep;

   tuples::ColumnDefAttr scanDef = scanOp.getRef();
   scanDef.getColumn().type = scanOp.getRef().getColumn().type;
   tuples::ColumnRefAttr scanRef = cm.createRef(&scanDef.getColumn());

   llvm::SmallVector<subop::NestedMapOp, 4> nestedMaps;
   for (mlir::Operation* parent = matOp->getParentOp(); parent; parent = parent->getParentOp()) {
      if (auto nested = mlir::dyn_cast<subop::NestedMapOp>(parent))
         nestedMaps.push_back(nested);
   }
   std::reverse(nestedMaps.begin(), nestedMaps.end());
   if (nestedMaps.empty()) return;

   mlir::Value currentValue;
   for (subop::NestedMapOp nested : nestedMaps) {
      if (currentValue && !findNestedMapParameter(nested, scanRef, cm)) {
         mlir::Value stream = nested.getStream();
         materializeColumnOnStreamBefore(nested.getOperation(), stream, scanDef, currentValue);
         nested->setOperand(0, stream);
      }

      mlir::BlockArgument nestedArg = ensureNestedMapParameter(nested, scanRef, cm);
      subop::NestedExecutionGroupOp neg = firstNestedExecutionGroup(nested);
      mlir::BlockArgument negArg = ensureNestedExecutionGroupInput(neg, nestedArg);
      subop::ExecutionStepOp innerStep = firstExecutionStep(neg);
      currentValue = ensureExecutionStepInput(innerStep, negArg);
   }

   if (!currentValue) return;
   mlir::Value stream = matOp.getStream();
   materializeColumnOnStreamBefore(matOp.getOperation(), stream, scanDef, currentValue);
   matOp->setOperand(0, stream);
}

static bool opIsNestedInside(mlir::Operation* maybeAncestor, mlir::Operation* op) {
   for (mlir::Operation* parent = op; parent; parent = parent->getParentOp()) {
      if (parent == maybeAncestor) return true;
   }
   return false;
}

static bool tupleStreamCarriesColumn(mlir::Value stream, tuples::ColumnRefAttr ref) {
   tuples::Column* target = &ref.getColumn();
   llvm::DenseSet<void*> seenStreams;
   llvm::SmallVector<mlir::Value, 8> worklist{stream};
   while (!worklist.empty()) {
      mlir::Value cur = worklist.pop_back_val();
      if (!cur || !mlir::isa<tuples::TupleStreamType>(cur.getType())) continue;
      if (!seenStreams.insert(cur.getAsOpaquePointer()).second) continue;
      mlir::Operation* def = cur.getDefiningOp();
      if (!def) continue;
      std::unordered_set<tuples::Column*> created =
         subop::ColumnCreationAnalysis::getCreatedColumnsForOp(def);
      if (created.contains(target)) return true;
      for (mlir::Value operand : def->getOperands()) {
         if (mlir::isa<tuples::TupleStreamType>(operand.getType())) worklist.push_back(operand);
      }
   }
   return false;
}

static tuples::ColumnRefAttr findColumnRefBySemanticOnStream(mlir::Value stream, const std::string& semantic,
                                                             tuples::ColumnManager& cm) {
   llvm::DenseSet<void*> seenStreams;
   llvm::SmallVector<mlir::Value, 8> worklist{stream};
   auto checkDef = [&](tuples::ColumnDefAttr def) -> tuples::ColumnRefAttr {
      auto [scope, leaf] = cm.getName(&def.getColumn());
      if (columnSemanticKey(scope, leaf) == semantic) return cm.createRef(&def.getColumn());
      return {};
   };
   while (!worklist.empty()) {
      mlir::Value cur = worklist.pop_back_val();
      if (!cur || !mlir::isa<tuples::TupleStreamType>(cur.getType())) continue;
      if (!seenStreams.insert(cur.getAsOpaquePointer()).second) continue;
      mlir::Operation* defOp = cur.getDefiningOp();
      if (!defOp) continue;
      if (auto gather = mlir::dyn_cast<subop::GatherOp>(defOp)) {
         for (auto& [member, def] : gather.getMapping().getMapping())
            if (tuples::ColumnRefAttr ref = checkDef(def)) return ref;
      } else if (auto map = mlir::dyn_cast<subop::MapOp>(defOp)) {
         for (auto attr : map.getComputedCols())
            if (tuples::ColumnRefAttr ref = checkDef(mlir::cast<tuples::ColumnDefAttr>(attr))) return ref;
      } else if (auto rename = mlir::dyn_cast<subop::RenamingOp>(defOp)) {
         for (auto attr : rename.getColumns())
            if (tuples::ColumnRefAttr ref = checkDef(mlir::cast<tuples::ColumnDefAttr>(attr))) return ref;
      }
      for (mlir::Value operand : defOp->getOperands()) {
         if (mlir::isa<tuples::TupleStreamType>(operand.getType())) worklist.push_back(operand);
      }
   }
   return {};
}

static tuples::ColumnRefAttr threadColumnRefFromSourceToMaterializeStream(subop::MaterializeOp matOp,
                                                                          mlir::Operation* sourceOp,
                                                                          tuples::ColumnDefAttr def,
                                                                          tuples::ColumnManager& cm) {
   tuples::ColumnRefAttr ref = cm.createRef(&def.getColumn());

   llvm::SmallVector<subop::NestedMapOp, 4> nestedMaps;
   for (mlir::Operation* parent = matOp->getParentOp(); parent; parent = parent->getParentOp()) {
      if (auto nested = mlir::dyn_cast<subop::NestedMapOp>(parent)) {
         if (!opIsNestedInside(nested.getOperation(), sourceOp)) nestedMaps.push_back(nested);
      }
   }
   std::reverse(nestedMaps.begin(), nestedMaps.end());

   mlir::Value currentValue;
   for (subop::NestedMapOp nested : nestedMaps) {
      if (!currentValue && !tupleStreamCarriesColumn(nested.getStream(), ref)) continue;
      if (currentValue && !findNestedMapParameter(nested, ref, cm)) {
         mlir::Value stream = nested.getStream();
         materializeColumnOnStreamBefore(nested.getOperation(), stream, def, currentValue);
         nested->setOperand(0, stream);
      }

      mlir::BlockArgument nestedArg = ensureNestedMapParameter(nested, ref, cm);
      subop::NestedExecutionGroupOp neg = firstNestedExecutionGroup(nested);
      mlir::BlockArgument negArg = ensureNestedExecutionGroupInput(neg, nestedArg);
      subop::ExecutionStepOp innerStep = firstExecutionStep(neg);
      currentValue = ensureExecutionStepInput(innerStep, negArg);
   }

   if (!currentValue) return ref;

   mlir::Value stream = matOp.getStream();
   materializeColumnOnStreamBefore(matOp.getOperation(), stream, def, currentValue);
   matOp->setOperand(0, stream);
   return ref;
}

static llvm::SmallVector<tuples::ColumnRefAttr, 8>
threadPredicateRefsToMaterializeStream(subop::MaterializeOp matOp,
                                       llvm::ArrayRef<tuples::ColumnRefAttr> predRefs,
                                       tuples::ColumnManager& cm) {
   llvm::SmallVector<tuples::ColumnRefAttr, 8> threaded;
   threaded.reserve(predRefs.size());
   for (tuples::ColumnRefAttr ref : predRefs) {
      tuples::ColumnDefAttr def = cm.createDef(&ref.getColumn());
      threaded.push_back(threadColumnRefFromSourceToMaterializeStream(matOp, /*sourceOp=*/nullptr, def, cm));
   }
   return threaded;
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
         bool relevantGather = lookupEntryRefStateHasLayout(r.getColumn().type, producerHiv);
         bool changed = syncLookupEntryRefToHiv(r);
         auto m = op.getMapping();
         llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> out;
         for (auto [mem, def] : m.getMapping()) {
            tuples::ColumnDefAttr d = def;
            if (lookupEntryRefStateHasLayout(d.getColumn().type, producerHiv)) relevantGather = true;
            if (syncLookupEntryDefToHiv(d)) changed = true;
            out.push_back({mem, d});
         }
         if (changed) {
            op.setRefAttr(r);
            op.setMappingAttr(subop::ColumnDefMemberMappingAttr::get(ctx, out));
         }
         if (relevantGather || changed) syncMapInputColsFromGather(op, cm);
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

static bool isJoinProbeCompilerScope(llvm::StringRef scope);

static void syncGatherMembersToMixedHivLayouts(mlir::ModuleOp module, const llvm::DenseSet<void*>* closureFilter) {
   auto* ctx = module.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();

   auto mixedStateFromGatherRef = [](tuples::ColumnRefAttr ref) -> subop::HashIndexedViewType {
      if (!ref) return nullptr;
      auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(ref.getColumn().type);
      if (!ler) return nullptr;
      return asHashIndexedViewLayoutType(ler.getState());
   };

   llvm::DenseMap<const void*, llvm::StringMap<subop::Member>> globalNewMemberBySemantic;
   module.walk([&](subop::GatherOp gather) {
      if (closureFilter && !opOperandsOrNestedBlockArgsTouchClosure(gather.getOperation(), *closureFilter)) return;
      auto mixed = mixedStateFromGatherRef(gather.getRef());
      if (!mixed) return;
      llvm::DenseSet<subop::Member> mixedMembers(mixed.getValueMembers().getMembers().begin(),
                                                 mixed.getValueMembers().getMembers().end());
      auto& bySemantic = globalNewMemberBySemantic[mixed.getAsOpaquePointer()];
      for (auto& [member, def] : gather.getMapping().getMapping()) {
         if (!mixedMembers.contains(member)) continue;
         if (parseFilterPredMemberSlot(mm.getName(member))) continue;
         auto [defScope, defLeaf] = cm.getName(&def.getColumn());
         if (isJoinProbeCompilerScope(defScope) && isPayloadMemberSlotName(defLeaf)) continue;
         bySemantic.try_emplace(columnSemanticKey(defScope, defLeaf), member);
      }
   });

   module.walk([&](subop::ExecutionStepOp step) {
      if (closureFilter && !opOperandsOrNestedBlockArgsTouchClosure(step.getOperation(), *closureFilter)) return;

      struct ScopeGatherLayout {
         subop::HashIndexedViewType mixed;
         llvm::SmallVector<subop::Member, 8> newNonPredMembers;
         llvm::DenseSet<subop::Member> mixedMembers;
         llvm::StringMap<subop::Member> newMemberBySemantic;
         llvm::DenseMap<subop::Member, std::string> oldMemberSemantic;
         llvm::DenseMap<subop::Member, unsigned> oldMemberPayloadOrdinal;
      };
      llvm::StringMap<ScopeGatherLayout> byScope;

      step.walk([&](subop::GatherOp gather) {
         if (closureFilter && !opOperandsOrNestedBlockArgsTouchClosure(gather.getOperation(), *closureFilter)) return;
         auto mixed = mixedStateFromGatherRef(gather.getRef());
         if (!mixed) return;
         auto [scope, leaf] = cm.getName(&gather.getRef().getColumn());
         (void)leaf;
         ScopeGatherLayout& layout = byScope[scope];
         if (!layout.mixed) {
            layout.mixed = mixed;
            for (subop::Member member : mixed.getValueMembers().getMembers()) {
               layout.mixedMembers.insert(member);
               if (parseFilterPredMemberSlot(mm.getName(member))) continue;
               layout.newNonPredMembers.push_back(member);
            }
            if (auto itGlobal = globalNewMemberBySemantic.find(mixed.getAsOpaquePointer());
                itGlobal != globalNewMemberBySemantic.end()) {
               for (const auto& kv : itGlobal->second)
                  layout.newMemberBySemantic.try_emplace(kv.getKey(), kv.getValue());
            }
         } else {
            assert(layout.mixed == mixed && "one lookup ref scope must carry a single mixed HIV layout");
         }

         for (auto& [member, def] : gather.getMapping().getMapping()) {
            if (parseFilterPredMemberSlot(mm.getName(member))) continue;
            auto [defScope, defLeaf] = cm.getName(&def.getColumn());
            if (isJoinProbeCompilerScope(defScope) && isPayloadMemberSlotName(defLeaf)) {
               if (!layout.mixedMembers.contains(member)) layout.oldMemberSemantic.try_emplace(member, "");
               continue;
            }
            std::string semantic = columnSemanticKey(defScope, defLeaf);
            if (layout.mixedMembers.contains(member)) {
               layout.newMemberBySemantic.try_emplace(semantic, member);
               continue;
            }
            auto [itOld, inserted] = layout.oldMemberSemantic.try_emplace(member, semantic);
            if (!inserted && itOld->second.empty()) itOld->second = semantic;
         }

         unsigned payloadIdx = 0;
         for (auto& [member, def] : gather.getMapping().getMapping()) {
            if (parseFilterPredMemberSlot(mm.getName(member))) continue;
            auto [defScope, defLeaf] = cm.getName(&def.getColumn());
            if (isJoinProbeCompilerScope(defScope) && isPayloadMemberSlotName(defLeaf)) {
               if (!layout.mixedMembers.contains(member)) {
                  layout.oldMemberSemantic.try_emplace(member, "");
                  layout.oldMemberPayloadOrdinal.try_emplace(member, payloadIdx);
               }
               ++payloadIdx;
               continue;
            }
            if (payloadIdx < layout.newNonPredMembers.size()) {
               subop::Member candidate = layout.newNonPredMembers[payloadIdx];
               if (mm.getType(candidate) == def.getColumn().type) {
                  layout.newMemberBySemantic.try_emplace(columnSemanticKey(defScope, defLeaf), candidate);
               }
            }
            ++payloadIdx;
         }
      });

      llvm::StringMap<llvm::DenseMap<subop::Member, subop::Member>> oldToNewByScope;
      for (auto& [scope, layout] : byScope) {
         llvm::DenseMap<subop::Member, subop::Member>& oldToNew = oldToNewByScope[scope];
         for (auto& [oldMember, semantic] : layout.oldMemberSemantic) {
            if (auto member = lookupMemberBySemanticOrUniqueLeaf(layout.newMemberBySemantic, semantic)) {
               oldToNew[oldMember] = *member;
            } else if (semantic.empty()) {
               auto itOrdinal = layout.oldMemberPayloadOrdinal.find(oldMember);
               if (itOrdinal == layout.oldMemberPayloadOrdinal.end()) continue;
               unsigned payloadIdx = itOrdinal->second;
               if (payloadIdx >= layout.newNonPredMembers.size()) continue;
               subop::Member candidate = layout.newNonPredMembers[payloadIdx];
               if (mm.getType(candidate) == mm.getType(oldMember)) oldToNew[oldMember] = candidate;
            }
         }
      }

      step.walk([&](subop::GatherOp gather) {
         if (closureFilter && !opOperandsOrNestedBlockArgsTouchClosure(gather.getOperation(), *closureFilter)) return;
         auto mixed = mixedStateFromGatherRef(gather.getRef());
         if (!mixed) return;
         auto [scope, leaf] = cm.getName(&gather.getRef().getColumn());
         (void)leaf;
         auto itMap = oldToNewByScope.find(scope);
         if (itMap == oldToNewByScope.end()) return;

         bool changed = false;
         llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> out;
         llvm::StringMap<tuples::ColumnRefAttr> oldDefRefsToNewDefs;
         out.reserve(gather.getMapping().getMapping().size());
         for (auto& [member, def] : gather.getMapping().getMapping()) {
            subop::Member outMember = member;
            tuples::ColumnDefAttr outDef = def;
            if (auto it = itMap->second.find(member); it != itMap->second.end()) {
               outMember = it->second;
               changed = true;
            }
            auto [defScope, defLeaf] = cm.getName(&outDef.getColumn());
            if (outMember != member && isJoinProbeCompilerScope(defScope) &&
                isPayloadMemberSlotName(defLeaf) && defLeaf != mm.getName(outMember)) {
               std::string oldDefKey = columnSemanticKey(defScope, defLeaf);
               outDef = cm.createDef(defScope, mm.getName(outMember));
               outDef.getColumn().type = mm.getType(outMember);
               oldDefRefsToNewDefs[oldDefKey] = cm.createRef(&outDef.getColumn());
               changed = true;
            }
            out.push_back({outMember, outDef});
         }
         if (!changed) return;
         gather.setMappingAttr(subop::ColumnDefMemberMappingAttr::get(ctx, out));
         syncMapInputColsFromGather(gather, cm, &oldDefRefsToNewDefs);
      });
   });
}

static void syncMixedLookupResultCarriers(mlir::ModuleOp module, const llvm::DenseSet<void*>* closureFilter) {
   auto* ctx = module.getContext();
   module.walk([&](subop::LookupOp lookup) {
      if (closureFilter && !opOperandsOrNestedBlockArgsTouchClosure(lookup.getOperation(), *closureFilter)) return;
      auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(lookup.getState().getType());
      if (!mixed) return;
      auto expectedListTy = subop::ListType::get(ctx, subop::LookupEntryRefType::get(ctx, mixed));
      auto ref = lookup.getRef();
      if (ref.getColumn().type != expectedListTy) {
         ref.getColumn().type = expectedListTy;
         lookup.setRefAttr(ref);
      }
      for (mlir::Operation* user : lookup.getResult().getUsers()) {
         auto nested = mlir::dyn_cast<subop::NestedMapOp>(user);
         if (!nested || nested.getRegion().empty()) continue;
         for (mlir::BlockArgument arg : nested.getRegion().front().getArguments()) {
            if (mlir::isa<subop::ListType>(arg.getType()) && arg.getType() != expectedListTy) {
               arg.setType(expectedListTy);
            }
         }
      }
   });
}

static void syncMixedProbeCarriers(mlir::ModuleOp module, const llvm::DenseSet<void*>* closureFilter) {
   syncMixedLookupResultCarriers(module, closureFilter);
   syncProbeListCarriersInClosure(module, closureFilter);
   syncGatherMembersToMixedHivLayouts(module, closureFilter);
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
   syncMixedProbeCarriers(module, &closureFilter);
}

static void patchBufferBuildStepForUnion(mlir::ModuleOp synthetic, subop::ExecutionStepOp buildStep,
                                         const JoinBufferUnionPlan& plan,
                                         const ModuleReuseInfo& reuseSynthetic,
                                         unsigned donorReuseSlot,
                                         llvm::ArrayRef<std::pair<mlir::ModuleOp, mlir::Value>> peerHivs,
                                         llvm::ArrayRef<const ModuleReuseInfo*> peerReuses,
                                         llvm::ArrayRef<unsigned> peerReuseSlots,
                                         llvm::DenseMap<const void*, uint64_t>& columnHashes) {
   assert(peerHivs.size() == peerReuses.size() && "join superset: peer HIVs and reuse metadata must align");
   assert(peerHivs.size() == peerReuseSlots.size() && "join superset: peer HIV slots must align");
   mlir::Block& body = buildStep.getSubOps().front();
   auto* ctx = synthetic.getContext();
   auto* subDialect = ctx->getLoadedDialect<subop::SubOperatorDialect>();
   auto* tupleDialect = ctx->getLoadedDialect<tuples::TupleStreamDialect>();
   auto& mm = subDialect->getMemberManager();
   auto& cm = tupleDialect->getColumnManager();

   subop::ScanRefsOp scanOp = findUniqueTableScanRefsInStep(buildStep);
   if (!scanOp) {
      bool scansNonTableState = false;
	      buildStep.walk([&](subop::ScanRefsOp scan) {
	         mlir::Value state = peelBlockArgsToEnclosingOperands(scan.getState());
	         if (!state) state = scan.getState();
	         if (!mlir::isa<subop::TableType, subop::SharedTableType>(state.getType()))
	            scansNonTableState = true;
	      });
      assert(scansNonTableState &&
             "join superset: payload union table rewrite requires a unique table scan source");
      return;
   }

   auto rewriteDonorExternalTableForUnion = [&]() {
      mlir::Value tableState = scanOp.getState();
      ExternalDatasourceProperty mergedDs;
      bool haveDs = false;
      llvm::StringRef donorTableName;
      subop::TableType donorTableTy;
      bool haveDonorExternal = resolveScannedTableExternal(buildStep, tableState, reuseSynthetic, donorTableName,
                                                           mergedDs, haveDs, donorTableTy) &&
                               haveDs;
      if (!haveDonorExternal) return;

      llvm::SmallVector<std::pair<unsigned, ExternalDatasourceProperty>, 4> filterSourcesBySlot;
      llvm::SmallVector<ExternalDatasourceProperty, 4> originalFilterSources;
      for (size_t pi = 0; pi < peerHivs.size(); ++pi) {
         auto [peerMod, peerHiv] = peerHivs[pi];
         if (!peerMod || !peerHiv) continue;
         const ModuleReuseInfo& reusePeer = *peerReuses[pi];
         subop::ExecutionStepOp peerBuild = findJoinBufferBuildStepForHiv(peerMod, peerHiv, reusePeer);
         if (!peerBuild) continue;
         if (auto resolved = resolveExternalTableScanForDonor(peerBuild, reusePeer, donorTableName)) {
            originalFilterSources.push_back(resolved->datasource);
            filterSourcesBySlot.push_back({peerReuseSlots[pi], std::move(resolved->datasource)});
         }
         mergePeerExternalFromBuildStepScan(mergedDs, haveDs, peerBuild, reusePeer, donorTableName);
      }
      assert(haveDs && "join superset: merged external datasource required for donor table");
      assert(!filterSourcesBySlot.empty() &&
             "join superset: merged external filters must come from explicit original query slots");
      mergeExternalFiltersForOrReuseBySlot(mergedDs, std::move(filterSourcesBySlot));

      llvm::StringMap<mlir::Type> filterOnlyTypeByIdentifier;
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
         if (auto it = filterOnlyTypeByIdentifier.find(normalizeColumnIdentifier(identifier));
             it != filterOnlyTypeByIdentifier.end())
            return it->second;
         return {};
      };
      addFilterOnlyColumnsToExternalMapping(mergedDs, ctx, lookupPeerColumnType,
                                            filterOnlyTypeByIdentifier, originalFilterSources);

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
      deriveNonNullColumnsFromWidenedNullableGathers(buildStep);
   };
   rewriteDonorExternalTableForUnion();

   subop::MaterializeOp matOp = findJoinBufferMaterializeInStep(buildStep);
   assert(matOp && "join superset: buffer build step must materialize into join buffer");
   auto materializeWritesPlanBuffer = [&](subop::MaterializeOp mat) {
      subop::BufferType bufTy = getInnerBufferTypeForMaterializeState(mat.getState().getType());
      if (!bufTy) return false;
      llvm::ArrayRef<subop::Member> members = bufTy.getMembers().getMembers();
      return llvm::is_contained(members, plan.linkMember) && llvm::is_contained(members, plan.hashMember);
   };
   buildStep.walk([&](subop::MaterializeOp mat) {
      if (materializeWritesPlanBuffer(mat))
         alignMaterializedPayloadMembersToUnionPlan(mat, plan, mm, columnHashes);
   });

   subop::MapOp hashMapOp = findJoinHashMapBeforeMaterialize(body, matOp);
   assert(hashMapOp);

   auto appendMaterializeMapping = [&](subop::Member bufMem, tuples::ColumnDefAttr colDef, uint64_t semanticHash) {
      tuples::ColumnRefAttr colRef = cm.createRef(&colDef.getColumn());
      llvm::SmallVector<subop::RefMappingPairT> matPairs;
      for (auto pr : matOp.getMapping().getMapping()) matPairs.push_back(pr);
      matPairs.push_back({bufMem, colRef});
      matOp.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, matPairs));
      columnHashes[&colDef.getColumn()] = semanticHash;
   };
   auto appendMaterializeMappingRef = [&](subop::Member bufMem, tuples::ColumnRefAttr colRef,
                                          uint64_t semanticHash) {
      llvm::SmallVector<subop::RefMappingPairT> matPairs;
      for (auto pr : matOp.getMapping().getMapping()) matPairs.push_back(pr);
      matPairs.push_back({bufMem, colRef});
      matOp.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, matPairs));
      columnHashes[&colRef.getColumn()] = semanticHash;
   };

   auto isPayloadMaterialized = [&](const PayloadColumnSpec& spec) {
      unsigned predIdx = 0;
      const bool isPred = parseFilterPredLayoutSemanticKey(spec.semanticKey, predIdx);
      for (auto& [member, colRef] : matOp.getMapping().getMapping()) {
         if (member == plan.linkMember || member == plan.hashMember) continue;
         if (colRef.getColumn().type != cloneTypeToContext(spec.colType, ctx)) continue;
         if (isPred) {
            if (parseFilterPredMemberSlot(mm.getName(member)) == predIdx) return true;
            continue;
         }
         if (payloadColumnIdentityHash(colRef, columnHashes) == spec.semanticHash) return true;
      }
      return false;
   };

   auto appendUnionPayloadGather = [&](const PayloadColumnSpec& spec, subop::Member bufMem) {
      assert(!spec.scope.empty() && "join superset: payload column scope required");

      subop::ScanRefsOp specScan = scanOp;
      mlir::Type tableTy = specScan.getState().getType();
      assert(tableDataMembersForType(tableTy) && "join superset: table-like type required for payload column gather");
      llvm::StringRef physicalLeaf = sourceColumnIdentifierForPayloadLeaf(spec.leaf);
      subop::Member tableMem = tableMemberForIdentifier(tableTy, mm, physicalLeaf);
      if (!tableMem) {
         if (tuples::ColumnRefAttr existing =
                findColumnRefBySemanticOnStream(matOp.getStream(), spec.semanticKey, cm)) {
            appendMaterializeMappingRef(bufMem, existing, spec.semanticHash);
            return;
         }
      }
      assert(tableMem && "join superset: widened external table must contain union payload source column");

      tuples::ColumnDefAttr gatheredDef = cm.createDef(spec.scope, physicalLeaf);
      gatheredDef.getColumn().type = cloneTypeToContext(mm.getType(tableMem), ctx);
      if (specScan->getBlock() != matOp->getBlock())
         threadTableRefToMaterializeStream(matOp, specScan, cm);

      // Chain union-only gathers on the current materialize stream (clone build chain + prior
      // union-only gathers), not from hashMapOp each time.
      mlir::Value mapStream = matOp.getStream();
      mlir::Operation* streamAnchor = mapStream.getDefiningOp();
      if (!streamAnchor) streamAnchor = hashMapOp;
      mlir::OpBuilder gb(streamAnchor);
      gb.setInsertionPointAfter(streamAnchor);
      auto mapping = subop::ColumnDefMemberMappingAttr::get(
         ctx, llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>>{{tableMem, gatheredDef}});
      auto gatherTy = mapStream.getType();
      auto refDef = specScan.getRef();
      auto gatherRef = cm.createRef(&refDef.getColumn());
      auto newGather =
         gb.create<subop::GatherOp>(mlir::UnknownLoc::get(ctx), gatherTy, mapStream, gatherRef, mapping);
      mlir::Value payloadStream = newGather.getRes();
      tuples::ColumnDefAttr payloadDef = gatheredDef;
      mlir::Type targetTy = cloneTypeToContext(spec.colType, ctx);
      if (gatheredDef.getColumn().type != targetTy) {
         auto nullableTy = mlir::dyn_cast<db::NullableType>(gatheredDef.getColumn().type);
         (void)nullableTy;
         assert(nullableTy && nullableTy.getType() == targetTy &&
                "join superset: union payload type mismatch must be nullable source to non-null target");
         payloadDef = cm.createDef(spec.scope, spec.leaf);
         payloadDef.getColumn().type = targetTy;
         tuples::ColumnRefAttr gatheredRef = cm.createRef(&gatheredDef.getColumn());
         MapCreationHelper helper(ctx);
         helper.buildBlock(gb, [&](mlir::OpBuilder& rb) {
            mlir::Value nullableValue = helper.access(gatheredRef, matOp.getLoc());
            mlir::Value value = rb.create<db::NullableGetVal>(matOp.getLoc(), targetTy, nullableValue);
            rb.create<tuples::ReturnOp>(matOp.getLoc(), mlir::ValueRange{value});
         });
         mlir::OpBuilder mb(newGather);
         mb.setInsertionPointAfter(newGather);
         auto mapOp = mb.create<subop::MapOp>(matOp.getLoc(), tuples::TupleStreamType::get(ctx),
                                              payloadStream, mb.getArrayAttr({payloadDef}),
                                              helper.getColRefs());
         mapOp.getFn().push_back(helper.getMapBlock());
         payloadStream = mapOp.getResult();
      }
      matOp->setOperand(0, payloadStream);
      appendMaterializeMapping(bufMem, payloadDef, spec.semanticHash);
      return;
   };

   auto appendSharedTablePredicateGather = [&](unsigned predIdx, subop::Member bufMem, uint64_t semanticHash) {
      subop::ScanRefsOp specScan = scanOp;
      auto sharedTy = mlir::dyn_cast<subop::SharedTableType>(specScan.getState().getType());
      if (!sharedTy) return false;
      std::string predMemberName = ("filter_pred$" + llvm::Twine(predIdx)).str();
      subop::Member predMember;
      for (subop::Member m : sharedTy.getPredicateMembers().getMembers()) {
         if (mm.getName(m) == predMemberName) {
            predMember = m;
            break;
         }
      }
      if (!predMember) return false;

      tuples::ColumnDefAttr predDef = cm.createDef(predMemberName, "filter_pred");
      predDef.getColumn().type = cloneTypeToContext(mm.getType(predMember), ctx);
      if (specScan->getBlock() != matOp->getBlock())
         threadTableRefToMaterializeStream(matOp, specScan, cm);

      mlir::Value mapStream = matOp.getStream();
      mlir::Operation* streamAnchor = mapStream.getDefiningOp();
      if (!streamAnchor) streamAnchor = hashMapOp;
      mlir::OpBuilder gb(streamAnchor);
      gb.setInsertionPointAfter(streamAnchor);
      auto mapping = subop::ColumnDefMemberMappingAttr::get(
         ctx, llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>>{{predMember, predDef}});
      auto gatherRef = cm.createRef(&specScan.getRef().getColumn());
      auto newGather =
         gb.create<subop::GatherOp>(mlir::UnknownLoc::get(ctx), mapStream.getType(), mapStream, gatherRef, mapping);
      matOp->setOperand(0, newGather.getRes());
      appendMaterializeMapping(bufMem, predDef, semanticHash);
      return true;
   };

   for (size_t i = 0; i < plan.payloadColumns.size(); ++i) {
      const PayloadColumnSpec& spec = plan.payloadColumns[i];
      unsigned reusePredQueryIdx = 0;
      if (parseFilterPredLayoutSemanticKey(spec.semanticKey, reusePredQueryIdx)) {
         if (isPayloadMaterialized(spec)) continue;
         assert(i < plan.payloadMembers.size() && "join superset: payload slot out of range");
         if (appendSharedTablePredicateGather(reusePredQueryIdx, plan.payloadMembers[i], spec.semanticHash))
            continue;
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
                                         unsigned donorReuseSlot,
                                         mlir::ModuleOp query0Module, mlir::Value hivA, const ModuleReuseInfo& reuseA,
                                         unsigned firstPeerReuseSlot,
                                         mlir::ModuleOp query1Module, mlir::Value hivB,
                                         const ModuleReuseInfo& reuseB,
                                         llvm::ArrayRef<std::pair<mlir::ModuleOp, mlir::Value>> extraPeerHivs = {},
                                         llvm::ArrayRef<const ModuleReuseInfo*> extraPeerReuses = {},
                                         llvm::ArrayRef<unsigned> extraPeerReuseSlots = {}) {
   assert(extraPeerHivs.size() == extraPeerReuses.size() &&
          "join superset: extra peer HIVs and reuse metadata must align");
   assert(extraPeerHivs.size() == extraPeerReuseSlots.size() &&
          "join superset: extra peer HIV slots must align");
   auto* ctx = synthetic.getContext();
   auto* subDialect = ctx->getLoadedDialect<subop::SubOperatorDialect>();
   auto& mm = subDialect->getMemberManager();
   SyntheticJoinBuildSite site = findSyntheticJoinBuildSite(synthetic, syntheticHiv, reuseSynthetic);
   assert(site.materialize && "join superset: synthetic build step must materialize into merged join buffer");
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   llvm::DenseMap<const void*, uint64_t> syntheticColumnHashes =
      collectStateConstructionColumnHashes(synthetic, syntheticHiv);
   assignPayloadMembersForPlan(mm, cm, site.materialize, plan, syntheticColumnHashes);
   llvm::DenseMap<uint64_t, subop::Member> targetMemberByColumnHash;
   assert(plan.payloadColumns.size() == plan.payloadMembers.size());
   for (size_t i = 0; i < plan.payloadColumns.size(); ++i)
      targetMemberByColumnHash[plan.payloadColumns[i].semanticHash] = plan.payloadMembers[i];
   auto targetMembers = bufferMembersForPlan(ctx, plan);
   llvm::SmallVector<mlir::Value, 4> roots = {site.mergedBuffer};

   applyBufferLayoutToSsaClosure(synthetic, roots, targetMembers, reuseSynthetic, &syntheticColumnHashes,
                                 &targetMemberByColumnHash);

   if (site.buildStep) {
      llvm::SmallVector<std::pair<mlir::ModuleOp, mlir::Value>, 8> peerHivs = {
         {query1Module, hivB},
         {query0Module, hivA},
      };
      llvm::SmallVector<const ModuleReuseInfo*, 8> peerReuses = {&reuseB, &reuseA};
      llvm::SmallVector<unsigned, 8> peerReuseSlots = {firstPeerReuseSlot, donorReuseSlot};
      peerHivs.append(extraPeerHivs.begin(), extraPeerHivs.end());
      peerReuses.append(extraPeerReuses.begin(), extraPeerReuses.end());
      peerReuseSlots.append(extraPeerReuseSlots.begin(), extraPeerReuseSlots.end());
      patchBufferBuildStepForUnion(synthetic, site.buildStep, plan, reuseSynthetic, donorReuseSlot,
                                   peerHivs, peerReuses, peerReuseSlots, syntheticColumnHashes);
   }

   applyBufferLayoutToSsaClosure(synthetic, roots, targetMembers, reuseSynthetic, &syntheticColumnHashes,
                                 &targetMemberByColumnHash);
   syncMaterializeMappingsToBufferMembers(synthetic, targetMembers, nullptr, &syntheticColumnHashes,
                                          &targetMemberByColumnHash);
   auto realignBuildStepMaterializes = [&]() {
      alignMaterializedPayloadMembersToUnionPlan(site.materialize, plan, mm, syntheticColumnHashes);
   };
   realignBuildStepMaterializes();
   syncMaterializeMappingsToActualBufferTypes(synthetic);
   realignBuildStepMaterializes();
   finalizeSyntheticJoinProducerClosure(synthetic, roots, plan, reuseSynthetic);
   realignBuildStepMaterializes();
}

static bool typeEmbedsHashIndexedView(mlir::Type t) {
   if (mlir::isa<subop::HashIndexedViewType, subop::MixedHashIndexedViewType>(t)) return true;
   if (auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(t)) return !!ler.getState();
   if (auto list = mlir::dyn_cast<subop::ListType>(t)) return typeEmbedsHashIndexedView(list.getT());
   return false;
}

static bool lookupEntryRefEmbedsHashIndexedView(subop::LookupEntryRefType ler) {
   return !!asHashIndexedViewLayoutType(ler.getState());
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

static std::optional<subop::Member> payloadOrdinalRemapMember(subop::HashIndexedViewType oldHiv,
                                                              subop::HashIndexedViewType alignedHiv,
                                                              subop::Member member) {
   if (!oldHiv || !alignedHiv || oldHiv == alignedHiv) return std::nullopt;
   if (!sameHashIndexedViewJoinKey(oldHiv, alignedHiv)) return std::nullopt;
   llvm::ArrayRef<subop::Member> oldVals = oldHiv.getValueMembers().getMembers();
   llvm::ArrayRef<subop::Member> alignedVals = alignedHiv.getValueMembers().getMembers();
   if (oldVals.size() != alignedVals.size()) return std::nullopt;
   auto it = llvm::find(oldVals, member);
   if (it == oldVals.end()) return std::nullopt;
   size_t idx = static_cast<size_t>(std::distance(oldVals.begin(), it));
   auto& mm = oldHiv.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   if (mm.getType(oldVals[idx]) != mm.getType(alignedVals[idx])) return std::nullopt;
   return alignedVals[idx];
}

static std::optional<subop::LookupOp> findUniqueUpstreamLookupOpInReuse(mlir::Value value,
                                                                        bool& multiple);

static std::optional<std::string> semanticForColumnOrSingleInputMap(tuples::ColumnRefAttr ref,
                                                                    mlir::ModuleOp module) {
   if (!ref) return std::nullopt;
   auto& cm = module.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   std::optional<std::string> fromProducer;
   module.walk([&](subop::MapOp map) {
      if (fromProducer) return;
      bool computesRef = false;
      for (mlir::Attribute attr : map.getComputedCols()) {
         auto def = mlir::dyn_cast<tuples::ColumnDefAttr>(attr);
         if (def && &def.getColumn() == &ref.getColumn()) {
            computesRef = true;
            break;
         }
      }
      if (!computesRef || map.getInputCols().size() != 1) return;
      auto input = mlir::dyn_cast<tuples::ColumnRefAttr>(*map.getInputCols().begin());
      if (!input) return;
      auto [scope, leaf] = cm.getName(&input.getColumn());
      fromProducer = columnSemanticKey(scope, leaf);
   });
   if (fromProducer) return fromProducer;
   auto [scope, leaf] = cm.getName(&ref.getColumn());
   return columnSemanticKey(scope, leaf);
}

static std::optional<subop::Member> payloadMemberForHashKeySource(subop::Member keyMember,
                                                                  subop::HashIndexedViewType alignedHiv,
                                                                  mlir::ModuleOp module) {
   auto& cm = module.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   llvm::DenseSet<subop::Member> valueMembers;
   for (subop::Member member : alignedHiv.getValueMembers().getMembers()) valueMembers.insert(member);

   std::optional<subop::Member> found;
   module.walk([&](subop::MaterializeOp mat) {
      if (found) return;
      tuples::ColumnRefAttr keyRef;
      for (auto [member, ref] : mat.getMapping().getMapping()) {
         if (member == keyMember) {
            keyRef = ref;
            break;
         }
      }
      if (!keyRef) return;
      std::optional<std::string> keySourceSemantic = semanticForColumnOrSingleInputMap(keyRef, module);
      if (!keySourceSemantic) return;
      for (auto [member, ref] : mat.getMapping().getMapping()) {
         if (!valueMembers.contains(member)) continue;
         auto [scope, leaf] = cm.getName(&ref.getColumn());
         if (columnSemanticKey(scope, leaf) != *keySourceSemantic) continue;
         if (found && *found != member)
            llvm::report_fatal_error("HIV key source semantic must map to one payload member");
         found = member;
      }
   });
   return found;
}

static llvm::StringMap<subop::Member> lookupKeySemanticAliasesForScanList(subop::ScanListOp scanList,
                                                                          subop::HashIndexedViewType alignedHiv) {
   llvm::StringMap<subop::Member> aliases;
   bool multiple = false;
   std::optional<subop::LookupOp> lookup = findUniqueUpstreamLookupOpInReuse(scanList.getList(), multiple);
   if (!lookup || multiple) return aliases;
   auto lookupHiv = asHashIndexedViewLayoutType(lookup->getState().getType());
   if (!lookupHiv || !sameHashIndexedViewJoinKey(lookupHiv, alignedHiv)) return aliases;
   mlir::ModuleOp module = scanList->getParentOfType<mlir::ModuleOp>();
   llvm::ArrayRef<subop::Member> keyMembers = alignedHiv.getKeyMembers().getMembers();
   if (lookup->getKeys().size() != keyMembers.size()) return aliases;
   for (auto [idx, attr] : llvm::enumerate(lookup->getKeys())) {
      auto keyRef = mlir::dyn_cast<tuples::ColumnRefAttr>(attr);
      if (!keyRef) continue;
      std::optional<std::string> lookupKeySemantic = semanticForColumnOrSingleInputMap(keyRef, module);
      if (!lookupKeySemantic) continue;
      std::optional<subop::Member> payloadMember =
         payloadMemberForHashKeySource(keyMembers[idx], alignedHiv, module);
      if (!payloadMember) continue;
      aliases[*lookupKeySemantic] = *payloadMember;
   }
   return aliases;
}

static bool typeEmbedsHashIndexedViewState(mlir::Type t, subop::HashIndexedViewType hiv) {
   if (!t || !hiv) return false;
   if (auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(t)) {
      if (!lookupEntryRefEmbedsHashIndexedView(ler)) return false;
      auto st = asHashIndexedViewLayoutType(ler.getState());
      return st && sameHashIndexedViewLayout(st, hiv);
   }
   if (auto list = mlir::dyn_cast<subop::ListType>(t)) {
      return typeEmbedsHashIndexedViewState(list.getT(), hiv);
   }
   return false;
}

static mlir::Type replaceEmbeddedHivStateInType(mlir::MLIRContext* ctx, mlir::Type t,
                                                mlir::Type replacementHiv,
                                                subop::HashIndexedViewType consumerHivBeforeAlign) {
   if (!t) return t;
   auto replacementLayout = asHashIndexedViewLayoutType(replacementHiv);
   assert(replacementLayout && "replacement must be an HIV-like state");
   if (asHashIndexedViewLayoutType(t)) {
      if (t == replacementHiv) return t;
      auto currentLayout = asHashIndexedViewLayoutType(t);
      if (sameHashIndexedViewLayout(currentLayout, replacementLayout)) return replacementHiv;
      if (consumerHivBeforeAlign && !sameHashIndexedViewLayout(currentLayout, consumerHivBeforeAlign)) return t;
      return replacementHiv;
   }
   if (auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(t)) {
      if (!lookupEntryRefEmbedsHashIndexedView(ler)) return t;
      if (ler.getState() == replacementHiv) return t;
      auto currentLayout = asHashIndexedViewLayoutType(ler.getState());
      if (sameHashIndexedViewLayout(currentLayout, replacementLayout))
         return subop::LookupEntryRefType::get(ctx, mlir::cast<subop::LookupAbleState>(replacementHiv));
      if (consumerHivBeforeAlign && !sameHashIndexedViewLayout(currentLayout, consumerHivBeforeAlign)) return t;
      return subop::LookupEntryRefType::get(ctx, mlir::cast<subop::LookupAbleState>(replacementHiv));
   }
   if (auto list = mlir::dyn_cast<subop::ListType>(t)) {
      mlir::Type nt = replaceEmbeddedHivStateInType(ctx, list.getT(), replacementHiv, consumerHivBeforeAlign);
      if (nt == list.getT()) return t;
      return subop::ListType::get(ctx, mlir::cast<subop::StateEntryReference>(nt));
   }
   return t;
}

static mlir::Type replaceEmbeddedHivInType(mlir::MLIRContext* ctx, mlir::Type t,
                                           subop::HashIndexedViewType producerHiv,
                                           subop::HashIndexedViewType consumerHivBeforeAlign) {
   return replaceEmbeddedHivStateInType(ctx, t, producerHiv, consumerHivBeforeAlign);
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

static void setValueCarrierHivStateType(mlir::Value v, mlir::Type replacementHiv,
                                        subop::HashIndexedViewType consumerHivBeforeAlign,
                                        const ProbeAlignDebugCtx* dbg = nullptr) {
   mlir::MLIRContext* ctx = v.getContext();
   mlir::Type oldTy = v.getType();
   if (mlir::Type nt = replaceEmbeddedHivStateInType(ctx, oldTy, replacementHiv, consumerHivBeforeAlign);
       nt != oldTy) {
      debugProbeAlign(dbg, [&](llvm::raw_ostream& os) {
         os << "setValueCarrierHivStateType value=" << v << " old_type=" << mlirTypeToString(oldTy)
            << " new_type=" << mlirTypeToString(nt) << " replacement_hiv=" << mlirTypeToString(replacementHiv);
      });
      v.setType(nt);
   }
}

static mlir::Type alignedHivStatePreservingMixedPred(mlir::MLIRContext* ctx, mlir::Type carrier,
                                                     subop::HashIndexedViewType alignedHiv) {
   auto stateWithMaybeMixedPred = [](mlir::Type type) -> mlir::Type {
      if (auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(type)) return mixed;
      if (auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(type)) return ler.getState();
      if (auto list = mlir::dyn_cast<subop::ListType>(type)) {
         if (auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(list.getT())) return ler.getState();
      }
      return {};
   };
   auto mixed = mlir::dyn_cast_or_null<subop::MixedHashIndexedViewType>(stateWithMaybeMixedPred(carrier));
   if (!mixed) return alignedHiv;
   return subop::MixedHashIndexedViewType::get(ctx, alignedHiv.getKeyMembers(), alignedHiv.getValueMembers(),
                                               alignedHiv.getCompareHashForLookup(),
                                               mixed.getFilterPredMemberName());
}

static subop::LookupEntryRefType alignedEntryRefPreservingMixedPred(mlir::MLIRContext* ctx,
                                                                    mlir::Type carrier,
                                                                    subop::HashIndexedViewType alignedHiv) {
   return subop::LookupEntryRefType::get(
      ctx, mlir::cast<subop::LookupAbleState>(alignedHivStatePreservingMixedPred(ctx, carrier, alignedHiv)));
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

template <typename LayoutT>
static unsigned appendPayloadLayoutSlot(LayoutT& layout, llvm::StringRef semanticKey,
                                        subop::Member member, mlir::Type type) {
   unsigned idx = static_cast<unsigned>(layout.payloadMembers.size());
   layout.payloadSemanticKeys.push_back(semanticKey.str());
   layout.payloadMembers.push_back(member);
   layout.payloadColumnTypes.push_back(type);
   return idx;
}

static CachedJoinBufferLayout layoutFromUnionPlan(subop::HashIndexedViewType producerHiv,
                                                  const JoinBufferUnionPlan& plan) {
   CachedJoinBufferLayout out;
   out.producerHiv = producerHiv;
   llvm::DenseMap<subop::Member, unsigned> planIndexByMember;
   for (auto [idx, member] : llvm::enumerate(plan.payloadMembers)) {
      planIndexByMember[member] = static_cast<unsigned>(idx);
   }
   out.payloadMembers.reserve(plan.payloadMembers.size());
   out.payloadColumnTypes.reserve(plan.payloadMembers.size());
   out.payloadSemanticKeys.reserve(plan.payloadMembers.size());
   for (subop::Member producerMember : producerHiv.getValueMembers().getMembers()) {
      auto itPlan = planIndexByMember.find(producerMember);
      if (itPlan == planIndexByMember.end()) continue;
      unsigned planIdx = itPlan->second;
      const PayloadColumnSpec& spec = plan.payloadColumns[planIdx];
      unsigned unionIdx = appendPayloadLayoutSlot(out, spec.semanticKey, producerMember,
                                                  plan.payloadMemberTypes[planIdx]);
      for (const auto& sourceMember : spec.sourceMemberByQueryIndex) {
         if (sourceMember.second != producerMember)
            out.probeGatherMemberRemap[sourceMember.second] = producerMember;
      }
      for (unsigned queryIndex : spec.queryIndices) {
         out.querySemanticKeysById[queryIndex].push_back(spec.semanticKey);
         out.querySlotInUnionById[queryIndex].push_back(unionIdx);
      }
      if (llvm::is_contained(spec.queryIndices, 0)) {
         out.query0SemanticKeys.push_back(spec.semanticKey);
         out.query0SlotInUnion.push_back(unionIdx);
      }
      if (llvm::is_contained(spec.queryIndices, 1)) {
         out.query1SemanticKeys.push_back(spec.semanticKey);
         out.query1SlotInUnion.push_back(unionIdx);
      }
   }
   assert(out.payloadMembers.size() == plan.payloadMembers.size() &&
          "cached join layout must cover all planned payload members");
   return out;
}

static CachedJoinBufferLayout layoutInProducerValueOrder(const CachedJoinBufferLayout& layout,
                                                         subop::HashIndexedViewType producerHiv) {
   CachedJoinBufferLayout out = layout;
   out.producerHiv = producerHiv;
   out.payloadMembers.clear();
   out.payloadColumnTypes.clear();
   out.payloadSemanticKeys.clear();
   out.query0SemanticKeys.clear();
   out.query1SemanticKeys.clear();
   out.querySemanticKeysById.clear();
   out.query0SlotInUnion.clear();
   out.query1SlotInUnion.clear();
   out.querySlotInUnionById.clear();

   llvm::DenseMap<subop::Member, unsigned> oldIdxByMember;
   llvm::StringMap<unsigned> oldIdxByMemberName;
   auto& mm = producerHiv.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   for (auto [idx, member] : llvm::enumerate(layout.payloadMembers)) {
      oldIdxByMember[member] = static_cast<unsigned>(idx);
      oldIdxByMemberName[mm.getName(member)] = static_cast<unsigned>(idx);
   }
   auto queryOwnsSemantic = [&](unsigned query, llvm::StringRef semantic) {
      if (auto it = layout.querySemanticKeysById.find(query); it != layout.querySemanticKeysById.end())
         return llvm::is_contained(it->second, semantic);
      if (query == 0) return llvm::is_contained(layout.query0SemanticKeys, semantic);
      if (query == 1) return llvm::is_contained(layout.query1SemanticKeys, semantic);
      return false;
   };
   llvm::SmallVector<unsigned, 8> knownQueries;
   for (const auto& kv : layout.querySemanticKeysById) knownQueries.push_back(kv.first);
   if (!llvm::is_contained(knownQueries, 0)) knownQueries.push_back(0);
   if (!llvm::is_contained(knownQueries, 1)) knownQueries.push_back(1);
   llvm::sort(knownQueries);
   knownQueries.erase(std::unique(knownQueries.begin(), knownQueries.end()), knownQueries.end());

   for (auto [producerIdx, producerMember] : llvm::enumerate(producerHiv.getValueMembers().getMembers())) {
      auto itOld = oldIdxByMember.find(producerMember);
      std::optional<unsigned> oldIdxOpt;
      if (itOld != oldIdxByMember.end()) {
         oldIdxOpt = itOld->second;
      } else if (auto itName = oldIdxByMemberName.find(mm.getName(producerMember));
                 itName != oldIdxByMemberName.end()) {
         oldIdxOpt = itName->second;
      } else if (producerIdx < layout.payloadMembers.size()) {
         oldIdxOpt = static_cast<unsigned>(producerIdx);
      }
      if (!oldIdxOpt) continue;
      unsigned oldIdx = *oldIdxOpt;
      llvm::StringRef semantic = layout.payloadSemanticKeys[oldIdx];
      unsigned newIdx = appendPayloadLayoutSlot(out, semantic, producerMember,
                                                layout.payloadColumnTypes[oldIdx]);
      for (unsigned query : knownQueries) {
         if (!queryOwnsSemantic(query, semantic)) continue;
         out.querySemanticKeysById[query].push_back(std::string(semantic));
         out.querySlotInUnionById[query].push_back(newIdx);
         if (query == 0) {
            out.query0SemanticKeys.push_back(std::string(semantic));
            out.query0SlotInUnion.push_back(newIdx);
         } else if (query == 1) {
            out.query1SemanticKeys.push_back(std::string(semantic));
            out.query1SlotInUnion.push_back(newIdx);
         }
      }
   }
   assert(out.payloadMembers.size() == layout.payloadMembers.size() &&
          "cached join layout must cover all producer value members");
   return out;
}

static CachedJoinBufferLayout layoutFromMaterializeMapping(subop::HashIndexedViewType producerHiv,
                                                           subop::MaterializeOp mat,
                                                           const CachedJoinBufferLayout* previousLayout) {
   auto* ctx = mat.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

   llvm::DenseMap<subop::Member, tuples::ColumnRefAttr> colByMember;
   for (auto& [member, colRef] : mat.getMapping().getMapping()) colByMember[member] = colRef;

   llvm::DenseMap<subop::Member, std::string> oldSemanticByMember;
   if (previousLayout) {
      for (size_t i = 0; i < previousLayout->payloadMembers.size() &&
                         i < previousLayout->payloadSemanticKeys.size(); ++i) {
         oldSemanticByMember[previousLayout->payloadMembers[i]] = previousLayout->payloadSemanticKeys[i];
      }
   }
   auto queryOwnsSemantic = [&](unsigned query, llvm::StringRef semantic) {
      if (!previousLayout) return false;
      if (auto it = previousLayout->querySemanticKeysById.find(query);
          it != previousLayout->querySemanticKeysById.end()) {
         return llvm::is_contained(it->second, semantic);
      }
      if (query == 0) return llvm::is_contained(previousLayout->query0SemanticKeys, semantic);
      if (query == 1) return llvm::is_contained(previousLayout->query1SemanticKeys, semantic);
      return false;
   };
   llvm::SmallVector<unsigned, 8> knownQueries;
   if (previousLayout) {
      for (const auto& kv : previousLayout->querySemanticKeysById) knownQueries.push_back(kv.first);
   }
   if (!llvm::is_contained(knownQueries, 0)) knownQueries.push_back(0);
   if (!llvm::is_contained(knownQueries, 1)) knownQueries.push_back(1);
   llvm::sort(knownQueries);
   knownQueries.erase(std::unique(knownQueries.begin(), knownQueries.end()), knownQueries.end());

   CachedJoinBufferLayout out;
   out.producerHiv = producerHiv;
   for (subop::Member member : producerHiv.getValueMembers().getMembers()) {
      std::string semantic;
      if (auto predSlot = parseFilterPredMemberSlot(mm.getName(member))) {
         if (auto itOld = oldSemanticByMember.find(member); itOld != oldSemanticByMember.end()) {
            semantic = itOld->second;
         } else {
            semantic = reuseFilterPredSemanticKey(*predSlot);
         }
      } else {
         auto itCol = colByMember.find(member);
         if (itCol != colByMember.end()) {
            auto [scope, leaf] = cm.getName(&itCol->second.getColumn());
            semantic = columnSemanticKey(scope, leaf);
            if (isJoinProbeCompilerScope(scope) && isPayloadMemberSlotName(leaf)) {
               if (auto itOld = oldSemanticByMember.find(member); itOld != oldSemanticByMember.end())
                  semantic = itOld->second;
            }
         } else {
            auto itOld = oldSemanticByMember.find(member);
            assert(itOld != oldSemanticByMember.end() && "join layout refresh needs materialized payload member");
            semantic = itOld->second;
         }
      }
      unsigned newIdx = appendPayloadLayoutSlot(out, semantic, member, mm.getType(member));
      for (unsigned query : knownQueries) {
         if (!queryOwnsSemantic(query, semantic)) continue;
         out.querySemanticKeysById[query].push_back(semantic);
         out.querySlotInUnionById[query].push_back(newIdx);
         if (query == 0) {
            out.query0SemanticKeys.push_back(semantic);
            out.query0SlotInUnion.push_back(newIdx);
         } else if (query == 1) {
            out.query1SemanticKeys.push_back(semantic);
            out.query1SlotInUnion.push_back(newIdx);
         }
      }
   }
   if (previousLayout) out.probeGatherMemberRemap = previousLayout->probeGatherMemberRemap;
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

static llvm::StringMap<subop::Member> semKeyToAlignedHivMemberMap(const CachedJoinBufferLayout& layout,
                                                                  subop::HashIndexedViewType alignedHiv) {
   llvm::StringMap<subop::Member> semKeyToMember = semKeyToMemberMap(layout);
   auto alignedMembers = alignedHiv.getValueMembers().getMembers();
   assert(layout.payloadSemanticKeys.size() <= alignedMembers.size() &&
          "aligned HIV must contain every cached payload semantic slot");
   for (size_t i = 0; i < layout.payloadSemanticKeys.size(); ++i) {
      semKeyToMember[layout.payloadSemanticKeys[i]] = alignedMembers[i];
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

enum class AggregatePayloadKind { Identity, Count, Sum, Min, Max, Payload };

static AggregatePayloadKind aggregatePayloadKindForSemantic(llvm::StringRef semanticKey) {
   if (semanticKey == "identity") return AggregatePayloadKind::Identity;
   if (semanticKey == "count:*") return AggregatePayloadKind::Count;
   if (semanticKey.starts_with("sum:")) return AggregatePayloadKind::Sum;
   if (semanticKey.starts_with("min:")) return AggregatePayloadKind::Min;
   if (semanticKey.starts_with("max:")) return AggregatePayloadKind::Max;
   return AggregatePayloadKind::Payload;
}

static std::optional<AggregatePayloadKind>
classifyMinMaxAggregateSelect(mlir::Value returned, mlir::Value current, mlir::Value input) {
   auto select = mlir::dyn_cast_or_null<mlir::arith::SelectOp>(returned.getDefiningOp());
   if (!select) return std::nullopt;
   auto cmp = mlir::dyn_cast_or_null<db::CmpOp>(
      stripCastLikeForAggregateSemantic(select.getCondition()).getDefiningOp());
   if (!cmp) return std::nullopt;

   mlir::Value trueValue = stripCastLikeForAggregateSemantic(select.getTrueValue());
   mlir::Value falseValue = stripCastLikeForAggregateSemantic(select.getFalseValue());
   mlir::Value lhs = stripCastLikeForAggregateSemantic(cmp.getLeft());
   mlir::Value rhs = stripCastLikeForAggregateSemantic(cmp.getRight());
   bool trueIsInput = trueValue == input;
   bool falseIsCurrent = falseValue == current;
   bool trueIsCurrent = trueValue == current;
   bool falseIsInput = falseValue == input;
   if (!((trueIsInput && falseIsCurrent) || (trueIsCurrent && falseIsInput))) return std::nullopt;

   enum class CmpShape { CurrentLessInput, CurrentGreaterInput, InputLessCurrent, InputGreaterCurrent };
   std::optional<CmpShape> shape;
   using P = db::DBCmpPredicate;
   if (lhs == current && rhs == input) {
      if (cmp.getPredicate() == P::lt || cmp.getPredicate() == P::lte) shape = CmpShape::CurrentLessInput;
      if (cmp.getPredicate() == P::gt || cmp.getPredicate() == P::gte) shape = CmpShape::CurrentGreaterInput;
   } else if (lhs == input && rhs == current) {
      if (cmp.getPredicate() == P::lt || cmp.getPredicate() == P::lte) shape = CmpShape::InputLessCurrent;
      if (cmp.getPredicate() == P::gt || cmp.getPredicate() == P::gte) shape = CmpShape::InputGreaterCurrent;
   }
   if (!shape) return std::nullopt;

   if (trueIsInput) {
      if (*shape == CmpShape::CurrentGreaterInput || *shape == CmpShape::InputLessCurrent)
         return AggregatePayloadKind::Min;
      if (*shape == CmpShape::CurrentLessInput || *shape == CmpShape::InputGreaterCurrent)
         return AggregatePayloadKind::Max;
   } else {
      if (*shape == CmpShape::CurrentLessInput || *shape == CmpShape::InputGreaterCurrent)
         return AggregatePayloadKind::Min;
      if (*shape == CmpShape::CurrentGreaterInput || *shape == CmpShape::InputLessCurrent)
         return AggregatePayloadKind::Max;
   }
   return std::nullopt;
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
      mlir::Value input = block.getArgument(*inputIdx);
      if (aggregateExprContainsValue(returned, current)) {
         if (auto kind = classifyMinMaxAggregateSelect(returned, current, input)) {
            if (*kind == AggregatePayloadKind::Min) return "min:" + aggregateColumnSemanticKey(col, cm);
            if (*kind == AggregatePayloadKind::Max) return "max:" + aggregateColumnSemanticKey(col, cm);
         }
         abortAggregateUnionUnsupported("unsupported aggregate payload update expression using current value");
      }
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
   else if (semanticKey.consume_front("min:")) kind = "min";
   else if (semanticKey.consume_front("max:")) kind = "max";
   assert(sourceColumn && "aggregate payload must have a source column identity hash");
   uint64_t h = hashPayloadString("aggregate_payload");
   h = combinePayloadHash(h, hashPayloadString(kind));
   h = combinePayloadHash(h, hashPayloadString(semanticKey));
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

static subop::ExecutionStepOp tryFindAggregateBuildStepForHt(mlir::Value aggregateState,
                                                             const ModuleReuseInfo& reuse) {
   llvm::SmallVector<subop::ExecutionStepOp, 2> candidates;
   auto considerState = [&](mlir::Value state) {
      auto it = reuse.writerStepsByState.find(state);
      if (it == reuse.writerStepsByState.end()) return;
      for (subop::ExecutionStepOp step : it->second) {
         subop::ReduceOp reduce;
         step.walk([&](subop::ReduceOp op) {
            assert(!reduce && "aggregate union: expected a single reduce in aggregate build step");
            reduce = op;
         });
         if (reduce && reduce->getParentOfType<subop::ExecutionStepOp>() == step) candidates.push_back(step);
      }
   };
   considerState(aggregateState);
   forEachShadowChainPredecessor(aggregateState, reuse, considerState);
   if (candidates.size() != 1) return {};
   return candidates.front();
}

static subop::ExecutionStepOp findAggregateBuildStepForHt(mlir::Value aggregateState,
                                                          const ModuleReuseInfo& reuse) {
   subop::ExecutionStepOp step = tryFindAggregateBuildStepForHt(aggregateState, reuse);
   assert(step && "aggregate union: expected one reduce build step for aggregate state");
   return step;
}

static llvm::SmallVector<AggregatePayloadMemberInfo, 16>
collectAggregatePayloadMembers(mlir::ModuleOp module, mlir::Value aggregateState,
                               subop::ExecutionStepOp buildStep) {
   auto* ctx = module.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   llvm::DenseMap<const void*, uint64_t> columnHashes =
      collectStateConstructionColumnHashes(module, aggregateState);
   subop::PreAggrHtType ht = aggregateHtTypeFromState(aggregateState);
   assert(ht && "aggregate payload collection requires optimistic_ht-like state");

   llvm::SmallVector<subop::ReduceOp, 4> reduceOps;
   buildStep.walk([&](subop::ReduceOp reduce) {
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

static llvm::SmallVector<AggregatePayloadMemberInfo, 16>
collectAggregatePayloadMembers(mlir::ModuleOp module, mlir::Value aggregateState,
                               const ModuleReuseInfo& reuse) {
   return collectAggregatePayloadMembers(module, aggregateState,
                                         findAggregateBuildStepForHt(aggregateState, reuse));
}

static void recordAggregateConsumerPayloadLayout(CachedAggregateLayout& layout, unsigned queryId,
                                                 mlir::ModuleOp module, mlir::Value state,
                                                 const ModuleReuseInfo& reuse) {
   llvm::SmallVector<AggregatePayloadMemberInfo, 16> infos =
      collectAggregatePayloadMembers(module, state, reuse);
   auto& keys = layout.querySemanticKeysById[queryId];
   auto& members = layout.queryMembersById[queryId];
   keys.clear();
   members.clear();
   for (const auto& p : infos) {
      keys.push_back(p.semanticKey);
      members.push_back(p.member);
   }
}

static subop::ExecutionStepOp findAggregateMergeStepForHt(mlir::Value aggregateState,
                                                          const ModuleReuseInfo& reuse) {
   auto it = reuse.writerStepsByState.find(aggregateState);
   assert(it != reuse.writerStepsByState.end() && "aggregate union: optimistic_ht must have writer step");
   subop::ExecutionStepOp found;
   for (subop::ExecutionStepOp step : it->second) {
      subop::MergeOp merge;
      step.walk([&](subop::MergeOp op) {
         assert(!merge && "aggregate union: expected a single merge in aggregate writer step");
         merge = op;
      });
      if (!merge) continue;
      assert(!found && "aggregate union: expected a single optimistic_ht merge writer");
      found = step;
   }
   assert(found && "aggregate union: expected merge writer for optimistic_ht");
   return found;
}

static subop::MergeOp findAggregateMergeInStep(subop::ExecutionStepOp step) {
   subop::MergeOp found;
   step.walk([&](subop::MergeOp merge) {
      assert(!found && "aggregate union: expected a single merge in aggregate merge step");
      found = merge;
   });
   assert(found && "aggregate union: aggregate merge step must contain merge");
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

static subop::ScanRefsOp findAggregateSharedTableScanInStep(subop::ExecutionStepOp step) {
   subop::ScanRefsOp found;
   step.walk([&](subop::ScanRefsOp scan) {
      if (!mlir::isa<subop::SharedTableType>(scan.getState().getType())) return;
      assert(!found && "aggregate union: expected a single shared table scan in aggregate build step");
      found = scan;
   });
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

struct AggregatePeerRewriteInput {
   unsigned queryId = 0;
   unsigned filterSlot = 0;
   mlir::ModuleOp module;
   mlir::Value state;
   subop::ExecutionStepOp buildStep;
   ModuleReuseInfo reuse;
};

struct AggregateUnionGroupInput {
   const CrossQueryStateMatchEntry* entry = nullptr;
   unsigned queryId = 0;
   unsigned filterSlot = 0;
   mlir::ModuleOp module;
   mlir::Value state;
   subop::ExecutionStepOp buildStep;
   ModuleReuseInfo reuse;
};

static llvm::SmallVector<AggregateUnionGroupInput, 8>
collectAggregateUnionGroupInputs(const CrossQueryStateMatchGroup& group,
                                 llvm::ArrayRef<mlir::ModuleOp> queries) {
   llvm::SmallVector<AggregateUnionGroupInput, 8> inputs;
   for (const CrossQueryStateMatchEntry& entry : group.entries) {
      if (entry.query < 0 || static_cast<size_t>(entry.query) >= queries.size()) continue;
      if (!entry.state) continue;
      assert(mlir::isa<subop::PreAggrHtType>(entry.state.getType()) &&
             "aggregate union group must contain optimistic_ht states");
      ModuleReuseInfo reuse = collectModuleReuseInfo(queries[entry.query]);
      subop::ExecutionStepOp buildStep = tryFindAggregateBuildStepForHt(entry.state, reuse);
      if (!buildStep) return {};
      inputs.push_back(AggregateUnionGroupInput{&entry, static_cast<unsigned>(entry.query),
                                                aggregateReuseSlotForEntry(entry), queries[entry.query],
                                                entry.state, buildStep, std::move(reuse)});
   }
   if (inputs.size() < 2) return {};
   return inputs;
}

static const AggregateUnionGroupInput*
selectAggregateUnionDonor(llvm::ArrayRef<AggregateUnionGroupInput> inputs) {
   const AggregateUnionGroupInput* donor = nullptr;
   for (const AggregateUnionGroupInput& input : inputs) {
      if (!donor || input.queryId < donor->queryId) donor = &input;
   }
   return donor;
}

static AggregatePeerRewriteInput
aggregatePeerRewriteInputFromGroupInput(const AggregateUnionGroupInput& input) {
   AggregatePeerRewriteInput peer;
   peer.queryId = input.queryId;
   peer.filterSlot = input.filterSlot;
   peer.module = input.module;
   peer.state = input.state;
   peer.buildStep = input.buildStep;
   peer.reuse = input.reuse;
   return peer;
}

struct AggregateUnionRewritePlan {
   CachedAggregateLayout layout;
   llvm::SmallVector<AggregatePeerRewriteInput, 8> peers;
   ModuleReuseInfo reuseSynthetic;
   unsigned donorFilterSlot = 0;
};

static std::optional<AggregateUnionRewritePlan>
buildAggregateUnionRewritePlan(mlir::ModuleOp synthetic, mlir::Value syntheticState,
                               const CrossQueryStateMatchGroup& group,
                               llvm::ArrayRef<AggregateUnionGroupInput> groupInputs) {
   const AggregateUnionGroupInput* donor = selectAggregateUnionDonor(groupInputs);
   if (!donor) return std::nullopt;

   AggregateUnionRewritePlan plan;
   plan.layout.producerHt = mlir::cast<subop::PreAggrHtType>(syntheticState.getType());
   plan.layout.mixedByQueryId = group.enableFilterPredReuse;
   plan.reuseSynthetic = collectModuleReuseInfo(synthetic);
   plan.donorFilterSlot = donor->filterSlot;

   recordAggregateConsumerPayloadLayout(plan.layout, donor->queryId, donor->module, donor->state,
                                        donor->reuse);
   for (const auto& p : collectAggregatePayloadMembers(synthetic, syntheticState, plan.reuseSynthetic)) {
      appendPayloadLayoutSlot(plan.layout, p.semanticKey, p.member, p.type);
   }

   for (const AggregateUnionGroupInput& input : groupInputs) {
      if (&input == donor) continue;
      AggregatePeerRewriteInput peer = aggregatePeerRewriteInputFromGroupInput(input);
      recordAggregateConsumerPayloadLayout(plan.layout, peer.queryId, peer.module, peer.state, peer.reuse);
      plan.peers.push_back(std::move(peer));
   }
   assert(!plan.peers.empty() && "aggregate union group must have peer entries");
   return plan;
}

static void widenAggregateExternalTableForPeers(mlir::ModuleOp synthetic, subop::ExecutionStepOp buildStep,
                                                subop::ScanRefsOp scanOp,
                                                llvm::ArrayRef<AggregatePeerRewriteInput> peers,
                                                const ModuleReuseInfo& reuseSynthetic,
                                                bool mergeFiltersForOrReuse = false);

static void widenAggregateExternalTableForPeer(mlir::ModuleOp synthetic, subop::ExecutionStepOp buildStep,
                                               subop::ScanRefsOp scanOp, subop::ExecutionStepOp peerBuild,
                                               const ModuleReuseInfo& reuseSynthetic,
                                               const ModuleReuseInfo& reusePeer,
                                               bool mergeFiltersForOrReuse = false) {
   AggregatePeerRewriteInput peer;
   peer.buildStep = peerBuild;
   peer.reuse = reusePeer;
   llvm::SmallVector<AggregatePeerRewriteInput, 1> peers;
   peers.push_back(peer);
   widenAggregateExternalTableForPeers(synthetic, buildStep, scanOp, peers, reuseSynthetic,
                                       mergeFiltersForOrReuse);
}

static void widenAggregateExternalTableForPeers(mlir::ModuleOp synthetic, subop::ExecutionStepOp buildStep,
                                                subop::ScanRefsOp scanOp,
                                                llvm::ArrayRef<AggregatePeerRewriteInput> peers,
                                                const ModuleReuseInfo& reuseSynthetic,
                                                bool mergeFiltersForOrReuse) {
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
   llvm::SmallVector<ExternalDatasourceProperty, 2> filterSources;
   if (mergeFiltersForOrReuse) filterSources.push_back(mergedDs);
   if (mergeFiltersForOrReuse) {
      for (const AggregatePeerRewriteInput& peer : peers) {
         auto resolved = resolveExternalTableScanForDonor(peer.buildStep, peer.reuse, donorTableName);
         assert(resolved && "aggregate mixed reuse: peer build must scan donor table");
         filterSources.push_back(resolved->datasource);
      }
   }
   for (const AggregatePeerRewriteInput& peer : peers)
      mergePeerExternalFromBuildStepScan(mergedDs, haveDs, peer.buildStep, peer.reuse, donorTableName);
   if (mergeFiltersForOrReuse) mergeExternalFiltersForOrReuse(mergedDs, filterSources);
   llvm::StringMap<mlir::Type> filterOnlyTypeByIdentifier;
   auto lookupExistingPeerColumnType = [&](llvm::StringRef identifier) -> mlir::Type {
      for (const AggregatePeerRewriteInput& peer : peers) {
         if (mlir::Type ty = columnTypeForIdentifierFromPeerBuildScan(peer.buildStep, peer.reuse,
                                                                      donorTableName, identifier))
            return ty;
      }
      return {};
   };
   auto lookupPeerColumnType = [&](llvm::StringRef identifier) -> mlir::Type {
      if (mlir::Type ty = lookupExistingPeerColumnType(identifier)) return ty;
      if (auto it = filterOnlyTypeByIdentifier.find(normalizeColumnIdentifier(identifier));
          it != filterOnlyTypeByIdentifier.end())
         return it->second;
      llvm_unreachable("peer scanned table must contain union payload column");
   };
   if (mergeFiltersForOrReuse)
      addFilterOnlyColumnsToExternalMapping(mergedDs, ctx, lookupExistingPeerColumnType,
                                            filterOnlyTypeByIdentifier, filterSources);
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
   deriveNonNullColumnsFromWidenedNullableGathers(buildStep);
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

static subop::Member makeOrGetAggregateQueryIdMember(mlir::MLIRContext* ctx) {
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   return mm.getOrCreateMemberDirect("reuse_query_id$0", mlir::IntegerType::get(ctx, 64),
                                     /*allowTypeUpdate=*/false);
}

static mlir::Value createZeroForType(mlir::OpBuilder& b, mlir::Location loc, mlir::Type ty) {
   return b.create<db::ConstantOp>(loc, ty, b.getI64IntegerAttr(0));
}

static mlir::Value createOneForType(mlir::OpBuilder& b, mlir::Location loc, mlir::Type ty) {
   return b.create<db::ConstantOp>(loc, ty, b.getI64IntegerAttr(1));
}

static mlir::Value createInitialForAggregatePayload(mlir::OpBuilder& b, mlir::Location loc,
                                                    mlir::Type ty, AggregatePayloadKind kind) {
   if (kind == AggregatePayloadKind::Min) {
      auto intTy = mlir::dyn_cast<mlir::IntegerType>(ty);
      if (!intTy) abortAggregateUnionUnsupported("min aggregate payload initial value requires integer type");
      return b.create<db::ConstantOp>(loc, ty,
                                      b.getIntegerAttr(ty, llvm::APInt::getSignedMaxValue(intTy.getWidth())));
   }
   return createZeroForType(b, loc, ty);
}

static mlir::Value createAggregatePayloadUpdate(mlir::OpBuilder& b, mlir::Location loc,
                                                AggregatePayloadKind kind, mlir::Value current,
                                                mlir::Value input, mlir::Type ty) {
   switch (kind) {
      case AggregatePayloadKind::Count:
         return b.create<db::AddOp>(loc, current, createOneForType(b, loc, ty));
      case AggregatePayloadKind::Sum:
      case AggregatePayloadKind::Payload:
         return b.create<db::AddOp>(loc, current, input);
      case AggregatePayloadKind::Min: {
         mlir::Value useInput = b.create<db::CmpOp>(loc, db::DBCmpPredicate::gt, current, input);
         return b.create<mlir::arith::SelectOp>(loc, useInput, input, current);
      }
      case AggregatePayloadKind::Max: {
         mlir::Value useInput = b.create<db::CmpOp>(loc, db::DBCmpPredicate::lt, current, input);
         return b.create<mlir::arith::SelectOp>(loc, useInput, input, current);
      }
      case AggregatePayloadKind::Identity:
         abortAggregateUnionUnsupported("cannot insert identity aggregate payload update");
   }
   llvm_unreachable("unknown aggregate payload kind");
}

static mlir::Value createAggregatePayloadCombine(mlir::OpBuilder& b, mlir::Location loc,
                                                 AggregatePayloadKind kind, mlir::Value left,
                                                 mlir::Value right) {
   switch (kind) {
      case AggregatePayloadKind::Count:
      case AggregatePayloadKind::Sum:
      case AggregatePayloadKind::Payload:
         return b.create<db::AddOp>(loc, left, right);
      case AggregatePayloadKind::Min: {
         mlir::Value keepLeft = b.create<db::CmpOp>(loc, db::DBCmpPredicate::lt, left, right);
         return b.create<mlir::arith::SelectOp>(loc, keepLeft, left, right);
      }
      case AggregatePayloadKind::Max: {
         mlir::Value keepRight = b.create<db::CmpOp>(loc, db::DBCmpPredicate::lt, left, right);
         return b.create<mlir::arith::SelectOp>(loc, keepRight, right, left);
      }
      case AggregatePayloadKind::Identity:
         abortAggregateUnionUnsupported("cannot insert identity aggregate payload combine");
   }
   llvm_unreachable("unknown aggregate payload kind");
}

static void insertReturnOperand(tuples::ReturnOp ret, unsigned idx, mlir::Value v) {
   mlir::OpBuilder b(ret);
   llvm::SmallVector<mlir::Value> operands(ret->getOperands().begin(), ret->getOperands().end());
   operands.insert(operands.begin() + idx, v);
   auto newRet = b.create<tuples::ReturnOp>(ret.getLoc(), operands);
   ret.erase();
   (void)newRet;
}

static void insertAggregateLookupInitial(subop::LookupOrInsertOp lookup, unsigned idx,
                                         mlir::Type ty, AggregatePayloadKind kind) {
   mlir::Block& block = lookup.getInitFn().front();
   auto ret = mlir::cast<tuples::ReturnOp>(block.getTerminator());
   mlir::OpBuilder b(ret);
   insertReturnOperand(ret, idx, createInitialForAggregatePayload(b, lookup.getLoc(), ty, kind));
}

static void insertAggregateReduceUpdate(subop::ReduceOp reduce, unsigned idx, tuples::ColumnRefAttr sourceCol,
                                        subop::Member member, mlir::Type ty,
                                        AggregatePayloadKind kind) {
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
   mlir::Value updated = createAggregatePayloadUpdate(b, loc, kind, current, input, ty);
   insertReturnOperand(ret, idx, updated);
}

static void insertPairwiseAggregateRegionResult(mlir::Region& region, unsigned idx, mlir::Type ty,
                                                mlir::Location loc, AggregatePayloadKind kind) {
   mlir::Block& block = region.front();
   auto ret = mlir::cast<tuples::ReturnOp>(block.getTerminator());
   unsigned oldNumValues = ret.getNumOperands();
   mlir::BlockArgument left = block.insertArgument(idx, ty, loc);
   mlir::BlockArgument right = block.insertArgument(oldNumValues + 1 + idx, ty, loc);
   mlir::OpBuilder b(ret);
   insertReturnOperand(ret, idx, createAggregatePayloadCombine(b, loc, kind, left, right));
}

static void insertAggregateReduceCombine(subop::ReduceOp reduce, unsigned idx,
                                         mlir::Type ty, AggregatePayloadKind kind) {
   insertPairwiseAggregateRegionResult(reduce.getCombine(), idx, ty, reduce.getLoc(), kind);
}

static void insertAggregateMergeCombine(subop::MergeOp merge, unsigned idx,
                                        mlir::Type ty, AggregatePayloadKind kind) {
   insertPairwiseAggregateRegionResult(merge.getCombineFn(), idx, ty, merge.getLoc(), kind);
}

static void appendAggregateMergeEqKey(subop::MergeOp merge, unsigned oldKeyCount, mlir::Type keyTy) {
   mlir::Block& eqBlock = merge.getEqFn().front();
   assert(eqBlock.getNumArguments() == oldKeyCount * 2 &&
          "aggregate merge equality must contain paired key arguments before query_id append");
   mlir::Location loc = merge.getLoc();
   mlir::BlockArgument storedKey = eqBlock.insertArgument(oldKeyCount, keyTy, loc);
   mlir::BlockArgument peerKey = eqBlock.insertArgument(oldKeyCount * 2 + 1, keyTy, loc);
   auto ret = mlir::cast<tuples::ReturnOp>(eqBlock.getTerminator());
   assert(ret.getNumOperands() == 1 && "aggregate merge equality must return one predicate");
   mlir::OpBuilder b(ret);
   mlir::Value keyMatches =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, storedKey, peerKey);
   mlir::Value combined = b.create<mlir::arith::AndIOp>(loc, ret.getOperand(0), keyMatches);
   b.create<tuples::ReturnOp>(loc, mlir::ValueRange{combined});
   ret.erase();
}

static void updateSyntheticAggregateStateTypes(mlir::ModuleOp synthetic, const llvm::DenseSet<void*>& closure,
                                               subop::PreAggrHtFragmentType oldFrag,
                                               subop::PreAggrHtFragmentType newFrag, subop::PreAggrHtType oldHt,
                                               subop::PreAggrHtType newHt) {
   auto oldTl = subop::ThreadLocalType::get(synthetic.getContext(), oldFrag);
   auto newTl = subop::ThreadLocalType::get(synthetic.getContext(), newFrag);
   auto update = [&](mlir::Value v) {
      if (!opaqueClosureContains(closure, v)) return;
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
   synchronizeExecutionStepPortTypes(synthetic, &closure);
}

static void syncAggregateEntryRefAttrs(mlir::ModuleOp module, const llvm::DenseSet<void*>& closure,
                                       subop::PreAggrHtFragmentType oldFrag,
                                       subop::PreAggrHtFragmentType newFrag, subop::PreAggrHtType oldHt,
                                       subop::PreAggrHtType newHt) {
   auto* ctx = module.getContext();
   auto newLookupRef = subop::LookupEntryRefType::get(ctx, newFrag);
   auto oldPreAggrRef = subop::PreAggrHTEntryRefType::get(ctx, oldHt);
   auto newPreAggrRef = subop::PreAggrHTEntryRefType::get(ctx, newHt);
   module.walk([&](subop::LookupOrInsertOp op) {
      if (!opOperandsOrNestedBlockArgsTouchClosure(op.getOperation(), closure)) return;
      auto ref = op.getRef();
      if (ref.getColumn().type == subop::LookupEntryRefType::get(ctx, oldFrag) ||
          ref.getColumn().type == newLookupRef) {
         ref.getColumn().type = newLookupRef;
         op.setRefAttr(ref);
      }
   });
   module.walk([&](subop::ReduceOp op) {
      if (!opOperandsOrNestedBlockArgsTouchClosure(op.getOperation(), closure)) return;
      auto ref = op.getRef();
      if (ref.getColumn().type == subop::LookupEntryRefType::get(ctx, oldFrag) ||
          ref.getColumn().type == newLookupRef) {
         ref.getColumn().type = newLookupRef;
         op.setRefAttr(ref);
      }
   });
   module.walk([&](subop::ScanRefsOp op) {
      if (!opaqueClosureContains(closure, op.getState())) return;
      auto ref = op.getRef();
      if (ref.getColumn().type == oldPreAggrRef || ref.getColumn().type == newPreAggrRef) {
         ref.getColumn().type = newPreAggrRef;
         op.setRefAttr(ref);
      }
   });
}

static tuples::ColumnRefAttr insertSyntheticAggregateQueryIdColumn(
   subop::ExecutionStepOp buildStep, llvm::ArrayRef<RuntimeFilterIdClause> clauses, unsigned defaultId) {
   assert(!clauses.empty() && "aggregate mixed reuse requires query id filter clauses");
   if (subop::ScanRefsOp sharedScan = findAggregateSharedTableScanInStep(buildStep)) {
      auto* ctx = buildStep.getContext();
      auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
      auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      auto shared = mlir::cast<subop::SharedTableType>(sharedScan.getState().getType());

      llvm::DenseMap<unsigned, subop::Member> predMemberBySlot;
      for (subop::Member member : shared.getPredicateMembers().getMembers()) {
         if (std::optional<unsigned> slot = parseFilterPredMemberSlot(mm.getName(member)))
            predMemberBySlot[*slot] = member;
      }

      auto predicateSlotForClause = [&](unsigned clauseIdx) {
         for (unsigned prev = 0; prev < clauseIdx; ++prev) {
            if (filterClauseEquals(clauses[prev].filters, clauses[clauseIdx].filters)) return prev;
         }
         return clauseIdx;
      };

      subop::LookupOrInsertOp lookup = findAggregateLookupInStep(buildStep);
      mlir::OpBuilder b(lookup);
      mlir::Location loc = lookup.getLoc();
      mlir::Value stream = lookup.getStream();
      llvm::SmallVector<tuples::ColumnRefAttr, 8> predRefs;
      for (unsigned i = 0; i < clauses.size(); ++i) {
         unsigned predSlot = predicateSlotForClause(i);
         auto it = predMemberBySlot.find(predSlot);
         assert(it != predMemberBySlot.end() && "aggregate union: widened shared table missing filter_pred slot");
         tuples::ColumnDefAttr predDef =
            cm.createDef(cm.getUniqueScope("agg_reuse_shared_pred$" + llvm::Twine(predSlot).str()), "filter_pred");
         predDef.getColumn().type = mlir::IntegerType::get(ctx, 1);
         auto gather = b.create<subop::GatherOp>(
            loc, stream.getType(), stream, cm.createRef(&sharedScan.getRef().getColumn()),
            subop::ColumnDefMemberMappingAttr::get(ctx, {{it->second, predDef}}));
         stream = gather.getRes();
         predRefs.push_back(cm.createRef(&predDef.getColumn()));
      }

      tuples::ColumnDefAttr qidDef = cm.createDef(cm.getUniqueScope("agg_reuse_query_id"), "query_id");
      qidDef.getColumn().type = mlir::IntegerType::get(ctx, 64);
      tuples::ColumnRefAttr qidRef = cm.createRef(&qidDef.getColumn());
      llvm::SmallVector<mlir::Attribute, 8> inputAttrs;
      for (tuples::ColumnRefAttr predRef : predRefs) inputAttrs.push_back(predRef);
      auto map = b.create<subop::MapOp>(loc, tuples::TupleStreamType::get(ctx), stream,
                                        b.getArrayAttr({qidDef}), b.getArrayAttr(inputAttrs));
      mlir::Block* block = new mlir::Block();
      for (tuples::ColumnRefAttr predRef : predRefs) block->addArgument(predRef.getColumn().type, loc);
      map.getFn().push_back(block);

      mlir::OpBuilder rb(ctx);
      rb.setInsertionPointToStart(block);
      mlir::Value qid = rb.create<db::ConstantOp>(loc, qidDef.getColumn().type,
                                                  rb.getI64IntegerAttr(static_cast<int64_t>(defaultId)));
      for (unsigned i = 0; i < clauses.size(); ++i) {
         mlir::Value slotId = rb.create<db::ConstantOp>(loc, qidDef.getColumn().type,
                                                        rb.getI64IntegerAttr(static_cast<int64_t>(clauses[i].id)));
         qid = rb.create<mlir::arith::SelectOp>(loc, block->getArgument(i), slotId, qid);
      }
      rb.create<tuples::ReturnOp>(loc, mlir::ValueRange{qid});
      lookup->setOperand(0, map.getResult());
      return qidRef;
   }

   subop::ScanRefsOp scan = findAggregateTableScanInStep(buildStep);
   auto [qidStream, qidRef] = materializeRuntimeFilterClausesAsIdColumnAfterScanRefs(
      scan, clauses, defaultId, "query_id", /*rewireDownstreamUses=*/true);
   (void)qidStream;
   return qidRef;
}

static std::optional<tuples::ColumnRefAttr> insertSyntheticAggregateQueryIdFromMixedPreds(
   subop::ExecutionStepOp buildStep, llvm::ArrayRef<unsigned> filterSlots, unsigned defaultId) {
   if (filterSlots.empty()) return std::nullopt;
   subop::LookupOrInsertOp lookup = findAggregateLookupInStep(buildStep);
   if (!lookup) return std::nullopt;

   auto* ctx = buildStep.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

   llvm::SmallVector<unsigned, 8> slots(filterSlots.begin(), filterSlots.end());
   llvm::sort(slots);
   slots.erase(std::unique(slots.begin(), slots.end()), slots.end());

   struct PredSource {
      tuples::ColumnRefAttr sourceRef;
      subop::Member member;
   };
   llvm::DenseMap<unsigned, llvm::SmallVector<PredSource, 2>> sourcesBySlot;
   buildStep.walk([&](subop::ScanListOp scan) {
      auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(scan.getElem().getColumn().type);
      if (!ler) return;
      auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(ler.getState());
      if (!mixed) return;
      for (subop::Member member : mixed.getValueMembers().getMembers()) {
         std::optional<unsigned> slot = parseFilterPredMemberSlot(mm.getName(member));
         if (!slot || !llvm::is_contained(slots, *slot)) continue;
         sourcesBySlot[*slot].push_back(PredSource{cm.createRef(&scan.getElem().getColumn()), member});
      }
   });
   buildStep.walk([&](subop::ScanRefsOp scan) {
      auto shared = mlir::dyn_cast<subop::SharedTableType>(scan.getState().getType());
      if (!shared) return;
      for (subop::Member member : shared.getPredicateMembers().getMembers()) {
         std::optional<unsigned> slot = parseFilterPredMemberSlot(mm.getName(member));
         if (!slot || !llvm::is_contained(slots, *slot)) continue;
         sourcesBySlot[*slot].push_back(PredSource{cm.createRef(&scan.getRef().getColumn()), member});
      }
   });
   for (unsigned slot : slots) {
      if (sourcesBySlot[slot].empty()) return std::nullopt;
   }

   mlir::OpBuilder b(lookup);
   mlir::Location loc = lookup.getLoc();
   mlir::Value stream = lookup.getStream();
   llvm::DenseMap<unsigned, llvm::SmallVector<tuples::ColumnRefAttr, 2>> predRefsBySlot;
   for (unsigned slot : slots) {
      for (PredSource source : sourcesBySlot[slot]) {
         tuples::ColumnDefAttr predDef =
            cm.createDef(cm.getUniqueScope("agg_reuse_mixed_pred$" + llvm::Twine(slot).str()), "filter_pred");
         predDef.getColumn().type = mlir::IntegerType::get(ctx, 1);
         auto gather = b.create<subop::GatherOp>(
            loc, stream.getType(), stream, source.sourceRef,
            subop::ColumnDefMemberMappingAttr::get(ctx, {{source.member, predDef}}));
         stream = gather.getRes();
         predRefsBySlot[slot].push_back(cm.createRef(&predDef.getColumn()));
      }
   }

   tuples::ColumnDefAttr qidDef = cm.createDef(cm.getUniqueScope("agg_reuse_query_id"), "query_id");
   qidDef.getColumn().type = mlir::IntegerType::get(ctx, 64);
   tuples::ColumnRefAttr qidRef = cm.createRef(&qidDef.getColumn());
   llvm::SmallVector<mlir::Attribute, 8> inputAttrs;
   for (unsigned slot : slots) {
      for (tuples::ColumnRefAttr predRef : predRefsBySlot[slot]) inputAttrs.push_back(predRef);
   }
   auto map = b.create<subop::MapOp>(loc, tuples::TupleStreamType::get(ctx), stream,
                                     b.getArrayAttr({qidDef}), b.getArrayAttr(inputAttrs));
   mlir::Block* block = new mlir::Block();
   for (mlir::Attribute attr : inputAttrs) {
      auto ref = mlir::cast<tuples::ColumnRefAttr>(attr);
      block->addArgument(ref.getColumn().type, loc);
   }
   map.getFn().push_back(block);

   mlir::OpBuilder rb(ctx);
   rb.setInsertionPointToStart(block);
   mlir::Value qid = rb.create<db::ConstantOp>(loc, qidDef.getColumn().type,
                                               rb.getI64IntegerAttr(static_cast<int64_t>(defaultId)));
   unsigned argIdx = 0;
   for (unsigned slot : slots) {
      mlir::Value pred;
      for (size_t i = 0, e = predRefsBySlot[slot].size(); i < e; ++i) {
         mlir::Value next = block->getArgument(argIdx++);
         pred = pred ? rb.create<mlir::arith::AndIOp>(loc, pred, next).getResult() : next;
      }
      mlir::Value slotId = rb.create<db::ConstantOp>(loc, qidDef.getColumn().type,
                                                     rb.getI64IntegerAttr(static_cast<int64_t>(slot)));
      qid = rb.create<mlir::arith::SelectOp>(loc, pred, slotId, qid);
   }
   rb.create<tuples::ReturnOp>(loc, mlir::ValueRange{qid});

   lookup->setOperand(0, map.getResult());
   return qidRef;
}

static void appendAggregateLookupKey(subop::LookupOrInsertOp lookup, tuples::ColumnRefAttr key) {
   llvm::SmallVector<mlir::Attribute> keys(lookup.getKeys().begin(), lookup.getKeys().end());
   unsigned oldKeyCount = keys.size();
   keys.push_back(key);
   lookup.setKeysAttr(mlir::ArrayAttr::get(lookup.getContext(), keys));

   mlir::Block& eqBlock = lookup.getEqFn().front();
   mlir::Location loc = lookup.getLoc();
   mlir::BlockArgument storedKey =
      eqBlock.insertArgument(oldKeyCount, key.getColumn().type, loc);
   mlir::BlockArgument lookupKey =
      eqBlock.insertArgument(oldKeyCount * 2 + 1, key.getColumn().type, loc);
   auto ret = mlir::cast<tuples::ReturnOp>(eqBlock.getTerminator());
   assert(ret.getNumOperands() == 1 && "aggregate lookup equality must return one predicate");
   mlir::OpBuilder b(ret);
   mlir::Value keyMatches = b.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, storedKey, lookupKey);
   mlir::Value combined = b.create<mlir::arith::AndIOp>(loc, ret.getOperand(0), keyMatches);
   b.create<tuples::ReturnOp>(loc, mlir::ValueRange{combined});
   ret.erase();
}

static llvm::SmallVector<unsigned, 8>
aggregateFilterSlotsForMixedUnion(unsigned donorFilterSlot,
                                  llvm::ArrayRef<AggregatePeerRewriteInput> peers) {
   llvm::SmallVector<unsigned, 8> filterSlots;
   filterSlots.push_back(donorFilterSlot);
   for (const AggregatePeerRewriteInput& peer : peers) filterSlots.push_back(peer.filterSlot);
   sortUniqueSlots(filterSlots);
   return filterSlots;
}

static tuples::ColumnRefAttr insertSyntheticAggregateQueryIdForUnion(
   mlir::ModuleOp synthetic, subop::ExecutionStepOp buildStep,
   subop::LookupOrInsertOp lookup, llvm::ArrayRef<AggregatePeerRewriteInput> peers,
   const ModuleReuseInfo& reuseSynthetic, unsigned donorFilterSlot) {
   llvm::SmallVector<unsigned, 8> filterSlots = aggregateFilterSlotsForMixedUnion(donorFilterSlot, peers);
   std::optional<tuples::ColumnRefAttr> qidRef =
      insertSyntheticAggregateQueryIdFromMixedPreds(buildStep, filterSlots, donorFilterSlot);
   if (qidRef) return *qidRef;

   llvm::SmallVector<RuntimeFilterIdClause, 8> clauses;
   llvm::DenseSet<unsigned> seenClauseSlots;
   RuntimeFilterIdClause donorClause;
   donorClause.id = donorFilterSlot;
   donorClause.filters = decodeFiltersFromTableScanInExecutionStep(buildStep);
   assert(!donorClause.filters.empty() && "mixed aggregate reuse requires donor simple filters");
   clauses.push_back(std::move(donorClause));
   seenClauseSlots.insert(donorFilterSlot);
   for (const AggregatePeerRewriteInput& peer : peers) {
      if (!seenClauseSlots.insert(peer.filterSlot).second) continue;
      RuntimeFilterIdClause clause;
      clause.id = peer.filterSlot;
      clause.filters = decodeFiltersFromTableScanInExecutionStep(peer.buildStep);
      assert(!clause.filters.empty() && "mixed aggregate reuse requires peer simple filters");
      clauses.push_back(std::move(clause));
   }
   subop::ScanRefsOp scanOp = findAggregateTableScanInStep(buildStep);
   widenAggregateExternalTableForPeers(synthetic, buildStep, scanOp, peers, reuseSynthetic,
                                       /*mergeFiltersForOrReuse=*/true);
   return insertSyntheticAggregateQueryIdColumn(buildStep, clauses, donorFilterSlot);
}

static void appendMixedAggregateQueryIdKey(
   mlir::ModuleOp synthetic, CachedAggregateLayout& layout,
   llvm::SmallVectorImpl<subop::Member>& newKeyMembers,
   subop::ExecutionStepOp buildStep, subop::LookupOrInsertOp lookup,
   llvm::ArrayRef<AggregatePeerRewriteInput> peers,
   const ModuleReuseInfo& reuseSynthetic, unsigned donorFilterSlot) {
   auto* ctx = synthetic.getContext();
   layout.queryIdMember = makeOrGetAggregateQueryIdMember(ctx);
   newKeyMembers.push_back(layout.queryIdMember);
   tuples::ColumnRefAttr qidRef =
      insertSyntheticAggregateQueryIdForUnion(synthetic, buildStep, lookup, peers, reuseSynthetic,
                                              donorFilterSlot);
   appendAggregateLookupKey(lookup, qidRef);
}

struct AggregateProducerPayloadIndex {
   llvm::DenseMap<uint64_t, AggregatePayloadMemberInfo> byHash;
   unsigned nextSlot = 0;

   bool containsEquivalentPayload(const AggregatePayloadMemberInfo& payload) const {
      return byHash.contains(payload.semanticHash);
   }

   void record(const AggregatePayloadMemberInfo& payload, subop::MemberManager& mm) {
      byHash[payload.semanticHash] = payload;
      if (auto slot = parseAggregateValueSlot(mm.getName(payload.member))) nextSlot = std::max(nextSlot, *slot + 1);
   }
};

static AggregateProducerPayloadIndex buildAggregateProducerPayloadIndex(
   llvm::ArrayRef<AggregatePayloadMemberInfo> producerInfos, subop::MemberManager& mm) {
   AggregateProducerPayloadIndex index;
   for (const auto& payload : producerInfos) index.record(payload, mm);
   return index;
}

struct InsertedAggregatePayload {
   unsigned idx;
   mlir::Type type;
   AggregatePayloadKind kind;
};

static InsertedAggregatePayload insertAggregatePayloadIntoSyntheticBuild(
   CachedAggregateLayout& layout,
   llvm::SmallVectorImpl<subop::Member>& newMembers,
   llvm::SmallVectorImpl<mlir::Type>& newTypes,
   subop::LookupOrInsertOp lookup,
   subop::ReduceOp reduce,
   AggregateProducerPayloadIndex& producerIndex,
   subop::MemberManager& mm,
   const AggregatePayloadMemberInfo& payload,
   tuples::ColumnRefAttr source,
   mlir::Type slotTy) {
   AggregatePayloadKind kind = aggregatePayloadKindForSemantic(payload.semanticKey);
   subop::Member member = allocUnusedAggregateValueSlot(mm, slotTy, producerIndex.nextSlot);
   unsigned insertIdx = layout.payloadSemanticKeys.size();
   newMembers.insert(newMembers.begin() + insertIdx, member);
   newTypes.insert(newTypes.begin() + insertIdx, slotTy);
   layout.payloadSemanticKeys.insert(layout.payloadSemanticKeys.begin() + insertIdx, payload.semanticKey);
   layout.payloadMembers.insert(layout.payloadMembers.begin() + insertIdx, member);
   layout.payloadColumnTypes.insert(layout.payloadColumnTypes.begin() + insertIdx, slotTy);
   insertAggregateLookupInitial(lookup, insertIdx, slotTy, kind);
   insertAggregateReduceUpdate(reduce, insertIdx, source, member, slotTy, kind);
   insertAggregateReduceCombine(reduce, insertIdx, slotTy, kind);
   producerIndex.record(AggregatePayloadMemberInfo{payload.semanticKey, payload.semanticHash, member, slotTy,
                                                   payload.sourceColumn},
                        mm);
   return InsertedAggregatePayload{insertIdx, slotTy, kind};
}

static void applySyntheticAggregatePayloadUnion(mlir::ModuleOp synthetic, mlir::Value syntheticHtState,
                                                CachedAggregateLayout& layout,
                                                llvm::ArrayRef<AggregatePeerRewriteInput> peers,
                                                const ModuleReuseInfo& reuseSynthetic,
                                                unsigned donorQueryId) {
   auto* ctx = synthetic.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   subop::ExecutionStepOp buildStep = findAggregateBuildStepForHt(syntheticHtState, reuseSynthetic);
   subop::ExecutionStepOp mergeStep = findAggregateMergeStepForHt(syntheticHtState, reuseSynthetic);
   subop::MergeOp merge = findAggregateMergeInStep(mergeStep);
   subop::ReduceOp reduce = findAggregateReduceInStep(buildStep);
   subop::LookupOrInsertOp lookup = findAggregateLookupInStep(buildStep);
   subop::PreAggrHtType oldHt = mlir::cast<subop::PreAggrHtType>(syntheticHtState.getType());
   subop::PreAggrHtFragmentType oldFrag = fragmentTypeForAggregateHt(oldHt);
   llvm::SmallVector<subop::Member> newKeyMembers(oldHt.getKeyMembers().getMembers().begin(),
                                                  oldHt.getKeyMembers().getMembers().end());

   if (layout.mixedByQueryId) {
      appendMixedAggregateQueryIdKey(synthetic, layout, newKeyMembers, buildStep, lookup, peers,
                                     reuseSynthetic, donorQueryId);
   }

   llvm::SmallVector<AggregatePayloadMemberInfo, 16> producerInfos =
      collectAggregatePayloadMembers(synthetic, syntheticHtState, buildStep);
   AggregateProducerPayloadIndex producerIndex = buildAggregateProducerPayloadIndex(producerInfos, mm);

   llvm::SmallVector<subop::Member> newMembers(oldHt.getValueMembers().getMembers().begin(),
                                               oldHt.getValueMembers().getMembers().end());
   llvm::SmallVector<mlir::Type> newTypes;
   for (subop::Member m : newMembers) newTypes.push_back(mm.getType(m));

   subop::MapOp producerMap;
   llvm::SmallVector<InsertedAggregatePayload, 4> insertedPayloads;
   for (const AggregatePeerRewriteInput& peer : peers) {
      llvm::SmallVector<AggregatePayloadMemberInfo, 16> peerInfos =
         collectAggregatePayloadMembers(peer.module, peer.state, peer.buildStep);
      for (unsigned peerIdx = 0; peerIdx < peerInfos.size(); ++peerIdx) {
         const auto& p = peerInfos[peerIdx];
         if (producerIndex.containsEquivalentPayload(p)) continue;
         if (p.semanticKey == "count:*") continue;
         tuples::ColumnRefAttr source = cloneColumnRefToContext(p.sourceColumn, ctx);
         auto sourceProducerMap = findMapProducingColumn(buildStep, source, cm);
         if (sourceProducerMap) {
            source = mapComputedRefForSemantic(sourceProducerMap, columnSemanticKey(source, cm), cm);
         } else {
            subop::MapOp peerMap = findMapProducingColumn(peer.buildStep, p.sourceColumn,
                                                          p.sourceColumn.getContext()
                                                             ->getLoadedDialect<tuples::TupleStreamDialect>()
                                                             ->getColumnManager());
            assert(peerMap && "aggregate union: missing computed aggregate payload requires peer map");
            if (!producerMap) producerMap = findProducerAggregateMap(buildStep);
            source = clonePeerMapResultIntoProducer(buildStep, producerMap, peerMap, p.sourceColumn,
                                                   reuseSynthetic, peer.reuse, synthetic);
         }
         mlir::Type slotTy = cloneTypeToContext(p.type, ctx);
         insertedPayloads.push_back(insertAggregatePayloadIntoSyntheticBuild(
            layout, newMembers, newTypes, lookup, reduce, producerIndex, mm, p, source, slotTy));
      }
   }

   auto newKeyAttr = subop::StateMembersAttr::get(ctx, newKeyMembers);
   auto newFrag = subop::PreAggrHtFragmentType::get(ctx, newKeyAttr,
                                                    subop::StateMembersAttr::get(ctx, newMembers),
                                                    oldHt.getWithLock());
   auto newHt = subop::PreAggrHtType::get(ctx, newKeyAttr,
                                          subop::StateMembersAttr::get(ctx, newMembers),
                                          oldHt.getWithLock());
   if (layout.mixedByQueryId) {
      appendAggregateMergeEqKey(merge, oldHt.getKeyMembers().getMembers().size(), mm.getType(layout.queryIdMember));
   }
   for (auto payload : insertedPayloads)
      insertAggregateMergeCombine(merge, payload.idx, payload.type, payload.kind);

   llvm::DenseSet<void*> closure;
   auto seedValue = [&](mlir::Value v) {
      if (v) closure.insert(v.getAsOpaquePointer());
   };
   auto seedStep = [&](subop::ExecutionStepOp step) {
      for (mlir::Value operand : step->getOperands()) seedValue(operand);
      for (mlir::Value result : step->getResults()) seedValue(result);
      step.walk([&](mlir::Operation* op) {
         for (mlir::Value operand : op->getOperands()) seedValue(operand);
         for (mlir::Value result : op->getResults()) seedValue(result);
         for (mlir::Region& region : op->getRegions())
            for (mlir::Block& block : region)
               for (mlir::BlockArgument arg : block.getArguments()) seedValue(arg);
      });
   };
   seedValue(syntheticHtState);
   seedStep(buildStep);
   seedStep(mergeStep);
   auto seedCreateOnlyStepForState = [&](mlir::Value state) {
      auto it = reuseSynthetic.createOnlyStepForState.find(state);
      if (it != reuseSynthetic.createOnlyStepForState.end()) seedStep(it->second);
   };
   seedCreateOnlyStepForState(syntheticHtState);
   forEachShadowChainPredecessor(syntheticHtState, reuseSynthetic, [&](mlir::Value state) {
      seedValue(state);
      seedCreateOnlyStepForState(state);
   });
   for (;;) {
      size_t before = closure.size();
      expandClosureThroughExecutionStepPorts(synthetic, closure);
      if (closure.size() == before) break;
   }

   updateSyntheticAggregateStateTypes(synthetic, closure, oldFrag, newFrag, oldHt, newHt);
   syncAggregateEntryRefAttrs(synthetic, closure, oldFrag, newFrag, oldHt, newHt);
   layout.producerHt = newHt;
}

static llvm::StringMap<llvm::SmallVector<subop::Member, 4>>
aggregateSemanticToConsumerMembers(const CachedAggregateLayout& layout, unsigned consumerIdx) {
   llvm::StringMap<llvm::SmallVector<subop::Member, 4>> out;
   llvm::ArrayRef<std::string> keys;
   llvm::ArrayRef<subop::Member> members;
   auto itKeys = layout.querySemanticKeysById.find(consumerIdx);
   auto itMembers = layout.queryMembersById.find(consumerIdx);
   if (itKeys != layout.querySemanticKeysById.end() && itMembers != layout.queryMembersById.end()) {
      keys = itKeys->second;
      members = itMembers->second;
   } else {
      keys = consumerIdx == 0 ? layout.query0SemanticKeys : layout.query1SemanticKeys;
      members = consumerIdx == 0 ? layout.query0Members : layout.query1Members;
   }
   assert(keys.size() == members.size());
   for (size_t i = 0; i < keys.size(); ++i) out[keys[i]].push_back(members[i]);
   return out;
}

struct ConsumerAggregateAlignment {
   subop::PreAggrHtType ht;
   subop::Member queryIdMember;
   llvm::DenseMap<subop::Member, subop::Member> memberRemap;
   llvm::DenseMap<subop::Member, subop::Member> keyMemberRemap;
};

static llvm::DenseMap<subop::Member, subop::Member>
buildAggregateKeyMemberRemap(subop::PreAggrHtType oldHt, subop::PreAggrHtType alignedHt) {
   llvm::DenseMap<subop::Member, subop::Member> remap;
   auto& mm = oldHt.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   llvm::ArrayRef<subop::Member> oldKeys = oldHt.getKeyMembers().getMembers();
   llvm::ArrayRef<subop::Member> alignedKeys = alignedHt.getKeyMembers().getMembers();
   assert((oldKeys.size() == alignedKeys.size() || oldKeys.size() + 1 == alignedKeys.size()) &&
          "aggregate cache_get key layout may only append query_id");
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
   if (layout.mixedByQueryId) {
      assert(!keyMembers.empty() && "mixed aggregate layout must append query_id key");
      out.queryIdMember = keyMembers.back();
   }
   out.memberRemap = std::move(remap);
   return out;
}

static void insertAggregateQueryIdConsumerFilter(subop::ScanRefsOp scan,
                                                 subop::Member queryIdMember,
                                                 unsigned consumerIdx) {
   assert(queryIdMember && "mixed aggregate consumer filter requires query_id member");
   auto* ctx = scan.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();

   tuples::ColumnDefAttr qidDef = cm.createDef(cm.getUniqueScope("agg_reuse_query_id"), "query_id");
   qidDef.getColumn().type = mm.getType(queryIdMember);
   tuples::ColumnRefAttr qidRef = cm.createRef(&qidDef.getColumn());
   tuples::ColumnDefAttr predDef = cm.createDef(cm.getUniqueScope("agg_reuse_query_id_filter"), "pred");
   predDef.getColumn().type = mlir::IntegerType::get(ctx, 1);
   tuples::ColumnRefAttr predRef = cm.createRef(&predDef.getColumn());

   mlir::OpBuilder b(scan);
   b.setInsertionPointAfter(scan);
   auto gather = b.create<subop::GatherOp>(
      scan.getLoc(), scan.getRes(), cm.createRef(&scan.getRef().getColumn()),
      subop::ColumnDefMemberMappingAttr::get(
         ctx, llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>>{{queryIdMember, qidDef}}));

   b.setInsertionPointAfter(gather);
   auto map = b.create<subop::MapOp>(scan.getLoc(), tuples::TupleStreamType::get(ctx), gather.getRes(),
                                     b.getArrayAttr({predDef}), b.getArrayAttr({qidRef}));
   mlir::Block* block = new mlir::Block();
   block->addArgument(qidRef.getColumn().type, scan.getLoc());
   map.getFn().push_back(block);
   mlir::OpBuilder rb(ctx);
   rb.setInsertionPointToStart(block);
   mlir::Value want = rb.create<db::ConstantOp>(scan.getLoc(), qidRef.getColumn().type,
                                                rb.getI64IntegerAttr(static_cast<int64_t>(consumerIdx)));
   mlir::Value pred = rb.create<mlir::arith::CmpIOp>(scan.getLoc(), mlir::arith::CmpIPredicate::eq,
                                                     block->getArgument(0), want);
   rb.create<tuples::ReturnOp>(scan.getLoc(), mlir::ValueRange{pred});

   b.setInsertionPointAfter(map);
   auto filter = b.create<subop::FilterOp>(scan.getLoc(), map.getResult(),
                                           subop::FilterSemantic::all_true,
                                           b.getArrayAttr({predRef}));
   rewireStreamUsesAfterAnchorInStep(scan.getRes(), filter.getRes(), filter.getOperation(),
                                     llvm::ArrayRef<mlir::Operation*>{
                                        gather.getOperation(), map.getOperation(), filter.getOperation()});
}

using ProbeMemberSemanticMap = llvm::StringMap<llvm::DenseMap<subop::Member, std::string>>;

static ProbeMemberSemanticMap collectProbeMemberSemanticsFromGathers(
   mlir::ModuleOp module, subop::HashIndexedViewType targetHiv,
   const llvm::StringSet<>* probeLookupScopes = nullptr,
   const llvm::DenseSet<void*>* ssaClosure = nullptr) {
   ProbeMemberSemanticMap byProbeScope;
   auto& cm = module.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   module.walk([&](subop::GatherOp gather) {
      auto gatherRef = gather.getRef();
      auto [refScope, refLeaf] = cm.getName(&gatherRef.getColumn());
      (void)refLeaf;
      if (!isJoinProbeCompilerScope(refScope)) return;
      (void)probeLookupScopes;
      (void)ssaClosure;
      auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(gatherRef.getColumn().type);
      auto st = ler ? asHashIndexedViewLayoutType(ler.getState()) : nullptr;
      if (!st || (targetHiv && !sameHashIndexedViewJoinKey(st, targetHiv))) return;

      auto& scopeMap = byProbeScope[refScope];
      for (auto& [mem, def] : gather.getMapping().getMapping()) {
         auto [defScope, defLeaf] = cm.getName(&def.getColumn());
         if (isJoinProbeCompilerScope(defScope) && isPayloadMemberSlotName(defLeaf)) continue;
         scopeMap.try_emplace(mem, columnSemanticKey(defScope, defLeaf));
      }
   });
   return byProbeScope;
}

static void mergeProbeMemberSemantics(ProbeMemberSemanticMap& dst, const ProbeMemberSemanticMap& src) {
   for (const auto& scopeIt : src) {
      auto& dstScope = dst[scopeIt.getKey()];
      for (const auto& memberIt : scopeIt.second) {
         dstScope.try_emplace(memberIt.first, memberIt.second);
      }
   }
}

static ProbeMemberSemanticMap collectProbeMemberSemanticsFromMapComparisons(mlir::ModuleOp module) {
   ProbeMemberSemanticMap byProbeScope;
   auto& cm = module.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = module.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();

   module.walk([&](subop::MapOp map) {
      if (map.getFn().empty()) return;
      mlir::Block& block = map.getFn().front();
      auto inputRefForArg = [&](mlir::Value v) -> tuples::ColumnRefAttr {
         auto arg = mlir::dyn_cast<mlir::BlockArgument>(v);
         if (!arg || arg.getOwner() != &block) return {};
         unsigned idx = arg.getArgNumber();
         if (idx >= map.getInputCols().size()) return {};
         return mlir::dyn_cast<tuples::ColumnRefAttr>(map.getInputCols()[idx]);
      };
      auto probePayloadMemberForArg = [&](mlir::Value v) -> std::optional<std::pair<std::string, subop::Member>> {
         tuples::ColumnRefAttr ref = inputRefForArg(v);
         if (!ref) return std::nullopt;
         auto [scope, leaf] = cm.getName(&ref.getColumn());
         if (!isJoinProbeCompilerScope(scope) || !isPayloadMemberSlotName(leaf)) return std::nullopt;
         if (!mm.hasMemberDirect(leaf)) return std::nullopt;
         return std::make_pair(scope, mm.lookupMember(leaf));
      };
      auto semanticForArg = [&](mlir::Value v) -> std::optional<std::string> {
         tuples::ColumnRefAttr ref = inputRefForArg(v);
         if (!ref) return std::nullopt;
         auto [scope, leaf] = cm.getName(&ref.getColumn());
         if (isJoinProbeCompilerScope(scope) && isPayloadMemberSlotName(leaf)) return std::nullopt;
         return columnSemanticKey(scope, leaf);
      };
      auto record = [&](mlir::Value maybePayload, mlir::Value maybeSemantic) {
         auto payload = probePayloadMemberForArg(maybePayload);
         if (!payload) return;
         auto semantic = semanticForArg(maybeSemantic);
         if (!semantic) return;
         byProbeScope[payload->first].try_emplace(payload->second, *semantic);
      };

      map.walk([&](db::CmpOp cmp) {
         record(cmp.getLeft(), cmp.getRight());
         record(cmp.getRight(), cmp.getLeft());
      });
      map.walk([&](mlir::arith::CmpIOp cmp) {
         record(cmp.getLhs(), cmp.getRhs());
         record(cmp.getRhs(), cmp.getLhs());
      });
   });
   return byProbeScope;
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
   auto listHiv = asHashIndexedViewLayoutType(listElemLer.getState());
   if (!listHiv) return;

   mlir::Type alignedCarrierHiv =
      alignedHivStatePreservingMixedPred(ctx, scanList.getList().getType(), alignedHiv);
   setValueCarrierHivStateType(scanList.getList(), alignedCarrierHiv, consumerHivBeforeAlign);
   listTy = mlir::dyn_cast<subop::ListType>(scanList.getList().getType());
   if (!listTy) return;
   listElemLer = mlir::dyn_cast<subop::LookupEntryRefType>(listTy.getT());
   if (!listElemLer) return;
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   subop::LookupEntryRefType expectedLer =
      alignedEntryRefPreservingMixedPred(ctx, listTy, alignedHiv);
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

   llvm::StringMap<subop::Member> semKeyToMember = semKeyToAlignedHivMemberMap(consumerLayout, alignedHiv);
   llvm::DenseSet<subop::Member> hivMemberSet(consumerLayout.payloadMembers.begin(),
                                                consumerLayout.payloadMembers.end());
   llvm::StringMap<subop::Member> hivMemberByName;
   for (subop::Member m : alignedHiv.getValueMembers().getMembers()) {
      hivMemberByName[mm.getName(m)] = m;
   }
   ProbeMemberSemanticMap probeMemberSemantic =
      collectProbeMemberSemanticsFromGathers(scanList->getParentOfType<mlir::ModuleOp>(), alignedHiv);
   mergeProbeMemberSemantics(
      probeMemberSemantic,
      collectProbeMemberSemanticsFromMapComparisons(scanList->getParentOfType<mlir::ModuleOp>()));
   llvm::StringMap<subop::Member> lookupKeySemanticAliases =
      lookupKeySemanticAliasesForScanList(scanList, alignedHiv);

   auto resolveGatherMember = [&](llvm::StringRef refScope, subop::Member useMem,
                                  tuples::ColumnDefAttr def) -> subop::Member {
      auto [scope, leaf] = cm.getName(&def.getColumn());
      std::string semKey = columnSemanticKey(scope, leaf);
      if (!(isJoinProbeCompilerScope(scope) && isPayloadMemberSlotName(leaf))) {
         if (auto member = lookupMemberBySemanticOrUniqueLeaf(semKeyToMember, semKey)) return *member;
         if (auto alias = lookupKeySemanticAliases.find(semKey); alias != lookupKeySemanticAliases.end())
            return alias->second;
      }
      if (auto byScope = probeMemberSemantic.find(refScope); byScope != probeMemberSemantic.end()) {
         if (auto sem = byScope->second.find(useMem); sem != byScope->second.end()) {
            if (auto member = lookupMemberBySemanticOrUniqueLeaf(semKeyToMember, sem->second)) return *member;
            if (auto alias = lookupKeySemanticAliases.find(sem->second); alias != lookupKeySemanticAliases.end())
               return alias->second;
         }
      }
      if (auto itRemap = consumerLayout.probeGatherMemberRemap.find(useMem);
          itRemap != consumerLayout.probeGatherMemberRemap.end()) {
         return itRemap->second;
      }
      if (hivMemberSet.contains(useMem)) return useMem;
      if (auto it = hivMemberByName.find(mm.getName(useMem)); it != hivMemberByName.end()) return it->second;
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
      auto lerBefore = mlir::dyn_cast<subop::LookupEntryRefType>(gatherRef.getColumn().type);
      auto stBefore = lerBefore ? asHashIndexedViewLayoutType(lerBefore.getState()) : nullptr;
      debugProbeAlign(dbg, [&](llvm::raw_ostream& os) {
         os << "  rewrite gather op=" << gather.getOperation() << " ref_scope=" << refScope
            << " leaf=" << refLeaf << " ref_type_before=" << refTyBefore;
      });

      gatherRef.getColumn().type = expectedLer;
      gather.setRefAttr(gatherRef);

      llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> newMapping;
      llvm::StringMap<tuples::ColumnRefAttr> oldDefRefsToNewDefs;
      bool mappingChanged = false;
      for (auto& pr : gather.getMapping().getMapping()) {
         subop::Member nm = resolveGatherMember(refScope, pr.first, pr.second);
         if (nm == pr.first)
            nm = payloadOrdinalRemapMember(stBefore, alignedHiv, pr.first).value_or(nm);
         if (nm != pr.first) mappingChanged = true;
         tuples::ColumnDefAttr def = pr.second;
         auto [defScope, defLeaf] = cm.getName(&def.getColumn());
         if (isJoinProbeCompilerScope(defScope) && !lookupProbeRefScopesMatch(entryScope, defScope)) {
            debugProbeAlign(dbg, [&](llvm::raw_ostream& os) {
               os << "    mapping payload@" << defScope << "::" << defLeaf << " -> @" << entryScope
                  << "::" << mm.getName(nm);
            });
            std::string oldDefKey = columnSemanticKey(defScope, defLeaf);
            def = cm.createDef(entryScope, mm.getName(nm));
            def.getColumn().type = mm.getType(nm);
            oldDefRefsToNewDefs[oldDefKey] = cm.createRef(&def.getColumn());
            mappingChanged = true;
         } else if (auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(def.getColumn().type)) {
            auto st = asHashIndexedViewLayoutType(ler.getState());
            if (st && lookupProbeRefScopesMatch(entryScope, defScope) &&
                sameHashIndexedViewLayout(st, alignedHiv) && def.getColumn().type != expectedLer) {
               def.getColumn().type = expectedLer;
               mappingChanged = true;
            }
         }
         newMapping.push_back({nm, def});
      }
      if (mappingChanged) gather.setMappingAttr(subop::ColumnDefMemberMappingAttr::get(ctx, newMapping));
      syncMapInputColsFromGather(gather, cm, &oldDefRefsToNewDefs);

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
            if (listRef.getColumn().type != expectedListTy) {
               listRef.getColumn().type = expectedListTy;
               lookup.setRefAttr(listRef);
            }
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
   auto st = asHashIndexedViewLayoutType(ler.getState());
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
   // The list operand may still carry a pre-align embedded HIV predicate slot while cache_get has already
   // been retagged to the consumer slot. Key-layout equality is enough to keep the probed stream in scope.
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
                                                  llvm::StringMap<ConsumerColumnBinding>& out,
                                                  subop::HashIndexedViewType targetHiv = nullptr) {
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
      if (!ler) return;
      auto st = asHashIndexedViewLayoutType(ler.getState());
      if (!st) return;
      if (targetHiv && !sameHashIndexedViewLayout(st, targetHiv)) return;
      for (auto [mem, def] : gather.getMapping().getMapping()) {
         recordOnHivState(mem, def);
      }
   });
   consumer.walk([&](subop::ScatterOp scatter) {
      auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(scatter.getRef().getColumn().type);
      if (!ler) return;
      auto st = asHashIndexedViewLayoutType(ler.getState());
      if (!st) return;
      if (targetHiv && !sameHashIndexedViewLayout(st, targetHiv)) return;
      for (auto [mem, colRef] : scatter.getMapping().getMapping()) {
         auto [scope, leaf] = cm.getName(&colRef.getColumn());
         record(mem, scope, leaf, colRef.getColumn().type);
      }
   });
   if (targetHiv) return;
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
   if (auto it = layout.querySemanticKeysById.find(consumerQueryIndex);
       it != layout.querySemanticKeysById.end()) {
      return llvm::is_contained(it->second, semKey);
   }
   if (consumerQueryIndex == 0) return llvm::is_contained(layout.query0SemanticKeys, semKey);
   if (consumerQueryIndex == 1) return llvm::is_contained(layout.query1SemanticKeys, semKey);
   return false;
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

   llvm::StringMap<subop::Member> semKeyToMember = semKeyToAlignedHivMemberMap(consumerLayout, alignedHiv);

   llvm::StringMap<ConsumerColumnBinding> semanticToConsumer;
   collectConsumerPayloadSemanticMembers(consumer, semanticToConsumer);
   ProbeMemberSemanticMap probeMemberSemantic =
      collectProbeMemberSemanticsFromGathers(consumer, alignedHiv, probeLookupScopes, ssaClosure);
   mergeProbeMemberSemantics(probeMemberSemantic, collectProbeMemberSemanticsFromMapComparisons(consumer));

   auto resolveGatherMember = [&](llvm::StringRef refScope, subop::Member useMem,
                                  tuples::ColumnDefAttr def) -> subop::Member {
      auto [scope, leaf] = cm.getName(&def.getColumn());
      std::string semKey = columnSemanticKey(scope, leaf);
      if (!(isJoinProbeCompilerScope(scope) && isPayloadMemberSlotName(leaf))) {
         if (auto member = lookupMemberBySemanticOrUniqueLeaf(semKeyToMember, semKey)) return *member;
      }
      if (auto itRemap = consumerLayout.probeGatherMemberRemap.find(useMem);
          itRemap != consumerLayout.probeGatherMemberRemap.end()) {
         return itRemap->second;
      }
      if (auto byScope = probeMemberSemantic.find(refScope); byScope != probeMemberSemantic.end()) {
         if (auto sem = byScope->second.find(useMem); sem != byScope->second.end()) {
            if (auto member = lookupMemberBySemanticOrUniqueLeaf(semKeyToMember, sem->second)) return *member;
         }
      }
      if (hivMemberSet.contains(useMem)) return useMem;
      if (auto it = hivMemberByName.find(mm.getName(useMem)); it != hivMemberByName.end()) return it->second;
      if (auto it = semanticToConsumer.find(semKey); it != semanticToConsumer.end()) {
         if (hivMemberSet.contains(it->second.member)) return it->second.member;
         if (auto bn = hivMemberByName.find(mm.getName(it->second.member)); bn != hivMemberByName.end()) {
            return bn->second;
         }
      }
      if (isPayloadMemberSlotName(leaf) && isJoinProbeCompilerScope(scope)) {
         // Probe gathers name columns `@lookup_u_*::@member$N`; map pre-union consumer slots via semantics.
         for (const auto& e : semanticToConsumer) {
            if (e.second.member != useMem) continue;
            if (auto member = lookupMemberBySemanticOrUniqueLeaf(semKeyToMember, e.getKey())) return *member;
         }
      }
      return useMem;
   };

   consumer.walk([&](subop::GatherOp gather) {
      auto gatherRef = gather.getRef();
      auto [refScope, refLeaf] = cm.getName(&gatherRef.getColumn());
      if (!isJoinProbeCompilerScope(refScope)) return;
      bool inProbeScope = probeLookupScopes && probeLookupScopes->contains(refScope);
      bool inClosure = ssaClosure && opOperandsOrNestedBlockArgsTouchClosure(gather.getOperation(), *ssaClosure);
      if (probeLookupScopes && !probeLookupScopes->empty() && !inProbeScope && !inClosure) {
         debugProbeAlign(dbg, [&](llvm::raw_ostream& os) {
            os << "remapClosure consider gather outside traversal op=" << gather.getOperation()
               << " ref_scope=" << refScope;
         });
      }
      auto lerBefore = mlir::dyn_cast<subop::LookupEntryRefType>(gatherRef.getColumn().type);
      auto stBefore = lerBefore ? asHashIndexedViewLayoutType(lerBefore.getState()) : nullptr;
      if (!stBefore || !sameHashIndexedViewJoinKey(stBefore, alignedHiv)) return;
      auto expectedEntryRef =
         alignedEntryRefPreservingMixedPred(ctx, gatherRef.getColumn().type, alignedHiv);

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
      auto st = asHashIndexedViewLayoutType(ler.getState());
      if (!st || !sameHashIndexedViewLayout(st, alignedHiv)) return;
      llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> out;
      llvm::StringMap<tuples::ColumnRefAttr> oldDefRefsToNewDefs;
      bool changed = false;
      for (auto& pr : gather.getMapping().getMapping()) {
         subop::Member nm = resolveGatherMember(refScope, pr.first, pr.second);
         if (nm == pr.first)
            nm = payloadOrdinalRemapMember(stBefore, alignedHiv, pr.first).value_or(nm);
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
            auto defHiv = asHashIndexedViewLayoutType(lerDef.getState());
            if (defHiv && sameHashIndexedViewLayout(defHiv, alignedHiv) &&
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
/// All union \c filter_pred$N slots are retained (same as synthetic). When known, \p consumerReuseQueryIndex
/// also decides which non-predicate payload columns belonged to this consumer before union widening.
static subop::HashIndexedViewType buildConsumerAlignedHivType(mlir::ModuleOp consumer,
                                                                subop::HashIndexedViewType producerHiv,
                                                                const CachedJoinBufferLayout& cachedLayout,
                                                                subop::HashIndexedViewType consumerHivBeforeAlign,
                                                                std::optional<unsigned> consumerReuseQueryIndex,
                                                                CachedJoinBufferLayout& outConsumerLayout) {
   auto* ctx = consumer.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   CachedJoinBufferLayout layout = layoutInProducerValueOrder(cachedLayout, producerHiv);

   llvm::StringMap<ConsumerColumnBinding> semanticToConsumer;
   collectConsumerPayloadSemanticMembers(consumer, semanticToConsumer, consumerHivBeforeAlign);

   llvm::DenseMap<subop::Member, subop::Member> producerPayloadToConsumer;
   llvm::StringMap<subop::Member> assignedBySemantic;
   llvm::DenseSet<subop::Member> assignedMembers;
   llvm::StringSet<> assignedMemberNames;
   llvm::DenseMap<subop::Member, subop::Member> probeGatherMemberRemap = layout.probeGatherMemberRemap;
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
   auto recordProducerPayloadToConsumer = [&](subop::Member producerMem, subop::Member consumerMem) {
      producerPayloadToConsumer[producerMem] = consumerMem;
      llvm::SmallVector<subop::Member, 4> eraseRemaps;
      llvm::SmallVector<std::pair<subop::Member, subop::Member>, 4> translatedRemaps;
      for (const auto& remap : layout.probeGatherMemberRemap) {
         if (remap.second != producerMem) continue;
         if (remap.first == consumerMem) {
            eraseRemaps.push_back(remap.first);
         } else {
            translatedRemaps.push_back({remap.first, consumerMem});
         }
      }
      for (subop::Member key : eraseRemaps) probeGatherMemberRemap.erase(key);
      for (auto [from, to] : translatedRemaps) probeGatherMemberRemap[from] = to;
   };
   auto explicitConsumerMemberFromProbeRemap = [&](subop::Member producerMem,
                                                   mlir::Type producerSlotTy) -> std::optional<subop::Member> {
      if (!consumerHivBeforeAlign) return std::nullopt;
      std::optional<subop::Member> found;
      for (const auto& remap : layout.probeGatherMemberRemap) {
         if (remap.second != producerMem) continue;
         std::optional<subop::Member> consumerMem =
            findConsumerMemberByName(mm.getName(remap.first), consumerHivBeforeAlign);
         if (!consumerMem) continue;
         mlir::Type expectedTy = cloneTypeToContext(producerSlotTy, ctx);
         if (mm.getType(*consumerMem) != expectedTy)
            llvm::report_fatal_error("probe gather source member remap type mismatch");
         if (found && *found != *consumerMem)
            llvm::report_fatal_error("probe gather source member remap must identify one consumer member");
         found = *consumerMem;
      }
      return found;
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
         assignedMemberNames.insert(mm.getName(consumerMem));
         recordProducerPayloadToConsumer(producerMem, consumerMem);
         continue;
      }

      if (!consumerReuseQueryIndex) {
         subop::Member consumerMem =
            cloneMemberToContext(producerMem, producerHiv.getContext(), ctx, /*allowMemberTypeUpdate=*/false);
         assignedBySemantic[semKey] = consumerMem;
         assignedMembers.insert(consumerMem);
         assignedMemberNames.insert(mm.getName(consumerMem));
         recordProducerPayloadToConsumer(producerMem, consumerMem);
         continue;
      }

      bool payloadBelongsToConsumer = consumerQueryHadPayloadColumn(layout, semKey, *consumerReuseQueryIndex);

      std::optional<subop::Member> reused;
      if (payloadBelongsToConsumer) {
         reused = explicitConsumerMemberFromProbeRemap(producerMem, producerSlotTy);
         if (!reused) {
            auto it = semanticToConsumer.find(semKey);
            if (it != semanticToConsumer.end()) reused = it->second.member;
         }
         // Probe gathers use @lookup_u_* column defs; reuse the consumer's pre-cache_get HIV slot name
         // only for payloads that already existed in this query. Union-only payloads must get fresh slots.
         if (!reused && consumerHivBeforeAlign) {
            auto& producerMm =
               producerHiv.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
            reused = findConsumerMemberByName(producerMm.getName(producerMem), consumerHivBeforeAlign);
            if (!reused) {
               llvm::ArrayRef<subop::Member> oldVals = consumerHivBeforeAlign.getValueMembers().getMembers();
               if (consumerReuseQueryIndex) {
                  llvm::ArrayRef<std::string> qKeys;
                  if (auto it = layout.querySemanticKeysById.find(*consumerReuseQueryIndex);
                      it != layout.querySemanticKeysById.end()) {
                     qKeys = it->second;
                  } else if (*consumerReuseQueryIndex <= 1) {
                     qKeys = *consumerReuseQueryIndex == 0 ? llvm::ArrayRef<std::string>(layout.query0SemanticKeys)
                                                           : llvm::ArrayRef<std::string>(layout.query1SemanticKeys);
                  }
                  if (qKeys.size() == 1 && qKeys[0] == semKey && oldVals.size() == 1) {
                     probeGatherRemapFrom = oldVals[0];
                     if (!assignedMembers.contains(oldVals[0])) reused = oldVals[0];
                  }
               }
            }
         }
      }
      if (reused && (assignedMembers.contains(*reused) || assignedMemberNames.contains(mm.getName(*reused))))
         reused = std::nullopt;
      subop::Member consumerMem = ensureConsumerMember(reused, producerSlotTy);
      if (assignedMemberNames.contains(mm.getName(consumerMem))) {
         llvm::SmallVector<subop::Member, 8> bump;
         if (consumerHivBeforeAlign) {
            for (subop::Member m : consumerHivBeforeAlign.getValueMembers().getMembers()) bump.push_back(m);
         }
         for (subop::Member m : assignedMembers) bump.push_back(m);
         unsigned nextSlot = nextPayloadMemberSlot(mm, assignedBySemantic, bump);
         mlir::Type slotTy = cloneTypeToContext(producerSlotTy, ctx);
         consumerMem = allocUnusedPayloadMemberSlot(mm, slotTy, nextSlot);
      }
      if (probeGatherRemapFrom) {
         probeGatherMemberRemap[*probeGatherRemapFrom] = consumerMem;
      } else if (reused && consumerHivBeforeAlign && *reused != consumerMem) {
         probeGatherMemberRemap[*reused] = consumerMem;
      }
      if (producerMem != consumerMem) {
         bool producerNameAlreadyExistsInConsumer = false;
         if (consumerHivBeforeAlign) {
            producerNameAlreadyExistsInConsumer =
               static_cast<bool>(findConsumerMemberByName(mm.getName(producerMem), consumerHivBeforeAlign));
         }
         if (!producerNameAlreadyExistsInConsumer) probeGatherMemberRemap[producerMem] = consumerMem;
      }
      assignedBySemantic[semKey] = consumerMem;
      assignedMembers.insert(consumerMem);
      assignedMemberNames.insert(mm.getName(consumerMem));
      recordProducerPayloadToConsumer(producerMem, consumerMem);
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
      appendPayloadLayoutSlot(outConsumerLayout, layout.payloadSemanticKeys[i], consumerMem,
                              mm.getType(consumerMem));
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

bool aggregateHashTablePayloadUnionSupported(mlir::Value aggregateState, const ModuleReuseInfo& reuse) {
   return static_cast<bool>(tryFindAggregateBuildStepForHt(aggregateState, reuse));
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

static mlir::Value findHashIndexedViewBuiltByStep(mlir::ModuleOp module,
                                                  subop::ExecutionStepOp buildStep,
                                                  const ModuleReuseInfo& reuseSynthetic) {
   mlir::Value found;
   module.walk([&](subop::CreateHashIndexedView create) {
      if (found) return;
      subop::ExecutionStepOp producer = findStrictJoinBufferBuildStepForHiv(module, create.getResult(), reuseSynthetic);
      if (producer != buildStep) return;
      auto parentStep = create->getParentOfType<subop::ExecutionStepOp>();
      assert(parentStep && "hash-indexed view create must be inside an execution_step");
      auto ret = mlir::cast<subop::ExecutionStepReturnOp>(parentStep.getSubOps().front().getTerminator());
      for (unsigned i = 0; i < ret.getOperands().size(); ++i) {
         if (ret.getOperand(i) == create.getResult()) {
            found = parentStep.getResult(i);
            return;
         }
      }
      llvm_unreachable("hash-indexed view create result must be returned by its execution_step");
   });
   return found;
}

ClonedJoinBufferBuildSitesByKey recordClonedJoinBufferBuildSites(mlir::ModuleOp synthetic,
                                                                 llvm::ArrayRef<CacheTarget> targetsInSynthetic,
                                                                 const ModuleReuseInfo& reuseSynthetic) {
   ClonedJoinBufferBuildSitesByKey out;
   for (const CacheTarget& t : targetsInSynthetic) {
      if (!t.state || !asHashIndexedViewLayoutType(t.state.getType())) continue;
      subop::ExecutionStepOp buildStep = findStrictJoinBufferBuildStepForHiv(synthetic, t.state, reuseSynthetic);
      assert(buildStep && "recordClonedJoinBufferBuildSites: cache target must have an exact join-buffer build step");
	      ClonedJoinBufferBuildSite site;
	      site.cacheKey = t.cacheKey;
	      site.syntheticHiv = t.state;
	      site.syntheticMergedBuffer = resolveJoinMergedBuffer(site.syntheticHiv, synthetic, reuseSynthetic);
	      site.syntheticBuildStep = buildStep;
	      out[t.cacheKey] = site;
   }
   return out;
}

void refreshClonedJoinBufferBuildSiteStates(mlir::ModuleOp synthetic,
	                                            ClonedJoinBufferBuildSitesByKey& buildSites,
	                                            const ModuleReuseInfo& reuseSynthetic) {
	   for (auto& [key, site] : buildSites) {
	      assert(site.syntheticHiv && "refreshClonedJoinBufferBuildSiteStates: build site must keep its HIV SSA value");
	      site.syntheticMergedBuffer = resolveJoinMergedBuffer(site.syntheticHiv, synthetic, reuseSynthetic);
	   }
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
   auto& cm = synthetic.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

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
         FilterPredLayoutSlot predSlot =
            classifyFilterPredLayoutSlot(layout.payloadSemanticKeys[i], mm.getName(layout.payloadMembers[i]));
         if (!predSlot.isPred) continue;
         if (predSlot.isUnion) {
            unionPredMembers.push_back(layout.payloadMembers[i]);
            continue;
         }
         unsigned qIdx = predSlot.slot;
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
      bool hasUnrestrictedPredSlot = false;
      for (unsigned qIdx : predQueryIndices) {
         auto peerFilters = decodeFiltersFromTableScanInExecutionStep(peerBuilds[qIdx]);
         if (peerFilters.empty()) {
            hasUnrestrictedPredSlot = true;
            break;
         }
      }

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
         FilterPredLayoutSlot predSlot =
            classifyFilterPredLayoutSlot(layout.payloadSemanticKeys[i], mm.getName(layout.payloadMembers[i]));
         if (!predSlot.isPred || predSlot.isUnion) continue;
         unsigned qIdx = predSlot.slot;
         llvm::StringRef predName = mm.getName(layout.payloadMembers[i]);
         subop::ExecutionStepOp peerBuild = peerBuilds.lookup(qIdx);
         assert(peerBuild && "insertSyntheticFilterPreds: missing peer build for filter_pred slot");
         // Per-query predicates come from the peer query's donor table get_external descr, not from HIV
         // writer steps (cache_put / create_hash_indexed_view do not read !subop.table).
         auto peerFilters = decodeFiltersFromTableScanInExecutionStep(peerBuild);
         auto filters = restrictFiltersToTableScanInExecutionStep(buildStep, peerFilters);
         if (filters.empty() && materializePredMemberFromUpstreamMixedScanList(buildStep, predName)) {
            continue;
         }
         insertWriteSidePredIntoBufferConstructionStepForPredMember(
            buildStep, peerFilters, predName,
            /*allowSharedScanPredicate=*/!hasUnrestrictedPredSlot,
            /*clearScanPushdown=*/hasUnrestrictedPredSlot);
      }
      if (!unionPredMembers.empty()) {
         subop::MaterializeOp mat = findJoinBufferMaterializeInStep(buildStep);
         assert(mat && "insertSyntheticFilterPreds: synthetic build step must materialize join buffer");
         llvm::SmallVector<tuples::ColumnRefAttr, 8> predRefs;
         for (auto& kv : predMembers) {
            if (tuples::ColumnRefAttr ref = materializedColumnForMember(mat, kv.second))
               predRefs.push_back(ref);
         }
         if (predRefs.size() >= 2) {
            predRefs = threadPredicateRefsToMaterializeStream(mat, predRefs, cm);
            if (hasUnrestrictedPredSlot || !streamIsAlreadyUnionFilteredBySharedTableScan(mat.getStream()))
               insertResidualFilterUnionAfterPredicates(mat.getStream(), predRefs);
         }
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
   const ClonedJoinBufferBuildSitesByKey& buildSites,
   const llvm::DenseMap<uint64_t, llvm::DenseMap<unsigned, unsigned>>* consumerSlotByCacheKeyAndQuery) {
   llvm::SmallVector<ModuleReuseInfo, 8> reuseByQuery;
   reuseByQuery.reserve(queries.size());
   for (mlir::ModuleOp query : queries) reuseByQuery.push_back(collectModuleReuseInfo(query));

   llvm::DenseMap<uint64_t, const CrossQueryStateMatchGroup*> groupByKey;
   for (const CrossQueryStateMatchGroup& group : groups) {
      if (group.entries.size() >= 2) groupByKey[group.cacheKey] = &group;
   }

   auto& mm = synthetic.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   auto& cm = synthetic.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

   for (const CacheTarget& t : targetsInSynthetic) {
      auto itL = layoutsByKey.find(t.cacheKey);
      auto itG = groupByKey.find(t.cacheKey);
      auto itB = buildSites.find(t.cacheKey);
      if (itL == layoutsByKey.end() || itG == groupByKey.end() || itB == buildSites.end()) continue;
      const CachedJoinBufferLayout& layout = itL->second;
      const CrossQueryStateMatchGroup& group = *itG->second;
      bool layoutHasFilterPredSlots = false;
      for (size_t i = 0; i < layout.payloadMembers.size(); ++i) {
         llvm::StringRef semantic;
         if (i < layout.payloadSemanticKeys.size()) semantic = layout.payloadSemanticKeys[i];
         if (classifyFilterPredLayoutSlot(semantic, mm.getName(layout.payloadMembers[i])).isPred) {
            layoutHasFilterPredSlots = true;
            break;
         }
      }
      if (!group.enableFilterPredReuse && !layoutHasFilterPredSlots) continue;
      subop::ExecutionStepOp buildStep = itB->second.syntheticBuildStep;
      if (!buildStep) continue;

      llvm::DenseMap<unsigned, mlir::Value> hivBySlot;
      llvm::DenseMap<unsigned, ModuleReuseInfo*> reuseInfoBySlot;
      llvm::DenseMap<unsigned, mlir::ModuleOp> moduleBySlot;
      for (const CrossQueryStateMatchEntry& entry : group.entries) {
         if (entry.query < 0 || static_cast<size_t>(entry.query) >= queries.size() || !entry.state) continue;
         unsigned slot = reuseSlotForEntry(entry);
         if (hivBySlot.contains(slot)) continue;
         ModuleReuseInfo& reuse = reuseByQuery[entry.query];
         hivBySlot[slot] = resolveCacheTargetStateForReuse(entry.state, reuse);
         reuseInfoBySlot[slot] = &reuse;
         moduleBySlot[slot] = queries[entry.query];
      }

      llvm::DenseMap<unsigned, subop::ExecutionStepOp> peerBuilds;
      llvm::DenseMap<unsigned, subop::Member> predMembers;
      llvm::SmallVector<subop::Member, 2> unionPredMembers;
      for (size_t i = 0; i < layout.payloadMembers.size(); ++i) {
         llvm::StringRef semantic;
         if (i < layout.payloadSemanticKeys.size()) semantic = layout.payloadSemanticKeys[i];
         FilterPredLayoutSlot predSlot =
            classifyFilterPredLayoutSlot(semantic, mm.getName(layout.payloadMembers[i]));
         if (!predSlot.isPred) continue;
         if (predSlot.isUnion) {
            unionPredMembers.push_back(layout.payloadMembers[i]);
            continue;
         }
         unsigned qIdx = predSlot.slot;
         auto itH = hivBySlot.find(qIdx);
         auto itR = reuseInfoBySlot.find(qIdx);
         auto itM = moduleBySlot.find(qIdx);
         if (itH == hivBySlot.end() || itR == reuseInfoBySlot.end() || itM == moduleBySlot.end()) {
            unionPredMembers.push_back(layout.payloadMembers[i]);
            continue;
         }
         predMembers[qIdx] = layout.payloadMembers[i];
         peerBuilds[qIdx] = findJoinBufferBuildStepForHiv(itM->second, itH->second, *itR->second);
         assert(peerBuilds.lookup(qIdx) &&
                "insertSyntheticFilterPredsForGroups: peer join-buffer build step with table scan_refs required");
      }

	      llvm::SmallVector<unsigned, 8> predQueryIndices;
	      for (auto& kv : predMembers) predQueryIndices.push_back(kv.first);
	      llvm::sort(predQueryIndices);
	      subop::MaterializeOp existingMat = findJoinBufferMaterializeInStep(buildStep);
	      assert(existingMat && "insertSyntheticFilterPredsForGroups: synthetic build step must materialize join buffer");
	      bool allPredMembersAlreadyMaterialized = !predMembers.empty();
	      for (auto& kv : predMembers) {
	         if (!materializedColumnForMember(existingMat, kv.second)) {
	            allPredMembersAlreadyMaterialized = false;
	            break;
	         }
	      }
	      if (allPredMembersAlreadyMaterialized) {
	         for (subop::Member unionPredMember : unionPredMembers) {
	            if (!materializedColumnForMember(existingMat, unionPredMember)) {
	               allPredMembersAlreadyMaterialized = false;
	               break;
	            }
	         }
	      }
	      if (allPredMembersAlreadyMaterialized) continue;
	      bool hasUnrestrictedPredSlot = false;
	      for (unsigned qIdx : predQueryIndices) {
	         auto peerFilters = decodeFiltersFromTableScanInExecutionStep(peerBuilds[qIdx]);
         if (peerFilters.empty()) {
            hasUnrestrictedPredSlot = true;
            break;
         }
      }

      if (predQueryIndices.size() >= 2) {
         subop::MaterializeOp mat = findJoinBufferMaterializeInStep(buildStep);
         assert(mat && "insertSyntheticFilterPredsForGroups: synthetic build step must materialize join buffer");
         llvm::DenseMap<unsigned, llvm::SmallVector<runtime::FilterDescription, 8>> simpleFilters;
         for (unsigned qIdx : predQueryIndices) {
            auto peerFilters = decodeFiltersFromTableScanInExecutionStep(peerBuilds[qIdx]);
            simpleFilters[qIdx] = restrictFiltersToTableScanInExecutionStep(buildStep, peerFilters);
         }
         if (rewriteSyntheticResidualFiltersAsFilterPredsForSlots(
                buildStep, predQueryIndices, peerBuilds, mat, predMembers, simpleFilters,
                group.cacheKey, consumerSlotByCacheKeyAndQuery)) {
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
	         auto filters = restrictFiltersToTableScanInExecutionStep(buildStep, peerFilters);
	         bool hasTableScanSource = !!findUniqueTableScanRefsInStep(buildStep);
	         if (consumerSlotByCacheKeyAndQuery) {
	            llvm::SmallVector<tuples::ColumnRefAttr, 4> upstreamRefs =
	               threadMappedUpstreamMixedPredColumnsToMaterializeStream(
	                  buildStep, qIdx, group.cacheKey, *consumerSlotByCacheKeyAndQuery);
	            if (!upstreamRefs.empty()) {
	               subop::MaterializeOp mat = findJoinBufferMaterializeInStep(buildStep);
	               assert(mat && "insertSyntheticFilterPredsForGroups: synthetic build step must materialize join buffer");
	               llvm::SmallVector<tuples::ColumnRefAttr, 4> predRefs;
	               if (hasTableScanSource && !peerFilters.empty()) {
	                  insertWriteSidePredIntoBufferConstructionStepForPredMember(
	                     buildStep, peerFilters, predName,
	                     /*allowSharedScanPredicate=*/!hasUnrestrictedPredSlot,
	                     /*clearScanPushdown=*/hasUnrestrictedPredSlot);
	                  tuples::ColumnRefAttr localRef = materializedColumnForMember(mat, predMembers[qIdx]);
	                  assert(localRef && "mapped upstream predicate rewrite: local predicate materialize missing");
	                  predRefs.push_back(localRef);
	               }
	               predRefs.append(upstreamRefs.begin(), upstreamRefs.end());
	               tuples::ColumnRefAttr predRef =
	                  insertAndPredsBeforeMaterialize(mat, predRefs,
	                                                  ("reuse_mapped_upstream_pred$" + llvm::Twine(qIdx)).str());
	               setMaterializeMapping(mat, predMembers[qIdx], predRef);
	               continue;
	            }
	         }
	         if ((!hasTableScanSource || filters.empty()) &&
	             materializePredMemberFromUpstreamMixedScanList(buildStep, predName)) {
	            continue;
	         }
         assert(hasTableScanSource &&
                "filter_pred insertion without a table scan must inherit an upstream predicate");
         insertWriteSidePredIntoBufferConstructionStepForPredMember(
            buildStep, peerFilters, predName,
            /*allowSharedScanPredicate=*/!hasUnrestrictedPredSlot,
            /*clearScanPushdown=*/hasUnrestrictedPredSlot);
      }
      if (!unionPredMembers.empty()) {
         subop::MaterializeOp mat = findJoinBufferMaterializeInStep(buildStep);
         assert(mat && "insertSyntheticFilterPredsForGroups: synthetic build step must materialize join buffer");
         llvm::SmallVector<tuples::ColumnRefAttr, 8> predRefs;
         for (auto& kv : predMembers) {
            if (tuples::ColumnRefAttr ref = materializedColumnForMember(mat, kv.second))
               predRefs.push_back(ref);
         }
         if (predRefs.size() >= 2) {
            predRefs = threadPredicateRefsToMaterializeStream(mat, predRefs, cm);
            if (hasUnrestrictedPredSlot || !streamIsAlreadyUnionFilteredBySharedTableScan(mat.getStream()))
               insertResidualFilterUnionAfterPredicates(mat.getStream(), predRefs);
         }
         for (subop::Member unionPredMember : unionPredMembers) {
            materializeConstantTruePredMemberOnBufferMaterialize(
               mat, mm.getName(unionPredMember), /*updateStreamOperand=*/true);
         }
      }
   }
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
   syncGatherMembersToMixedHivLayouts(module, nullptr);
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
      remapAlignedHivClosureGathers(consumer, nullptr, probe.alignedHiv, probe.consumerLayout,
                                    nullptr, probe.cacheKey, "remap-global");
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
         if (auto st = asHashIndexedViewLayoutType(ler.getState())) return st;
      }
   }
   if (auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(scanList.getElem().getColumn().type)) {
      if (auto st = asHashIndexedViewLayoutType(ler.getState())) return st;
   }
   return std::nullopt;
}

static subop::MixedHashIndexedViewType mixedHivTypeForPredMember(mlir::MLIRContext* ctx,
                                                                 subop::HashIndexedViewType hiv,
                                                                 subop::Member predMember) {
   assert(hiv && "probe predicate retag requires an HIV-like layout");
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   assert(valueMembersContainMemberNamed(ctx, hiv.getValueMembers(), mm.getName(predMember)) &&
          "probe predicate retag requires the predicate member in the cached HIV layout");
   return subop::MixedHashIndexedViewType::get(ctx, hiv.getKeyMembers(), hiv.getValueMembers(),
                                              hiv.getCompareHashForLookup(),
                                              mlir::StringAttr::get(ctx, mm.getName(predMember)));
}

static bool scanListTargetsAlignedHivPredSlot(mlir::MLIRContext* ctx, subop::ScanListOp scanList,
                                               subop::HashIndexedViewType alignedHiv, subop::Member predMember,
                                               subop::HashIndexedViewType consumerHivBeforeAlign) {
   mlir::Type predAlignedHiv = mixedHivTypeForPredMember(ctx, alignedHiv, predMember);
   setValueCarrierHivStateType(scanList.getList(), predAlignedHiv, consumerHivBeforeAlign);
   std::optional<subop::HashIndexedViewType> stOpt = hashIndexedViewFromScanListCarrier(scanList);
   if (!stOpt) return false;
   subop::HashIndexedViewType st = *stOpt;
   subop::HashIndexedViewType predLayout = asHashIndexedViewLayoutType(predAlignedHiv);
   if (st.getCompareHashForLookup() != predLayout.getCompareHashForLookup()) return false;
   if (st.getKeyMembers().getMembers().size() != predLayout.getKeyMembers().getMembers().size()) return false;
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   llvm::StringRef predName = mm.getName(predMember);
   return valueMembersContainMemberNamed(ctx, st.getValueMembers(), predName) &&
          valueMembersContainMemberNamed(ctx, predLayout.getValueMembers(), predName);
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
   predSpec.stableOrder = nextPayloadStableOrder(plan.payloadColumns);
   plan.payloadColumns.push_back(std::move(predSpec));
   refreshPayloadMemberTypes(plan);
}

struct JoinGroupPrimaryEntries {
   const CrossQueryStateMatchEntry* donor = nullptr;
   const CrossQueryStateMatchEntry* firstPeer = nullptr;
};

static bool joinGroupEntryValidForQueryList(const CrossQueryStateMatchEntry& entry, size_t numQueries) {
   return entry.query >= 0 && static_cast<size_t>(entry.query) < numQueries && entry.state;
}

static JoinGroupPrimaryEntries selectJoinGroupPrimaryEntries(
   const CrossQueryStateMatchGroup& group,
   size_t numQueries) {
   JoinGroupPrimaryEntries selected;
   for (const CrossQueryStateMatchEntry& entry : group.entries) {
      if (!joinGroupEntryValidForQueryList(entry, numQueries)) continue;
      if (!selected.donor || entry.query < selected.donor->query) selected.donor = &entry;
   }
   if (!selected.donor) return selected;

   auto findPeer = [&](bool requireDifferentReuseSlot) -> const CrossQueryStateMatchEntry* {
      for (const CrossQueryStateMatchEntry& entry : group.entries) {
         if (&entry == selected.donor) continue;
         if (!joinGroupEntryValidForQueryList(entry, numQueries)) continue;
         if (requireDifferentReuseSlot &&
             reuseSlotForEntry(entry) == reuseSlotForEntry(*selected.donor))
            continue;
         return &entry;
      }
      return nullptr;
   };
   selected.firstPeer = findPeer(/*requireDifferentReuseSlot=*/group.enableFilterPredReuse);
   if (!selected.firstPeer) selected.firstPeer = findPeer(/*requireDifferentReuseSlot=*/false);
   return selected;
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

      JoinGroupPrimaryEntries primary = selectJoinGroupPrimaryEntries(group, queries.size());
      if (!primary.donor || !primary.firstPeer) continue;
      const CrossQueryStateMatchEntry* donor = primary.donor;
      const CrossQueryStateMatchEntry* firstPeer = primary.firstPeer;

      const ModuleReuseInfo& reuseDonor = reuseByQuery[donor->query];
      const ModuleReuseInfo& reusePeer = reuseByQuery[firstPeer->query];
      mlir::Value hivDonor = resolveCacheTargetStateForReuse(donor->state, reuseDonor);
      mlir::Value hivPeer = resolveCacheTargetStateForReuse(firstPeer->state, reusePeer);
      if (!asHashIndexedViewLayoutType(hivDonor.getType()) ||
          !asHashIndexedViewLayoutType(hivPeer.getType())) {
         continue;
      }
      if (!findStrictJoinBufferBuildStepForHiv(queries[donor->query], hivDonor, reuseDonor) ||
          !findStrictJoinBufferBuildStepForHiv(queries[firstPeer->query], hivPeer, reusePeer)) {
         continue;
      }

      llvm::SmallVector<std::pair<mlir::ModuleOp, mlir::Value>, 8> extraPeerHivs;
      llvm::SmallVector<const ModuleReuseInfo*, 8> extraPeerReuses;
      llvm::SmallVector<unsigned, 8> extraPeerReuseSlots;
      for (const CrossQueryStateMatchEntry& entry : group.entries) {
         if (entry.query == donor->query || entry.query == firstPeer->query) continue;
         if (entry.query < 0 || static_cast<size_t>(entry.query) >= queries.size() || !entry.state) continue;
         const ModuleReuseInfo& reuseExtra = reuseByQuery[entry.query];
         mlir::Value hivExtra = resolveCacheTargetStateForReuse(entry.state, reuseExtra);
         if (!asHashIndexedViewLayoutType(hivExtra.getType())) continue;
         if (!findStrictJoinBufferBuildStepForHiv(queries[entry.query], hivExtra, reuseExtra)) continue;
         extraPeerHivs.push_back({queries[entry.query], hivExtra});
         extraPeerReuses.push_back(&reuseExtra);
         extraPeerReuseSlots.push_back(reuseSlotForEntry(entry));
      }
      if (!group.enableFilterPredReuse) {
         bool allLayoutsPhysicallyCompatible = hashIndexedViewLayoutsArePhysicallyCompatible(hivDonor, hivPeer);
         for (auto [extraMod, extraHiv] : extraPeerHivs) {
            (void)extraMod;
            allLayoutsPhysicallyCompatible &=
               hashIndexedViewLayoutsArePhysicallyCompatible(hivDonor, extraHiv);
         }
         if (allLayoutsPhysicallyCompatible) continue;
      }

      JoinBufferUnionPlan plan =
         buildUnionPlan(hivDonor, hivPeer, reuseDonor, reusePeer, queries[donor->query],
                        queries[firstPeer->query], reuseSlotForEntry(*donor),
                        reuseSlotForEntry(*firstPeer), /*enableFilterPredReuse=*/false,
                        extraPeerHivs, extraPeerReuses, extraPeerReuseSlots);
      if (group.enableFilterPredReuse) {
         llvm::SmallVector<unsigned, 8> queryIndices;
         llvm::DenseSet<unsigned> seenSlots;
         for (const CrossQueryStateMatchEntry& entry : group.entries) {
            if (entry.query < 0 || static_cast<size_t>(entry.query) >= queries.size()) continue;
            unsigned slot = reuseSlotForEntry(entry);
            if (!seenSlots.insert(slot).second) continue;
            queryIndices.push_back(slot);
            ensureReuseFilterPredColumn(plan, slot, synthetic.getContext());
         }
         unsigned unionSlot = filterPredUnionSlotForQueryIndices(queryIndices);
         ensureFilterPredUnionColumn(plan, unionSlot, synthetic.getContext());
      }
      if (plan.payloadColumns.empty()) continue;

	      auto reuseSynthetic = collectModuleReuseInfo(synthetic);
	      applyUnionPlanToSyntheticHiv(synthetic, t.state, plan, reuseSynthetic,
	                                   reuseSlotForEntry(*donor), queries[donor->query],
	                                   hivDonor, reuseDonor, reuseSlotForEntry(*firstPeer),
	                                   queries[firstPeer->query], hivPeer, reusePeer,
	                                   extraPeerHivs, extraPeerReuses, extraPeerReuseSlots);

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
	      applyUnionPlanToSyntheticHiv(synthetic, synthHiv, plan, reuseSynthetic,
	                                   static_cast<unsigned>(m.queryA), query0, hivA, reuse0,
	                                   static_cast<unsigned>(m.queryB), query1, hivB, reuse1);

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

static llvm::SmallVector<unsigned, 8> sortedFilterPredSlotsInMembers(mlir::MLIRContext* ctx,
                                                                     subop::StateMembersAttr members) {
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   llvm::SmallVector<unsigned, 8> slots;
   for (subop::Member member : members.getMembers()) {
      if (auto slot = parseFilterPredMemberSlot(mm.getName(member))) slots.push_back(*slot);
   }
   sortUniqueSlots(slots);
   return slots;
}

static llvm::SmallVector<unsigned, 8> localFilterPredSlotsInMixedMembers(mlir::MLIRContext* ctx,
                                                                         subop::StateMembersAttr members) {
   llvm::SmallVector<unsigned, 8> slots = sortedFilterPredSlotsInMembers(ctx, members);
   assert(!slots.empty() && "mixed HIV must expose at least one filter_pred slot");
   slots.pop_back();
   return slots;
}

static llvm::SmallVector<unsigned, 8> inheritedMixedPredSlotsInBuildStep(subop::ExecutionStepOp buildStep) {
   llvm::DenseSet<unsigned> seen;
   llvm::SmallVector<unsigned, 8> slots;
   buildStep.walk([&](subop::ScanListOp scanList) {
      auto ler = mlir::dyn_cast<subop::LookupEntryRefType>(scanList.getElem().getColumn().type);
      if (!ler) return;
      auto mixed = mlir::dyn_cast<subop::MixedHashIndexedViewType>(ler.getState());
      if (!mixed) return;
      for (unsigned slot : localFilterPredSlotsInMixedMembers(buildStep.getContext(), mixed.getValueMembers())) {
         if (seen.insert(slot).second) slots.push_back(slot);
      }
   });
   llvm::sort(slots);
   return slots;
}

static llvm::SmallVector<uint64_t, 4> inheritedMixedCacheDepsInBuildStep(subop::ExecutionStepOp buildStep) {
   llvm::DenseSet<uint64_t> seen;
   llvm::SmallVector<uint64_t, 4> deps;
   auto recordCacheGetStep = [&](subop::ExecutionStepOp step) {
      step.walk([&](subop::CacheGetOp get) {
         auto layout = asLocalHivLayoutType(get.getResult().getType());
         if (!localHivLayoutHasFilterPredMember(buildStep.getContext(), layout)) return;
         uint64_t key = static_cast<uint64_t>(get.getKey());
         if (seen.insert(key).second) deps.push_back(key);
      });
   };
   for (mlir::Value operand : buildStep.getInputs()) {
      if (auto producer = operand.getDefiningOp<subop::ExecutionStepOp>()) {
         recordCacheGetStep(producer);
      }
   }
   return deps;
}

static llvm::SmallVector<unsigned, 8> consumerSlotsForCacheKey(
   uint64_t cacheKey,
   const llvm::DenseMap<uint64_t, llvm::DenseMap<unsigned, unsigned>>* consumerSlotByCacheKeyAndQuery) {
   llvm::SmallVector<unsigned, 8> slots;
   if (!consumerSlotByCacheKeyAndQuery) return slots;
   auto itSlotsByQuery = consumerSlotByCacheKeyAndQuery->find(cacheKey);
   if (itSlotsByQuery == consumerSlotByCacheKeyAndQuery->end()) return slots;
   for (const auto& [queryIdx, slot] : itSlotsByQuery->second) {
      (void)queryIdx;
      slots.push_back(slot);
   }
   sortUniqueSlots(slots);
   return slots;
}

static llvm::DenseSet<unsigned> filterPredSlotsInMembers(mlir::MLIRContext* ctx,
                                                         subop::StateMembersAttr members) {
   llvm::DenseSet<unsigned> slots;
   for (unsigned slot : sortedFilterPredSlotsInMembers(ctx, members)) slots.insert(slot);
   return slots;
}

static llvm::DenseSet<unsigned> filterPredSlotsInHivLikeType(mlir::Type type) {
   if (auto hiv = asHashIndexedViewLayoutType(type))
      return filterPredSlotsInMembers(type.getContext(), hiv.getValueMembers());
   return {};
}

static subop::StateMembersAttr joinBufferMembersWithFilterPredSlots(mlir::MLIRContext* ctx,
                                                                    subop::BufferType bufTy,
                                                                    llvm::ArrayRef<unsigned> predSlots,
                                                                    unsigned unionSlot) {
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   llvm::SmallVector<subop::Member> members(bufTy.getMembers().getMembers().begin(),
                                            bufTy.getMembers().getMembers().end());
   llvm::StringSet<> existing;
   for (subop::Member member : members) existing.insert(mm.getName(member));
   auto appendPredSlot = [&](unsigned slot) {
      subop::Member predMember = makeOrGetPredMemberForSlot(ctx, slot);
      if (existing.insert(mm.getName(predMember)).second) members.push_back(predMember);
   };
   for (unsigned slot : predSlots) appendPredSlot(slot);
   appendPredSlot(unionSlot);
   return subop::StateMembersAttr::get(ctx, members);
}

static llvm::SmallVector<unsigned, 8> filterPredSlotsForScanRefsBufferGroup(
   const CrossQueryStateMatchGroup& group,
   const llvm::DenseMap<uint64_t, llvm::DenseMap<unsigned, unsigned>>* consumerSlotByCacheKeyAndQuery) {
   llvm::DenseSet<unsigned> seen;
   llvm::SmallVector<unsigned, 8> slots;
   for (const CrossQueryStateMatchEntry& entry : group.entries) {
      unsigned slot = reuseSlotForEntry(entry);
      if (seen.insert(slot).second) slots.push_back(slot);
   }
   if (consumerSlotByCacheKeyAndQuery) {
      for (unsigned slot : sourceMixedPredSlotsForMappedTargetSlots(group,
                                                                    *consumerSlotByCacheKeyAndQuery)) {
         if (seen.insert(slot).second) slots.push_back(slot);
      }
   }
   llvm::sort(slots);
   return slots;
}

static subop::ExecutionStepOp findBufferBuildStepForTarget(mlir::Value buffer,
                                                           const ModuleReuseInfo& reuse) {
   if (subop::ExecutionStepOp step = findBufferBuildStepWithTableMaterialize(buffer, reuse)) return step;
   auto itW = findReuseMap(reuse.writerStepsByState, buffer);
   if (itW == reuse.writerStepsByState.end()) return {};
   for (subop::ExecutionStepOp step : itW->second) {
      subop::MaterializeOp mat;
      step.walk([&](subop::MaterializeOp op) {
         if (mat) return;
         if (materializeWritesExactState(op, buffer)) mat = op;
      });
      if (mat) return step;
   }
   return {};
}

void extendSyntheticScanRefsBuffersWithFilterPredsForGroups(
   mlir::ModuleOp synthetic, llvm::ArrayRef<CrossQueryStateMatchGroup> groups,
   llvm::ArrayRef<CacheTarget> targetsInSynthetic,
   const llvm::DenseMap<uint64_t, llvm::DenseMap<unsigned, unsigned>>* consumerSlotByCacheKeyAndQuery) {
   if (groups.empty() || targetsInSynthetic.empty()) return;

   llvm::DenseMap<uint64_t, const CrossQueryStateMatchGroup*> groupByKey;
   for (const CrossQueryStateMatchGroup& group : groups) {
      if (group.requiresBufferScanRefsUnion) groupByKey[group.cacheKey] = &group;
   }
   if (groupByKey.empty()) return;

   auto* ctx = synthetic.getContext();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();

   for (const CacheTarget& target : targetsInSynthetic) {
      auto itG = groupByKey.find(target.cacheKey);
      if (itG == groupByKey.end()) continue;
      if (!mlir::isa<subop::BufferType>(target.state.getType())) continue;

      ModuleReuseInfo reuseSynthetic = collectModuleReuseInfo(synthetic);
      subop::ExecutionStepOp buildStep = findBufferBuildStepForTarget(target.state, reuseSynthetic);
      assert(buildStep && "scan_refs buffer reuse requires a synthetic buffer materialize step");
      llvm::SmallVector<unsigned, 8> predSlots =
         filterPredSlotsForScanRefsBufferGroup(*itG->second, consumerSlotByCacheKeyAndQuery);
      if (predSlots.empty()) continue;
      unsigned unionSlot = filterPredUnionSlotForQueryIndices(predSlots);
      subop::BufferType bufTy = mlir::cast<subop::BufferType>(target.state.getType());
      subop::StateMembersAttr targetMembers =
         joinBufferMembersWithFilterPredSlots(ctx, bufTy, predSlots, unionSlot);

      applyBufferLayoutToSsaClosure(synthetic, {target.state}, targetMembers, reuseSynthetic);
      syncMaterializeMappingsToActualBufferTypes(synthetic);
      synchronizeExecutionStepPortTypes(synthetic, nullptr);

      reuseSynthetic = collectModuleReuseInfo(synthetic);
      buildStep = findBufferBuildStepForTarget(target.state, reuseSynthetic);
      assert(buildStep && "scan_refs buffer reuse requires a synthetic buffer materialize step");
      subop::MaterializeOp mat = findJoinBufferMaterializeInStep(buildStep);
      assert(mat && "scan_refs buffer reuse requires buffer materialize");

      llvm::SmallVector<tuples::ColumnRefAttr, 8> predRefs;
      for (unsigned slot : predSlots) {
         std::string predName = ("filter_pred$" + llvm::Twine(slot)).str();
         bool materialized = false;
         if (consumerSlotByCacheKeyAndQuery) {
            llvm::SmallVector<tuples::ColumnRefAttr, 4> upstreamRefs =
               threadMappedUpstreamMixedPredColumnsToMaterializeStream(
                  buildStep, slot, target.cacheKey, *consumerSlotByCacheKeyAndQuery);
            if (!upstreamRefs.empty()) {
               tuples::ColumnRefAttr predRef =
                  insertAndPredsBeforeMaterialize(mat, upstreamRefs,
                                                  ("reuse_upstream_pred$" + llvm::Twine(slot)).str());
               subop::Member predMember;
               for (subop::Member member : getInnerBufferTypeForMaterializeState(mat.getState().getType()).getMembers().getMembers()) {
                  if (mm.getName(member) == predName) {
                     predMember = member;
                     break;
                  }
               }
               assert(predMember && "scan_refs buffer predicate member must exist in synthetic layout");
               llvm::SmallVector<subop::RefMappingPairT> pairs;
               for (auto& pr : mat.getMapping().getMapping()) {
                  if (pr.first != predMember) pairs.push_back(pr);
               }
               pairs.push_back({predMember, predRef});
               mat.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, pairs));
               materialized = true;
            }
         }
         if (!materialized && consumerSlotByCacheKeyAndQuery) {
            llvm::SmallVector<tuples::ColumnRefAttr, 4> upstreamRefs =
               threadSourceMixedPredColumnToMaterializeStream(buildStep, slot);
            if (!upstreamRefs.empty()) {
               tuples::ColumnRefAttr predRef =
                  insertAndPredsBeforeMaterialize(mat, upstreamRefs,
                                                  ("reuse_source_pred$" + llvm::Twine(slot)).str());
               subop::Member predMember;
               for (subop::Member member :
                    getInnerBufferTypeForMaterializeState(mat.getState().getType()).getMembers().getMembers()) {
                  if (mm.getName(member) == predName) {
                     predMember = member;
                     break;
                  }
               }
               assert(predMember && "scan_refs buffer source predicate member must exist in synthetic layout");
               llvm::SmallVector<subop::RefMappingPairT> pairs;
               for (auto& pr : mat.getMapping().getMapping()) {
                  if (pr.first != predMember) pairs.push_back(pr);
               }
               pairs.push_back({predMember, predRef});
               mat.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(ctx, pairs));
               materialized = true;
            }
         }
         if (!materialized && !materializePredMemberFromUpstreamMixedScanList(buildStep, predName))
            materializeConstantTruePredMemberOnBufferMaterialize(mat, predName, /*updateStreamOperand=*/true);
         for (auto& [member, ref] : mat.getMapping().getMapping()) {
            if (mm.getName(member) == predName) predRefs.push_back(ref);
         }
      }
      if (!predRefs.empty()) {
         insertResidualFilterUnionAfterPredicates(mat.getStream(), predRefs);
      }
      std::string unionPredName = ("filter_pred$" + llvm::Twine(unionSlot)).str();
      // The synthetic stream was filtered by the slot predicates above; this marker is true only for kept rows.
      materializeConstantTruePredMemberOnBufferMaterialize(mat, unionPredName, /*updateStreamOperand=*/true);

      syncMaterializeMappingsToActualBufferTypes(synthetic);
      synchronizeExecutionStepPortTypes(synthetic, nullptr);
   }
}

void alignConsumerScanRefsBufferCacheGetWithFilterPreds(
   mlir::ModuleOp consumer, uint64_t cacheKey, subop::StateMembersAttr producerMembers,
   llvm::ArrayRef<unsigned> predSlots,
   unsigned consumerReuseSlot) {
   if (predSlots.empty()) return;
   assert(producerMembers && "scan_refs buffer cache_get alignment requires producer layout");
   auto* ctx = consumer.getContext();
   unsigned unionSlot = filterPredUnionSlotForQueryIndices(predSlots);
   subop::Member predMember = makeOrGetPredMemberForSlot(ctx, consumerReuseSlot);

   consumer.walk([&](subop::CacheGetOp get) {
      if (static_cast<uint64_t>(get.getKey()) != cacheKey) return;
      auto bufTy = mlir::dyn_cast<subop::BufferType>(get.getResult().getType());
      if (!bufTy) return;
      subop::StateMembersAttr targetMembers =
         joinBufferMembersWithFilterPredSlots(ctx, subop::BufferType::get(ctx, producerMembers), predSlots, unionSlot);
      llvm::DenseMap<subop::Member, subop::Member> memberRemap;
      llvm::ArrayRef<subop::Member> oldMembers = bufTy.getMembers().getMembers();
      llvm::ArrayRef<subop::Member> newMembers = targetMembers.getMembers();
      assert(oldMembers.size() <= newMembers.size() &&
             "scan_refs buffer consumer layout may only append predicate members");
      for (size_t i = 0; i < oldMembers.size(); ++i) memberRemap[oldMembers[i]] = newMembers[i];
      ModuleReuseInfo reuse = collectModuleReuseInfo(consumer);
      applyBufferLayoutToSsaClosure(consumer, {get.getResult()}, targetMembers, reuse);
      syncMaterializeMappingsToActualBufferTypes(consumer);
      synchronizeExecutionStepPortTypes(consumer, nullptr);

      mlir::Value root = get.getResult();
      consumer.walk([&](subop::ScanRefsOp scan) {
         mlir::Value scanned = peelBlockArgsToEnclosingOperands(scan.getState());
         if (canonicalizeStateValueForReuse(scanned) != canonicalizeStateValueForReuse(root)) return;
         auto scanBufTy = mlir::dyn_cast<subop::BufferType>(scan.getState().getType());
         if (!scanBufTy) return;
         auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
         auto ref = scan.getRef();
         ref.getColumn().type = subop::EntryRefType::get(ctx, mlir::cast<subop::State>(scanBufTy));
         scan.setRefAttr(ref);
         tuples::ColumnRefAttr entryRef = cm.createRef(&scan.getRef().getColumn());
         insertProbePredFilterImmediatelyAfterScanProducer(scan.getOperation(), scan.getRes(), entryRef,
                                                           predMember);
      });
      consumer.walk([&](subop::GatherOp gather) {
         auto refTy = mlir::dyn_cast<subop::EntryRefType>(gather.getRef().getColumn().type);
         if (!refTy || refTy.getState() != subop::BufferType::get(ctx, targetMembers)) return;
         llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> pairs;
         bool changed = false;
         for (auto [member, col] : gather.getMapping().getMapping()) {
            subop::Member outMember = member;
            if (auto it = memberRemap.find(member); it != memberRemap.end()) {
               outMember = it->second;
               changed |= outMember != member;
            }
            pairs.push_back({outMember, col});
         }
         if (changed) gather.setMappingAttr(subop::ColumnDefMemberMappingAttr::get(ctx, pairs));
      });
   });
   synchronizeExecutionStepPortTypes(consumer, nullptr);
}

void extendSyntheticJoinBuffersWithInheritedMixedPreds(
   mlir::ModuleOp synthetic, llvm::ArrayRef<CacheTarget> targetsInSynthetic,
   CachedJoinBufferLayoutsByKey* outLayouts,
   llvm::DenseMap<uint64_t, llvm::SmallVector<uint64_t, 4>>* outInheritedDepsByCacheKey,
   llvm::DenseMap<uint64_t, llvm::DenseMap<unsigned, unsigned>>* consumerSlotByCacheKeyAndQuery) {
   if (targetsInSynthetic.empty()) return;
   ModuleReuseInfo reuseSynthetic = collectModuleReuseInfo(synthetic);
   auto& mm = synthetic.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   llvm::DenseMap<uint64_t, llvm::SmallVector<uint64_t, 4>> inheritedDepsByTargetKey;
   if (consumerSlotByCacheKeyAndQuery) {
      for (const CacheTarget& target : targetsInSynthetic) {
         mlir::Value targetState = target.state;
         synthetic.walk([&](subop::CachePutOp put) {
            if (static_cast<uint64_t>(put.getKey()) == target.cacheKey) targetState = put.getState();
         });
         if (!targetState || !asHashIndexedViewLayoutType(targetState.getType())) continue;
         subop::ExecutionStepOp buildStep = findStrictJoinBufferBuildStepForHiv(synthetic, targetState, reuseSynthetic);
         if (!buildStep) continue;
         llvm::SmallVector<uint64_t, 4> deps = inheritedMixedCacheDepsInBuildStep(buildStep);
         if (!deps.empty()) inheritedDepsByTargetKey[target.cacheKey] = deps;
      }
      for (unsigned iter = 0; iter < 8; ++iter) {
         for (const auto& [targetKey, deps] : inheritedDepsByTargetKey)
            inheritSlotMapForSyntheticTarget(targetKey, deps, *consumerSlotByCacheKeyAndQuery);
      }
   }
   for (const CacheTarget& target : targetsInSynthetic) {
      mlir::Value targetState = target.state;
      synthetic.walk([&](subop::CachePutOp put) {
         if (static_cast<uint64_t>(put.getKey()) == target.cacheKey) targetState = put.getState();
      });
      if (!targetState || !asHashIndexedViewLayoutType(targetState.getType())) continue;
      subop::ExecutionStepOp buildStep = findStrictJoinBufferBuildStepForHiv(synthetic, targetState, reuseSynthetic);
      if (!buildStep) continue;
      llvm::SmallVector<uint64_t, 4> inheritedDeps;
      if (auto itDeps = inheritedDepsByTargetKey.find(target.cacheKey); itDeps != inheritedDepsByTargetKey.end())
         inheritedDeps = itDeps->second;
      else
         inheritedDeps = inheritedMixedCacheDepsInBuildStep(buildStep);
      if (inheritedDeps.empty()) continue;
      if (consumerSlotByCacheKeyAndQuery)
         inheritSlotMapForSyntheticTarget(target.cacheKey, inheritedDeps, *consumerSlotByCacheKeyAndQuery);
      llvm::SmallVector<unsigned, 8> predSlots = inheritedMixedPredSlotsInBuildStep(buildStep);
      llvm::SmallVector<unsigned, 8> consumerSlots =
         consumerSlotsForCacheKey(target.cacheKey, consumerSlotByCacheKeyAndQuery);
      if (!consumerSlots.empty()) {
         predSlots = std::move(consumerSlots);
      } else if (auto mixedTarget = mlir::dyn_cast<subop::MixedHashIndexedViewType>(targetState.getType())) {
            llvm::SmallVector<unsigned, 8> targetSlots =
               localFilterPredSlotsInMixedMembers(synthetic.getContext(), mixedTarget.getValueMembers());
            if (!targetSlots.empty()) predSlots = std::move(targetSlots);
      }
      if (predSlots.empty()) continue;
      sortUniqueSlots(predSlots);
      unsigned unionSlot = filterPredUnionSlotForQueryIndices(predSlots);
      if (outInheritedDepsByCacheKey) (*outInheritedDepsByCacheKey)[target.cacheKey] = inheritedDeps;
      subop::MaterializeOp mat = findJoinBufferMaterializeInStep(buildStep);
      assert(mat && "inherited mixed pred target must materialize a join buffer");
      auto& cm = synthetic.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      bool layoutAlreadyHasPreds = false;
      bool layoutContainsRequiredPreds = false;
      if (outLayouts) {
         auto itExistingLayout = outLayouts->find(target.cacheKey);
         layoutAlreadyHasPreds =
            itExistingLayout != outLayouts->end() &&
            llvm::any_of(itExistingLayout->second.payloadMembers, [&](subop::Member member) {
               return parseFilterPredMemberSlot(mm.getName(member)).has_value();
            });
         if (itExistingLayout != outLayouts->end()) {
            llvm::DenseSet<unsigned> layoutSlots;
            for (subop::Member member : itExistingLayout->second.payloadMembers) {
               if (auto slot = parseFilterPredMemberSlot(mm.getName(member))) layoutSlots.insert(*slot);
            }
            layoutContainsRequiredPreds = layoutSlots.contains(unionSlot);
            for (unsigned slot : predSlots)
               layoutContainsRequiredPreds = layoutContainsRequiredPreds && layoutSlots.contains(slot);
         }
      }
      if (!layoutContainsRequiredPreds) {
         llvm::DenseSet<unsigned> typeSlots = filterPredSlotsInHivLikeType(targetState.getType());
         layoutContainsRequiredPreds = typeSlots.contains(unionSlot);
         for (unsigned slot : predSlots)
            layoutContainsRequiredPreds = layoutContainsRequiredPreds && typeSlots.contains(slot);
      }

      if (!layoutAlreadyHasPreds || !layoutContainsRequiredPreds) {
         subop::BufferType bufTy = getInnerBufferTypeForMaterializeState(mat.getState().getType());
         assert(bufTy && "inherited mixed pred target must materialize a join buffer");
         mlir::Value mergedBuffer = resolveJoinMergedBuffer(targetState, synthetic, reuseSynthetic);
         subop::StateMembersAttr targetMembers =
            joinBufferMembersWithFilterPredSlots(synthetic.getContext(), bufTy, predSlots, unionSlot);
         applyBufferLayoutToSsaClosure(synthetic, {mergedBuffer}, targetMembers, reuseSynthetic);
         auto targetBufTy = subop::BufferType::get(synthetic.getContext(), targetMembers);
         auto targetTlTy = subop::ThreadLocalType::get(synthetic.getContext(), mlir::cast<subop::State>(targetBufTy));
         if (mlir::isa<subop::BufferType>(mat.getState().getType())) {
            mat.getState().setType(targetBufTy);
         } else if (auto tl = mlir::dyn_cast<subop::ThreadLocalType>(mat.getState().getType())) {
            if (mlir::isa<subop::BufferType>(tl.getWrapped())) mat.getState().setType(targetTlTy);
         }
         syncMaterializeMappingsToActualBufferTypes(synthetic);
         synchronizeExecutionStepPortTypes(synthetic, nullptr);
         mat = findJoinBufferMaterializeInStep(buildStep);
         assert(mat && "inherited mixed pred target must still materialize a join buffer");
      }

      llvm::SmallVector<tuples::ColumnRefAttr, 8> inheritedPredRefs;
      for (unsigned slot : predSlots) {
         std::string predName = ("filter_pred$" + llvm::Twine(slot)).str();
         subop::Member predMember;
         subop::BufferType bufTy = getInnerBufferTypeForMaterializeState(mat.getState().getType());
         assert(bufTy && "inherited mixed pred target must materialize a join buffer");
         for (subop::Member member : bufTy.getMembers().getMembers()) {
            if (mm.getName(member) == predName) {
               predMember = member;
               break;
            }
         }
         assert(predMember && "inherited mixed pred target buffer must contain predicate member");

         if (tuples::ColumnRefAttr existing = materializedColumnForMember(mat, predMember)) {
            if (isReuseCombinedPredicateColumn(existing, cm)) {
               inheritedPredRefs.push_back(existing);
               continue;
            }
         }

         llvm::SmallVector<tuples::ColumnRefAttr, 4> upstreamRefs;
         if (consumerSlotByCacheKeyAndQuery) {
            upstreamRefs = threadMappedUpstreamMixedPredColumnsToMaterializeStream(
               buildStep, slot, target.cacheKey, *consumerSlotByCacheKeyAndQuery);
         }
         assert(consumerSlotByCacheKeyAndQuery &&
                "inherited mixed pred rewrite requires explicit slot mapping");
         llvm::SmallVector<tuples::ColumnRefAttr, 4> refs;
         if (tuples::ColumnRefAttr existing = materializedColumnForMember(mat, predMember)) {
            if (upstreamRefs.empty() || !isReuseGeneratedUpstreamPredicateColumn(existing, cm))
               refs.push_back(existing);
         }
         refs.append(upstreamRefs.begin(), upstreamRefs.end());
         if (refs.empty()) {
            llvm_unreachable("inherited mixed pred target must materialize upstream predicate");
         }
         tuples::ColumnRefAttr combined =
            insertAndPredsBeforeMaterialize(mat, refs,
                                            ("reuse_inherited_combined_pred$" + llvm::Twine(slot)).str());
         setMaterializeMapping(mat, predMember, combined);
         inheritedPredRefs.push_back(combined);
      }
      if (!inheritedPredRefs.empty()) {
         removeTrailingGeneratedUpstreamOnlyUnionFilter(mat);
         insertResidualFilterUnionAfterPredicates(mat.getStream(), inheritedPredRefs);
      }
      std::string unionPredName = ("filter_pred$" + llvm::Twine(unionSlot)).str();
      materializeConstantTruePredMemberOnBufferMaterialize(mat, unionPredName, /*updateStreamOperand=*/true);

      syncMaterializeMappingsToActualBufferTypes(synthetic);
      syncMixedProbeCarriers(synthetic, nullptr);
      synchronizeExecutionStepPortTypes(synthetic, nullptr);

      ModuleReuseInfo afterPreds = collectModuleReuseInfo(synthetic);
      mlir::Value producer = findHashIndexedViewBuiltByStep(synthetic, buildStep, afterPreds);
      assert(producer && "inherited mixed pred target must still produce a hash indexed view");
      if (outLayouts) {
         const CachedJoinBufferLayout* previousLayout = nullptr;
         if (auto itPrev = outLayouts->find(target.cacheKey); itPrev != outLayouts->end())
            previousLayout = &itPrev->second;
         (*outLayouts)[target.cacheKey] =
            layoutFromMaterializeMapping(asHashIndexedViewLayoutType(producer.getType()), mat, previousLayout);
      }
   }
   syncMaterializeMappingsToActualBufferTypes(synthetic);
   syncMixedProbeCarriers(synthetic, nullptr);
   synchronizeExecutionStepPortTypes(synthetic, nullptr);
}

void extendSyntheticAggregateHashTablesToPayloadUnion(mlir::ModuleOp synthetic, mlir::ModuleOp query0,
                                                      mlir::ModuleOp query1,
                                                      llvm::ArrayRef<CrossQueryStateMatchPair> matches,
                                                      llvm::ArrayRef<CacheTarget> targetsInSynthetic,
                                                      const mlir::IRMapping& donorToSynthetic,
                                                      CachedAggregateLayoutsByKey* outLayouts) {
   (void)donorToSynthetic;
   if (matches.empty() || targetsInSynthetic.empty()) return;

   llvm::SmallVector<mlir::ModuleOp, 2> queries{query0, query1};
   llvm::SmallVector<CrossQueryStateMatchGroup, 8> groups;
   for (const CrossQueryStateMatchPair& m : matches) {
      if (!m.stateA || !m.stateB) continue;
      CrossQueryStateMatchGroup g;
      g.cacheKey = m.cacheKey;
      g.enableFilterPredReuse = m.enableFilterPredReuse;
      g.entries.push_back(CrossQueryStateMatchEntry{0, m.stateA});
      g.entries.push_back(CrossQueryStateMatchEntry{1, m.stateB});
      groups.push_back(std::move(g));
   }
   extendSyntheticAggregateHashTablesToPayloadUnionForGroups(synthetic, queries, groups, targetsInSynthetic,
                                                             outLayouts);
}

void extendSyntheticAggregateHashTablesToPayloadUnionForGroups(
   mlir::ModuleOp synthetic, llvm::ArrayRef<mlir::ModuleOp> queries,
   llvm::ArrayRef<CrossQueryStateMatchGroup> groups,
   llvm::ArrayRef<CacheTarget> targetsInSynthetic,
   CachedAggregateLayoutsByKey* outLayouts) {
   if (groups.empty() || targetsInSynthetic.empty()) return;

   llvm::DenseMap<uint64_t, const CrossQueryStateMatchGroup*> groupByKey;
   for (const CrossQueryStateMatchGroup& g : groups) groupByKey[g.cacheKey] = &g;

   for (const CacheTarget& t : targetsInSynthetic) {
      auto itG = groupByKey.find(t.cacheKey);
      if (itG == groupByKey.end()) continue;
      const CrossQueryStateMatchGroup& group = *itG->second;
      if (group.requiresSplitMaterialize) continue;
      if (!mlir::isa<subop::PreAggrHtType>(t.state.getType())) continue;
      assert(group.entries.size() >= 2 && "aggregate union group must have at least two entries");
      llvm::SmallVector<AggregateUnionGroupInput, 8> groupInputs =
         collectAggregateUnionGroupInputs(group, queries);
      std::optional<AggregateUnionRewritePlan> plan =
         buildAggregateUnionRewritePlan(synthetic, t.state, group, groupInputs);
      if (!plan) continue;
      applySyntheticAggregatePayloadUnion(synthetic, t.state, plan->layout, plan->peers,
                                          plan->reuseSynthetic, plan->donorFilterSlot);
      if (outLayouts) (*outLayouts)[t.cacheKey] = std::move(plan->layout);
   }
}

void alignConsumerModulesToCachedAggregateLayout(mlir::ModuleOp consumer, const CachedAggregateLayout& layout,
                                                 std::optional<uint64_t> cacheKey,
                                                 std::optional<unsigned> consumerReuseQueryIndex,
                                                 std::optional<unsigned> consumerReuseFilterSlot) {
   if (!consumerReuseQueryIndex) return;
   if (!consumerReuseFilterSlot) consumerReuseFilterSlot = consumerReuseQueryIndex;
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
         if (layout.mixedByQueryId) insertAggregateQueryIdConsumerFilter(scan, alignment.queryIdMember,
                                                                         *consumerReuseFilterSlot);
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
   consumer.walk([&](subop::CacheGetOp get) {
      if (cacheKey && static_cast<uint64_t>(get.getKey()) != *cacheKey) return;
      auto canonicalHiv = asHashIndexedViewLayoutType(get.getResult().getType());
      if (!canonicalHiv) return;

      ConsumerCachedHivSites sites;
      // Only rewrite carriers that already embed this cache_get HIV (not other local join buffers).
      CachedJoinBufferLayout emptyLayout;
      emptyLayout.producerHiv = canonicalHiv;
      traverseConsumerHivUsesFromRoot(get.getResult(), canonicalHiv, canonicalHiv, emptyLayout, sites,
                                      /*dbg=*/nullptr);
      expandConsumerProbeClosureThroughPorts(consumer, sites.ssaClosure);
      synchronizeExecutionStepPortTypes(consumer, &sites.ssaClosure);
      syncMixedProbeCarriers(consumer, nullptr);
      synchronizeExecutionStepPortTypes(consumer, nullptr);
   });
}

void setMixedLookupPredSlotForCacheGet(mlir::ModuleOp module, uint64_t cacheKey, unsigned slot) {
   (void)module;
   (void)cacheKey;
   (void)slot;
   // Predicate selection is encoded by retagging lookup-entry/list carriers to
   // MixedHIV<..., filter_pred$slot>. Lookup keys remain the real hash keys;
   // scan_list lowering reads the stored predicate from each hash-table entry.
}

void applyProbePredFiltersForConsumerClosures(mlir::ModuleOp consumer,
                                              llvm::MutableArrayRef<ConsumerCacheGetProbeClosure> probeClosures) {
   auto* ctx = consumer.getContext();
   for (ConsumerCacheGetProbeClosure& probe : probeClosures) {
      if (!probe.alignedHiv) continue;
      subop::Member predMember;
      if (probe.consumerReuseQueryIndex) {
         assert(probe.cacheKey && "mixed HIV probe predicate slot rewrite requires a cache_get key");
         predMember = makeOrGetPredMemberForSlot(ctx, *probe.consumerReuseQueryIndex);
         auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
         if (!valueMembersContainMemberNamed(ctx, probe.alignedHiv.getValueMembers(), mm.getName(predMember)))
            continue;
         setMixedLookupPredSlotForCacheGet(consumer, *probe.cacheKey, *probe.consumerReuseQueryIndex);
         probe.cacheGetRoot.setType(mixedHivTypeForPredMember(ctx, probe.alignedHiv, predMember));
      } else if (auto found = findFilterPredMemberOnHashIndexedView(probe.alignedHiv)) {
         predMember = *found;
      } else {
         continue;
      }

      refreshProbeClosureFromCacheGetRoot(consumer, probe);
      alignScanListForProbeClosure(consumer, probe, "probe_pred_align");

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
   syncMixedProbeCarriers(consumer, nullptr);
   for (ConsumerCacheGetProbeClosure& probe : probeClosures) {
      if (!probe.alignedHiv) continue;
      remapAlignedHivClosureGathers(consumer, nullptr, probe.alignedHiv, probe.consumerLayout,
                                    nullptr, probe.cacheKey, "probe_pred_remap_global");
   }
   synchronizeExecutionStepPortTypes(consumer, nullptr);
}

void refreshCachedJoinLayoutsFromSyntheticCachePuts(mlir::ModuleOp synthetic,
                                                    llvm::ArrayRef<CacheTarget> targetsInSynthetic,
                                                    CachedJoinBufferLayoutsByKey& layoutsByKey) {
   for (const CacheTarget& t : targetsInSynthetic) {
      auto itPrevLayout = layoutsByKey.find(t.cacheKey);
      if (itPrevLayout == layoutsByKey.end()) continue;
      synthetic.walk([&](subop::CachePutOp put) {
         if (static_cast<uint64_t>(put.getKey()) != t.cacheKey) return;
         auto hivTy = asHashIndexedViewLayoutType(put.getState().getType());
         if (!hivTy) return;
         const CachedJoinBufferLayout* prev = &itPrevLayout->second;
         layoutsByKey[t.cacheKey] = layoutInProducerValueOrder(*prev, hivTy);
      });
   }
}

} // namespace lingodb::compiler::dialect::subop
