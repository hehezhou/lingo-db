#include "lingodb/compiler/Dialect/SubOperator/Transforms/CrossQueryStateReuse.h"

#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsAttributes.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateExtraction.h"
#include "lingodb/compiler/Dialect/SubOperator/Utils.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstdlib>
#include <optional>
#include <string>

namespace lingodb::compiler::dialect::subop {
namespace {

static subop::Member makeOrGetPredMember(mlir::MLIRContext* ctx) {
   auto* d = ctx->getLoadedDialect<subop::SubOperatorDialect>();
   assert(d && "SubOperatorDialect must be loaded");
   auto& mm = d->getMemberManager();
   // Stable name. If it already exists, type must match.
   return mm.createMemberDirect("filter_pred$0", mlir::IntegerType::get(ctx, 1));
}

static subop::StateMembersAttr appendMember(mlir::MLIRContext* ctx, subop::StateMembersAttr members, subop::Member m) {
   llvm::SmallVector<subop::Member> ms;
   for (auto x : members.getMembers()) ms.push_back(x);
   ms.push_back(m);
   return subop::StateMembersAttr::get(ctx, std::move(ms));
}

static bool valueMembersContainMemberNamed(mlir::MLIRContext* ctx, subop::StateMembersAttr members,
                                           llvm::StringRef name) {
   auto* d = ctx->getLoadedDialect<subop::SubOperatorDialect>();
   assert(d);
   auto& mm = d->getMemberManager();
   for (auto m : members.getMembers()) {
      if (mm.getName(m) == name) return true;
   }
   return false;
}

static mlir::Type substituteJoinBufferTypes(mlir::Type t,
                                            const llvm::DenseMap<mlir::Type, mlir::Type>& bufOldToNew) {
   auto* ctx = t.getContext();
   if (auto it = bufOldToNew.find(t); it != bufOldToNew.end()) return it->second;
   if (auto tl = mlir::dyn_cast<subop::ThreadLocalType>(t)) {
      mlir::Type inner = substituteJoinBufferTypes(tl.getWrapped(), bufOldToNew);
      if (inner != tl.getWrapped()) return subop::ThreadLocalType::get(ctx, mlir::cast<subop::State>(inner));
   }
   return t;
}

static void syncCreateHashIndexedViewTypesFromBuffers(mlir::ModuleOp module) {
   auto* ctx = module.getContext();
   module.walk([&](subop::CreateHashIndexedView chiv) {
      auto bufTy = mlir::dyn_cast<subop::BufferType>(chiv.getSource().getType());
      if (!bufTy) return;
      if (!valueMembersContainMemberNamed(ctx, bufTy.getMembers(), "filter_pred$0")) return;
      subop::Member linkM = chiv.getLinkMember().getMember();
      subop::Member hashM = chiv.getHashMember().getMember();
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
   });
}

static void patchLookupRefsForHashIndexedViewStates(mlir::ModuleOp module) {
   auto* ctx = module.getContext();
   module.walk([&](subop::LookupOp lookup) {
      auto stTy = lookup.getState().getType();
      if (!mlir::isa<subop::HashIndexedViewType>(stTy)) return;
      auto hiv = mlir::cast<subop::HashIndexedViewType>(stTy);
      auto refDef = lookup.getRef();
      refDef.getColumn().type = subop::LookupEntryRefType::get(ctx, hiv);
      lookup.setRefAttr(refDef);
   });
}

/// `create_hash_indexed_view` updates its result type; align `execution_step` SSA result types.
static void syncExecutionStepResultTypesFromReturns(mlir::ModuleOp module) {
   module.walk([&](subop::ExecutionStepReturnOp ret) {
      auto* parent = ret->getParentOp();
      auto step = mlir::dyn_cast<subop::ExecutionStepOp>(parent);
      if (!step || step.getNumResults() != ret.getNumOperands()) return;
      for (unsigned i = 0; i < step.getNumResults(); ++i) {
         mlir::Type t = ret.getOperand(i).getType();
         if (t != step.getResult(i).getType()) step.getResult(i).setType(t);
      }
   });
}

static void syncExecutionStepBlockArgTypesFromOperands(mlir::ModuleOp module) {
   module.walk([&](subop::ExecutionStepOp step) {
      mlir::Block& body = step.getSubOps().front();
      for (unsigned i = 0; i < step.getNumOperands(); ++i) {
         if (i >= body.getNumArguments()) break;
         mlir::Type wt = step.getOperand(i).getType();
         if (body.getArgument(i).getType() != wt) body.getArgument(i).setType(wt);
      }
   });
}

/// Join pipelines use `!subop.hashmap<...>` (often under `thread_local`). Extend value members with
/// `filter_pred$0 : i1` so table scan filters can be materialized once and re-checked on `scan_refs`.
static mlir::Type extendHashMapTypeWithPred(mlir::Type t, subop::Member predMember) {
   auto* ctx = t.getContext();
   if (auto tl = mlir::dyn_cast<subop::ThreadLocalType>(t)) {
      mlir::Type w = tl.getWrapped();
      mlir::Type nw = extendHashMapTypeWithPred(w, predMember);
      if (nw == w) return t;
      return subop::ThreadLocalType::get(ctx, mlir::cast<subop::State>(nw));
   }
   if (auto hm = mlir::dyn_cast<subop::HashMapType>(t)) {
      if (valueMembersContainMemberNamed(ctx, hm.getValueMembers(), "filter_pred$0")) return t;
      auto newVals = appendMember(ctx, hm.getValueMembers(), predMember);
      return subop::HashMapType::get(ctx, hm.getKeyMembers(), newVals, hm.getWithLock());
   }
   return t;
}

static constexpr llvm::StringLiteral kHashmapLookupInitPredExtendedAttr = "lingo.hashmap_lookup_init_pred_extended";
static constexpr llvm::StringLiteral kHashmapMergeCombinePredExtendedAttr = "lingo.hashmap_merge_combine_pred_extended";

/// Extend join `hashmap` types globally, and extend matching `buffer` / `thread_local<buffer>` only for
/// states targeted for cross-query reuse (`extendJoinBufferStates`), then align `hash_indexed_view` with buffers.
static void rewriteHashmapTypesInModule(mlir::ModuleOp module,
                                        llvm::ArrayRef<mlir::Value> extendJoinBufferStates = {}) {
   auto* ctx = module.getContext();
   subop::Member predMember = makeOrGetPredMember(ctx);

   llvm::DenseMap<mlir::Type, mlir::Type> bufOldToNew;
   for (mlir::Value v : extendJoinBufferStates) {
      auto bt = mlir::dyn_cast<subop::BufferType>(v.getType());
      if (!bt) continue;
      if (valueMembersContainMemberNamed(ctx, bt.getMembers(), "filter_pred$0")) continue;
      bufOldToNew[bt] = subop::BufferType::get(ctx, appendMember(ctx, bt.getMembers(), predMember));
   }

   module.walk([&](mlir::Operation* op) {
      for (mlir::OpResult r : op->getResults()) {
         mlir::Type t = r.getType();
         mlir::Type nt = extendHashMapTypeWithPred(t, predMember);
         if (!bufOldToNew.empty()) nt = substituteJoinBufferTypes(nt, bufOldToNew);
         if (nt != t) r.setType(nt);
      }
      for (mlir::Region& reg : op->getRegions()) {
         for (mlir::Block& b : reg) {
            for (mlir::BlockArgument a : b.getArguments()) {
               mlir::Type t = a.getType();
               mlir::Type nt = extendHashMapTypeWithPred(t, predMember);
               if (!bufOldToNew.empty()) nt = substituteJoinBufferTypes(nt, bufOldToNew);
               if (nt != t) a.setType(nt);
            }
         }
      }
   });

   if (!bufOldToNew.empty()) {
      syncCreateHashIndexedViewTypesFromBuffers(module);
      for (unsigned iter = 0; iter < 8; ++iter) {
         syncExecutionStepResultTypesFromReturns(module);
         syncExecutionStepBlockArgTypesFromOperands(module);
      }
      patchLookupRefsForHashIndexedViewStates(module);
   }

   module.walk([&](subop::ScanRefsOp scan) {
      if (!mlir::isa<subop::HashMapType>(scan.getState().getType())) return;
      auto hmTy = mlir::cast<subop::HashMapType>(scan.getState().getType());
      auto newRefTy = subop::HashMapEntryRefType::get(ctx, hmTy);
      auto refDef = scan.getRef();
      refDef.getColumn().type = newRefTy;
      scan.setRefAttr(refDef);
   });
   module.walk([&](subop::LookupOrInsertOp loi) {
      if (!mlir::isa<subop::HashMapType>(loi.getState().getType())) return;
      auto hmTy = mlir::cast<subop::HashMapType>(loi.getState().getType());
      auto newRefTy = subop::LookupEntryRefType::get(ctx, hmTy);
      auto refDef = loi.getRef();
      refDef.getColumn().type = newRefTy;
      loi.setRefAttr(refDef);
   });

   module.walk([&](subop::LookupOrInsertOp loi) {
      if (!mlir::isa<subop::HashMapType>(loi.getState().getType())) return;
      if (loi->hasAttr(kHashmapLookupInitPredExtendedAttr)) return;
      assert(!loi.getInitFn().empty());
      mlir::Block& b = loi.getInitFn().front();
      auto ret = mlir::cast<tuples::ReturnOp>(b.getTerminator());
      mlir::OpBuilder bb(ret);
      mlir::Value t = bb.create<mlir::arith::ConstantIntOp>(loi.getLoc(), 1, 1);
      llvm::SmallVector<mlir::Value, 32> outs(ret.getOperands().begin(), ret.getOperands().end());
      outs.push_back(t);
      bb.create<tuples::ReturnOp>(loi.getLoc(), outs);
      ret.erase();
      loi->setAttr(kHashmapLookupInitPredExtendedAttr, mlir::UnitAttr::get(ctx));
   });

   module.walk([&](subop::MergeOp mergeOp) {
      if (!mlir::isa<subop::HashMapType>(mergeOp.getResult().getType())) return;
      if (mergeOp->hasAttr(kHashmapMergeCombinePredExtendedAttr)) return;
      if (mergeOp.getCombineFn().empty()) return;
      mlir::Block& b = mergeOp.getCombineFn().front();
      auto ret = mlir::cast<tuples::ReturnOp>(b.getTerminator());
      auto loc = mergeOp.getLoc();
      b.addArgument(mlir::IntegerType::get(ctx, 1), loc);
      b.addArgument(mlir::IntegerType::get(ctx, 1), loc);
      mlir::Value oldPred = b.getArgument(b.getNumArguments() - 2);
      mlir::Value newPred = b.getArgument(b.getNumArguments() - 1);

      mlir::OpBuilder bb(ret);
      mlir::Value mergedPred = bb.create<mlir::arith::AndIOp>(loc, oldPred, newPred);

      llvm::SmallVector<mlir::Value, 32> outs(ret.getOperands().begin(), ret.getOperands().end());
      outs.push_back(mergedPred);
      bb.create<tuples::ReturnOp>(loc, outs);
      ret.erase();
      mergeOp->setAttr(kHashmapMergeCombinePredExtendedAttr, mlir::UnitAttr::get(ctx));
   });
}

// Convert runtime filter descriptions into MLIR subop.map + subop.filter.
// For now: only integer column + int64 constant comparisons. Everything else asserts.
static mlir::Value materializeRuntimeFiltersAsSubopFilter(mlir::OpBuilder& b,
                                                          mlir::Location loc,
                                                          mlir::Value stream,
                                                          const llvm::DenseMap<llvm::StringRef, tuples::ColumnRefAttr>& colByName,
                                                          llvm::ArrayRef<runtime::FilterDescription> filters) {
   if (filters.empty()) return stream;

   // Build a single predicate column: AND over all supported filters.
   auto [predDef, predRef] = [&]() -> std::pair<tuples::ColumnDefAttr, tuples::ColumnRefAttr> {
      auto& cm = b.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      std::string scopeName = cm.getUniqueScope("delay_filter");
      tuples::ColumnDefAttr def = cm.createDef(scopeName, "pred");
      def.getColumn().type = b.getI1Type();
      return {def, cm.createRef(&def.getColumn())};
   }();

   subop::MapCreationHelper helper(b.getContext());
   helper.buildBlock(b, [&](mlir::OpBuilder& rb) {
      mlir::Value acc;
      for (auto& f : filters) {
         auto it = colByName.find(f.columnName);
         assert(it != colByName.end() && "delay_filter: missing filter column in gathered columns");
         mlir::Value colV = helper.access(it->second, loc);

         mlir::Value pred;
         if (f.op == runtime::FilterOp::NOTNULL) {
            pred = rb.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
         } else {
            assert(std::holds_alternative<int64_t>(f.value) && "delay_filter: only int64 filter values supported");
            int64_t v = std::get<int64_t>(f.value);
            auto itTy = mlir::dyn_cast<mlir::IntegerType>(colV.getType());
            assert(itTy && "delay_filter: only integer-typed columns supported");
            mlir::Value c = rb.create<mlir::arith::ConstantIntOp>(loc, v, itTy.getWidth());
            using P = mlir::arith::CmpIPredicate;
            P p;
            switch (f.op) {
               case runtime::FilterOp::EQ: p = P::eq; break;
               case runtime::FilterOp::NEQ: p = P::ne; break;
               case runtime::FilterOp::LT: p = P::slt; break;
               case runtime::FilterOp::LTE: p = P::sle; break;
               case runtime::FilterOp::GT: p = P::sgt; break;
               case runtime::FilterOp::GTE: p = P::sge; break;
               default: assert(0 && "delay_filter: unsupported filter op");
            }
            pred = rb.create<mlir::arith::CmpIOp>(loc, p, colV, c);
         }
         acc = acc ? rb.create<mlir::arith::AndIOp>(loc, acc, pred) : pred;
      }
      rb.create<tuples::ReturnOp>(loc, mlir::ValueRange{acc});
   });

   auto mapOp = b.create<subop::MapOp>(loc,
                                       tuples::TupleStreamType::get(b.getContext()),
                                       stream,
                                       b.getArrayAttr({predDef}),
                                       helper.getColRefs());
   mapOp.getFn().push_back(helper.getMapBlock());
   auto filterOp = b.create<subop::FilterOp>(loc,
                                             mapOp.getResult(),
                                             subop::FilterSemantic::all_true,
                                             b.getArrayAttr({predRef}));
   return filterOp.getRes();
}

// Build a predicate column (AND over supported filters) and append it to the stream.
// The returned stream has an additional boolean column named `predLeafName` in a fresh scope.
static std::pair<mlir::Value, tuples::ColumnDefAttr>
materializeRuntimeFiltersAsPredicateColumn(mlir::OpBuilder& b,
                                           mlir::Location loc,
                                           mlir::Value stream,
                                           const llvm::DenseMap<llvm::StringRef, tuples::ColumnRefAttr>& colByName,
                                           llvm::ArrayRef<runtime::FilterDescription> filters,
                                           llvm::StringRef predLeafName) {
   assert(!filters.empty());

   auto& cm = b.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   std::string scopeName = cm.getUniqueScope("delay_filter_pred");
   tuples::ColumnDefAttr predDef = cm.createDef(scopeName, predLeafName.str());
   predDef.getColumn().type = b.getI1Type();
   // Keep predicate column def alive; column ref created where needed.

   subop::MapCreationHelper helper(b.getContext());
   helper.buildBlock(b, [&](mlir::OpBuilder& rb) {
      mlir::Value acc;
      for (auto& f : filters) {
         auto it = colByName.find(f.columnName);
         assert(it != colByName.end() && "delay_filter_pred: missing filter column in gathered columns");
         mlir::Value colV = helper.access(it->second, loc);

         mlir::Value pred;
         if (f.op == runtime::FilterOp::NOTNULL) {
            pred = rb.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
         } else {
            // Minimal support for TPCH Q1: date column compared to string literal.
            if (std::holds_alternative<int64_t>(f.value)) {
               int64_t v = std::get<int64_t>(f.value);
               auto itTy = mlir::dyn_cast<mlir::IntegerType>(colV.getType());
               assert(itTy && "delay_filter_pred: only integer-typed columns supported");
               mlir::Value c = rb.create<mlir::arith::ConstantIntOp>(loc, v, itTy.getWidth());
               using P = mlir::arith::CmpIPredicate;
               P p;
               switch (f.op) {
                  case runtime::FilterOp::EQ: p = P::eq; break;
                  case runtime::FilterOp::NEQ: p = P::ne; break;
                  case runtime::FilterOp::LT: p = P::slt; break;
                  case runtime::FilterOp::LTE: p = P::sle; break;
                  case runtime::FilterOp::GT: p = P::sgt; break;
                  case runtime::FilterOp::GTE: p = P::sge; break;
                  default: assert(0 && "delay_filter_pred: unsupported filter op");
               }
               pred = rb.create<mlir::arith::CmpIOp>(loc, p, colV, c);
            } else if (std::holds_alternative<std::string>(f.value)) {
               auto s = std::get<std::string>(f.value);
               assert((mlir::isa<lingodb::compiler::dialect::db::DateType>(colV.getType()) ||
                       mlir::isa<lingodb::compiler::dialect::db::CharType>(colV.getType())) &&
                      "delay_filter_pred: string literal filters only supported for db.date/db.char");
               auto rhs = rb.create<lingodb::compiler::dialect::db::ConstantOp>(
                  loc, colV.getType(), rb.getStringAttr(s));
               using P = lingodb::compiler::dialect::db::DBCmpPredicate;
               P p;
               switch (f.op) {
                  case runtime::FilterOp::EQ: p = P::eq; break;
                  case runtime::FilterOp::NEQ: p = P::neq; break;
                  case runtime::FilterOp::LT: p = P::lt; break;
                  case runtime::FilterOp::LTE: p = P::lte; break;
                  case runtime::FilterOp::GT: p = P::gt; break;
                  case runtime::FilterOp::GTE: p = P::gte; break;
                  default: assert(0 && "delay_filter_pred: unsupported filter op");
               }
               auto cmp = rb.create<lingodb::compiler::dialect::db::CmpOp>(loc, p, colV, rhs);
               // cmp may be nullable depending on operands; `derive_truth` normalizes to i1.
               pred = rb.create<lingodb::compiler::dialect::db::DeriveTruth>(loc, cmp);
            } else {
               assert(0 && "delay_filter_pred: unsupported filter literal type");
            }
         }
         acc = acc ? rb.create<mlir::arith::AndIOp>(loc, acc, pred) : pred;
      }
      rb.create<tuples::ReturnOp>(loc, mlir::ValueRange{acc});
   });

   auto mapOp = b.create<subop::MapOp>(loc,
                                       tuples::TupleStreamType::get(b.getContext()),
                                       stream,
                                       b.getArrayAttr({predDef}),
                                       helper.getColRefs());
   mapOp.getFn().push_back(helper.getMapBlock());
   return {mapOp.getResult(), predDef};
}

static llvm::SmallVector<runtime::FilterDescription, 8>
decodeFiltersForStateFromWriterSteps(mlir::Value state, const ModuleReuseInfo& reuse) {
   llvm::DenseMap<mlir::Operation*, const ModuleReuseInfo::StepRW*> rwByStepOp;
   for (const auto& e : reuse.steps) {
      rwByStepOp[const_cast<ExecutionStepOp&>(e.step).getOperation()] = &e;
   }
   llvm::SmallVector<runtime::FilterDescription, 8> decoded;
   auto itW = reuse.writerStepsByState.find(canonicalizeStateValueForReuse(state));
   if (itW == reuse.writerStepsByState.end()) itW = reuse.writerStepsByState.find(state);
   if (itW == reuse.writerStepsByState.end()) return decoded;
   for (ExecutionStepOp ws : itW->second) {
      const ModuleReuseInfo::StepRW* rw = rwByStepOp.lookup(ws.getOperation());
      if (!rw) continue;
      for (mlir::Value r : rw->reads) {
         if (!mlir::isa<subop::TableType>(r.getType())) continue;
         mlir::Value tableV = r;
         if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(r)) {
            if (ba.getOwner() == &ws.getSubOps().front()) {
               assert(static_cast<unsigned>(ba.getArgNumber()) < ws.getNumOperands());
               tableV = ws.getOperand(ba.getArgNumber());
            }
         }
         mlir::Value key = canonicalizeStateValueForReuse(tableV);
         auto it = reuse.externalDatasourceByTableState.find(key);
         if (it == reuse.externalDatasourceByTableState.end()) it = reuse.externalDatasourceByTableState.find(tableV);
         if (it == reuse.externalDatasourceByTableState.end()) continue;
         for (auto& f : it->second.filterDescriptions) decoded.push_back(f);
      }
   }
   return decoded;
}

static mlir::Operation* findFirstOpConsumingInBlock(mlir::Block& block, mlir::Value v) {
   for (mlir::Operation& op : block.without_terminator()) {
      for (mlir::Value in : op.getOperands()) {
         if (in == v) return &op;
      }
   }
   return nullptr;
}

// Join-style plans (e.g. TPCH Q2) thread merged state through `nested_execution_group` and inner
// `execution_step` before `lookup` / `reduce`.
// Follow that chain so delay_filter can still reach the TupleStream fed into those ops.
static mlir::Operation* drillToStateConsumerSkippingNestedScopes(mlir::Block& startBlock, mlir::Value startState) {
   mlir::Block* curBlock = &startBlock;
   mlir::Value curState = startState;
   for (;;) {
      mlir::Operation* op = findFirstOpConsumingInBlock(*curBlock, curState);
      if (!op) return nullptr;
      if (auto nest = mlir::dyn_cast<subop::NestedExecutionGroupOp>(op)) {
         unsigned idx = 0;
         for (; idx < nest.getNumOperands(); ++idx) {
            if (nest.getOperand(idx) == curState) break;
         }
         if (idx >= nest.getNumOperands()) return nullptr;
         curBlock = &nest.getSubOps().front();
         curState = curBlock->getArgument(idx);
         continue;
      }
      if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(op)) {
         unsigned idx = 0;
         for (; idx < step.getNumOperands(); ++idx) {
            if (step.getOperand(idx) == curState) break;
         }
         if (idx >= step.getNumOperands()) return nullptr;
         curBlock = &step.getSubOps().front();
         curState = curBlock->getArgument(idx);
         continue;
      }
      return op;
   }
}

static void insertWriteSidePredIntoHashMapConstructionStep(ExecutionStepOp step,
                                                            llvm::ArrayRef<runtime::FilterDescription> filters) {
   if (filters.empty()) return;
   mlir::Block& body = step.getSubOps().front();

   // Find the table scan_refs (input table) and a gather that follows it.
   subop::ScanRefsOp scanOp;
   for (mlir::Operation& op : body.without_terminator()) {
      auto s = mlir::dyn_cast<subop::ScanRefsOp>(&op);
      if (!s) continue;
      if (mlir::isa<subop::TableType>(s.getState().getType())) { scanOp = s; break; }
   }
   assert(scanOp && "write_pred: expected a scan_refs over table");

   // Ensure scan_refs produced table_entry_ref contains all filter columns.
   {
      auto tableTy = mlir::cast<subop::TableType>(scanOp.getState().getType());
      auto stripSuffix = [](llvm::StringRef s) -> llvm::StringRef {
         size_t pos = s.find('$');
         if (pos == llvm::StringRef::npos) return s;
         return s.take_front(pos);
      };
      auto refDef = scanOp.getRef();
      auto refTy = mlir::dyn_cast<subop::TableEntryRefType>(refDef.getColumn().type);
      assert(refTy && "write_pred: expected scan_refs ref to be table_entry_ref");
      llvm::SmallVector<subop::Member> cols = refTy.getTableColumns().getMembers();
      llvm::DenseSet<subop::Member> have;
      for (auto m : cols) have.insert(m);
      auto& mm = scanOp->getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
      for (auto& f : filters) {
         if (f.op == runtime::FilterOp::NOTNULL) continue;
         subop::Member mem;
         for (auto m : tableTy.getMembers().getMembers()) {
            if (stripSuffix(mm.getName(m)) == f.columnName) { mem = m; break; }
         }
         assert(mem && "write_pred: could not find table member for filter column");
         if (have.insert(mem).second) cols.push_back(mem);
      }
      auto newCols = subop::StateMembersAttr::get(step.getContext(), cols);
      refDef.getColumn().type = subop::TableEntryRefType::get(step.getContext(), newCols);
      scanOp.setRefAttr(refDef);
   }

   subop::GatherOp firstGather;
   for (mlir::Operation& op : body.without_terminator()) {
      if (!scanOp->isBeforeInBlock(&op)) continue;
      auto g = mlir::dyn_cast<subop::GatherOp>(&op);
      if (!g) continue;
      if (g.getStream() == scanOp.getRes()) { firstGather = g; break; }
   }
   assert(firstGather && "write_pred: expected a gather right after table scan_refs");

   // Ensure filter columns exist in the gathered stream by extending the gather mapping (member->coldef).
   auto& cm = firstGather->getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = firstGather->getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   llvm::DenseMap<llvm::StringRef, tuples::ColumnRefAttr> colByName;
   for (auto& p : firstGather.getMapping().getMapping()) {
      auto def = p.second;
      auto [scope, leaf] = cm.getName(&def.getColumn());
      (void)scope;
      colByName[llvm::StringRef(leaf)] = cm.createRef(&def.getColumn());
   }
   auto tableTy = mlir::cast<subop::TableType>(scanOp.getState().getType());
   auto stripSuffix = [](llvm::StringRef s) -> llvm::StringRef {
      size_t pos = s.find('$');
      if (pos == llvm::StringRef::npos) return s;
      return s.take_front(pos);
   };
   bool needExtend = false;
   for (auto& f : filters) {
      if (f.op == runtime::FilterOp::NOTNULL) continue;
      if (colByName.contains(f.columnName)) continue;
      needExtend = true;
      break;
   }
   if (needExtend) {
      llvm::StringRef reusedScope = "write_pred";
      if (!firstGather.getMapping().getMapping().empty()) {
         auto def0 = firstGather.getMapping().getMapping().begin()->second;
         auto [scope0, leaf0] = cm.getName(&def0.getColumn());
         (void)leaf0;
         reusedScope = scope0;
      }
      llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> mappingPairs;
      for (auto& p : firstGather.getMapping().getMapping()) mappingPairs.push_back({p.first, p.second});
      for (auto& f : filters) {
         if (f.op == runtime::FilterOp::NOTNULL) continue;
         if (colByName.contains(f.columnName)) continue;
         subop::Member mem;
         for (auto m : tableTy.getMembers().getMembers()) {
            if (stripSuffix(mm.getName(m)) == f.columnName) { mem = m; break; }
         }
         assert(mem && "write_pred: could not find table member for filter column");
         tuples::ColumnDefAttr def = cm.createDef(reusedScope.str(), f.columnName);
         def.getColumn().type = mm.getType(mem);
         colByName[f.columnName] = cm.createRef(&def.getColumn());
         mappingPairs.push_back({mem, def});
      }
      mlir::OpBuilder gb(firstGather);
      gb.setInsertionPointAfter(firstGather);
      auto newMapping = subop::ColumnDefMemberMappingAttr::get(firstGather.getContext(), mappingPairs);
      auto g2 = gb.create<subop::GatherOp>(firstGather.getLoc(), firstGather.getRes(), firstGather.getRef(), newMapping);
      // Replace uses of old gather stream after it with new gather.
      mlir::Value oldS = firstGather.getRes();
      mlir::Value newS = g2.getRes();
      oldS.replaceUsesWithIf(newS, [&](mlir::OpOperand& ou) {
         if (ou.getOwner() == g2.getOperation()) return false;
         if (ou.getOwner()->getBlock() != &body) return false;
         return firstGather->isBeforeInBlock(ou.getOwner());
      });
      firstGather = g2;
   }

   // Insert a map that computes `filter_pred` bool column.
   mlir::OpBuilder pb(firstGather);
   pb.setInsertionPointAfter(firstGather);
   auto [predStream, predDef] =
      materializeRuntimeFiltersAsPredicateColumn(pb, firstGather.getLoc(), firstGather.getRes(), colByName, filters, "filter_pred");

   // Replace uses of gathered stream with predStream for subsequent ops in this step.
   mlir::Value oldS = firstGather.getRes();
   oldS.replaceUsesWithIf(predStream, [&](mlir::OpOperand& ou) {
      if (ou.getOwner() == predStream.getDefiningOp()) return false;
      if (ou.getOwner()->getBlock() != &body) return false;
      return firstGather->isBeforeInBlock(ou.getOwner());
   });

   // Also reduce the computed predicate into the ht fragment by inserting an extra reduce
   // that only updates `filter_pred$0`. This avoids rewriting the existing big reduce.
   subop::ReduceOp bigReduce;
   for (mlir::Operation& op : body.without_terminator()) {
      auto r = mlir::dyn_cast<subop::ReduceOp>(&op);
      if (!r) continue;
      auto refTy = mlir::dyn_cast<subop::LookupEntryRefType>(r.getRef().getColumn().type);
      if (!refTy) continue;
      if (!mlir::isa<subop::HashMapType>(refTy.getState())) continue;
      bigReduce = r;
      break;
   }
   assert(bigReduce && "write_pred: expected reduce into hashmap");

   // Find the pred member in the hashmap value tuple.
   subop::Member predMember;
   {
      auto refTy = mlir::cast<subop::LookupEntryRefType>(bigReduce.getRef().getColumn().type);
      auto hmTy = mlir::cast<subop::HashMapType>(refTy.getState());
      auto& mm = bigReduce->getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
      for (auto m : hmTy.getValueMembers().getMembers()) {
         if (mm.getName(m) == "filter_pred$0") { predMember = m; break; }
      }
      assert(predMember && "write_pred: could not find filter_pred$0 member in hashmap type");
   }

   tuples::ColumnRefAttr predColRef = cm.createRef(&predDef.getColumn());
   {
      mlir::OpBuilder rb(bigReduce);
      rb.setInsertionPointAfter(bigReduce);
      auto cols = mlir::ArrayAttr::get(step.getContext(), mlir::ArrayRef<mlir::Attribute>{predColRef});
      auto mems = mlir::ArrayAttr::get(step.getContext(), mlir::ArrayRef<mlir::Attribute>{subop::MemberAttr::get(step.getContext(), predMember)});
      auto predReduce =
         rb.create<subop::ReduceOp>(bigReduce.getLoc(), bigReduce.getStream(), bigReduce.getRef(), cols, mems);

      // Region 0: (predCol, currPred) -> incomingPred (= predCol)
      {
         mlir::Block* b = new mlir::Block();
         predReduce.getRegion().push_back(b);
         b->addArgument(predDef.getColumn().type, bigReduce.getLoc());
         b->addArgument(mlir::IntegerType::get(step.getContext(), 1), bigReduce.getLoc());
         mlir::OpBuilder bb(step.getContext());
         bb.setInsertionPointToStart(b);
         bb.create<tuples::ReturnOp>(bigReduce.getLoc(), mlir::ValueRange{b->getArgument(0)});
      }
      // Combine region: (currPred, incomingPred) -> (currPred & incomingPred)
      {
         mlir::Block* b = new mlir::Block();
         predReduce.getCombine().push_back(b);
         b->addArgument(mlir::IntegerType::get(step.getContext(), 1), bigReduce.getLoc());
         b->addArgument(mlir::IntegerType::get(step.getContext(), 1), bigReduce.getLoc());
         mlir::OpBuilder bb(step.getContext());
         bb.setInsertionPointToStart(b);
         mlir::Value merged = bb.create<mlir::arith::AndIOp>(bigReduce.getLoc(), b->getArgument(0), b->getArgument(1));
         bb.create<tuples::ReturnOp>(bigReduce.getLoc(), mlir::ValueRange{merged});
      }
   }
}

/// Join hash build via `growing buffer` + `create_hash_indexed_view`: materialize `filter_pred$0` from table scan.
static void insertWriteSidePredIntoBufferConstructionStep(ExecutionStepOp step,
                                                          llvm::ArrayRef<runtime::FilterDescription> filters) {
   if (filters.empty()) return;
   mlir::Block& body = step.getSubOps().front();

   subop::ScanRefsOp scanOp;
   for (mlir::Operation& op : body.without_terminator()) {
      auto s = mlir::dyn_cast<subop::ScanRefsOp>(&op);
      if (!s) continue;
      if (mlir::isa<subop::TableType>(s.getState().getType())) {
         scanOp = s;
         break;
      }
   }
   assert(scanOp && "write_pred_buf: expected a scan_refs over table");

   {
      auto tableTy = mlir::cast<subop::TableType>(scanOp.getState().getType());
      auto stripSuffix = [](llvm::StringRef s) -> llvm::StringRef {
         size_t pos = s.find('$');
         if (pos == llvm::StringRef::npos) return s;
         return s.take_front(pos);
      };
      auto refDef = scanOp.getRef();
      auto refTy = mlir::dyn_cast<subop::TableEntryRefType>(refDef.getColumn().type);
      assert(refTy && "write_pred_buf: expected scan_refs ref to be table_entry_ref");
      llvm::SmallVector<subop::Member> cols = refTy.getTableColumns().getMembers();
      llvm::DenseSet<subop::Member> have;
      for (auto m : cols) have.insert(m);
      auto& mm = scanOp->getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
      for (auto& f : filters) {
         if (f.op == runtime::FilterOp::NOTNULL) continue;
         subop::Member mem;
         for (auto m : tableTy.getMembers().getMembers()) {
            if (stripSuffix(mm.getName(m)) == f.columnName) {
               mem = m;
               break;
            }
         }
         assert(mem && "write_pred_buf: could not find table member for filter column");
         if (have.insert(mem).second) cols.push_back(mem);
      }
      auto newCols = subop::StateMembersAttr::get(step.getContext(), cols);
      refDef.getColumn().type = subop::TableEntryRefType::get(step.getContext(), newCols);
      scanOp.setRefAttr(refDef);
   }

   subop::GatherOp firstGather;
   for (mlir::Operation& op : body.without_terminator()) {
      if (!scanOp->isBeforeInBlock(&op)) continue;
      auto g = mlir::dyn_cast<subop::GatherOp>(&op);
      if (!g) continue;
      if (g.getStream() == scanOp.getRes()) {
         firstGather = g;
         break;
      }
   }
   assert(firstGather && "write_pred_buf: expected a gather right after table scan_refs");

   auto& cm = firstGather->getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = firstGather->getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
   llvm::DenseMap<llvm::StringRef, tuples::ColumnRefAttr> colByName;
   for (auto& p : firstGather.getMapping().getMapping()) {
      auto def = p.second;
      auto [scope, leaf] = cm.getName(&def.getColumn());
      (void)scope;
      colByName[llvm::StringRef(leaf)] = cm.createRef(&def.getColumn());
   }
   auto tableTy = mlir::cast<subop::TableType>(scanOp.getState().getType());
   auto stripSuffix = [](llvm::StringRef s) -> llvm::StringRef {
      size_t pos = s.find('$');
      if (pos == llvm::StringRef::npos) return s;
      return s.take_front(pos);
   };
   bool needExtend = false;
   for (auto& f : filters) {
      if (f.op == runtime::FilterOp::NOTNULL) continue;
      if (colByName.contains(f.columnName)) continue;
      needExtend = true;
      break;
   }
   if (needExtend) {
      llvm::StringRef reusedScope = "write_pred";
      if (!firstGather.getMapping().getMapping().empty()) {
         auto def0 = firstGather.getMapping().getMapping().begin()->second;
         auto [scope0, leaf0] = cm.getName(&def0.getColumn());
         (void)leaf0;
         reusedScope = scope0;
      }
      llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> mappingPairs;
      for (auto& p : firstGather.getMapping().getMapping()) mappingPairs.push_back({p.first, p.second});
      for (auto& f : filters) {
         if (f.op == runtime::FilterOp::NOTNULL) continue;
         if (colByName.contains(f.columnName)) continue;
         subop::Member mem;
         for (auto m : tableTy.getMembers().getMembers()) {
            if (stripSuffix(mm.getName(m)) == f.columnName) {
               mem = m;
               break;
            }
         }
         assert(mem && "write_pred_buf: could not find table member for filter column");
         tuples::ColumnDefAttr def = cm.createDef(reusedScope.str(), f.columnName);
         def.getColumn().type = mm.getType(mem);
         colByName[f.columnName] = cm.createRef(&def.getColumn());
         mappingPairs.push_back({mem, def});
      }
      mlir::OpBuilder gb(firstGather);
      gb.setInsertionPointAfter(firstGather);
      auto newMapping = subop::ColumnDefMemberMappingAttr::get(firstGather.getContext(), mappingPairs);
      auto g2 = gb.create<subop::GatherOp>(firstGather.getLoc(), firstGather.getRes(), firstGather.getRef(), newMapping);
      mlir::Value oldS = firstGather.getRes();
      mlir::Value newS = g2.getRes();
      oldS.replaceUsesWithIf(newS, [&](mlir::OpOperand& ou) {
         if (ou.getOwner() == g2.getOperation()) return false;
         if (ou.getOwner()->getBlock() != &body) return false;
         return firstGather->isBeforeInBlock(ou.getOwner());
      });
      firstGather = g2;
   }

   mlir::OpBuilder pb(firstGather);
   pb.setInsertionPointAfter(firstGather);
   auto [predStream, predDef] =
      materializeRuntimeFiltersAsPredicateColumn(pb, firstGather.getLoc(), firstGather.getRes(), colByName, filters, "filter_pred");

   mlir::Value oldS = firstGather.getRes();
   oldS.replaceUsesWithIf(predStream, [&](mlir::OpOperand& ou) {
      if (ou.getOwner() == predStream.getDefiningOp()) return false;
      if (ou.getOwner()->getBlock() != &body) return false;
      return firstGather->isBeforeInBlock(ou.getOwner());
   });

   subop::MaterializeOp matOp;
   step.getOperation()->walk([&](subop::MaterializeOp m) {
      auto bufTy = mlir::dyn_cast<subop::BufferType>(m.getState().getType());
      if (!bufTy) return mlir::WalkResult::advance();
      if (!valueMembersContainMemberNamed(step.getContext(), bufTy.getMembers(), "filter_pred$0"))
         return mlir::WalkResult::advance();
      matOp = m;
      return mlir::WalkResult::interrupt();
   });
   assert(matOp && "write_pred_buf: expected materialize into join buffer with filter_pred$0");

   subop::Member predMember;
   {
      auto bufTy = mlir::cast<subop::BufferType>(matOp.getState().getType());
      for (auto m : bufTy.getMembers().getMembers()) {
         if (mm.getName(m) == "filter_pred$0") {
            predMember = m;
            break;
         }
      }
      assert(predMember && "write_pred_buf: filter_pred$0 missing on buffer type");
   }

   tuples::ColumnRefAttr predColRef = cm.createRef(&predDef.getColumn());
   llvm::SmallVector<subop::RefMappingPairT> pairs;
   for (auto& pr : matOp.getMapping().getMapping()) pairs.push_back(pr);
   pairs.push_back({predMember, predColRef});
   matOp.setMappingAttr(subop::ColumnRefMemberMappingAttr::get(step.getContext(), pairs));
}

// For scan_refs over join `hashmap`: filter by stored predicate member `filter_pred$0`.
static void insertScanRefsPredFilter(ExecutionStepOp step, subop::Member predMember) {
   mlir::Block& body = step.getSubOps().front();
   auto& cm = step.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = step.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();

   subop::ScanRefsOp scanOp;
   for (mlir::Operation& op : body.without_terminator()) {
      auto s = mlir::dyn_cast<subop::ScanRefsOp>(&op);
      if (!s) continue;
      if (mlir::isa<subop::HashMapType>(s.getState().getType())) { scanOp = s; break; }
   }
   if (!scanOp) return;

   // Find first gather after scan, fed by scan result.
   subop::GatherOp gatherOp;
   for (mlir::Operation& op : body.without_terminator()) {
      if (!scanOp->isBeforeInBlock(&op)) continue;
      auto g = mlir::dyn_cast<subop::GatherOp>(&op);
      if (!g) continue;
      if (g.getStream() == scanOp.getRes()) { gatherOp = g; break; }
   }
   assert(gatherOp && "pred_filter: expected gather directly after scan_refs(ht)");

   // Ensure gather includes predicate column.
   bool hasPred = false;
   for (auto& p : gatherOp.getMapping().getMapping()) {
      if (p.first == predMember) { hasPred = true; break; }
   }
   tuples::ColumnRefAttr predRef;
   if (!hasPred) {
      // Reuse existing scope and add a new column def.
      llvm::StringRef scope = "pred_filter";
      if (!gatherOp.getMapping().getMapping().empty()) {
         auto def0 = gatherOp.getMapping().getMapping().begin()->second;
         auto [scope0, leaf0] = cm.getName(&def0.getColumn());
         (void)leaf0;
         scope = scope0;
      }
      tuples::ColumnDefAttr predDef = cm.createDef(scope.str(), "filter_pred");
      predDef.getColumn().type = mm.getType(predMember);
      predRef = cm.createRef(&predDef.getColumn());

      llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> mappingPairs;
      for (auto& p : gatherOp.getMapping().getMapping()) mappingPairs.push_back({p.first, p.second});
      mappingPairs.push_back({predMember, predDef});

      mlir::OpBuilder gb(gatherOp);
      gb.setInsertionPointAfter(gatherOp);
      auto newMapping = subop::ColumnDefMemberMappingAttr::get(gatherOp.getContext(), mappingPairs);
      auto g2 = gb.create<subop::GatherOp>(gatherOp.getLoc(), gatherOp.getRes(), gatherOp.getRef(), newMapping);
      gatherOp = g2;
   } else {
      // Find existing pred column def.
      for (auto& p : gatherOp.getMapping().getMapping()) {
         if (p.first == predMember) {
            predRef = cm.createRef(&p.second.getColumn());
            break;
         }
      }
      assert(predRef);
   }

   // Insert filter(all_true [predRef]) after gatherOp.
   mlir::OpBuilder fb(gatherOp);
   fb.setInsertionPointAfter(gatherOp);
   auto filterOp = fb.create<subop::FilterOp>(gatherOp.getLoc(), gatherOp.getRes(), subop::FilterSemantic::all_true,
                                              fb.getArrayAttr({predRef}));

   // Rewrite uses of gathered stream after insertion point.
   mlir::Value orig = gatherOp.getRes();
   mlir::Value filtered = filterOp.getRes();
   mlir::Block* gatherBlock = gatherOp->getBlock();
   orig.replaceUsesWithIf(filtered, [&](mlir::OpOperand& ou) {
      mlir::Operation* owner = ou.getOwner();
      if (owner == filtered.getDefiningOp()) return false;
      if (owner->getBlock() != gatherBlock) return false;
      return gatherOp->isBeforeInBlock(owner);
   });
}

/// After `lookup` / `scan_list` on `!subop.hash_indexed_view<...>` with `filter_pred$0`, re-check the stored bool.
static void insertHashIndexedViewGatherPredFilters(ExecutionStepOp step, subop::Member predMember) {
   auto* ctx = step.getContext();
   auto& cm = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto& mm = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();

   llvm::SmallVector<subop::GatherOp> gathers;
   step.getOperation()->walk([&](subop::GatherOp g) { gathers.push_back(g); });

   for (subop::GatherOp gatherOp : gathers) {
      auto refTy = mlir::dyn_cast<subop::LookupEntryRefType>(gatherOp.getRef().getColumn().type);
      if (!refTy) continue;
      auto hiv = mlir::dyn_cast<subop::HashIndexedViewType>(refTy.getState());
      if (!hiv) continue;
      if (!valueMembersContainMemberNamed(ctx, hiv.getValueMembers(), "filter_pred$0")) continue;

      bool hasPred = false;
      for (auto& p : gatherOp.getMapping().getMapping()) {
         if (p.first == predMember) {
            hasPred = true;
            break;
         }
      }
      tuples::ColumnRefAttr predRef;
      if (!hasPred) {
         llvm::StringRef scope = "pred_filter";
         if (!gatherOp.getMapping().getMapping().empty()) {
            auto def0 = gatherOp.getMapping().getMapping().begin()->second;
            auto [scope0, leaf0] = cm.getName(&def0.getColumn());
            (void)leaf0;
            scope = scope0;
         }
         tuples::ColumnDefAttr predDef = cm.createDef(scope.str(), "filter_pred");
         predDef.getColumn().type = mm.getType(predMember);
         predRef = cm.createRef(&predDef.getColumn());

         llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> mappingPairs;
         for (auto& p : gatherOp.getMapping().getMapping()) mappingPairs.push_back({p.first, p.second});
         mappingPairs.push_back({predMember, predDef});

         mlir::OpBuilder gb(gatherOp);
         gb.setInsertionPointAfter(gatherOp);
         auto newMapping = subop::ColumnDefMemberMappingAttr::get(gatherOp.getContext(), mappingPairs);
         auto g2 = gb.create<subop::GatherOp>(gatherOp.getLoc(), gatherOp.getRes(), gatherOp.getRef(), newMapping);
         gatherOp = g2;
      } else {
         for (auto& p : gatherOp.getMapping().getMapping()) {
            if (p.first == predMember) {
               predRef = cm.createRef(&p.second.getColumn());
               break;
            }
         }
         assert(predRef);
      }

      mlir::OpBuilder fb(gatherOp);
      fb.setInsertionPointAfter(gatherOp);
      auto filterOp = fb.create<subop::FilterOp>(gatherOp.getLoc(), gatherOp.getRes(), subop::FilterSemantic::all_true,
                                                 fb.getArrayAttr({predRef}));

      mlir::Value orig = gatherOp.getRes();
      mlir::Value filtered = filterOp.getRes();
      mlir::Block* gatherBlock = gatherOp->getBlock();
      orig.replaceUsesWithIf(filtered, [&](mlir::OpOperand& ou) {
         mlir::Operation* owner = ou.getOwner();
         if (owner == filtered.getDefiningOp()) return false;
         if (owner->getBlock() != gatherBlock) return false;
         return gatherOp->isBeforeInBlock(owner);
      });
   }
}

struct CacheTarget {
   mlir::Value state;
   uint64_t cacheKey;
};

static bool isPipelineStateValue(mlir::Value v) {
   mlir::Type t = v.getType();
   if (mlir::isa<subop::State>(t)) return true;
   auto tl = mlir::dyn_cast<subop::ThreadLocalType>(t);
   return tl && mlir::isa<subop::State>(tl.getWrapped());
}

/// `ModuleReuseInfo` maps are keyed by values produced inside `collectModuleReuseInfo`, which uses
/// `canonicalizeStateValueForReuse` for merge pairing and writer/create registration. Match that
/// here when `neededStates` / targets carry a different SSA view of the same state.
template <typename MapT>
static auto findReuseMap(const MapT& m, mlir::Value st) -> decltype(m.find(st)) {
   mlir::Value c = canonicalizeStateValueForReuse(st);
   auto it = m.find(c);
   if (it != m.end()) return it;
   if (c != st) return m.find(st);
   return m.end();
}

/// If \p v is an operand of an enclosing `execution_step`, map it to the body block argument that aliases it.
static mlir::Value mapStateThroughExecutionStepOperands(mlir::Value v, mlir::Operation* user) {
   for (mlir::Operation* p = user; p; p = p->getParentOp()) {
      auto step = mlir::dyn_cast<subop::ExecutionStepOp>(p);
      if (!step) continue;
      for (unsigned i = 0; i < step.getNumOperands(); ++i) {
         if (step.getOperand(i) != v) continue;
         mlir::Region& r = step.getSubOps();
         if (r.empty()) continue;
         mlir::Block& body = r.front();
         if (i < body.getNumArguments()) return body.getArgument(i);
      }
   }
   return {};
}

/// SSA seeds for tracing a join build buffer to `create_hash_indexed_view`: the merge-result buffer,
/// its paired `thread_local` writer state, and any buffer/thread_local values recorded as **written**
/// on writer steps for those keys (`ModuleReuseInfo` / `analyzeStepStateRW`).
static void collectJoinBufferReachabilitySeeds(mlir::Value canonicalMergedBuffer, const ModuleReuseInfo& reuse,
                                               llvm::SmallVectorImpl<mlir::Value>& seeds) {
   llvm::DenseSet<void*> seen;
   auto push = [&](mlir::Value v) {
      if (!v) return;
      if (!mlir::isa<subop::BufferType, subop::ThreadLocalType>(v.getType())) return;
      if (!seen.insert(v.getAsOpaquePointer()).second) return;
      seeds.push_back(v);
   };

   push(canonicalMergedBuffer);
   if (auto it = findReuseMap(reuse.mergedFromThreadLocal, canonicalMergedBuffer);
       it != reuse.mergedFromThreadLocal.end()) {
      push(it->second);
   }

   llvm::DenseMap<mlir::Operation*, const ModuleReuseInfo::StepRW*> rwByStepOp;
   for (const auto& e : reuse.steps) {
      // `reuse` is const; `ExecutionStepOp::getOperation()` is non-const in this MLIR build.
      rwByStepOp[const_cast<subop::ExecutionStepOp&>(e.step).getOperation()] = &e;
   }
   auto pushWriterBuffers = [&](mlir::Value key) {
      auto itW = findReuseMap(reuse.writerStepsByState, key);
      if (itW == reuse.writerStepsByState.end()) return;
      for (ExecutionStepOp ws : itW->second) {
         const ModuleReuseInfo::StepRW* rw = rwByStepOp.lookup(ws.getOperation());
         if (!rw) continue;
         for (mlir::Value w : rw->writes) {
            push(w);
         }
      }
   };
   pushWriterBuffers(canonicalMergedBuffer);
   if (auto it = findReuseMap(reuse.mergedFromThreadLocal, canonicalMergedBuffer);
       it != reuse.mergedFromThreadLocal.end()) {
      pushWriterBuffers(it->second);
   }
}

/// Forward walk (uses + `merge` + `execution_step` region args) until we see `subop.create_hash_indexed_view`
/// whose `source` is a tracked `!subop.buffer` SSA value.
static bool joinBufferSeedsReachCreateHashIndexedView(llvm::ArrayRef<mlir::Value> seeds) {
   llvm::DenseSet<void*> visited;
   llvm::SmallVector<mlir::Value> stack;
   auto push = [&](mlir::Value v) {
      if (!v) return;
      if (!mlir::isa<subop::BufferType, subop::ThreadLocalType>(v.getType())) return;
      if (!visited.insert(v.getAsOpaquePointer()).second) return;
      stack.push_back(v);
   };
   for (mlir::Value s : seeds) push(s);

   while (!stack.empty()) {
      mlir::Value v = stack.back();
      stack.pop_back();
      for (mlir::OpOperand& use : v.getUses()) {
         mlir::Operation* op = use.getOwner();
         if (mlir::isa<subop::BufferType>(v.getType())) {
            if (auto chiv = mlir::dyn_cast<subop::CreateHashIndexedView>(op)) {
               if (chiv.getSource() == v) return true;
            }
         }
         if (auto merge = mlir::dyn_cast<subop::MergeOp>(op)) {
            if (merge.getThreadLocal() == v) push(merge.getResult());
         }
         if (mlir::Value inner = mapStateThroughExecutionStepOperands(v, op)) push(inner);
      }
   }
   return false;
}

/// Filter `!subop.buffer` reuse targets to those that (per `reuse` RW + SSA forward walk) feed a HIV build.
static llvm::SmallVector<mlir::Value, 8> collectJoinBuffersFeedingHashIndexedView(llvm::ArrayRef<mlir::Value> candidates,
                                                                                 const ModuleReuseInfo& reuse) {
   llvm::SmallVector<mlir::Value, 8> out;
   for (mlir::Value cand : candidates) {
      auto bt = mlir::dyn_cast_or_null<subop::BufferType>(cand.getType());
      if (!bt) continue;
      if (valueMembersContainMemberNamed(bt.getContext(), bt.getMembers(), "filter_pred$0")) continue;

      llvm::SmallVector<mlir::Value, 8> seeds;
      collectJoinBufferReachabilitySeeds(cand, reuse, seeds);
      if (joinBufferSeedsReachCreateHashIndexedView(seeds)) out.push_back(cand);
   }
   return out;
}

ExecutionGroupOp getSingleExecutionGroup(mlir::ModuleOp module) {
   llvm::SmallVector<ExecutionGroupOp, 4> groups;
   module.walk([&](ExecutionGroupOp g) { groups.push_back(g); });
   assert(groups.size() == 1 && "expected exactly one execution_group per module");
   return groups.front();
}

/// Map any `execution_step` (including nested regions) to the enclosing top-level step that is a
/// direct child of `donor`'s main block — cloning only ever inserts those top-level ops.
///
/// Returns a step different from \p s when \p s lives under `nested_execution_group` (see
/// PrepareForLowering::splitNested): `collectModuleReuseInfo` records those nested writers, but
/// `cloneExecutionStepsToQuery0` only clones top-level steps under the donor `execution_group`.
static ExecutionStepOp liftToTopLevelStepInDonor(ExecutionGroupOp donor, ExecutionStepOp s) {
   for (mlir::Operation* op = s.getOperation(); ;) {
      if (auto es = mlir::dyn_cast<ExecutionStepOp>(op)) {
         if (es->getParentOp() == donor.getOperation()) {
            return es;
         }
      }
      mlir::Region* pr = op->getParentRegion();
      assert(pr);
      op = pr->getParentOp();
      assert(op);
   }
}

/// BFS over states: start from matched `targets0` states (plus paired thread_local from
/// `mergedFromThreadLocal`), then for every state S pull in any state read by a writer of S
/// (`ModuleReuseInfo::StepRW::reads`). This matches how construction depends on prerequisite
/// states without relying on SSA `getDefiningOp` alone.
static llvm::DenseSet<mlir::Value> expandNeededStatesFromTargets(llvm::ArrayRef<CacheTarget> targets0,
                                                                 const ModuleReuseInfo& reuse) {
   llvm::DenseMap<mlir::Operation*, const ModuleReuseInfo::StepRW*> rwByStepOp;
   for (const auto& e : reuse.steps) {
      ExecutionStepOp stepOp = e.step;
      rwByStepOp[stepOp.getOperation()] = &e;
   }

   llvm::DenseSet<mlir::Value> needed;
   llvm::SmallVector<mlir::Value, 32> queue;
   auto enqueue = [&](mlir::Value v) {
      if (!isPipelineStateValue(v)) return;
      if (needed.insert(v).second) queue.push_back(v);
   };

   for (auto& t : targets0) {
      enqueue(t.state);
      auto it = findReuseMap(reuse.mergedFromThreadLocal, t.state);
      if (it != reuse.mergedFromThreadLocal.end()) {
         enqueue(it->second);
      }
   }

   for (size_t qi = 0; qi < queue.size(); ++qi) {
      mlir::Value s = queue[qi];

      if (auto itM = findReuseMap(reuse.mergedFromThreadLocal, s);
          itM != reuse.mergedFromThreadLocal.end()) {
         enqueue(itM->second);
      }

      auto itW = findReuseMap(reuse.writerStepsByState, s);
      if (itW == reuse.writerStepsByState.end()) continue;
      for (ExecutionStepOp w : itW->second) {
         const ModuleReuseInfo::StepRW* rw = rwByStepOp.lookup(w.getOperation());
         for (mlir::Value r : rw->reads) {
            enqueue(r);
         }
      }
   }
   return needed;
}

/// For every needed state, include:
/// - each top-level `execution_step` listed in `writerStepsByState` (all writes / updates), and
/// - the top-level `execution_step` that **defines** the state SSA result, if any (create path).
/// - the create-only `execution_step` registered in `ModuleReuseInfo::createOnlyStepForState`, if any.
static llvm::SmallVector<ExecutionStepOp, 32>
collectCreateAndWriteStepsForStates(ExecutionGroupOp donor, const ModuleReuseInfo& reuse,
                                    const llvm::DenseSet<mlir::Value>& neededStates) {
   llvm::DenseSet<mlir::Operation*> seen;
   llvm::SmallVector<ExecutionStepOp, 32> out;
   for (mlir::Value s : neededStates) {
      assert(canonicalizeStateValueForReuse(s) == s);
      if (auto itC = reuse.createOnlyStepForState.find(s); itC != reuse.createOnlyStepForState.end()) {
         ExecutionStepOp top = liftToTopLevelStepInDonor(donor, itC->second);
         if (seen.insert(top.getOperation()).second) out.push_back(top);
      }
      auto itW = findReuseMap(reuse.writerStepsByState, s);
      if (itW == reuse.writerStepsByState.end()) continue;
      for (ExecutionStepOp w : itW->second) {
         ExecutionStepOp top = liftToTopLevelStepInDonor(donor, w);
         if (seen.insert(top.getOperation()).second) out.push_back(top);
      }
   }
   llvm::sort(out, [](ExecutionStepOp a, ExecutionStepOp b) { return a->isBeforeInBlock(b); });
   return out;
}

static int executionStepDonorIndex(ExecutionGroupOp donor, ExecutionStepOp target) {
   int i = 0;
   for (mlir::Operation& op : donor.getSubOps().front()) {
      if (auto es = mlir::dyn_cast<ExecutionStepOp>(&op)) {
         if (es == target) return i;
         ++i;
      }
   }
   return -1;
}

static const ModuleReuseInfo::StepRW* lookupStepRw(const ModuleReuseInfo& reuse, ExecutionStepOp step) {
   for (const auto& e : reuse.steps) {
      if (e.step == step) return &e;
   }
   return nullptr;
}

mlir::IRMapping cloneExecutionStepsToQuery0(ExecutionGroupOp dstGroup,
                                            llvm::ArrayRef<ExecutionStepOp> steps) {
   auto& dstBlock = dstGroup.getSubOps().front();
   auto* dstTerminator = dstBlock.getTerminator();
   assert(dstTerminator && "dst execution_group must have a terminator");

   mlir::IRMapping mapping;
   for (auto s : steps) {
      auto* cloned = s.getOperation()->clone(mapping);
      dstBlock.getOperations().insert(mlir::Block::iterator(dstTerminator), cloned);
   }
   return mapping;
}

void insertCachePutsForTargets(mlir::ModuleOp producerModule, llvm::ArrayRef<CacheTarget> targets) {
   if (targets.empty()) return;
   // Join hashmaps: extend schema with `filter_pred$0` when we will materialize table filters on the write path.
   auto reusePreRewrite = collectModuleReuseInfo(producerModule);
   llvm::SmallVector<mlir::Value, 8> bufferCandidates;
   for (auto& t : targets) {
      if (mlir::isa<subop::BufferType>(t.state.getType())) bufferCandidates.push_back(t.state);
   }
   llvm::SmallVector<mlir::Value, 8> extendJoinBuffers =
      collectJoinBuffersFeedingHashIndexedView(bufferCandidates, reusePreRewrite);
   rewriteHashmapTypesInModule(producerModule, extendJoinBuffers);
   auto reuse = collectModuleReuseInfo(producerModule);

   llvm::DenseSet<uint64_t> seenKeys;
   for (auto& t : targets) {
      if (!t.state) continue;
      assert(!mlir::isa<ThreadLocalType>(t.state.getType()) &&
             "insertCachePutsForTargets must never be called for thread_local-wrapped states");
      if (!seenKeys.insert(t.cacheKey).second) continue;

      mlir::Value st = t.state;
      mlir::Operation* insertAfter = st.getDefiningOp();
      if (auto itW = reuse.writerStepsByState.find(st); itW != reuse.writerStepsByState.end()) {
         for (auto s : itW->second) {
            auto* op = s.getOperation();
            if (!insertAfter) insertAfter = op;
            else if (insertAfter->isBeforeInBlock(op)) insertAfter = op;
         }
      }
      if (!insertAfter) continue;

      // Write-side: join `hashmap` — materialize table descr filters right after scan, store bool in `filter_pred$0`.
      if (mlir::isa<subop::HashMapType>(st.getType())) {
         auto itTL = reuse.mergedFromThreadLocal.find(st);
         assert(itTL != reuse.mergedFromThreadLocal.end() && "hashmap merge result must have paired thread_local");
         mlir::Value tl = itTL->second;

         auto decoded = decodeFiltersForStateFromWriterSteps(tl, reuse);
         if (!decoded.empty()) {
            auto itW = reuse.writerStepsByState.find(tl);
            assert(itW != reuse.writerStepsByState.end());
            ExecutionStepOp construction;
            for (auto ws : itW->second) {
               mlir::Block& body = ws.getSubOps().front();
               bool hasLoi = false, hasRed = false;
               for (auto& op : body.without_terminator()) {
                  if (mlir::isa<subop::LookupOrInsertOp>(&op)) hasLoi = true;
                  if (mlir::isa<subop::ReduceOp>(&op)) hasRed = true;
               }
               if (hasLoi && hasRed) { construction = ws; break; }
            }
            assert(construction && "write_pred: could not find construction step for join hashmap");
            insertWriteSidePredIntoHashMapConstructionStep(construction, decoded);
         }
      }

      auto loc = insertAfter->getLoc();
      mlir::OpBuilder builder(st.getContext());
      builder.setInsertionPointAfter(insertAfter);
      auto resTy = st.getType();
      auto step = builder.create<ExecutionStepOp>(loc, mlir::TypeRange{}, mlir::ValueRange{st},
                                                  builder.getArrayAttr({builder.getBoolAttr(false)}));
      auto& block = step.getSubOps().emplaceBlock();
      block.addArgument(resTy, loc);
      builder.setInsertionPointToStart(&block);
      builder.create<CachePutOp>(loc, builder.getI64IntegerAttr(static_cast<int64_t>(t.cacheKey)), block.getArgument(0));
      builder.create<ExecutionStepReturnOp>(loc, mlir::ValueRange{});
   }
}

void injectCacheGetsAndDeleteConstructionSteps(mlir::ModuleOp consumerModule,
                                               llvm::ArrayRef<CacheTarget> targets) {
   if (targets.empty()) return;

   auto reusePreRewrite = collectModuleReuseInfo(consumerModule);
   llvm::SmallVector<mlir::Value, 8> bufferCandidates;
   for (auto& t : targets) {
      if (mlir::isa<subop::BufferType>(t.state.getType())) bufferCandidates.push_back(t.state);
   }
   llvm::SmallVector<mlir::Value, 8> extendJoinBuffers =
      collectJoinBuffersFeedingHashIndexedView(bufferCandidates, reusePreRewrite);
   rewriteHashmapTypesInModule(consumerModule, extendJoinBuffers);
   auto reuse = collectModuleReuseInfo(consumerModule);

   auto findEnclosingExecutionGroup = [&](mlir::Value v) -> ExecutionGroupOp {
      if (auto* defOp = v.getDefiningOp()) {
         if (auto g = defOp->getParentOfType<ExecutionGroupOp>()) return g;
      }
      for (auto& use : v.getUses()) {
         if (auto g = use.getOwner()->getParentOfType<ExecutionGroupOp>()) return g;
      }
      llvm_unreachable("state must be inside an execution_group");
   };

   auto insertCacheGetAtExecutionGroupStart = [&](ExecutionGroupOp group, mlir::Type stateTy, uint64_t key) -> mlir::Value {
      auto loc = group->getLoc();
      mlir::OpBuilder builder(stateTy.getContext());
      builder.setInsertionPointToStart(&group.getSubOps().front());
      auto step = builder.create<ExecutionStepOp>(loc, mlir::TypeRange{stateTy}, mlir::ValueRange{},
                                                  builder.getArrayAttr({builder.getBoolAttr(false)}));
      auto& block = step.getSubOps().emplaceBlock();
      builder.setInsertionPointToStart(&block);
      auto got = builder.create<CacheGetOp>(loc, stateTy, builder.getI64IntegerAttr(static_cast<int64_t>(key)));
      builder.create<ExecutionStepReturnOp>(loc, mlir::ValueRange{got.getRes()});
      return step.getResult(0);
   };

   llvm::DenseMap<uint64_t, mlir::Value> keyToCached;
   llvm::DenseSet<mlir::Operation*> uniqOpsToErase;
   llvm::SmallVector<mlir::Operation*, 128> opsToErase;
   llvm::DenseSet<mlir::Operation*> joinBufPredProbeInjectedGroups;

   // Precompute decoded table filters per target state **before** we start rewriting (cache_get insertion
   // and replaceAllUsesWith can otherwise hide the original get_external-derived table values).
   llvm::DenseMap<mlir::Value, llvm::SmallVector<runtime::FilterDescription, 8>> decodedFiltersByTarget;
   {
      llvm::DenseMap<mlir::Operation*, const ModuleReuseInfo::StepRW*> rwByStepOp;
      for (const auto& e : reuse.steps) {
         rwByStepOp[const_cast<ExecutionStepOp&>(e.step).getOperation()] = &e;
      }
      for (auto& t : targets) {
         if (!t.state) continue;
         mlir::Value state = t.state;
         llvm::SmallVector<mlir::Value, 2> statesToScan;
         statesToScan.push_back(state);
         if (auto itTL = findReuseMap(reuse.mergedFromThreadLocal, state);
             itTL != reuse.mergedFromThreadLocal.end()) {
            statesToScan.push_back(itTL->second);
         }

         llvm::SmallVector<ExecutionStepOp, 32> writerSteps;
         for (auto st : statesToScan) {
            auto itW = reuse.writerStepsByState.find(st);
            if (itW == reuse.writerStepsByState.end()) continue;
            for (auto s : itW->second) writerSteps.push_back(s);
         }
         if (writerSteps.empty()) continue;

         llvm::SmallVector<runtime::FilterDescription, 8> decodedFilters;
         for (ExecutionStepOp ws : writerSteps) {
            const ModuleReuseInfo::StepRW* rw = rwByStepOp.lookup(ws.getOperation());
            if (!rw) continue;
            for (mlir::Value r : rw->reads) {
               if (!mlir::isa<subop::TableType>(r.getType())) continue;
               // Table values often appear as block arguments of the writer step body; map those
               // back to the corresponding step operand so we can hit the table_ref-recorded key.
               mlir::Value tableV = r;
               if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(r)) {
                  if (ba.getOwner() == &ws.getSubOps().front()) {
                     assert(static_cast<unsigned>(ba.getArgNumber()) < ws.getNumOperands());
                     tableV = ws.getOperand(ba.getArgNumber());
                  }
               }
               auto it = findReuseMap(reuse.externalDatasourceByTableState, tableV);
               if (it == reuse.externalDatasourceByTableState.end()) continue;
               for (auto& f : it->second.filterDescriptions) decodedFilters.push_back(f);
            }
         }
         if (!decodedFilters.empty()) {
            decodedFiltersByTarget[state] = std::move(decodedFilters);
         }
      }
   }

   // Consumer join buffers: `rewriteHashmapTypesInModule` extends the buffer layout; any remaining
   // thread_local writer that materializes into that buffer must also persist `filter_pred$0`
   // (the producer synthetic module is patched in `insertCachePutsForTargets`, but live pipelines
   // here must stay layout-consistent until construction steps are erased).
   llvm::DenseSet<mlir::Operation*> bufferWritePredPatchedSteps;
   for (auto& t : targets) {
      if (!t.state) continue;
      if (!mlir::isa<subop::BufferType>(t.state.getType())) continue;
      auto itF = decodedFiltersByTarget.find(t.state);
      if (itF == decodedFiltersByTarget.end() || itF->second.empty()) continue;
      auto itTL = findReuseMap(reuse.mergedFromThreadLocal, t.state);
      if (itTL == reuse.mergedFromThreadLocal.end()) continue;
      auto itW = reuse.writerStepsByState.find(itTL->second);
      if (itW == reuse.writerStepsByState.end()) continue;
      for (ExecutionStepOp ws : itW->second) {
         bool hasBufMat = false;
         ws.getOperation()->walk([&](subop::MaterializeOp m) {
            if (mlir::isa<subop::BufferType>(m.getState().getType())) hasBufMat = true;
         });
         if (!hasBufMat) continue;
         if (!bufferWritePredPatchedSteps.insert(ws.getOperation()).second) continue;
         insertWriteSidePredIntoBufferConstructionStep(ws, itF->second);
      }
   }

   auto rewriteOne = [&](mlir::Value state, uint64_t cacheKey) {
      assert(!mlir::isa<ThreadLocalType>(state.getType()) &&
             "rewrite must never target thread_local-wrapped states");

      llvm::SmallVector<mlir::Value, 2> statesToDelete;
      statesToDelete.push_back(state);
      if (auto itTL = findReuseMap(reuse.mergedFromThreadLocal, state);
          itTL != reuse.mergedFromThreadLocal.end()) {
         // If the match is on a merge result, we still must delete the thread_local side steps,
         // but we must never cache that thread_local value.
         statesToDelete.push_back(itTL->second);
      }

      llvm::SmallVector<ExecutionStepOp, 32> writerSteps;
      for (auto st : statesToDelete) {
         auto itW = reuse.writerStepsByState.find(st);
         if (itW == reuse.writerStepsByState.end()) continue;
         for (auto s : itW->second) writerSteps.push_back(s);
      }
      if (writerSteps.empty()) return;

      llvm::ArrayRef<runtime::FilterDescription> decodedFilters;
      if (auto itF = decodedFiltersByTarget.find(state); itF != decodedFiltersByTarget.end()) {
         decodedFilters = itF->second;
      }
      const bool isAggHt = mlir::isa<subop::PreAggrHtType>(state.getType());
      const bool isJoinHm = mlir::isa<subop::HashMapType>(state.getType());
      const bool isJoinBuf = mlir::isa<subop::BufferType>(state.getType());

      auto group = findEnclosingExecutionGroup(state);
      mlir::Value cached;
      if (auto it = keyToCached.find(cacheKey); it != keyToCached.end()) {
         cached = it->second;
      } else {
         cached = insertCacheGetAtExecutionGroupStart(group, state.getType(), cacheKey);
         keyToCached[cacheKey] = cached;
      }

      // Replace the state uses with cached value.
      state.replaceAllUsesWith(cached);

      // Re-apply construction filter at use sites for state kinds that still use stream-level delay filters.
      // Aggregate `optimistic_ht` and join `hashmap` / join `buffer`+`hash_indexed_view` are excluded: aggr ht unchanged;
      // join structures use `filter_pred$0` on scan_refs / probe gathers.
      if (!decodedFilters.empty() && !isAggHt && !isJoinHm && !isJoinBuf) {
         // Collect uses first; we'll mutate IR while iterating.
         llvm::SmallVector<mlir::OpOperand*, 16> uses;
         for (auto& u : cached.getUses()) {
            uses.push_back(&u);
         }
         for (mlir::OpOperand* u : uses) {
            mlir::Operation* owner = u->getOwner();
            // Most uses of a state are as an operand of an `execution_step` op itself; the step body
            // then receives it as a block argument. Handle both cases:
            // - owner is a step-internal op (rare)
            // - owner is `ExecutionStepOp` (common)
            subop::ExecutionStepOp step;
            mlir::Value stateInBody;
            if (auto stepOwner = mlir::dyn_cast<subop::ExecutionStepOp>(owner)) {
               step = stepOwner;
               unsigned argNo = u->getOperandNumber();
               assert(argNo < step.getNumOperands());
               stateInBody = step.getSubOps().front().getArgument(argNo);
            } else {
               step = owner->getParentOfType<subop::ExecutionStepOp>();
               if (!step) continue;
               stateInBody = u->get();
            }
            mlir::Block& body = step.getSubOps().front();

            // We want to inject filter *before* the actual consumer of the cached state in the step body.
            mlir::Operation* useOp = drillToStateConsumerSkippingNestedScopes(body, stateInBody);
            if (!useOp) {
               // No op in this step's body consumes the state SSA (unusual); skip delay_filter here.
               continue;
            }

            // Find the TupleStream operand that feeds this useOp.
            mlir::OpOperand* streamOperand = nullptr;
            for (mlir::OpOperand& ou : useOp->getOpOperands()) {
               if (mlir::isa<tuples::TupleStreamType>(ou.get().getType())) {
                  streamOperand = &ou;
                  break;
               }
            }
            // Two supported consumer shapes:
            // 1) consumer has a TupleStream operand (e.g. lookup_or_insert) → filter that stream
            // 2) consumer is scan_refs(state) → filter the first gathered stream produced from its result
            mlir::Value inStream;
            subop::GatherOp gatherOp;
            if (streamOperand) {
               inStream = streamOperand->get();
               gatherOp = mlir::dyn_cast_or_null<subop::GatherOp>(inStream.getDefiningOp());
               if (!gatherOp) {
                  // Stream may be produced by map/filter/etc.; skip delay_filter for this use.
                  continue;
               }
            } else if (auto scan = mlir::dyn_cast<subop::ScanRefsOp>(useOp)) {
               assert(scan->getNumResults() == 1);
               mlir::Value scanStream = scan.getResult();
               mlir::Block* scanBlock = scan->getBlock();
               for (mlir::Operation& op : scanBlock->without_terminator()) {
                  if (!scan->isBeforeInBlock(&op)) continue;
                  auto g = mlir::dyn_cast<subop::GatherOp>(&op);
                  if (!g) continue;
                  if (g.getStream() == scanStream) { gatherOp = g; break; }
               }
               if (!gatherOp) continue;
            } else {
               // e.g. state-only consumer; skip delay_filter for this use site.
               continue;
            }

            // Build columnName -> ColumnRefAttr mapping from gather's column defs.
            llvm::DenseMap<llvm::StringRef, tuples::ColumnRefAttr> colByName;
            auto& cm = gatherOp->getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
            for (auto& pair : gatherOp.getMapping().getMapping()) {
               auto def = pair.second;
               auto [scope, leaf] = cm.getName(&def.getColumn());
               (void)scope;
               colByName[llvm::StringRef(leaf)] = cm.createRef(&def.getColumn());
            }

            // If the gathered stream is missing columns required by the runtime filter, try to
            // gather them from the table entry ref (still available in the stream via the ref
            // column) by inserting an extra gather right after the existing one.
            //
            // This is the key "columns may have been dropped" scenario for optimistic_ht-like
            // constructions: filter lives on the table, but the stream driving lookup/insert only
            // gathers key columns. We must re-gather the filter columns before applying it.
            bool anyMissing = false;
            for (auto& f : decodedFilters) {
               if (f.op == runtime::FilterOp::NOTNULL) continue;
               if (colByName.contains(f.columnName)) continue;
               anyMissing = true;
               break;
            }
            if (anyMissing) {
               // For scan_refs-based use sites (i.e., reading from a reused state), we currently
               // require the filter columns to already be present in the gathered stream.
               // If they are missing (e.g. optimistic_ht doesn't store shipdate), we cannot re-apply
               // the filter here yet. Just skip (future work: insert needed columns into the state).
               if (!streamOperand) {
                  continue;
               }

               // Find a table `scan_refs` in the same block before `gatherOp` (join plans may nest
               // the gather under `nested_execution_group`, outside the top-level step body).
               subop::ScanRefsOp scanOp;
               mlir::Block* gBlock = gatherOp->getBlock();
               for (mlir::Operation& op : gBlock->without_terminator()) {
                  if (&op == gatherOp.getOperation()) break;
                  if (auto s = mlir::dyn_cast<subop::ScanRefsOp>(&op)) {
                     if (mlir::isa<subop::TableType>(s.getState().getType())) scanOp = s;
                  }
               }
               assert(scanOp && "delay_filter: missing filter columns but could not find scan_refs before gather");
               auto tableTy = mlir::dyn_cast<subop::TableType>(scanOp.getState().getType());
               assert(tableTy && "delay_filter: scan_refs state must be a subop.table");

               // Pick a stable scope for new gathered columns: reuse the scope of any existing def.
               llvm::StringRef reusedScope = "delay_filter";
               if (!gatherOp.getMapping().getMapping().empty()) {
                  auto def0 = gatherOp.getMapping().getMapping().begin()->second;
                  auto [scope0, leaf0] = cm.getName(&def0.getColumn());
                  (void)leaf0;
                  reusedScope = scope0;
               }

               // Collect new mapping entries for missing filter columns.
               auto& mm = gatherOp->getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
               llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>, 8> extra;
               auto stripSuffix = [](llvm::StringRef s) -> llvm::StringRef {
                  size_t pos = s.find('$');
                  if (pos == llvm::StringRef::npos) return s;
                  return s.take_front(pos);
               };
               for (auto& f : decodedFilters) {
                  if (f.op == runtime::FilterOp::NOTNULL) continue;
                  if (colByName.contains(f.columnName)) continue;

                  subop::Member mem;
                  for (auto m : tableTy.getMembers().getMembers()) {
                     if (stripSuffix(mm.getName(m)) == f.columnName) {
                        mem = m;
                        break;
                     }
                  }
                  assert(mem && "delay_filter: could not find table member for filter column");

                  tuples::ColumnDefAttr def = cm.createDef(reusedScope.str(), f.columnName);
                  def.getColumn().type = mm.getType(mem);
                  colByName[f.columnName] = cm.createRef(&def.getColumn());
                  extra.push_back({mem, def});
               }

               if (!extra.empty()) {
                  llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> mappingPairs;
                  for (auto& p : gatherOp.getMapping().getMapping()) {
                     mappingPairs.push_back({p.first, p.second});
                  }
                  for (auto& e : extra) mappingPairs.push_back(e);

                  mlir::OpBuilder gb(gatherOp);
                  gb.setInsertionPointAfter(gatherOp);
                  auto newMapping = subop::ColumnDefMemberMappingAttr::get(gatherOp.getContext(), mappingPairs);
                  auto g2 = gb.create<subop::GatherOp>(gatherOp.getLoc(), gatherOp.getRes(),
                                                       gatherOp.getRef(), newMapping);
                  gatherOp = g2;
               }
            }

            mlir::OpBuilder builder(gatherOp);
            builder.setInsertionPointAfter(gatherOp);
            mlir::Value filteredStream =
               materializeRuntimeFiltersAsSubopFilter(builder, gatherOp.getLoc(), gatherOp.getRes(), colByName, decodedFilters);

            if (streamOperand) {
               // Feed the filtered stream into the cached-state consumer.
               streamOperand->set(filteredStream);
            } else {
               // scan_refs path: rewrite uses of the gathered stream after insertion point.
               mlir::Value orig = gatherOp.getRes();
               mlir::Block* gatherBlock = gatherOp->getBlock();
               orig.replaceUsesWithIf(filteredStream, [&](mlir::OpOperand& ou) {
                  mlir::Operation* ouOwner = ou.getOwner();
                  if (ouOwner == filteredStream.getDefiningOp()) return false;
                  if (ouOwner->getBlock() != gatherBlock) return false;
                  return gatherOp->isBeforeInBlock(ouOwner);
               });
            }
         }
      }

      if (isJoinHm && !decodedFilters.empty()) {
         subop::Member predMember = makeOrGetPredMember(state.getContext());
         for (auto& u : cached.getUses()) {
            auto step = mlir::dyn_cast<subop::ExecutionStepOp>(u.getOwner());
            if (!step) continue;
            insertScanRefsPredFilter(step, predMember);
         }
      }

      if (isJoinBuf && !decodedFilters.empty()) {
         if (joinBufPredProbeInjectedGroups.insert(group.getOperation()).second) {
            subop::Member predMember = makeOrGetPredMember(state.getContext());
            for (mlir::Operation& op : group.getSubOps().front()) {
               if (auto step = mlir::dyn_cast<subop::ExecutionStepOp>(&op)) {
                  insertHashIndexedViewGatherPredFilters(step, predMember);
               }
            }
         }
      }

      // Delete create steps too (create_only steps are part of construction but may not be marked as writes).
      for (auto st : statesToDelete) {
         if (auto defStep = mlir::dyn_cast_or_null<ExecutionStepOp>(st.getDefiningOp())) {
            auto* op = defStep.getOperation();
            if (uniqOpsToErase.insert(op).second) opsToErase.push_back(op);
         }
      }
      for (auto s : writerSteps) {
         auto* op = s.getOperation();
         if (uniqOpsToErase.insert(op).second) opsToErase.push_back(op);
      }
   };

   for (auto& t : targets) {
      if (!t.state) continue;
      rewriteOne(t.state, t.cacheKey);
   }

   // Actually erase the now-dead construction / writer steps.
   // Only erase `execution_step` ops that are direct children of the module's top-level
   // `execution_group`. Walk the group block in reverse: pipeline steps that appear later in the
   // block are erased first, so producers (earlier construction) still exist while their users are
   // removed (TPCH Q2 has long SSA chains across many steps).
   if (!opsToErase.empty()) {
      ExecutionGroupOp eg = getSingleExecutionGroup(consumerModule);
      mlir::Operation* egOp = eg.getOperation();
      mlir::Block& egBody = eg.getSubOps().front();
      llvm::DenseSet<mlir::Operation*> eraseSet;
      for (mlir::Operation* o : opsToErase) {
         if (!mlir::isa<ExecutionStepOp>(o)) continue;
         if (o->getParentRegion()->getParentOp() != egOp) continue;
         if (o->getBlock() != &egBody) continue;
         eraseSet.insert(o);
      }
      // Erasing construction steps after `cache_get` is still being hardened for join-shaped
      // pipelines (TPCH Q2). Until then, set `LINGODB_REUSE_ERASE=1` to enable erasure.
      if (std::getenv("LINGODB_REUSE_ERASE")) {
         for (mlir::Operation& op : llvm::make_early_inc_range(llvm::reverse(egBody.without_terminator()))) {
            if (auto es = mlir::dyn_cast<ExecutionStepOp>(&op)) {
               if (eraseSet.contains(es.getOperation())) es.erase();
            }
         }
      }
   }
}

} // namespace

ReusePlanRewriteResult rewritePlansWithSyntheticQuery0(
   mlir::ModuleOp query0,
   mlir::ModuleOp query1,
   llvm::ArrayRef<CrossQueryStateMatchPair> matches) {
   ReusePlanRewriteResult res;

   llvm::SmallVector<CacheTarget, 64> targets0;
   llvm::SmallVector<CacheTarget, 64> targets1;
   targets0.reserve(matches.size());
   targets1.reserve(matches.size());

   for (auto& m : matches) {
      if (m.stateA) {
         assert(!mlir::isa<ThreadLocalType>(m.stateA.getType()) &&
                "match pairs must never target thread_local-wrapped states; match should use merge result");
         targets0.push_back(CacheTarget{m.stateA, m.cacheKey});
      }
      if (m.stateB) {
         assert(!mlir::isa<ThreadLocalType>(m.stateB.getType()) &&
                "match pairs must never target thread_local-wrapped states; match should use merge result");
         targets1.push_back(CacheTarget{m.stateB, m.cacheKey});
      }
   }

   res.numTargetsQuery0 = targets0.size();
   res.numTargetsQuery1 = targets1.size();

   if (targets0.empty() || targets1.empty()) {
      return res;
   }

   // Build synthetic query0 module from scratch in the same context as query0, then clone steps into it.
   mlir::MLIRContext* q0Ctx = query0.getContext();
   res.query0 = mlir::OwningOpRef<mlir::ModuleOp>(mlir::ModuleOp::create(mlir::UnknownLoc::get(q0Ctx)));

   auto donorGroup = getSingleExecutionGroup(query0);
   auto donorMain = query0.lookupSymbol<mlir::func::FuncOp>("main");
   assert(donorMain && "expected func @main");

   auto q0Main = mlir::func::FuncOp::create(donorMain.getLoc(), "main", donorMain.getFunctionType());
   res.query0->push_back(q0Main);
   auto* entry = q0Main.addEntryBlock();
   mlir::OpBuilder fb = mlir::OpBuilder::atBlockBegin(entry);

   auto q0Group = fb.create<ExecutionGroupOp>(donorGroup.getLoc(), mlir::TypeRange{}, mlir::ValueRange{});
   auto& q0Block = q0Group.getSubOps().emplaceBlock();
   mlir::OpBuilder gb = mlir::OpBuilder::atBlockBegin(&q0Block);
   gb.create<ExecutionGroupReturnOp>(donorGroup.getLoc(), mlir::ValueRange{});
   fb.create<mlir::func::ReturnOp>(donorMain.getLoc());

   // Clone construction into synthetic query0: seed **states** (matched targets), BFS to all
   // prerequisite states via reads on writer steps (`ModuleReuseInfo`), then take every state's
   // defining step, `createOnlyStepForState`, and all writer steps (`writerStepsByState`), then SSA
   // predecessors for remaining operand edges (merge, hash views, …).
   auto reuse0 = collectModuleReuseInfo(query0);
   llvm::DenseSet<mlir::Value> neededStates = expandNeededStatesFromTargets(targets0, reuse0);
   auto stepsToClone = collectCreateAndWriteStepsForStates(donorGroup, reuse0, neededStates);

   // Join-buffer descr filters: mutate the **donor** query0 before cloning so cloned synthetic steps
   // already materialize `filter_pred$0` (synthetic modules may not register the same writer-step index
   // keys as the donor for `insertCachePutsForTargets` to rediscover the construction step).
   {
      llvm::SmallVector<mlir::Value, 8> bufferCandidates;
      for (auto& t : targets0) {
         if (mlir::isa<subop::BufferType>(t.state.getType())) bufferCandidates.push_back(t.state);
      }
      llvm::SmallVector<mlir::Value, 8> extendBufStates =
         collectJoinBuffersFeedingHashIndexedView(bufferCandidates, reuse0);
      if (!extendBufStates.empty()) {
         rewriteHashmapTypesInModule(query0, extendBufStates);
         llvm::DenseSet<mlir::Operation*> stepSet;
         for (ExecutionStepOp s : stepsToClone) stepSet.insert(s.getOperation());
         for (auto& t : targets0) {
            if (!mlir::isa<subop::BufferType>(t.state.getType())) continue;
            auto itTL = reuse0.mergedFromThreadLocal.find(t.state);
            if (itTL == reuse0.mergedFromThreadLocal.end()) continue;
            llvm::SmallVector<runtime::FilterDescription, 8> dec =
               decodeFiltersForStateFromWriterSteps(itTL->second, reuse0);
            if (dec.empty()) continue;
            auto itW = reuse0.writerStepsByState.find(itTL->second);
            if (itW == reuse0.writerStepsByState.end()) continue;
            for (ExecutionStepOp ws : itW->second) {
               if (!stepSet.contains(ws.getOperation())) continue;
               bool hasBufMat = false;
               ws.getOperation()->walk([&](subop::MaterializeOp m) {
                  if (mlir::isa<subop::BufferType>(m.getState().getType())) hasBufMat = true;
               });
               if (!hasBufMat) continue;
               insertWriteSidePredIntoBufferConstructionStep(ws, dec);
            }
         }
      }
   }

   auto mapping = cloneExecutionStepsToQuery0(q0Group, stepsToClone);

   // Map targets into query0.
   llvm::SmallVector<CacheTarget, 64> targetsQ0;
   targetsQ0.reserve(targets0.size());
   for (auto& t : targets0) {
      if (!t.state) continue;
      mlir::Value mapped = mapping.lookupOrNull(t.state);
      if (!mapped) {
         // If the matched state value is a canonical step input (BlockArgument), cache the value produced
         // by the last writer step for this state.
         if (!t.state.getDefiningOp()) {
            if (auto itW = reuse0.writerStepsByState.find(t.state); itW != reuse0.writerStepsByState.end()) {
               if (!itW->second.empty()) {
                  auto lastWriter = itW->second.back();
                  if (lastWriter.getNumResults() == 1) {
                     mapped = mapping.lookupOrNull(lastWriter.getResult(0));
                  }
               }
            }
         }
      }
      if (!mapped) continue;
      targetsQ0.push_back(CacheTarget{mapped, t.cacheKey});
   }
   res.numTargetsQuery0Mapped = targetsQ0.size();
   if (targetsQ0.empty()) {
      return res;
   }

   // Producer: cache_puts.
   insertCachePutsForTargets(*res.query0, targetsQ0);

   // Consumers: cache_get + delete construction steps.
   injectCacheGetsAndDeleteConstructionSteps(query0, targets0);
   injectCacheGetsAndDeleteConstructionSteps(query1, targets1);

   return res;
}

} // namespace lingodb::compiler::dialect::subop

