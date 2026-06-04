#include "lingodb/compiler/Dialect/RelAlg/Transforms/CardinalityEstimation.h"

#include "lingodb/catalog/Catalog.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/compiler/mlir-support/eval.h"
#include "lingodb/compiler/mlir-support/parsing.h"
#include "lingodb/runtime/ExternalDataSourceProperty.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cassert>

#include <arrow/api.h>

namespace lingodb::compiler::dialect::relalg {
namespace {

static bool filterDescrLess(const lingodb::runtime::FilterDescription& a,
                            const lingodb::runtime::FilterDescription& b) {
   if (a.columnName != b.columnName) return a.columnName < b.columnName;
   if (a.columnId != b.columnId) return a.columnId < b.columnId;
   if (a.op != b.op) return static_cast<uint8_t>(a.op) < static_cast<uint8_t>(b.op);
   if (a.value != b.value) return a.value < b.value;
   return a.values < b.values;
}

static bool filterClauseEquals(llvm::ArrayRef<lingodb::runtime::FilterDescription> a,
                               llvm::ArrayRef<lingodb::runtime::FilterDescription> b) {
   if (a.size() != b.size()) return false;
   llvm::SmallVector<lingodb::runtime::FilterDescription, 8> sa(a.begin(), a.end());
   llvm::SmallVector<lingodb::runtime::FilterDescription, 8> sb(b.begin(), b.end());
   llvm::sort(sa, filterDescrLess);
   llvm::sort(sb, filterDescrLess);
   return std::equal(sa.begin(), sa.end(), sb.begin());
}

static bool allExternalFilterSourcesIdentical(
   llvm::ArrayRef<lingodb::runtime::ExternalDatasourceProperty> filterSources) {
   assert(!filterSources.empty());
   const auto& a = filterSources.front();
   for (const auto& b : filterSources.drop_front()) {
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
   }
   return true;
}

static lingodb::runtime::ExternalDatasourceProperty mergeExternalFiltersForOrReuse(
   llvm::ArrayRef<lingodb::runtime::ExternalDatasourceProperty> filterSources) {
   assert(!filterSources.empty());
   lingodb::runtime::ExternalDatasourceProperty merged = filterSources.front();
   if (allExternalFilterSourcesIdentical(filterSources)) {
      merged.filterDescriptions = filterSources.front().filterDescriptions;
      merged.orFilterClauses.clear();
      return merged;
   }

   llvm::SmallVector<llvm::SmallVector<lingodb::runtime::FilterDescription, 8>, 4> uniqueClauses;
   auto tryAddClause = [&](llvm::ArrayRef<lingodb::runtime::FilterDescription> clause) {
      if (clause.empty()) return;
      for (const auto& existing : uniqueClauses) {
         if (filterClauseEquals(existing, clause)) return;
      }
      uniqueClauses.emplace_back(clause.begin(), clause.end());
   };
   for (const auto& src : filterSources) {
      tryAddClause(src.filterDescriptions);
      for (const auto& clause : src.orFilterClauses) tryAddClause(clause);
   }
   merged.filterDescriptions.clear();
   merged.orFilterClauses.clear();
   if (uniqueClauses.empty()) return merged;
   merged.filterDescriptions.assign(uniqueClauses.front().begin(), uniqueClauses.front().end());
   for (size_t i = 1; i < uniqueClauses.size(); ++i) {
      merged.orFilterClauses.emplace_back(uniqueClauses[i].begin(), uniqueClauses[i].end());
   }
   return merged;
}

static std::tuple<::arrow::Type::type, uint32_t, uint32_t>
sampleEvalTypeForField(const std::shared_ptr<::arrow::DataType>& type) {
   switch (type->id()) {
      case ::arrow::Type::BOOL: return {::arrow::Type::BOOL, 0, 0};
      case ::arrow::Type::INT8: return {::arrow::Type::INT8, 0, 0};
      case ::arrow::Type::INT16: return {::arrow::Type::INT16, 0, 0};
      case ::arrow::Type::INT32: return {::arrow::Type::INT32, 0, 0};
      case ::arrow::Type::INT64: return {::arrow::Type::INT64, 0, 0};
      case ::arrow::Type::UINT8: return {::arrow::Type::UINT8, 0, 0};
      case ::arrow::Type::UINT16: return {::arrow::Type::UINT16, 0, 0};
      case ::arrow::Type::UINT32: return {::arrow::Type::UINT32, 0, 0};
      case ::arrow::Type::UINT64: return {::arrow::Type::UINT64, 0, 0};
      case ::arrow::Type::FLOAT: return {::arrow::Type::FLOAT, 0, 0};
      case ::arrow::Type::DOUBLE: return {::arrow::Type::DOUBLE, 0, 0};
      case ::arrow::Type::STRING: return {::arrow::Type::STRING, 0, 0};
      case ::arrow::Type::DATE32: return {::arrow::Type::DATE32, 0, 0};
      case ::arrow::Type::DATE64: return {::arrow::Type::DATE64, 0, 0};
      case ::arrow::Type::TIMESTAMP: {
         auto ts = std::static_pointer_cast<::arrow::TimestampType>(type);
         return {::arrow::Type::TIMESTAMP, static_cast<uint32_t>(ts->unit()), 0};
      }
      case ::arrow::Type::DECIMAL128: {
         auto dec = std::static_pointer_cast<::arrow::Decimal128Type>(type);
         return {::arrow::Type::DECIMAL128, static_cast<uint32_t>(dec->precision()),
                 static_cast<uint32_t>(dec->scale())};
      }
      case ::arrow::Type::FIXED_SIZE_BINARY: {
         auto fixed = std::static_pointer_cast<::arrow::FixedSizeBinaryType>(type);
         return {::arrow::Type::FIXED_SIZE_BINARY, static_cast<uint32_t>(fixed->byte_width()), 0};
      }
      default: llvm_unreachable("sample CE: unsupported sample column type");
   }
}

static std::string filterColumnName(const lingodb::runtime::FilterDescription& f,
                                    const std::shared_ptr<::arrow::Schema>& schema) {
   if (!f.columnName.empty()) return f.columnName;
   assert(f.columnId < static_cast<size_t>(schema->num_fields()) &&
          "sample CE: filter column id must be in sample schema");
   return schema->field(static_cast<int>(f.columnId))->name();
}

static std::variant<int64_t, double, std::string>
parseFilterLiteral(const std::variant<std::string, int64_t, double>& value,
                   const std::tuple<::arrow::Type::type, uint32_t, uint32_t>& type) {
   auto [id, param1, param2] = type;
   std::variant<int64_t, double, std::string> parseArg;
   std::visit([&](const auto& v) { parseArg = v; }, value);
   return lingodb::compiler::support::parse(parseArg, id, param1, param2);
}

static std::unique_ptr<lingodb::compiler::support::eval::expr>
buildFilterLiteral(const std::variant<std::string, int64_t, double>& value,
                   const std::tuple<::arrow::Type::type, uint32_t, uint32_t>& type) {
   return lingodb::compiler::support::eval::createLiteral(parseFilterLiteral(value, type), type);
}

static std::unique_ptr<lingodb::compiler::support::eval::expr>
buildFilterPredicate(const lingodb::runtime::FilterDescription& f,
                     const std::shared_ptr<::arrow::Schema>& schema) {
   using lingodb::runtime::FilterOp;
   namespace eval = lingodb::compiler::support::eval;
   std::string colName = filterColumnName(f, schema);
   auto field = schema->GetFieldByName(colName);
   assert(field && "sample CE: filtered column must exist in sample schema");
   auto type = sampleEvalTypeForField(field->type());

   if (f.op == FilterOp::NOTNULL) {
      return eval::createNot(eval::createIsNull(eval::createAttrRef(colName)));
   }
   if (f.op == FilterOp::IN) {
      std::vector<std::unique_ptr<eval::expr>> disjuncts;
      std::visit([&](const auto& vals) {
         for (const auto& v : vals) {
            std::variant<std::string, int64_t, double> literalValue = v;
            disjuncts.push_back(eval::createEq(eval::createAttrRef(colName), buildFilterLiteral(literalValue, type)));
         }
      },
                 f.values);
      assert(!disjuncts.empty() && "sample CE: IN filters must have at least one value");
      return eval::createOr(disjuncts);
   }

   auto lhs = eval::createAttrRef(colName);
   auto rhs = buildFilterLiteral(f.value, type);
   switch (f.op) {
      case FilterOp::EQ: return eval::createEq(std::move(lhs), std::move(rhs));
      case FilterOp::NEQ: return eval::createNot(eval::createEq(std::move(lhs), std::move(rhs)));
      case FilterOp::LT: return eval::createLt(std::move(lhs), std::move(rhs));
      case FilterOp::LTE: return eval::createLte(std::move(lhs), std::move(rhs));
      case FilterOp::GT: return eval::createGt(std::move(lhs), std::move(rhs));
      case FilterOp::GTE: return eval::createGte(std::move(lhs), std::move(rhs));
      default: llvm_unreachable("sample CE: unexpected filter op");
   }
}

static std::optional<std::unique_ptr<lingodb::compiler::support::eval::expr>>
buildFilterClausePredicate(llvm::ArrayRef<lingodb::runtime::FilterDescription> clause,
                           const std::shared_ptr<::arrow::Schema>& schema) {
   namespace eval = lingodb::compiler::support::eval;
   if (clause.empty()) return std::nullopt;
   std::vector<std::unique_ptr<eval::expr>> conjuncts;
   for (const auto& f : clause) conjuncts.push_back(buildFilterPredicate(f, schema));
   return eval::createAnd(conjuncts);
}

} // namespace

double estimateExternalDatasourceOrRowsFromSample(
   llvm::ArrayRef<lingodb::runtime::ExternalDatasourceProperty> filterSources,
   lingodb::catalog::Catalog& catalog,
   llvm::ArrayRef<relalg::MapOp> mapOps,
   llvm::ArrayRef<relalg::SelectionOp> filterOps) {
   assert(mapOps.empty() && "sample CE: map op predicates are reserved for future implementation");
   assert(filterOps.empty() && "sample CE: filter op predicates are reserved for future implementation");
   assert(!filterSources.empty() && "sample CE: expected at least one filter source");
   for (const auto& src : filterSources) {
      assert(src.tableName == filterSources.front().tableName &&
             "sample CE: all filter sources must scan the same table");
   }
   auto tableEntry = catalog.getTypedEntry<lingodb::catalog::TableCatalogEntry>(filterSources.front().tableName);
   assert(tableEntry && "sample CE: table must exist in catalog");
   auto sample = tableEntry.value()->getSample();
   assert(sample && "sample CE: table must have a sample");
   auto batch = sample.getSampleData();
   assert(batch && batch->num_rows() > 0 && "sample CE: sample must be non-empty");

   for (const auto& src : filterSources) {
      if (src.filterDescriptions.empty() && src.orFilterClauses.empty()) {
         return static_cast<double>(tableEntry.value()->getNumRows());
      }
   }

   auto merged = mergeExternalFiltersForOrReuse(filterSources);
   llvm::SmallVector<llvm::ArrayRef<lingodb::runtime::FilterDescription>, 4> clauses;
   clauses.push_back(merged.filterDescriptions);
   for (const auto& c : merged.orFilterClauses) clauses.push_back(c);
   if (llvm::any_of(clauses, [](auto c) { return c.empty(); })) {
      return static_cast<double>(tableEntry.value()->getNumRows());
   }

   std::vector<std::unique_ptr<lingodb::compiler::support::eval::expr>> disjuncts;
   for (auto clause : clauses) {
      auto expr = buildFilterClausePredicate(clause, batch->schema());
      assert(expr.has_value() && "sample CE: non-empty clause must produce predicate");
      disjuncts.push_back(std::move(expr.value()));
   }
   assert(!disjuncts.empty() && "sample CE: filtered datasource must have clauses");
   auto optionalCount = lingodb::compiler::support::eval::countResults(
      batch, lingodb::compiler::support::eval::createOr(disjuncts));
   assert(optionalCount.has_value() && "sample CE: sample predicate must be evaluable");
   size_t count = optionalCount.value();
   if (count == 0) count = 1;
   return static_cast<double>(tableEntry.value()->getNumRows()) *
      static_cast<double>(count) / static_cast<double>(batch->num_rows());
}

} // namespace lingodb::compiler::dialect::relalg
