#include "lingodb/compiler/Dialect/RelAlg/Transforms/CardinalityEstimation.h"

#include "lingodb/catalog/Catalog.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
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
namespace eval = lingodb::compiler::support::eval;

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
   if (clause.empty()) return std::nullopt;
   std::vector<std::unique_ptr<eval::expr>> conjuncts;
   for (const auto& f : clause) conjuncts.push_back(buildFilterPredicate(f, schema));
   return eval::createAnd(conjuncts);
}

static std::optional<std::unique_ptr<eval::expr>>
buildExternalDatasourcePredicate(const lingodb::runtime::ExternalDatasourceProperty& src,
                                 const std::shared_ptr<::arrow::Schema>& schema) {
   llvm::SmallVector<llvm::ArrayRef<lingodb::runtime::FilterDescription>, 4> clauses;
   if (!src.filterDescriptions.empty()) clauses.push_back(src.filterDescriptions);
   for (const auto& c : src.orFilterClauses) clauses.push_back(c);
   if (clauses.empty()) return std::nullopt;
   if (llvm::any_of(clauses, [](auto c) { return c.empty(); })) return std::nullopt;

   std::vector<std::unique_ptr<eval::expr>> disjuncts;
   for (auto clause : clauses) {
      auto expr = buildFilterClausePredicate(clause, schema);
      assert(expr.has_value() && "sample CE: non-empty clause must produce predicate");
      disjuncts.push_back(std::move(expr.value()));
   }
   return eval::createOr(disjuncts);
}

static std::unique_ptr<eval::expr>
buildConstant(mlir::Type type, std::variant<int64_t, double, std::string> parseArg) {
   ::arrow::Type::type typeConstant = ::arrow::Type::type::NA;
   uint32_t param1 = 0, param2 = 0;
   if (isIntegerType(type, 1)) {
      typeConstant = ::arrow::Type::type::BOOL;
   } else if (auto intWidth = getIntegerWidth(type, false)) {
      switch (intWidth) {
         case 8: typeConstant = ::arrow::Type::type::INT8; break;
         case 16: typeConstant = ::arrow::Type::type::INT16; break;
         case 32: typeConstant = ::arrow::Type::type::INT32; break;
         case 64: typeConstant = ::arrow::Type::type::INT64; break;
      }
   } else if (auto uIntWidth = getIntegerWidth(type, true)) {
      switch (uIntWidth) {
         case 8: typeConstant = ::arrow::Type::type::UINT8; break;
         case 16: typeConstant = ::arrow::Type::type::UINT16; break;
         case 32: typeConstant = ::arrow::Type::type::UINT32; break;
         case 64: typeConstant = ::arrow::Type::type::UINT64; break;
      }
   } else if (auto decimalType = mlir::dyn_cast_or_null<db::DecimalType>(type)) {
      typeConstant = ::arrow::Type::type::DECIMAL128;
      param1 = decimalType.getP();
      param2 = decimalType.getS();
   } else if (auto floatType = mlir::dyn_cast_or_null<mlir::FloatType>(type)) {
      switch (floatType.getWidth()) {
         case 16: typeConstant = ::arrow::Type::type::HALF_FLOAT; break;
         case 32: typeConstant = ::arrow::Type::type::FLOAT; break;
         case 64: typeConstant = ::arrow::Type::type::DOUBLE; break;
      }
   } else if (mlir::isa<db::StringType>(type)) {
      typeConstant = ::arrow::Type::type::STRING;
   } else if (auto dateType = mlir::dyn_cast_or_null<db::DateType>(type)) {
      typeConstant = dateType.getUnit() == db::DateUnitAttr::day ? ::arrow::Type::type::DATE32
                                                                 : ::arrow::Type::type::DATE64;
   } else if (auto charType = mlir::dyn_cast_or_null<db::CharType>(type)) {
      if (charType.getLen() <= 1) {
         typeConstant = ::arrow::Type::type::FIXED_SIZE_BINARY;
         param1 = charType.getLen() * 4;
      } else {
         typeConstant = ::arrow::Type::type::STRING;
      }
   } else if (auto timestampType = mlir::dyn_cast_or_null<db::TimestampType>(type)) {
      typeConstant = ::arrow::Type::type::TIMESTAMP;
      param1 = static_cast<uint32_t>(timestampType.getUnit());
   }
   assert(typeConstant != ::arrow::Type::type::NA && "sample CE: unsupported constant type");

   auto parseResult = lingodb::compiler::support::parse(parseArg, typeConstant, param1);
   return eval::createLiteral(parseResult, std::make_tuple(typeConstant, param1, param2));
}

static std::unique_ptr<eval::expr>
buildEvalExpr(mlir::Value val, llvm::DenseMap<mlir::Value, std::string>& mapping) {
   if (auto it = mapping.find(val); it != mapping.end()) return eval::createAttrRef(it->second);
   mlir::Operation* op = val.getDefiningOp();
   assert(op && "sample CE: residual predicate value must be an op result or mapped block argument");
   if (auto constantOp = mlir::dyn_cast<db::ConstantOp>(op)) {
      std::variant<int64_t, double, std::string> parseArg;
      if (auto integerAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(constantOp.getValue())) {
         parseArg = integerAttr.getInt();
      } else if (auto floatAttr = mlir::dyn_cast_or_null<mlir::FloatAttr>(constantOp.getValue())) {
         parseArg = floatAttr.getValueAsDouble();
      } else if (auto stringAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(constantOp.getValue())) {
         parseArg = stringAttr.str();
      } else {
         llvm_unreachable("sample CE: unsupported constant attribute");
      }
      return buildConstant(getBaseType(constantOp.getType()), parseArg);
   }
   if (auto cmpOp = mlir::dyn_cast<db::CmpOp>(op)) {
      if (cmpOp.isEqualityPred(false))
         return eval::createEq(buildEvalExpr(cmpOp.getLeft(), mapping), buildEvalExpr(cmpOp.getRight(), mapping));
      if (cmpOp.isLessPred(false))
         return eval::createLt(buildEvalExpr(cmpOp.getLeft(), mapping), buildEvalExpr(cmpOp.getRight(), mapping));
      if (cmpOp.isGreaterPred(false))
         return eval::createGt(buildEvalExpr(cmpOp.getLeft(), mapping), buildEvalExpr(cmpOp.getRight(), mapping));
      if (cmpOp.isLessPred(true))
         return eval::createLte(buildEvalExpr(cmpOp.getLeft(), mapping), buildEvalExpr(cmpOp.getRight(), mapping));
      if (cmpOp.isGreaterPred(true))
         return eval::createGte(buildEvalExpr(cmpOp.getLeft(), mapping), buildEvalExpr(cmpOp.getRight(), mapping));
      if (cmpOp.isUnequalityPred())
         return eval::createNot(eval::createEq(buildEvalExpr(cmpOp.getLeft(), mapping),
                                               buildEvalExpr(cmpOp.getRight(), mapping)));
      llvm_unreachable("sample CE: unsupported compare predicate");
   }
   if (auto betweenOp = mlir::dyn_cast<db::BetweenOp>(op)) {
      std::vector<std::unique_ptr<eval::expr>> expressions;
      expressions.push_back(betweenOp.getLowerInclusive()
                               ? eval::createGte(buildEvalExpr(betweenOp.getVal(), mapping),
                                                 buildEvalExpr(betweenOp.getLower(), mapping))
                               : eval::createGt(buildEvalExpr(betweenOp.getVal(), mapping),
                                                buildEvalExpr(betweenOp.getLower(), mapping)));
      expressions.push_back(betweenOp.getUpperInclusive()
                               ? eval::createLte(buildEvalExpr(betweenOp.getVal(), mapping),
                                                 buildEvalExpr(betweenOp.getUpper(), mapping))
                               : eval::createLt(buildEvalExpr(betweenOp.getVal(), mapping),
                                                buildEvalExpr(betweenOp.getUpper(), mapping)));
      return eval::createAnd(expressions);
   }
   if (auto oneOfOp = mlir::dyn_cast<db::OneOfOp>(op)) {
      std::vector<std::unique_ptr<eval::expr>> expressions;
      for (auto v : oneOfOp.getVals())
         expressions.push_back(eval::createEq(buildEvalExpr(oneOfOp.getVal(), mapping), buildEvalExpr(v, mapping)));
      return eval::createOr(expressions);
   }
   if (auto notOp = mlir::dyn_cast<db::NotOp>(op)) return eval::createNot(buildEvalExpr(notOp.getVal(), mapping));
   if (auto deriveOp = mlir::dyn_cast<db::DeriveTruth>(op)) return buildEvalExpr(deriveOp.getVal(), mapping);
   if (auto isNullOp = mlir::dyn_cast<db::IsNullOp>(op)) return eval::createIsNull(buildEvalExpr(isNullOp.getVal(), mapping));
   if (auto andOp = mlir::dyn_cast<db::AndOp>(op)) {
      std::vector<std::unique_ptr<eval::expr>> expressions;
      for (auto v : andOp.getVals()) expressions.push_back(buildEvalExpr(v, mapping));
      return eval::createAnd(expressions);
   }
   if (auto orOp = mlir::dyn_cast<db::OrOp>(op)) {
      std::vector<std::unique_ptr<eval::expr>> expressions;
      for (auto v : orOp.getVals()) expressions.push_back(buildEvalExpr(v, mapping));
      return eval::createOr(expressions);
   }
   if (auto runtimeCall = mlir::dyn_cast<db::RuntimeCall>(op)) {
      if (runtimeCall.getFn() == "ConstLike" || runtimeCall.getFn() == "Like") {
         auto constantOp = mlir::dyn_cast<db::ConstantOp>(runtimeCall.getArgs()[1].getDefiningOp());
         assert(constantOp && "sample CE: LIKE pattern must be constant");
         return eval::createLike(buildEvalExpr(runtimeCall.getArgs()[0], mapping),
                                 mlir::cast<mlir::StringAttr>(constantOp.getValue()).str());
      }
   }
   if (auto castOp = mlir::dyn_cast<db::CastOp>(op)) {
      auto fromType = getBaseType(castOp.getVal().getType());
      auto toType = getBaseType(castOp.getRes().getType());
      auto fromExpr = buildEvalExpr(castOp.getVal(), mapping);
      if (fromType == toType) return fromExpr;
      if (auto charType = mlir::dyn_cast_or_null<db::CharType>(fromType);
          charType && mlir::isa<db::StringType>(toType)) {
         assert(charType.getLen() > 1 && "sample CE: char<1> should be stored as fixed binary");
         return fromExpr;
      }
   }
   llvm_unreachable("sample CE: unsupported residual predicate expression op");
}

static std::string sampleColumnNameForMapInput(tuples::ColumnRefAttr ref,
                                               const std::shared_ptr<::arrow::Schema>& schema) {
   auto [scope, leaf] = ref.getContext()
                         ->getLoadedDialect<tuples::TupleStreamDialect>()
                         ->getColumnManager()
                         .getName(&ref.getColumn());
   (void)scope;
   if (schema->GetFieldByName(leaf)) return leaf;
   llvm::StringRef normalized = leaf;
   if (auto pos = normalized.rfind('$'); pos != llvm::StringRef::npos)
      normalized = normalized.take_front(pos);
   assert(schema->GetFieldByName(normalized.str()) &&
          "sample CE: residual predicate input column must exist in sample schema");
   return normalized.str();
}

static std::optional<std::unique_ptr<eval::expr>>
buildResidualPredicate(mlir::Operation* mapOpRaw, mlir::Operation* filterOpRaw,
                       const std::shared_ptr<::arrow::Schema>& schema) {
   if (!mapOpRaw && !filterOpRaw) return std::nullopt;
   auto mapOp = mlir::dyn_cast<subop::MapOp>(mapOpRaw);
   auto filterOp = mlir::dyn_cast<subop::FilterOp>(filterOpRaw);
   assert(mapOp && filterOp && "sample CE: residual predicate must be subop.map + subop.filter");

   llvm::DenseMap<mlir::Value, std::string> mapping;
   mlir::Block& block = mapOp.getFn().front();
   for (unsigned i = 0; i < mapOp.getInputCols().size(); ++i) {
      auto ref = mlir::cast<tuples::ColumnRefAttr>(mapOp.getInputCols()[i]);
      mapping[block.getArgument(i)] = sampleColumnNameForMapInput(ref, schema);
   }

   auto ret = mlir::cast<tuples::ReturnOp>(block.getTerminator());
   std::vector<std::unique_ptr<eval::expr>> predicates;
   for (auto attr : filterOp.getConditions()) {
      auto cond = mlir::cast<tuples::ColumnRefAttr>(attr);
      bool found = false;
      for (unsigned i = 0; i < mapOp.getComputedCols().size(); ++i) {
         auto def = mlir::cast<tuples::ColumnDefAttr>(mapOp.getComputedCols()[i]);
         if (&def.getColumn() != &cond.getColumn()) continue;
         predicates.push_back(buildEvalExpr(ret.getOperand(i), mapping));
         found = true;
         break;
      }
      (void)found;
      assert(found && "sample CE: residual filter condition must be produced by map");
   }
   assert(!predicates.empty() && "sample CE: residual filter must contain predicates");
   return eval::createAnd(predicates);
}

} // namespace

double estimateExternalDatasourceOrRowsFromSample(
   llvm::ArrayRef<lingodb::runtime::ExternalDatasourceProperty> filterSources,
   lingodb::catalog::Catalog& catalog,
   llvm::ArrayRef<mlir::Operation*> mapOps,
   llvm::ArrayRef<mlir::Operation*> filterOps) {
   assert(!filterSources.empty() && "sample CE: expected at least one filter source");
   assert(mapOps.size() == filterOps.size() && "sample CE: map/filter op arrays must align");
   assert((mapOps.empty() || mapOps.size() == filterSources.size()) &&
          "sample CE: predicate op arrays must be empty or aligned with filter sources");
   for (const auto& src : filterSources) {
      (void)src;
      assert(src.tableName == filterSources.front().tableName &&
             "sample CE: all filter sources must scan the same table");
   }
   auto tableEntry = catalog.getTypedEntry<lingodb::catalog::TableCatalogEntry>(filterSources.front().tableName);
   assert(tableEntry && "sample CE: table must exist in catalog");
   auto sample = tableEntry.value()->getSample();
   assert(sample && "sample CE: table must have a sample");
   auto batch = sample.getSampleData();
   assert(batch && batch->num_rows() > 0 && "sample CE: sample must be non-empty");

   std::vector<std::unique_ptr<eval::expr>> disjuncts;
   for (size_t i = 0; i < filterSources.size(); ++i) {
      std::vector<std::unique_ptr<eval::expr>> conjuncts;
      auto sourceExpr = buildExternalDatasourcePredicate(filterSources[i], batch->schema());
      if (sourceExpr) conjuncts.push_back(std::move(sourceExpr.value()));
      if (!mapOps.empty()) {
         auto residualExpr = buildResidualPredicate(mapOps[i], filterOps[i], batch->schema());
         if (residualExpr) conjuncts.push_back(std::move(residualExpr.value()));
      }
      if (conjuncts.empty()) {
         return static_cast<double>(tableEntry.value()->getNumRows());
      }
      disjuncts.push_back(eval::createAnd(conjuncts));
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
