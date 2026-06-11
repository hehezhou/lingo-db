#include "lingodb/compiler/Dialect/TupleStream/ColumnManager.h"
#include <iostream>
#include <llvm/Support/raw_ostream.h>
namespace lingodb::compiler::dialect::tuples {
using namespace mlir;
void ColumnManager::setContext(MLIRContext* context) {
   this->context = context;
}
static std::string makeCombinedKey(StringRef scope, StringRef attribute, Type type = {}) {
   std::string combined = std::string(scope) + "##" + std::string(attribute);
   if (type) {
      std::string typeStr;
      llvm::raw_string_ostream os(typeStr);
      type.print(os);
      combined += "##type=" + os.str();
   }
   return combined;
}
std::shared_ptr<Column> ColumnManager::get(StringRef scope, StringRef attribute) {
   if (!scopeUnifier.contains(std::string(scope))) {
      scopeUnifier[std::string(scope)] = 0;
   }
   std::string combined = makeCombinedKey(scope, attribute);
   auto pair = std::make_pair(std::string(scope), std::string(attribute));
   if (!attributes.count(combined)) {
      auto attr = std::make_shared<Column>();
      attributes[combined] = attr;
      attributesRev[attr.get()] = pair;
      attributesByPtr[attr.get()] = attr;
      return attr;
   }
   return attributes[combined];
}
std::shared_ptr<Column> ColumnManager::get(StringRef scope, StringRef attribute, Type type) {
   if (!type) return get(scope, attribute);
   if (!scopeUnifier.contains(std::string(scope))) {
      scopeUnifier[std::string(scope)] = 0;
   }
   std::string combined = makeCombinedKey(scope, attribute, type);
   auto pair = std::make_pair(std::string(scope), std::string(attribute));
   if (!attributes.count(combined)) {
      auto attr = std::make_shared<Column>();
      attr->type = type;
      attributes[combined] = attr;
      attributesRev[attr.get()] = pair;
      attributesByPtr[attr.get()] = attr;
      return attr;
   }
   return attributes[combined];
}
ColumnDefAttr ColumnManager::createDef(SymbolRefAttr name, Attribute fromExisting) {
   assert(name.getNestedReferences().size() == 1);
   auto attribute = get(name.getRootReference().getValue(), name.getLeafReference().getValue());
   return ColumnDefAttr::get(context, name, attribute, fromExisting);
}
ColumnDefAttr ColumnManager::createDef(StringRef scope, StringRef name, Attribute fromExisting) {
   auto attribute = get(scope, name);
   std::vector<FlatSymbolRefAttr> nested;
   nested.push_back(FlatSymbolRefAttr::get(context, name));
   return ColumnDefAttr::get(context, SymbolRefAttr::get(context, scope, nested), attribute, fromExisting);
}
ColumnDefAttr ColumnManager::createDef(StringRef scope, StringRef name, Type type, Attribute fromExisting) {
   auto attribute = get(scope, name, type);
   std::vector<FlatSymbolRefAttr> nested;
   nested.push_back(FlatSymbolRefAttr::get(context, name));
   return ColumnDefAttr::get(context, SymbolRefAttr::get(context, scope, nested), attribute, fromExisting);
}
ColumnRefAttr ColumnManager::createRef(SymbolRefAttr name) {
   assert(name.getNestedReferences().size() == 1);
   auto attribute = get(name.getRootReference().getValue(), name.getLeafReference().getValue());
   return tuples::ColumnRefAttr::get(context, name, attribute);
}
ColumnRefAttr ColumnManager::createRef(StringRef scope, StringRef name) {
   auto attribute = get(scope, name);
   std::vector<FlatSymbolRefAttr> nested;
   nested.push_back(FlatSymbolRefAttr::get(context, name));
   return tuples::ColumnRefAttr::get(context, SymbolRefAttr::get(context, scope, nested), attribute);
}
ColumnRefAttr ColumnManager::createRef(StringRef scope, StringRef name, Type type) {
   auto attribute = get(scope, name, type);
   std::vector<FlatSymbolRefAttr> nested;
   nested.push_back(FlatSymbolRefAttr::get(context, name));
   return tuples::ColumnRefAttr::get(context, SymbolRefAttr::get(context, scope, nested), attribute);
}
ColumnRefAttr ColumnManager::createRef(const Column* attr) {
   auto [scope, name] = attributesRev[attr];
   std::vector<FlatSymbolRefAttr> nested;
   nested.push_back(FlatSymbolRefAttr::get(context, name));
   return tuples::ColumnRefAttr::get(context, SymbolRefAttr::get(context, scope, nested), attributesByPtr[attr]);
}
ColumnDefAttr ColumnManager::createDef(const Column* attr, Attribute fromExisting) {
   auto [scope, name] = attributesRev[attr];
   std::vector<FlatSymbolRefAttr> nested;
   nested.push_back(FlatSymbolRefAttr::get(context, name));
   return ColumnDefAttr::get(context, SymbolRefAttr::get(context, scope, nested), attributesByPtr[attr], fromExisting);
}

std::pair<std::string, std::string> ColumnManager::getName(const Column* attr) {
   return attributesRev.at(attr);
}
} // namespace lingodb::compiler::dialect::tuples
