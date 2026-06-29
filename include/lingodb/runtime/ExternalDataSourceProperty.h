#ifndef LINGODB_RUNTIME_EXTERNALDATASOURCEPROPERTY_H
#define LINGODB_RUNTIME_EXTERNALDATASOURCEPROPERTY_H
#include "lingodb/utility/Serialization.h"
#include "storage/TableStorage.h"

#include <cassert>
namespace lingodb::runtime {
struct ExternalDatasourceProperty {
   struct Mapping {
      std::string memberName;
      std::string identifier;
      void serialize(lingodb::utility::Serializer& serializer) const {
         serializer.writeProperty(0, memberName);
         serializer.writeProperty(1, identifier);
      }
      bool operator==(const Mapping& other) const {
         return other.memberName == memberName && other.identifier == identifier;
      }
      static Mapping deserialize(lingodb::utility::Deserializer& deserializer) {
         Mapping map{};
         map.memberName = deserializer.readProperty<std::string>(0);
         map.identifier = deserializer.readProperty<std::string>(1);
         return map;
      }
   };
   std::string tableName;
   std::vector<Mapping> mapping;
   /// Conjunctive filters: `filterDescriptions[0] AND filterDescriptions[1] AND ...`
   std::vector<runtime::FilterDescription> filterDescriptions{};
   /// Disjunctive AND-clauses: overall predicate is
   /// `(filterDescriptions...) OR (orFilterClauses[0]...) OR (orFilterClauses[1]...) OR ...`
   std::vector<std::vector<runtime::FilterDescription>> orFilterClauses{};
   /// Shared-scan clauses are pushed down like OR clauses, and a scan_refs containing
   /// `filter_pred$N` can read the Nth clause result directly from the scan.
   std::vector<std::vector<runtime::FilterDescription>> sharedPredicateClauses{};
   /// Optional logical slot number for each physical shared predicate clause.
   /// When empty, physical slot `i` maps to `filter_pred$i`.
   std::vector<uint64_t> sharedPredicateSlots{};
   std::string index;
   std::string indexType;

   void serialize(lingodb::utility::Serializer& serializer) const {
      serializer.writeProperty(0, tableName);
      serializer.writeProperty(1, mapping);
      serializer.writeProperty(2, filterDescriptions);
      serializer.writeProperty(3, index);
      serializer.writeProperty(4, indexType);
      serializer.writeProperty(5, orFilterClauses);
      serializer.writeProperty(6, sharedPredicateClauses);
      serializer.writeProperty(7, sharedPredicateSlots);
   }
   bool operator==(const ExternalDatasourceProperty& other) const {
      return other.index == index && other.indexType == indexType && other.mapping == mapping && other.tableName == tableName &&
         other.filterDescriptions == filterDescriptions && other.orFilterClauses == orFilterClauses &&
         other.sharedPredicateClauses == sharedPredicateClauses && other.sharedPredicateSlots == sharedPredicateSlots;
   }

   static ExternalDatasourceProperty deserialize(lingodb::utility::Deserializer& deserializer) {
      ExternalDatasourceProperty prop{};
      prop.tableName = deserializer.readProperty<std::string>(0);
      prop.mapping = deserializer.readProperty<std::vector<Mapping>>(1);
      prop.filterDescriptions = deserializer.readProperty<std::vector<runtime::FilterDescription>>(2);
      prop.index = deserializer.readProperty<std::string>(3);
      prop.indexType = deserializer.readProperty<std::string>(4);
      utility::marker_t next = deserializer.readMarker();
      if (next == 5) {
         prop.orFilterClauses =
            deserializer.readSerializedPayload<std::vector<std::vector<runtime::FilterDescription>>>();
         utility::marker_t end = deserializer.readMarker();
         assert(end == 5 && "Expected orFilterClauses property end marker");
         (void)end;
         next = deserializer.readMarker();
      }
      if (next == 6) {
         prop.sharedPredicateClauses =
            deserializer.readSerializedPayload<std::vector<std::vector<runtime::FilterDescription>>>();
         utility::marker_t end = deserializer.readMarker();
         assert(end == 6 && "Expected sharedPredicateClauses property end marker");
         (void)end;
         next = deserializer.readMarker();
      }
      if (next == 7) {
         prop.sharedPredicateSlots =
            deserializer.readSerializedPayload<std::vector<uint64_t>>();
         utility::marker_t end = deserializer.readMarker();
         assert(end == 7 && "Expected sharedPredicateSlots property end marker");
         (void)end;
      } else {
         deserializer.pushBackMarker(next);
      }

      return prop;
   }
};
} // namespace lingodb::runtime

#endif // LINGODB_RUNTIME_EXTERNALDATASOURCEPROPERTY_H
