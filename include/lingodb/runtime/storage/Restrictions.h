#ifndef LINGODB_RUNTIME_STORAGE_RESTRICTIONS_H
#define LINGODB_RUNTIME_STORAGE_RESTRICTIONS_H
#include "lingodb/runtime/ArrowView.h"
#include "lingodb/runtime/storage/TableStorage.h"
#include <cstddef>
#include <memory>
#include <vector>
namespace lingodb::runtime {
class Filter {
   public:
   virtual size_t filter(size_t len, const uint16_t* currSelVec, uint16_t* nextSelVec, const lingodb::runtime::ArrayView* arrayView, size_t offset) = 0;
   virtual ~Filter() {}
};
class Restrictions {
   /// Each entry is one AND-clause; rows pass if they satisfy any clause (OR).
   std::vector<std::vector<std::pair<std::unique_ptr<lingodb::runtime::Filter>, size_t>>> andClauses;

   std::pair<size_t, uint16_t*> applyAndClause(size_t offset, size_t length, uint16_t* selVec1, uint16_t* selVec2,
                                               const std::vector<std::pair<std::unique_ptr<Filter>, size_t>>& clause,
                                               std::function<const ArrayView*(size_t)> getArrayView) const;

   public:
   std::pair<size_t, uint16_t*> applyFilters(size_t offset, size_t length, uint16_t* selVec1, uint16_t* selVec2, std::function<const ArrayView*(size_t)> getArrayView);
   std::pair<size_t, uint16_t*> applyFiltersWithClauseResults(
      size_t offset, size_t length, uint16_t* selVec1, uint16_t* selVec2,
      std::function<const ArrayView*(size_t)> getArrayView,
      const std::vector<size_t>& predicateClauseIds,
      const std::vector<uint16_t*>& predicateColumns);
   static std::unique_ptr<Restrictions> create(std::vector<FilterDescription> filterDescs, const arrow::Schema& schema);
   /// `clauses[0]` is the conjunctive `filterDescriptions` group; further entries are `orFilterClauses`.
   static std::unique_ptr<Restrictions> createFromFilterClauses(
      std::vector<std::vector<FilterDescription>> clauses, const arrow::Schema& schema);
};
} // namespace lingodb::runtime

#endif //LINGODB_RUNTIME_STORAGE_RESTRICTIONS_H
