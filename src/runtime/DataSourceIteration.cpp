#include "lingodb/runtime/DataSourceIteration.h"
//TODO remove
#include "../../include/lingodb/runtime/DatasourceRestrictionProperty.h"
#include "json.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/runtime/ExternalDataSourceProperty.h"
#include "lingodb/runtime/storage/TableStorage.h"
#include "lingodb/scheduler/Scheduler.h"
#include "lingodb/scheduler/Tasks.h"
#include <iterator>

#include "lingodb/utility/Tracer.h"

#include <arrow/array.h>
#include <arrow/table.h>
namespace utility = lingodb::utility;
namespace {

static utility::Tracer::Event tableScan("Tablescan", "tableScan");

class TableSource : public lingodb::runtime::DataSource {
   lingodb::runtime::TableStorage& tableStorage;
   std::unordered_map<std::string, std::string> memberToColumn;
   std::vector<lingodb::runtime::FilterDescription> filters;
   std::vector<std::vector<lingodb::runtime::FilterDescription>> orFilterClauses;
   std::vector<std::vector<lingodb::runtime::FilterDescription>> sharedPredicateClauses;

   public:
   TableSource(lingodb::runtime::TableStorage& tableStorage, std::unordered_map<std::string, std::string> memberToColumn,
               std::vector<lingodb::runtime::FilterDescription> filters,
               std::vector<std::vector<lingodb::runtime::FilterDescription>> orFilterClauses,
               std::vector<std::vector<lingodb::runtime::FilterDescription>> sharedPredicateClauses)
      : tableStorage(tableStorage), memberToColumn(memberToColumn), filters(std::move(filters)),
        orFilterClauses(std::move(orFilterClauses)), sharedPredicateClauses(std::move(sharedPredicateClauses)) {}
   void iterate(bool parallel, std::vector<std::string> members, bool exportPredicateResults,
                const std::function<void(lingodb::runtime::BatchView*)>& cb) override {
      std::vector<std::string> columns;
      std::vector<size_t> predicateClauseIds;
      const bool exportSharedPredicates = exportPredicateResults && !sharedPredicateClauses.empty();
      size_t numPredicateMembers = exportSharedPredicates ? sharedPredicateClauses.size() : 0;
      assert(members.size() >= numPredicateMembers);
      size_t numDataMembers = members.size() - numPredicateMembers;
      for (size_t i = 0; i < numDataMembers; ++i) {
         const std::string& member = members[i];
         columns.push_back(memberToColumn.at(member));
      }
      for (size_t i = 0; i < numPredicateMembers; ++i)
         predicateClauseIds.push_back(i);
      std::vector<lingodb::runtime::FilterDescription> scanFilters = filters;
      std::vector<std::vector<lingodb::runtime::FilterDescription>> scanOrClauses = orFilterClauses;
      if (!sharedPredicateClauses.empty()) {
         scanFilters.clear();
         scanOrClauses.clear();
         scanFilters = sharedPredicateClauses.front();
         scanOrClauses.insert(scanOrClauses.end(), std::next(sharedPredicateClauses.begin()),
                              sharedPredicateClauses.end());
      }
      auto scanTask = tableStorage.createScanTask(
         {parallel, columns, std::move(scanFilters), std::move(scanOrClauses),
          exportSharedPredicates, std::move(predicateClauseIds), cb});
      lingodb::scheduler::awaitChildTask(std::move(scanTask));
   }
};
} // end namespace

void lingodb::runtime::DataSourceIteration::end(DataSourceIteration* iteration) {
   delete iteration;
}

lingodb::runtime::DataSourceIteration* lingodb::runtime::DataSourceIteration::init(DataSource* dataSource, lingodb::runtime::VarLen32 rawMembers) {
   //TODO remove init
   nlohmann::json descr = nlohmann::json::parse(rawMembers.str());
   std::vector<std::string> members;
   for (std::string c : descr.get<nlohmann::json::array_t>()) {
      members.push_back(c);
   }
   auto* it = new DataSourceIteration(dataSource, members, false);
   getCurrentExecutionContext()->registerState({it, [](void* ptr) { delete reinterpret_cast<DataSourceIteration*>(ptr); }});
   return it;
}
lingodb::runtime::DataSourceIteration* lingodb::runtime::DataSourceIteration::initShared(DataSource* dataSource, lingodb::runtime::VarLen32 rawMembers) {
   nlohmann::json descr = nlohmann::json::parse(rawMembers.str());
   std::vector<std::string> members;
   for (std::string c : descr.get<nlohmann::json::array_t>()) {
      members.push_back(c);
   }
   auto* it = new DataSourceIteration(dataSource, members, true);
   getCurrentExecutionContext()->registerState({it, [](void* ptr) { delete reinterpret_cast<DataSourceIteration*>(ptr); }});
   return it;
}
lingodb::runtime::DataSourceIteration::DataSourceIteration(DataSource* dataSource, const std::vector<std::string>& members,
                                                           bool exportPredicateResults)
   : dataSource(dataSource), members(members), exportPredicateResults(exportPredicateResults) {
}

lingodb::runtime::DataSource* lingodb::runtime::DataSource::get(lingodb::runtime::VarLen32 description) {
   lingodb::runtime::ExecutionContext* executionContext = lingodb::runtime::getCurrentExecutionContext();
   std::string tableName;
   std::unordered_map<std::string, std::string> memberToColumn;
   std::unordered_set<FilterDescription> uniqueRestrictions;
   std::vector<FilterDescription> filters;

   std::string dataSourceRaw = description.str();
   auto dataSource = utility::deserializeFromHexString<ExternalDatasourceProperty>(dataSourceRaw);
   tableName = dataSource.tableName;
   for (auto& filterDesc : dataSource.filterDescriptions) {
      if (uniqueRestrictions.contains(filterDesc)) {
         continue;
      }
      uniqueRestrictions.insert(filterDesc);
      filters.push_back(filterDesc);
   }
   std::vector<std::vector<FilterDescription>> orFilterClauses = std::move(dataSource.orFilterClauses);
   std::vector<std::vector<FilterDescription>> sharedPredicateClauses =
      std::move(dataSource.sharedPredicateClauses);
   for (auto& mapping : dataSource.mapping) {
      memberToColumn[mapping.memberName] = mapping.identifier;
   }
   auto& session = executionContext->getSession();
   if (auto maybeRelation = session.getCatalog()->getTypedEntry<catalog::TableCatalogEntry>(tableName)) {
      auto relation = maybeRelation.value();

      auto* ts = new TableSource(relation->getTableStorage(), memberToColumn, std::move(filters),
                                 std::move(orFilterClauses), std::move(sharedPredicateClauses));
      getCurrentExecutionContext()->registerState({ts, [](void* ptr) { delete reinterpret_cast<TableSource*>(ptr); }});
      return ts;

   } else {
      throw std::runtime_error("could not find relation");
   }
}

void lingodb::runtime::DataSourceIteration::iterate(bool parallel, void (*forEachChunk)(lingodb::runtime::BatchView*, void*), void* context) {
   utility::Tracer::Trace trace(tableScan);
   dataSource->iterate(parallel, members, exportPredicateResults, [context, forEachChunk](lingodb::runtime::BatchView* recordBatchInfo) {
      forEachChunk(recordBatchInfo, context);
   });
   trace.stop();
}
