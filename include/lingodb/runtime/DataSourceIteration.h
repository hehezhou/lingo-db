#ifndef LINGODB_RUNTIME_DATASOURCEITERATION_H
#define LINGODB_RUNTIME_DATASOURCEITERATION_H
#include "ArrowTable.h"
#include "ArrowView.h"
#include "lingodb/runtime/ExecutionContext.h"
#include "lingodb/runtime/helpers.h"
#include <cassert>
namespace lingodb::runtime {
class DataSource {
   public:
   virtual void iterate(bool parallel, std::vector<std::string> members,
                        const std::function<void(BatchView*)>& cb) = 0;
   virtual void iterateShared(bool parallel, std::vector<std::string> members,
                              const std::function<void(BatchView*)>& cb) {
      (void)parallel;
      (void)members;
      (void)cb;
      assert(false && "datasource does not support shared predicate scans");
   }
   virtual ~DataSource() {}
   static DataSource* get(runtime::VarLen32 description);
   //static DataSource* getFromTable(ArrowTable* arrowTable, runtime::VarLen32 mappingVal,runtime::VarLen32 columnArray);
};
class DataSourceIteration {
   std::shared_ptr<arrow::RecordBatch> currChunk;
   DataSource* dataSource;
   std::vector<std::string> members;
   bool sharedPredicateScan;

   public:
   DataSourceIteration(DataSource* dataSource, const std::vector<std::string>& members,
                       bool sharedPredicateScan);

   static DataSourceIteration* init(DataSource* dataSource, runtime::VarLen32 members);
   static DataSourceIteration* initShared(DataSource* dataSource, runtime::VarLen32 members);
   static void end(DataSourceIteration*);
   void iterate(bool parallel, void (*forEachChunk)(BatchView*, void*), void*);
};
} // end namespace lingodb::runtime
#endif // LINGODB_RUNTIME_DATASOURCEITERATION_H
