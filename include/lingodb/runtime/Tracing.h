#ifndef LINGODB_RUNTIME_TRACING_H
#define LINGODB_RUNTIME_TRACING_H

#include "helpers.h"
#include <cstdint>

namespace lingodb::runtime {
class ExecutionStepTracing {
   public:
   static uint8_t* start(runtime::VarLen32 step);
   static void end(uint8_t* tracing);
};

class RuntimeScanCpuProfiler {
   public:
   class Scope {
      bool active;
      uint64_t startNs;

      public:
      Scope();
      ~Scope();
   };

   static bool enabled();
   static void reset();
   static uint64_t scanCpuNs();
   static uint64_t scanCalls();
   static uint64_t processCpuNs();
   static void addScanCpuNs(uint64_t ns);
};
}; // namespace lingodb::runtime

#endif //LINGODB_RUNTIME_TRACING_H
