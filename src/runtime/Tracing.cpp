#include "lingodb/runtime/Tracing.h"
#include "lingodb/utility/Tracer.h"
#include <atomic>
#include <cstdlib>
#include <ctime>
#include <string_view>

namespace {
std::atomic<uint64_t> runtimeScanCpuNs{0};
std::atomic<uint64_t> runtimeScanCalls{0};

uint64_t clockNs(clockid_t clockId) {
   timespec ts{};
   if (clock_gettime(clockId, &ts) != 0) return 0;
   return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}
} // namespace

#ifdef TRACER
static lingodb::utility::Tracer::StringMetaDataEvent executionStepEvent("Execution", "Step", "location", false);
uint8_t* lingodb::runtime::ExecutionStepTracing::start(lingodb::runtime::VarLen32 step) {
   return reinterpret_cast<uint8_t*>(new utility::Tracer::MetaDataTrace<utility::Tracer::StringMetaDataEvent, std::string>(executionStepEvent, step.str()));
}
void lingodb::runtime::ExecutionStepTracing::end(uint8_t* tracing) {
   delete reinterpret_cast<utility::Tracer::MetaDataTrace<utility::Tracer::StringMetaDataEvent, std::string>*>(tracing);
}
#else
uint8_t* lingodb::runtime::ExecutionStepTracing::start(lingodb::runtime::VarLen32 step) {
   return nullptr;
}
void lingodb::runtime::ExecutionStepTracing::end(uint8_t* tracing) {
}

#endif

bool lingodb::runtime::RuntimeScanCpuProfiler::enabled() {
   static bool isEnabled = [] {
      const char* v = std::getenv("LINGODB_SCAN_CPU_PROFILE");
      return v && v[0] != '\0' && std::string_view(v) != "0" && std::string_view(v) != "false" &&
             std::string_view(v) != "off" && std::string_view(v) != "no";
   }();
   return isEnabled;
}

void lingodb::runtime::RuntimeScanCpuProfiler::reset() {
   runtimeScanCpuNs.store(0, std::memory_order_relaxed);
   runtimeScanCalls.store(0, std::memory_order_relaxed);
}

uint64_t lingodb::runtime::RuntimeScanCpuProfiler::scanCpuNs() {
   return runtimeScanCpuNs.load(std::memory_order_relaxed);
}

uint64_t lingodb::runtime::RuntimeScanCpuProfiler::scanCalls() {
   return runtimeScanCalls.load(std::memory_order_relaxed);
}

uint64_t lingodb::runtime::RuntimeScanCpuProfiler::processCpuNs() {
   return clockNs(CLOCK_PROCESS_CPUTIME_ID);
}

void lingodb::runtime::RuntimeScanCpuProfiler::addScanCpuNs(uint64_t ns) {
   runtimeScanCpuNs.fetch_add(ns, std::memory_order_relaxed);
   runtimeScanCalls.fetch_add(1, std::memory_order_relaxed);
}

lingodb::runtime::RuntimeScanCpuProfiler::Scope::Scope()
   : active(RuntimeScanCpuProfiler::enabled()), startNs(active ? clockNs(CLOCK_THREAD_CPUTIME_ID) : 0) {
}

lingodb::runtime::RuntimeScanCpuProfiler::Scope::~Scope() {
   if (!active) return;
   uint64_t endNs = clockNs(CLOCK_THREAD_CPUTIME_ID);
   if (endNs >= startNs) RuntimeScanCpuProfiler::addScanCpuNs(endNs - startNs);
}
