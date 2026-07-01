#include "lingodb/runtime/ExecutionContext.h"
#include <cassert>
#include <mutex>
#include <unordered_map>

void lingodb::runtime::ExecutionContext::setResult(uint32_t id, uint8_t* ptr) {
   auto* context = getCurrentExecutionContext();
   assert(context);
   context->results[id] = ptr;
}
void lingodb::runtime::ExecutionContext::clearResult(uint32_t id) {
   auto* context = getCurrentExecutionContext();
   context->results.erase(id);
}

void lingodb::runtime::ExecutionContext::setTupleCount(uint32_t id, int64_t tupleCount) {
   auto* context = getCurrentExecutionContext();
   context->tupleCounts[id] = tupleCount;
}

namespace {
std::mutex gCachedStatesMutex;
std::unordered_map<uint64_t, uint8_t*> gCachedStates;

void freeState(const lingodb::runtime::State& state) {
   if (state.ptr && state.freeFn) {
      state.freeFn(state.ptr);
   }
}
} // namespace

void lingodb::runtime::ExecutionContext::putCachedState(uint64_t key, uint8_t* ptr) {
   assert(ptr && "putCachedState requires non-null ptr");
   auto* ctx = getCurrentExecutionContext();
   std::lock_guard<std::mutex> lock(gCachedStatesMutex);
   gCachedStates[key] = ptr;
   ctx->cachedStateKeys.push_back(key);
}

uint8_t* lingodb::runtime::ExecutionContext::getCachedState(uint64_t key) {
   std::lock_guard<std::mutex> lock(gCachedStatesMutex);
   auto it = gCachedStates.find(key);
   assert(it != gCachedStates.end() && "getCachedState: missing key (cache not populated?)");
   return it->second;
}

void lingodb::runtime::ExecutionContext::clearAllCachedStates() {
   std::lock_guard<std::mutex> lock(gCachedStatesMutex);
   gCachedStates.clear();
}

void lingodb::runtime::ExecutionContext::markPermanentMemoryCheckpoint() {
   permanentMemoryCheckpoint.perWorkerStateSizes.clear();
   permanentMemoryCheckpoint.perWorkerStateSizes.reserve(perWorkerStates.size());
   for (const auto& states : perWorkerStates) {
      permanentMemoryCheckpoint.perWorkerStateSizes.push_back(states.size());
   }

   permanentMemoryCheckpoint.stringArenaCheckpoints.clear();
   permanentMemoryCheckpoint.stringArenaCheckpoints.reserve(stringArenas.size());
   for (const auto& arena : stringArenas) {
      permanentMemoryCheckpoint.stringArenaCheckpoints.push_back(arena.checkpoint());
   }

   for (auto& workerAllocators : allocators) {
      for (auto& [_, state] : workerAllocators) {
         if (state.ptr && state.markPermanentFn) {
            state.markPermanentFn(state.ptr);
         }
      }
   }
   permanentMemoryCheckpoint.allocatorStates = allocators;
   hasPermanentMemoryCheckpoint = true;
}

void lingodb::runtime::ExecutionContext::flushTransientMemory() {
   if (!hasPermanentMemoryCheckpoint) {
      markPermanentMemoryCheckpoint();
   }

   results.clear();
   tupleCounts.clear();

   for (size_t worker = 0; worker < perWorkerStates.size(); ++worker) {
      size_t checkpointSize = worker < permanentMemoryCheckpoint.perWorkerStateSizes.size()
                                 ? permanentMemoryCheckpoint.perWorkerStateSizes[worker]
                                 : 0;
      auto& states = perWorkerStates[worker];
      while (states.size() > checkpointSize) {
         freeState(states.back());
         states.pop_back();
      }
   }

   for (size_t worker = 0; worker < stringArenas.size(); ++worker) {
      if (worker < permanentMemoryCheckpoint.stringArenaCheckpoints.size()) {
         stringArenas[worker].rollbackTo(permanentMemoryCheckpoint.stringArenaCheckpoints[worker]);
      }
   }

   for (size_t worker = 0; worker < allocators.size(); ++worker) {
      auto& current = allocators[worker];
      const auto& checkpoint = permanentMemoryCheckpoint.allocatorStates[worker];
      for (auto it = current.begin(); it != current.end();) {
         auto checkpointIt = checkpoint.find(it->first);
         if (checkpointIt == checkpoint.end()) {
            freeState(it->second);
            it = current.erase(it);
            continue;
         }
         const State& checkpointState = checkpointIt->second;
         if (it->second.ptr != checkpointState.ptr) {
            freeState(it->second);
            it->second = checkpointState;
         } else if (it->second.ptr && it->second.flushTransientFn) {
            it->second.flushTransientFn(it->second.ptr);
         }
         ++it;
      }
      for (const auto& [group, checkpointState] : checkpoint) {
         current.try_emplace(group, checkpointState);
      }
   }
}

lingodb::runtime::ExecutionContext::~ExecutionContext() {
   {
      std::lock_guard<std::mutex> lock(gCachedStatesMutex);
      for (uint64_t key : cachedStateKeys) {
         gCachedStates.erase(key);
      }
   }
   cachedStateKeys.clear();
   for (auto threadLocal : perWorkerStates) {
      for (auto s : threadLocal) {
         freeState(s);
      }
   }
   for (auto local : allocators) {
      for (auto a : local) {
         freeState(a.second);
      }
   }
   allocators.clear();
   perWorkerStates.clear();
}

uint8_t* lingodb::runtime::ExecutionContext::allocStateRaw(size_t size) {
   auto* context = getCurrentExecutionContext();
   assert(context);
   uint8_t* ptr = static_cast<uint8_t*>(malloc(size));
   context->registerState({ptr, [](void* p) { free(p); }});
   return ptr;
}

namespace {
thread_local lingodb::runtime::ExecutionContext* currentExecutionContext = nullptr;
} // end namespace
void lingodb::runtime::setCurrentExecutionContext(lingodb::runtime::ExecutionContext* context) {
   currentExecutionContext = context;
}

lingodb::runtime::ExecutionContext* lingodb::runtime::getCurrentExecutionContext() {
   assert(currentExecutionContext);
   return currentExecutionContext;
}
