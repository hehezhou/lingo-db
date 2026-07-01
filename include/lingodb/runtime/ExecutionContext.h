#ifndef LINGODB_RUNTIME_EXECUTIONCONTEXT_H
#define LINGODB_RUNTIME_EXECUTIONCONTEXT_H
#include <functional>
#include <memory>
#include <optional>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ConcurrentMap.h"
#include "Session.h"
#include <lingodb/scheduler/Scheduler.h>
namespace lingodb::runtime {
class Database;
//some state required for query processing;
struct State {
   void* ptr = nullptr;
   std::function<void(void*)> freeFn;
   std::function<void(void*)> markPermanentFn;
   std::function<void(void*)> flushTransientFn;

   State() = default;
   State(void* ptr, std::function<void(void*)> freeFn) : ptr(ptr), freeFn(std::move(freeFn)) {}
};
struct Arena {
   static constexpr size_t thresholdDirectAlloc = 16 * 1024; // 16 KiB
   static constexpr size_t chunkSize = 1024 * 1024; // 1 MiB
   // chunk allocations + direct allocations
   std::vector<uint8_t*> allocations;
   // remaining from current Chunk
   size_t remainingBytes = 0;
   // pointer to still free space in current chunk
   uint8_t* currentChunkStart = nullptr;

   struct Checkpoint {
      size_t numAllocations = 0;
      size_t remainingBytes = 0;
      uint8_t* currentChunkStart = nullptr;
   };

   uint8_t* alloc(size_t bytes) {
      //case 1: bytes are larger than threshold, allocate directly
      if (bytes >= thresholdDirectAlloc) {
         uint8_t* ptr = (uint8_t*) malloc(bytes);
         allocations.push_back(ptr);
         return ptr;
      } else {
         if (remainingBytes < bytes) {
            //case 2: not enough space in current chunk, allocate new chunk
            uint8_t* newChunk = (uint8_t*) malloc(chunkSize);
            allocations.push_back(newChunk);
            currentChunkStart = newChunk;
            remainingBytes = chunkSize;
         }
         // now we have enough space in the current chunk
         uint8_t* ptr = currentChunkStart;
         currentChunkStart += bytes;
         remainingBytes -= bytes;
         return ptr;
      }
   }

   Checkpoint checkpoint() const {
      return {allocations.size(), remainingBytes, currentChunkStart};
   }

   void rollbackTo(Checkpoint checkpoint) {
      while (allocations.size() > checkpoint.numAllocations) {
         free(allocations.back());
         allocations.pop_back();
      }
      remainingBytes = checkpoint.remainingBytes;
      currentChunkStart = checkpoint.currentChunkStart;
   }

   ~Arena() {
      for (auto& alloc : allocations) {
         free(alloc);
      }
   }
};

class ExecutionContext {
   std::unordered_map<uint32_t, uint8_t*> results;
   std::unordered_map<uint32_t, int64_t> tupleCounts;
   std::vector<std::unordered_map<size_t, State>> allocators;
   std::vector<std::vector<State>> perWorkerStates;
   std::vector<Arena> stringArenas;
   Session& session;
   /// Keys registered via `putCachedState` while this context was current; removed in `~ExecutionContext`
   /// so the global cache does not retain pointers into freed `registerState` storage.
   std::vector<uint64_t> cachedStateKeys;
   struct MemoryCheckpoint {
      std::vector<size_t> perWorkerStateSizes;
      std::vector<Arena::Checkpoint> stringArenaCheckpoints;
      std::vector<std::unordered_map<size_t, State>> allocatorStates;
   };
   MemoryCheckpoint permanentMemoryCheckpoint;
   bool hasPermanentMemoryCheckpoint = false;

   public:
   ExecutionContext(Session& session) : session(session) {
      allocators.resize(lingodb::scheduler::getNumWorkers());
      stringArenas.resize(lingodb::scheduler::getNumWorkers());
      perWorkerStates.resize(lingodb::scheduler::getNumWorkers());
   }
   Session& getSession() {
      return session;
   }
   template <class T>
   std::optional<T*> getResultOfType(uint32_t id) {
      if (results.contains(id)) {
         return (T*) results[id];
      } else {
         return {};
      }
   }
   std::optional<int64_t> getTupleCount(uint32_t id) {
      if (tupleCounts.contains(id)) {
         return tupleCounts[id];
      } else {
         return {};
      }
   }
   const std::unordered_map<uint32_t, int64_t>& getTupleCounts() const {
      return tupleCounts;
   }

   uint8_t* allocString(size_t bytes) {
      return stringArenas[lingodb::scheduler::currentWorkerId()].alloc(bytes);
   }
   static void setResult(uint32_t id, uint8_t* ptr);
   static uint8_t* allocStateRaw(size_t size);
   static void clearResult(uint32_t id);
   static void setTupleCount(uint32_t id, int64_t tupleCount);
   static void putCachedState(uint64_t key, uint8_t* ptr);
   static uint8_t* getCachedState(uint64_t key);
   /// Clears the experimental cross-query `cache_put` / `cache_get` map (not scoped to a single
   /// `ExecutionContext`). Call when cached pointers are no longer valid (e.g. end of a tool run).
   static void clearAllCachedStates();
   void markPermanentMemoryCheckpoint();
   void flushTransientMemory();
   void registerState(const State& s) {
      perWorkerStates[lingodb::scheduler::currentWorkerId()].push_back(s);
   }
   State& getAllocator(size_t group) {
      return allocators[lingodb::scheduler::currentWorkerId()][group];
   }
   ~ExecutionContext();
};

void setCurrentExecutionContext(ExecutionContext* context);
ExecutionContext* getCurrentExecutionContext();
} // end namespace lingodb::runtime

#endif // LINGODB_RUNTIME_EXECUTIONCONTEXT_H
