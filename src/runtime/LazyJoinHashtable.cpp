#include "lingodb/runtime/LazyJoinHashtable.h"
#include "lingodb/runtime/GrowingBuffer.h"
#include "lingodb/utility/Tracer.h"

#include <algorithm>
#include <atomic>
#include <vector>

namespace {
static lingodb::utility::Tracer::Event buildEvent("HashIndexedView", "build");
} // end namespace
lingodb::runtime::HashIndexedView* lingodb::runtime::HashIndexedView::buildInternal(lingodb::runtime::GrowingBuffer* buffer, size_t predSlotCount, const size_t* filterPredOffsets) {
   utility::Tracer::Trace trace(buildEvent);
   auto* executionContext = lingodb::runtime::getCurrentExecutionContext();
   auto& values = buffer->getValues();
   size_t htSize = std::max(nextPow2(values.getLen() * 1.25), static_cast<uint64_t>(1));
   size_t htMask = htSize - 1;
   auto* htView = new HashIndexedView(htSize, htMask);
   executionContext->registerState({htView, [](void* ptr) { delete reinterpret_cast<lingodb::runtime::HashIndexedView*>(ptr); }});
   values.iterateParallel([&](uint8_t* ptr) {
      auto* entry = (Entry*) ptr;
      size_t hash = (size_t) entry->hashValue;
      auto pos = hash & htMask;
      std::atomic_ref<Entry*> slot(htView->ht[pos]);
      Entry* current = slot.load();
      Entry* newEntry;
      thread_local std::vector<uint64_t> predSelectorWords;
      if (predSlotCount) {
         predSelectorWords.assign((predSlotCount + 63) / 64, 0);
         for (size_t predOrdinal = 0; predOrdinal < predSlotCount; ++predOrdinal) {
            if (*(ptr + filterPredOffsets[predOrdinal])) predSelectorWords[predOrdinal / 64] |= (uint64_t{1} << (predOrdinal % 64));
         }
      }
      do {
         entry->next = lingodb::runtime::untag(current);
         newEntry = predSlotCount ? lingodb::runtime::tagWithFilterPredWords(entry, current, hash, predSelectorWords.data(), predSlotCount) : lingodb::runtime::tag(entry, current, hash);
      } while (!slot.compare_exchange_weak(current, newEntry));
   });
   trace.stop();
   return htView;
}
lingodb::runtime::HashIndexedView* lingodb::runtime::HashIndexedView::build(lingodb::runtime::GrowingBuffer* buffer) {
   return buildInternal(buffer, 0, nullptr);
}
lingodb::runtime::HashIndexedView* lingodb::runtime::HashIndexedView::buildWithPredFlagOffsets(lingodb::runtime::GrowingBuffer* buffer, size_t predSlotCount, const size_t* filterPredOffsets) {
   return buildInternal(buffer, predSlotCount, filterPredOffsets);
}
void lingodb::runtime::HashIndexedView::destroy(lingodb::runtime::HashIndexedView* ht) {
   delete ht;
}
lingodb::runtime::HashIndexedView::HashIndexedView(size_t htSize, size_t htMask) : ht(lingodb::runtime::FixedSizedBuffer<Entry*>::createZeroed(htSize)), htMask(htMask) {}
lingodb::runtime::HashIndexedView::~HashIndexedView() {
   lingodb::runtime::FixedSizedBuffer<Entry*>::deallocate(ht, htMask + 1);
}
