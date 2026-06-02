#include "lingodb/runtime/LazyJoinHashtable.h"
#include "lingodb/runtime/GrowingBuffer.h"
#include "lingodb/utility/Tracer.h"

#include <algorithm>
#include <atomic>

namespace {
static lingodb::utility::Tracer::Event buildEvent("HashIndexedView", "build");
} // end namespace
lingodb::runtime::HashIndexedView* lingodb::runtime::HashIndexedView::buildInternal(lingodb::runtime::GrowingBuffer* buffer, bool withPredFlags, size_t filterPred0Offset, size_t filterPred1Offset) {
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
      uint8_t predSelectors = 0;
      if (withPredFlags) {
         if (*(ptr + filterPred0Offset)) predSelectors |= lingodb::runtime::filterPred0Selector;
         if (*(ptr + filterPred1Offset)) predSelectors |= lingodb::runtime::filterPred1Selector;
      }
      do {
         entry->next = lingodb::runtime::untag(current);
         newEntry = withPredFlags ? lingodb::runtime::tagWithFilterPreds(entry, current, hash, predSelectors) : lingodb::runtime::tag(entry, current, hash);
      } while (!slot.compare_exchange_weak(current, newEntry));
   });
   trace.stop();
   return htView;
}
lingodb::runtime::HashIndexedView* lingodb::runtime::HashIndexedView::build(lingodb::runtime::GrowingBuffer* buffer) {
   return buildInternal(buffer, false, 0, 0);
}
lingodb::runtime::HashIndexedView* lingodb::runtime::HashIndexedView::buildWithPredFlags(lingodb::runtime::GrowingBuffer* buffer, size_t filterPred0Offset, size_t filterPred1Offset) {
   return buildInternal(buffer, true, filterPred0Offset, filterPred1Offset);
}
void lingodb::runtime::HashIndexedView::destroy(lingodb::runtime::HashIndexedView* ht) {
   delete ht;
}
lingodb::runtime::HashIndexedView::HashIndexedView(size_t htSize, size_t htMask) : ht(lingodb::runtime::FixedSizedBuffer<Entry*>::createZeroed(htSize)), htMask(htMask) {}
lingodb::runtime::HashIndexedView::~HashIndexedView() {
   lingodb::runtime::FixedSizedBuffer<Entry*>::deallocate(ht, htMask + 1);
}
