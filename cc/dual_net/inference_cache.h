// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CC_DUAL_NET_INFERENCE_CACHE_H_
#define CC_DUAL_NET_INFERENCE_CACHE_H_

#include <array>

#include "cc/constants.h"
#include "cc/dual_net/dual_net.h"

#include "absl/container/node_hash_map.h"

namespace minigo {

// An LRU cache for inferences.
// Not thread safe.
class InferenceCache {
 public:
  using CompressedFeatures =
      std::array<uint64_t, (DualNet::kNumBoardFeatures + 63) / 64>;

  // Compresses the given BoardFeatures into a more compact representation.
  static CompressedFeatures CompressFeatures(
      const DualNet::BoardFeatures& features);

  // Calculates a reasonable approximation for how many elements can fit in
  // an InferenceCache of size_mb MB.
  static size_t CalculateCapacity(size_t size_mb);

  explicit InferenceCache(size_t capacity);

  // Adds the (features, inference output) pair to the cache.
  // If the cache is full, the least-recently-used pair is evicted.
  void Add(const CompressedFeatures& f, DualNet::Output& o);

  // Looks up the inference output for the given features.
  // If found, the features are marked as most-recently-used.
  bool TryGet(const CompressedFeatures& f, DualNet::Output* o);

 private:
  struct ListNode {
    ListNode* prev;
    ListNode* next;
  };

  struct Element : public ListNode {
    // The constructor doesn't initialize the features pointer because it will
    // point to the key inside map_, the address of which isn't known until
    // after the element is inserted.
    explicit Element(const DualNet::Output& output) : output(output) {}

    // A pointer to the compressed features, which is stored as the map key.
    // We don't want to store the compressed features here by value because
    // they're 768 bytes and we already have to store them in the map as keys.
    // We can't store a map iterator instead because iterators get invalidated
    // when the map performs a rehash operation.
    const CompressedFeatures* features;

    // The result of inference.
    DualNet::Output output;
  };

  // Removes the given element from the LRU list.
  void Unlink(Element* elem) {
    elem->next->prev = elem->prev;
    elem->prev->next = elem->next;
  }

  // Pushes the given element to the front of the LRU list.
  // The element must have been newly constructed, or previously unlinked from
  // the list.
  void PushFront(Element* elem) {
    elem->prev = &list_;
    elem->next = list_.next;
    list_.next->prev = elem;
    list_.next = elem;
  }

  // Circular intrusive list sentinel node.
  // The list head is list_.next.
  // The list tail is list_.prev.
  // The list is empty if list_.next == list_.prev.
  ListNode list_;

  // We use a node_hash_map because it guarantees that the address of both
  // the key and the value do not change during a rehash operation. This is
  // important the linked list keeps pointers to the map's keys.
  using Map = absl::node_hash_map<CompressedFeatures, Element>;
  Map map_;

  const size_t capacity_;
};

}  // namespace minigo

#endif  // CC_DUAL_NET_INFERENCE_CACHE_H_
