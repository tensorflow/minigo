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

#ifndef CC_MODEL_INFERENCE_CACHE_H_
#define CC_MODEL_INFERENCE_CACHE_H_

#include <array>
#include <memory>
#include <ostream>
#include <vector>

#include "absl/container/node_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "cc/constants.h"
#include "cc/coord.h"
#include "cc/model/model.h"
#include "cc/position.h"
#include "cc/zobrist.h"

namespace minigo {

// An LRU cache for inferences.
class InferenceCache {
 public:
  // The key used for the inference cache.
  // Takes into account:
  //  - the stones on the board.
  //  - who is to play.
  //  - which moves are legal.
  //  - whether the previous move was a pass.
  class Key {
   public:
    // Constructs a test key directly.
    // Provided to make testing possible.
    static Key CreateTestKey(zobrist::Hash cache_hash,
                             zobrist::Hash stone_hash);

    // Constructs a cache key from the given position and previous move made to
    // get to that position.
    Key(Coord prev_move, const Position& position);

    template <typename H>
    friend H AbslHashValue(H h, Key key) {
      return H::combine(std::move(h), key.cache_hash_);
    }

    friend bool operator==(Key a, Key b) {
      return a.cache_hash_ == b.cache_hash_ && a.stone_hash_ == b.stone_hash_;
    }

    int Shard(int num_shards) const { return cache_hash_ % num_shards; }

    friend std::ostream& operator<<(std::ostream& os, Key key);

   private:
    Key(zobrist::Hash cache_hash, zobrist::Hash stone_hash)
        : cache_hash_(cache_hash), stone_hash_(stone_hash) {}

    // There is a vanishingly small chance that two Positions could have
    // different stone hashes but the same computed hash for the inference
    // cache key. To avoid potential crashes in this case, the key compares both
    // for equality. Note that it's sufficient to use cache_hash_ for the Key's
    // actual hash value.
    zobrist::Hash cache_hash_;
    zobrist::Hash stone_hash_;
  };

  virtual ~InferenceCache();

  // Clears the cache.
  virtual void Clear() = 0;

  // Adds the (features, inference output) pair to the cache.
  // If the cache is full, the least-recently-used pair is evicted.
  virtual void Add(Key key, const Model::Output& output) = 0;

  // Looks up the inference output for the given features.
  // If found, the features are marked as most-recently-used.
  virtual bool TryGet(Key key, Model::Output* output) = 0;
};

// Not thread safe.
class BasicInferenceCache : public InferenceCache {
 public:
  // Calculates a reasonable approximation for how many elements can fit in
  // an InferenceCache of size_mb MB.
  static size_t CalculateCapacity(size_t size_mb);

  explicit BasicInferenceCache(size_t capacity);

  void Clear() override;
  void Add(Key key, const Model::Output& output) override;
  bool TryGet(Key key, Model::Output* output) override;

 private:
  struct ListNode {
    ListNode* prev;
    ListNode* next;
  };

  struct Element : public ListNode {
    Element(Key key, const Model::Output& output) : key(key), output(output) {}
    Key key;
    Model::Output output;
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
  using Map = absl::node_hash_map<Key, Element>;
  Map map_;

  const size_t capacity_;
};

// Thread safe wrapper around BasicInferenceCache.
// In order to reduce lock contention when playing large numbers of games in
// parallel, ThreadSafeInferenceCache can use multiple BasicInferenceCaches,
// each guarded by their own mutex lock. The cache an element is assigned to is
// determined by InferenceCache::Key::Shard.
class ThreadSafeInferenceCache : public InferenceCache {
 public:
  static size_t CalculateCapacity(size_t size_mb) {
    // Ignore the size taken up by shards_ for now.
    return BasicInferenceCache::CalculateCapacity(size_mb);
  }

  // `total_capacity` is the total number of elements the cache can hold.
  // `num_shards` is the number BasicInferenceCaches to shard between.
  ThreadSafeInferenceCache(size_t total_capacity, int num_shards);

  // Note that each shard is locked and cleared in turn: if a Clear call is
  // made concurrently with multiple Add calls, there may never be a point in
  // time where the cache is completely empty (unless num_shards == 1).
  void Clear() override;

  void Add(Key key, const Model::Output& output) override;

  bool TryGet(Key key, Model::Output* output) override;

 private:
  struct Shard {
    explicit Shard(size_t capacity) : cache(capacity) {}
    absl::Mutex mutex;
    BasicInferenceCache cache;
  };

  std::vector<std::unique_ptr<Shard>> shards_;
};

}  // namespace minigo

#endif  // CC_MODEL_INFERENCE_CACHE_H_
