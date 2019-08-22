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

#include "cc/dual_net/inference_cache.h"

#include <tuple>

#include "absl/memory/memory.h"

namespace minigo {

std::ostream& operator<<(std::ostream& os, InferenceCache::Key key) {
  return os << absl::StreamFormat("%016x:%016x", key.cache_hash_,
                                  key.stone_hash_);
}

InferenceCache::Key InferenceCache::Key::CreateTestKey(
    zobrist::Hash cache_hash, zobrist::Hash stone_hash) {
  return InferenceCache::Key(cache_hash, stone_hash);
}

InferenceCache::Key::Key(Coord prev_move, const Position& position)
    : Key(position.stone_hash(), position.stone_hash()) {
  cache_hash_ ^= zobrist::ToPlayHash(position.to_play());

  if (prev_move == Coord::kPass) {
    cache_hash_ ^= zobrist::OpponentPassedHash();
  }

  const auto& stones = position.stones();
  for (int i = 0; i < kN * kN; ++i) {
    if (stones[i].color() == Color::kEmpty && !position.legal_move(i)) {
      cache_hash_ ^= zobrist::IllegalEmptyPointHash(i);
    }
  }
}

InferenceCache::~InferenceCache() = default;

size_t BasicInferenceCache::CalculateCapacity(size_t size_mb) {
  // Minimum load factory of an absl::node_hash_map at the time of writing,
  // taken from https://abseil.io/docs/cpp/guides/container.
  // This is a pessimistic estimate of the cache's load factor but since the
  // size of each node pointer is much smaller than that of the node itself, it
  // shouldn't make that much difference either way.
  float load_factor = 0.4375;

  // absl::node_hash_map allocates each (key, value) pair on the heap and stores
  // pointers to those pairs in the table itself, along with one byte of hash
  // for each element.
  float element_size =
      sizeof(Map::value_type) + (sizeof(Map::value_type*) + 1) / load_factor;

  return static_cast<size_t>(size_mb * 1024.0f * 1024.0f / element_size);
}

BasicInferenceCache::BasicInferenceCache(size_t capacity)
    : capacity_(capacity) {
  MG_CHECK(capacity_ > 0);
  Clear();
}

void BasicInferenceCache::Clear() {
  // Init the LRU list.
  list_.prev = &list_;
  list_.next = &list_;
  map_.clear();
}

void BasicInferenceCache::Add(Key key, const Model::Output& output) {
  if (map_.size() == capacity_) {
    // Cache is full, remove an element.
    auto it = map_.find(static_cast<Element*>(list_.prev)->key);
    MG_CHECK(it != map_.end());
    Unlink(&it->second);
    map_.erase(it);
  }

  // Insert the key into the map without making temporary copies of the Element.
  auto result =
      map_.emplace(std::piecewise_construct, std::forward_as_tuple(key),
                   std::forward_as_tuple(key, output));
  auto* elem = &result.first->second;
  if (!result.second) {
    Unlink(elem);  // The element was already in the cache.
  }
  PushFront(elem);
}

bool BasicInferenceCache::TryGet(Key key, Model::Output* output) {
  auto it = map_.find(key);
  if (it == map_.end()) {
    return false;
  }

  auto* elem = &it->second;
  Unlink(elem);
  PushFront(elem);
  *output = elem->output;
  return true;
}

ThreadSafeInferenceCache::ThreadSafeInferenceCache(size_t total_capacity,
                                                   int num_shards) {
  shards_.reserve(num_shards);
  size_t shard_capacity_sum = 0;
  for (int i = 0; i < num_shards; ++i) {
    auto a = i * total_capacity / num_shards;
    auto b = (i + 1) * total_capacity / num_shards;
    auto shard_capacity = b - a;
    shard_capacity_sum += shard_capacity;
    shards_.push_back(absl::make_unique<Shard>(shard_capacity));
  }
  MG_CHECK(shard_capacity_sum == total_capacity);
}

void ThreadSafeInferenceCache::Clear() {
  for (auto& shard : shards_) {
    absl::MutexLock lock(&shard->mutex);
    shard->cache.Clear();
  }
}

void ThreadSafeInferenceCache::Add(Key key, const Model::Output& output) {
  auto* shard = shards_[key.Shard(shards_.size())].get();
  absl::MutexLock lock(&shard->mutex);
  shard->cache.Add(key, output);
}

bool ThreadSafeInferenceCache::TryGet(Key key, Model::Output* output) {
  auto* shard = shards_[key.Shard(shards_.size())].get();
  absl::MutexLock lock(&shard->mutex);
  return shard->cache.TryGet(key, output);
}

}  // namespace minigo
