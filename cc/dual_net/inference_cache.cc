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

namespace minigo {

std::ostream& operator<<(std::ostream& os, InferenceCache::Key key) {
  return os << absl::StreamFormat("%016x:%016x", key.cache_hash_,
                                  key.stone_hash_);
}

InferenceCache::Key::Key(Coord prev_move, const Position& position)
    : cache_hash_(position.stone_hash()), stone_hash_(position.stone_hash()) {
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

size_t InferenceCache::CalculateCapacity(size_t size_mb) {
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

InferenceCache::InferenceCache(size_t capacity) : capacity_(capacity) {
  // Init the LRU list.
  list_.prev = &list_;
  list_.next = &list_;
}

void InferenceCache::Add(Key key, const DualNet::Output& output) {
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
  MG_CHECK(result.second);
  auto* elem = &result.first->second;
  PushFront(elem);
}

bool InferenceCache::TryGet(Key key, DualNet::Output* output) {
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

}  // namespace minigo
