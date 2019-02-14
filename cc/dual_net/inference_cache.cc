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

namespace minigo {

InferenceCache::CompressedFeatures InferenceCache::CompressFeatures(
    const DualNet::BoardFeatures& features) {
  CompressedFeatures result;
  auto num_full_u64s = (features.size() + 63) / 64 - 1;
  const auto* src = features.data();
  for (size_t i = 0; i < num_full_u64s; ++i) {
    uint64_t bits = 0;
    for (uint64_t j = 0; j < 64; ++j) {
      bits |= static_cast<uint64_t>(*src++ != 0) << j;
    }
    result[i] = bits;
  }
  uint64_t bits = 0;
  for (uint64_t j = 0; src != features.end(); ++j) {
    bits |= static_cast<uint64_t>(*src++ != 0) << j;
  }
  result[num_full_u64s] = bits;
  return result;
}

size_t InferenceCache::CalculateCapacity(size_t size_mb) {
  // Maximum load factory of an absl::node_hash_map.
  float max_load_factor = 0.875;

  // absl::node_hash_map allocates each (key, value) pair on the heap and stores
  // pointers to those pairs in the table itself.
  float element_size =
      sizeof(Map::value_type) + sizeof(Map::value_type*) / max_load_factor;

  return static_cast<size_t>(size_mb * 1024.0f * 1024.0f / element_size);
}

InferenceCache::InferenceCache(size_t capacity) : capacity_(capacity) {
  // Init the LRU list.
  list_.prev = &list_;
  list_.next = &list_;
}

void InferenceCache::Add(const CompressedFeatures& f, DualNet::Output& o) {
  if (map_.size() == capacity_) {
    // Cache is full, remove an element.
    auto it = map_.find(*static_cast<Element*>(list_.prev)->features);
    MG_CHECK(it != map_.end());
    Unlink(&it->second);
    map_.erase(it);
  }

  auto result = map_.emplace(f, o);
  MG_CHECK(result.second);
  auto* elem = &result.first->second;
  elem->features = &result.first->first;
  PushFront(elem);
}

bool InferenceCache::TryGet(const CompressedFeatures& f, DualNet::Output* o) {
  auto it = map_.find(f);
  if (it == map_.end()) {
    return false;
  }

  auto* elem = &it->second;
  Unlink(elem);
  PushFront(elem);
  *o = elem->output;
  return true;
}

}  // namespace minigo
