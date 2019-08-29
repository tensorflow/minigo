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

#include "cc/model/inference_cache.h"

#include <tuple>

#include "absl/memory/memory.h"

namespace minigo {

std::ostream& operator<<(std::ostream& os, InferenceCache::Key key) {
  return os << absl::StreamFormat("%016x:%016x", key.cache_hash_,
                                  key.stone_hash_);
}

InferenceCache::Key InferenceCache::Key::CreateTestKey(
    zobrist::Hash cache_hash, zobrist::Hash stone_hash) {
  InferenceCache::Key key;
  key.cache_hash_ = cache_hash;
  key.stone_hash_ = stone_hash;
  return key;
}

InferenceCache::Key::Key(Coord prev_move, symmetry::Symmetry canonical_sym,
                         const Position& position) {
  cache_hash_ ^= zobrist::ToPlayHash(position.to_play());
  if (prev_move == Coord::kPass) {
    cache_hash_ ^= zobrist::OpponentPassedHash();
  }

  const auto& coord_symmetry = symmetry::kCoords[canonical_sym];
  const auto& stones = position.stones();
  for (int real_c = 0; real_c < kN * kN; ++real_c) {
    auto symmetric_c = coord_symmetry[real_c];
    auto h = zobrist::MoveHash(symmetric_c, stones[real_c].color());
    stone_hash_ ^= h;
    cache_hash_ ^= h;
    if (stones[real_c].color() == Color::kEmpty &&
        !position.legal_move(real_c)) {
      cache_hash_ ^= zobrist::IllegalEmptyPointHash(symmetric_c);
    }
  }
}

InferenceCache::~InferenceCache() = default;

std::ostream& operator<<(std::ostream& os, const InferenceCache::Stats& stats) {
  auto num_lookups =
      stats.num_hits + stats.num_complete_misses + stats.num_symmetry_misses;
  auto hit_rate =
      static_cast<float>(stats.num_hits) / static_cast<float>(num_lookups);
  auto full =
      static_cast<float>(stats.size) / static_cast<float>(stats.capacity);

  return os << "size:" << stats.size << " capacity:" << stats.capacity
            << " full:" << (100 * full) << "%"
            << " hits:" << stats.num_hits
            << " complete_misses:" << stats.num_complete_misses
            << " symmetry_misses:" << stats.num_symmetry_misses
            << " hit_rate:" << (100 * hit_rate) << "%";
}

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

BasicInferenceCache::BasicInferenceCache(size_t capacity) {
  MG_CHECK(capacity > 0);
  stats_.capacity = capacity;
  Clear();
}

void BasicInferenceCache::Clear() {
  // Init the LRU list.
  list_.prev = &list_;
  list_.next = &list_;
  map_.clear();
}

void BasicInferenceCache::Merge(Key key, symmetry::Symmetry canonical_sym,
                                symmetry::Symmetry inference_sym,
                                Model::Output* output) {
  if (map_.size() == stats_.capacity) {
    // Cache is full, remove the last element from the LRU queue.
    auto it = map_.find(static_cast<Element*>(list_.prev)->key);
    MG_CHECK(it != map_.end());
    Unlink(&it->second);
    map_.erase(it);
  } else {
    stats_.size += 1;
  }

  // Symmetry that converts the model output into canonical form.
  auto inverse_canonical_sym = symmetry::Inverse(canonical_sym);

  auto canonical_inference_sym =
      symmetry::Concat(inference_sym, inverse_canonical_sym);
  int sym_bit = (1 << canonical_inference_sym);

  auto result = map_.try_emplace(key, key, canonical_inference_sym);
  auto inserted = result.second;
  auto* elem = &result.first->second;

  if (inserted) {
    // Transform the model output into canonical form.
    Model::ApplySymmetry(inverse_canonical_sym, *output, &elem->output);
    elem->valid_symmetry_bits = sym_bit;
    elem->num_valid_symmetries = 1;
  } else {
    // The element was already in the cache.
    Unlink(elem);

    if ((elem->valid_symmetry_bits & sym_bit) == 0) {
      const auto& coord_symmetry = symmetry::kCoords[inverse_canonical_sym];

      // This is a new symmetry for this key: merge it in.
      float n = static_cast<float>(elem->num_valid_symmetries);
      float a = n / (n + 1);
      float b = 1 / (n + 1);

      auto& cached = elem->output;
      for (size_t i = 0; i < kNumMoves; ++i) {
        cached.policy[i] =
            a * cached.policy[i] + b * output->policy[coord_symmetry[i]];
      }
      cached.value = a * cached.value + b * output->value;

      elem->valid_symmetry_bits |= sym_bit;
      elem->num_valid_symmetries += 1;
    }

    Model::ApplySymmetry(canonical_sym, elem->output, output);
  }
  PushFront(elem);
}

bool BasicInferenceCache::TryGet(Key key, symmetry::Symmetry canonical_sym,
                                 symmetry::Symmetry inference_sym,
                                 Model::Output* output) {
  auto it = map_.find(key);
  if (it == map_.end()) {
    stats_.num_complete_misses += 1;
    return false;
  }

  auto* elem = &it->second;
  Unlink(elem);
  PushFront(elem);

  // Symmetry that converts the model output into canonical form.
  auto inverse_canonical_sym = symmetry::Inverse(canonical_sym);

  auto canonical_inference_sym =
      symmetry::Concat(inference_sym, inverse_canonical_sym);
  int sym_bit = (1 << canonical_inference_sym);

  if ((elem->valid_symmetry_bits & sym_bit) == 0) {
    // We have some symmetries for this position, just not the one requested.
    stats_.num_symmetry_misses += 1;
    return false;
  }

  Model::ApplySymmetry(canonical_sym, elem->output, output);
  stats_.num_hits += 1;
  return true;
}

BasicInferenceCache::Stats BasicInferenceCache::GetStats() const {
  return stats_;
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

void ThreadSafeInferenceCache::Merge(Key key, symmetry::Symmetry canonical_sym,
                                     symmetry::Symmetry inference_sym,
                                     Model::Output* output) {
  auto* shard = shards_[key.Shard(shards_.size())].get();
  absl::MutexLock lock(&shard->mutex);
  shard->cache.Merge(key, canonical_sym, inference_sym, output);
}

bool ThreadSafeInferenceCache::TryGet(Key key, symmetry::Symmetry canonical_sym,
                                      symmetry::Symmetry inference_sym,
                                      Model::Output* output) {
  auto* shard = shards_[key.Shard(shards_.size())].get();
  absl::MutexLock lock(&shard->mutex);
  return shard->cache.TryGet(key, canonical_sym, inference_sym, output);
}

InferenceCache::Stats ThreadSafeInferenceCache::GetStats() const {
  Stats result;
  for (auto& shard : shards_) {
    absl::MutexLock lock(&shard->mutex);
    auto s = shard->cache.GetStats();
    result.size += s.size;
    result.capacity += s.capacity;
    result.num_hits += s.num_hits;
    result.num_complete_misses += s.num_complete_misses;
    result.num_symmetry_misses += s.num_symmetry_misses;
  }
  return result;
}

}  // namespace minigo
