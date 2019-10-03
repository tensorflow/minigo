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

#include <algorithm>
#include <sstream>
#include <thread>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "cc/random.h"
#include "cc/symmetries.h"
#include "cc/test_utils.h"
#include "gtest/gtest.h"

namespace minigo {
namespace {

struct Inference {
  Inference(InferenceCache::Key key, const ModelOutput& output)
      : key(key), output(output) {}
  InferenceCache::Key key;
  ModelOutput output;
};

// Verify that all symmetries of a position generate the same cache key.
TEST(KeyTest, CanonicalSymmetry) {
  // Generate a position that isn't symmetric and contains some empty points
  // that black and white can't play in, so we can verify the hashing of illegal
  // empty points.
  Random rnd(234697, 1);
  TestablePosition canonical_position(R"(
X . X . X
X X X X X
O O . . .
. O . . .
O O . . .
. O . . .
O O . . .
)");
  Coord prev_move;
  for (int i = 0; i < 20; ++i) {
    prev_move = GetRandomLegalMove(canonical_position, &rnd);
    ASSERT_NE(Coord::kPass, prev_move);
    canonical_position.PlayMove(prev_move);
  }
  std::array<Color, kN * kN> canonical_stones;
  for (int i = 0; i < kN * kN; ++i) {
    canonical_stones[i] = canonical_position.stones()[i].color();
  }

  // Calculate the canonical position's cache key.
  InferenceCache::Key canonical_key(prev_move, symmetry::kIdentity,
                                    canonical_position);

  // Apply all symmetries to the canonical position, the cache key of each
  // symmetric position should be equal to the canonical position's key.
  for (auto canonical_sym : symmetry::kAllSymmetries) {
    // canonical_sym is the symmetry required to the canonical form of a
    // position into it's real one.
    std::array<Color, kN * kN> symmetric_stones;
    symmetry::ApplySymmetry<kN, 1>(canonical_sym, canonical_stones.data(),
                                   symmetric_stones.data());
    if (canonical_sym == symmetry::kIdentity) {
      ASSERT_EQ(canonical_stones, symmetric_stones);
    } else {
      ASSERT_NE(canonical_stones, symmetric_stones);
    }

    TestablePosition symmetric_position(symmetric_stones,
                                        canonical_position.to_play());
    InferenceCache::Key symmetric_key(
        symmetry::ApplySymmetry(canonical_sym, prev_move), canonical_sym,
        symmetric_position);

    EXPECT_EQ(canonical_key, symmetric_key);
  }
}

// Verify the LRU behavior of the basic cache.
TEST(BasicInferenceCacheTest, LruTest) {
  BasicInferenceCache cache(3);

  // Create some positions & inference outputs.
  Random rnd(614944751, 1);
  std::vector<Inference> inferences;
  auto prev_move = Coord::kInvalid;
  auto sym = symmetry::kIdentity;

  TestablePosition position("");
  for (int i = 0; i < 4; ++i) {
    InferenceCache::Key key(prev_move, sym, position);
    prev_move = GetRandomLegalMove(position, &rnd);
    position.PlayMove(prev_move);

    ModelOutput output;
    rnd.Uniform(&output.policy);
    output.value = rnd();
    inferences.emplace_back(key, output);
  }

  // Fill the cache.
  for (int i = 0; i < 3; ++i) {
    cache.Merge(inferences[i].key, sym, sym, &inferences[i].output);
  }

  // Verify that the elements stored in the cache are as expected.
  ModelOutput output;
  for (int i = 0; i < 3; ++i) {
    ASSERT_TRUE(cache.TryGet(inferences[i].key, sym, sym, &output));
    EXPECT_EQ(inferences[i].output.policy, output.policy);
    EXPECT_EQ(inferences[i].output.value, output.value);
  }

  // Adding a fourth element should evict the least recently used one.
  cache.Merge(inferences[3].key, sym, sym, &inferences[3].output);
  ASSERT_TRUE(cache.TryGet(inferences[3].key, sym, sym, &output));
  EXPECT_EQ(inferences[3].output.policy, output.policy);
  EXPECT_EQ(inferences[3].output.value, output.value);

  EXPECT_FALSE(cache.TryGet(inferences[0].key, sym, sym, &output));
}

// A basic test of putting a single symmetry of a position into the cache.
TEST(InferenceCacheTest, SingleSymmetryTest) {
  Random rnd(80379245, 1);

  // KeyTest.CanonicalSymmetry verifies that all symmetries of a position
  // produce the same key, so we don't have to worry about the actual input
  // position in this test.
  auto key = InferenceCache::Key::CreateTestKey(rnd.UniformUint64(),
                                                rnd.UniformUint64());

  // Generate the canonical output.
  ModelOutput canonical_output;
  for (int i = 0; i < kNumMoves; ++i) {
    canonical_output.policy[i] = static_cast<float>(i);
  }
  canonical_output.value = 0.25;

  auto canonical_symmetries = symmetry::kAllSymmetries;
  rnd.Shuffle(&canonical_symmetries);
  for (auto canonical_sym : canonical_symmetries) {
    // When running for real, the output from inference would be transformed by
    // the canonical symmetry.
    ModelOutput real_output;
    Model::ApplySymmetry(canonical_sym, canonical_output, &real_output);

    // The models handle inference symmetry internally so we don't actually
    // apply inference_sym anywhere in the test code, but the inference cache
    // will be transforming model outputs by the inference symmetry internally.
    auto inference_symmetries = symmetry::kAllSymmetries;
    rnd.Shuffle(&inference_symmetries);
    for (auto inference_sym : inference_symmetries) {
      BasicInferenceCache cache(3);

      // The cache should be empty.
      ModelOutput cached_output;
      EXPECT_FALSE(
          cache.TryGet(key, canonical_sym, inference_sym, &cached_output));

      // Merging the first symmetry for a position should not change the output.
      ModelOutput before_merge_output = real_output;
      cache.Merge(key, canonical_sym, inference_sym, &real_output);

      EXPECT_EQ(before_merge_output.policy, real_output.policy);
      EXPECT_EQ(before_merge_output.value, real_output.value);

      // Make sure the cached output matches what we put in.
      EXPECT_TRUE(
          cache.TryGet(key, canonical_sym, inference_sym, &cached_output));
      EXPECT_EQ(real_output.policy, cached_output.policy);
      EXPECT_EQ(real_output.value, cached_output.value);
    }
  }
}

// Test that different symmetries of a position get averaged together when
// merged.
TEST(InferenceCacheTest, MergeSymmetiesTest) {
  Random rnd(89072659, 1);

  // KeyTest.CanonicalSymmetry verifies that all symmetries of a position
  // produce the same key, so we don't have to worry about the actual input
  // position in this test.
  auto key = InferenceCache::Key::CreateTestKey(rnd.UniformUint64(),
                                                rnd.UniformUint64());

  // Generate canonical output.
  // The policy is initialized to 0.0 except for one point that doesn't lie on
  // any symmetry.
  ModelOutput canonical_output;
  for (auto& x : canonical_output.policy) {
    x = 0;
  }
  Coord canonical_non_zero_point(2, 1);
  canonical_output.policy[canonical_non_zero_point] = 1;
  canonical_output.value = 0;

  auto canonical_symmetries = symmetry::kAllSymmetries;
  rnd.Shuffle(&canonical_symmetries);
  for (auto canonical_sym : canonical_symmetries) {
    BasicInferenceCache cache(3);

    // Build the output from this inference.
    ModelOutput real_output;
    Model::ApplySymmetry(canonical_sym, canonical_output, &real_output);
    auto real_non_zero_point =
        symmetry::ApplySymmetry(canonical_sym, canonical_non_zero_point);

    absl::flat_hash_set<Coord> expected_non_zero_points;
    auto inference_symmetries = symmetry::kAllSymmetries;
    rnd.Shuffle(&inference_symmetries);
    for (int symmetry_merged = 0; symmetry_merged < symmetry::kNumSymmetries;
         ++symmetry_merged) {
      auto inference_sym = inference_symmetries[symmetry_merged];

      ModelOutput inference_output;
      Model::ApplySymmetry(inference_sym, real_output, &inference_output);

      // In order to reasonably test that the kPass policy gets averaged
      // correctly, each symmetry is assigned a kPass policy of N, which starts
      // at 0 and increments each iteration.
      // On the i'th iteration, the average of all these values is:
      //   (N_i + N_i-1) / (i + 1)
      // This works out to: i / 2
      inference_output.policy[Coord::kPass] = symmetry_merged;

      // We do a similar thing for the output's value.
      inference_output.value = 3.0f * symmetry_merged;

      // We haven't put this symmetry into the cache yet.
      ModelOutput cached_output;
      EXPECT_FALSE(
          cache.TryGet(key, canonical_sym, inference_sym, &cached_output));

      expected_non_zero_points.insert(
          symmetry::ApplySymmetry(inference_sym, real_non_zero_point));

      cache.Merge(key, canonical_sym, inference_sym, &inference_output);

      for (int i = 0; i < kN * kN; ++i) {
        if (expected_non_zero_points.contains(i)) {
          EXPECT_NEAR(1.0f / (symmetry_merged + 1), inference_output.policy[i],
                      0.0001);
        } else {
          EXPECT_EQ(0, inference_output.policy[i]);
        }
      }
      EXPECT_NEAR(symmetry_merged / 2.0f, inference_output.policy[Coord::kPass],
                  0.0001);

      EXPECT_NEAR(3.0f * symmetry_merged / 2.0f, inference_output.value,
                  0.0001);

      EXPECT_TRUE(
          cache.TryGet(key, canonical_sym, inference_sym, &cached_output));
      EXPECT_EQ(inference_output.policy, cached_output.policy);
      EXPECT_EQ(inference_output.value, cached_output.value);
    }
  }
}

TEST(ThreadSafeInferenceCacheTest, SimpleTest) {
  ThreadSafeInferenceCache cache(4, 2);

  // Create some positions & inference outputs.
  Random rnd(35374, 1);
  std::vector<Inference> inferences;
  auto prev_move = Coord::kInvalid;
  auto sym = symmetry::kIdentity;

  TestablePosition position("");
  for (int i = 0; i < 4; ++i) {
    auto key = InferenceCache::Key::CreateTestKey(i, i);
    prev_move = GetRandomLegalMove(position, &rnd);
    position.PlayMove(prev_move);

    ModelOutput output;
    rnd.Uniform(&output.policy);
    output.value = rnd();
    inferences.emplace_back(key, output);
  }

  // Fill the cache.
  for (auto& inference : inferences) {
    cache.Merge(inference.key, sym, sym, &inference.output);
  }

  // Verify that the elements stored in the cache are as expected.
  ModelOutput output;
  for (const auto& inference : inferences) {
    ASSERT_TRUE(cache.TryGet(inference.key, sym, sym, &output));
    EXPECT_EQ(inference.output.policy, output.policy);
    EXPECT_EQ(inference.output.value, output.value);
  }
}

TEST(ThreadSafeInferenceCacheTest, StressTest) {
  constexpr int kCacheSize = 32;
  constexpr int kNumThreads = 10;
  constexpr int kNumShards = 3;
  constexpr int kNumIterations = 10000;
  auto sym = symmetry::kIdentity;

  ThreadSafeInferenceCache cache(kCacheSize, kNumShards);
  std::vector<std::thread> threads;
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&cache, i, sym]() {
      int hits = 0;
      int misses = 0;
      ModelOutput output;
      Random rnd(27, i);
      for (int i = 0; i < kNumIterations; ++i) {
        TestablePosition position("");
        // Create cache keys that only differ by a few bits so that the test
        // gets a roughly 50/50 split of cache hits and misses.
        auto key = InferenceCache::Key::CreateTestKey(rnd.UniformInt(0, 7),
                                                      rnd.UniformInt(0, 8));
        if (cache.TryGet(key, sym, sym, &output)) {
          hits += 1;
        } else {
          misses += 1;
        }
        cache.Merge(key, sym, sym, &output);
      }
      MG_LOG(INFO) << "thread:" << i << " hits:" << hits
                   << " misses:" << misses;
    });
  }
  for (auto& t : threads) {
    t.join();
  }
}

}  // namespace
}  // namespace minigo

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::minigo::zobrist::Init(375089798);
  return RUN_ALL_TESTS();
}
