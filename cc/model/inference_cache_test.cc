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

#include <thread>
#include <vector>

#include "cc/random.h"
#include "cc/test_utils.h"
#include "gtest/gtest.h"

namespace minigo {
namespace {

// Note that the features returned aren't valid input features for the model:
struct Inference {
  Inference(InferenceCache::Key key, const Model::Output& output)
      : key(key), output(output) {}
  InferenceCache::Key key;
  Model::Output output;
};

// Verify the LRU behavior of the basic cache.
TEST(InferenceCacheTest, Basic) {
  BasicInferenceCache cache(3);

  // Create some positions & inference outputs.
  Random rnd(614944751);
  std::vector<Inference> inferences;
  auto prev_move = Coord::kInvalid;
  TestablePosition position("");
  for (int i = 0; i < 4; ++i) {
    InferenceCache::Key key(prev_move, position);
    prev_move = GetRandomLegalMove(position, &rnd);
    position.PlayMove(prev_move);

    Model::Output output;
    rnd.Uniform(&output.policy);
    output.value = rnd();
    inferences.emplace_back(key, output);
  }

  // Fill the cache.
  for (int i = 0; i < 3; ++i) {
    cache.Add(inferences[i].key, inferences[i].output);
  }

  // Verify that the elements stored in the cache are as expected.
  Model::Output output;
  for (int i = 0; i < 3; ++i) {
    ASSERT_TRUE(cache.TryGet(inferences[i].key, &output));
    EXPECT_EQ(inferences[i].output.policy, output.policy);
    EXPECT_EQ(inferences[i].output.value, output.value);
  }

  // Adding a fourth element should evict the least recently used one.
  cache.Add(inferences[3].key, inferences[3].output);
  ASSERT_TRUE(cache.TryGet(inferences[3].key, &output));
  EXPECT_EQ(inferences[3].output.policy, output.policy);
  EXPECT_EQ(inferences[3].output.value, output.value);

  EXPECT_FALSE(cache.TryGet(inferences[0].key, &output));
}

TEST(InferenceCacheTest, ThreadSafe) {
  ThreadSafeInferenceCache cache(4, 2);

  // Create some positions & inference outputs.
  Random rnd(614944751);
  std::vector<Inference> inferences;
  auto prev_move = Coord::kInvalid;
  TestablePosition position("");
  for (int i = 0; i < 4; ++i) {
    auto key = InferenceCache::Key::CreateTestKey(i, i);
    prev_move = GetRandomLegalMove(position, &rnd);
    position.PlayMove(prev_move);

    Model::Output output;
    rnd.Uniform(&output.policy);
    output.value = rnd();
    inferences.emplace_back(key, output);
  }

  // Fill the cache.
  for (const auto& inference : inferences) {
    cache.Add(inference.key, inference.output);
  }

  // Verify that the elements stored in the cache are as expected.
  Model::Output output;
  for (const auto& inference : inferences) {
    ASSERT_TRUE(cache.TryGet(inference.key, &output));
    EXPECT_EQ(inference.output.policy, output.policy);
    EXPECT_EQ(inference.output.value, output.value);
  }
}

TEST(InferenceCacheTest, StressTest) {
  constexpr int kCacheSize = 32;
  constexpr int kNumThreads = 10;
  constexpr int kNumShards = 3;
  constexpr int kNumIterations = 100000;

  ThreadSafeInferenceCache cache(kCacheSize, kNumShards);
  std::vector<std::thread> threads;
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&cache, i]() {
      int hits = 0;
      int misses = 0;
      Model::Output output;
      Random rnd((i + 31) * 27);
      for (int i = 0; i < kNumIterations; ++i) {
        TestablePosition position("");
        // Create cache keys that only differ by a few bits so that the test
        // gets a roughly 50/50 split of cache hits and misses.
        auto key = InferenceCache::Key::CreateTestKey(rnd.UniformInt(0, 7),
                                                      rnd.UniformInt(0, 8));
        if (cache.TryGet(key, &output)) {
          hits += 1;
        } else {
          misses += 1;
        }
        cache.Add(key, output);
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
  ::minigo::zobrist::Init(614944751);
  return RUN_ALL_TESTS();
}
