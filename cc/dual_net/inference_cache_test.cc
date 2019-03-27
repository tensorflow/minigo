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

#include <vector>

#include "cc/random.h"
#include "cc/test_utils.h"
#include "gtest/gtest.h"

namespace minigo {
namespace {

// Note that the features returned aren't valid input features for the model:
struct Inference {
  Inference(InferenceCache::Key key, const DualNet::Output& output)
      : key(key), output(output) {}
  InferenceCache::Key key;
  DualNet::Output output;
};

// Verify the LRU behavior of the cache.
TEST(InferenceCacheTest, LruCache) {
  InferenceCache cache(3);

  // Create some positions & inference outputs.
  Random rnd(614944751);
  std::vector<Inference> inferences;
  auto prev_move = Coord::kInvalid;
  TestablePosition position("");
  for (int i = 0; i < 4; ++i) {
    InferenceCache::Key key(prev_move, position);
    prev_move = GetRandomLegalMove(position, &rnd);
    position.PlayMove(prev_move);

    DualNet::Output output;
    rnd.Uniform(&output.policy);
    output.value = rnd();
    inferences.emplace_back(key, output);
  }

  // Fill the cache.
  for (int i = 0; i < 3; ++i) {
    cache.Add(inferences[i].key, inferences[i].output);
  }

  // Verify that the elements stored in the cache are as expected.
  DualNet::Output output;
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

}  // namespace
}  // namespace minigo

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::minigo::zobrist::Init(614944751);
  return RUN_ALL_TESTS();
}
