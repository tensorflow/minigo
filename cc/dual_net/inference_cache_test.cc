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
#include "gtest/gtest.h"

namespace minigo {
namespace {

// Note that the features returned aren't valid input features for the model:
// there may be a 1 in both the black & white feature planes.
DualNet::BoardFeatures RandomFeatures(Random* rnd) {
  DualNet::BoardFeatures features;
  for (auto& f : features) {
    f = static_cast<float>(rnd->UniformInt(0, 1));
  }
  return features;
}

struct Inference {
  Inference(const InferenceCache::CompressedFeatures& features,
            const DualNet::Output& output)
      : features(features), output(output) {}
  InferenceCache::CompressedFeatures features;
  DualNet::Output output;
};

// Verify that compressing features works correctly.
TEST(InferenceCacheTest, CompressFeatures) {
  Random rnd(614944751);
  for (int iteration = 0; iteration < 10; ++iteration) {
    auto original = RandomFeatures(&rnd);
    auto compressed = InferenceCache::CompressFeatures(original);
    for (size_t i = 0; i < original.size(); ++i) {
      auto expected = original[i];
      auto actual = static_cast<float>((compressed[i / 64] >> i) & 1);
      ASSERT_EQ(expected, actual) << "index: " << i;
    }
  }
}

// Verify the LRU behavior of the cache.
TEST(InferenceCacheTest, LruCache) {
  InferenceCache cache(3);

  // Create some random compressed features & inference outputs.
  Random rnd(614944751);
  std::vector<Inference> inferences;
  for (int i = 0; i < 4; ++i) {
    auto features = InferenceCache::CompressFeatures(RandomFeatures(&rnd));
    DualNet::Output output;
    rnd.Uniform(&output.policy);
    output.value = rnd();
    inferences.emplace_back(features, output);
  }

  // Fill the cache.
  for (int i = 0; i < 3; ++i) {
    cache.Add(inferences[0].features, inferences[0].output);
    cache.Add(inferences[1].features, inferences[1].output);
    cache.Add(inferences[2].features, inferences[2].output);
  }

  // Verify that the elements stored in the cache are as expected.
  DualNet::Output output;
  for (int i = 0; i < 3; ++i) {
    ASSERT_TRUE(cache.TryGet(inferences[i].features, &output));
    EXPECT_EQ(inferences[i].output.policy, output.policy);
    EXPECT_EQ(inferences[i].output.value, output.value);
  }

  // Adding a fourth element should evict the least recently used one.
  cache.Add(inferences[3].features, inferences[3].output);
  ASSERT_TRUE(cache.TryGet(inferences[3].features, &output));
  EXPECT_EQ(inferences[3].output.policy, output.policy);
  EXPECT_EQ(inferences[3].output.value, output.value);

  EXPECT_FALSE(cache.TryGet(inferences[0].features, &output));
}

}  // namespace
}  // namespace minigo
