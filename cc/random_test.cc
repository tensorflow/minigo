// Copyright 2018 Google LLC
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

#include "cc/random.h"

#include <algorithm>
#include <array>
#include <functional>

#include "gtest/gtest.h"

namespace minigo {
namespace {

TEST(RandomTest, TestUniformArray) {
  Random rnd(66, 45243);

  std::array<float, 100> sum;
  for (auto& s : sum) {
    s = 0;
  }

  for (int iter = 0; iter < 10000; ++iter) {
    std::array<float, 100> samples;
    rnd.Uniform(5, 10, &samples);
    for (int i = 0; i < 100; ++i) {
      sum[i] += samples[i];
    }
  }

  std::array<float, 100> avg;
  for (int i = 0; i < 100; ++i) {
    avg[i] = sum[i] / 10000;
  }
  for (const auto& a : avg) {
    EXPECT_NEAR(7.5, a, 0.05);
  }
}

TEST(RandomTest, TestOperator) {
  Random rnd(42, 897692);
  float sum = 0;
  for (int iter = 0; iter < 10000; ++iter) {
    sum += rnd();
  }
  float avg = sum / 10000;
  EXPECT_NEAR(0.5, avg, 0.01);
}

TEST(RandomTest, Dirichlet) {
  Random rnd(777, 8724784);

  std::array<float, 40> sum;
  for (auto& s : sum) {
    s = 0;
  }

  for (int iter = 0; iter < 10000; ++iter) {
    std::array<float, 40> samples;
    rnd.Dirichlet(0.03, &samples);
    std::sort(samples.begin(), samples.end(), std::greater<float>());
    for (int i = 0; i < 40; ++i) {
      sum[i] += samples[i];
    }
  }

  std::array<float, 40> avg;
  for (int i = 0; i < 40; ++i) {
    avg[i] = sum[i] / 10000;
  }

  EXPECT_NEAR(0.60, avg[0], 0.01);
  EXPECT_NEAR(0.21, avg[1], 0.01);
  EXPECT_NEAR(0.09, avg[2], 0.01);
  EXPECT_NEAR(0.05, avg[3], 0.01);
  EXPECT_NEAR(0.02, avg[4], 0.01);
  EXPECT_NEAR(0.01, avg[5], 0.01);
  for (int i = 6; i < 40; ++i) {
    EXPECT_NEAR(0, avg[i], 0.01);
  }
}

TEST(RandomTest, SampleCdf) {
  Random rnd(893745, 73462594);
  std::array<float, 8> cdf;
  for (int i = 0; i < 8; ++i) {
    cdf[i] = i < 3 ? 0 : 10;
  }
  for (int iter = 0; iter < 10000; ++iter) {
    EXPECT_EQ(3, rnd.SampleCdf(absl::MakeSpan(cdf)));
  }
}

TEST(RandomTest, Streams) {
  constexpr uint64_t seed = 9872659;
  Random a(seed, 1);
  Random b(seed, 2);
  for (int iter = 0; iter < 10000; ++iter) {
    EXPECT_NE(a.UniformUint64(), b.UniformUint64());
  }
}

TEST(RandomTest, Shuffle) {
  Random rnd(0, 0);

  std::vector<int> original;
  for (int i = 0; i < 1000; ++i) {
    original.push_back(i);
  }

  auto shuffled = original;
  rnd.Shuffle(&shuffled);
  EXPECT_NE(original, shuffled);

  auto shuffled_again = original;
  rnd.Shuffle(&shuffled_again);
  EXPECT_NE(shuffled, shuffled_again);
}

}  // namespace
}  // namespace minigo
