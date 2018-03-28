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
#include <functional>

#include "gtest/gtest.h"

namespace minigo {
namespace {

TEST(RandomTest, TestUniformArray) {
  Random rnd(66);

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
    EXPECT_NEAR(a, 7.5, 0.05);
  }
}

TEST(RandomTest, TestOperator) {
  Random rnd(42);
  float sum = 0;
  for (int iter = 0; iter < 10000; ++iter) {
    sum += rnd();
  }
  float avg = sum / 10000;
  EXPECT_NEAR(avg, 0.5, 0.01);
}

TEST(RandomTest, Dirichlet) {
  Random rnd(777);

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

}  // namespace
}  // namespace minigo
