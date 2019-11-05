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

#include "cc/algorithm.h"

#include <array>

#include "cc/random.h"
#include "gtest/gtest.h"

namespace minigo {
namespace {

TEST(AlgorithmTest, ArgMaxSseRandom) {
  Random rnd(Random::kUniqueSeed, Random::kUniqueStream);

  std::array<float, 1237> vals;
  for (int iter = 0; iter < 100; ++iter) {
    rnd.Uniform(&vals);

    auto sse_result = ArgMaxSse(vals);
    auto c_result = ArgMax(vals);
    ASSERT_EQ(sse_result, c_result);
  }
}

TEST(AlgorithmTest, ArgMaxSseTieBreak) {
  std::array<float, 15> vals{};
  vals[3] = 1;
  vals[7] = 1;
  ASSERT_EQ(3, ArgMaxSse(vals));

  vals[14] = 1;
  ASSERT_EQ(3, ArgMaxSse(vals));

  vals[14] = 2;
  ASSERT_EQ(14, ArgMaxSse(vals));
  vals[13] = 2;
  ASSERT_EQ(13, ArgMaxSse(vals));
  vals[12] = 2;
  ASSERT_EQ(12, ArgMaxSse(vals));
  vals[11] = 2;
  ASSERT_EQ(11, ArgMaxSse(vals));
}

TEST(AlgorithmTest, ArgMax4) {
  std::array<float, 4> vals{1, 1, 1, 1};
  ASSERT_EQ(0, ArgMaxSse(vals));

  vals[1] = 3;
  ASSERT_EQ(1, ArgMaxSse(vals));
}

TEST(AlgorithmTest, ArgMax3) {
  std::array<float, 3> vals{1, 1, 1};
  ASSERT_EQ(0, ArgMaxSse(vals));

  vals[0] = 3;
  ASSERT_EQ(0, ArgMaxSse(vals));
  vals[1] = 4;
  ASSERT_EQ(1, ArgMaxSse(vals));
  vals[2] = 5;
  ASSERT_EQ(2, ArgMaxSse(vals));
}



}  // namespace
}  // namespace minigo
