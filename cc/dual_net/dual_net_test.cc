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

#include "cc/dual_net/dual_net.h"

#include <array>

#include "cc/position.h"
#include "cc/test_utils.h"
#include "gtest/gtest.h"

namespace minigo {
namespace {

using StoneFeatures = DualNet::StoneFeatures;
using BoardFeatures = DualNet::BoardFeatures;

StoneFeatures GetStoneFeatures(const BoardFeatures& features, Coord c) {
  StoneFeatures result;
  for (int i = 0; i < DualNet::kNumStoneFeatures; ++i) {
    result[i] = features[c * DualNet::kNumStoneFeatures + i];
  }
  return result;
}

// Verifies InitializeFeatures for an empty board with black to play.
TEST(DualNetTest, TestInitializeFeaturesBlackToPlay) {
  TestablePosition board("", Color::kBlack);

  BoardFeatures features;
  DualNet::InitializeFeatures(board, &features);
  for (int c = 0; c < kN * kN; ++c) {
    auto f = GetStoneFeatures(features, c);
    for (int i = 0; i < DualNet::kPlayerFeature; ++i) {
      EXPECT_EQ(0, f[i]);
    }
    EXPECT_EQ(1, f[DualNet::kPlayerFeature]);
  }
}

// Verifies InitializeFeatures for an empty board with white to play.
TEST(DualNetTest, TestInitializeFeaturesWhiteToPlay) {
  TestablePosition board("", Color::kWhite);

  BoardFeatures features;
  DualNet::InitializeFeatures(board, &features);
  for (int c = 0; c < kN * kN; ++c) {
    auto f = GetStoneFeatures(features, c);
    for (int i = 0; i < DualNet::kPlayerFeature; ++i) {
      EXPECT_EQ(0, f[i]);
    }
    EXPECT_EQ(0, f[DualNet::kPlayerFeature]);
  }
}

// Verifies UpdateFeatures.
TEST(DualNetTest, TestUpdateFeatures) {
  TestablePosition board("");

  BoardFeatures features;
  DualNet::InitializeFeatures(board, &features);

  std::vector<std::string> moves = {"B9", "H9", "A8", "J9"};
  for (const auto& move : moves) {
    board.PlayMove(move);
    DualNet::UpdateFeatures(features, board, &features);
  }

  //                  B0 W0 B1 W1 B2 W2 B3 W3 B4 W4 B5 W5 B6 W6 B7 W7 C
  StoneFeatures b9 = {1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
  StoneFeatures h9 = {0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
  StoneFeatures a8 = {1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
  StoneFeatures j9 = {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};

  EXPECT_EQ(b9, GetStoneFeatures(features, Coord::FromString("B9")));
  EXPECT_EQ(h9, GetStoneFeatures(features, Coord::FromString("H9")));
  EXPECT_EQ(a8, GetStoneFeatures(features, Coord::FromString("A8")));
  EXPECT_EQ(j9, GetStoneFeatures(features, Coord::FromString("J9")));
}

// Verfies that features work as expected when capturing.
TEST(DualNetTest, TestStoneFeaturesWithCapture) {
  TestablePosition board("");

  BoardFeatures features;
  DualNet::InitializeFeatures(board, &features);

  std::vector<std::string> moves = {"J3", "pass", "H2", "J2",
                                    "J1", "pass", "J2"};
  for (const auto& move : moves) {
    board.PlayMove(move);
    DualNet::UpdateFeatures(features, board, &features);
  }

  //                  W0 B0 W1 B1 W2 B2 W3 B3 W4 B4 W5 B5 W6 B6 W7 B7 C
  StoneFeatures j2 = {0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  EXPECT_EQ(j2, GetStoneFeatures(features, Coord::FromString("J2")));
}

// Verifies that UpdateFeatures generates the correct features when the same
// object is passed for both old_features and new_features.
TEST(DualNetTest, TestUpdateFeaturesSameObject) {
  TestablePosition board("");

  BoardFeatures a, b;
  DualNet::InitializeFeatures(board, &a);
  std::vector<std::string> moves = {"A9", "B9", "A8", "D3"};
  for (const auto& move : moves) {
    board.PlayMove(move);
    DualNet::UpdateFeatures(a, board, &b);
    DualNet::UpdateFeatures(a, board, &a);
    ASSERT_EQ(a, b);
  }
}

}  // namespace
}  // namespace minigo
