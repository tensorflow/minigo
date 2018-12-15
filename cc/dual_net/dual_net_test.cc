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

#include <array>
#include <deque>
#include <map>
#include <vector>

#include "cc/dual_net/dual_net.h"
#include "cc/position.h"
#include "cc/random.h"
#include "cc/symmetries.h"
#include "cc/test_utils.h"
#include "gtest/gtest.h"

#if MG_ENABLE_TF_DUAL_NET
#include "cc/dual_net/tf_dual_net.h"
#endif
#if MG_ENABLE_LITE_DUAL_NET
#include "cc/dual_net/lite_dual_net.h"
#endif
#if MG_ENABLE_TRT_DUAL_NET
#include "cc/dual_net/trt_dual_net.h"
#endif

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

// Verifies SetFeatures an empty board with black to play.
TEST(DualNetTest, TestEmptyBoardBlackToPlay) {
  Position::Stones stones;
  std::vector<const Position::Stones*> history = {&stones};
  DualNet::BoardFeatures features;
  DualNet::SetFeatures(history, Color::kBlack, &features);

  for (int c = 0; c < kN * kN; ++c) {
    auto f = GetStoneFeatures(features, c);
    for (int i = 0; i < DualNet::kPlayerFeature; ++i) {
      EXPECT_EQ(0, f[i]);
    }
    EXPECT_EQ(1, f[DualNet::kPlayerFeature]);
  }
}

// Verifies SetFeatures for an empty board with white to play.
TEST(DualNetTest, TestEmptyBoardWhiteToPlay) {
  Position::Stones stones;
  std::vector<const Position::Stones*> history = {&stones};
  DualNet::BoardFeatures features;
  DualNet::SetFeatures(history, Color::kWhite, &features);

  for (int c = 0; c < kN * kN; ++c) {
    auto f = GetStoneFeatures(features, c);
    for (int i = 0; i < DualNet::kPlayerFeature; ++i) {
      EXPECT_EQ(0, f[i]);
    }
    EXPECT_EQ(0, f[DualNet::kPlayerFeature]);
  }
}

// Verifies SetFeatures.
TEST(DualNetTest, TestSetFeatures) {
  TestablePosition board("");

  std::vector<std::string> moves = {"B9", "H9", "A8", "J9"};
  std::deque<Position::Stones> positions;
  for (const auto& move : moves) {
    board.PlayMove(move);
    positions.push_front(board.stones());
  }

  std::vector<const Position::Stones*> history;
  for (const auto& p : positions) {
    history.push_back(&p);
  }

  DualNet::BoardFeatures features;
  DualNet::SetFeatures(history, board.to_play(), &features);

  //                   B0 W0 B1 W1 B2 W2 B3 W3 B4 W4 B5 W5 B6 W6 B7 W7 C
  StoneFeatures b9 = {{1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}};
  StoneFeatures h9 = {{0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}};
  StoneFeatures a8 = {{1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}};
  StoneFeatures j9 = {{0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}};

  EXPECT_EQ(b9, GetStoneFeatures(features, Coord::FromString("B9")));
  EXPECT_EQ(h9, GetStoneFeatures(features, Coord::FromString("H9")));
  EXPECT_EQ(a8, GetStoneFeatures(features, Coord::FromString("A8")));
  EXPECT_EQ(j9, GetStoneFeatures(features, Coord::FromString("J9")));
}

// Verfies that features work as expected when capturing.
TEST(DualNetTest, TestStoneFeaturesWithCapture) {
  TestablePosition board("");

  std::vector<std::string> moves = {"J3", "pass", "H2", "J2",
                                    "J1", "pass", "J2"};
  std::deque<Position::Stones> positions;
  for (const auto& move : moves) {
    board.PlayMove(move);
    positions.push_front(board.stones());
  }

  std::vector<const Position::Stones*> history;
  for (const auto& p : positions) {
    history.push_back(&p);
  }

  BoardFeatures features;
  DualNet::SetFeatures(history, board.to_play(), &features);

  //                   W0 B0 W1 B1 W2 B2 W3 B3 W4 B4 W5 B5 W6 B6 W7 B7 C
  StoneFeatures j2 = {{0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  EXPECT_EQ(j2, GetStoneFeatures(features, Coord::FromString("J2")));
}

// Checks that the different backends produce the same result.
TEST(DualNetTest, TestBackendsEqual) {
  struct Test {
    Test(std::unique_ptr<DualNetFactory> factory, std::string extension)
        : factory(std::move(factory)), extension(std::move(extension)) {}
    std::unique_ptr<DualNetFactory> factory;
    std::string extension;
  };

  std::map<std::string, Test> tests;

#if MG_ENABLE_TF_DUAL_NET
  tests.emplace("TfDualNet",
                Test(absl::make_unique<TfDualNetFactory>(), ".pb"));
#endif
#if MG_ENABLE_LITE_DUAL_NET
  tests.emplace("LiteDualNet",
                Test(absl::make_unique<LiteDualNetFactory>(), ".tflite"));
#endif
#if MG_ENABLE_TRT_DUAL_NET
  tests.emplace("TrtDualNet",
                Test(absl::make_unique<TrtDualNetFactory>(), ".uff"));
#endif

  DualNet::BoardFeatures nhwc_features;
  Random().Uniform(0.0f, 1.0f, absl::MakeSpan(nhwc_features));
  DualNet::BoardFeatures nchw_features;
  using OutIter =
      symmetry::NchwOutputIterator<kN, DualNet::kNumStoneFeatures, float>;
  std::copy(nhwc_features.begin(), nhwc_features.end(),
            OutIter(nchw_features.data()));

  DualNet::Output ref_output;
  std::string ref_name;

  auto policy_string = [](const std::array<float, kNumMoves>& policy) {
    std::ostringstream oss;
    std::copy(policy.begin(), policy.end(),
              std::ostream_iterator<float>(oss, " "));
    return oss.str();
  };

  for (const auto& kv : tests) {
    const auto& name = kv.first;
    auto& test = kv.second;
    MG_LOG(INFO) << "Running " << name;

    auto dual_net = test.factory->NewDualNet(
        absl::StrCat("cc/dual_net/test_model", test.extension));

    auto* features = dual_net->GetInputLayout() == DualNet::InputLayout::kNHWC
                         ? &nhwc_features
                         : &nchw_features;
    DualNet::Output output;
    dual_net->RunMany({features}, {&output}, nullptr);

    if (ref_name.empty()) {
      ref_output = output;
      ref_name = name;
      continue;
    }

    auto pred = [](float left, float right) {
      return std::abs(left - right) <
             0.0001f * (1.0f + std::abs(left) + std::abs(right));
    };
    EXPECT_EQ(std::equal(output.policy.begin(), output.policy.end(),
                         ref_output.policy.begin(), pred),
              true)
        << name << ": " << policy_string(output.policy) << "\n"
        << ref_name << ": " << policy_string(ref_output.policy);
    EXPECT_NEAR(output.value, ref_output.value, 0.0001f)
        << name << " vs " << ref_name;
  }
}
}  // namespace
}  // namespace minigo
