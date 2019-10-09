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
#include <type_traits>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "cc/model/features.h"
#include "cc/position.h"
#include "cc/random.h"
#include "cc/symmetries.h"
#include "cc/test_utils.h"
#include "gtest/gtest.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"

#if MG_ENABLE_TF_DUAL_NET
#include "cc/dual_net/tf_dual_net.h"
#endif
#if MG_ENABLE_LITE_DUAL_NET
#include "cc/dual_net/lite_dual_net.h"
#endif

namespace minigo {
namespace {

template <typename T>
std::vector<T> GetStoneFeatures(const Tensor<T>& features, Coord c) {
  std::vector<T> result;
  MG_CHECK(features.n == 1);
  for (int i = 0; i < features.c; ++i) {
    result.push_back(features.data[c * features.c + i]);
  }
  return result;
}

template <typename F>
class DualNetTest : public ::testing::Test {};

using TestFeatureTypes = ::testing::Types<AgzFeatures, ExtraFeatures>;
TYPED_TEST_CASE(DualNetTest, TestFeatureTypes);

// Verifies SetFeatures an empty board with black to play.
TYPED_TEST(DualNetTest, TestEmptyBoardBlackToPlay) {
  using FeatureType = TypeParam;

  TestablePosition board("");
  ModelInput input;
  input.sym = symmetry::kIdentity;
  input.position_history.push_back(&board);

  BoardFeatureBuffer<float> buffer;
  Tensor<float> features = {1, kN, kN, FeatureType::kNumPlanes, buffer.data()};
  FeatureType::Set({&input}, &features);

  for (int c = 0; c < kN * kN; ++c) {
    auto f = GetStoneFeatures(features, c);
    for (size_t i = 0; i < f.size(); ++i) {
      if (i != FeatureType::template GetPlaneIdx<ToPlayFeature>()) {
        EXPECT_EQ(0, f[i]);
      } else {
        EXPECT_EQ(1, f[i]);
      }
    }
  }
}

// Verifies SetFeatures for an empty board with white to play.
TYPED_TEST(DualNetTest, TestEmptyBoardWhiteToPlay) {
  using FeatureType = TypeParam;

  TestablePosition board("", Color::kWhite);
  ModelInput input;
  input.sym = symmetry::kIdentity;
  input.position_history.push_back(&board);

  BoardFeatureBuffer<float> buffer;
  Tensor<float> features = {1, kN, kN, FeatureType::kNumPlanes, buffer.data()};
  FeatureType::Set({&input}, &features);

  for (int c = 0; c < kN * kN; ++c) {
    auto f = GetStoneFeatures(features, c);
    for (size_t i = 0; i < f.size(); ++i) {
      EXPECT_EQ(0, f[i]);
    }
  }
}

// Verifies SetFeatures.
TYPED_TEST(DualNetTest, TestSetFeatures) {
  using FeatureType = TypeParam;

  TestablePosition board("");

  std::vector<std::string> moves = {"B9", "H9", "A8", "J9",
                                    "D5", "A1", "A2", "J1"};
  std::deque<TestablePosition> positions;
  for (const auto& move : moves) {
    board.PlayMove(move);
    positions.push_front(board);
  }

  ModelInput input;
  input.sym = symmetry::kIdentity;
  for (const auto& p : positions) {
    input.position_history.push_back(&p);
  }

  BoardFeatureBuffer<float> buffer;
  Tensor<float> features = {1, kN, kN, FeatureType::kNumPlanes, buffer.data()};
  FeatureType::Set({&input}, &features);

  //                        B0 W0 B1 W1 B2 W2 B3 W3 B4 W4 B5 W5 B6 W6 B7 W7 C
  std::vector<float> b9 = {{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}};
  std::vector<float> h9 = {{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1}};
  std::vector<float> a8 = {{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1}};
  std::vector<float> j9 = {{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1}};
  std::vector<float> d5 = {{1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}};
  std::vector<float> a1 = {{0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}};
  std::vector<float> a2 = {{1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}};
  std::vector<float> j1 = {{0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}};
  std::vector<float> b1 = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}};

  if (std::is_same<FeatureType, ExtraFeatures>::value) {
    //                   L1 L2 L3 C1 C2 C3 C4 C5 C6 C7 C8
    b9.insert(b9.end(), {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0});
    h9.insert(h9.end(), {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0});
    a8.insert(a8.end(), {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0});
    j9.insert(j9.end(), {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0});
    d5.insert(d5.end(), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    a1.insert(a1.end(), {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    a2.insert(a2.end(), {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    j1.insert(j1.end(), {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    b1.insert(b1.end(), {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0});
  }

  EXPECT_EQ(b9, GetStoneFeatures(features, Coord::FromString("B9")));
  EXPECT_EQ(h9, GetStoneFeatures(features, Coord::FromString("H9")));
  EXPECT_EQ(a8, GetStoneFeatures(features, Coord::FromString("A8")));
  EXPECT_EQ(j9, GetStoneFeatures(features, Coord::FromString("J9")));
  EXPECT_EQ(d5, GetStoneFeatures(features, Coord::FromString("D5")));
  EXPECT_EQ(a1, GetStoneFeatures(features, Coord::FromString("A1")));
  EXPECT_EQ(a2, GetStoneFeatures(features, Coord::FromString("A2")));
  EXPECT_EQ(j1, GetStoneFeatures(features, Coord::FromString("J1")));
  EXPECT_EQ(b1, GetStoneFeatures(features, Coord::FromString("B1")));
}

// Verfies that features work as expected when capturing.
TYPED_TEST(DualNetTest, TestStoneFeaturesWithCapture) {
  using FeatureType = TypeParam;

  TestablePosition board("");

  std::vector<std::string> moves = {"J3", "pass", "H2", "J2",
                                    "J1", "pass", "J2"};
  std::deque<TestablePosition> positions;
  for (const auto& move : moves) {
    board.PlayMove(move);
    positions.push_front(board);
  }

  ModelInput input;
  input.sym = symmetry::kIdentity;
  for (const auto& p : positions) {
    input.position_history.push_back(&p);
  }

  BoardFeatureBuffer<float> buffer;
  Tensor<float> features = {1, kN, kN, FeatureType::kNumPlanes, buffer.data()};
  FeatureType::Set({&input}, &features);

  //                        W0 B0 W1 B1 W2 B2 W3 B3 W4 B4 W5 B5 W6 B6 W7 B7 C
  std::vector<float> j2 = {{0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  if (std::is_same<FeatureType, ExtraFeatures>::value) {
    //                   L1 L2 L3 C1 C2 C3 C4 C5 C6 C7 C8
    j2.insert(j2.end(), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  }
  EXPECT_EQ(j2, GetStoneFeatures(features, Coord::FromString("J2")));
}

// Checks that the different backends produce the same result.
TYPED_TEST(DualNetTest, TestBackendsEqual) {
  using FeatureType = TypeParam;

  if (!std::is_same<FeatureType, AgzFeatures>::value) {
    // TODO(tommadams): generate models for other feature types.
    return;
  }

  struct Test {
    Test(std::unique_ptr<ModelFactory> factory, std::string basename)
        : factory(std::move(factory)), basename(std::move(basename)) {}
    std::unique_ptr<ModelFactory> factory;
    std::string basename;
  };

  std::map<std::string, Test> tests;
#if MG_ENABLE_TF_DUAL_NET
  tests.emplace("TfDualNet",
                Test(absl::make_unique<TfDualNetFactory>(std::vector<int>()),
                     "test_model.pb"));
#endif
#if MG_ENABLE_LITE_DUAL_NET
  tests.emplace("LiteDualNet", Test(absl::make_unique<LiteDualNetFactory>(),
                                    "test_model.tflite"));
#endif

  Random rnd(Random::kUniqueSeed, Random::kUniqueStream);
  ModelInput input;
  input.sym = symmetry::kIdentity;
  TestablePosition position("");
  for (int i = 0; i < kN * kN; ++i) {
    auto c = GetRandomLegalMove(position, &rnd);
    position.PlayMove(c);
  }
  input.position_history.push_back(&position);

  ModelOutput ref_output;
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

    auto model =
        test.factory->NewModel(absl::StrCat("cc/dual_net/", test.basename));

    ModelOutput output;
    std::vector<const ModelInput*> inputs = {&input};
    std::vector<ModelOutput*> outputs = {&output};
    model->RunMany(inputs, &outputs, nullptr);

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

TEST(WouldCaptureTest, WouldCaptureBlack) {
  TestablePosition board(R"(
      OOOX.XOOX
      OXX....X.
      .OOX.....
      OOOOX....
      XXXXX....)");

  ModelInput input;
  input.sym = symmetry::kIdentity;
  input.position_history.push_back(&board);

  BoardFeatureBuffer<float> buffer;
  Tensor<float> features = {1, kN, kN, ExtraFeatures::kNumPlanes,
                            buffer.data()};
  ExtraFeatures::Set({&input}, &features);

  //                        W0 B0 W1 B1 W2 B2 W3 B3 W4 B4 W5 B5 W6 B6 W7 B7 C
  std::vector<float> a7 = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}};
  std::vector<float> g8 = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}};
  //                   L1 L2 L3 C1 C2 C3 C4 C5 C6 C7 C8
  a7.insert(a7.end(), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1});
  g8.insert(g8.end(), {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0});
  EXPECT_EQ(a7, GetStoneFeatures(features, Coord::FromString("A7")));
  EXPECT_EQ(g8, GetStoneFeatures(features, Coord::FromString("G8")));
}

TEST(WouldCaptureTest, WouldCaptureWhite) {
  TestablePosition board(R"(
      XXXO.OXXO
      XOO....O.
      .XXO.....
      XXXXO....
      OOOOO....)",
                         Color::kWhite);

  ModelInput input;
  input.sym = symmetry::kIdentity;
  input.position_history.push_back(&board);

  BoardFeatureBuffer<float> buffer;
  Tensor<float> features = {1, kN, kN, ExtraFeatures::kNumPlanes,
                            buffer.data()};
  ExtraFeatures::Set({&input}, &features);

  //                        W0 B0 W1 B1 W2 B2 W3 B3 W4 B4 W5 B5 W6 B6 W7 B7 C
  std::vector<float> a7 = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  std::vector<float> g8 = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  //                   L1 L2 L3 C1 C2 C3 C4 C5 C6 C7 C8
  a7.insert(a7.end(), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1});
  g8.insert(g8.end(), {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0});
  EXPECT_EQ(a7, GetStoneFeatures(features, Coord::FromString("A7")));
  EXPECT_EQ(g8, GetStoneFeatures(features, Coord::FromString("G8")));
}

}  // namespace
}  // namespace minigo
