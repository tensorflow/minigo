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
#include "cc/model/loader.h"
#include "cc/model/model.h"
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
  MG_CHECK(features.shape.is({1, kN, kN, -1}));
  for (int i = 0; i < features.shape[3]; ++i) {
    result.push_back(features.data[c * features.shape[3] + i]);
  }
  return result;
}

// Helper function to make AlphaGo Zero-like features (stones and to-play
// features) for input features that use different sizes of move history.
template <typename T, typename F>
std::vector<T> MakeZeroFeatures(std::initializer_list<T> stones, T to_play) {
  // Make sure that caller has provided enough stone features.
  MG_CHECK(stones.size() >= F::kNumStonePlanes);

  // Take only the required number of stone features.
  std::vector<T> result(stones);
  result.resize(F::kNumStonePlanes);

  // Append the to-play feature.
  result.push_back(to_play);
  return result;
}

template <typename F>
class DualNetTest : public ::testing::Test {};

using TestFeatureTypes = ::testing::Types<AgzFeatures, Mlperf07Features>;
TYPED_TEST_CASE(DualNetTest, TestFeatureTypes);

// Verifies SetFeatures an empty board with black to play.
TYPED_TEST(DualNetTest, TestEmptyBoardBlackToPlay) {
  using FeatureType = TypeParam;

  TestablePosition board("");
  ModelInput input;
  input.sym = symmetry::kIdentity;
  input.position_history.push_back(&board);

  BoardFeatureBuffer<float> buffer;
  Tensor<float> features = {{1, kN, kN, FeatureType::kNumPlanes},
                            buffer.data()};
  FeatureType::SetNhwc({&input}, &features);

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
  Tensor<float> features = {{1, kN, kN, FeatureType::kNumPlanes},
                            buffer.data()};
  FeatureType::SetNhwc({&input}, &features);

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
  Tensor<float> features = {{1, kN, kN, FeatureType::kNumPlanes},
                            buffer.data()};
  FeatureType::SetNhwc({&input}, &features);

  const auto& mzf = MakeZeroFeatures<float, FeatureType>;
  //             B0 W0 B1 W1 B2 W2 B3 W3 B4 W4 B5 W5 B6 W6 B7 W7  C
  auto b9 = mzf({1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, 1);
  auto h9 = mzf({0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0}, 1);
  auto a8 = mzf({1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0}, 1);
  auto j9 = mzf({0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0}, 1);
  auto d5 = mzf({1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1);
  auto a1 = mzf({0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1);
  auto a2 = mzf({1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1);
  auto j1 = mzf({0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1);
  auto b1 = mzf({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1);

  if (std::is_same<FeatureType, Mlperf07Features>::value) {
    //                   L1 L2 L3 WC
    b9.insert(b9.end(), {0, 0, 1, 0});
    h9.insert(h9.end(), {0, 0, 1, 0});
    a8.insert(a8.end(), {0, 0, 1, 0});
    j9.insert(j9.end(), {0, 0, 1, 0});
    d5.insert(d5.end(), {0, 0, 1, 0});
    a1.insert(a1.end(), {1, 0, 0, 0});
    a2.insert(a2.end(), {0, 1, 0, 0});
    j1.insert(j1.end(), {0, 1, 0, 0});
    b1.insert(b1.end(), {0, 0, 0, 1});
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
  Tensor<float> features = {{1, kN, kN, FeatureType::kNumPlanes},
                            buffer.data()};
  FeatureType::SetNhwc({&input}, &features);

  const auto& mzf = MakeZeroFeatures<float, FeatureType>;
  //             W0 B0 W1 B1 W2 B2 W3 B3 W4 B4 W5 B5 W6 B6 W7 B7  C
  auto j2 = mzf({0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0);
  if (std::is_same<FeatureType, Mlperf07Features>::value) {
    //                   L1 L2 L3 WC
    j2.insert(j2.end(), {0, 0, 1, 0});
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

  std::vector<std::string> model_basenames;
#if MG_ENABLE_TF_DUAL_NET
  model_basenames.push_back("test_tf.minigo");
#endif
#if MG_ENABLE_LITE_DUAL_NET
  model_basenames.push_back("test_lite.minigo");
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

  for (const auto& name : model_basenames) {
    MG_LOG(INFO) << "Loading " << name;
    auto model = NewModel(absl::StrCat("cc/dual_net/", name), "");

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

  ShutdownModelFactories();
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
  Tensor<float> features = {{1, kN, kN, Mlperf07Features::kNumPlanes},
                            buffer.data()};
  Mlperf07Features::SetNhwc({&input}, &features);

  const auto& mzf = MakeZeroFeatures<float, Mlperf07Features>;
  //             W0 B0 W1 B1 W2 B2 W3 B3 W4 B4 W5 B5 W6 B6 W7 B7  C
  auto a7 = mzf({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1);
  auto g8 = mzf({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1);
  //                   L1 L2 L3 WC
  a7.insert(a7.end(), {0, 0, 0, 1});
  g8.insert(g8.end(), {0, 0, 0, 1});
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
  Tensor<float> features = {{1, kN, kN, Mlperf07Features::kNumPlanes},
                            buffer.data()};
  Mlperf07Features::SetNhwc({&input}, &features);

  const auto& mzf = MakeZeroFeatures<float, Mlperf07Features>;
  //             W0 B0 W1 B1 W2 B2 W3 B3 W4 B4 W5 B5 W6 B6 W7 B7  C
  auto a7 = mzf({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0);
  auto g8 = mzf({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0);
  //                   L1 L2 L3 WC
  a7.insert(a7.end(), {0, 0, 0, 1});
  g8.insert(g8.end(), {0, 0, 0, 1});
  EXPECT_EQ(a7, GetStoneFeatures(features, Coord::FromString("A7")));
  EXPECT_EQ(g8, GetStoneFeatures(features, Coord::FromString("G8")));
}

}  // namespace
}  // namespace minigo
