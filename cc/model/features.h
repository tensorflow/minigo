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

#ifndef CC_MODEL_FEATURES_H_
#define CC_MODEL_FEATURES_H_

#include <array>
#include <utility>
#include <vector>

#include "cc/color.h"
#include "cc/constants.h"
#include "cc/logging.h"
#include "cc/model/features_internal.h"
#include "cc/model/types.h"
#include "cc/platform/utils.h"
#include "cc/position.h"

// This header contains the different kinds of input features that we pass to
// models. Each set of input features (the stones on the board, whose turn
// it is, liberty counts, etc) is represented as a templated struct where the
// template type T is the feature type (uint8, float).
//
// Each feature struct has two static members:
//  - kNumPlanes : the number of planes for this feature.
//  - Set : a method that sets the feature planes on an input tensor.

namespace minigo {

// Input feature planes for stones on the board over the most recent N moves.
// Up to 8 feature planes X_t indicate the presence of the current player's
// stones at time t. A further up to 8 feature planes Y_t indicate the presence
// of the opposing player's stones at time t:
//   [X_t, Y_t, X_t-1, Y_t-1, ..., X_t-7, Y_t-7].
struct StoneFeatures {
  static constexpr int kNumPlanes = 2 * kMaxPositionHistory;

  template <typename T>
  MG_ALWAYS_INLINE static void Set(const ModelInput& input, int num_planes,
                                   T* dst) {
    auto my_color = input.position_history[0]->to_play();
    auto their_color = OtherColor(my_color);

    // Write the features for the position history that we have.
    int j = 0;
    for (; j < input.position_history.size(); ++j) {
      const auto* src = input.position_history[j]->stones().data();
      const auto* end = dst + kN * kN * num_planes;
      for (auto* d = dst + j * 2; d < end; d += num_planes) {
        auto color = src->color();
        src += 1;
        d[0] = color == my_color ? 1 : 0;
        d[1] = color == their_color ? 1 : 0;
      }
    }

    // Pad the features with zeros if we have fewer than 8 moves of history.
    for (; j < kMaxPositionHistory; ++j) {
      const auto* end = dst + kN * kN * num_planes;
      for (auto* d = dst + j * 2; d < end; d += num_planes) {
        d[0] = 0;
        d[1] = 0;
      }
    }
  }
};

// Input feature plane containing all 1s if it's blacks turn to play, or all 0s.
struct ToPlayFeature {
  static constexpr int kNumPlanes = 1;

  template <typename T>
  MG_ALWAYS_INLINE static void Set(const ModelInput& input, int num_planes,
                                   T* dst) {
    T f = input.position_history[0]->to_play() == Color::kBlack ? 1 : 0;
    const auto* end = dst + kN * kN * num_planes;
    for (auto* d = dst; d < end; d += num_planes) {
      d[0] = f;
    }
  }
};

// Input feature planes that describe chains with only a few remaining
// liberties.
struct LibertyFeatures {
  static constexpr int kNumPlanes = 3;

  template <typename T>
  MG_ALWAYS_INLINE static void Set(const ModelInput& input, int num_planes,
                                   T* dst) {
    const auto& position = *input.position_history[0];
    for (int i = 0; i < kN * kN; ++i) {
      auto num_liberties = position.num_chain_liberties(i);
      dst[0] = num_liberties == 1 ? 1 : 0;
      dst[1] = num_liberties == 2 ? 1 : 0;
      dst[2] = num_liberties >= 3 ? 1 : 0;
      dst += num_planes;
    }
  }
};

struct WouldCaptureFeature {
  static constexpr int kNumPlanes = 1;

  template <typename T>
  MG_ALWAYS_INLINE static void Set(const ModelInput& input, int num_planes,
                                   T* dst) {
    const auto& position = *input.position_history[0];
    auto my_color = position.to_play();
    auto their_color = OtherColor(my_color);
    const auto& stones = position.stones();

    for (int i = 0; i < kN * kN; ++i) {
      int f = 0;
      if (position.legal_move(i)) {
        for (auto nc : kNeighborCoords[i]) {
          if (stones[nc].color() == their_color &&
              position.num_chain_liberties(nc) == 1) {
            f = 1;
            break;
          }
        }
      }
      dst[0] = f;
      dst += num_planes;
    }
  }
};

// TODO(tommadams): Move Features and FeaturesDescriptor into another header so
// that features.h only contains the feature types structs.
// TODO(tommadams): Move the framework tests from features_test.cc to
// model_test.cc too.

// `Features` encodes the input tensor type `T` and the list of input features
// to be used `Fs`.
template <typename... Fs>
struct Features {
  using Impl = internal::FeaturesImpl<Fs...>;

  // Total number of input feature planes.
  static constexpr int kNumPlanes = Impl::kNumPlanes;

  // Generate the input features in `Fs` from `input` and write them to
  // the `features` tensor.
  // CHECK fails if the number of channels in the `features` doesn't match the
  // number of feature planes `kNumPlanes`.
  template <typename T>
  static void Set(const std::vector<const ModelInput*>& inputs,
                  Tensor<T>* features) {
    MG_CHECK(features->h == kN && features->w && kN &&
             features->c == Impl::kNumPlanes)
        << features->h << " " << features->w << " " << features->c;
    int stride = features->h * features->w * features->c;
    auto* data = features->data;
    std::array<T, kN * kN * kNumPlanes> raw_features;
    for (const auto* input : inputs) {
      Impl::SetAll(*input, features->c, raw_features.data());
      symmetry::ApplySymmetry<kN, kNumPlanes>(input->sym, raw_features.data(),
                                              data);
      data += stride;
    }
  }

  // Returns the index in the list of feature planes of `FeatureType`, or -1
  // if `FeatureType` isn't in the list.
  // For example:
  //   using MyFeatures = Features<StoneFeatures, ToPlayFeature>;
  //
  //   // StoneFeatures is the first set of features, so trivially has index 0.
  //   MyFeatures::GetPlaneIdx<StoneFeatures>() == 0;
  //
  //   // StoneFeatures has 16 planes.
  //   MyFeatures::GetPlaneIdx<ToPlayFeature>() == 16;
  //
  //   // int isn't in the list of features.
  //   MyFeatures::GetPlaneIdx<int>() == -1;
  template <typename FeatureType>
  static constexpr int GetPlaneIdx() {
    return Impl::template GetPlaneIdx<FeatureType>(0);
  }
};

// Descriptor for a model's input features.
// Basically it turns compile-time information about a set of input features
// encoded as a `Features<...>` type into run-time information.
struct FeatureDescriptor {
  template <typename T>
  using SetFeatures = void (*)(const std::vector<const ModelInput*>&,
                               Tensor<T>*);

  template <typename FeatureType>
  static FeatureDescriptor Create() {
    return FeatureDescriptor{FeatureType::kNumPlanes,
                             &FeatureType::template Set<uint8_t>,
                             &FeatureType::template Set<float>};
  }

  int num_planes;
  SetFeatures<uint8_t> set_bytes;
  SetFeatures<float> set_floats;
};

using AgzFeatures = Features<StoneFeatures, ToPlayFeature>;
using ExtraFeatures = Features<StoneFeatures, ToPlayFeature, LibertyFeatures,
                               WouldCaptureFeature>;

// Maximum number of feature planes used by these features.
constexpr int kMaxNumFeaturePlanes =
    internal::GetMaxNumFeaturePlanes<AgzFeatures, ExtraFeatures>();

// A buffer large enough to hold features for all input types.
template <typename T>
using BoardFeatureBuffer = std::array<T, kN * kN * kMaxNumFeaturePlanes>;

}  // namespace minigo

#endif  //  CC_MODEL_FEATURES_H_
