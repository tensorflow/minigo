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

#include <emmintrin.h>

#include <array>
#include <utility>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
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
//  - SetNhwc : a method that sets NHWC feature on an input tensor.
//  - SetNchw : a method that sets NCWH feature on an input tensor.

namespace minigo {

// Input feature planes for stones on the board over the most recent N moves.
// Up to 8 feature planes X_t indicate the presence of the current player's
// stones at time t. A further up to 8 feature planes Y_t indicate the presence
// of the opposing player's stones at time t:
//   [X_t, Y_t, X_t-1, Y_t-1, ..., X_t-7, Y_t-7].
template <int PositionHistory>
struct StoneFeatures {
  static_assert(PositionHistory <= kMaxPositionHistory,
                "PositionHistory too large");

  static constexpr int kNumPlanes = 2 * PositionHistory;

  template <typename T>
  MG_ALWAYS_INLINE static void SetNhwc(const ModelInput& input, int num_planes,
                                       T* dst) {
    auto my_color = input.position_history[0]->to_play();
    auto their_color = OtherColor(my_color);

    auto n = std::min(input.position_history.size(), PositionHistory);

    // Write the features for the position history that we have.
    int j = 0;
    for (; j < n; ++j) {
      const auto* src = input.position_history[j]->stones().data();
      const auto* end = dst + kNumPoints * num_planes;
      for (auto* d = dst + j * 2; d < end; d += num_planes) {
        auto color = src->color();
        src += 1;
        d[0] = color == my_color;
        d[1] = color == their_color;
      }
    }

    // Pad the features with zeros if we have fewer than 8 moves of history.
    for (; j < PositionHistory; ++j) {
      const auto* end = dst + kNumPoints * num_planes;
      for (auto* d = dst + j * 2; d < end; d += num_planes) {
        d[0] = 0;
        d[1] = 0;
      }
    }
  }

  MG_ALWAYS_INLINE static void SetNchw(const ModelInput& input, float* dst) {
    auto my_color = input.position_history[0]->to_play();
    auto their_color = OtherColor(my_color);

    auto n = std::min(input.position_history.size(), PositionHistory);

    // Write the features for the position history that we have.
    int j = 0;
    for (; j < n; ++j) {
      const auto& stones = input.position_history[j]->stones();
      for (const auto& stone : stones) {
        *dst++ = stone.color() == my_color;
      }
      for (const auto& stone : stones) {
        *dst++ = stone.color() == their_color;
      }
    }

    // Pad the features with zeros if we have fewer than 8 moves of history.
    for (; j < PositionHistory; ++j) {
      for (int i = 0; i < 2 * kNumPoints; ++i) {
        *dst++ = 0;
      }
    }
  }

  MG_ALWAYS_INLINE static void SetNchw(const ModelInput& input, uint8_t* dst) {
    auto my_color = input.position_history[0]->to_play();
    auto their_color = OtherColor(my_color);

    auto n = std::min(input.position_history.size(), PositionHistory);

    // Stones are stored as bit-packed uint16_t, with the color stored in the
    // bottom two bits.
    static_assert(sizeof(Stone) == 2, "");
    __m128i color_mask = _mm_set1_epi16(3);

    // Initialize useful single byte values.
    __m128i one = _mm_set1_epi8(1);
    __m128i my_color_mm = _mm_set1_epi8(static_cast<int>(my_color));
    __m128i their_color_mm = _mm_set1_epi8(static_cast<int>(their_color));

    // Write the features for the position history that we have.
    int safe_size = (kNumPoints / 16) * 16;
    int j = 0;
    for (; j < n; ++j) {
      const auto* stones = input.position_history[j]->stones().data();
      int i = 0;
      for (; i < safe_size; i += 16) {
        // Load 16 stones.
        __m128i a =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(stones + i));
        __m128i b =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(stones + i + 8));

        // Clear all but the bottom two bits of each stone, leaving just their
        // colors.
        a = _mm_and_si128(a, color_mask);
        b = _mm_and_si128(b, color_mask);

        // Extract the color values from a & b and pack them into a single.
        __m128i col = _mm_packus_epi16(a, b);

        // Generate 16 input features for two planes: current player's stones
        // and opponent's stones.
        // my[i] = col[i] == my_color ? 1 : 0.
        // their[i] = col[i] == their_color ? 1 : 0.
        __m128i my = _mm_and_si128(one, _mm_cmpeq_epi8(col, my_color_mm));
        __m128i their = _mm_and_si128(one, _mm_cmpeq_epi8(col, their_color_mm));

        // Store the input features.
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), my);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + kNumPoints), their);
        dst += 16;
      }

      // Finish up the last few input features.
      for (; i < kNumPoints; ++i) {
        auto color = stones[i].color();
        dst[0] = color == my_color;
        dst[kNumPoints] = color == their_color;
        dst += 1;
      }
      dst += kNumPoints;
    }

    // Pad the features with zeros if we have fewer than 8 moves of history.
    for (; j < PositionHistory; ++j) {
      for (int i = 0; i < 2 * kNumPoints; ++i) {
        *dst++ = 0;
      }
    }
  }
};

// Input feature plane containing all 1s if it's blacks turn to play, or all 0s.
struct ToPlayFeature {
  static constexpr int kNumPlanes = 1;

  template <typename T>
  MG_ALWAYS_INLINE static void SetNhwc(const ModelInput& input, int num_planes,
                                       T* dst) {
    T f = input.position_history[0]->to_play() == Color::kBlack;
    const auto* end = dst + kNumPoints * num_planes;
    for (auto* d = dst; d < end; d += num_planes) {
      d[0] = f;
    }
  }

  template <typename T>
  MG_ALWAYS_INLINE static void SetNchw(const ModelInput& input, T* dst) {
    T f = input.position_history[0]->to_play() == Color::kBlack;
    for (int i = 0; i < kNumPoints; ++i) {
      *dst++ = f;
    }
  }
};

// Input feature planes that describe chains with only a few remaining
// liberties.
struct LibertyFeatures {
  static constexpr int kNumPlanes = 3;

  template <typename T>
  MG_ALWAYS_INLINE static void SetNhwc(const ModelInput& input, int num_planes,
                                       T* dst) {
    const auto& position = *input.position_history[0];
    for (int i = 0; i < kNumPoints; ++i) {
      auto num_liberties = position.num_chain_liberties(i);
      dst[0] = num_liberties == 1;
      dst[1] = num_liberties == 2;
      dst[2] = num_liberties >= 3;
      dst += num_planes;
    }
  }

  template <typename T>
  MG_ALWAYS_INLINE static void SetNchw(const ModelInput& input, T* dst) {
    const auto& position = *input.position_history[0];
    auto* dst0 = dst;
    auto* dst1 = dst0 + kNumPoints;
    auto* dst2 = dst1 + kNumPoints;
    for (int i = 0; i < kNumPoints; ++i) {
      auto num_liberties = position.num_chain_liberties(i);
      *dst0++ = num_liberties == 1;
      *dst1++ = num_liberties == 2;
      *dst2++ = num_liberties >= 3;
    }
  }
};

struct WouldCaptureFeature {
  static constexpr int kNumPlanes = 1;

  template <typename T>
  MG_ALWAYS_INLINE static void SetNhwc(const ModelInput& input, int num_planes,
                                       T* dst) {
    const auto& position = *input.position_history[0];
    auto my_color = position.to_play();
    auto their_color = OtherColor(my_color);
    const auto& stones = position.stones();

    for (int i = 0; i < kNumPoints; ++i) {
      int f = 0;
      if (position.legal_move(i)) {
        for (auto nc : kNeighborCoords[i]) {
          f |= ((stones[nc].color() == their_color) &
                (position.num_chain_liberties(nc) == 1));
        }
      }
      dst[0] = f;
      dst += num_planes;
    }
  }

  template <typename T>
  MG_ALWAYS_INLINE static void SetNchw(const ModelInput& input, T* dst) {
    const auto& position = *input.position_history[0];
    auto my_color = position.to_play();
    auto their_color = OtherColor(my_color);
    const auto& stones = position.stones();

    for (int i = 0; i < kNumPoints; ++i) {
      int f = 0;
      if (position.legal_move(i)) {
        for (auto nc : kNeighborCoords[i]) {
          f |= ((stones[nc].color() == their_color) &
                (position.num_chain_liberties(nc) == 1));
        }
      }
      *dst++ = f;
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
  static void SetNhwc(absl::Span<const ModelInput* const> inputs,
                      Tensor<T>* features) {
    MG_CHECK(features->shape.is({-1, kN, kN, Impl::kNumPlanes}))
        << features->shape;

    int stride = features->shape[1] * features->shape[2] * features->shape[3];
    auto* data = features->data;
    std::array<T, kNumPoints * kNumPlanes> raw_features;
    for (const auto* input : inputs) {
      Impl::SetAllNhwc(*input, Impl::kNumPlanes, raw_features.data());
      symmetry::ApplySymmetry<kN, kNumPlanes>(input->sym, raw_features.data(),
                                              data);
      data += stride;
    }
  }

  template <typename T>
  static void SetNchw(absl::Span<const ModelInput* const> inputs,
                      Tensor<T>* features) {
    MG_CHECK(features->shape.is({-1, Impl::kNumPlanes, kN, kN}))
        << features->shape;

    int stride = features->shape[1] * features->shape[2] * features->shape[3];
    auto* data = features->data;
    std::array<T, kNumPoints * kNumPlanes> raw_features;
    for (const auto* input : inputs) {
      Impl::SetAllNchw(*input, raw_features.data());
      symmetry::ApplySymmetryPlanar<kN, kNumPlanes>(input->sym,
                                                    raw_features.data(), data);
      data += stride;
    }
  }

  // Returns the index in the list of feature planes of `FeatureType`, or -1
  // if `FeatureType` isn't in the list.
  // For example:
  //   using MyFeatures = Features<StoneFeatures<8>, ToPlayFeature>;
  //
  //   // StoneFeatures is the first set of features, so trivially has index 0.
  //   MyFeatures::GetPlaneIdx<StoneFeatures<8>>() == 0;
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
  enum class Layout {
    kNhwc,
    kNchw,
  };

  template <typename T>
  using SetFeaturesFn = void (*)(absl::Span<const ModelInput* const>,
                                 Tensor<T>*);

  template <typename FeatureType>
  static FeatureDescriptor Create(Layout layout) {
    switch (layout) {
      case Layout::kNhwc:
        return FeatureDescriptor{FeatureType::kNumPlanes, layout,
                                 &FeatureType::template SetNhwc<uint8_t>,
                                 &FeatureType::template SetNhwc<float>};
      case Layout::kNchw:
        return FeatureDescriptor{FeatureType::kNumPlanes, layout,
                                 &FeatureType::template SetNchw<uint8_t>,
                                 &FeatureType::template SetNchw<float>};
    }
    MG_LOG(FATAL) << "Invalid layout" << static_cast<int>(layout);
    return {};
  }

  static FeatureDescriptor Create(absl::string_view input_features,
                                  absl::string_view input_layout);

  // Returns the feature tensor shape for a batch size of `n` that uses the
  // descriptors input layout.
  TensorShape GetInputShape(int n) const {
    switch (layout) {
      case Layout::kNhwc:
        return TensorShape({n, kN, kN, num_planes});
      case Layout::kNchw:
        return TensorShape({n, num_planes, kN, kN});
    }
    MG_LOG(FATAL) << "Invalid layout" << static_cast<int>(layout);
    return {};
  }

  void SetFeatures(absl::Span<const ModelInput* const> inputs,
                   Tensor<uint8_t>* features) const {
    set_bytes(inputs, features);
  }

  void SetFeatures(absl::Span<const ModelInput* const> inputs,
                   Tensor<float>* features) const {
    set_floats(inputs, features);
  }

  int num_planes;
  Layout layout;
  SetFeaturesFn<uint8_t> set_bytes;
  SetFeaturesFn<float> set_floats;
};

using AgzFeatures = Features<StoneFeatures<8>, ToPlayFeature>;

// TODO(tommadams): rename ExtraFeatures to Mlperf07Features.
using ExtraFeatures = Features<StoneFeatures<4>, ToPlayFeature, LibertyFeatures,
                               WouldCaptureFeature>;

// Maximum number of feature planes used by these features.
constexpr int kMaxNumFeaturePlanes =
    internal::GetMaxNumFeaturePlanes<AgzFeatures, ExtraFeatures>();

// A buffer large enough to hold features for all input types.
template <typename T>
using BoardFeatureBuffer = std::array<T, kNumPoints * kMaxNumFeaturePlanes>;

}  // namespace minigo

#endif  //  CC_MODEL_FEATURES_H_
