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

#include "cc/color.h"
#include "cc/constants.h"
#include "wtf/macros.h"

namespace minigo {

// Generates the board features from the history of recent moves, where
// history[0] is the current board position, and history[i] is the board
// position from i moves ago.
// history.size() must be <= kMoveHistory.
// TODO(tommadams): make this a templated class, so it can set either uint8 or
// float features.
void DualNet::SetInputs(const std::vector<const Input*>& model_inputs,
                        FeatureType feature_type, Tensor* features) {
  MG_CHECK(static_cast<int>(model_inputs.size()) <= features->n);
  MG_CHECK(features->w == kN && features->h == kN);

  std::array<float, kMaxBoardFeaturesSize> raw_features;
  for (size_t input_idx = 0; input_idx < model_inputs.size(); ++input_idx) {
    const auto& model_input = *model_inputs[input_idx];
    const auto& history = model_input.position_history;
    const auto& position = *history[0];

    MG_CHECK(history.size() <= DualNet::kMoveHistory);

    Color my_color = position.to_play();
    Color their_color = OtherColor(my_color);

    auto features_size = features->w * features->h * features->c;

    // Write the features for the position history that we have.
    int j = 0;
    for (j = 0; j < history.size(); ++j) {
      auto* dst = raw_features.data() + j * 2;
      const auto* end = dst + features_size;
      const auto* src = history[j]->stones().data();
      while (dst < end) {
        auto color = src->color();
        ++src;
        dst[0] = color == my_color ? 1 : 0;
        dst[1] = color == their_color ? 1 : 0;
        dst += features->c;
      }
    }

    // Pad the features with zeros if we have fewer than 8 moves of history.
    for (; j < DualNet::kMoveHistory; ++j) {
      auto* dst = raw_features.data() + j * 2;
      const auto* end = dst + features_size;
      while (dst < end) {
        dst[0] = 0;
        dst[1] = 0;
        dst += features->c;
      }
    }

    // Set the "to play" feature plane.
    float to_play_feature = my_color == Color::kBlack ? 1 : 0;
    auto* dst = raw_features.data() + DualNet::kPlayerFeature;
    const auto* end = dst + features_size;
    while (dst < end) {
      dst[0] = to_play_feature;
      dst += features->c;
    }

    switch (feature_type) {
      case FeatureType::kAgz:
        // No extra features to set.
        MG_CHECK(features->c == kNumAgzStoneFeatures);
        symmetry::ApplySymmetry<kN, DualNet::kNumAgzStoneFeatures>(
            model_input.sym, raw_features.data(),
            features->data + input_idx * features_size);
        break;

      case FeatureType::kExtra: {
        // Set the liberties features.
        dst = raw_features.data() + kNumAgzStoneFeatures;
        for (int i = 0; i < kN * kN; ++i) {
          auto num_liberties = position.num_chain_liberties(i);
          dst[0] = dst[1] = dst[2] = 0;
          if (num_liberties >= 1 && num_liberties <= 3) {
            dst[num_liberties - 1] = 1;
          }
          dst += features->c;
        }

        MG_CHECK(features->c == kNumExtraStoneFeatures);
        symmetry::ApplySymmetry<kN, DualNet::kNumExtraStoneFeatures>(
            model_input.sym, raw_features.data(),
            features->data + input_idx * features_size);
        break;
      }

      default:
        MG_LOG(FATAL) << "unexpected features type "
                      << static_cast<int>(feature_type);
    }
  }
}

void DualNet::GetOutputs(const std::vector<const Input*>& model_inputs,
                         const DualNet::Tensor& policy,
                         const DualNet::Tensor& value,
                         std::vector<Output*>* model_outputs) {
  MG_CHECK(model_outputs->size() == model_inputs.size());
  MG_CHECK(policy.n == value.n);
  MG_CHECK(static_cast<int>(model_inputs.size()) <= policy.n);

  // Copy the policy and value out of the output tensors.
  for (size_t input_idx = 0; input_idx < model_inputs.size(); ++input_idx) {
    const auto sym = model_inputs[input_idx]->sym;
    const auto* raw_policy = policy.data + kNumMoves * input_idx;
    const auto* raw_value = value.data + input_idx;
    auto& model_output = *(*model_outputs)[input_idx];

    symmetry::ApplySymmetry<kN, 1>(symmetry::Inverse(sym), raw_policy,
                                   model_output.policy.data());
    model_output.policy[Coord::kPass] = raw_policy[Coord::kPass];
    model_output.value = *raw_value;
  }
}

DualNet::~DualNet() = default;

}  // namespace minigo
