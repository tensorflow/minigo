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

constexpr int DualNet::kNumStoneFeatures;
constexpr int DualNet::kNumBoardFeatures;

// Generates the board features from the history of recent moves, where
// history[0] is the current board position, and history[i] is the board
// position from i moves ago.
// history.size() must be <= kMoveHistory.
// TODO(tommadams): make this a templated class, so it can set either uint8 or
// float features.
void DualNet::SetFeatures(absl::Span<const Position::Stones* const> history,
                          Color to_play, DualNet::BoardFeatures* features) {
  MG_CHECK(history.size() <= DualNet::kMoveHistory);
  Color my_color = to_play;
  Color their_color = OtherColor(my_color);

  // Write the features for the position history that we have.
  size_t j = 0;
  for (j = 0; j < history.size(); ++j) {
    auto* dst = features->data() + j * 2;
    const auto* end = dst + DualNet::kNumBoardFeatures;
    const auto* src = history[j]->data();
    while (dst < end) {
      auto color = src->color();
      ++src;
      dst[0] = color == my_color ? 1 : 0;
      dst[1] = color == their_color ? 1 : 0;
      dst += DualNet::kNumStoneFeatures;
    }
  }

  // Pad the features with zeros if we have fewer than 8 moves of history.
  for (; j < DualNet::kMoveHistory; ++j) {
    auto* dst = features->data() + j * 2;
    const auto* end = dst + DualNet::kNumBoardFeatures;
    while (dst < end) {
      dst[0] = 0;
      dst[1] = 0;
      dst += DualNet::kNumStoneFeatures;
    }
  }

  // Set the "to play" feature plane.
  float to_play_feature = to_play == Color::kBlack ? 1 : 0;
  auto* dst = features->data() + DualNet::kPlayerFeature;
  const auto* end = dst + DualNet::kNumBoardFeatures;
  while (dst < end) {
    dst[0] = to_play_feature;
    dst += DualNet::kNumStoneFeatures;
  }
}

void DualNet::RunMany(const std::vector<const Input*>& inputs,
                      std::vector<Output*>* outputs, std::string* model_name) {
  WTF_SCOPE("DualNet::RunMany", size_t)(inputs.size());

  MG_CHECK(inputs.size() == outputs->size());

  // Generate input features & apply symmetries.
  // TODO(tommadams): apply symmetries in place.
  BoardFeatures raw_features;
  features_.resize(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    SetFeatures(inputs[i]->position_history, inputs[i]->to_play, &raw_features);
    symmetry::ApplySymmetry<kN, DualNet::kNumStoneFeatures>(
        inputs[i]->sym, raw_features.data(), features_[i].data());
  }

  raw_outputs_.resize(inputs.size());

  // Run inference.
  RunManyImpl(model_name);

  // Undo symmetries.
  for (size_t i = 0; i < raw_outputs_.size(); ++i) {
    const auto& raw_output = raw_outputs_[i];
    auto* final_output = (*outputs)[i];
    symmetry::ApplySymmetry<kN, 1>(symmetry::Inverse(inputs[i]->sym),
                                   raw_output.policy.data(),
                                   final_output->policy.data());
    final_output->policy[Coord::kPass] = raw_output.policy[Coord::kPass];
    final_output->value = raw_output.value;
  }
}

DualNet::DualNet(std::string name) : Model(std::move(name), 1) {}

DualNet::~DualNet() = default;

}  // namespace minigo
