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

#include "cc/dual_net.h"

#include "cc/constants.h"
#include "cc/symmetries.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"

using tensorflow::DT_FLOAT;
using tensorflow::Env;
using tensorflow::GraphDef;
using tensorflow::NewSession;
using tensorflow::ReadBinaryProto;
using tensorflow::Session;
using tensorflow::SessionOptions;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::error::CANCELLED;

namespace minigo {

constexpr int DualNet::kNumStoneFeatures;
constexpr int DualNet::kNumBoardFeatures;

void DualNet::InitializeFeatures(const Position& position,
                                 BoardFeatures* features) {
  const auto my_color = position.to_play();
  const auto their_color = OtherColor(my_color);
  const float to_play = my_color == Color::kBlack ? 1 : 0;

  for (int i = 0; i < kN * kN; ++i) {
    int j = i * kNumStoneFeatures;
    auto stone_color = position.stones()[i].color();
    auto my_stone = stone_color == my_color ? 1 : 0;
    auto their_stone = stone_color == their_color ? 1 : 0;
    for (int plane = 0; plane < kPlayerFeature; plane += 2) {
      (*features)[j++] = my_stone;
      (*features)[j++] = their_stone;
    }
    (*features)[j++] = to_play;
  }
}

// The update loop here is a little tricky.
//
// The chart below shows, for each move, how the stones from the last 8 moves
// should be distributed through the input planes.
//
//                                     planes
//   move | to play |   0    1    2    3    4    5   ...  16
//  ------+---------+-----------------------------------------
//     1  |    B    |  B_1  W_1   -    -    -    -   ...   1
//     2  |    W    |  W_2  B_2  W_1  B_1   -    -   ...   0
//     3  |    B    |  B_3  W_3  B_2  W_2  B_1  W_1  ...   1
//     4  |    W    |  W_4  B_4  W_3  B_3  W_2  B_2  ...   0
//    ... |   ...   |  ...  ...  ...  ...  ...  ...  ...  ...
//
// For example: on move 3, planes 0 & 1 hold the black & white stones that are
// on the board before move 3 is played, planes 2 & 3 hold the stones that were
// on the board before move 2 was played, planes 4 & 5 hold the stones that
// were on the board before move 1 was played, etc.
//
// So... to update the features, we need to do four things:
//  1) Shuffle the planes for moves t .. t-6 over to the planes for moves
//     t-1 .. t-7.
//  2) Swap the black and white planes for moves t-1 .. t-7.
//  3) Write the new black and white stones into planes 0 & 1 (or planes 1 & 0
//     depending on who is to play first).
//  4) Write the "to play" feature into plane 16.
//
// Steps 3 and 4 are trivial.
//
// Steps 1 and 2 can be accomplished in one by the following:
//  1) Copy even planes from plane N to plane N + 3.
//  2) Copy odd planes from plane N to plane N + 1.
//
// The code below does this slightly differently, updated the planes in the
// reverse order because that allows old_features and new_features to point to
// the same array, but the end result is the same.
void DualNet::UpdateFeatures(const BoardFeatures& old_features,
                             const Position& position,
                             BoardFeatures* new_features) {
  const auto my_color = position.to_play();
  const auto their_color = OtherColor(my_color);
  const float to_play = my_color == Color::kBlack ? 1 : 0;

  for (int i = 0; i < kN * kN; ++i) {
    auto stone_color = position.stones()[i].color();
    const auto* src = old_features.data() + i * kNumStoneFeatures;
    auto* dst = new_features->data() + i * kNumStoneFeatures;

    dst[kPlayerFeature] = to_play;
    for (int j = kPlayerFeature - 2; j > 0; j -= 2) {
      dst[j + 1] = src[j - 2];
      dst[j] = src[j - 1];
    }
    dst[1] = stone_color == their_color ? 1 : 0;
    dst[0] = stone_color == my_color ? 1 : 0;
  }
}

DualNet::DualNet() = default;

DualNet::~DualNet() {
  if (session_ != nullptr) {
    session_->Close();
  }
}

Status DualNet::Initialize(const std::string& graph_path) {
  GraphDef graph_def;
  TF_RETURN_IF_ERROR(ReadBinaryProto(Env::Default(), graph_path, &graph_def));
  session_.reset(NewSession(SessionOptions()));
  TF_RETURN_IF_ERROR(session_->Create(graph_def));

  inputs_.clear();
  inputs_.emplace_back(
      "pos_tensor",
      Tensor(DT_FLOAT, TensorShape({1, kN, kN, kNumStoneFeatures})));

  output_names_.clear();
  output_names_.push_back("policy_output");
  output_names_.push_back("value_output");

  return Status::OK();
}

void DualNet::RunMany(absl::Span<const BoardFeatures* const> features,
                      absl::Span<Output> outputs, Random* rnd) {
  assert(features.size() == outputs.size());

  int batch_size = static_cast<int>(features.size());
  auto& feature_tensor = inputs_[0].second;
  if (feature_tensor.dim_size(0) != batch_size) {
    feature_tensor =
        Tensor(DT_FLOAT, TensorShape({batch_size, kN, kN, kNumStoneFeatures}));
  }

  // Select symmetry operations to apply.
  symmetries_used_.clear();
  if (rnd != nullptr) {
    symmetries_used_.reserve(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      symmetries_used_.push_back(static_cast<symmetry::Symmetry>(
          rnd->UniformInt(0, symmetry::kNumSymmetries - 1)));
    }
  } else {
    symmetries_used_.resize(batch_size, symmetry::kIdentity);
  }

  // Copy the features into the input tensor.
  for (int i = 0; i < batch_size; ++i) {
    symmetry::ApplySymmetry<float, kN, kNumStoneFeatures>(
        symmetries_used_[i], features[i]->data(),
        feature_tensor.flat<float>().data() + i * kNumBoardFeatures);
  }

  // Run the model.
  auto status = session_->Run(inputs_, output_names_, {}, &outputs_);
  assert(status.ok());

  // Copy the policy and value out of the output tensors.
  const auto& policy_tensor = outputs_[0].flat<float>();
  const auto& value_tensor = outputs_[1].flat<float>();
  for (int i = 0; i < batch_size; ++i) {
    const auto* policy_tensor_data = policy_tensor.data() + i * kNumMoves;

    symmetry::ApplySymmetry<float, kN, 1>(
        symmetry::Inverse(symmetries_used_[i]), policy_tensor_data,
        outputs[i].policy.data());
    outputs[i].policy[Coord::kPass] = policy_tensor_data[Coord::kPass];

    outputs[i].value = value_tensor.data()[i];
  }
}

}  // namespace minigo
