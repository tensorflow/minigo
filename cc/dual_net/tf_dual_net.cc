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

#include "cc/dual_net/tf_dual_net.h"

#include "cc/check.h"
#include "cc/constants.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"

using tensorflow::DT_FLOAT;
using tensorflow::Env;
using tensorflow::GraphDef;
using tensorflow::NewSession;
using tensorflow::ReadBinaryProto;
using tensorflow::SessionOptions;
using tensorflow::Tensor;
using tensorflow::TensorShape;

namespace minigo {

TfDualNet::TfDualNet(const std::string& graph_path) {
  GraphDef graph_def;
  TF_CHECK_OK(ReadBinaryProto(Env::Default(), graph_path, &graph_def));
  session_.reset(NewSession(SessionOptions()));
  TF_CHECK_OK(session_->Create(graph_def));

  inputs_.clear();
  inputs_.emplace_back(
      "pos_tensor",
      Tensor(DT_FLOAT, TensorShape({1, kN, kN, kNumStoneFeatures})));

  output_names_.clear();
  output_names_.push_back("policy_output");
  output_names_.push_back("value_output");

  // Tensorflow lazily initializes the first time Session::Run is called, which
  // can take hundreds of milliseconds. This intefers with time control, so
  // explicitly run inference once during construction.
  Output output;
  BoardFeatures features;
  const auto* features_ptr = &features;
  RunMany({&features_ptr, 1}, {&output, 1});
}

TfDualNet::~TfDualNet() {
  if (session_ != nullptr) {
    session_->Close();
  }
}

void TfDualNet::RunMany(absl::Span<const BoardFeatures* const> features,
                        absl::Span<Output> outputs, Random* rnd) {
  MG_DCHECK(features.size() == outputs.size());

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
  TF_CHECK_OK(session_->Run(inputs_, output_names_, {}, &outputs_));

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
