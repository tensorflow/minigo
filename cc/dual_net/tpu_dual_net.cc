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

#include <algorithm>

#include "cc/dual_net/tpu_dual_net.h"

#include "absl/strings/str_cat.h"
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

TpuDualNet::TpuDualNet(const std::string& graph_path,
                       const std::string& tpu_name)
    : graph_path_(graph_path) {
  GraphDef graph_def;

  // If we can't find the specified graph, try adding a .pb extension.
  auto* env = Env::Default();
  if (!env->FileExists(graph_path_).ok()) {
    auto alt_path = absl::StrCat(graph_path_, ".pb");
    if (env->FileExists(alt_path).ok()) {
      std::cerr << graph_path << " doesn't exist, using " << alt_path
                << std::endl;
      graph_path_ = alt_path;
    }
  }

  TF_CHECK_OK(ReadBinaryProto(env, graph_path_, &graph_def));

  SessionOptions options;
  options.target = tpu_name;
  options.config.set_allow_soft_placement(true);
  options.config.set_log_device_placement(true);
  session_.reset(NewSession(options));
  TF_CHECK_OK(session_->Create(graph_def));

  std::cerr << "Initializing TPU" << std::endl;
  TF_CHECK_OK(session_->Run({}, {}, {"ConfigureDistributedTPU"}, nullptr));

  // TODO(tommadams): automatically figure out the number of replicas from the
  // GraphDef.
  num_replicas_ = 8;
  for (int i = 0; i < num_replicas_; ++i) {
    inputs_.emplace_back(
        absl::StrCat("pos_tensor_", i),
        Tensor(DT_FLOAT, TensorShape({1, kN, kN, kNumStoneFeatures})));
    output_names_.push_back(absl::StrCat("policy_output_", i));
    output_names_.push_back(absl::StrCat("value_output_", i));
  }

  // Tensorflow lazily initializes the first time Session::Run is called, which
  // can take hundreds of milliseconds. This intefers with time control, so
  // explicitly run inference once during construction.
  BoardFeatures features;
  Output output;
  RunMany({&features}, {&output}, nullptr);
}

TpuDualNet::~TpuDualNet() {
  if (session_ != nullptr) {
    std::cerr << "Shutting down TPU" << std::endl;
    TF_CHECK_OK(session_->Run({}, {}, {"ShutdownDistributedTPU"}, nullptr));
    std::cerr << "Closing session" << std::endl;
    TF_CHECK_OK(session_->Close());
  }
}

void TpuDualNet::RunMany(std::vector<const BoardFeatures*> features,
                         std::vector<Output*> outputs, std::string* model) {
  MG_DCHECK(features.size() == outputs.size());

  auto batch_size = static_cast<int>(features.size());
  auto sub_batch_size =
      std::max((batch_size + num_replicas_ - 1) / num_replicas_, 1);

  // Split the input features across all replicas.
  for (int replica = 0; replica < num_replicas_; ++replica) {
    auto& feature_tensor = inputs_[replica].second;

    // All input tensors must have the same batch size.
    if (feature_tensor.dim_size(0) != sub_batch_size) {
      feature_tensor = Tensor(
          DT_FLOAT, TensorShape({sub_batch_size, kN, kN, kNumStoneFeatures}));
    }

    int begin = replica * sub_batch_size;
    int end = std::min(batch_size, (replica + 1) * sub_batch_size);
    auto* feature_data = feature_tensor.flat<float>().data();
    for (int i = begin; i < end; ++i) {
      feature_data =
          std::copy(features[i]->begin(), features[i]->end(), feature_data);
    }
  }

  // Run the model.
  TF_CHECK_OK(session_->Run(inputs_, output_names_, {}, &outputs_));

  // Copy the policy and value out of the output tensors.
  for (int i = 0; i < batch_size; ++i) {
    int replica = i / sub_batch_size;
    int j = i % sub_batch_size;

    const auto& policy_tensor = outputs_[replica * 2].flat<float>();
    const auto& value_tensor = outputs_[replica * 2 + 1].flat<float>();
    memcpy(outputs[i]->policy.data(), policy_tensor.data() + j * kNumMoves,
           sizeof(outputs[i]->policy));
    outputs[i]->value = value_tensor.data()[j];
  }

  if (model != nullptr) {
    *model = graph_path_;
  }
}

}  // namespace minigo
