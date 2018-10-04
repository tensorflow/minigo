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

#include "absl/memory/memory.h"
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

TfDualNet::TfDualNet(std::string graph_path) : graph_path_(graph_path) {
  GraphDef graph_def;

  // If we can't find the specified graph, try adding a .pb extension.
  auto* env = Env::Default();
  if (!env->FileExists(graph_path).ok()) {
    graph_path = absl::StrCat(graph_path, ".pb");
  }

  TF_CHECK_OK(ReadBinaryProto(env, graph_path, &graph_def));

  SessionOptions options;
  options.config.mutable_gpu_options()->set_allow_growth(true);
  session_.reset(NewSession(options));
  TF_CHECK_OK(session_->Create(graph_def));

  inputs_.emplace_back("pos_tensor",
                       Tensor(DT_FLOAT, TensorShape({FLAGS_batch_size, kN, kN,
                                                     kNumStoneFeatures})));

  output_names_.emplace_back("policy_output");
  output_names_.emplace_back("value_output");
}

TfDualNet::~TfDualNet() {
  if (session_ != nullptr) {
    TF_CHECK_OK(session_->Close());
  }
}

void TfDualNet::RunMany(std::vector<const BoardFeatures*> features,
                        std::vector<Output*> outputs, std::string* model) {
  MG_DCHECK(features.size() == outputs.size());

  int batch_size = static_cast<int>(features.size());
  auto* feature_tensor = inputs_.front().second.flat<float>().data();
  // Copy the features into the input tensor.
  for (const auto* feature : features) {
    feature_tensor =
        std::copy(feature->begin(), feature->end(), feature_tensor);
  }

  // Run the model.
  TF_CHECK_OK(session_->Run(inputs_, output_names_, {}, &outputs_));

  // Copy the policy and value out of the output tensors.
  const auto& policy_tensor = outputs_[0].flat<float>();
  const auto& value_tensor = outputs_[1].flat<float>();
  for (int i = 0; i < batch_size; ++i) {
    memcpy(outputs[i]->policy.data(), policy_tensor.data() + i * kNumMoves,
           sizeof(outputs[i]->policy));
    outputs[i]->value = value_tensor.data()[i];
  }

  if (model != nullptr) {
    *model = graph_path_;
  }
}

std::unique_ptr<DualNet> NewTfDualNet(const std::string& graph_path) {
  return absl::make_unique<TfDualNet>(graph_path);
}

}  // namespace minigo
