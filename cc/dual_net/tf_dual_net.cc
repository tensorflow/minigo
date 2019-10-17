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

#include <algorithm>
#include <thread>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "cc/constants.h"
#include "cc/file/path.h"
#include "cc/logging.h"
#include "cc/thread_safe_queue.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"
#include "wtf/macros.h"

namespace minigo {
namespace {

void PlaceOnDevice(tensorflow::GraphDef* graph_def, const std::string& device) {
  for (auto& node : *graph_def->mutable_node()) {
    if (node.op() == "Const") {
      auto it = node.attr().find("dtype");
      if (it != node.attr().end() &&
          it->second.type() == tensorflow::DT_INT32) {
        continue;  // Const nodes of type int32 need to be in CPU.
      }
    }
    node.set_device(device);
  }
}

class TfDualNet : public Model {
 public:
  TfDualNet(const std::string& graph_path,
            const FeatureDescriptor& feature_desc,
            const tensorflow::GraphDef& graph_def, int device);
  ~TfDualNet() override;

  void RunMany(const std::vector<const ModelInput*>& inputs,
               std::vector<ModelOutput*>* outputs,
               std::string* model_name) override;

 private:
  void Reserve(int capacity);

  std::unique_ptr<tensorflow::Session> session_;
  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs_;
  std::vector<std::string> output_names_;
  std::vector<tensorflow::Tensor> outputs_;
  const std::string graph_path_;
  int batch_capacity_ = 0;
};

TfDualNet::TfDualNet(const std::string& graph_path,
                     const FeatureDescriptor& feature_desc,
                     const tensorflow::GraphDef& graph_def, int device)
    : Model(std::string(file::Stem(file::Basename(graph_path))), feature_desc),
      graph_path_(graph_path) {
  tensorflow::SessionOptions options;
  options.config.mutable_gpu_options()->set_allow_growth(true);
  if (device >= 0) {
    options.config.mutable_gpu_options()->set_visible_device_list(
        absl::StrCat(device));
  }
  session_.reset(tensorflow::NewSession(options));
  TF_CHECK_OK(session_->Create(graph_def));

  output_names_.emplace_back("policy_output");
  output_names_.emplace_back("value_output");
}

TfDualNet::~TfDualNet() {
  if (session_ != nullptr) {
    TF_CHECK_OK(session_->Close());
  }
}

void TfDualNet::RunMany(const std::vector<const ModelInput*>& inputs,
                        std::vector<ModelOutput*>* outputs,
                        std::string* model_name) {
  Reserve(inputs.size());

  WTF_SCOPE("TfDualNet::Run", size_t, int)(inputs.size(), batch_capacity_);
  MG_CHECK(inputs.size() == outputs->size());

  Tensor<float> features(batch_capacity_, kN, kN,
                         feature_descriptor().num_planes,
                         inputs_[0].second.flat<float>().data());
  {
    WTF_SCOPE("Features::Set", int)(batch_capacity_);
    feature_descriptor().set_floats(inputs, &features);
  }

  // Run the model.
  {
    WTF_SCOPE("Session::Run", int)(batch_capacity_);
    TF_CHECK_OK(session_->Run(inputs_, output_names_, {}, &outputs_));
  }

  Tensor<float> policy(batch_capacity_, 1, 1, kNumMoves,
                       outputs_[0].flat<float>().data());
  Tensor<float> value(batch_capacity_, 1, 1, 1,
                      outputs_[1].flat<float>().data());
  {
    WTF_SCOPE("Model::GetOutputs", int)(batch_capacity_);
    Model::GetOutputs(inputs, policy, value, outputs);
  }

  if (model_name != nullptr) {
    *model_name = graph_path_;
  }
}

void TfDualNet::Reserve(int capacity) {
  MG_CHECK(capacity > 0);
  if (capacity <= batch_capacity_ && capacity > 3 * batch_capacity_ / 4) {
    return;
  }
  inputs_.clear();
  inputs_.emplace_back(
      "pos_tensor",
      tensorflow::Tensor(tensorflow::DT_FLOAT,
                         {capacity, kN, kN, feature_descriptor().num_planes}));
  batch_capacity_ = capacity;
}

}  // namespace

TfDualNetFactory::TfDualNetFactory(int device) : device_(device) {}

std::unique_ptr<Model> TfDualNetFactory::NewModel(
    const std::string& descriptor) {
  tensorflow::GraphDef graph_def;
  auto* env = tensorflow::Env::Default();
  TF_CHECK_OK(tensorflow::ReadBinaryProto(env, descriptor, &graph_def));

  // Check that we're not loading a TPU model.
  for (const auto& node : graph_def.node()) {
    MG_CHECK(!absl::StartsWithIgnoreCase(node.name(), "tpu"))
        << "found node named \"" << node.name()
        << "\", this model looks like it was compiled for TPU";
  }

  // Look at the shape of the feature tensor to figure out what type of model
  // it is.
  // TODO(tommadams): We'll need something more sophisticated if we want to
  // support arbitrary combinations of features. This will do to start with
  // though.
  int num_feature_planes = 0;
  for (const auto& node : graph_def.node()) {
    if (node.name() == "pos_tensor") {
      auto it = node.attr().find("shape");
      MG_CHECK(it != node.attr().end());
      MG_CHECK(it->second.has_shape());
      MG_CHECK(it->second.shape().dim().size() == 4);
      num_feature_planes = it->second.shape().dim(3).size();
      break;
    }
  }
  MG_CHECK(num_feature_planes != 0)
      << "Couldn't determine model type from GraphDef: pos_tensor not found";

  FeatureDescriptor feature_desc;
  switch (num_feature_planes) {
    case AgzFeatures::kNumPlanes:
      feature_desc = FeatureDescriptor::Create<AgzFeatures>();
      break;
    case ExtraFeatures::kNumPlanes:
      feature_desc = FeatureDescriptor::Create<ExtraFeatures>();
      break;
    default:
      MG_LOG(FATAL) << "unrecognized number of features: "
                    << num_feature_planes;
      return nullptr;
  }

  // TODO(tommadams): support running on multiple GPUs
  if (device_ >= 0) {
    PlaceOnDevice(&graph_def, "/gpu:0");
  }
  return absl::make_unique<TfDualNet>(descriptor, feature_desc, graph_def,
                                      device_);
}

}  // namespace minigo
