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
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/public/session.h"
#include "wtf/macros.h"

namespace minigo {
namespace {

void PlaceOnDevice(tensorflow::GraphDef* graph_def, const std::string& device) {
  for (auto& node : *graph_def->mutable_node()) {
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
  tensorflow::Session::CallableHandle handle_;
  std::vector<tensorflow::Tensor> inputs_;
  std::vector<tensorflow::Tensor> outputs_;
  const std::string graph_path_;
  int batch_capacity_ = 0;
  tensorflow::DataType input_type_ = tensorflow::DT_INVALID;
};

TfDualNet::TfDualNet(const std::string& graph_path,
                     const FeatureDescriptor& feature_desc,
                     const tensorflow::GraphDef& graph_def, int device)
    : Model(std::string(file::Stem(file::Basename(graph_path))), feature_desc),
      graph_path_(graph_path) {
  tensorflow::SessionOptions session_options;
  session_options.config.mutable_gpu_options()->set_allow_growth(true);
  if (device >= 0) {
    session_options.config.mutable_gpu_options()->set_visible_device_list(
        absl::StrCat(device));
  }

  // session_options.config.set_inter_op_parallelism_threads(1);
  // auto* thread_pool_options =
  //     session_options.config.add_session_inter_op_thread_pool();
  // thread_pool_options->set_num_threads(1);
  // thread_pool_options->set_global_name("TfDualNet");

  session_.reset(tensorflow::NewSession(session_options));
  TF_CHECK_OK(session_->Create(graph_def));

  tensorflow::CallableOptions callable_options;
  callable_options.add_feed("pos_tensor");
  callable_options.add_fetch("policy_output");
  callable_options.add_fetch("value_output");
  callable_options.add_target("policy_output");
  callable_options.add_target("value_output");

  // Timeout after 30 seconds.
  callable_options.mutable_run_options()->set_timeout_in_ms(30 * 1000);

  TF_CHECK_OK(session_->MakeCallable(callable_options, &handle_));

  for (const auto& node : graph_def.node()) {
    if (node.name() == "pos_tensor") {
      auto it = node.attr().find("dtype");
      MG_CHECK(it != node.attr().end());
      input_type_ = it->second.type();
      break;
    }
  }
  const auto* desc =
      google::protobuf::GetEnumDescriptor<tensorflow::DataType>();
  const auto* value = desc->FindValueByNumber(input_type_);
  MG_CHECK(value != nullptr);
  MG_LOG(INFO) << "Model " << graph_path_ << " has input type "
               << value->name();
  MG_CHECK(input_type_ == tensorflow::DT_FLOAT ||
           input_type_ == tensorflow::DT_BOOL)
      << input_type_;
}

TfDualNet::~TfDualNet() {
  if (session_ != nullptr) {
    TF_CHECK_OK(session_->ReleaseCallable(handle_));
    TF_CHECK_OK(session_->Close());
  }
}

void TfDualNet::RunMany(const std::vector<const ModelInput*>& inputs,
                        std::vector<ModelOutput*>* outputs,
                        std::string* model_name) {
  Reserve(inputs.size());

  WTF_SCOPE("TfDualNet::Run: inputs, capacity", size_t, int)
  (inputs.size(), batch_capacity_);
  MG_CHECK(inputs.size() == outputs->size());

  if (input_type_ == tensorflow::DT_FLOAT) {
    WTF_SCOPE("Features::SetFloat: inputs", int)(inputs.size());
    Tensor<float> features(
        {batch_capacity_, kN, kN, feature_descriptor().num_planes},
        inputs_[0].flat<float>().data());
    feature_descriptor().set_floats(inputs, &features);
  } else {
    WTF_SCOPE("Features::SetBool: inputs", size_t)(inputs.size());
    static_assert(sizeof(bool) == sizeof(uint8_t), "bool must be 1 byte");
    Tensor<uint8_t> features(
        {batch_capacity_, kN, kN, feature_descriptor().num_planes},
        reinterpret_cast<uint8_t*>(inputs_[0].flat<bool>().data()));
    feature_descriptor().set_bytes(inputs, &features);
  }

  // Run the model.
  {
    WTF_SCOPE("Session::Run: capacity", int)(batch_capacity_);
    outputs_.clear();
    TF_CHECK_OK(session_->RunCallable(handle_, inputs_, &outputs_, nullptr));
  }

  Tensor<float> policy({batch_capacity_, kNumMoves},
                       outputs_[0].flat<float>().data());
  Tensor<float> value({batch_capacity_}, outputs_[1].flat<float>().data());
  {
    WTF_SCOPE("Model::GetOutputs: outputs", size_t)(outputs->size());
    Model::GetOutputs(inputs, policy, value, absl::MakeSpan(*outputs));
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

  // pos_tensor
  inputs_.emplace_back(
      input_type_, tensorflow::TensorShape(
                       {capacity, kN, kN, feature_descriptor().num_planes}));

  batch_capacity_ = capacity;
}

}  // namespace

TfDualNetFactory::TfDualNetFactory(int device) : device_(device) {}

std::unique_ptr<Model> TfDualNetFactory::NewModel(
    const std::string& descriptor) {
  tensorflow::GraphDef graph_def;
  auto* env = tensorflow::Env::Default();
  TF_CHECK_OK(env->FileExists(descriptor));
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
