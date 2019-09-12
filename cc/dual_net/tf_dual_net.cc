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
#include "absl/strings/str_join.h"
#include "absl/synchronization/notification.h"
#include "cc/constants.h"
#include "cc/file/path.h"
#include "cc/logging.h"
#include "cc/model/buffered_model.h"
#include "cc/thread_safe_queue.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"
#include "wtf/macros.h"

#if MINIGO_ENABLE_GPU
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/stream_executor/platform.h"
#endif

using tensorflow::DT_FLOAT;
using tensorflow::DT_INT32;
using tensorflow::Env;
using tensorflow::GraphDef;
using tensorflow::NewSession;
using tensorflow::ReadBinaryProto;
using tensorflow::Session;
using tensorflow::SessionOptions;
using tensorflow::Tensor;
using tensorflow::TensorShape;

namespace minigo {
namespace {

void PlaceOnDevice(GraphDef* graph_def, const std::string& device) {
  for (auto& node : *graph_def->mutable_node()) {
    if (node.op() == "Const") {
      auto it = node.attr().find("dtype");
      if (it != node.attr().end() && it->second.type() == DT_INT32) {
        continue;  // Const nodes of type int32 need to be in CPU.
      }
    }
    node.set_device(device);
  }
}

class TfDualNet : public DualNet {
 public:
  TfDualNet(const std::string& graph_path, const GraphDef& graph_def,
            const std::vector<int>& devices);
  ~TfDualNet() override;

 private:
  void RunManyImpl(std::string* model_name) override;
  void Reserve(size_t capacity);

  std::unique_ptr<Session> session_;
  std::vector<std::pair<std::string, Tensor>> inputs_;
  std::vector<std::string> output_names_;
  std::vector<Tensor> outputs_;
  size_t batch_capacity_ = 0;
  const std::string graph_path_;
};

TfDualNet::TfDualNet(const std::string& graph_path, const GraphDef& graph_def,
                     const std::vector<int>& devices)
    : DualNet(std::string(file::Stem(graph_path))), graph_path_(graph_path) {
  SessionOptions options;
  options.config.mutable_gpu_options()->set_allow_growth(true);
  if (!devices.empty()) {
    options.config.mutable_gpu_options()->set_visible_device_list(
        absl::StrJoin(devices, ","));
  }
  session_.reset(NewSession(options));
  TF_CHECK_OK(session_->Create(graph_def));

  output_names_.emplace_back("policy_output");
  output_names_.emplace_back("value_output");
}

TfDualNet::~TfDualNet() {
  if (session_ != nullptr) {
    TF_CHECK_OK(session_->Close());
  }
}

void TfDualNet::RunManyImpl(std::string* model_name) {
  size_t num_features = features_.size();
  Reserve(num_features);

  auto* feature_data = inputs_[0].second.flat<float>().data();
  // Copy the features into the input tensor.
  for (const auto& feature : features_) {
    feature_data = std::copy(feature.begin(), feature.end(), feature_data);
  }

  // Run the model.
  {
    WTF_SCOPE("Session::Run", size_t)(batch_capacity_);
    TF_CHECK_OK(session_->Run(inputs_, output_names_, {}, &outputs_));
  }

  // Copy the policy and value out of the output tensors.
  const auto& policy_tensor = outputs_[0].flat<float>();
  const auto& value_tensor = outputs_[1].flat<float>();
  for (size_t i = 0; i < num_features; ++i) {
    auto& output = raw_outputs_[i];
    memcpy(output.policy.data(), policy_tensor.data() + i * kNumMoves,
           sizeof(output.policy));
    output.value = value_tensor.data()[i];
  }

  if (model_name != nullptr) {
    *model_name = graph_path_;
  }
}

void TfDualNet::Reserve(size_t capacity) {
  MG_CHECK(capacity > 0);
  if (capacity <= batch_capacity_ && capacity > 3 * batch_capacity_ / 4) {
    return;
  }
  inputs_.clear();
  inputs_.emplace_back(
      "pos_tensor", Tensor(DT_FLOAT, TensorShape({static_cast<int>(capacity),
                                                  kN, kN, kNumStoneFeatures})));
  batch_capacity_ = capacity;
}

}  // namespace

TfDualNetFactory::TfDualNetFactory(std::vector<int> devices)
    : devices_(std::move(devices)) {
#if MINIGO_ENABLE_GPU
  if (devices_.empty() && tensorflow::ValidateGPUMachineManager().ok()) {
    int num_devices = tensorflow::GPUMachineManager()->VisibleDeviceCount();
    for (int i = 0; i < num_devices; ++i) {
      devices_.push_back(i);
    }
  }
#endif
}

std::unique_ptr<Model> TfDualNetFactory::NewModel(
    const std::string& descriptor) {
  GraphDef graph_def;
  auto* env = Env::Default();
  TF_CHECK_OK(ReadBinaryProto(env, descriptor, &graph_def));

  // Check that we're not loading a TPU model.
  for (const auto& node : graph_def.node()) {
    MG_CHECK(!absl::StartsWithIgnoreCase(node.name(), "tpu"))
        << "found node named \"" << node.name()
        << "\", this model looks like it was compiled for TPU";
  }

  std::vector<std::unique_ptr<Model>> models;
  // Create two worker models per device (or two threads for CPU inference if
  // there are no accelerator devices present).
  for (size_t i = 0; i < std::max<size_t>(devices_.size(), 1); ++i) {
    if (!devices_.empty()) {
      PlaceOnDevice(&graph_def, absl::StrCat("/gpu:", i));
    }
    models.push_back(
        absl::make_unique<TfDualNet>(descriptor, graph_def, devices_));
    models.push_back(
        absl::make_unique<TfDualNet>(descriptor, graph_def, devices_));
  }

  return absl::make_unique<BufferedModel>(std::move(models));
}

}  // namespace minigo
