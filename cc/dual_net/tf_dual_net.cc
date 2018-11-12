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

#include <thread>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "cc/check.h"
#include "cc/constants.h"
#include "cc/thread_safe_queue.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/stream_executor/platform.h"

using tensorflow::DT_FLOAT;
using tensorflow::Env;
using tensorflow::GraphDef;
using tensorflow::NewSession;
using tensorflow::ReadBinaryProto;
using tensorflow::SessionOptions;
using tensorflow::Tensor;
using tensorflow::TensorShape;

namespace minigo {
namespace {

class TfDualNet : public DualNet {
  class TfWorker {
   public:
    explicit TfWorker(const tensorflow::GraphDef& graph_def)
        : batch_capacity_(0) {
      tensorflow::SessionOptions options;
      options.config.mutable_gpu_options()->set_allow_growth(true);
      session_.reset(tensorflow::NewSession(options));
      TF_CHECK_OK(session_->Create(graph_def));

      output_names_.emplace_back("policy_output");
      output_names_.emplace_back("value_output");
    }

    ~TfWorker() {
      if (session_ != nullptr) {
        TF_CHECK_OK(session_->Close());
      }
    }

    void RunMany(std::vector<const BoardFeatures*> features,
                 std::vector<Output*> outputs) {
      size_t num_features = features.size();
      Reserve(num_features);

      auto* feature_data = inputs_[0].second.flat<float>().data();
      // Copy the features into the input tensor.
      for (const auto* feature : features) {
        feature_data =
            std::copy(feature->begin(), feature->end(), feature_data);
      }

      // Run the model.
      TF_CHECK_OK(session_->Run(inputs_, output_names_, {}, &outputs_));

      // Copy the policy and value out of the output tensors.
      const auto& policy_tensor = outputs_[0].flat<float>();
      const auto& value_tensor = outputs_[1].flat<float>();
      for (size_t i = 0; i < num_features; ++i) {
        memcpy(outputs[i]->policy.data(), policy_tensor.data() + i * kNumMoves,
               sizeof(outputs[i]->policy));
        outputs[i]->value = value_tensor.data()[i];
      }
    }

   private:
    void Reserve(size_t capacity) {
      MG_CHECK(capacity > 0);
      if (capacity <= batch_capacity_) {
        return;
      }
      inputs_.clear();
      inputs_.emplace_back(
          "pos_tensor", tensorflow::Tensor(tensorflow::DT_FLOAT,
                                           tensorflow::TensorShape(
                                               {static_cast<int>(capacity), kN,
                                                kN, kNumStoneFeatures})));
      batch_capacity_ = capacity;
    }

    std::unique_ptr<tensorflow::Session> session_;
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs_;
    std::vector<std::string> output_names_;
    std::vector<tensorflow::Tensor> outputs_;
    size_t batch_capacity_;
  };

  struct InferenceData {
    std::vector<const DualNet::BoardFeatures*> features;
    std::vector<DualNet::Output*> outputs;
    absl::Notification* notification;
  };

 public:
  explicit TfDualNet(std::string graph_path);

  ~TfDualNet() override;

  void RunMany(std::vector<const BoardFeatures*> features,
               std::vector<Output*> outputs, std::string* model) override;

  int GetBufferCount() const override { return worker_threads_.size(); }

 private:
  static void PlaceOnDevice(tensorflow::GraphDef* graph_def,
                            const std::string& device) {
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

  std::string graph_path_;
  ThreadSafeQueue<InferenceData> inference_queue_;
  std::vector<std::thread> worker_threads_;
  std::atomic<bool> running_;
  int device_count_;
};

TfDualNet::TfDualNet(std::string graph_path)
    : graph_path_(graph_path), running_(true) {
  GraphDef graph_def;

  // If we can't find the specified graph, try adding a .pb extension.
  auto* env = Env::Default();
  if (!env->FileExists(graph_path).ok()) {
    absl::StrAppend(&graph_path, ".pb");
  }

  TF_CHECK_OK(ReadBinaryProto(env, graph_path, &graph_def));

  auto functor = [this](const tensorflow::GraphDef& graph_def) {
    TfWorker worker(graph_def);
    while (running_) {
      InferenceData inference;
      if (inference_queue_.PopWithTimeout(&inference, absl::Seconds(1))) {
        worker.RunMany(std::move(inference.features),
                       std::move(inference.outputs));
        inference.notification->Notify();
      }
    }
  };

  // Create two worker threads per GPU.
  if (tensorflow::ValidateGPUMachineManager().ok()) {
    int device_count = tensorflow::GPUMachineManager()->VisibleDeviceCount();
    for (int device_id = 0; device_id < device_count; ++device_id) {
      auto device = std::to_string(device_id);
      PlaceOnDevice(&graph_def, "/gpu:" + device);
      worker_threads_.emplace_back(functor, graph_def);
      worker_threads_.emplace_back(functor, graph_def);
    }
    if (device_count) {
      return;
    }
  }

  // No GPUs available, use CPU instead.
  worker_threads_.emplace_back(functor, graph_def);
  worker_threads_.emplace_back(functor, graph_def);
}

TfDualNet::~TfDualNet() {
  running_ = false;
  for (auto& thread : worker_threads_) {
    thread.join();
  }
}

void TfDualNet::RunMany(std::vector<const BoardFeatures*> features,
                        std::vector<Output*> outputs, std::string* model) {
  MG_DCHECK(features.size() == outputs.size());

  absl::Notification notification;
  inference_queue_.Push(
      {std::move(features), std::move(outputs), &notification});
  notification.WaitForNotification();

  if (model != nullptr) {
    *model = graph_path_;
  }
}
}  // namespace

std::unique_ptr<DualNet> NewTfDualNet(const std::string& graph_path) {
  return absl::make_unique<TfDualNet>(graph_path);
}

}  // namespace minigo
