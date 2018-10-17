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

#include "cc/dual_net/tpu_dual_net.h"

#include <algorithm>
#include <thread>

#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
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

TpuDualNet::Worker::Worker(const tensorflow::GraphDef& graph_def,
                           const std::string& tpu_name, int num_replicas,
                           int max_batch_size)
    : num_replicas_(num_replicas),
      max_sub_batch_size_((max_batch_size + num_replicas_ - 1) /
                          num_replicas_) {
  SessionOptions options;
  options.target = tpu_name;
  options.config.set_allow_soft_placement(true);
  options.config.set_log_device_placement(true);
  session_.reset(NewSession(options));
  TF_CHECK_OK(session_->Create(graph_def));

  for (int i = 0; i < num_replicas_; ++i) {
    inputs_.emplace_back(
        absl::StrCat("tpu_pos_tensor_", i),
        Tensor(DT_FLOAT,
               TensorShape({max_sub_batch_size_, kN, kN, kNumStoneFeatures})));
    output_names_.push_back(absl::StrCat("tpu_policy_output_", i));
    output_names_.push_back(absl::StrCat("tpu_value_output_", i));
  }
}

TpuDualNet::Worker::~Worker() {
  std::cerr << "Closing session" << std::endl;
  TF_CHECK_OK(session_->Close());
}

void TpuDualNet::Worker::InitializeTpu() {
  std::cerr << "Initializing TPU" << std::endl;
  TF_CHECK_OK(session_->Run({}, {}, {"ConfigureDistributedTPU"}, nullptr));
}

void TpuDualNet::Worker::ShutdownTpu() {
  std::cerr << "Shutting down TPU" << std::endl;
  TF_CHECK_OK(session_->Run({}, {}, {"ShutdownDistributedTPU"}, nullptr));
}

void TpuDualNet::Worker::RunMany(std::vector<const BoardFeatures*> features,
                                 std::vector<Output*> outputs) {
  MG_CHECK(features.size() == outputs.size());

  auto batch_size = static_cast<int>(features.size());
  auto sub_batch_size = (batch_size + num_replicas_ - 1) / num_replicas_;
  MG_CHECK(sub_batch_size <= max_sub_batch_size_);

  // Split the input features across all replicas.
  for (int replica = 0; replica < num_replicas_; ++replica) {
    int begin = replica * sub_batch_size;
    int end = std::min(batch_size, (replica + 1) * sub_batch_size);
    auto* data = inputs_[replica].second.flat<float>().data();
    for (int i = begin; i < end; ++i) {
      data = std::copy(features[i]->begin(), features[i]->end(), data);
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
}

TpuDualNet::TpuDualNet(const std::string& graph_path,
                       const std::string& tpu_name, int buffering,
                       int max_batch_size)
    : graph_path_(graph_path) {
  MG_CHECK(buffering >= 1);

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
  GraphDef graph_def;
  TF_CHECK_OK(ReadBinaryProto(env, graph_path_, &graph_def));

  // Count the number of times the model is replicated. There should be eight,
  // one replica for each TPU core.
  int num_replicas = 0;
  for (const auto& node : graph_def.node()) {
    absl::string_view name = node.name();
    if (absl::ConsumePrefix(&name, "tpu_pos_tensor_")) {
      int replica;
      MG_CHECK(absl::SimpleAtoi(name, &replica));
      num_replicas = std::max(num_replicas, replica + 1);
    }
  }
  std::cerr << "Found " << num_replicas << " model replicas in graph "
            << graph_path << std::endl;
  MG_CHECK(num_replicas > 0);

  for (int i = 0; i < buffering; ++i) {
    workers_.Push(absl::make_unique<TpuDualNet::Worker>(
        graph_def, tpu_name, num_replicas, max_batch_size));
  }

  // Use one of the workers to initialize the TPU.
  auto worker = workers_.Pop();
  worker->InitializeTpu();
  workers_.Push(std::move(worker));

  // Run warm-up inferences on all sessions.
  // Tensorflow lazily initializes the first time Session::Run is called,
  // which can take hundreds of milliseconds. This intefers with time control,
  // so explicitly run inference once during construction.
  std::cerr << "Running warm-up inferences" << std::endl;
  std::vector<std::thread> threads;
  for (int i = 0; i < buffering; ++i) {
    threads.emplace_back([this]() {
      BoardFeatures features;
      Output output;
      RunMany({&features}, {&output}, nullptr);
    });
  }
  for (auto& t : threads) {
    t.join();
  }
}

TpuDualNet::~TpuDualNet() {
  // Use one of the workers to shutdown the TPU.
  auto worker = workers_.Pop();
  worker->ShutdownTpu();
  workers_.Push(std::move(worker));
}

void TpuDualNet::RunMany(std::vector<const BoardFeatures*> features,
                         std::vector<Output*> outputs, std::string* model) {
  auto worker = workers_.Pop();
  worker->RunMany(features, outputs);
  workers_.Push(std::move(worker));

  if (model != nullptr) {
    *model = graph_path_;
  }
}

}  // namespace minigo
