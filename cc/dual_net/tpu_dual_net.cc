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
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "cc/constants.h"
#include "cc/logging.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"

using tensorflow::DT_FLOAT;
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
// Use double buffering: one running the current set of batches, the other
// filling up the next set of batches.
constexpr int kBufferCount = 2;

// A GraphDef containing the ops required to initialize and shutdown a TPU.
// This proto was generated from the script oneoffs/generate_tpu_graph_def.py.
constexpr auto kTpuOpsGraphDef = R"(
node {
  name: "ConfigureDistributedTPU"
  op: "ConfigureDistributedTPU"
  device: "/device:TPU_SYSTEM:0"
  attr {
    key: "embedding_config"
    value {
      s: ""
    }
  }
  attr {
    key: "is_global_init"
    value {
      b: false
    }
  }
  attr {
    key: "tpu_embedding_config"
    value {
      s: ""
    }
  }
}
node {
  name: "ShutdownDistributedTPU"
  op: "ShutdownDistributedTPU"
  device: "/device:TPU_SYSTEM:0"
}
library {
}
)";

std::unique_ptr<Session> CreateSession(const GraphDef& graph_def,
                                       const std::string& tpu_name) {
  SessionOptions options;
  options.target = tpu_name;
  options.config.set_allow_soft_placement(true);
  options.config.set_log_device_placement(true);
  std::unique_ptr<Session> session(NewSession(options));
  TF_CHECK_OK(session->Create(graph_def));
  return session;
}

}  // namespace

TpuDualNet::Worker::Worker(const tensorflow::GraphDef& graph_def,
                           const std::string& tpu_name, int num_replicas)
    : num_replicas_(num_replicas), batch_capacity_(0) {
  session_ = CreateSession(graph_def, tpu_name);
  for (int i = 0; i < num_replicas_; ++i) {
    output_names_.push_back(absl::StrCat("policy_output_", i));
    output_names_.push_back(absl::StrCat("value_output_", i));
  }
}

TpuDualNet::Worker::~Worker() {
  MG_LOG(INFO) << "Closing worker session";
  TF_CHECK_OK(session_->Close());
}

void TpuDualNet::Worker::RunMany(std::vector<const BoardFeatures*> features,
                                 std::vector<Output*> outputs) {
  MG_CHECK(features.size() == outputs.size());

  size_t num_features = features.size();
  size_t batch_size = (num_features + num_replicas_ - 1) / num_replicas_;
  Reserve(batch_size);

  // Split the input features across all replicas.
  for (int replica = 0; replica < num_replicas_; ++replica) {
    size_t begin = replica * batch_size;
    size_t end = std::min(num_features, (replica + 1) * batch_size);
    auto* data = inputs_[replica].second.flat<float>().data();
    for (size_t i = begin; i < end; ++i) {
      data = std::copy(features[i]->begin(), features[i]->end(), data);
    }
  }

  // Run the model.
  TF_CHECK_OK(session_->Run(inputs_, output_names_, {}, &outputs_));

  // Copy the policy and value out of the output tensors.
  for (size_t i = 0; i < num_features; ++i) {
    size_t replica = i / batch_size;
    size_t j = i % batch_size;

    const auto& policy_tensor = outputs_[replica * 2].flat<float>();
    const auto& value_tensor = outputs_[replica * 2 + 1].flat<float>();
    memcpy(outputs[i]->policy.data(), policy_tensor.data() + j * kNumMoves,
           sizeof(outputs[i]->policy));
    outputs[i]->value = value_tensor.data()[j];
  }
}

void TpuDualNet::Worker::Reserve(size_t capacity) {
  MG_CHECK(capacity > 0);
  if (capacity <= batch_capacity_) {
    return;
  }

  inputs_.clear();
  for (int i = 0; i < num_replicas_; ++i) {
    inputs_.emplace_back(
        absl::StrCat("pos_tensor_", i),
        Tensor(DT_FLOAT, TensorShape({static_cast<int>(capacity), kN, kN,
                                      kNumStoneFeatures})));
  }
  batch_capacity_ = capacity;
}

TpuDualNet::TpuDualNet(const std::string& tpu_name,
                       const std::string& graph_path)
    : graph_path_(graph_path) {
  // Make sure tpu_name looks like a valid name.
  MG_CHECK(absl::StartsWith(tpu_name, "grpc://"));

  auto* env = Env::Default();
  GraphDef graph_def;
  TF_CHECK_OK(ReadBinaryProto(env, graph_path_, &graph_def));

  // Check that we're actually loading a TPU model.
  bool found_tpu_op = false;
  for (const auto& node : graph_def.node()) {
    if (absl::StartsWithIgnoreCase(node.name(), "tpu")) {
      found_tpu_op = true;
      break;
    }
  }
  MG_CHECK(found_tpu_op) << "didn't find any ops starting with \"tpu\" this "
                            "model looks like it wasn't compiled for TPU";

  // Count the number of times the model is replicated. There should be eight,
  // one replica for each TPU core.
  int num_replicas = 0;
  for (const auto& node : graph_def.node()) {
    absl::string_view name = node.name();
    if (absl::ConsumePrefix(&name, "pos_tensor_")) {
      int replica;
      MG_CHECK(absl::SimpleAtoi(name, &replica));
      num_replicas = std::max(num_replicas, replica + 1);
    }
  }
  MG_LOG(INFO) << "Found " << num_replicas << " model replicas in graph "
               << graph_path;
  MG_CHECK(num_replicas > 0);

  for (int i = 0; i < kBufferCount; ++i) {
    workers_.Push(absl::make_unique<TpuDualNet::Worker>(graph_def, tpu_name,
                                                        num_replicas));
  }

  // Run warm-up inferences on all sessions.
  // Tensorflow lazily initializes the first time Session::Run is called,
  // which can take hundreds of milliseconds. This interfers with time control,
  // so explicitly run inference once during construction.
  MG_LOG(INFO) << "Running warm-up inferences";
  std::vector<std::thread> threads;
  for (int i = 0; i < kBufferCount; ++i) {
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

TpuDualNet::~TpuDualNet() = default;

void TpuDualNet::RunMany(std::vector<const BoardFeatures*> features,
                         std::vector<Output*> outputs, std::string* model) {
  auto worker = workers_.Pop();
  worker->RunMany(features, outputs);
  workers_.Push(std::move(worker));

  if (model != nullptr) {
    *model = graph_path_;
  }
}

TpuDualNetFactory::TpuDualNetFactory(std::string tpu_name)
    : tpu_name_(std::move(tpu_name)) {
  // Create a session containing ops for initializing & shutting down a TPU.
  GraphDef graph_def;
  ::tensorflow::protobuf::TextFormat::ParseFromString(
      kTpuOpsGraphDef, &graph_def);
  main_session_ = CreateSession(graph_def, tpu_name_);

  MG_LOG(INFO) << "Initializing TPU " << tpu_name_;
  TF_CHECK_OK(main_session_->Run({}, {}, {"ConfigureDistributedTPU"}, nullptr));
}

TpuDualNetFactory::~TpuDualNetFactory() {
  MG_LOG(INFO) << "Shutting down TPU " << tpu_name_;
  TF_CHECK_OK(main_session_->Run({}, {}, {"ShutdownDistributedTPU"}, nullptr));

  MG_LOG(INFO) << "Closing main session";
  TF_CHECK_OK(main_session_->Close());
}

int TpuDualNetFactory::GetBufferCount() const { return kBufferCount; }

std::unique_ptr<DualNet> TpuDualNetFactory::NewDualNet(
    const std::string& model) {
  return absl::make_unique<TpuDualNet>(tpu_name_, model);
}

}  // namespace minigo
