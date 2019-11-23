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

#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/types/span.h"
#include "cc/constants.h"
#include "cc/file/path.h"
#include "cc/logging.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"
#include "wtf/macros.h"

namespace minigo {

namespace {

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

std::unique_ptr<tensorflow::Session> CreateSession(
    const tensorflow::GraphDef& graph_def, const std::string& tpu_name) {
  // Make sure tpu_name looks like a valid name.
  MG_CHECK(absl::StartsWith(tpu_name, "grpc://"));

  tensorflow::SessionOptions options;
  options.target = tpu_name;
  options.config.set_allow_soft_placement(true);
  options.config.set_log_device_placement(true);
  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
  TF_CHECK_OK(session->Create(graph_def));
  return session;
}

}  // namespace

TpuDualNet::TpuDualNet(const std::string& graph_path,
                       const FeatureDescriptor& feature_desc,
                       tensorflow::DataType input_type,
                       std::shared_ptr<tensorflow::Session> session,
                       int num_replicas, TpuDualNetFactory* factory)
    : Model(std::string(file::Stem(graph_path)), feature_desc),
      session_(std::move(session)),
      num_replicas_(num_replicas),
      graph_path_(graph_path),
      input_type_(input_type),
      factory_(factory) {
  tensorflow::CallableOptions callable_options;
  for (int i = 0; i < num_replicas_; ++i) {
    callable_options.add_feed(absl::StrCat("pos_tensor_", i));
    callable_options.add_fetch(absl::StrCat("policy_output_", i));
    callable_options.add_fetch(absl::StrCat("value_output_", i));
    callable_options.add_target(absl::StrCat("policy_output_", i));
    callable_options.add_target(absl::StrCat("value_output_", i));
  }

  // Timeout after 30 seconds.
  callable_options.mutable_run_options()->set_timeout_in_ms(30 * 1000);

  TF_CHECK_OK(session_->MakeCallable(callable_options, &handle_));
}

TpuDualNet::~TpuDualNet() {
  TF_CHECK_OK(session_->ReleaseCallable(handle_));
  session_.reset();
  factory_->CloseOrphanedSessions();
}

void TpuDualNet::RunMany(const std::vector<const ModelInput*>& inputs,
                         std::vector<ModelOutput*>* outputs,
                         std::string* model_name) {
  auto batch_size =
      static_cast<int>((inputs.size() + num_replicas_ - 1) / num_replicas_);
  Reserve(batch_size);

  WTF_SCOPE("TpuDualNet::Run: inputs, capacity", size_t, size_t)
  (inputs.size(), num_replicas_ * batch_capacity_);

  auto input_span = absl::MakeConstSpan(inputs);
  auto output_span = absl::MakeSpan(*outputs);

  {
    WTF_SCOPE("SetFeatures: inputs", size_t)(inputs.size());
    // Split the input features across all replicas.
    for (int replica = 0; replica < num_replicas_; ++replica) {
      int begin = replica * batch_size;
      int end = std::min<int>(inputs.size(), (replica + 1) * batch_size);
      if (end <= begin) {
        continue;
      }
      auto replica_inputs = input_span.subspan(begin, end - begin);

      // TODO(tommadams): pull this out into shared cc/dual_net/tf_utils.cc
      if (input_type_ == tensorflow::DT_FLOAT) {
        Tensor<float> features(
            {end - begin, kN, kN, feature_descriptor().num_planes},
            inputs_[replica].flat<float>().data());
        feature_descriptor().set_floats(replica_inputs, &features);
      } else {
        static_assert(sizeof(bool) == sizeof(uint8_t), "bool must be 1 byte");
        Tensor<uint8_t> features(
            {end - begin, kN, kN, feature_descriptor().num_planes},
            reinterpret_cast<uint8_t*>(inputs_[replica].flat<bool>().data()));
        feature_descriptor().set_bytes(replica_inputs, &features);
      }
    }
  }

  // Run the model.
  {
    WTF_SCOPE("Session::Run: inputs, capacity", size_t, size_t)
    (inputs.size(), num_replicas_ * batch_capacity_);
    outputs_.clear();
    TF_CHECK_OK(session_->RunCallable(handle_, inputs_, &outputs_, nullptr));
  }

  // Copy the policy and value out of the output tensors.
  {
    WTF_SCOPE("GetOutputs: outputs", size_t)(outputs_.size());
    for (int replica = 0; replica < num_replicas_; ++replica) {
      int begin = replica * batch_size;
      int end = std::min<int>(inputs.size(), (replica + 1) * batch_size);
      if (end <= begin) {
        continue;
      }
      auto replica_inputs = input_span.subspan(begin, end - begin);
      auto replica_outputs = output_span.subspan(begin, end - begin);

      const auto& policy_tensor = outputs_[replica * 2].flat<float>();
      const auto& value_tensor = outputs_[replica * 2 + 1].flat<float>();

      Tensor<float> policy({end - begin, kNumMoves}, policy_tensor.data());
      Tensor<float> value({end - begin}, value_tensor.data());
      Model::GetOutputs(replica_inputs, policy, value, replica_outputs);
    }
  }

  if (model_name != nullptr) {
    *model_name = graph_path_;
  }
}

void TpuDualNet::Reserve(size_t capacity) {
  MG_CHECK(capacity > 0);
  if (capacity <= batch_capacity_) {
    // TODO(tommadams): for now, never shrink the tensor sent for inference.
    // Resizing TPU tensors can take up to a second and we're focusing on using
    // TPUs for continuous selfplay at the moment.
    return;
  }

  // Use flattened input features because they're 35x faster to transfer to
  // the device on a v3 TPU.
  auto size =
      static_cast<int>(capacity * kN * kN * feature_descriptor().num_planes);

  inputs_.clear();
  for (int i = 0; i < num_replicas_; ++i) {
    inputs_.emplace_back(input_type_, tensorflow::TensorShape({size}));
  }
  batch_capacity_ = capacity;
}

TpuDualNetFactory::TpuDualNetFactory(std::string tpu_name)
    : tpu_name_(std::move(tpu_name)) {
  // Create a session containing ops for initializing & shutting down a TPU.
  tensorflow::GraphDef graph_def;
  ::tensorflow::protobuf::TextFormat::ParseFromString(kTpuOpsGraphDef,
                                                      &graph_def);
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

TpuDualNetFactory::LoadedModel TpuDualNetFactory::GetModel(
    const ModelDefinition& def) {
  absl::MutexLock lock(&mutex_);
  auto it = models_.find(def.path);
  if (it != models_.end()) {
    return it->second;
  }

  // Load the GraphDef.
  tensorflow::protobuf::io::CodedInputStream coded_stream(
      reinterpret_cast<const uint8_t*>(def.model_bytes.data()),
      def.model_bytes.size());
  coded_stream.SetTotalBytesLimit(1024 * 1024 * 1024);
  tensorflow::GraphDef graph_def;
  MG_CHECK(graph_def.ParseFromCodedStream(&coded_stream) &&
           coded_stream.ConsumedEntireMessage());

  // Find the data type of the input features.
  tensorflow::DataType dt;
  const auto& input_type = def.metadata.Get<std::string>("input_type");
  if (input_type == "bool") {
    dt = tensorflow::DT_BOOL;
  } else if (input_type == "float") {
    dt = tensorflow::DT_FLOAT;
  } else {
    MG_LOG(FATAL) << "Unsupported input type \"" << input_type << "\"";
  }

  tensorflow::SessionOptions options;
  options.target = tpu_name_;
  options.config.set_allow_soft_placement(true);
  options.config.set_log_device_placement(true);
  // options.config.set_intra_op_parallelism_threads(1);
  // options.config.set_inter_op_parallelism_threads(-1);

  LoadedModel model;
  model.input_type = dt;
  model.num_replicas =
      static_cast<int>(def.metadata.Get<uint64_t>("num_replicas"));
  model.session.reset(tensorflow::NewSession(options));
  model.feature_desc = FeatureDescriptor::Create(
      def.metadata.Get<std::string>("input_features"));

  TF_CHECK_OK(model.session->Create(graph_def));
  models_.emplace(def.path, model);

  return model;
}

void TpuDualNetFactory::CloseOrphanedSessions() {
  absl::MutexLock lock(&mutex_);
  std::vector<std::string> to_erase;
  for (const auto& kv : models_) {
    if (kv.second.session.use_count() == 1) {
      MG_LOG(INFO) << "Closing orphaned model session: " << kv.first;
      TF_CHECK_OK(kv.second.session->Close());
      to_erase.push_back(kv.first);
    }
  }
  for (const auto& k : to_erase) {
    models_.erase(k);
  }
}

std::unique_ptr<Model> TpuDualNetFactory::NewModel(const ModelDefinition& def) {
  MG_CHECK(def.metadata.Get<std::string>("engine") == "tpu");
  auto model = GetModel(def);
  return absl::make_unique<TpuDualNet>(def.path, model.feature_desc,
                                       model.input_type, model.session,
                                       model.num_replicas, this);
}

}  // namespace minigo
