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
//
// Standalone test code that runs the Minigo model on a Cloud TPU.
// Helpful for debugging issues.
//
// Example usage (you will need to supply your own values for tpu_name, model_a
// and model_b):
//   bazel build --define=tf=1 --define=tpu=1 cc:tpu_test
//   ./bazel-bin/cc/tpu_test \
//     --tpu_name=grpc://10.240.2.10:8470 \
//     --model_a=gs://tmadams-sandbox/tpu_cpp/000674-neptune.pb \
//     --model_b=gs://tmadams-sandbox/tpu_cpp/000001-bootstrap.pb

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "gflags/gflags.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

DEFINE_string(
    tpu_name, "",
    "Cloud TPU name to run inference on, e.g. \"grpc://10.240.2.10:8470\"");
DEFINE_string(model_a, "", "Path to first model to load");
DEFINE_string(model_b, "", "Path to second model to load");

using tensorflow::DT_FLOAT;
using tensorflow::Env;
using tensorflow::GraphDef;
using tensorflow::NewSession;
using tensorflow::ReadBinaryProto;
using tensorflow::Session;
using tensorflow::SessionOptions;
using tensorflow::Tensor;
using tensorflow::TensorShape;

constexpr int kNumReplicas = 8;
constexpr int kN = 19;
constexpr int kNumMoves = kN * kN + 1;
constexpr int kNumStoneFeatures = 17;
constexpr int kNumBoardFeatures = kN * kN * kNumStoneFeatures;

using Features = std::array<float, kNumBoardFeatures>;
using Policy = std::array<float, kNumMoves>;

// Simple wrapper around the Minigo model.
class Model {
 public:
  explicit Model(const std::string& path) : path_(path) {
    // Load model.
    auto* env = Env::Default();
    GraphDef graph_def;
    TF_CHECK_OK(ReadBinaryProto(env, path, &graph_def));

    // Create a session.
    SessionOptions options;
    options.target = FLAGS_tpu_name;
    options.config.set_allow_soft_placement(true);
    options.config.set_log_device_placement(true);
    session_.reset(NewSession(options));
    TF_CHECK_OK(session_->Create(graph_def));

    // Initialize model inputs & outputs.
    for (int i = 0; i < kNumReplicas; ++i) {
      inputs_.emplace_back(
          absl::StrCat("pos_tensor_", i),
          Tensor(DT_FLOAT, TensorShape({1, kN, kN, kNumStoneFeatures})));
      output_names_.push_back(absl::StrCat("policy_output_", i));
      output_names_.push_back(absl::StrCat("value_output_", i));
    }
  }

  ~Model() {
    Log() << "Closing session" << std::endl;
    TF_CHECK_OK(session_->Close());
  }

  void InitializeTpu() {
    Log() << "Initializing TPU" << std::endl;
    TF_CHECK_OK(session_->Run({}, {}, {"ConfigureDistributedTPU"}, nullptr));
  }

  void ShutdownTpu() {
    Log() << "Shutting down TPU" << std::endl;
    TF_CHECK_OK(session_->Run({}, {}, {"ShutdownDistributedTPU"}, nullptr));
  }

  void Run(const Features& features) {
    Log() << "Running inference" << std::endl;

    // Copy features into all input tensors.
    for (int replica = 0; replica < kNumReplicas; ++replica) {
      auto* data = inputs_[replica].second.flat<float>().data();
      memcpy(data, features.data(), sizeof(features));
    }

    // Run inference.
    TF_CHECK_OK(session_->Run(inputs_, output_names_, {}, &outputs_));

    // Copy results out of the output tensors.
    Policy policy[kNumReplicas];
    float value[kNumReplicas];
    for (int replica = 0; replica < kNumReplicas; ++replica) {
      const auto* policy_data = outputs_[replica * 2].flat<float>().data();
      const auto* value_data = outputs_[replica * 2 + 1].flat<float>().data();
      memcpy(policy[replica].data(), policy_data, sizeof(Policy));
      value[replica] = *value_data;
    }

    // Check the outputs from all replicas are the same.
    for (int replica = 1; replica < kNumReplicas; ++replica) {
      for (int i = 0; i < kNumMoves; ++i) {
        if (policy[0][i] != policy[replica][i]) {
          Log() << absl::StreamFormat("policy[0][%d] == %f\n",
                                      i, policy[0][i]);
          Log() << absl::StreamFormat("policy[%d][%d] == %f\n",
                                      replica, i, policy[replica][i]);
          LOG(FATAL) << ":(";
        }
        if (value[0] != value[replica]) {
          Log() << absl::StreamFormat("value[0] == %f\n", value[0]);
          Log() << absl::StreamFormat("value[%d] == %f\n",
                                      replica, value[replica]);
          LOG(FATAL) << ":(";
        }
      }
    }

    // Log the output of the first replica (since the other replica outputs
    // all match the first).
    for (int i = 0; i < kNumMoves; ++i) {
      if (i != 0 && (i % kN) == 0) {
        std::cerr << "\n";
      }
      std::cerr << absl::StrFormat(" %0.2f", policy[0][i]);
    }
    std::cerr << "\n" << value[0] << "\n";
  }

 private:
  std::ostream& Log() {
    return std::cerr << "(" << path_ << ") ";
  }

  std::string path_;
  std::unique_ptr<Session> session_;
  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs_;
  std::vector<tensorflow::Tensor> outputs_;
  std::vector<std::string> output_names_;
};

void SimpleTest() {
  // Initialize some features that represent an empty board.
  Features features;
  int i = 0;
  for (int pos = 0; pos < kN * kN; ++pos) {
    for (int f = 0; f < kNumStoneFeatures; ++f) {
      features[i++] = (f == kNumStoneFeatures - 1) ? 1 : 0;
    }
  }

  Model model_a(FLAGS_model_a);
  Model model_b(FLAGS_model_b);

  // -----------------------------------

  // This works as expected: the outputs of model_a.Run() and model_b.Run()
  // are different.
  // model_a.InitializeTpu();
  // model_a.Run(features);
  // model_a.ShutdownTpu();
  //
  // model_b.InitializeTpu();
  // model_b.Run(features);
  // model_b.ShutdownTpu();

  // -----------------------------------

  // This does not work.
  model_a.InitializeTpu();
  model_a.Run(features);

  model_b.InitializeTpu();
  model_b.Run(features);

  // This call produces the output from model_b.
  model_a.Run(features);

  model_a.ShutdownTpu();
  // Calling model_b.ShutdownTpu() here crashes because the TPU is already
  // shutdown.

  // -----------------------------------
}

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  SimpleTest();

  return 0;
}
