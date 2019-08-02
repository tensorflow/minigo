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

#include "cc/dual_net/trt_dual_net.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "cc/constants.h"
#include "cc/file/path.h"
#include "cc/logging.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

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

class TrtDualNet : public DualNet {
 public:
  TrtDualNet(std::string graph_path, size_t batch_size);
  ~TrtDualNet() override;

  void RunMany(std::vector<const BoardFeatures*> features,
               std::vector<Output*> outputs, std::string* model) override;

 private:
  const std::string graph_path_;
  const size_t batch_size_;
  std::unique_ptr<Session> session_;
  std::vector<std::pair<std::string, Tensor>> inputs_;
  std::vector<std::string> output_names_;
  std::vector<Tensor> outputs_;
};

TrtDualNet::TrtDualNet(std::string graph_path, size_t batch_size)
    : DualNet(std::string(file::Stem(graph_path))),
      graph_path_(graph_path),
      batch_size_(batch_size) {
  GraphDef graph_def;

  auto* env = Env::Default();
  TF_CHECK_OK(ReadBinaryProto(env, graph_path, &graph_def));

  SessionOptions options;
  options.config.mutable_gpu_options()->set_allow_growth(true);
  session_.reset(NewSession(options));
  TF_CHECK_OK(session_->Create(graph_def));

  output_names_.emplace_back("policy_output");
  output_names_.emplace_back("value_output");

  inputs_.emplace_back(
      "pos_tensor", Tensor(DT_FLOAT, TensorShape({static_cast<int>(batch_size_),
                                                  kN, kN, kNumStoneFeatures})));
}

TrtDualNet::~TrtDualNet() {
  if (session_ != nullptr) {
    TF_CHECK_OK(session_->Close());
  }
}

void TrtDualNet::RunMany(std::vector<const BoardFeatures*> features,
                         std::vector<Output*> outputs, std::string* model) {
  MG_CHECK(features.size() <= batch_size_);

  auto* feature_data = inputs_[0].second.flat<float>().data();
  for (const auto* feature : features) {
    feature_data = std::copy(feature->begin(), feature->end(), feature_data);
  }

  // Input should already be ready here
  // Run the model.
  TF_CHECK_OK(session_->Run(inputs_, output_names_, {}, &outputs_));

  // Copy the policy and value out of the output tensors.
  // TODO(tommadams): figure out if there's a way to avoid this copy by having
  // the client code read the contents of the output tensors directly. This is
  // complicated by multi-threading & batching.
  const auto& policy_tensor = outputs_[0].flat<float>();
  const auto& value_tensor = outputs_[1].flat<float>();

  for (size_t i = 0; i < features.size(); ++i) {
    memcpy(outputs[i]->policy.data(), policy_tensor.data() + i * kNumMoves,
           sizeof(outputs[i]->policy));
    outputs[i]->value = value_tensor.data()[i];
  }

  if (model != nullptr) {
    *model = graph_path_;
  }
}
}  // namespace

TrtDualNetFactory::TrtDualNetFactory(size_t batch_size)
    : batch_size_(batch_size) {}

int TrtDualNetFactory::GetBufferCount() const {
  // TODO(tommamdams): support multiple GPUs.
  return 1;
}

std::unique_ptr<DualNet> TrtDualNetFactory::NewDualNet(
    const std::string& model) {
  return absl::make_unique<TrtDualNet>(model, batch_size_);
}

}  // namespace minigo
