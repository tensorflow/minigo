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

#include "absl/strings/str_cat.h"
#include "cc/check.h"
// TODO(tommadams): remove the dual_net.h include
#include "absl/strings/str_join.h"
#include "cc/dual_net/dual_net.h"
#include "cc/dual_net/inference_service.grpc.pb.h"
#include "cc/init.h"
#include "gflags/gflags.h"
#include "grpc++/grpc++.h"
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

DEFINE_string(model, "",
              "Path to a minigo model serialized as a GraphDef proto.");
DEFINE_string(address, "localhost", "Inference server addresss.");
DEFINE_int32(port, 50051, "Inference server port.");

// void GetFeatures(InferenceService::Stub* stub,
//                  std::vector<DualNet::BoardFeatures>* batch) {
//   GetFeaturesRequest request;
//   GetFeaturesResponse response;
//
//   grpc::ClientContext context;
//   auto status = stub->GetFeatures(&context, request, &response);
//   MG_CHECK(status.ok()) << "RPC failed: " << status.error_message() << ": "
//                         << status.error_details();
//
//   /*
//   MG_CHECK(response.dtype() == dual_net::tf::DT_FLOAT);
//   MG_CHECK(response.tensor_shape().dim().size() == 4);
//   MG_CHECK(response.tensor_shape().dim(1).size() == kN);
//   MG_CHECK(response.tensor_shape().dim(2).size() == kN);
//   MG_CHECK(response.tensor_shape().dim(3).size() ==
//   DualNet::kNumStoneFeatures);
//
//   batch->clear();
//   for (int j = 0; j < response.tensor_shape().dim(0).size(); ++j) {
//     batch->emplace_back();
//     auto* dst = batch->back().data();
//     for (int i = 0; i < DualNet::kNumBoardFeatures; ++i) {
//       dst[i] = response.float_val(i + j * DualNet::kNumBoardFeatures);
//     }
//   }
//   */
//
//   const auto& packed_features = response.features();
//   MG_CHECK(packed_features.size() % DualNet::kNumBoardFeatures == 0);
//
//   batch->clear();
//   int batch_size = packed_features.size() / DualNet::kNumBoardFeatures;
//   for (int j = 0; j < batch_size; ++j) {
//     batch->emplace_back();
//     auto* dst = batch->back().data();
//     for (int i = 0; i < DualNet::kNumBoardFeatures; ++i) {
//       dst[i] = packed_features[i + j * DualNet::kNumBoardFeatures];
//     }
//   }
// }

void PutOutputs(InferenceService::Stub* stub,
                const std::vector<DualNet::Output>& outputs) {
  PutOutputsRequest request;
  PutOutputsResponse response;

  for (const auto& src : outputs) {
    for (int i = 0; i < kNumMoves; ++i) {
      request.add_policy(src.policy[i]);
    }
    request.add_value(src.value);
  }

  grpc::ClientContext context;
  context.set_wait_for_ready(true);
  auto status = stub->PutOutputs(&context, request, &response);
  MG_CHECK(status.ok()) << "RPC failed: " << status.error_message() << ": "
                        << status.error_details();
}

void RunClient() {
  GraphDef graph_def;
  TF_CHECK_OK(ReadBinaryProto(Env::Default(), FLAGS_model, &graph_def));

  for (const auto& node : graph_def.node()) {
    std::cerr << node.op() << " : " << node.name() << "["
              << absl::StrJoin(node.input(), ", ") << "]" << std::endl;
  }

  std::unique_ptr<Session> session(NewSession(SessionOptions()));
  TF_CHECK_OK(session->Create(graph_def));

  std::vector<std::string> output_names;
  output_names.push_back("policy_output");
  output_names.push_back("value_output");

  std::vector<Tensor> output_tensors;

  auto channel =
      grpc::CreateChannel(absl::StrCat(FLAGS_address, ":", FLAGS_port),
                          grpc::InsecureChannelCredentials());
  auto stub = InferenceService::NewStub(channel);

  for (;;) {
    TF_CHECK_OK(session->Run({}, output_names, {}, &output_tensors));

    // Copy the policy and value out of the output tensors.
    const auto& policy_tensor = output_tensors[0].flat<float>();
    const auto& value_tensor = output_tensors[1].flat<float>();
    // TODO(tommadams): Read batch size from GraphDef.
    int batch_size = 8;
    std::vector<DualNet::Output> outputs(batch_size);
    for (int j = 0; j < batch_size; ++j) {
      const auto* policy_tensor_data = policy_tensor.data() + j * kNumMoves;
      for (int i = 0; i < kNumMoves; ++i) {
        outputs[j].policy[i] = policy_tensor_data[j];
      }
      outputs[j].value = value_tensor.data()[j];
    }

    PutOutputs(stub.get(), outputs);
  }
}

}  // namespace minigo

int main(int argc, char* argv[]) {
  minigo::Init(&argc, &argv);
  minigo::RunClient();
  return 0;
}
