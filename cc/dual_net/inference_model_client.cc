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
#include "cc/dual_net/inference_service.grpc.pb.h"
#include "cc/dual_net/tf_dual_net.h"
#include "cc/init.h"
#include "gflags/gflags.h"
#include "grpc++/grpc++.h"

namespace minigo {

DEFINE_string(model, "",
              "Path to a minigo model serialized as a GraphDef proto.");
DEFINE_string(address, "localhost", "Inference server addresss.");
DEFINE_int32(port, 50051, "Inference server port.");

void GetFeatures(InferenceService::Stub* stub,
                 DualNet::BoardFeatures* features) {
  GetFeaturesRequest get_features_request;
  GetFeaturesResponse get_features_response;

  grpc::ClientContext context;
  auto status =
      stub->GetFeatures(&context, get_features_request, &get_features_response);
  MG_CHECK(status.ok()) << "RPC failed: " << status.error_message() << ": "
                        << status.error_details();
  MG_CHECK(get_features_response.features().size() ==
           DualNet::kNumBoardFeatures);

  for (int i = 0; i < DualNet::kNumBoardFeatures; ++i) {
    (*features)[i] = get_features_response.features(i);
  }
}

void PutOutputs(InferenceService::Stub* stub, const DualNet::Output& output) {
  PutOutputsRequest put_outputs_request;
  PutOutputsResponse put_outputs_response;

  for (int i = 0; i < kNumMoves; ++i) {
    put_outputs_request.add_policy(output.policy[i]);
  }
  put_outputs_request.set_value(output.value);

  grpc::ClientContext context;
  auto status =
      stub->PutOutputs(&context, put_outputs_request, &put_outputs_response);
  MG_CHECK(status.ok()) << "RPC failed: " << status.error_message() << ": "
                        << status.error_details();
}

void RunClient() {
  auto stub = InferenceService::NewStub(
      grpc::CreateChannel(absl::StrCat(FLAGS_address, ":", FLAGS_port),
                          grpc::InsecureChannelCredentials()));

  TfDualNet dual_net(FLAGS_model);

  DualNet::BoardFeatures features;
  DualNet::Output output;
  auto* features_ptr = &features;
  for (;;) {
    GetFeatures(stub.get(), &features);
    dual_net.RunMany({&features_ptr, 1}, {&output, 1});
    PutOutputs(stub.get(), output);
  }
}

}  // namespace minigo

int main(int argc, char* argv[]) {
  minigo::Init(&argc, &argv);
  minigo::RunClient();
  return 0;
}
