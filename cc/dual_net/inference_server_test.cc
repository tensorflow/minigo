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

#include "cc/dual_net/inference_server.h"

#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "cc/constants.h"
#include "cc/dual_net/fake_net.h"
#include "cc/dual_net/inference_service.grpc.pb.h"
#include "gmock/gmock.h"
#include "grpc++/create_channel.h"
#include "grpc/status.h"
#include "gtest/gtest.h"

namespace minigo {
namespace {

class InferenceServerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::vector<float> priors;
    priors.resize(kNumMoves, 0.3);
    dual_net_ = absl::make_unique<FakeNet>(priors, 0.1);
    server_ = absl::make_unique<InferenceServer>();
    client_ = server_->NewDualNet();
  }

  std::unique_ptr<DualNet> dual_net_;
  std::unique_ptr<InferenceServer> server_;
  std::unique_ptr<DualNet> client_;
};

TEST_F(InferenceServerTest, Test) {
  std::thread thread([this]() {
    InferenceService::Stub stub(grpc::CreateChannel(
        "localhost:50051", grpc::InsecureChannelCredentials()));

    grpc::Status status;

    // Get the features.
    GetFeaturesRequest get_features_request;
    GetFeaturesResponse get_features_response;
    {
      grpc::ClientContext context;
      status = stub.GetFeatures(&context, get_features_request,
                                &get_features_response);
      MG_CHECK(status.ok()) << "RPC failed: " << status.error_message() << ": "
                            << status.error_details();
    }
    MG_CHECK(get_features_response.features().size() ==
             DualNet::kNumBoardFeatures);

    // Run the model.
    DualNet::BoardFeatures features;
    auto* features_ptr = &features;
    for (int i = 0; i < DualNet::kNumBoardFeatures; ++i) {
      features[i] = get_features_response.features(i);
    }
    DualNet::Output output;
    dual_net_->RunMany({&features_ptr, 1}, {&output, 1});

    // Put the outputs.
    PutOutputsRequest put_outputs_request;
    PutOutputsResponse put_outputs_response;
    for (int i = 0; i < kNumMoves; ++i) {
      put_outputs_request.add_policy(output.policy[i]);
    }
    put_outputs_request.set_value(output.value);
    {
      grpc::ClientContext context;
      status =
          stub.PutOutputs(&context, put_outputs_request, &put_outputs_response);
      MG_CHECK(status.ok()) << "RPC failed: " << status.error_message() << ": "
                            << status.error_details();
    }
  });

  DualNet::BoardFeatures features;
  auto* features_ptr = &features;
  DualNet::Output output;
  client_->RunMany({&features_ptr, 1}, {&output, 1});

  for (int j = 0; j < kN; ++j) {
    for (int i = 0; i < kN; ++i) {
      std::cerr << "  " << output.policy[j * kN + i];
    }
    std::cerr << "\n";
  }
  std::cerr << "\n";
  std::cerr << output.value << "\n";

  thread.join();
}

}  // namespace
}  // namespace minigo
