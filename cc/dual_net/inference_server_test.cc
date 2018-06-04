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
#include "absl/strings/str_cat.h"
#include "cc/constants.h"
#include "cc/dual_net/fake_net.h"
#include "cc/random.h"
#include "gmock/gmock.h"
#include "grpc++/create_channel.h"
#include "grpc/status.h"
#include "gtest/gtest.h"
#include "proto/inference_service.grpc.pb.h"

namespace minigo {
namespace {

class InferenceServerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    Random rnd;
    for (int i = 0; i < kNumMoves; ++i) {
      priors_.push_back(rnd());
    }
    value_ = 0.1;
    dual_net_ = absl::make_unique<FakeNet>(priors_, value_);

    server_ = absl::make_unique<InferenceServer>(port_);
    client_ = server_->NewDualNet();
  }

  int port_ = 50051;

  std::vector<float> priors_;
  float value_;

  std::unique_ptr<DualNet> dual_net_;
  std::unique_ptr<InferenceServer> server_;
  std::unique_ptr<DualNet> client_;
};

TEST_F(InferenceServerTest, Test) {
  // Run a fake inference worker on a separate thread.
  // Unlike the real inference worker, this fake worker doesn't loop, and
  // doesn't add any RPC ops to the TensorFlow graph. Instead, the RPCs and
  // proto marshalling is performed manually.
  std::thread thread([this]() {
    InferenceService::Stub stub(grpc::CreateChannel(
        absl::StrCat("localhost:", port_), grpc::InsecureChannelCredentials()));

    grpc::Status status;

    // Get the server config.
    GetConfigRequest get_config_request;
    GetConfigResponse get_config_response;
    {
      grpc::ClientContext context;
      status =
          stub.GetConfig(&context, get_config_request, &get_config_response);
      ASSERT_TRUE(status.ok()) << "RPC failed: " << status.error_message()
                               << ": " << status.error_details();
    }

    int board_size = get_config_response.board_size();
    int batch_size = get_config_response.batch_size();

    ASSERT_EQ(kN, board_size);
    ASSERT_LT(0, batch_size);

    // Get the features.
    GetFeaturesRequest get_features_request;
    GetFeaturesResponse get_features_response;
    {
      grpc::ClientContext context;
      status = stub.GetFeatures(&context, get_features_request,
                                &get_features_response);
      ASSERT_TRUE(status.ok()) << "RPC failed: " << status.error_message()
                               << ": " << status.error_details();
    }
    ASSERT_EQ(batch_size * DualNet::kNumBoardFeatures,
              get_features_response.features().size());

    // Run the model.
    std::vector<DualNet::BoardFeatures> features(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < DualNet::kNumBoardFeatures; ++j) {
        features[i][j] =
            get_features_response.features(i * DualNet::kNumBoardFeatures + j);
      }
    }
    std::vector<DualNet::Output> outputs(batch_size);
    std::vector<const DualNet::BoardFeatures*> feature_ptrs;
    for (const auto& f : features) {
      feature_ptrs.push_back(&f);
    }
    dual_net_->RunMany(feature_ptrs, {outputs.data(), outputs.size()});

    // Put the outputs.
    PutOutputsRequest put_outputs_request;
    PutOutputsResponse put_outputs_response;
    for (const auto& output : outputs) {
      for (int i = 0; i < kNumMoves; ++i) {
        put_outputs_request.add_policy(output.policy[i]);
      }
      put_outputs_request.add_value(output.value);
    }
    put_outputs_request.set_batch_id(get_features_response.batch_id());
    {
      grpc::ClientContext context;
      status =
          stub.PutOutputs(&context, put_outputs_request, &put_outputs_response);
      ASSERT_TRUE(status.ok()) << "RPC failed: " << status.error_message()
                               << ": " << status.error_details();
    }
  });

  DualNet::BoardFeatures features;
  auto* features_ptr = &features;
  DualNet::Output output;
  client_->RunMany({&features_ptr, 1}, {&output, 1});

  ASSERT_EQ(value_, output.value);
  for (int i = 0; i < kNumMoves; ++i) {
    ASSERT_EQ(priors_[i], output.policy[i]);
  }

  thread.join();
}

}  // namespace
}  // namespace minigo
