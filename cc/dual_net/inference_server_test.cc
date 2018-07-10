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

    server_ = absl::make_unique<InferenceServer>(virtual_losses_, games_per_inference_, port_);
    for (int i = 0; i < games_per_inference_; ++i) {
      clients_.push_back(server_->NewDualNet());
    }
  }

  int port_ = 50051;
  int virtual_losses_ = 8;
  int games_per_inference_ = 2;

  std::vector<float> priors_;
  float value_;

  std::unique_ptr<DualNet> dual_net_;
  std::unique_ptr<InferenceServer> server_;
  std::vector<std::unique_ptr<DualNet>> clients_;
};

TEST_F(InferenceServerTest, Test) {
  // Run a fake inference worker on a separate thread.
  // Unlike the real inference worker, this fake worker doesn't loop, and
  // doesn't add any RPC ops to the TensorFlow graph. Instead, the RPCs and
  // proto marshalling is performed manually.
  std::thread server_thread([this]() {
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
    int vlosses = get_config_response.virtual_losses();
    int games_per_inference = get_config_response.games_per_inference();

    ASSERT_EQ(kN, board_size);
    ASSERT_LT(0, vlosses);
    ASSERT_LT(0, games_per_inference);
    int batch_size = vlosses * games_per_inference;

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
    const std::string& src = get_features_response.features();
    std::vector<DualNet::BoardFeatures> features(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < DualNet::kNumBoardFeatures; ++j) {
        features[i][j] = static_cast<float>(src[i * DualNet::kNumBoardFeatures + j]);
      }
    }
    std::vector<DualNet::Output> outputs(batch_size);
    dual_net_->RunMany(features, absl::MakeSpan(outputs));

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

  int vlosses = virtual_losses_;
  std::vector<DualNet::BoardFeatures> features(vlosses * clients_.size());
  std::vector<DualNet::Output> outputs(vlosses * clients_.size());
  std::vector<std::thread> client_threads;
  for (size_t i = 0; i < clients_.size(); ++i) {
    auto* client = clients_[i].get();
    auto* client_features = &features[i * vlosses];
    auto* client_output = &outputs[i * vlosses];
    client_threads.emplace_back([=]() {
      auto size = static_cast<size_t>(vlosses);
      client->RunMany({client_features, size}, {client_output, size});
    });
  }
  for (auto& thread : client_threads) {
    thread.join();
  }

  for (size_t i = 0; i < clients_.size(); ++i) {
    for (int vloss = 0; vloss < vlosses; ++vloss) {
      const auto& output = outputs[i * vlosses + vloss];
      ASSERT_EQ(value_, output.value);
      for (int i = 0; i < kNumMoves; ++i) {
        ASSERT_EQ(priors_[i], output.policy[i]);
      }
    }
  }

  server_thread.join();
}

}  // namespace
}  // namespace minigo
