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

#include <functional>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "cc/dual_net/inference_service.grpc.pb.h"
#include "grpc++/grpc++.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using grpc::StatusCode;

namespace minigo {

namespace {

class ServiceImpl final : public InferenceService::Service {
 public:
  ServiceImpl() {}

  Status GetFeatures(ServerContext* context, const GetFeaturesRequest* request,
                     GetFeaturesResponse* response) override {
    RemoteInference inference = request_queue_.Pop();
    output_queue_.Push(inference);

    const auto& features = *inference.features;
    for (size_t i = 0; i < features.size(); ++i) {
      response->add_features(features[i]);
    }

    return Status::OK;
  }

  Status PutOutputs(ServerContext* context, const PutOutputsRequest* request,
                    PutOutputsResponse* response) override {
    RemoteInference inference;
    if (!output_queue_.TryPop(&inference)) {
      return Status(StatusCode::FAILED_PRECONDITION, "nothing in output queue",
                    "GetFeatures must be called before PutOutputs");
    }

    const auto& src_policy = request->policy();
    auto& dst_policy = inference.output->policy;
    for (int i = 0; i < src_policy.size(); ++i) {
      dst_policy[i] = src_policy[i];
    }
    inference.output->value = request->value();
    inference.counter->DecrementCount();

    return Status::OK;
  }

  ThreadSafeQueue<RemoteInference>* request_queue() { return &request_queue_; }

 private:
  ThreadSafeQueue<RemoteInference> request_queue_;
  ThreadSafeQueue<RemoteInference> output_queue_;
};

class InferenceClient : public DualNet {
 public:
  explicit InferenceClient(std::function<void(RemoteInference)> run_inference)
      : run_inference_(run_inference) {}

  void RunMany(absl::Span<const BoardFeatures* const> features,
               absl::Span<Output> outputs, Random* rnd) override {
    absl::BlockingCounter pending_count(features.size());
    for (size_t i = 0; i < features.size(); ++i) {
      run_inference_({features[i], &outputs[i], &pending_count});
    }
    pending_count.Wait();
  }

 private:
  std::function<void(RemoteInference)> run_inference_;
};

}  // namespace

InferenceServer::InferenceServer() {
  std::string server_address("0.0.0.0:50051");

  auto impl = absl::make_unique<ServiceImpl>();
  request_queue_ = impl->request_queue();
  service_ = std::move(impl);

  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(service_.get());
  server_ = builder.BuildAndStart();
  std::cerr << "Inference server listening on " << server_address << std::endl;

  thread_ = std::thread([this]() { server_->Wait(); });
}

InferenceServer::~InferenceServer() {
  server_->Shutdown();
  thread_.join();
}

std::unique_ptr<DualNet> InferenceServer::NewDualNet() {
  return absl::make_unique<InferenceClient>(
      [this](const RemoteInference& inference) {
        request_queue_->Push(inference);
      });
}

}  // namespace minigo
