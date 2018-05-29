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

#include <atomic>
#include <functional>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/time/time.h"
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
  ServiceImpl() : batch_size_(8), batch_timeout_(absl::Microseconds(100)) {}

  Status GetFeatures(ServerContext* context, const GetFeaturesRequest* request,
                     GetFeaturesResponse* response) override {
    absl::MutexLock lock(&m_);

    MG_CHECK(batch_.empty());

    // Wait "forever" until an inference is pushed onto the request_queue_.
    RemoteInference inference;
    auto timeout = absl::Milliseconds(100);
    while (!request_queue_.PopWithTimeout(&inference, timeout)) {
      if (context->IsCancelled()) {
        return Status(StatusCode::CANCELLED, "connection terminated");
      }
    }
    batch_.push_back(inference);

    // Once we have the first inference, wait up to batch_timeout_ for more
    // inferences to arrive until we reach the maximum batch_size_.
    auto deadline = absl::Now() + batch_timeout_;
    while (batch_.size() < batch_size_) {
      auto now = absl::Now();
      if (now >= deadline ||
          !request_queue_.PopWithTimeout(&inference, deadline - now)) {
        break;
      }
      batch_.push_back(inference);
    }

    // Response with the batch.
    for (const auto& inference : batch_) {
      const auto& src = *inference.features;
      for (size_t i = 0; i < src.size(); ++i) {
        response->add_features(src[i]);
      }
    }

    return Status::OK;
  }

  Status PutOutputs(ServerContext* context, const PutOutputsRequest* request,
                    PutOutputsResponse* response) override {
    absl::MutexLock lock(&m_);

    // There should be one value for each inference.
    MG_CHECK(request->value().size() == static_cast<int>(batch_.size()))
        << request->value().size() << " != " << batch_.size();

    // There should be (N * N + 1) policy values for each inference.
    MG_CHECK(request->policy().size() ==
             static_cast<int>(batch_.size() * kNumMoves));

    for (int j = 0; j < request->value().size(); ++j) {
      auto& inference = batch_[j];
      auto& dst_policy = inference.output->policy;
      for (int i = 0; i < kNumMoves; ++i) {
        dst_policy[i] = request->policy(i + j * kNumMoves);
      }
      inference.output->value = request->value(j);
      inference.counter->DecrementCount();
    }

    batch_.clear();

    return Status::OK;
  }

  ThreadSafeQueue<RemoteInference>* request_queue() { return &request_queue_; }

 private:
  // GetFeatures will attempt to pop up to this many inference requests off
  // request_queue_ before replying.
  const size_t batch_size_;

  // After successfully popping the first request off request_queue, GetFeatures
  // will wait for up to the batch_timeout_ for more inference requests before
  // replying.
  absl::Duration batch_timeout_;

  ThreadSafeQueue<RemoteInference> request_queue_;

  absl::Mutex m_;
  std::vector<RemoteInference> batch_ GUARDED_BY(&m_);
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
  // Passing inf past to Shutdown makes it shutdown immediately.
  server_->Shutdown(gpr_inf_past(GPR_CLOCK_REALTIME));
  thread_.join();
}

std::unique_ptr<DualNet> InferenceServer::NewDualNet() {
  return absl::make_unique<InferenceClient>(
      [this](const RemoteInference& inference) {
        request_queue_->Push(inference);
      });
}

}  // namespace minigo
