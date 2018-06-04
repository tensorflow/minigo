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
#include "grpc++/grpc++.h"
#include "proto/inference_service.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using grpc::StatusCode;

namespace minigo {

namespace {

class ServiceImpl final : public InferenceService::Service {
 public:
  // TODO(tommadams): tune batch_timeout_.
  // TODO(tommadams): try having the InferenceServer keep track of how many
  // InferenceClients are actively wanting to make inference requests (and their
  // desired batch size). That way it should be possible to know exactly how
  // many inference requests to wait for when batching.
  ServiceImpl()
      : batch_size_(8),
        batch_id_(1),
        batch_timeout_(absl::Microseconds(1000)) {}

  Status GetConfig(ServerContext* context, const GetConfigRequest* request,
                   GetConfigResponse* response) override {
    response->set_board_size(kN);
    response->set_batch_size(batch_size_);
    return Status::OK;
  }

  Status GetFeatures(ServerContext* context, const GetFeaturesRequest* request,
                     GetFeaturesResponse* response) override {
    // std::cerr << "### GetFeatures()" << std::endl;

    std::vector<RemoteInference> batch;

    {
      // Lock batch_mutex_ while popping inference requests off the
      // request_queue_: we want make sure that each request fills up as much
      // of a batch as possible. If multiple threads all popped inference
      // requests off the queue in parallel, we'd likely end up with multiple
      // partially empty batches.
      absl::MutexLock lock(&batch_mutex_);

      // Wait forever until an inference is pushed onto the request_queue_.
      RemoteInference inference;
      auto timeout = absl::Milliseconds(100);
      while (!request_queue_.PopWithTimeout(&inference, timeout)) {
        if (context->IsCancelled()) {
          // std::cerr << "### GetFeatures() CANCELLED" << std::endl;
          return Status(StatusCode::CANCELLED, "connection terminated");
        }
      }

      batch.push_back(inference);

      // Once we have the first inference, wait up to batch_timeout_ for more
      // inferences to arrive until we reach the maximum batch_size_.
      auto deadline = absl::Now() + batch_timeout_;
      while (batch.size() < batch_size_) {
        auto now = absl::Now();
        if (now >= deadline ||
            !request_queue_.PopWithTimeout(&inference, deadline - now)) {
          break;
        }
        batch.push_back(inference);
      }
    }

    // Populate the response with the batch we just built.
    response->set_batch_id(batch_id_++);
    for (const auto& inference : batch) {
      const auto& src = *inference.features;
      for (size_t i = 0; i < src.size(); ++i) {
        response->add_features(src[i]);
      }
    }

    // The RPC ops in the worker graph seem to require that the batch size is
    // known at graph build time, so make sure we always send a batch of size
    // batch_size_.
    int padding = batch_size_ - batch.size();
    for (int i = 0; i < padding; ++i) {
      for (size_t j = 0; j < DualNet::kNumBoardFeatures; ++j) {
        response->add_features(0);
      }
    }

    {
      absl::MutexLock lock(&pending_batches_mutex_);
      pending_batches_[response->batch_id()] = std::move(batch);
    }

    return Status::OK;
  }

  Status PutOutputs(ServerContext* context, const PutOutputsRequest* request,
                    PutOutputsResponse* response) override {
    // std::cerr << "### PutOutputs(" << request->batch_id() << ")" <<
    // std::endl;

    std::vector<RemoteInference> batch;
    {
      absl::MutexLock pending_batches_lock(&pending_batches_mutex_);
      auto it = pending_batches_.find(request->batch_id());
      MG_CHECK(it != pending_batches_.end());
      batch = std::move(it->second);
      pending_batches_.erase(it);
    }

    // Check we got the expected number of values.
    // (Note that if the prior GetFeatures response was padded, we may have
    // more values than batch_.size()).
    MG_CHECK(request->value().size() == static_cast<int>(batch_size_))
        << "Expected " << batch_size_ << " values, got "
        << request->value().size();

    // There should be (N * N + 1) policy values for each inference.
    MG_CHECK(request->policy().size() ==
             static_cast<int>(batch.size() * kNumMoves));

    // Because of padding, it's possible that we have more value & policy
    // results than were requested: match sure to only extract the first
    // batch.size() outputs.
    for (size_t j = 0; j < batch.size(); ++j) {
      auto& inference = batch[j];
      auto& dst_policy = inference.output->policy;
      for (int i = 0; i < kNumMoves; ++i) {
        dst_policy[i] = request->policy(i + j * kNumMoves);
      }
      inference.output->value = request->value(j);
      inference.counter->DecrementCount();
    }

    // std::cerr << "### PutOutputs() OK" << std::endl;
    return Status::OK;
  }

  ThreadSafeQueue<RemoteInference>* request_queue() { return &request_queue_; }

 private:
  // GetFeatures will attempt to pop up to this many inference requests off
  // request_queue_ before replying.
  const size_t batch_size_;

  std::atomic<int32_t> batch_id_{1};

  // After successfully popping the first request off request_queue, GetFeatures
  // will wait for up to the batch_timeout_ for more inference requests before
  // replying.
  absl::Duration batch_timeout_;

  ThreadSafeQueue<RemoteInference> request_queue_;

  absl::Mutex batch_mutex_;

  absl::Mutex pending_batches_mutex_;
  std::map<int32_t, std::vector<RemoteInference>> pending_batches_
      GUARDED_BY(&pending_batches_mutex_);
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
