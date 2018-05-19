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

#ifndef CC_DUAL_NET_INFERENCE_SERVER_H_
#define CC_DUAL_NET_INFERENCE_SERVER_H_

#include <memory>
#include <queue>
#include <thread>

#include "absl/synchronization/blocking_counter.h"
#include "cc/dual_net/dual_net.h"
#include "cc/thread_safe_queue.h"
#include "grpc++/server.h"

namespace minigo {

struct RemoteInference {
  const DualNet::BoardFeatures* features;
  DualNet::Output* output;
  absl::BlockingCounter* counter;
};

class InferenceServer {
 public:
  InferenceServer();
  ~InferenceServer();

  void RunInference(const DualNet::BoardFeatures* features,
                    DualNet::Output* output, absl::BlockingCounter* counter);

 private:
  std::thread thread_;
  // request_queue_ is owned by the service_ implementation.
  ThreadSafeQueue<RemoteInference>* request_queue_;
  std::unique_ptr<grpc::Server> server_;
  std::unique_ptr<grpc::Service> service_;
};

class InferenceClient : public DualNet {
 public:
  explicit InferenceClient(InferenceServer* server);

  void RunMany(absl::Span<const BoardFeatures* const> features,
               absl::Span<Output> outputs, Random* rnd = nullptr) override;

 private:
  InferenceServer* server_;
};

}  // namespace minigo

#endif  // CC_DUAL_NET_INFERENCE_SERVER_H_
