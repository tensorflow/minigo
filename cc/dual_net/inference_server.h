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

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "absl/synchronization/notification.h"
#include "cc/dual_net/dual_net.h"
#include "cc/thread_safe_queue.h"
#include "grpc++/server.h"

namespace minigo {

namespace internal {
class InferenceServiceImpl;
}  // namespace internal

// A batch of inference requests.
struct RemoteInference {
  // A batch of features to run inference on.
  absl::Span<const DualNet::BoardFeatures> features;

  // Inference output for the batch.
  absl::Span<DualNet::Output> outputs;

  // Model used for the inference.
  std::string* model;

  // Notified when the batch is ready.
  absl::Notification* notification;
};

class InferenceServer {
 public:
  InferenceServer(int virtual_losses, int games_per_inference, int port);
  ~InferenceServer();

  // Return a new DualNet instance whose inference requests are performed
  // by this InferenceServer.
  std::unique_ptr<DualNet> NewDualNet();

 private:
  std::thread thread_;
  // request_queue_ is owned by the service_ implementation.
  std::unique_ptr<grpc::Server> server_;
  std::unique_ptr<internal::InferenceServiceImpl> service_;
};

}  // namespace minigo

#endif  // CC_DUAL_NET_INFERENCE_SERVER_H_
