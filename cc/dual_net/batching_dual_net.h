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

#ifndef CC_DUAL_NET_BATCHING_DUAL_NET_H_
#define CC_DUAL_NET_BATCHING_DUAL_NET_H_

#include <atomic>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "cc/dual_net/dual_net.h"

namespace minigo {

namespace internal {

// The ModelBatcher is responsible for batching up inference requests from
// from multiple BatchingDualNet clients into larger (and therefore more
// efficient) inferences.
// Each ModelBatcher instance is responsible for batching inference requests
// for a single model.
class ModelBatcher {
 public:
  // A single inference request from a client, possibly containing multiple
  // individual inferences because of virtual losses.
  struct InferenceRequest {
    ModelBatcher* other_batcher;
    std::vector<const DualNet::BoardFeatures*> features;
    std::vector<DualNet::Output*> outputs;
    std::string* model_name;
    absl::Notification* notification;
  };

  ModelBatcher(std::unique_ptr<DualNet> model_impl, size_t buffering);
  ~ModelBatcher();

  const std::string& name() const { return model_impl_->name(); }

  void StartGame() LOCKS_EXCLUDED(&mutex_);
  void EndGame() LOCKS_EXCLUDED(&mutex_);
  DualNet::InputLayout GetInputLayout() const;
  void RunMany(ModelBatcher* other_batcher,
               std::vector<const DualNet::BoardFeatures*> features,
               std::vector<DualNet::Output*> outputs, std::string* model_name);

 private:
  size_t GetBatchSize() const EXCLUSIVE_LOCKS_REQUIRED(&mutex_);

  void MaybeRunBatchesLocked() EXCLUSIVE_LOCKS_REQUIRED(&mutex_);
  void RunBatch() EXCLUSIVE_LOCKS_REQUIRED(&mutex_);

  absl::Mutex mutex_;
  std::unique_ptr<DualNet> model_impl_;
  const size_t buffering_;
  std::queue<InferenceRequest> queue_ GUARDED_BY(&mutex_);

  // Number of clients of this batcher that are playing in a two player game
  // and are currently waiting for the other player to play a move. These
  // clients are not going to make an inference request until it's their turn
  // and the batcher shouldn't wait for them to make a request.
  std::atomic<size_t> num_waiting_{0};

  // Number of clients of this batcher that are currently playing a game.
  size_t num_active_clients_ GUARDED_BY(&mutex_) = 0;

  // Stats that get reported when the ModelBatcher is destroyed.
  size_t num_batches_ GUARDED_BY(&mutex_) = 0;
  size_t num_inferences_ GUARDED_BY(&mutex_) = 0;
};

}  // namespace internal

// The BatchingDualNet is a thin client for a ModelBatcher, which does all
// the real work. The only tricky thing here is that in two player games,
// BatchingDualNet keeps track of who the other player is so that its
// ModelBatcher knows whose turn it is.
class BatchingDualNet : public DualNet {
 public:
  explicit BatchingDualNet(std::shared_ptr<internal::ModelBatcher> batcher);

  void RunMany(std::vector<const BoardFeatures*> features,
               std::vector<Output*> outputs, std::string* model) override;
  InputLayout GetInputLayout() const override;
  void StartGame();
  void EndGame();
  void SetOther(BatchingDualNet* other);

 private:
  friend class internal::ModelBatcher;

  // The ModelBatcher used to batch our RunMany calls.
  std::shared_ptr<internal::ModelBatcher> batcher_;

  // In a two player game where StartGame was called with different
  // BatchingDualNet instances, other_batcher_ points to the ModelBatcher
  // used by the other player in the game. It's possible that batcher_ ==
  // other_batcher_ if both players are using the same model.
  std::shared_ptr<internal::ModelBatcher> other_batcher_ = nullptr;
};

// BatchingDualNetFactory managers the per-model ModelBatchers and creates
// their BatchingDualNet clients.
class BatchingDualNetFactory : public DualNetFactory {
 public:
  BatchingDualNetFactory(std::unique_ptr<DualNetFactory> factory_impl);

  int GetBufferCount() const;

  std::unique_ptr<DualNet> NewDualNet(const std::string& model_path);

  static void StartGame(DualNet* black, DualNet* white);
  static void EndGame(DualNet* black, DualNet* white);

 private:
  absl::Mutex mutex_;
  std::unique_ptr<DualNetFactory> factory_impl_;

  // Map from model to BatchingService for that model.
  absl::flat_hash_map<std::string, std::shared_ptr<internal::ModelBatcher>>
      batchers_ GUARDED_BY(&mutex_);
};

}  // namespace minigo

#endif  // CC_DUAL_NET_BATCHING_DUAL_NET_H_
