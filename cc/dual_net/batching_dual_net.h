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

// TODO(tommadams): rename file to batching_model.h

#include <atomic>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "cc/model/model.h"

namespace minigo {

struct BatchingModelStats {
  explicit BatchingModelStats(size_t buffer_count)
      : buffer_count(buffer_count) {}
  size_t num_inferences = 0;
  size_t buffer_count = 0;
  absl::Duration run_batch_time;
  absl::Duration run_many_time;
};

namespace internal {

// The ModelBatcher is responsible for batching up inference requests from
// from multiple BatchingModel clients into larger (and therefore more
// efficient) inferences.
// Each ModelBatcher instance is responsible for batching inference requests
// for a single model.
class ModelBatcher {
 public:
  // A single inference request from a client, possibly containing multiple
  // individual inferences because of virtual losses.
  struct InferenceRequest {
    ModelBatcher* other_batcher;
    const std::vector<const Model::Input*>* inputs;
    std::vector<Model::Output*>* outputs;
    std::string* model_name;
    absl::Notification* notification;
  };

  // model_impl: the model that will evaluate the batched inferences.
  explicit ModelBatcher(std::unique_ptr<Model> model_impl);
  ~ModelBatcher();

  const std::string& name() const { return model_impl_->name(); }

  void StartGame() LOCKS_EXCLUDED(&mutex_);
  void EndGame() LOCKS_EXCLUDED(&mutex_);
  void RunMany(ModelBatcher* other_batcher,
               const std::vector<const Model::Input*>& inputs,
               std::vector<Model::Output*>* outputs, std::string* model_name);
  BatchingModelStats FlushStats() LOCKS_EXCLUDED(&mutex_);

 private:
  size_t GetBatchSize() const EXCLUSIVE_LOCKS_REQUIRED(&mutex_);

  void MaybeRunBatchesLocked() EXCLUSIVE_LOCKS_REQUIRED(&mutex_);
  void RunBatch() EXCLUSIVE_LOCKS_REQUIRED(&mutex_);

  absl::Mutex mutex_;
  std::unique_ptr<Model> model_impl_;
  std::queue<InferenceRequest> queue_ GUARDED_BY(&mutex_);
  BatchingModelStats stats_ GUARDED_BY(&mutex_);

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

// The BatchingModel is a thin client for a ModelBatcher, which does all
// the real work. The only tricky thing here is that in two player games,
// BatchingModel keeps track of who the other player is so that its
// ModelBatcher knows whose turn it is.
class BatchingModel : public Model {
 public:
  explicit BatchingModel(std::shared_ptr<internal::ModelBatcher> batcher);

  void RunMany(const std::vector<const Input*>& inputs,
               std::vector<Output*>* outputs, std::string* model_name) override;

  void StartGame();
  void EndGame();
  void SetOther(BatchingModel* other);

 private:
  friend class internal::ModelBatcher;

  // The ModelBatcher used to batch our RunMany calls.
  std::shared_ptr<internal::ModelBatcher> batcher_;

  // In a two player game where StartGame was called with different
  // BatchingModel instances, other_batcher_ points to the ModelBatcher
  // used by the other player in the game. It's possible that batcher_ ==
  // other_batcher_ if both players are using the same model.
  std::shared_ptr<internal::ModelBatcher> other_batcher_ = nullptr;
};

// BatchingModelFactory managers the per-model ModelBatchers and creates
// their BatchingModel clients.
// TODO(tommadams): Don't derive BatchingModelFactory from ModelFactory, that
// way NewModel can return a unique_ptr<BatchingModel> and we can pass
// BatchingModel pointers to StartGame and EndGame. That way we can get rid of
// the dynamic_casts in those methods and have a compile-time guarantee that the
// models are of the correct type.
class BatchingModelFactory : public ModelFactory {
 public:
  BatchingModelFactory(std::unique_ptr<ModelFactory> factory_impl);

  std::unique_ptr<Model> NewModel(const std::string& descriptor) override;

  static void StartGame(Model* black, Model* white);
  static void EndGame(Model* black, Model* white);

  std::vector<std::pair<std::string, BatchingModelStats>> FlushStats();

 private:
  absl::Mutex mutex_;
  std::unique_ptr<ModelFactory> factory_impl_;

  // Map from model to BatchingService for that model.
  absl::flat_hash_map<std::string, std::shared_ptr<internal::ModelBatcher>>
      batchers_ GUARDED_BY(&mutex_);
};

}  // namespace minigo

#endif  // CC_DUAL_NET_BATCHING_DUAL_NET_H_
