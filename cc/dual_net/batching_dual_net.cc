#include "cc/dual_net/batching_dual_net.h"

#include <atomic>
#include <queue>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "cc/check.h"

namespace minigo {
namespace {

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
    ModelBatcher* batcher;
    ModelBatcher* other_batcher;
    std::vector<const DualNet::BoardFeatures*> features;
    std::vector<DualNet::Output*> outputs;
    std::string* model_name;
    absl::Notification* notification;
  };

  ModelBatcher(std::unique_ptr<DualNet> model_impl, size_t buffering)
      : model_impl_(std::move(model_impl)), buffering_(buffering) {}

  ~ModelBatcher() {
    std::cerr << "Ran " << num_batches_ << " batches with an average size of "
              << static_cast<float>(num_inferences_) / num_batches_ << ".\n";
  }

  void StartGame() LOCKS_EXCLUDED(&mutex_) {
    absl::MutexLock lock(&mutex_);
    num_active_clients_ += 1;
  }

  void EndGame() LOCKS_EXCLUDED(&mutex_) {
    absl::MutexLock lock(&mutex_);
    num_active_clients_ -= 1;
    MaybeRunBatchesLocked();
  }

  DualNet::InputLayout GetInputLayout() const {
    return model_impl_->GetInputLayout();
  }

  void RunMany(ModelBatcher* other_batcher,
               std::vector<const DualNet::BoardFeatures*> features,
               std::vector<DualNet::Output*> outputs, std::string* model_name) {
    MG_CHECK(features.size() == outputs.size());

    absl::Notification notification;

    {
      absl::MutexLock lock(&mutex_);
      queue_.push({this, other_batcher, std::move(features), std::move(outputs),
                   model_name, &notification});
      if (other_batcher != nullptr) {
        other_batcher->num_waiting_ += 1;
      }
      MaybeRunBatchesLocked();
    }

    if (other_batcher != nullptr) {
      absl::MutexLock lock(&other_batcher->mutex_);
      other_batcher->MaybeRunBatchesLocked();
    }

    notification.WaitForNotification();
  }

 private:
  size_t GetBatchSize() const {
    return (num_active_clients_ + buffering_ - 1) / buffering_;
  }

  void MaybeRunBatchesLocked() EXCLUSIVE_LOCKS_REQUIRED(&mutex_) {
    while (!queue_.empty()) {
      auto queue_size = queue_.size();
      if (queue_size < GetBatchSize()) {
        // The queue doesn't have enough requests to fill a batch: see if we
        // can run a smaller batch instead.
        //
        // We run a small batch if all clients of this model have either
        // submitted inference requests, or are in a two player game and
        // waiting for the other player's inference.
        //
        // Additionally... when starting a bunch of games in parallel, we
        // will initially submit several smaller batches until all the
        // clients have been created. This has a ripple effect across all
        // subsequent reads, making the batching irregular. To counteract
        // this, we additionally enforce the constraint that a small batch
        // can't be run until at least half of the clients have submitted
        // inference requests. This has the effect of forcing those clients
        // to run their batches in lock-step.
        bool can_run_small_batch =
            queue_size >= num_active_clients_ / 2 &&
            queue_size + num_waiting_ >= num_active_clients_;
        if (!can_run_small_batch) {
          break;
        }
      }

      RunBatch();
    }
  }

  void RunBatch() EXCLUSIVE_LOCKS_REQUIRED(&mutex_) {
    auto batch_size = GetBatchSize();

    std::vector<const DualNet::BoardFeatures*> features;
    std::vector<DualNet::Output*> outputs;
    std::vector<InferenceRequest> inferences;
    features.reserve(batch_size);
    outputs.reserve(batch_size);
    inferences.reserve(batch_size);

    while (!queue_.empty() && inferences.size() < batch_size) {
      auto& inference = queue_.front();
      size_t num_features = inference.features.size();

      std::copy_n(inference.features.begin(), num_features,
                  std::back_inserter(features));
      std::copy_n(inference.outputs.begin(), num_features,
                  std::back_inserter(outputs));
      inferences.push_back(std::move(inference));

      queue_.pop();
    }

    num_batches_ += 1;
    num_inferences_ += features.size();

    // Unlock the mutex while running inference. This allows more inferences
    // to be enqueued while inference is running.
    mutex_.Unlock();

    std::string model_name;
    model_impl_->RunMany(std::move(features), std::move(outputs), &model_name);

    for (auto& inference : inferences) {
      if (inference.model_name != nullptr) {
        *inference.model_name = model_name;
      }
      // For all two player games, tell the batcher of the opponent model that
      // it isn't blocked on this inference any more.
      if (inference.other_batcher != nullptr) {
        inference.other_batcher->num_waiting_ -= 1;
      }
    }

    // All the required work is done, unblock all the waiting clients.
    for (auto& inference : inferences) {
      inference.notification->Notify();
    }

    // Lock the mutex again.
    mutex_.Lock();
  }

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
  size_t num_active_clients_ = 0 GUARDED_BY(&mutex_);

  // Stats that get reported when the ModelBatcher is destroyed.
  size_t num_batches_ = 0 GUARDED_BY(&mutex_);
  size_t num_inferences_ = 0 GUARDED_BY(&mutex_);
};

// The BatchingDualNet is a thin client for a ModelBatcher, which does all
// the real work. The only tricky thing here is that in two player games,
// BatchingDualNet keeps track of who the other player is so that its
// ModelBatcher knows whose turn it is.
class BatchingDualNet : public DualNet {
 public:
  explicit BatchingDualNet(std::shared_ptr<ModelBatcher> batcher)
      : batcher_(std::move(batcher)) {}

  void RunMany(std::vector<const BoardFeatures*> features,
               std::vector<Output*> outputs, std::string* model) override {
    batcher_->RunMany(other_batcher_.get(), std::move(features),
                      std::move(outputs), model);
  }
  InputLayout GetInputLayout() const override {
    return batcher_->GetInputLayout();
  }
  void StartGame() { batcher_->StartGame(); }
  void EndGame() { batcher_->EndGame(); }
  void SetOther(BatchingDualNet* other) {
    if (other == nullptr) {
      MG_CHECK(other_batcher_ != nullptr);
      other_batcher_ = nullptr;
    } else {
      MG_CHECK(other_batcher_ == nullptr);
      other_batcher_ = other->batcher_;
    }
  }

 protected:
  friend class ModelBatcher;

  // The ModelBatcher used to batch our RunMany calls.
  std::shared_ptr<ModelBatcher> batcher_;

  // In a two player game where StartGame was called with different
  // BatchingDualNet instances, other_batcher_ points to the ModelBatcher
  // used by the other player in the game. It's possible that batcher_ ==
  // other_batcher_ if both players are using the same model.
  std::shared_ptr<ModelBatcher> other_batcher_ = nullptr;
};

// BatchingDualNetFactory managers the per-model ModelBathers and creates
// their BatchingDualNet clients.
// TODO(tommadams): Reconsider the decision to have a single factory manage
// multiple models. Originally, I thought it would be required to make robust
// batching & model reloading work but that turned out not to be the case.
class BatchingDualNetFactory : public DualNetFactory {
 public:
  BatchingDualNetFactory(std::unique_ptr<DualNetFactory> factory_impl)
      : factory_impl_(std::move(factory_impl)) {}

  int GetBufferCount() const { return factory_impl_->GetBufferCount(); }

  std::unique_ptr<DualNet> NewDualNet(const std::string& model_path) {
    absl::MutexLock lock(&mutex_);

    // Find or create a service for the requested model.
    auto it = batchers_.find(model_path);
    if (it == batchers_.end()) {
      auto batcher = std::make_shared<ModelBatcher>(
          factory_impl_->NewDualNet(model_path), GetBufferCount());
      it = batchers_.emplace(model_path, std::move(batcher)).first;
    }

    auto model = absl::make_unique<BatchingDualNet>(it->second);

    // Take this opportunity to prune any services that have no clients.
    it = batchers_.begin();
    while (it != batchers_.end()) {
      // If the factory is the only one left with a reference to the batcher,
      // delete it.
      if (it->second.use_count() == 1) {
        batchers_.erase(it++);
      } else {
        ++it;
      }
    }

    return model;
  }

  void StartGame(DualNet* black, DualNet* white) {
    // TODO(tommadams): figure out if we can refactor the code somehow to take
    // BatchingDualNet pointers and avoid these dynamic_casts.
    auto* b = dynamic_cast<BatchingDualNet*>(black);
    auto* w = dynamic_cast<BatchingDualNet*>(white);
    MG_CHECK(b != nullptr && w != nullptr);

    if (b != w) {
      // This is a two player game, inform each client who the other one is.
      b->SetOther(w);
      w->SetOther(b);
    }

    b->StartGame();
    if (b != w) {
      w->StartGame();
    }
  }

  void EndGame(DualNet* black, DualNet* white) {
    // TODO(tommadams): figure out if we can refactor the code somehow to take
    // BatchingDualNet pointers and avoid these dynamic_casts.
    auto* b = dynamic_cast<BatchingDualNet*>(black);
    auto* w = dynamic_cast<BatchingDualNet*>(white);
    MG_CHECK(b != nullptr && w != nullptr);

    if (b != w) {
      b->SetOther(nullptr);
      w->SetOther(nullptr);
    }

    b->EndGame();
    if (b != w) {
      w->EndGame();
    }
  }

 private:
  absl::Mutex mutex_;
  std::unique_ptr<DualNetFactory> factory_impl_;

  // Map from model to BatchingService for that model.
  absl::flat_hash_map<std::string, std::shared_ptr<ModelBatcher>> batchers_
      GUARDED_BY(&mutex_);
};

}  // namespace

std::unique_ptr<DualNetFactory> NewBatchingDualNetFactory(
    std::unique_ptr<DualNetFactory> impl) {
  return absl::make_unique<BatchingDualNetFactory>(std::move(impl));
}

}  // namespace minigo
