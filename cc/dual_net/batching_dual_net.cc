#include "cc/dual_net/batching_dual_net.h"

#include <atomic>
#include <queue>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
// DO NOT CHECK IN
// DO NOT CHECK IN
// DO NOT CHECK IN
#include <sys/syscall.h>
#include <unistd.h>
#define gettid() syscall(SYS_gettid)
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "cc/file/path.h"
// DO NOT CHECK IN
// DO NOT CHECK IN
// DO NOT CHECK IN
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "cc/check.h"

namespace minigo {
namespace {

class BatchingDualNet;
class BatchingDualNetFactory;
struct ModelBatcher;

class BatchingDualNet : public DualNet {
 public:
  explicit BatchingDualNet(std::shared_ptr<ModelBatcher> batcher);

  void RunMany(std::vector<const BoardFeatures*> features,
               std::vector<Output*> outputs, std::string* model) override;
  InputLayout GetInputLayout() const override;

  void SetOtherBatcher(std::shared_ptr<ModelBatcher> batcher);

  // protected:

  // The ModelBatcher used to batch our RunMany calls.
  std::shared_ptr<ModelBatcher> batcher_;

  // In a two player game where StartGame was called with different
  // BatchingDualNet instances, other_batcher_ points to the ModelBatcher
  // used by the other player in the game. It's possible that batcher_ ==
  // other_batcher_ if both players are using the same model.
  std::shared_ptr<ModelBatcher> other_batcher_ = nullptr;
};

class BatchingDualNetFactory : public DualNetFactory {
 public:
  BatchingDualNetFactory(std::unique_ptr<DualNetFactory> factory_impl);
  int GetBufferCount() const;
  std::unique_ptr<DualNet> NewDualNet(const std::string& model);
  void StartGame(DualNet* black, DualNet* white);
  void EndGame(DualNet* black, DualNet* white);

 private:
  absl::Mutex mutex_;
  std::unique_ptr<DualNetFactory> factory_impl_;

  // Map from model to BatchingService for that model.
  absl::flat_hash_map<std::string, std::shared_ptr<ModelBatcher>> batchers_
      GUARDED_BY(&mutex_);
};

struct InferenceData {
  std::vector<const DualNet::BoardFeatures*> features;
  std::vector<DualNet::Output*> outputs;
  std::string* model_name;
  absl::Notification* notification;
};

struct ModelBatcher {
  // TODO(tommadams): remove model_path
  ModelBatcher(std::string model_path, std::unique_ptr<DualNet> model_impl,
               size_t buffering)
      : model_impl(std::move(model_impl)),
        buffering(buffering),
        model_path(std::move(model_path)) {}

  void StartGame() LOCKS_EXCLUDED(&mutex) {
    absl::MutexLock lock(&mutex);
    num_active_clients += 1;
  }

  void EndGame() LOCKS_EXCLUDED(&mutex) {
    absl::MutexLock lock(&mutex);
    num_active_clients -= 1;
    MaybeRunBatchesLocked();
  }

  DualNet::InputLayout GetInputLayout() const {
    return model_impl->GetInputLayout();
  }

  void RunMany(ModelBatcher* other_batcher,
               std::vector<const DualNet::BoardFeatures*> features,
               std::vector<DualNet::Output*> outputs, std::string* model_name) {
    MG_CHECK(features.size() == outputs.size());

    // -----------------------------------------------
    absl::Notification notification;

    {
      absl::MutexLock lock(&mutex);
      inference_queue.push(
          {std::move(features), std::move(outputs), model_name, &notification});
      if (other_batcher == this) {
        // Two player game. Players are using the same model.
        num_waiting += 1;
      }
      MaybeRunBatchesLocked();
    }

    if (other_batcher != nullptr && other_batcher != this) {
      // Two player game. Players are using different models.
      absl::MutexLock lock(&other_batcher->mutex);
      other_batcher->num_waiting += 1;
      other_batcher->MaybeRunBatchesLocked();
    }

    notification.WaitForNotification();

    if (other_batcher == this) {
      // Two player game. Players are using the same model.
      absl::MutexLock lock(&mutex);
      num_waiting -= 1;
    } else if (other_batcher != nullptr) {
      // Two player game. Players are using different models.
      absl::MutexLock lock(&other_batcher->mutex);
      other_batcher->num_waiting -= 1;
    }
  }

  // TODO(tommadams): make this private
  void MaybeRunBatchesLocked() EXCLUSIVE_LOCKS_REQUIRED(&mutex) {
    while (inference_queue.size() >=
               (num_active_clients + buffering - 1) / buffering ||
           inference_queue.size() + num_waiting >= num_active_clients) {
      if (inference_queue.empty()) {
        return;
      }
      auto str =
          absl::StrCat(absl::StrFormat("%x", gettid()), " MRB ",
                       file::Stem(model_path), " iq:", inference_queue.size(),
                       " nw:", num_waiting, " nc:", num_active_clients);
      std::cerr << absl::StrCat(str, "  YES\n");
      RunBatch();
    }
  }

  void RunBatch() EXCLUSIVE_LOCKS_REQUIRED(&mutex) {
    auto batch_size = (num_active_clients + buffering - 1) / buffering;
    std::vector<const DualNet::BoardFeatures*> features;
    std::vector<DualNet::Output*> outputs;
    features.reserve(batch_size);
    outputs.reserve(batch_size);

    std::vector<InferenceData> inferences;

    while (!inference_queue.empty() && inferences.size() < batch_size) {
      auto& inference = inference_queue.front();
      size_t num_features = inference.features.size();

      std::copy_n(inference.features.begin(), num_features,
                  std::back_inserter(features));
      std::copy_n(inference.outputs.begin(), num_features,
                  std::back_inserter(outputs));
      inferences.push_back(std::move(inference));

      inference_queue.pop();
    }

    std::vector<std::string> parts = {absl::StrFormat("%x", gettid()), "RUN",
                                      absl::StrCat(num_active_clients)};
    std::string model_name;

    // Unlock the mutex while running inference.
    mutex.Unlock();

    model_impl->RunMany(std::move(features), std::move(outputs), &model_name);
    for (const auto& inference : inferences) {
      if (inference.model_name != nullptr) {
        *inference.model_name = model_name;
        parts.emplace_back(file::Stem(model_name));
      }
      inference.notification->Notify();
    }
    parts.push_back("\n");
    std::cerr << absl::StrJoin(parts, " ");
    mutex.Lock();
  }

  absl::Mutex mutex;
  std::unique_ptr<DualNet> model_impl;
  const size_t buffering;
  std::queue<InferenceData> inference_queue GUARDED_BY(&mutex);

  // Number of clients of this batcher that are playing in a two player game and
  // are currently waiting for the other player to play a move. These clients
  // are not going to make an inference request until it's their turn and the
  // batcher shouldn't wait for them to make a request.
  size_t num_waiting = 0 GUARDED_BY(&mutex);

  // Number of clients of this batcher that are currently playing a game.
  size_t num_active_clients = 0 GUARDED_BY(&mutex);

  const std::string model_path;
};

BatchingDualNet::BatchingDualNet(std::shared_ptr<ModelBatcher> batcher)
    : batcher_(std::move(batcher)) {}

void BatchingDualNet::RunMany(std::vector<const BoardFeatures*> features,
                              std::vector<Output*> outputs,
                              std::string* model) {
  batcher_->RunMany(other_batcher_.get(), std::move(features),
                    std::move(outputs), model);
};

DualNet::InputLayout BatchingDualNet::GetInputLayout() const {
  return batcher_->GetInputLayout();
}

void BatchingDualNet::SetOtherBatcher(
    std::shared_ptr<ModelBatcher> other_batcher) {
  if (other_batcher == nullptr) {
    MG_CHECK(other_batcher_ != nullptr);
  } else {
    MG_CHECK(other_batcher_ == nullptr);
  }
  other_batcher_ = std::move(other_batcher);
}

BatchingDualNetFactory::BatchingDualNetFactory(
    std::unique_ptr<DualNetFactory> factory_impl)
    : factory_impl_(std::move(factory_impl)) {}

int BatchingDualNetFactory::GetBufferCount() const {
  return factory_impl_->GetBufferCount();
}

std::unique_ptr<DualNet> BatchingDualNetFactory::NewDualNet(
    const std::string& model_path) {
  absl::MutexLock lock(&mutex_);

  // Find or create a service for the requested model.
  auto it = batchers_.find(model_path);
  if (it == batchers_.end()) {
    auto batcher = std::make_shared<ModelBatcher>(
        model_path, factory_impl_->NewDualNet(model_path), GetBufferCount());
    it = batchers_.emplace(model_path, std::move(batcher)).first;
  }

  auto result = absl::make_unique<BatchingDualNet>(it->second);

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

  return result;
}

void BatchingDualNetFactory::StartGame(DualNet* black, DualNet* white) {
  auto* b = dynamic_cast<BatchingDualNet*>(black);
  auto* w = dynamic_cast<BatchingDualNet*>(white);
  MG_CHECK(b != nullptr && w != nullptr);

  if (b != w) {
    b->SetOtherBatcher(w->batcher_);
    w->SetOtherBatcher(b->batcher_);
  }

  b->batcher_->StartGame();
  if (b != w) {
    w->batcher_->StartGame();
  }
}

void BatchingDualNetFactory::EndGame(DualNet* black, DualNet* white) {
  auto* b = dynamic_cast<BatchingDualNet*>(black);
  auto* w = dynamic_cast<BatchingDualNet*>(white);
  MG_CHECK(b != nullptr && w != nullptr);

  if (b != w) {
    b->SetOtherBatcher(nullptr);
    w->SetOtherBatcher(nullptr);
  }

  b->batcher_->EndGame();
  if (b != w) {
    w->batcher_->EndGame();
  }
}

}  // namespace

std::unique_ptr<DualNetFactory> NewBatchingDualNetFactory(
    std::unique_ptr<DualNetFactory> impl) {
  return absl::make_unique<BatchingDualNetFactory>(std::move(impl));
}

}  // namespace minigo
