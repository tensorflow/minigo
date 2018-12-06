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

/*
class BatchingService {
 public:
  BatchingService(BatchingDualNetFactory* factory,
                  std::unique_ptr<DualNet> dual_net, size_t batch_size,
                  std::string model)
      : factory_(factory),
        dual_net_(std::move(dual_net)),
        num_clients_(0),
        queue_counter_(0),
        run_counter_(0),
        num_runs_(0),
        batch_size_(batch_size),
        model_(std::move(model)) {
    dual_net_->Reserve(batch_size);
  }

  ~BatchingService() {
    std::cerr << "Ran " << num_runs_ << " batches with an average size of "
              << static_cast<float>(run_counter_) / num_runs_ << ".\n";
  }

  void IncrementClientCount() {
    absl::MutexLock lock(&mutex_);
    ++num_clients_;
  }

  void DecrementClientCount() {
    absl::MutexLock lock(&mutex_);
    --num_clients_;
    factory_->MaybeRunBatches();
  }

  size_t num_clients() const {
    absl::MutexLock lock(&mutex_);
    return num_clients_;
  }

  void RunMany(std::vector<const DualNet::BoardFeatures*> features,
               std::vector<DualNet::Output*> outputs, std::string* model) {
    size_t num_features = features.size();
    MG_CHECK(num_features <= batch_size_);
    MG_CHECK(num_features == outputs.size());

    absl::Notification notification;

    {
      absl::MutexLock lock(&mutex_);

      queue_counter_ += num_features;
      inference_queue_.push(
          {std::move(features), std::move(outputs), model, &notification});

      factory_->MaybeRunBatches();
    }

    notification.WaitForNotification();
  }

  DualNet::InputLayout GetInputLayout() const {
    return dual_net_->GetInputLayout();
  }

 private:
  void MaybeRunBatches() EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    std::cerr << absl::StrCat(
        "### ", model_, "  qc-rc:", queue_counter_ - run_counter_,
        "  bs:", batch_size_, "  iq:", inference_queue_.size(),
        "  nc:", num_clients_, "\n");
    while (size_t batch_size =
               std::min(queue_counter_ - run_counter_, batch_size_)) {
      // Stop if we won't fill a batch and more clients will send requests.
      if (batch_size < batch_size_ && inference_queue_.size() != num_clients_) {
        break;
      }

      RunBatch(batch_size);
    }
  }

  void RunBatch(size_t batch_size) EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    std::vector<const DualNet::BoardFeatures*> features;
    std::vector<DualNet::Output*> outputs;
    features.reserve(batch_size);
    outputs.reserve(batch_size);

    std::vector<InferenceData> inferences;

    while (batch_size > 0) {
      auto& inference = inference_queue_.front();
      size_t num_features = inference.features.size();

      if (num_features > batch_size) {
        break;  // Request doesn't fit anymore.
      }

      std::copy_n(inference.features.begin(), num_features,
                  std::back_inserter(features));
      std::copy_n(inference.outputs.begin(), num_features,
                  std::back_inserter(outputs));
      inferences.push_back(std::move(inference));

      inference_queue_.pop();
      batch_size -= num_features;
      run_counter_ += num_features;
    }

    // Unlock the mutex while running inference.
    mutex_.Unlock();

    std::string model;
    dual_net_->RunMany(std::move(features), std::move(outputs), &model);
    for (const auto& inference : inferences) {
      if (inference.model != nullptr) {
        *inference.model = model;
      }
      inference.notification->Notify();
    }
    mutex_.Lock();

    ++num_runs_;
  }

  std::unique_ptr<DualNet> dual_net_;

  mutable absl::Mutex mutex_;

  size_t num_clients_ GUARDED_BY(&mutex_);

  std::queue<InferenceData> inference_queue_ GUARDED_BY(&mutex_);
  // Number of features pushed to inference queue.
  size_t queue_counter_ GUARDED_BY(&mutex_);
  // Number of features popped from inference queue.
  size_t run_counter_ GUARDED_BY(&mutex_);

  // For printing batching stats in the destructor only.
  size_t num_runs_ GUARDED_BY(&mutex_);

  const size_t batch_size_;

  const std::string model_;
};
*/

class BatchingDualNet;
class BatchingDualNetFactory;
struct ModelBatcher;

class BatchingDualNet : public DualNet {
 public:
  BatchingDualNet(BatchingDualNetFactory* factory,
                  std::shared_ptr<ModelBatcher> batcher);

  void RunMany(std::vector<const BoardFeatures*> features,
               std::vector<Output*> outputs, std::string* model) override;
  InputLayout GetInputLayout() const override;

  void SetOtherBatcher(std::shared_ptr<ModelBatcher> batcher);

  // protected:
  BatchingDualNetFactory* factory_;
  std::shared_ptr<ModelBatcher> batcher_;
  std::shared_ptr<ModelBatcher> other_batcher_ = nullptr;
};

class BatchingDualNetFactory : public DualNetFactory {
 public:
  BatchingDualNetFactory(std::unique_ptr<DualNetFactory> factory_impl,
                         size_t batch_size);
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

  const size_t batch_size_;
};

struct InferenceData {
  std::vector<const DualNet::BoardFeatures*> features;
  std::vector<DualNet::Output*> outputs;
  std::string* model_name;
  absl::Notification* notification;
};

struct ModelBatcher {
  // TODO(tommadams): remove model_path and add a DualNet::path() method.
  ModelBatcher(std::string model_path, std::unique_ptr<DualNet> model_impl,
               size_t batch_size)
      : model_impl(std::move(model_impl)),
        batch_size(batch_size),
        model_path(std::move(model_path)) {}

  void StartGame() LOCKS_EXCLUDED(&mutex) {
    absl::MutexLock lock(&mutex);
    num_active_games += 1;
  }

  void EndGame() LOCKS_EXCLUDED(&mutex) {
    absl::MutexLock lock(&mutex);
    num_active_games -= 1;
    MaybeRunBatchesLocked();
  }

  void MaybeRunBatches() LOCKS_EXCLUDED(&mutex) {
    absl::MutexLock lock(&mutex);
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
      mutex.Lock();
      inference_queue.push(
          {std::move(features), std::move(outputs), model_name, &notification});
      if (other_batcher != nullptr) {
        if (other_batcher == this) {
          num_waiting += 1;
          mutex.Unlock();
        } else {
          mutex.Unlock();
          absl::MutexLock other_lock(&other_batcher->mutex);
          other_batcher->num_waiting += 1;
          other_batcher->MaybeRunBatchesLocked();
        }
      }
      MaybeRunBatches();
    }

    notification.WaitForNotification();

    {
      mutex.Lock();
      if (other_batcher != nullptr) {
        if (other_batcher == this) {
          num_waiting -= 1;
          mutex.Unlock();
        } else {
          mutex.Unlock();
          absl::MutexLock other_lock(&other_batcher->mutex);
          other_batcher->num_waiting -= 1;
        }
      }
    }
  }

  // TODO(tommadams): make this private
  void MaybeRunBatchesLocked() EXCLUSIVE_LOCKS_REQUIRED(&mutex) {
    while (inference_queue.size() >= batch_size ||
           inference_queue.size() + num_waiting >= num_active_games) {
      auto str = absl::StrCat(
          "### ", absl::StrFormat("%x", gettid()), "  ", file::Stem(model_path),
          "  iq:", inference_queue.size(), "  bs:", batch_size,
          "  nw:", num_waiting, "  ng:", num_active_games);
      if (inference_queue.empty()) {
        std::cerr << absl::StrCat(str, "  NO\n");
        return;
      }
      std::cerr << absl::StrCat(str, "  YES\n");
      RunBatch();
    }
  }

  void RunBatch() EXCLUSIVE_LOCKS_REQUIRED(&mutex) {
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

    // Unlock the mutex while running inference.
    mutex.Unlock();

    std::vector<std::string> parts = {"### BATCH",
                                      absl::StrFormat("%x", gettid())};
    std::string model_name;
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
  const size_t batch_size;
  std::queue<InferenceData> inference_queue GUARDED_BY(&mutex);
  size_t num_waiting = 0;
  size_t num_active_games = 0 GUARDED_BY(&mutex);
  const std::string model_path;
};

BatchingDualNet::BatchingDualNet(BatchingDualNetFactory* factory,
                                 std::shared_ptr<ModelBatcher> batcher)
    : factory_(factory), batcher_(std::move(batcher)) {}

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
    std::unique_ptr<DualNetFactory> factory_impl, size_t batch_size)
    : factory_impl_(std::move(factory_impl)), batch_size_(batch_size) {}

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
        model_path, factory_impl_->NewDualNet(model_path), batch_size_);
    it = batchers_.emplace(model_path, std::move(batcher)).first;
  }

  auto result = absl::make_unique<BatchingDualNet>(this, it->second);

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

  if (b->batcher_ != w->batcher_) {
    b->SetOtherBatcher(w->batcher_);
    w->SetOtherBatcher(b->batcher_);
  }
  b->batcher_->StartGame();
  w->batcher_->StartGame();
}

void BatchingDualNetFactory::EndGame(DualNet* black, DualNet* white) {
  auto* b = dynamic_cast<BatchingDualNet*>(black);
  auto* w = dynamic_cast<BatchingDualNet*>(white);
  MG_CHECK(b != nullptr && w != nullptr);

  if (b->batcher_ != w->batcher_) {
    b->SetOtherBatcher(nullptr);
    w->SetOtherBatcher(nullptr);
  }
  b->batcher_->EndGame();
  w->batcher_->EndGame();
}

}  // namespace

std::unique_ptr<DualNetFactory> NewBatchingDualNetFactory(
    std::unique_ptr<DualNetFactory> impl, size_t batch_size) {
  return absl::make_unique<BatchingDualNetFactory>(std::move(impl), batch_size);
}

}  // namespace minigo
