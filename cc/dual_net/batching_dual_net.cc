#include "cc/dual_net/batching_dual_net.h"

#include <queue>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "cc/check.h"

namespace minigo {
namespace {
class BatchingService {
  struct InferenceData {
    std::vector<const DualNet::BoardFeatures*> features;
    std::vector<DualNet::Output*> outputs;
    std::string* model;
    absl::Notification* notification;
  };

 public:
  BatchingService(std::unique_ptr<DualNet> dual_net, size_t batch_size)
      : dual_net_(std::move(dual_net)),
        num_clients_(0),
        queue_counter_(0),
        run_counter_(0),
        num_runs_(0),
        batch_size_(batch_size) {
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
    MaybeRunBatches();
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

      MaybeRunBatches();
    }

    notification.WaitForNotification();
  }

  DualNet::InputLayout GetInputLayout() const {
    return dual_net_->GetInputLayout();
  }

 private:
  void MaybeRunBatches() EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
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
};

class BatchingDualNet : public DualNet {
 public:
  explicit BatchingDualNet(BatchingService* service) : service_(service) {
    service_->IncrementClientCount();
  }

  ~BatchingDualNet() override { service_->DecrementClientCount(); }

  void RunMany(std::vector<const BoardFeatures*> features,
               std::vector<Output*> outputs, std::string* model) override {
    service_->RunMany(std::move(features), std::move(outputs), model);
  };

  InputLayout GetInputLayout() const override {
    return service_->GetInputLayout();
  }

 protected:
  BatchingService* service_;
};

class BatchingFactory : public DualNetFactory {
 public:
  BatchingFactory(std::unique_ptr<DualNetFactory> impl, size_t batch_size)
      : impl_(std::move(impl)), batch_size_(batch_size) {}

  int GetBufferCount() const override { return impl_->GetBufferCount(); }

  std::unique_ptr<DualNet> NewDualNet(const std::string& model) override {
    absl::MutexLock lock(&mutex_);

    // Find or create a service for the requested model.
    auto it = services_.find(model);
    if (it == services_.end()) {
      std::unique_ptr<BatchingService> service;
      if (model == cached_model_) {
        service = std::move(cached_service_);
        cached_model_ = "";
      } else {
        service = absl::make_unique<BatchingService>(impl_->NewDualNet(model),
                                                     batch_size_);
      }
      it = services_.emplace(model, std::move(service)).first;
    }

    // Create a new client of the service.
    auto dual_net = absl::make_unique<BatchingDualNet>(it->second.get());

    // Take this opportunity to delete any services that have no clients.
    it = services_.begin();
    while (it != services_.end()) {
      if (it->second->num_clients() == 0) {
        cached_model_ = it->first;
        cached_service_ = std::move(it->second);
        services_.erase(it++);
      } else {
        ++it;
      }
    }

    return dual_net;
  }

 private:
  absl::Mutex mutex_;
  std::unique_ptr<DualNetFactory> impl_;

  // Map from model to BatchingService for that model. Once a service no longer
  // has clients, it's model to the cached_service_ and finally deleted.
  absl::flat_hash_map<std::string, std::unique_ptr<BatchingService>> services_
      GUARDED_BY(&mutex_);

  // The most recent BatchingService that no longer has clients. We don't delete
  // a service that no longer has clients immediately, because a common pattern
  // is to delete a DualNet, then create a new one with the same model.
  std::string cached_model_;
  std::unique_ptr<BatchingService> cached_service_;

  const size_t batch_size_;
};

}  // namespace

std::unique_ptr<DualNetFactory> NewBatchingDualNetFactory(
    std::unique_ptr<DualNetFactory> impl, size_t batch_size) {
  return absl::make_unique<BatchingFactory>(std::move(impl), batch_size);
}

}  // namespace minigo
