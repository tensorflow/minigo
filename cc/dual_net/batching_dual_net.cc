#include "cc/dual_net/batching_dual_net.h"

#include <future>
#include <queue>

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
        batch_size_(batch_size) {}

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

  absl::Mutex mutex_;

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
  BatchingFactory(std::unique_ptr<DualNet> dual_net, size_t batch_size)
      : service_(std::move(dual_net), batch_size) {}

 private:
  std::unique_ptr<DualNet> New() override {
    return absl::make_unique<BatchingDualNet>(&service_);
  }

  BatchingService service_;
};
}  // namespace

std::unique_ptr<DualNetFactory> NewBatchingFactory(
    std::unique_ptr<DualNet> dual_net, size_t batch_size) {
  return absl::make_unique<BatchingFactory>(std::move(dual_net), batch_size);
}
}  // namespace minigo
