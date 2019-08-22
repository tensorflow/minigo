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

#include "cc/dual_net/batching_dual_net.h"

#include <utility>

#include "absl/memory/memory.h"
#include "absl/time/clock.h"
#include "cc/logging.h"
#include "wtf/macros.h"

namespace minigo {

namespace internal {

ModelBatcher::ModelBatcher(std::unique_ptr<Model> model_impl)
    : model_impl_(std::move(model_impl)), stats_(model_impl_->buffer_count()) {}

ModelBatcher::~ModelBatcher() {
  MG_LOG(INFO) << "Ran " << num_batches_ << " batches with an average size of "
               << static_cast<float>(num_inferences_) / num_batches_;
}

void ModelBatcher::StartGame() {
  absl::MutexLock lock(&mutex_);
  num_active_clients_ += 1;
}

void ModelBatcher::EndGame() {
  absl::MutexLock lock(&mutex_);
  num_active_clients_ -= 1;
  MaybeRunBatchesLocked();
}

void ModelBatcher::RunMany(ModelBatcher* other_batcher,
                           const std::vector<const Model::Input*>& inputs,
                           std::vector<Model::Output*>* outputs,
                           std::string* model_name) {
  WTF_SCOPE("ModelBatcher::RunMany", size_t)(inputs.size());

  absl::Notification notification;

  {
    absl::MutexLock lock(&mutex_);
    queue_.push({other_batcher, &inputs, outputs, model_name, &notification});
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

BatchingModelStats ModelBatcher::FlushStats() {
  mutex_.Lock();
  auto result = stats_;
  stats_ = BatchingModelStats(model_impl_->buffer_count());
  mutex_.Unlock();
  return result;
}

size_t ModelBatcher::GetBatchSize() const {
  return std::max<size_t>(1, num_active_clients_ / model_impl_->buffer_count());
}

void ModelBatcher::MaybeRunBatchesLocked() {
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

void ModelBatcher::RunBatch() {
  WTF_SCOPE0("ModelBatcher::RunBatch");
  auto run_batch_start_time = absl::Now();

  auto batch_size = GetBatchSize();

  // TODO(tommadams): reserve GetBatchSize() * virtual_losses elements.
  std::vector<const Model::Input*> inputs;
  std::vector<Model::Output*> outputs;
  std::vector<InferenceRequest> inferences;

  while (!queue_.empty() && inferences.size() < batch_size) {
    auto& inference = queue_.front();
    size_t num_features = inference.inputs->size();

    std::copy_n(inference.inputs->begin(), num_features,
                std::back_inserter(inputs));
    std::copy_n(inference.outputs->begin(), num_features,
                std::back_inserter(outputs));
    inferences.push_back(inference);

    queue_.pop();
  }

  num_batches_ += 1;
  num_inferences_ += inputs.size();
  auto num_inferences_in_batch = inputs.size();

  // Unlock the mutex while running inference. This allows more inferences
  // to be enqueued while inference is running.
  mutex_.Unlock();

  std::string model_name;
  auto run_many_start_time = absl::Now();

  MG_CHECK(inputs.size() == outputs.size());
  model_impl_->RunMany(inputs, &outputs, &model_name);
  auto run_many_time = absl::Now() - run_many_start_time;

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

  stats_.run_batch_time +=
      (absl::Now() - run_batch_start_time) / model_impl_->buffer_count();
  stats_.run_many_time += (run_many_time) / model_impl_->buffer_count();
  stats_.num_inferences += num_inferences_in_batch;
}

}  // namespace internal

BatchingModel::BatchingModel(std::shared_ptr<internal::ModelBatcher> batcher)
    : Model(batcher->name(), 1), batcher_(std::move(batcher)) {}

void BatchingModel::RunMany(const std::vector<const Model::Input*>& inputs,
                            std::vector<Output*>* outputs, std::string* model) {
  batcher_->RunMany(other_batcher_.get(), inputs, outputs, model);
}

void BatchingModel::StartGame() { batcher_->StartGame(); }

void BatchingModel::EndGame() { batcher_->EndGame(); }

void BatchingModel::SetOther(BatchingModel* other) {
  if (other == nullptr) {
    MG_CHECK(other_batcher_ != nullptr);
    other_batcher_ = nullptr;
  } else {
    MG_CHECK(other_batcher_ == nullptr);
    other_batcher_ = other->batcher_;
  }
}

BatchingModelFactory::BatchingModelFactory(
    std::unique_ptr<ModelFactory> factory_impl)
    : factory_impl_(std::move(factory_impl)) {}

std::unique_ptr<Model> BatchingModelFactory::NewModel(
    const std::string& descriptor) {
  absl::MutexLock lock(&mutex_);

  // Find or create a service for the requested model.
  auto it = batchers_.find(descriptor);
  if (it == batchers_.end()) {
    auto batcher = std::make_shared<internal::ModelBatcher>(
        factory_impl_->NewModel(descriptor));
    it = batchers_.emplace(descriptor, std::move(batcher)).first;
  }

  auto model = absl::make_unique<BatchingModel>(it->second);

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

void BatchingModelFactory::StartGame(Model* black, Model* white) {
  // TODO(tommadams): figure out if we can refactor the code somehow to take
  // BatchingModel pointers and avoid these dynamic_casts.
  auto* b = dynamic_cast<BatchingModel*>(black);
  auto* w = dynamic_cast<BatchingModel*>(white);
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

void BatchingModelFactory::EndGame(Model* black, Model* white) {
  // TODO(tommadams): figure out if we can refactor the code somehow to take
  // BatchingModel pointers and avoid these dynamic_casts.
  auto* b = dynamic_cast<BatchingModel*>(black);
  auto* w = dynamic_cast<BatchingModel*>(white);
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

std::vector<std::pair<std::string, BatchingModelStats>>
BatchingModelFactory::FlushStats() {
  absl::MutexLock lock(&mutex_);
  std::vector<std::pair<std::string, BatchingModelStats>> result;
  for (const auto& kv : batchers_) {
    result.emplace_back(kv.first, kv.second->FlushStats());
  }
  return result;
}

}  // namespace minigo
