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

#include "cc/model/batching_model.h"

#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "cc/model/buffered_model.h"
#include "cc/model/model.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace minigo {
namespace {

class WaitingModel;
class WaitingModelFactory;

struct EvaluatedBatch {
  EvaluatedBatch() = default;
  EvaluatedBatch(std::string model_descriptor, size_t size)
      : model_descriptor(std::move(model_descriptor)), size(size) {}
  std::string model_descriptor;
  size_t size = 0;
};

// Why doesn't absl have this already?
class Semaphore {
 public:
  void Post() {
    absl::MutexLock lock(&mutex_);
    ++count_;
    cond_var_.Signal();
  }

  void Wait() {
    absl::MutexLock lock(&mutex_);
    while (count_ == 0) {
      cond_var_.Wait(&mutex_);
    }
    --count_;
  }

 private:
  absl::Mutex mutex_;
  absl::CondVar cond_var_;
  int count_ = 0;
};

// Model implementation whose RunMany method blocks until Notify is called.
// This is used in tests where multiple BatchingModel clients are running in
// parallel and we want to control the evaluation order of the implementation
// models.
// WaitingModel also records each RunMany call with its factory.
class WaitingModel : public Model {
 public:
  WaitingModel(WaitingModelFactory* factory, std::string model,
               int buffer_count);

  // Blocks until Notify is called.
  // Each call pushes an EvaluatedBatch instance onto the factory's queue
  // containing the model name and size of the batch.
  void RunMany(const std::vector<const ModelInput*>& inputs,
               std::vector<ModelOutput*>* outputs,
               std::string* model_name) override;

  // Allows one call to RunMany to complete.
  // Blocks until the executed RunMany call completes.
  void Notify();

 private:
  Semaphore before_;
  Semaphore after_;
  WaitingModelFactory* factory_;
  std::string model_name_;
};

class WaitingModelFactory : public ModelFactory {
 public:
  explicit WaitingModelFactory(int buffer_count);

  std::unique_ptr<Model> NewModel(const std::string& descriptor) override;

  // Notifies the batcher for the given model_descriptor, so that it can run a
  // single batch of inferences.
  // CHECK fails if batches_ is empty.
  // CHECK fails if the front element in the batches_ queue doesn't match the
  // given model_descriptor or expected_batch_size.
  void FlushBatch(const std::string& model_descriptor,
                  size_t expected_batch_size);

  // Called by WaitingModel::RunMany.
  void PushEvaluatedBatch(const std::string& model, size_t size);

 private:
  const int buffer_count_;
  mutable absl::Mutex mutex_;
  absl::flat_hash_map<std::string, WaitingModel*> models_ GUARDED_BY(&mutex_);
  std::queue<EvaluatedBatch> batches_ GUARDED_BY(&mutex_);
};

WaitingModel::WaitingModel(WaitingModelFactory* factory, std::string model_name,
                           int buffer_count)
    : Model("Waiting", FeatureDescriptor::Create<AgzFeatures>(), buffer_count),
      factory_(factory),
      model_name_(std::move(model_name)) {}

void WaitingModel::RunMany(const std::vector<const ModelInput*>& inputs,
                           std::vector<ModelOutput*>* outputs,
                           std::string* model_name) {
  before_.Wait();
  factory_->PushEvaluatedBatch(model_name_, inputs.size());
  if (model_name != nullptr) {
    *model_name = model_name_;
  }
  after_.Post();
}

void WaitingModel::Notify() {
  before_.Post();
  after_.Wait();
}

WaitingModelFactory::WaitingModelFactory(int buffer_count)
    : buffer_count_(buffer_count) {}

std::unique_ptr<Model> WaitingModelFactory::NewModel(
    const std::string& descriptor) {
  absl::MutexLock lock(&mutex_);

  auto model = absl::make_unique<WaitingModel>(this, descriptor, buffer_count_);
  MG_CHECK(models_.emplace(descriptor, model.get()).second);
  return model;
}

void WaitingModelFactory::FlushBatch(const std::string& model_descriptor,
                                     size_t expected_batch_size) {
  // Find the model for the given descriptor.
  WaitingModel* model;
  {
    absl::MutexLock lock(&mutex_);
    auto it = models_.find(model_descriptor);
    MG_CHECK(it != models_.end());
    model = it->second;
  }

  // Notify it, letting it run a single batch.
  model->Notify();

  // The model should now have pushed a batch on to the batches_ queue: pop it
  // off or die trying.
  EvaluatedBatch batch;
  {
    absl::MutexLock lock(&mutex_);
    MG_CHECK(!batches_.empty());
    batch = std::move(batches_.front());
    batches_.pop();
  }

  // Check the popped batch matches the expected batch.
  MG_CHECK(batch.model_descriptor == model_descriptor);
  MG_CHECK(batch.size == expected_batch_size);
}

void WaitingModelFactory::PushEvaluatedBatch(const std::string& model,
                                             size_t size) {
  absl::MutexLock lock(&mutex_);
  batches_.emplace(model, size);
}

class BatchingModelTest : public ::testing::Test {
 protected:
  void InitFactory(int buffer_count) {
    auto impl = absl::make_unique<WaitingModelFactory>(buffer_count);
    model_factory_ = impl.get();
    batcher_ = absl::make_unique<BatchingModelFactory>(std::move(impl));
  }

  std::unique_ptr<Model> NewModel(const std::string& descriptor) {
    return batcher_->NewModel(descriptor);
  }

  void StartGame(Model* black, Model* white) {
    batcher_->StartGame(black, white);
  }

  void EndGame(Model* black, Model* white) { batcher_->EndGame(black, white); }

  void FlushBatch(const std::string& model_descriptor,
                  size_t expected_batch_size) {
    model_factory_->FlushBatch(model_descriptor, expected_batch_size);
  }

 private:
  // Owned by batcher_.
  WaitingModelFactory* model_factory_ = nullptr;
  std::unique_ptr<BatchingModelFactory> batcher_;
};

TEST_F(BatchingModelTest, SelfPlay) {
  constexpr int kNumGames = 6;

  struct Game {
    ModelInput input;
    ModelOutput output;
    std::unique_ptr<Model> model;
    std::thread thread;
  };

  // Test single, double and triple buffering.
  for (int buffer_count = 1; buffer_count <= 3; ++buffer_count) {
    InitFactory(buffer_count);
    int expected_batch_size = kNumGames / buffer_count;

    std::vector<Game> games;
    for (int i = 0; i < kNumGames; ++i) {
      Game game;
      game.model = NewModel("a");
      StartGame(game.model.get(), game.model.get());
      games.push_back(std::move(game));
    }

    for (auto& game : games) {
      game.thread = std::thread([&game] {
        std::vector<const ModelInput*> inputs = {&game.input};
        std::vector<ModelOutput*> outputs = {&game.output};
        game.model->RunMany(inputs, &outputs, nullptr);
      });
    }

    for (int i = 0; i < kNumGames / expected_batch_size; ++i) {
      FlushBatch("a", expected_batch_size);
    }

    for (auto& game : games) {
      EndGame(game.model.get(), game.model.get());
      game.thread.join();
    }
  }
}

TEST_F(BatchingModelTest, EvalDoubleBuffer) {
  constexpr int kNumGames = 6;

  struct Game {
    ModelInput input;
    ModelOutput output;
    std::unique_ptr<Model> black;
    std::unique_ptr<Model> white;
    std::thread thread;
  };

  // Test single, double and triple buffering.
  for (int buffer_count = 1; buffer_count <= 3; ++buffer_count) {
    InitFactory(buffer_count);
    int expected_batch_size = kNumGames / buffer_count;

    std::vector<Game> games;
    for (int i = 0; i < kNumGames; ++i) {
      Game game;
      game.black = NewModel("black");
      game.white = NewModel("white");
      StartGame(game.black.get(), game.white.get());
      games.push_back(std::move(game));
    }

    for (auto& game : games) {
      game.thread = std::thread([&game] {
        std::vector<const ModelInput*> inputs = {&game.input};
        std::vector<ModelOutput*> outputs = {&game.output};
        game.black->RunMany(inputs, &outputs, nullptr);
        game.white->RunMany(inputs, &outputs, nullptr);
      });
    }

    for (int i = 0; i < kNumGames / expected_batch_size; ++i) {
      FlushBatch("black", expected_batch_size);
      FlushBatch("white", expected_batch_size);
    }

    for (auto& game : games) {
      EndGame(game.black.get(), game.black.get());
      game.thread.join();
    }
  }
}

}  // namespace
}  // namespace minigo
