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

#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "cc/dual_net/dual_net.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace minigo {
namespace {

class WaitingDualNet;
class WaitingDualNetFactory;

struct EvaluatedBatch {
  EvaluatedBatch() = default;
  EvaluatedBatch(std::string model_path, size_t size)
      : model_path(std::move(model_path)), size(size) {}
  std::string model_path;
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

// DualNet implementation whose RunMany method blocks until Notify is called.
// This is used in tests where multiple BatchingDualNet clients are running in
// parallel and we want to control the evaluation order of the implementation
// models.
// WaitingDualNet also records each RunMany call with its factory.
class WaitingDualNet : public DualNet {
 public:
  WaitingDualNet(WaitingDualNetFactory* factory, std::string model);

  // Blocks until Notify is called.
  // Each call pushes an EvaluatedBatch instance onto the factory's queue
  // containing the model name and size of the batch.
  void RunMany(std::vector<const BoardFeatures*> features,
               std::vector<Output*> outputs, std::string* model) override;

  // Allows one call to RunMany to complete.
  // Blocks until the executed RunMany call completes.
  void Notify();

 private:
  Semaphore before_;
  Semaphore after_;
  WaitingDualNetFactory* factory_;
  std::string model_;
};

class WaitingDualNetFactory : public DualNetFactory {
 public:
  explicit WaitingDualNetFactory(int buffer_count);

  int GetBufferCount() const override;
  std::unique_ptr<DualNet> NewDualNet(const std::string& model_path) override;

  // Notifies the batcher for the given model_path, so that it can run a single
  // batch of inferences.
  // CHECK fails if batches_ is empty.
  // CHECK fails if the front element in the batches_ queue doesn't match the
  // given model_path or expected_batch_size.
  void FlushBatch(const std::string& model_path, size_t expected_batch_size);

  // Called by WaitingDualNet::RunMany.
  void PushEvaluatedBatch(const std::string& model, size_t size);

 private:
  const int buffer_count_;
  mutable absl::Mutex mutex_;
  absl::flat_hash_map<std::string, WaitingDualNet*> models_ GUARDED_BY(&mutex_);
  std::queue<EvaluatedBatch> batches_ GUARDED_BY(&mutex_);
};

WaitingDualNet::WaitingDualNet(WaitingDualNetFactory* factory,
                               std::string model)
    : DualNet("Waiting"), factory_(factory), model_(std::move(model)) {}

void WaitingDualNet::RunMany(std::vector<const BoardFeatures*> features,
                             std::vector<Output*> outputs, std::string* model) {
  before_.Wait();
  factory_->PushEvaluatedBatch(model_, features.size());
  if (model != nullptr) {
    *model = model_;
  }
  after_.Post();
}

void WaitingDualNet::Notify() {
  before_.Post();
  after_.Wait();
}

WaitingDualNetFactory::WaitingDualNetFactory(int buffer_count)
    : buffer_count_(buffer_count) {}

int WaitingDualNetFactory::GetBufferCount() const { return buffer_count_; }

std::unique_ptr<DualNet> WaitingDualNetFactory::NewDualNet(
    const std::string& model_path) {
  absl::MutexLock lock(&mutex_);
  auto model = absl::make_unique<WaitingDualNet>(this, model_path);
  MG_CHECK(models_.emplace(model_path, model.get()).second);
  return model;
}

void WaitingDualNetFactory::FlushBatch(const std::string& model_path,
                                       size_t expected_batch_size) {
  // Find the model for the given model_path.
  WaitingDualNet* model;
  {
    absl::MutexLock lock(&mutex_);
    auto it = models_.find(model_path);
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
  MG_CHECK(batch.model_path == model_path);
  MG_CHECK(batch.size == expected_batch_size);
}

void WaitingDualNetFactory::PushEvaluatedBatch(const std::string& model,
                                               size_t size) {
  absl::MutexLock lock(&mutex_);
  batches_.emplace(model, size);
}

class BatchingDualNetTest : public ::testing::Test {
 protected:
  void InitFactory(int buffer_count) {
    auto impl = absl::make_unique<WaitingDualNetFactory>(buffer_count);
    model_factory_ = impl.get();
    batcher_ = absl::make_unique<BatchingDualNetFactory>(std::move(impl));
  }

  std::unique_ptr<DualNet> NewDualNet(const std::string& model_path) {
    return batcher_->NewDualNet(model_path);
  }

  void StartGame(DualNet* black, DualNet* white) {
    batcher_->StartGame(black, white);
  }

  void EndGame(DualNet* black, DualNet* white) {
    batcher_->EndGame(black, white);
  }

  void FlushBatch(const std::string& model_path, size_t expected_batch_size) {
    model_factory_->FlushBatch(model_path, expected_batch_size);
  }

 private:
  // Owned by batcher_.
  WaitingDualNetFactory* model_factory_ = nullptr;
  std::unique_ptr<BatchingDualNetFactory> batcher_;
};

TEST_F(BatchingDualNetTest, SelfPlay) {
  constexpr int kNumGames = 6;

  struct Game {
    DualNet::BoardFeatures features;
    DualNet::Output output;
    std::unique_ptr<DualNet> model;
    std::thread thread;
  };

  // Test single, double and triple buffering.
  for (int buffer_count = 1; buffer_count <= 3; ++buffer_count) {
    InitFactory(buffer_count);
    int expected_batch_size = kNumGames / buffer_count;

    std::vector<Game> games;
    for (int i = 0; i < kNumGames; ++i) {
      Game game;
      game.model = NewDualNet("a");
      StartGame(game.model.get(), game.model.get());
      games.push_back(std::move(game));
    }

    for (auto& game : games) {
      game.thread = std::thread([&game] {
        game.model->RunMany({&game.features}, {&game.output}, nullptr);
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

TEST_F(BatchingDualNetTest, EvalDoubleBuffer) {
  constexpr int kNumGames = 6;

  struct Game {
    DualNet::BoardFeatures features;
    DualNet::Output output;
    std::unique_ptr<DualNet> black;
    std::unique_ptr<DualNet> white;
    std::thread thread;
  };

  // Test single, double and triple buffering.
  for (int buffer_count = 1; buffer_count <= 3; ++buffer_count) {
    InitFactory(buffer_count);
    int expected_batch_size = kNumGames / buffer_count;

    std::vector<Game> games;
    for (int i = 0; i < kNumGames; ++i) {
      Game game;
      game.black = NewDualNet("black");
      game.white = NewDualNet("white");
      StartGame(game.black.get(), game.white.get());
      games.push_back(std::move(game));
    }

    for (auto& game : games) {
      game.thread = std::thread([&game] {
        game.black->RunMany({&game.features}, {&game.output}, nullptr);
        game.white->RunMany({&game.features}, {&game.output}, nullptr);
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
