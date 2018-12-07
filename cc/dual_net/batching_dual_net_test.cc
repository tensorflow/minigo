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

// DO NOT CHECK IN
// DO NOT CHECK IN
// DO NOT CHECK IN
#include <sys/syscall.h>
#include <unistd.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#define gettid() syscall(SYS_gettid)
// DO NOT CHECK IN
// DO NOT CHECK IN
// DO NOT CHECK IN
//
namespace minigo {
namespace {

class WaitingDualNet;
class WaitingDualNetFactory;

template <typename Format, typename... Args>
void Log(Format format, const Args&... args) {
  std::cerr << absl::StrCat(absl::StrFormat("%x ", gettid()),
                            absl::StrFormat(format, args...), "\n");
}

struct EvaluatedBatch {
  EvaluatedBatch(std::string model, size_t size)
      : model(std::move(model)), size(size) {}
  std::string model;
  size_t size;
};

bool operator==(const EvaluatedBatch& a, const EvaluatedBatch& b) {
  return a.model == b.model && a.size == b.size;
}

std::ostream& operator<<(std::ostream& os, const EvaluatedBatch& x) {
  return os << "{" << x.model << ", " << x.size << "}";
}

// Why doesn't absl have this already?
class Semaphore {
 public:
  void Post() {
    absl::MutexLock lock(&mutex_);
    ++count_;
    // Log("SEM Post %p %d", this, count_);
    cond_var_.Signal();
  }

  void Wait() {
    absl::MutexLock lock(&mutex_);
    // Log("SEM Wait %p %d", this, count_);
    while (count_ == 0) {
      // Log("SEM Waiting %p", this);
      cond_var_.Wait(&mutex_);
    }
    --count_;
    // Log("SEM Got %p %d", this, count_);
  }

 private:
  absl::Mutex mutex_;
  absl::CondVar cond_var_;
  int count_ = 0;
};

// DualNet implementation whos RunMany method blocks until Notify is called.
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

  // Notify the given model that it should run eval.
  void Notify(const std::string& model_path) const;

  void PushEvaluatedBatch(const std::string& model, size_t size);
  EvaluatedBatch PopEvaluatedBatch();

 private:
  const int buffer_count_;
  mutable absl::Mutex mutex_;
  absl::flat_hash_map<std::string, WaitingDualNet*> models_ GUARDED_BY(&mutex_);
  std::queue<EvaluatedBatch> batches_ GUARDED_BY(&mutex_);
};

WaitingDualNet::WaitingDualNet(WaitingDualNetFactory* factory,
                               std::string model)
    : factory_(factory), model_(std::move(model)) {}

void WaitingDualNet::RunMany(std::vector<const BoardFeatures*> features,
                             std::vector<Output*> outputs, std::string* model) {
  // Log("RunMany WAIT %p", &before_);
  before_.Wait();
  // Log("RunMany WAITED %p", &before_);
  factory_->PushEvaluatedBatch(model_, features.size());
  if (model != nullptr) {
    *model = model_;
  }
  // Log("RunMany POST %p", &after_);
  after_.Post();
  // Log("RunMany DONE");
}

void WaitingDualNet::Notify() {
  // Log("Notify POST %p", &before_);
  before_.Post();
  // Log("Notify WAIT %p", &after_);
  after_.Wait();
  // Log("Notify DONE");
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

void WaitingDualNetFactory::Notify(const std::string& model_path) const {
  WaitingDualNet* model;
  {
    absl::MutexLock lock(&mutex_);
    auto it = models_.find(model_path);
    MG_CHECK(it != models_.end());
    model = it->second;
  }
  model->Notify();
}

void WaitingDualNetFactory::PushEvaluatedBatch(const std::string& model,
                                               size_t size) {
  absl::MutexLock lock(&mutex_);
  Log("BATCH %s %d", model, (int)size);
  batches_.emplace(model, size);
}

EvaluatedBatch WaitingDualNetFactory::PopEvaluatedBatch() {
  absl::MutexLock lock(&mutex_);
  MG_CHECK(!batches_.empty());
  auto batch = std::move(batches_.front());
  batches_.pop();
  return batch;
}

class BatchingDualNetTest : public ::testing::Test {
 protected:
  void InitFactory(int buffer_count) {
    auto impl = absl::make_unique<WaitingDualNetFactory>(buffer_count);
    waiting_factory_ = impl.get();
    batching_factory_ = NewBatchingDualNetFactory(std::move(impl));
  }

  std::unique_ptr<DualNet> NewDualNet(const std::string& model_path) {
    return batching_factory_->NewDualNet(model_path);
  }

  void StartGame(DualNet* black, DualNet* white) {
    batching_factory_->StartGame(black, white);
  }

  void EndGame(DualNet* black, DualNet* white) {
    batching_factory_->EndGame(black, white);
  }

  void Notify(const std::string& model_path) const {
    waiting_factory_->Notify(model_path);
  }

  EvaluatedBatch PopEvaluatedBatch() {
    return waiting_factory_->PopEvaluatedBatch();
  }

 private:
  // Owned by batching_factory_.
  WaitingDualNetFactory* waiting_factory_ = nullptr;
  std::unique_ptr<DualNetFactory> batching_factory_;
};

// Simulate self play, where one player plays each game.
// Batching is configured to use a single buffer.
TEST_F(BatchingDualNetTest, SelfPlaySingleBuffer) {
  InitFactory(1);

  struct Game {
    std::unique_ptr<DualNet> model;
    std::thread thread;
  };

  constexpr int kNumGames = 4;
  constexpr int kNumReads = 2;

  std::vector<Game> games;
  for (int i = 0; i < kNumGames; ++i) {
    Game game;
    game.model = NewDualNet("a");
    StartGame(game.model.get(), game.model.get());
    games.push_back(std::move(game));
  }

  for (auto& game : games) {
    game.thread = std::thread([this, &game] {
      DualNet::BoardFeatures features;
      DualNet::Output output;
      for (int i = 0; i < kNumReads; ++i) {
        Log("RunMany(%d)", i);
        game.model->RunMany({&features}, {&output}, nullptr);
      }
      EndGame(game.model.get(), game.model.get());
      Log("END");
    });
  }

  for (int i = 0; i < kNumReads; ++i) {
    Notify("a");
    EXPECT_EQ(EvaluatedBatch("a", 4), PopEvaluatedBatch());
  }

  Log("JOINING");
  for (auto& game : games) {
    game.thread.join();
  }
  Log("JOINED");
}

}  // namespace
}  // namespace minigo
