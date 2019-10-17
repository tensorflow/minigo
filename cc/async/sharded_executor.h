// Copyright 2019 Google LLC
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

#ifndef CC_ASYNC_SHARDED_EXECUTOR_H_
#define CC_ASYNC_SHARDED_EXECUTOR_H_

#include <functional>
#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "cc/async/semaphore.h"
#include "cc/thread.h"

namespace minigo {

// Helper class for running a function over an array in parallel shards.
// A simple example for setting all elements of an array to 1:
//
// constexpr int kSize = 10000;
// constexpr int kNumShards = 4;
//
// ShardedExecutor executor(kNumShards);
// std::array<int, kSize> a;
// executor.Execute([&a](int shard, int num_shards) {
//   auto range = ShardedExecutor::GetShardRange(shard, num_shards, a.size());
//   for (int i = range.begin; i < range.end; ++i) {
//     a[i] = 1;
//   }
// });
//
// `ShardedExecutor` is thread-safe.
class ShardedExecutor {
 public:
  struct Range {
    int begin;
    int end;
  };

  // Helper function for mapping from a shard to a sub-range of elements.
  static Range GetShardRange(int shard_idx, int num_shards, int num_elements) {
    auto begin = shard_idx * num_elements / num_shards;
    auto end = (shard_idx + 1) * num_elements / num_shards;
    return { begin, end };
  }

  // If `num_shards == 1`, concurrent calls to `Execute` may be executed in
  // parallel.
  // If `num_shards > 1`, concurrent calls to `Execute` will be executed
  // sequentially.
  explicit ShardedExecutor (int num_shards);

  ~ShardedExecutor();

  // Invoke `fn` `num_shards` times.
  // The first argument to `fn` is the shard ID in the range [0, `num_shards`).
  // The second argument to `fn` is `num_shards`.
  // One invocation of `fn` happens on the calling thread; if `num_shards > 0`,
  // the remaining invocations happen in parallel on threads owned by the
  // `ShardedExecutor`.
  // Blocks until all shards of work are complete.
  void Execute(std::function<void(int, int)> fn);

 private:
  struct WorkerThread : public Thread {
    WorkerThread(int shard, int num_shards);

    void Execute(std::function<void(int, int)>* fn);
    void Wait();
    void Join() override;

   private:
    void Run() override;

    std::atomic<bool> running_{true};
    std::function<void(int, int)>* fn_;
    int shard_;
    int num_shards_;
    Semaphore ready_sem_;
    Semaphore done_sem_;
  };

  std::function<void(int, int)> fn_;
  absl::Mutex mutex_;
  std::vector<std::unique_ptr<WorkerThread>> threads_;
};

} // namespace minigo

#endif  // CC_ASYNC_WORK_SHARDED_EXECUTOR_H_
