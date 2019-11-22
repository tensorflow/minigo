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

#include "cc/async/sharded_executor.h"

#include "absl/strings/str_cat.h"
#include "wtf/macros.h"

namespace minigo {

ShardedExecutor::ShardedExecutor(int num_shards) {
  threads_.reserve(num_shards - 1);
  for (int i = 1; i < num_shards; ++i) {
    threads_.push_back(absl::make_unique<WorkerThread>(i, num_shards));
  }
  for (auto& t : threads_) {
    t->Start();
  }
}

ShardedExecutor::~ShardedExecutor() {
  for (auto& t : threads_) {
    t->Join();
  }
}

void ShardedExecutor::Execute(const std::function<void(int, int)>& fn) {
  WTF_SCOPE0("ShardedExecutor::Execute");
  if (!threads_.empty()) {
    auto num_shards = threads_.size() + 1;

    // Create a blocking counter initialized to the number of threads that will
    // do work. Each thread pops a shard of work off its queue, executes the
    // function, then decrements this counter.
    absl::BlockingCounter counter(threads_.size());

    // Push some work on to each thread's queue.
    for (auto& t : threads_) {
      t->Execute(&fn, &counter);
    }

    // Execute shard 0 on the caller's thread.
    fn(0, num_shards);

    // Wait for the worker threads.
    counter.Wait();
  } else {
    fn(0, 1);
  }
}

ShardedExecutor::WorkerThread::WorkerThread(int shard, int num_shards)
    : Thread(absl::StrCat("ShardExec:", shard)),
      shard_(shard),
      num_shards_(num_shards) {}

void ShardedExecutor::WorkerThread::Execute(
    const std::function<void(int, int)>* fn, absl::BlockingCounter* counter) {
  work_queue_.Push({fn, counter});
}

void ShardedExecutor::WorkerThread::Join() {
  // Pushing null work onto the queue tells the Run loop to stop.
  work_queue_.Push({nullptr, nullptr});
  Thread::Join();
}

void ShardedExecutor::WorkerThread::Run() {
  WTF_THREAD_ENABLE("ShardedExecutor");

  for (;;) {
    auto work = work_queue_.Pop();
    if (work.fn == nullptr) {
      break;
    }
    {
      WTF_SCOPE0("ShardedExecutor::Run");
      (*work.fn)(shard_, num_shards_);
    }
    work.counter->DecrementCount();
  }
}

}  // namespace minigo
