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

void ShardedExecutor::Execute(std::function<void(int, int)> fn) {
  WTF_SCOPE0("ShardedExecutor::Execute");
  auto num_shards = threads_.size() + 1;

  // Process shards 1 to num_shards - 1 on worker threads.
  // We don't need to lock the mutex if there are no background threads.
  if (!threads_.empty()) {
    mutex_.Lock();
    for (auto& t : threads_) {
      t->Execute(&fn);
    }
  }

  // Process shard 0 on the caller's thread.
  fn(0, num_shards);

  // Wait for the background threads to finish.
  if (!threads_.empty()) {
    for (auto& t : threads_) {
      t->Wait();
    }
    mutex_.Unlock();
  }
}

ShardedExecutor::WorkerThread::WorkerThread(int shard, int num_shards)
    : shard_(shard), num_shards_(num_shards) {}

void ShardedExecutor::WorkerThread::Execute(std::function<void(int, int)>* fn) {
  fn_ = fn;
  ready_sem_.Post();
}

void ShardedExecutor::WorkerThread::Wait() {
  done_sem_.Wait();
  fn_ = nullptr;
}

void ShardedExecutor::WorkerThread::Join() {
  running_ = false;
  ready_sem_.Post();
  Thread::Join();
}

void ShardedExecutor::WorkerThread::Run() {
  WTF_THREAD_ENABLE("ShardedExecutor");

  for (;;) {
    ready_sem_.Wait();
    if (!running_) {
      break;
    }
    {
      WTF_SCOPE0("ShardedExecutor::Run");
      (*fn_)(shard_, num_shards_);
    }
    done_sem_.Post();
  }
}

} // namespace minigo
