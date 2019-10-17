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

#ifndef CC_THREAD_H_
#define CC_THREAD_H_

#include <functional>
#include <thread>

#include "absl/synchronization/mutex.h"

namespace minigo {

class Thread {
 public:
  virtual ~Thread();

  Thread() = default;
  Thread(Thread&&) = default;
  Thread& operator=(Thread&&) = default;

  std::thread::native_handle_type handle() { return impl_.native_handle(); }

  virtual void Start();
  virtual void Join();

 private:
  virtual void Run() = 0;

  std::thread impl_;
};

class LambdaThread : public Thread {
 public:
  template <typename T>
  explicit LambdaThread(T closure) : closure_(std::move(closure)) {}

  LambdaThread(LambdaThread&&) = default;
  LambdaThread& operator=(LambdaThread&&) = default;

 private:
  void Run() override;

  std::function<void()> closure_;
};

// A thread whose `Start` method blocks until the thread is running and has
// called `SignalStarted` at least once.
// This can be useful to serialize the order in which threads start.
class BlockingStartThread : public Thread {
 public:
  void Start() override {
    Thread::Start();
    absl::MutexLock lock(&mutex_);
    while (!started_) {
      cond_var_.Wait(&mutex_);
    }
  }

 protected:
  void SignalStarted() {
    absl::MutexLock lock(&mutex_);
    started_ = true;
    cond_var_.Signal();
  }

 private:
  absl::Mutex mutex_;
  absl::CondVar cond_var_;
  bool started_ = false;
};

}  // namespace minigo

#endif  // CC_THREAD_H_
