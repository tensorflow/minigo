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

#ifndef CC_THREAD_SAFE_QUEUE_H_
#define CC_THREAD_SAFE_QUEUE_H_

#include <queue>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

namespace minigo {

template <typename T>
class ThreadSafeQueue {
 public:
  void Push(const T& x) {
    absl::MutexLock lock(&m_);
    queue_.push(x);
  }

  void Push(T&& x) {
    absl::MutexLock lock(&m_);
    queue_.push(std::move(x));
  }

  bool TryPop(T* x) {
    absl::MutexLock lock(&m_);
    if (queue_.empty()) {
      return false;
    }
    *x = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  T Pop() {
    absl::MutexLock lock(&m_);
    m_.Await(absl::Condition(this, &ThreadSafeQueue::has_elements));
    T x = std::move(queue_.front());
    queue_.pop();
    return x;
  }

  bool PopWithTimeout(T* x, absl::Duration timeout) {
    absl::MutexLock lock(&m_);
    m_.AwaitWithTimeout(absl::Condition(this, &ThreadSafeQueue::has_elements),
                        timeout);
    if (queue_.empty()) {
      return false;
    }
    *x = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  bool empty() const {
    absl::MutexLock lock(&m_);
    return queue_.empty();
  }

  // private:
  bool has_elements() const EXCLUSIVE_LOCKS_REQUIRED(&m_) {
    return !queue_.empty();
  }

  mutable absl::Mutex m_;
  std::queue<T> queue_ GUARDED_BY(&m_);
};

}  // namespace minigo

#endif  // CC_THREAD_SAFE_QUEUE_H_
