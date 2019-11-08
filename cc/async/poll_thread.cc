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

#include "cc/async/poll_thread.h"

#include <utility>

namespace minigo {

PollThread::PollThread(std::string thread_name, absl::Duration poll_interval,
                       std::function<void()> poll_fn)
    : Thread(std::move(thread_name)),
      poll_interval_(poll_interval),
      poll_fn_(std::move(poll_fn)) {}

PollThread::~PollThread() = default;

void PollThread::Join() {
  {
    absl::MutexLock lock(&mutex_);
    is_joining_ = true;
  }
  Thread::Join();
}

void PollThread::Run() {
  absl::MutexLock lock(&mutex_);
  for (;;) {
    poll_fn_();

    // AwaitWithTimout blocks until either `poll_interval_` has elapsed or
    // `is_joining_ == true`. It returns `true` if the polling loop should
    // exit.
    if (mutex_.AwaitWithTimeout(absl::Condition(this, &PollThread::IsJoining),
                                poll_interval_)) {
      break;
    }
  }
}

bool PollThread::IsJoining() const { return is_joining_; }

}  // namespace minigo

