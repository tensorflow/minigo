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

#include <functional>
#include <thread>

namespace minigo {

class Thread {
 public:
  virtual ~Thread();

  Thread() = default;
  Thread(Thread&&) = default;
  Thread& operator=(Thread&&) = default;

  std::thread::native_handle_type handle() { return impl_.native_handle(); }

  void Start();
  void Join();

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

}  // namespace minigo
