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

#ifndef CC_CHECK_H_
#define CC_CHECK_H_

#include <iostream>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"

namespace minigo {

// MG_CHECK(cond) and MG_DCHECK(cond) halt the program, printing the current
// the given condition `cond` is not true. MG_CHECK is always enabled, MG_DCHECK
// is only enabled for debug builds (i.e. when NDEBUG is not defined).
// Ideally we'd use glog's CHECK macro, which serves the same purpose.
// Unfortunately TensorFlow's public headers define conflicting macros, which
// makes depending on both TensorFlow and glog difficult, so we avoid using
// glog altogether and define our own macros.

namespace internal {

void ABSL_ATTRIBUTE_NOINLINE CheckFail(const char* cond, const char* file,
                                       int line);
class CheckFailStream {
 public:
  CheckFailStream(const char* cond, const char* file, int line)
      : cond_(cond), file_(file), line_(line) {}

  ~CheckFailStream() {
    std::cerr << std::endl;
    CheckFail(cond_, file_, line_);
  }

  template <typename T>
  CheckFailStream& operator<<(const T& t) {
    std::cerr << t;
    return *this;
  }

  operator bool() { return true; }

 private:
  const char* cond_;
  const char* file_;
  int line_;
};

}  // namespace internal

#define MG_CHECK(cond) \
  (cond) ? 0 : true & internal::CheckFailStream(#cond, __FILE__, __LINE__)

#ifndef NDEBUG
#define MG_DCHECK MG_CHECK
#else
#define MG_DCHECK(cond) MG_CHECK(true)
#endif

}  // namespace minigo

#endif  // CC_CHECK_H_
