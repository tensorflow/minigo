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

#ifndef CC_LOGGING_H_
#define CC_LOGGING_H_

#include <sstream>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"

namespace minigo {
namespace internal {

enum class LogLevel {
  INFO,
  WARNING,
  ERROR,
  FATAL,
};

// A simple thread-safe logging stream that replaces logging directly to
// stderr, which is not thread-safe.
// All logging is written to stderr.
// For log levels other than INFO, the line is prefixed with the log level and
// source code location of the message.
class LogStream {
 public:
  LogStream(const char* file, int line, LogLevel level);
  ~LogStream();

  template <typename T>
  LogStream& operator<<(const T& t) {
    stream_ << t;
    return *this;
  }

 private:
  std::stringstream stream_;
  LogLevel level_;
};

void ABSL_ATTRIBUTE_NOINLINE CheckFail(const char* cond, const char* file,
                                       int line);

class CheckFailStream {
 public:
  CheckFailStream(const char* cond, const char* file, int line);

  template <typename T>
  CheckFailStream& operator<<(const T& t) {
    impl_ << t;
    return *this;
  }

  operator bool() { return true; }

 private:
  LogStream impl_;
};

}  // namespace internal
}  // namespace minigo

#define MG_LOG(level)                               \
  ::minigo::internal::LogStream(__FILE__, __LINE__, \
                                ::minigo::internal::LogLevel::level)

// MG_CHECK(cond) and MG_DCHECK(cond) halt the program, printing the current
// the given condition `cond` is not true. MG_CHECK is always enabled, MG_DCHECK
// is only enabled for debug builds (i.e. when NDEBUG is not defined).
#define MG_CHECK(cond) \
  (cond)               \
      ? false          \
      : true & ::minigo::internal::CheckFailStream(#cond, __FILE__, __LINE__)

#ifndef NDEBUG
#define MG_DCHECK MG_CHECK
#else
#define MG_DCHECK(cond) MG_CHECK(true)
#endif

#endif  // CC_LOGGING_H_
