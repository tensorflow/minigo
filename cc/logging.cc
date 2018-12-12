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

#include "cc/logging.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "absl/debugging/stacktrace.h"
#include "absl/debugging/symbolize.h"
#include "absl/synchronization/mutex.h"

namespace minigo {
namespace internal {
namespace {
absl::Mutex* mutex() {
  static auto* m = new absl::Mutex();
  return m;
}
}  // namespace

// A simple thread-safe logging stream that replaces logging directly to
// stderr, which is not thread-safe.
// All logging is written to stderr.
// For log levels other than INFO, the line is prefixed with the log level and
// source code location of the message.
LogStream::LogStream(const char* file, int line, LogLevel level)
    : line_(line), level_(level) {
  char c;
  switch (level) {
    case LogLevel::INFO:
      c = 'I';
      break;
    case LogLevel::WARNING:
      c = 'W';
      break;
    case LogLevel::ERROR:
      c = 'E';
      break;
    case LogLevel::FATAL:
      c = 'F';
      break;
    default:
      c = 'U';
      break;
  }
  file_ = std::strrchr(file, '/');
  if (file_ == nullptr) {
    file_ = std::strrchr(file, '\\');
  }
  if (file_ == nullptr) {
    file_ = file;
  } else {
    file_ += 1;
  }

  // We don't add a prefix to MG_LOG(INFO) log lines because many things rely
  // on the exact string being printed (GTP, correct formatting of position &
  // node descriptions, etc).
  if (level != LogLevel::INFO) {
    *this << '[' << c << "] " << file_ << ':' << line_ << " : ";
  }
}

LogStream::~LogStream() {
  *this << '\n';
  {
    absl::MutexLock lock(mutex());
    std::cerr << stream_.rdbuf() << std::flush;
  }
  if (level_ == LogLevel::FATAL) {
    exit(1);
  }
}

CheckFailStream::CheckFailStream(const char* cond, const char* file, int line)
    : impl_(file, line, LogLevel::FATAL) {
  impl_ << "check failed: " << cond << '\n';
}

CheckFailStream::~CheckFailStream() {
  void* stack[64];
  int depth = absl::GetStackTrace(stack, 64, 1);
  char buffer[256];
  for (int i = 0; i < depth; ++i) {
    impl_ << "  " << stack[i] << "  ";
    if (absl::Symbolize(stack[i], buffer, 256)) {
      impl_ << buffer;
    } else {
      impl_ << "??";
    }
    impl_ << '\n';
  }
}

}  // namespace internal
}  // namespace minigo
