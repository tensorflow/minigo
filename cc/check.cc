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

#include "cc/check.h"

#include <cstdlib>
#include <iostream>

#include "absl/debugging/stacktrace.h"
#include "absl/debugging/symbolize.h"

namespace minigo {
namespace internal {

void CheckFail(const char* cond, const char* file, int line) {
  std::cerr << "Check failed at " << file << ":" << line << ": " << cond
            << std::endl;

  void* stack[64];
  int depth = absl::GetStackTrace(stack, 64, 1);
  char buffer[256];
  for (int i = 0; i < depth; ++i) {
    std::cerr << "  " << stack[i] << "  ";
    if (absl::Symbolize(stack[i], buffer, 256)) {
      std::cerr << buffer;
    } else {
      std::cerr << "??";
    }
    std::cerr << std::endl;
  }

  exit(1);
}

}  // namespace internal
}  // namespace minigo
