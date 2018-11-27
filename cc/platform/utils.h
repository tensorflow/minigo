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

#ifndef CC_PLATFORM_UTILS_H_
#define CC_PLATFORM_UTILS_H_

#if defined(_MSC_VER)

#define MG_PLATFORM_MSC
#define MG_ALIGN(x) __declspec(align(x))
#define MG_WARN_UNUSED_RESULT _Check_return_

#elif defined(__GNU_C__)

#define MG_PLATFORM_GCC
#define MG_ALIGN(x) __attribute__((aligned(x)))
#define MG_WARN_UNUSED_RESULT __attribute__((warn_unused_result)) 

#endif

namespace minigo {

// Returns the number of logical CPUs.
int GetNumLogicalCpus();

// Returns true if the given file descriptor supports ANSI color codes.
bool FdSupportsAnsiColors(int fd);

}  // namespace minigo

#endif  // CC_PLATFORM_UTILS_H_
