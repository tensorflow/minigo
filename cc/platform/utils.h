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

#include <string>

#if defined(_MSC_VER)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#define MG_ALIGN(x) __declspec(align(x))
#define MG_WARN_UNUSED_RESULT _Check_return_

#elif defined(__GNUC__)

#include <unistd.h>

#define MG_ALIGN(x) __attribute__((aligned(x)))
#define MG_WARN_UNUSED_RESULT __attribute__((warn_unused_result))

#endif

namespace minigo {

// The underlying type of a process ID is platform-specific, unfortunately.
#if defined(_MSC_VER)
using ProcessId = DWORD;
#elif defined(__GNUC__)
using ProcessId = pid_t;
#endif

// Returns the number of logical CPUs.
int GetNumLogicalCpus();

// Returns true if the given file descriptor supports ANSI color codes.
bool FdSupportsAnsiColors(int fd);

// Returns ID of this process.
ProcessId GetProcessId();

// Returns the hostname if possible, or "unknown".
std::string GetHostname();

}  // namespace minigo

#endif  // CC_PLATFORM_UTILS_H_
