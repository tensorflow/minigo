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

#include "cc/platform/utils.h"

#include <sys/sysctl.h>
#include <cstring>

#include "cc/logging.h"

namespace minigo {

int GetNumLogicalCpus() {
  int nproc = 0;
  size_t len;
  MG_CHECK(sysctlbyname("hw.logicalcpu", &nproc, &len, nullptr, 0) == 0);
  MG_CHECK(len == sizeof(nproc));
  return nproc;
}

bool FdSupportsAnsiColors(int fd) { return isatty(fd); }

ProcessId GetProcessId() { return getpid(); }

std::string GetHostname() {
  // Posix guarantees that a 256B buffer is large enough to hold the hostname,
  // so we don't need to worry about whether a truncated hostname is
  // null-terminated.
  char hostname[256];
  return gethostname(hostname, sizeof(hostname)) == 0 ? hostname : "hostname";
}

}  // namespace minigo
