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

#include "cc/file/utils.h"

#include <sys/stat.h>
#include <cstdio>
#include <iostream>
#include <string>

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "cc/file/path.h"

namespace minigo {
namespace file {

bool WriteFile(const std::string& path, absl::string_view contents) {
  FILE* f = fopen(path.c_str(), "wb");
  if (f == nullptr) {
    std::cerr << "Error opening " << path << " for write" << std::endl;
    return false;
  }
  bool ok = fwrite(contents.data(), contents.size(), 1, f) == 1;
  if (!ok) {
    std::cerr << "Error writing " << path << std::endl;
  }
  fclose(f);
  return ok;
}

bool ReadFile(const std::string& path, std::string* contents) {
  FILE* f = fopen(path.c_str(), "rb");
  if (f == nullptr) {
    std::cerr << "Error opening " << path << " for read" << std::endl;
    return false;
  }
  fseek(f, 0, SEEK_END);
  contents->resize(ftell(f));
  fseek(f, 0, SEEK_SET);
  bool ok = fread(&(*contents)[0], contents->size(), 1, f) == 1;
  if (!ok) {
    std::cerr << "Error reading " << path << std::endl;
  }
  fclose(f);
  return ok;
}

bool GetModTime(const std::string& path, uint64_t* mtime_usec) {
  struct stat s;
  int result = stat(path.c_str(), &s);
  if (result != 0) {
    std::cerr << "Error statting " << path << ": " << result << std::endl;
    return false;
  }
  *mtime_usec = static_cast<uint64_t>(s.st_mtime) * 1000 * 1000;
  return true;
}

}  // namespace file
}  // namespace minigo
