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

#include <dirent.h>
#include <sys/stat.h>
#include <cstdio>
#include <iostream>
#include <string>

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "cc/file/path.h"

namespace minigo {
namespace file {

namespace {

bool MaybeCreateDir(const std::string& path) {
  int ret = mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
  if (ret == 0) {
    return true;
  }

  if (errno != EEXIST) {
    return false;
  }

  if (path == "/") {
    return true;
  }

  struct stat st;
  ret = stat(path.c_str(), &st);
  if (ret != 0) {
    return false;
  }

  return S_ISDIR(st.st_mode);
}

bool RecursivelyCreateDirNormalized(const std::string& path) {
  if (MaybeCreateDir(path)) {
    return true;
  }

  if (!RecursivelyCreateDir(std::string(Dirname(path)))) {
    return false;
  }

  // Creates the directory knowing the parent already exists.
  return MaybeCreateDir(path);
}
}  // namespace

bool RecursivelyCreateDir(std::string path) {
  return RecursivelyCreateDirNormalized(NormalizeSlashes(path));
}

bool WriteFile(std::string path, absl::string_view contents) {
  path = NormalizeSlashes(path);

  FILE* f = fopen(path.c_str(), "wb");
  if (f == nullptr) {
    std::cerr << "Error opening " << path << " for write" << std::endl;
    return false;
  }

  bool ok = true;
  if (!contents.empty()) {
    ok = fwrite(contents.data(), contents.size(), 1, f) == 1;
    if (!ok) {
      std::cerr << "Error writing " << path << std::endl;
    }
  }
  fclose(f);
  return ok;
}

bool ReadFile(std::string path, std::string* contents) {
  path = NormalizeSlashes(path);

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

bool GetModTime(std::string path, uint64_t* mtime_usec) {
  path = NormalizeSlashes(path);

  struct stat s;
  int result = stat(path.c_str(), &s);
  if (result != 0) {
    std::cerr << "Error statting " << path << ": " << result << std::endl;
    return false;
  }
  *mtime_usec = static_cast<uint64_t>(s.st_mtime) * 1000 * 1000;
  return true;
}

bool ListDir(std::string directory, std::vector<std::string>* files) {
  directory = NormalizeSlashes(directory);

  DIR* dirp = opendir(directory.c_str());
  if (dirp == nullptr) {
    std::cerr << "Could not open directory " << directory << std::endl;
    return false;
  }
  files->clear();
  while (dirent* dp = readdir(dirp)) {
    absl::string_view p(dp->d_name);
    if (p == "." || p == "..") {
      continue;
    }
    files->push_back(dp->d_name);
  }
  closedir(dirp);

  return true;
}
}  // namespace file
}  // namespace minigo
