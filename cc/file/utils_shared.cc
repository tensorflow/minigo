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

#include <errno.h>
#include <sys/stat.h>

#include "absl/strings/match.h"
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

}  // namespace

bool RecursivelyCreateDir(absl::string_view path) {
  // GCS doesn't support empty directories (it pretends to by creating an
  // empty file in them) and files can be written without having to first
  // create a directory: just return success immediately.
  if (absl::StartsWith(path, "gs://")) {
    return true;
  }

  std::string path_str(path);
  if (MaybeCreateDir(path_str)) {
    return true;
  }

  if (!RecursivelyCreateDir(Dirname(path))) {
    return false;
  }
  // Creates the directory knowing the parent already exists.
  return MaybeCreateDir(path_str);
}

}  // namespace file
}  // namespace minigo
