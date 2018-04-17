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

#include "cc/file/path.h"

#include "absl/strings/str_cat.h"

namespace minigo {
namespace file {
namespace internal {

std::string JoinPathImpl(std::initializer_list<absl::string_view> paths) {
  std::string result;

  for (auto path : paths) {
    if (path.empty()) {
      continue;
    }

    if (result.empty()) {
      result.assign(path.data(), path.size());
      continue;
    }

    if (result.back() == '/') {
      if (path[0] == '/') {
        absl::StrAppend(&result, path.substr(1));
      } else {
        absl::StrAppend(&result, path);
      }
    } else {
      if (path[0] == '/') {
        absl::StrAppend(&result, path);
      } else {
        absl::StrAppend(&result, "/", path);
      }
    }
  }

  return result;
}

}  // namespace internal

std::pair<absl::string_view, absl::string_view> SplitPath(
    absl::string_view path) {
  auto pos = path.find_last_of('/');

  // No '/' in path.
  if (pos == absl::string_view::npos) {
    return std::make_pair(path.substr(0, 0), path);
  }

  // Leading '/'.
  if (pos == 0) {
    return std::make_pair(path.substr(0, 1), absl::ClippedSubstr(path, 1));
  }

  // General case.
  return std::make_pair(path.substr(0, pos),
                        absl::ClippedSubstr(path, pos + 1));
}

}  // namespace file
}  // namespace minigo
