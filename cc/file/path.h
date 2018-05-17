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

#ifndef CC_FILE_PATH_H_
#define CC_FILE_PATH_H_

#include <initializer_list>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"

namespace minigo {
namespace file {

namespace internal {
std::string JoinPathImpl(std::initializer_list<absl::string_view> paths);
}  // namespace internal

template <typename... T>
inline std::string JoinPath(const T&... args) {
  return internal::JoinPathImpl({args...});
}

std::pair<absl::string_view, absl::string_view> SplitPath(
    absl::string_view path);

inline absl::string_view Dirname(absl::string_view path) {
  return SplitPath(path).first;
}

inline absl::string_view Basename(absl::string_view path) {
  return SplitPath(path).second;
}

}  // namespace file
}  // namespace minigo

#endif  //  CC_FILE_PATH_H_
