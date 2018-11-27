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

// Platform-specific file path separator character.
extern const char kSepChar;

// Platform-specific file path separator as a string.
extern const char kSepStr[2];

namespace internal {
std::string JoinPathImpl(std::initializer_list<absl::string_view> paths);
}  // namespace internal

// Joins the given arguments into a file path.
// The caller is responsible for normalizing slashes.
template <typename... T>
inline std::string JoinPath(const T&... args) {
  return internal::JoinPathImpl({args...});
}

// Splits the path into directory and basename parts.
// On Windows both forward and bask slashes are treated as valid separators.
// On OSX and Linux, only forward slash is treated as a valid separator.
std::pair<absl::string_view, absl::string_view> SplitPath(
    absl::string_view path);

// Splits the path using SplitPath, then returns just the directory part.
inline absl::string_view Dirname(absl::string_view path) {
  return SplitPath(path).first;
}

// Splits the path using SplitPath, then returns just the basename part.
inline absl::string_view Basename(absl::string_view path) {
  return SplitPath(path).second;
}

inline absl::string_view Stem(absl::string_view path) {
  auto base = Basename(path);
  auto pos = base.find_last_of('.');
  if (pos == absl::string_view::npos) {
    return base;
  }
  return base.substr(0, pos);
}

// Normalizes the slashes in the given path.
// On Windows, all forward slashes are replaced with back slashes unless the
// path begins with the string "gs://", in which case back slashes are replaced
// with forward slashes.
// On OSX and Linux, all back slashes are replaced with forward slashes.
std::string NormalizeSlashes(std::string path);

}  // namespace file
}  // namespace minigo

#endif  //  CC_FILE_PATH_H_
