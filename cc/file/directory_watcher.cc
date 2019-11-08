// Copyright 2019 Google LLC
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

#include "cc/file/directory_watcher.h"

#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "cc/async/poll_thread.h"
#include "cc/file/path.h"
#include "cc/file/utils.h"
#include "cc/logging.h"

namespace minigo {

namespace {

bool ParseModelPathPattern(const std::string& pattern, std::string* directory,
                           std::string* basename_pattern) {
  auto pair = file::SplitPath(pattern);

  *directory = std::string(pair.first);
  if (directory->find('%') != std::string::npos ||
      directory->find('*') != std::string::npos) {
    MG_LOG(ERROR) << "invalid pattern \"" << pattern
                  << "\": directory part must not contain '*' or '%'";
    return false;
  }
  if (directory->empty()) {
    MG_LOG(ERROR) << "directory not be empty";
    return false;
  }

  *basename_pattern = std::string(pair.second);
  auto it = basename_pattern->find('%');
  if (it == std::string::npos || basename_pattern->find("%d") != it ||
      basename_pattern->rfind("%d") != it) {
    MG_LOG(ERROR) << "invalid pattern \"" << pattern
                  << "\": basename must contain "
                  << " exactly one \"%d\" and no other matchers";
    return false;
  }
  return true;
}

bool MatchBasename(const std::string& basename, const std::string& pattern,
                   int* generation) {
  int gen = 0;
  int n = 0;
  if (sscanf(basename.c_str(), pattern.c_str(), &gen, &n) != 1 ||
      n != static_cast<int>(basename.size())) {
    return false;
  }
  *generation = gen;
  return true;
}

}  // namespace

DirectoryWatcher::DirectoryWatcher(
    const std::string& pattern, absl::Duration poll_interval,
    std::function<void(const std::string&)> callback)
    : callback_(std::move(callback)) {
  MG_CHECK(ParseModelPathPattern(pattern, &directory_, &basename_pattern_));
  basename_and_length_pattern_ = absl::StrCat(basename_pattern_, "%n");

  poll_thread_ = absl::make_unique<PollThread>(
      "DirWatcher", poll_interval, std::bind(&DirectoryWatcher::Poll, this));
  poll_thread_->Start();
}

DirectoryWatcher::~DirectoryWatcher() {
  poll_thread_->Join();
}

void DirectoryWatcher::Poll() {
  // List all the files in the given directory.
  std::vector<std::string> basenames;
  if (!file::ListDir(directory_, &basenames)) {
    return;
  }

  // Find the file basename that contains the largest integer.
  const std::string* latest_basename = nullptr;
  int latest_generation = -1;
  for (const auto& basename : basenames) {
    int generation = 0;
    if (!MatchBasename(basename, basename_and_length_pattern_, &generation)) {
      continue;
    }
    if (latest_basename == nullptr || generation > latest_generation) {
      latest_basename = &basename;
      latest_generation = generation;
    }
  }

  if (latest_basename == nullptr) {
    // Didn't find any matching files.
    return;
  }

  // Build the full path to the latest model.
  auto path = file::JoinPath(directory_, *latest_basename);
  if (path == latest_path_) {
    // The latest path hasn't changed.
    return;
  }

  // Update the latest known path and invoke the callback.
  latest_path_ = std::move(path);
  callback_(latest_path_);
}

}  // namespace minigo
