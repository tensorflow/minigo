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

#ifndef CC_FILE_DIRECTORY_WATCHER_H_
#define CC_FILE_DIRECTORY_WATCHER_H_

#include <atomic>
#include <functional>
#include <string>

#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "cc/thread.h"

namespace minigo {

class DirectoryWatcher {
 public:
  // The `DirectoryWatcher` polls a directory at the given `poll_interval` for
  // new files that match file `pattern`.
  // `pattern` must be a file path that contains exactly one "%d" scanf
  // matcher in the basename part (not the dirname part).
  // `callback` will invoked whenever a later file that matches `pattern` is
  // found. The callback is invoked on a background thread (though a single
  // `DirectoryWatcher` instance only ever makes one call to `callback` at a
  // time).
  DirectoryWatcher(const std::string& pattern, absl::Duration poll_interval,
                   std::function<void(const std::string&)> callback);

  ~DirectoryWatcher();

 private:
  void Poll();

  // The directory we're watching for new files.
  std::string directory_;

  // Pattern used to match files in directory_.
  std::string basename_pattern_;

  // basename_pattern_ with "%n" appended, which is used to ensure that the full
  // basename matches the pattern (and not just a prefix).
  std::string basename_and_length_pattern_;

  std::string latest_path_;

  std::unique_ptr<Thread> poll_thread_;

  std::function<void(const std::string&)> callback_;
};

}  // namespace minigo

#endif  // CC_FILE_DIRECTORY_WATCHER_H_
