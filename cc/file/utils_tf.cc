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

#include <iostream>
#include <memory>

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"

namespace minigo {
namespace file {

bool WriteFile(const std::string& path, absl::string_view contents) {
  tensorflow::Status status;
  auto* env = tensorflow::Env::Default();

  std::unique_ptr<tensorflow::WritableFile> file;
  status = env->NewWritableFile(path, &file);
  if (!status.ok()) {
    std::cerr << "Error opening " << path << " for write: " << status
              << std::endl;
    return false;
  }

  // *sigh* absl::string_view doesn't automatically convert to
  // tensorflow::StringPiece even though I'm pretty sure they're exactly the
  // same.
  status = file->Append({contents.data(), contents.size()});
  if (!status.ok()) {
    std::cerr << "Error writing to " << path << ": " << status << std::endl;
    return false;
  }

  status = file->Close();
  if (!status.ok()) {
    std::cerr << "Error closing " << path << ": " << status << std::endl;
    return false;
  }
  return true;
}

bool ReadFile(const std::string& path, std::string* contents) {
  tensorflow::Status status;
  auto* env = tensorflow::Env::Default();

  tensorflow::uint64 size;
  status = env->GetFileSize(path, &size);
  if (!status.ok()) {
    std::cerr << "Error getting size of " << path << ": " << status
              << std::endl;
    return false;
  }

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  status = env->NewRandomAccessFile(path, &file);
  if (!status.ok()) {
    std::cerr << "Error opening " << path << " for read: " << status
              << std::endl;
    return false;
  }

  contents->resize(size);
  tensorflow::StringPiece s;
  status = file->Read(0u, size, &s, &(*contents)[0]);
  if (!status.ok()) {
    std::cerr << "Error reading " << path << ": " << status << std::endl;
    return false;
  }
  contents->resize(s.size());

  return true;
}

bool GetModTime(const std::string& path, uint64_t* mtime_usec) {
  tensorflow::Status status;
  auto* env = tensorflow::Env::Default();
  tensorflow::FileStatistics stat;

  status = env->Stat(path, &stat);
  if (!status.ok()) {
    std::cerr << "Error statting " << path << ": " << status << std::endl;
    return false;
  }

  *mtime_usec = static_cast<uint64_t>(stat.mtime_nsec / 1000);
  return true;
}

}  // namespace file
}  // namespace minigo
