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

#include <memory>

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "cc/file/path.h"
#include "cc/file/utils.h"
#include "cc/logging.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"

namespace minigo {
namespace file {

bool RecursivelyCreateDir(std::string path) {
  path = NormalizeSlashes(path);

  tensorflow::Status status;

  // GCS doesn't support empty directories (it pretends to by creating an
  // empty file in them) and files can be written without having to first
  // create a directory: just return success immediately.
  if (absl::StartsWith(path, "gs://")) {
    return true;
  }

  auto* env = tensorflow::Env::Default();
  status = env->RecursivelyCreateDir(path);
  if (!status.ok()) {
    MG_LOG(ERROR) << "error creating " << path << ": " << status;
    return false;
  }

  return true;
}

bool WriteFile(std::string path, absl::string_view contents) {
  path = NormalizeSlashes(path);

  tensorflow::Status status;
  auto* env = tensorflow::Env::Default();

  std::unique_ptr<tensorflow::WritableFile> file;
  status = env->NewWritableFile(path, &file);
  if (!status.ok()) {
    MG_LOG(ERROR) << "error opening " << path << " for write: " << status;
    return false;
  }

  // *sigh* absl::string_view doesn't automatically convert to
  // tensorflow::StringPiece even though I'm pretty sure they're exactly the
  // same.
  if (!contents.empty()) {
    status = file->Append({contents.data(), contents.size()});
    if (!status.ok()) {
      MG_LOG(ERROR) << "error writing to " << path << ": " << status;
      return false;
    }
  }

  status = file->Close();
  if (!status.ok()) {
    MG_LOG(ERROR) << "error closing " << path << ": " << status;
    return false;
  }
  return true;
}

bool ReadFile(std::string path, std::string* contents) {
  path = NormalizeSlashes(path);

  tensorflow::Status status;
  auto* env = tensorflow::Env::Default();

  tensorflow::uint64 size;
  status = env->GetFileSize(path, &size);
  if (!status.ok()) {
    MG_LOG(ERROR) << "error getting size of " << path << ": " << status;
    return false;
  }

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  status = env->NewRandomAccessFile(path, &file);
  if (!status.ok()) {
    MG_LOG(ERROR) << "error opening " << path << " for read: " << status;
    return false;
  }

  contents->resize(size);
  tensorflow::StringPiece s;
  status = file->Read(0u, size, &s, &(*contents)[0]);
  if (!status.ok()) {
    MG_LOG(ERROR) << "error reading " << path << ": " << status;
    return false;
  }
  contents->resize(s.size());

  return true;
}

bool GetModTime(std::string path, uint64_t* mtime_usec) {
  path = NormalizeSlashes(path);

  tensorflow::Status status;
  auto* env = tensorflow::Env::Default();
  tensorflow::FileStatistics stat;

  status = env->Stat(path, &stat);
  if (!status.ok()) {
    MG_LOG(ERROR) << "error statting " << path << ": " << status;
    return false;
  }

  *mtime_usec = static_cast<uint64_t>(stat.mtime_nsec / 1000);
  return true;
}

bool ListDir(std::string directory, std::vector<std::string>* files) {
  directory = NormalizeSlashes(directory);

  tensorflow::Status status;
  auto* env = tensorflow::Env::Default();

  status = env->GetChildren(directory, files);
  if (!status.ok()) {
    MG_LOG(ERROR) << "error getting " << directory << " content: " << status;
    return false;
  }

  return true;
}

bool FileExists(std::string path) {
  path = NormalizeSlashes(path);
  auto* env = tensorflow::Env::Default();
  return env->FileExists(path).ok();
}

}  // namespace file
}  // namespace minigo
