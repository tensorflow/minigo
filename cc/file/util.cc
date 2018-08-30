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

#include "cc/file/util.h"

#include <sys/stat.h>
#include <cstdio>
#include <iostream>
#include <string>

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

#ifdef MG_ENABLE_TF_DUAL_NET

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

  status = tensorflow::Env::Default()->Stat(path, &stat);
  if (!status.ok()) {
    std::cerr << "Error statting " << path << ": " << status << std::endl;
    return false;
  }

  *mtime_usec = static_cast<uint64_t>(stat.mtime_nsec / 1000);
  return true;
}

#else  // MG_ENABLE_TF_DUAL_NET

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

#endif  // MG_ENABLE_TF_DUAL_NET

}  // namespace file
}  // namespace minigo
