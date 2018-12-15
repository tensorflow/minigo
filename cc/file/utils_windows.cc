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

#include <cstdio>
#include <string>

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "cc/file/path.h"
#include "cc/logging.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace minigo {
namespace file {

namespace {

bool MaybeCreateDir(const std::string& path) {
  if (CreateDirectory(path.c_str(), nullptr)) {
    return true;
  }

  if (GetLastError() != ERROR_ALREADY_EXISTS) {
    return false;
  }

  auto attribs = GetFileAttributes(path.c_str());
  if (attribs == INVALID_FILE_ATTRIBUTES) {
    return false;
  }
  return (attribs & FILE_ATTRIBUTE_DIRECTORY) != 0;
}

bool RecursivelyCreateDirNormalized(const std::string& path) {
  if (MaybeCreateDir(path)) {
    return true;
  }

  if (!RecursivelyCreateDirNormalized(std::string(Dirname(path)))) {
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
    MG_LOG(ERROR) << "error opening " << path << " for write";
    return false;
  }
  bool ok = true;
  if (!contents.empty()) {
    ok = fwrite(contents.data(), contents.size(), 1, f) == 1;
    if (!ok) {
      MG_LOG(ERROR) << "error writing " << path;
    }
  }
  fclose(f);
  return ok;
}

bool ReadFile(std::string path, std::string* contents) {
  path = NormalizeSlashes(path);

  FILE* f = fopen(path.c_str(), "rb");
  if (f == nullptr) {
    MG_LOG(ERROR) << "error opening " << path << " for read";
    return false;
  }
  fseek(f, 0, SEEK_END);
  contents->resize(ftell(f));
  fseek(f, 0, SEEK_SET);
  bool ok = fread(&(*contents)[0], contents->size(), 1, f) == 1;
  if (!ok) {
    MG_LOG(ERROR) << "error reading " << path;
  }
  fclose(f);
  return ok;
}

bool GetModTime(std::string path, uint64_t* mtime_usec) {
  path = NormalizeSlashes(path);

  auto h =
      CreateFile(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                 nullptr, OPEN_EXISTING, 0, nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    return false;
  }
  FILETIME t;
  bool success = GetFileTime(h, nullptr, nullptr, &t);
  CloseHandle(h);

  if (success) {
    // FILETIME represents time in 100-nanosecond intervals. Convert to
    // microseconds.
    auto lo = static_cast<uint64_t>(t.dwLowDateTime);
    auto hi = static_cast<uint64_t>(t.dwHighDateTime) << 32;
    auto usec = (lo + hi) / 10;

    // Also, the Windows & Unix epochs are different.
    *mtime_usec = usec - 11644473600000000ull;
  }

  return success;
}

bool ListDir(std::string directory, std::vector<std::string>* files) {
  directory = JoinPath(NormalizeSlashes(directory), "*");

  WIN32_FIND_DATA ffd;
  auto h = FindFirstFile(directory.c_str(), &ffd);
  if (h == INVALID_HANDLE_VALUE) {
    return false;
  }

  files->clear();
  do {
    absl::string_view p(ffd.cFileName);
    if (p == "." || p == "..") {
      continue;
    }
    files->push_back(ffd.cFileName);
  } while (FindNextFile(h, &ffd));
  FindClose(h);

  return true;
}

}  // namespace file
}  // namespace minigo
