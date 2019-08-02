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

#ifndef CC_FILE_UTILS_H_
#define CC_FILE_UTILS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "cc/platform/utils.h"

namespace minigo {
namespace file {

MG_WARN_UNUSED_RESULT bool RecursivelyCreateDir(std::string path);

// Write a file in one shot.
// When compiled with --define=tf=1, uses TensorFlow's file APIs to enable
// access to GCS. Only allows local file access otherwise.
MG_WARN_UNUSED_RESULT bool WriteFile(std::string path,
                                     absl::string_view contents);

// Read a file in one shot.
// When compiled with --define=tf=1, uses TensorFlow's file APIs to enable
// access to GCS. Only allows local file access otherwise.
MG_WARN_UNUSED_RESULT bool ReadFile(std::string path, std::string* contents);

// Get the modification time for a file.
// When compiled with --define=tf=1, uses TensorFlow's file APIs to enable
// access to GCS. Only allows local file access otherwise.
MG_WARN_UNUSED_RESULT bool GetModTime(std::string path, uint64_t* mtime_usec);

// Fills 'files' with the names of the files in 'directory'.
// When compiled with --define=tf=1, uses TensorFlow's file APIs to enable
// access to GCS. Only allows local file access otherwise.
MG_WARN_UNUSED_RESULT bool ListDir(std::string directory,
                                   std::vector<std::string>* files);

}  // namespace file
}  // namespace minigo

#endif  //  CC_FILE_UTILS_H_
