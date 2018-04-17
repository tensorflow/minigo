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

#ifndef CC_MINIGO_FILE_HELPERS_H_

#include <string>

namespace minigo {
namespace file {

__attribute__((warn_unused_result)) bool SetContents(
    const std::string& path, const std::string& contents);

}  // namespace file
}  // namespace minigo

#endif  // CC_MINIGO_FILE_HELPERS_H_
