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

#include "cc/tf_utils.h"

#include "cc/check.h"

namespace minigo {
namespace tf_utils {

void WriteGameExamples(const std::string& output_dir,
                       const std::string& output_name,
                       const MctsPlayer& player) {
  MG_FATAL()
      << "Can't write TensorFlow examples without TensorFlow support enabled. "
         "Please recompile, passing --define=tf=1 to bazel build.";
}

}  // namespace tf_utils
}  // namespace minigo
