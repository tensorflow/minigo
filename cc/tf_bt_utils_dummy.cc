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

void WriteGameExamples(const std::string& gcp_project_name,
                       const std::string& instance_name,
                       const std::string& table_name,
                       const MctsPlayer& player) {
  MG_FATAL() << "Can't write TensorFlow examples to Bigtable without Bigtable "
                "support enabled. "
                "Please recompile, passing --define=bt=1 to bazel build.";
}

void WriteEvalRecord(const std::string& gcp_project_name,
                     const std::string& instance_name,
                     const std::string& table_name,
                     const MctsPlayer& player,
                     const std::string& black_player_name,
                     const std::string& white_player_name,
                     const std::string& sgf_name,
                     const std::string& tag) {
  MG_FATAL() << "Can't write eval record to Bigtable without Bigtable "
                "support enabled. "
                "Please recompile, passing --define=bt=1 to bazel build.";
}

uint64_t IncrementGameCounter(const std::string& gcp_project_name,
                              const std::string& instance_name,
                              const std::string& table_name,
                              const std::string& counter_name,
                              size_t delta) {
  MG_FATAL() << "Can't increment a Bigtable game counter without Bigtable "
                "support enabled. "
                "Please recompile, passing --define=bt=1 to bazel build.";
  return 0;
}

void PortGamesToBigtable(const std::string& gcp_project_name,
                         const std::string& instance_name,
                         const std::string& table_name,
                         const std::vector<std::string>& paths,
                         int64_t game_counter) {
  MG_FATAL() << "Can't port TFRecord ZLIB files to Bigtable without Bigtable "
                "support enabled. "
                "Please recompile, passing --define=bt=1 to bazel build.";
}

}  // namespace tf_utils
}  // namespace minigo
