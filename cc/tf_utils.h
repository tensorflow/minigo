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

#ifndef CC_TF_UTILS_H_
#define CC_TF_UTILS_H_

#include <string>

#include "cc/mcts_player.h"

namespace minigo {
namespace tf_utils {

// Writes a list of tensorflow Example protos to a zlib compressed TFRecord
// file, one for each position in the player's move history.
// Each example contains:
//   x: the input BoardFeatures as bytes.
//   pi: the search pi as a float array, serialized as bytes.
//   outcome: a single float containing the game result +/-1.
// CHECK fails if the binary was not compiled with --define=tf=1.
void WriteGameExamples(const std::string& output_dir,
                       const std::string& output_name,
                       const MctsPlayer& player);

// Writes a list of tensorflow Example protos to the specified
// Bigtable, one example per row, starting at the given row cursor.
// Returns the value of the row cursor after having written out
// the examples.
void WriteGameExamples(const std::string& gcp_project_name,
                       const std::string& instance_name,
                       const std::string& table_name, const MctsPlayer& player);

// Atomically increment the game counter in the given Bigtable by the given
// delta.  Returns the new value.  Prior value will be returned - delta.
uint64_t IncrementGameCounter(const std::string& gcp_project_name,
                              const std::string& instance_name,
                              const std::string& table_name, size_t delta);

// Port Minigo games from the given GCS files, which must be in
// `.tfrecord.zz` format.  If game_counter is >=0, use that
// and increment from there.  Otherwise, atomically increment
// and use the value from `table_state=metadata:game_counter`.
void PortGamesToBigtable(const std::string& gcp_project_name,
                         const std::string& instance_name,
                         const std::string& table_name,
                         const std::vector<std::string>& paths,
                         int64_t game_counter = -1);

}  // namespace tf_utils
}  // namespace minigo

#endif  // CC_TF_UTILS_H_
