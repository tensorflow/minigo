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

}  // namespace tf_utils
}  // namespace minigo

#endif  // CC_TF_UTILS_H_
