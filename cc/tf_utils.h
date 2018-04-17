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

#include <array>
#include <string>

#include "cc/constants.h"
#include "cc/dual_net.h"

#include "tensorflow/core/example/example.pb.h"

namespace minigo {

// Converts board features, and the pi & value outputs of MTCS to a tensorflow
// example proto.
tensorflow::Example MakeTfExample(const DualNet::BoardFeatures& features,
                                  const std::array<float, kNumMoves>& pi,
                                  float outcome);

// Writes a list of tensorflow Example protos to a zlib compressed TFRecord
// file.
void WriteTfExamples(const std::string& path,
                     absl::Span<const tensorflow::Example> examples);

}  // namespace minigo

#endif  // CC_TF_UTILS_H_
