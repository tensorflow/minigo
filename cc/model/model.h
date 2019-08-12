// Copyright 2019 Google LLC
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

#ifndef CC_MODEL_MODEL_H_
#define CC_MODEL_MODEL_H_

#include <string>
#include <vector>

#include "absl/types/span.h"
#include "cc/position.h"

namespace minigo {

class Model {
 public:
  struct Output {
    std::array<float, kNumMoves> policy;
    float value;
  };

  virtual ~Model();

  virtual void RunMany(absl::Span<const Position*> position_history,
                       std::vector<Output*> outputs,
                       std::string* model_name) = 0;
};

}  // namespace minigo

#endif  //  CC_MODEL_MODEL_H_
