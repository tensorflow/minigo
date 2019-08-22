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

#include "cc/color.h"
#include "cc/constants.h"
#include "cc/inline_vector.h"
#include "cc/position.h"

namespace minigo {

class Model {
 public:
  struct Input {
    Color to_play;

    // position_history[0] holds the current position and position_history[i]
    // holds the position from i moves ago.
    inline_vector<const Position::Stones*, kMaxPositionHistory>
        position_history;
  };

  struct Output {
    std::array<float, kNumMoves> policy;
    float value;
  };

  // TODO(tommadams): is there some way to avoid having buffer_count in the base
  // class? All subclasses except BufferedModel set this to 1.
  Model(std::string name, int buffer_count);
  virtual ~Model();

  const std::string& name() const { return name_; }

  // Returns the ideal number of inference requests in flight for this model.
  int buffer_count() const { return buffer_count_; }

  virtual void RunMany(const std::vector<const Input*>& inputs,
                       std::vector<Output*>* outputs,
                       std::string* model_name) = 0;

 private:
  const std::string name_;
  const int buffer_count_;
};

// Factory that creates Model instances.
// All implementations are required to be thread safe.
class ModelFactory {
 public:
  virtual ~ModelFactory();

  // Create a single model.
  virtual std::unique_ptr<Model> NewModel(const std::string& descriptor) = 0;
};

}  // namespace minigo

#endif  //  CC_MODEL_MODEL_H_
