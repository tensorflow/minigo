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

#ifndef CC_DUAL_NET_H_
#define CC_DUAL_NET_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cc/position.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

namespace minigo {

class DualNet {
 public:
  struct Output {
    tensorflow::Status status;
    tensorflow::Tensor policy;
    tensorflow::Tensor value;
  };

  DualNet();
  ~DualNet();

  tensorflow::Status Initialize(const std::string& graph_path);

  Output Run(const Position& position);

  const tensorflow::Tensor& features() const { return inputs_[0].second; }

 private:
  void UpdateFeatures(const Position& position);

  std::unique_ptr<tensorflow::Session> session_;
  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs_;
  std::vector<std::string> output_names_;
  std::vector<tensorflow::Tensor> outputs_;
};

}  // namespace minigo

#endif  // CC_DUAL_NET_H_
