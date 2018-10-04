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

#ifndef CC_DUAL_NET_LITE_DUAL_NET_H_
#define CC_DUAL_NET_LITE_DUAL_NET_H_

#include <memory>
#include <string>

#include "absl/types/span.h"
#include "cc/dual_net/dual_net.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/model.h"

namespace minigo {

// TODO(csigg): Move to implementation file.
class LiteDualNet : public DualNet {
 public:
  explicit LiteDualNet(const std::string& graph_path);

  void RunMany(std::vector<const BoardFeatures*> features,
               std::vector<Output*> outputs, std::string* model) override;

 private:
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;

  TfLiteTensor* input_ = nullptr;
  TfLiteTensor* policy_ = nullptr;
  TfLiteTensor* value_ = nullptr;

  std::string graph_path_;
};

std::unique_ptr<DualNet> NewLiteDualNet(const std::string& model_path);

}  // namespace minigo

#endif  // CC_DUAL_NET_LITE_DUAL_NET_H_
