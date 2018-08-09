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

#include "cc/dual_net/lite_dual_net.h"

#include <sys/sysinfo.h>
#include <iostream>

#include "absl/strings/string_view.h"
#include "cc/check.h"
#include "cc/constants.h"
#include "tensorflow/contrib/lite/kernels/register.h"

using tflite::FlatBufferModel;
using tflite::InterpreterBuilder;
using tflite::ops::builtin::BuiltinOpResolver;

namespace minigo {

LiteDualNet::LiteDualNet(const std::string& graph_path)
    : graph_path_(graph_path) {
  model_ = FlatBufferModel::BuildFromFile(graph_path.c_str());
  MG_CHECK(model_ != nullptr);

  BuiltinOpResolver resolver;
  InterpreterBuilder(*model_, resolver)(&interpreter_);
  MG_CHECK(interpreter_ != nullptr);

  // Let's just use all the processors we can.
  interpreter_->SetNumThreads(get_nprocs());

  // Initialize input.
  const auto& inputs = interpreter_->inputs();
  MG_CHECK(inputs.size() == 1);
  absl::string_view input_name = interpreter_->GetInputName(0);
  MG_CHECK(input_name == "pos_tensor");

  auto* input_tensor = interpreter_->tensor(inputs[0]);
  MG_CHECK(input_tensor != nullptr);
  MG_CHECK(input_tensor->dims->size == 4);
  MG_CHECK(input_tensor->dims->data[1] == kN);
  MG_CHECK(input_tensor->dims->data[2] == kN);
  MG_CHECK(input_tensor->dims->data[3] == DualNet::kNumStoneFeatures);

  // Initialize outputs.
  const auto& outputs = interpreter_->outputs();
  MG_CHECK(outputs.size() == 2);
  absl::string_view output_0_name = interpreter_->GetOutputName(0);
  absl::string_view output_1_name = interpreter_->GetOutputName(1);
  if (output_0_name == "policy_output") {
    MG_CHECK(output_1_name == "value_output") << output_1_name;
    policy_ = 0;
    value_ = 1;
  } else {
    MG_CHECK(output_1_name == "policy_output") << output_1_name;
    MG_CHECK(output_0_name == "value_output") << output_0_name;
    policy_ = 1;
    value_ = 0;
  }

  auto* policy_tensor = interpreter_->tensor(outputs[policy_]);
  MG_CHECK(policy_tensor != nullptr);
  MG_CHECK(policy_tensor->dims->size == 2) << policy_tensor->dims->size;

  auto* value_tensor = interpreter_->tensor(outputs[value_]);
  MG_CHECK(value_tensor != nullptr);
  MG_CHECK(value_tensor->dims->size == 1);

  MG_CHECK(interpreter_->AllocateTensors() == kTfLiteOk);
}

LiteDualNet::~LiteDualNet() {}

void LiteDualNet::RunMany(absl::Span<const BoardFeatures> features,
                          absl::Span<Output> outputs, std::string* model) {
  int batch_size = static_cast<int>(features.size());
  auto* input_tensor = interpreter_->tensor(interpreter_->inputs()[0]);
  MG_CHECK(input_tensor->dims->data[0] == batch_size);

  auto* data = interpreter_->typed_input_tensor<float>(0);
  memcpy(data, features.data(), features.size() * sizeof(BoardFeatures));

  MG_CHECK(interpreter_->Invoke() == kTfLiteOk);

  auto* policy_data = interpreter_->typed_output_tensor<float>(policy_);
  auto* value_data = interpreter_->typed_output_tensor<float>(value_);
  for (int i = 0; i < batch_size; ++i) {
    memcpy(outputs[i].policy.data(), policy_data + i * kNumMoves,
           sizeof(outputs[i].policy));
    outputs[i].value = value_data[i];
  }

  if (model != nullptr) {
    *model = graph_path_;
  }
}

}  // namespace minigo
