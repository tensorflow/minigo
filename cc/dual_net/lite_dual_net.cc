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

#include "absl/memory/memory.h"
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

  // Resize input tensor to batch size.
  interpreter_->ResizeInputTensor(
      0, {FLAGS_batch_size, kN, kN, DualNet::kNumStoneFeatures});

  MG_CHECK(interpreter_->AllocateTensors() == kTfLiteOk);

  // Initialize input.
  const auto& inputs = interpreter_->inputs();
  MG_CHECK(inputs.size() == 1);
  absl::string_view input_name = interpreter_->GetInputName(0);
  MG_CHECK(input_name == "pos_tensor");

  input_ = interpreter_->tensor(inputs[0]);
  MG_CHECK(input_ != nullptr);
  MG_CHECK(input_->type == kTfLiteUInt8);
  MG_CHECK(input_->dims->size == 4);
  MG_CHECK(input_->dims->data[0] == FLAGS_batch_size);
  MG_CHECK(input_->dims->data[1] == kN);
  MG_CHECK(input_->dims->data[2] == kN);
  MG_CHECK(input_->dims->data[3] == DualNet::kNumStoneFeatures);

  // Initialize outputs.
  const auto& outputs = interpreter_->outputs();
  MG_CHECK(outputs.size() == 2);
  absl::string_view output_0_name = interpreter_->GetOutputName(0);
  absl::string_view output_1_name = interpreter_->GetOutputName(1);
  if (output_0_name == "policy_output") {
    MG_CHECK(output_1_name == "value_output") << output_1_name;
    policy_ = interpreter_->tensor(outputs[0]);
    value_ = interpreter_->tensor(outputs[1]);
  } else {
    MG_CHECK(output_1_name == "policy_output") << output_1_name;
    MG_CHECK(output_0_name == "value_output") << output_0_name;
    policy_ = interpreter_->tensor(outputs[1]);
    value_ = interpreter_->tensor(outputs[0]);
  }

  MG_CHECK(policy_ != nullptr);
  MG_CHECK(policy_->type == kTfLiteUInt8);
  MG_CHECK(policy_->dims->size == 2);
  MG_CHECK(policy_->dims->data[0] == FLAGS_batch_size);
  MG_CHECK(policy_->dims->data[1] == kNumMoves);

  MG_CHECK(value_ != nullptr);
  MG_CHECK(value_->type == kTfLiteUInt8);
  MG_CHECK(value_->dims->size == 1);
  MG_CHECK(value_->dims->data[0] == FLAGS_batch_size);
}

// TODO(csigg): Support both float and uint8_t.
void LiteDualNet::RunMany(std::vector<const BoardFeatures*> features,
                          std::vector<Output*> outputs, std::string* model) {
  int batch_size = static_cast<int>(features.size());

  // Allow a smaller batch size than we run inference on because the first
  // inference made when starting the game has batch size 1 (instead of the
  // normal 8) to initialized the tree search.
  MG_CHECK(batch_size <= input_->dims->data[0]);

  // TODO(tommadams): Make BoardFeatures a uint8_t array and memcpy here.
  auto* data = input_->data.uint8;
  for (size_t j = 0; j < features.size(); ++j) {
    const auto& board = *features[j];
    for (size_t i = 0; i < board.size(); ++i) {
      // TODO(csigg): Apply dequantization parameters?
      data[j * kNumStoneFeatures + i] = static_cast<uint8_t>(board[i]);
    }
  }

  MG_CHECK(interpreter_->Invoke() == kTfLiteOk);

  auto* policy_data = policy_->data.uint8;
  auto* value_data = value_->data.uint8;
  const auto& policy_params = policy_->params;
  const auto& value_params = value_->params;
  for (int j = 0; j < batch_size; ++j) {
    for (int i = 0; i < kNumMoves; ++i) {
      outputs[j]->policy[i] =
          policy_params.scale *
          (policy_data[j * batch_size + i] - policy_params.zero_point);
    }
    outputs[j]->value =
        value_params.scale * (value_data[j] - value_params.zero_point);
  }

  if (model != nullptr) {
    *model = graph_path_;
  }
}

std::unique_ptr<DualNet> NewLiteDualNet(const std::string& model_path) {
  return absl::make_unique<LiteDualNet>(model_path);
}
}  // namespace minigo
