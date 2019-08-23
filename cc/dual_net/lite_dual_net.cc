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

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "cc/constants.h"
#include "cc/file/path.h"
#include "cc/logging.h"
#include "cc/platform/utils.h"
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

using tflite::FlatBufferModel;
using tflite::InterpreterBuilder;
using tflite::ops::builtin::BuiltinOpResolver;

namespace minigo {
namespace {

class LiteDualNet : public DualNet {
 public:
  LiteDualNet(std::string graph_path, bool random_symmetry,
              uint64_t random_seed);

 private:
  void RunManyImpl(std::string* model_name) override;
  void Reserve(size_t capacity);

  template <typename T>
  void RunManyImpl(T* feature_data, const T* policy_data, const T* value_data);

  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;

  int input_idx_;
  int policy_idx_;
  int value_idx_;

  TfLiteTensor* input_ = nullptr;
  TfLiteTensor* policy_ = nullptr;
  TfLiteTensor* value_ = nullptr;

  std::string graph_path_;
  size_t batch_capacity_;
};

LiteDualNet::LiteDualNet(std::string graph_path, bool random_symmetry,
                         uint64_t random_seed)
    : DualNet(std::string(file::Stem(graph_path)), random_symmetry,
              random_seed),
      graph_path_(std::move(graph_path)),
      batch_capacity_(0) {
  model_ = FlatBufferModel::BuildFromFile(graph_path_.c_str());
  MG_CHECK(model_ != nullptr);

  BuiltinOpResolver resolver;
  InterpreterBuilder(*model_, resolver)(&interpreter_);
  MG_CHECK(interpreter_ != nullptr);

  // Let's just use all the processors we can.
  interpreter_->SetNumThreads(GetNumLogicalCpus());

  const auto& inputs = interpreter_->inputs();
  MG_CHECK(inputs.size() == 1);
  absl::string_view input_name = interpreter_->GetInputName(0);
  MG_CHECK(input_name == "pos_tensor");
  input_idx_ = inputs[0];

  // Check that the model matches the board size and feature count.
  auto* input = interpreter_->tensor(input_idx_);
  MG_CHECK(input->dims->size == 4);
  MG_CHECK(input->dims->data[1] == kN);
  MG_CHECK(input->dims->data[2] == kN);
  MG_CHECK(input->dims->data[3] == kNumStoneFeatures);

  const auto& outputs = interpreter_->outputs();
  MG_CHECK(outputs.size() == 2);
  absl::string_view output_0_name = interpreter_->GetOutputName(0);
  absl::string_view output_1_name = interpreter_->GetOutputName(1);
  if (output_0_name == "policy_output") {
    MG_CHECK(output_1_name == "value_output") << output_1_name;
    policy_idx_ = outputs[0];
    value_idx_ = outputs[1];
  } else {
    MG_CHECK(output_1_name == "policy_output") << output_1_name;
    MG_CHECK(output_0_name == "value_output") << output_0_name;
    policy_idx_ = outputs[1];
    value_idx_ = outputs[0];
  }
}

void LiteDualNet::Reserve(size_t capacity) {
  MG_CHECK(capacity > 0);
  if (capacity <= batch_capacity_) {
    return;
  }

  // Resize input tensor to batch size.
  MG_CHECK(interpreter_->ResizeInputTensor(
               input_idx_, {static_cast<int>(capacity), kN, kN,
                            DualNet::kNumStoneFeatures}) == kTfLiteOk);
  MG_CHECK(interpreter_->AllocateTensors() == kTfLiteOk);

  // Get the new inputs and outputs after AllocateTensor().
  input_ = interpreter_->tensor(input_idx_);
  policy_ = interpreter_->tensor(policy_idx_);
  value_ = interpreter_->tensor(value_idx_);

  batch_capacity_ = capacity;
}

void LiteDualNet::RunManyImpl(std::string* model_name) {
  Reserve(features_.size());

  switch (input_->type) {
    case kTfLiteFloat32:
      return RunManyImpl(input_->data.f, policy_->data.f, value_->data.f);
    case kTfLiteUInt8:
      return RunManyImpl(input_->data.uint8, policy_->data.uint8,
                         value_->data.uint8);
    default:
      MG_LOG(FATAL) << "Unsupported input type";
  }

  if (model_name != nullptr) {
    *model_name = graph_path_;
  }
}

template <typename T, typename S>
T Convert(const TfLiteQuantizationParams&, const S& x) {
  return static_cast<T>(x);
}

// Dequantize.
template <>
float Convert<float, uint8_t>(const TfLiteQuantizationParams& params,
                              const uint8_t& x) {
  return (x - params.zero_point) * params.scale;
};

// Quantize.
template <>
uint8_t Convert<uint8_t, float>(const TfLiteQuantizationParams& params,
                                const float& x) {
  return static_cast<uint8_t>(x / params.scale + params.zero_point);
};

template <typename T>
void LiteDualNet::RunManyImpl(T* feature_data, const T* policy_data,
                              const T* value_data) {
  int num_features = static_cast<int>(features_.size());

  // Allow a smaller batch size than we run inference on because the first
  // inference made when starting the game has batch size 1 (instead of the
  // normal 8) to initialized the tree search.
  MG_CHECK(num_features <= input_->dims->data[0]);

  // TODO(tommadams): Make BoardFeatures a uint8_t array and memcpy here.
  const auto& input_params = input_->params;
  for (int j = 0; j < num_features; ++j) {
    const auto& board = features_[j];
    for (size_t i = 0; i < board.size(); ++i) {
      // TODO(csigg): Apply dequantization parameters?
      feature_data[j * kNumStoneFeatures + i] =
          Convert<T>(input_params, board[i]);
    }
  }

  MG_CHECK(interpreter_->Invoke() == kTfLiteOk);

  const auto& policy_params = policy_->params;
  const auto& value_params = value_->params;
  for (int j = 0; j < num_features; ++j) {
    for (int i = 0; i < kNumMoves; ++i) {
      raw_outputs_[j].policy[i] =
          Convert<float>(policy_params, policy_data[j * num_features + i]);
    }
    raw_outputs_[j].value = Convert<float>(value_params, value_data[j]);
  }
}
}  // namespace

LiteDualNetFactory::LiteDualNetFactory(bool random_symmetry,
                                       uint64_t random_seed)
    : DualNetFactory(random_symmetry, random_seed) {}

std::unique_ptr<Model> LiteDualNetFactory::NewModel(
    const std::string& descriptor) {
  return absl::make_unique<LiteDualNet>(descriptor, random_symmetry(),
                                        GetModelSeed());
}

}  // namespace minigo
