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

void Unquantize(const TfLiteQuantizationParams& params,
                const Tensor<uint8_t>& src, Tensor<float>* dst) {
  MG_CHECK(src.shape == dst->shape);
  int size = src.shape.num_elements();
  for (int i = 0; i < size; ++i) {
    dst->data[i] = (src.data[i] - params.zero_point) * params.scale;
  }
}

class LiteDualNet : public Model {
 public:
  LiteDualNet(std::string graph_path, const FeatureDescriptor& feature_desc);

  void RunMany(const std::vector<const ModelInput*>& inputs,
               std::vector<ModelOutput*>* outputs,
               std::string* model_name) override;

 private:
  void Reserve(int capacity);

  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;

  int input_idx_;
  int policy_idx_;
  int value_idx_;

  TfLiteTensor* input_ = nullptr;
  TfLiteTensor* policy_ = nullptr;
  TfLiteTensor* value_ = nullptr;

  std::string graph_path_;
  int batch_capacity_;

  BackedTensor<float> unquantized_policy_;
  BackedTensor<float> unquantized_value_;
};

LiteDualNet::LiteDualNet(std::string graph_path,
                         const FeatureDescriptor& feature_desc)
    : Model(std::string(file::Stem(graph_path)), feature_desc),
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
  MG_CHECK(input->dims->data[3] == feature_desc.num_planes);

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

void LiteDualNet::Reserve(int capacity) {
  MG_CHECK(capacity > 0);
  if (capacity == batch_capacity_) {
    return;
  }

  // Resize input tensor to batch size.
  MG_CHECK(interpreter_->ResizeInputTensor(
               input_idx_, {capacity, kN, kN,
                            feature_descriptor().num_planes}) == kTfLiteOk);
  MG_CHECK(interpreter_->AllocateTensors() == kTfLiteOk);

  // Get the new inputs and outputs after AllocateTensor().
  input_ = interpreter_->tensor(input_idx_);
  policy_ = interpreter_->tensor(policy_idx_);
  value_ = interpreter_->tensor(value_idx_);

  unquantized_policy_.resize({capacity, 1, 1, kNumMoves});
  unquantized_value_.resize({capacity, 1, 1, 1});

  batch_capacity_ = capacity;
}

void LiteDualNet::RunMany(const std::vector<const ModelInput*>& inputs,
                          std::vector<ModelOutput*>* outputs,
                          std::string* model_name) {
  MG_CHECK(inputs.size() == outputs->size());

  Reserve(inputs.size());

  Tensor<float> policy, value;
  const auto& dims = input_->dims->data;
  switch (input_->type) {
    case kTfLiteFloat32: {
      Tensor<float> features({dims[0], dims[1], dims[2], dims[3]},
                             input_->data.f);
      feature_descriptor().set_floats(inputs, &features);

      MG_CHECK(interpreter_->Invoke() == kTfLiteOk);

      policy = Tensor<float>({batch_capacity_, kNumMoves}, policy_->data.f);
      value = Tensor<float>({batch_capacity_}, value_->data.f);
      break;
    }
    case kTfLiteUInt8: {
      Tensor<uint8_t> features({dims[0], dims[1], dims[2], dims[3]},
                               input_->data.uint8);
      feature_descriptor().set_bytes(inputs, &features);
      MG_CHECK(interpreter_->Invoke() == kTfLiteOk);

      Tensor<uint8_t> quantized_policy({batch_capacity_, kNumMoves},
                                       policy_->data.uint8);
      Tensor<uint8_t> quantized_value({batch_capacity_}, value_->data.uint8);

      policy = unquantized_policy_.tensor();
      value = unquantized_value_.tensor();
      Unquantize(policy_->params, quantized_policy, &policy);
      Unquantize(value_->params, quantized_value, &value);
      break;
    }
    default:
      MG_LOG(FATAL) << "Unsupported input type" << input_->type;
      break;
  }

  Model::GetOutputs(inputs, policy, value, absl::MakeSpan(*outputs));

  if (model_name != nullptr) {
    *model_name = graph_path_;
  }
}

}  // namespace

std::unique_ptr<Model> LiteDualNetFactory::NewModel(
    const std::string& descriptor) {
  // TODO(tommadams): support extra feature types.
  auto feature_desc = FeatureDescriptor::Create<AgzFeatures>();
  return absl::make_unique<LiteDualNet>(descriptor, feature_desc);
}

}  // namespace minigo
