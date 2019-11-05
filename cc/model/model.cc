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

#include "cc/model/model.h"

#include <utility>

namespace minigo {

Model::Model(std::string name, const FeatureDescriptor& feature_desc)
    : name_(std::move(name)), feature_desc_(feature_desc) {}
Model::~Model() = default;

void Model::GetOutputs(const std::vector<const ModelInput*>& inputs,
                       const Tensor<float>& policy, const Tensor<float>& value,
                       std::vector<ModelOutput*>* outputs) {
  MG_CHECK(outputs->size() == inputs.size());
  MG_CHECK(policy.shape.is({value.shape[0], kNumMoves}));
  MG_CHECK(value.shape.is({policy.shape[0]}));
  MG_CHECK(static_cast<int>(inputs.size()) <= policy.shape[0]);

  // Copy the policy and value out of the output tensors.
  for (size_t input_idx = 0; input_idx < inputs.size(); ++input_idx) {
    const auto sym = inputs[input_idx]->sym;
    const auto* raw_policy = policy.data + kNumMoves * input_idx;
    const auto* raw_value = value.data + input_idx;
    auto& output = *(*outputs)[input_idx];

    symmetry::ApplySymmetry<kN, 1>(symmetry::Inverse(sym), raw_policy,
                                   output.policy.data());
    output.policy[Coord::kPass] = raw_policy[Coord::kPass];
    output.value = *raw_value;
  }
}

void Model::ApplySymmetry(symmetry::Symmetry sym, const ModelOutput& src,
                          ModelOutput* dst) {
  symmetry::ApplySymmetry<kN, 1>(sym, src.policy.data(), dst->policy.data());
  dst->policy[Coord::kPass] = src.policy[Coord::kPass];
  dst->value = src.value;
}

ModelFactory::~ModelFactory() = default;

}  // namespace minigo
