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

Model::Model(std::string name, FeatureType feature_type, int buffer_count)
    : name_(std::move(name)),
      feature_type_(feature_type),
      buffer_count_(buffer_count) {}
Model::~Model() = default;

int Model::GetNumFeaturePlanes(FeatureType feature_type) {
  switch (feature_type) {
    case Model::FeatureType::kAgz:
      return Model::kNumAgzFeaturePlanes;
    case Model::FeatureType::kExtra:
      return Model::kNumExtraFeaturePlanes;
    default:
      MG_LOG(FATAL) << "invalid feature type "
                    << static_cast<int>(feature_type);
      return 0;
  }
}

void Model::ApplySymmetry(symmetry::Symmetry sym, const Output& src,
                          Output* dst) {
  symmetry::ApplySymmetry<kN, 1>(sym, src.policy.data(), dst->policy.data());
  dst->policy[Coord::kPass] = src.policy[Coord::kPass];
  dst->value = src.value;
}

ModelFactory::~ModelFactory() = default;

}  // namespace minigo
