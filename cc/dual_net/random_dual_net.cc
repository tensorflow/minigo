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

#include "cc/dual_net/random_dual_net.h"

#include <cmath>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "cc/logging.h"
#include "cc/platform/utils.h"

namespace minigo {

RandomDualNet::RandomDualNet(std::string name,
                             const FeatureDescriptor& feature_desc,
                             uint64_t seed, float policy_stddev,
                             float value_stddev)
    : Model(std::move(name), feature_desc),
      rnd_(seed, Random::kUniqueStream),
      policy_stddev_(policy_stddev), value_stddev_(value_stddev) {}

void RandomDualNet::RunMany(const std::vector<const ModelInput*>& inputs,
                            std::vector<ModelOutput*>* outputs,
                            std::string* model_name) {
  for (auto* output : *outputs) {
    rnd_.NormalDistribution(0.5, policy_stddev_, &output->policy);
    for (auto& p : output->policy) {
      p = std::exp(p);
    }
    float sum = 0;
    for (auto p : output->policy) {
      sum += p;
    }
    for (auto& p : output->policy) {
      p /= sum;
    }

    do {
      output->value = rnd_.NormalDistribution(0, value_stddev_);
    } while (output->value < -1 || output->value > 1);
  }
  if (model_name != nullptr) {
    *model_name = name();
  }
}

std::unique_ptr<Model> RandomDualNetFactory::NewModel(
    const ModelDefinition& def) {
  const auto& metadata = def.metadata;
  uint64_t seed = metadata.Get<uint64_t>("seed");
  float policy_stddev = metadata.Get<float>("policy_stddev");
  float value_stddev = metadata.Get<float>("value_stddev");
  auto name = absl::StrCat("rnd:", seed, ":", policy_stddev, ":", value_stddev);

  auto feature_desc = FeatureDescriptor::Create(metadata.Get<std::string>("input_features"));
  return absl::make_unique<RandomDualNet>(name, feature_desc, seed,
                                          policy_stddev, value_stddev);
}

}  // namespace minigo
