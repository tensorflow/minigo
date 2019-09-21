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

#include "cc/dual_net/fake_dual_net.h"

#include "absl/memory/memory.h"
#include "cc/logging.h"

namespace minigo {

FakeDualNet::FakeDualNet(absl::Span<const float> priors, float value)
    : Model("fake", Model::FeatureType::kAgz, 1), value_(value) {
  if (!priors.empty()) {
    MG_CHECK(priors.size() == kNumMoves);
    for (int i = 0; i < kNumMoves; ++i) {
      priors_[i] = priors[i];
    }
  } else {
    for (auto& prior : priors_) {
      prior = 1.0 / kNumMoves;
    }
  }
}

void FakeDualNet::RunMany(const std::vector<const Input*>& inputs,
                          std::vector<Output*>* outputs, std::string* model) {
  for (auto* output : *outputs) {
    output->policy = priors_;
    output->value = value_;
  }
  if (model != nullptr) {
    *model = "FakeDualNet";
  }
}

std::unique_ptr<Model> FakeDualNetFactory::NewModel(
    const std::string& descriptor) {
  return absl::make_unique<FakeDualNet>();
}

}  // namespace minigo
