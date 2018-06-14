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

#include "cc/dual_net/fake_net.h"

#include "cc/check.h"

namespace minigo {

FakeNet::FakeNet(absl::Span<const float> priors, float value) : value_(value) {
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

void FakeNet::RunMany(absl::Span<const BoardFeatures> features,
                      absl::Span<Output> outputs) {
  for (auto& output : outputs) {
    output.policy = priors_;
    output.value = value_;
  }
}

}  // namespace minigo
