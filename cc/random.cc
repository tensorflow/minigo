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

#include "cc/random.h"

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

namespace minigo {

namespace {
uint64_t ChooseSeed(uint64_t seed) {
  return seed != 0 ? seed : absl::ToUnixMicros(absl::Now());
}
}  // namespace

Random::Random(uint64_t seed) : seed_(ChooseSeed(seed)), impl_(seed_) {}

void Random::Dirichlet(float alpha, absl::Span<float> samples) {
  std::gamma_distribution<float> distribution(alpha);

  float sum = 0;
  for (float& sample : samples) {
    sample = distribution(impl_);
    sum += sample;
  }
  float norm = 1 / sum;
  for (float& sample : samples) {
    sample *= norm;
  }
}

void Random::Uniform(float mn, float mx, absl::Span<float> samples) {
  std::uniform_real_distribution<float> distribution(mn, mx);
  for (float& sample : samples) {
    sample = distribution(impl_);
  }
}

}  // namespace minigo
