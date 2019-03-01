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

#include "absl/memory/memory.h"
#include "cc/logging.h"

namespace minigo {

RandomDualNet::RandomDualNet(uint64_t seed) : rnd_(seed) {}

void RandomDualNet::RunMany(std::vector<const DualNet::BoardFeatures*> features,
                          std::vector<Output*> outputs, std::string* model) {
  for (auto* output : outputs) {
    rnd_.Uniform(0, 1, &output->policy);
    float sum = 0;
    for (const auto& p : output->policy) {
      sum += p;
    }
    float scale = 1.0 / sum;
    for (auto& p : output->policy) {
      p *= scale;
    }
    output->value = 2.0 * rnd_() - 1;
  }
  if (model != nullptr) {
    *model = "RandomDualNet";
  }
}

RandomDualNetFactory::RandomDualNetFactory(uint64_t seed) : rnd_(13 * seed) {}

std::unique_ptr<DualNet> RandomDualNetFactory::NewDualNet(
    const std::string& model) {
  uint64_t seed;
  {
    absl::MutexLock lock(&mutex_);
    seed = rnd_.UniformUint64();
  }
  return absl::make_unique<RandomDualNet>(seed);
}

}  // namespace minigo

