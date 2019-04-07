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

namespace minigo {

RandomDualNet::RandomDualNet(std::string name, uint64_t seed,
                             float policy_stddev, float value_stddev)
    : DualNet(std::move(name)),
      rnd_(seed),
      policy_stddev_(policy_stddev),
      value_stddev_(value_stddev) {}

void RandomDualNet::RunMany(std::vector<const BoardFeatures*> features,
                            std::vector<Output*> outputs, std::string* model) {
  for (auto* output : outputs) {
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
  if (model != nullptr) {
    *model = name();
  }
}

RandomDualNetFactory::RandomDualNetFactory(uint64_t seed) : rnd_(seed) {}

std::unique_ptr<DualNet> RandomDualNetFactory::NewDualNet(
    const std::string& model) {
  std::vector<absl::string_view> parts = absl::StrSplit(model, ':');
  MG_CHECK(parts.size() == 2);

  float policy_stddev, value_stddev;
  MG_CHECK(absl::SimpleAtof(parts[0], &policy_stddev));
  MG_CHECK(absl::SimpleAtof(parts[1], &value_stddev));

  uint64_t seed;
  {
    absl::MutexLock lock(&mutex_);
    seed = rnd_.UniformUint64();
  }
  return absl::make_unique<RandomDualNet>(absl::StrCat("rnd:", model), seed,
                                          policy_stddev, value_stddev);
}

}  // namespace minigo
