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

#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "cc/constants.h"
#include "cc/init.h"
#include "cc/logging.h"
#include "cc/model/features.h"
#include "cc/model/types.h"
#include "cc/position.h"
#include "cc/random.h"

namespace minigo {

template <typename T>
void BenchmarkFeatures(absl::string_view input_features,
                       absl::string_view input_layout,
                       absl::string_view input_type) {
  constexpr int kBatchSize = 50000;

  const auto& desc = FeatureDescriptor::Create(input_features, input_layout);
  auto shape = desc.GetInputShape(kBatchSize);

  Random rnd(23423, 23454);

  std::vector<Position> positions;
  for (int i = 0; i < kBatchSize; ++i) {
    positions.emplace_back(Color::kBlack);
  }

  std::vector<ModelInput> inputs;
  for (int i = 0; i < kBatchSize; ++i) {
    ModelInput input;
    input.sym = static_cast<symmetry::Symmetry>(rnd.UniformUint64() %
                                                symmetry::kNumSymmetries);
    input.sym = symmetry::kIdentity;
    for (int j = 0; j < kMaxPositionHistory; ++j) {
      input.position_history.push_back(
          &positions[rnd.UniformUint64() % positions.size()]);
    }
    inputs.push_back(input);
  }

  std::vector<const ModelInput*> input_ptrs;
  for (int i = 0; i < kBatchSize; ++i) {
    input_ptrs.push_back(&inputs[i]);
  }

  BackedTensor<T> features(shape);
  auto start = absl::Now();
  desc.SetFeatures(input_ptrs, &features.tensor());
  auto duration = absl::Now() - start;

  MG_LOG(INFO) << kN << "x" << kN << ":" << input_features << ":"
               << input_layout << ":" << input_type << " " << duration;
}

void RunBenchmark() {
  const char* features[] = {"agz", "mlperf07"};
  const char* layouts[] = {"nhwc", "nchw"};

  for (const char* f : features) {
    for (const char* l : layouts) {
      BenchmarkFeatures<float>(f, l, "float");
    }
    for (const char* l : layouts) {
      BenchmarkFeatures<uint8_t>(f, l, "uint8");
    }
    MG_LOG(INFO) << "";
  }
}

}  // namespace minigo

int main(int argc, char* argv[]) {
  minigo::Init(&argc, &argv);
  minigo::RunBenchmark();
  return 0;
}
