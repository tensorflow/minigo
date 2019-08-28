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

#include <atomic>

namespace minigo {

namespace {
std::atomic<int> unique_stream_id{0};

uint64_t ChooseSeed(uint64_t seed) {
  if (seed == 0) {
    std::random_device rd;
    seed = rd();
    if (sizeof(std::random_device::result_type) < 8) {
      seed = (seed << 32) | rd();
    }
  }
  return seed;
}

int ChooseStream(int stream) {
  if (stream == 0) {
    stream = unique_stream_id.fetch_add(1);
  }
  return stream;
}
}  // namespace

Random::Random(uint64_t seed, int stream)
    : seed_(ChooseSeed(seed)), impl_(seed_, ChooseStream(stream)) {}

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

float Random::NormalDistribution(float mean, float stddev) {
  return std::normal_distribution<float>(mean, stddev)(impl_);
}

void Random::NormalDistribution(float mean, float stddev,
                                absl::Span<float> samples) {
  std::normal_distribution<float> distribution(mean, stddev);
  for (float& sample : samples) {
    sample = distribution(impl_);
  }
}

int Random::SampleCdf(absl::Span<float> cdf) {
  // Take care to handle the case where the first elements in the CDF have zero
  // probability: discard any 0.0 values that the random number generator
  // produces. Admittedly, this isn't going to happen very often.
  float e;
  do {
    e = operator()();
  } while (e == 0);

  float x = cdf.back() * e;
  return std::distance(cdf.begin(),
                       std::lower_bound(cdf.begin(), cdf.end(), x));
}

}  // namespace minigo
