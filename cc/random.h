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

#ifndef CC_RANDOM_H_
#define CC_RANDOM_H_

#include <cstdint>
#include <random>

#include "absl/types/span.h"

namespace minigo {

// The C++ random library functionality is about as user friendly as its time
// library.
class Random {
 public:
  explicit Random(uint64_t seed = 0);

  // Draw samples from a Dirichlet distribution.
  void Dirichlet(float alpha, absl::Span<float> samples);

  // Draw samples from a Dirichlet distribution.
  template <typename T>
  void Dirichlet(float alpha, T* array_like) {
    Dirichlet(alpha, {array_like->data(), array_like->size()});
  }

  // Draw multiple unform random samples in the half-open range [mn, mx).
  void Uniform(float mn, float mx, absl::Span<float> samples);

  // Draw multiple unform random samples in the half-open range [mn, mx).
  template <typename T>
  void Uniform(float mn, float mx, T* array_like) {
    Uniform(mn, mx, {array_like->data(), array_like->size()});
  }

  // Draw multiple unform random samples in the half-open range [0, 1).
  template <typename T>
  void Uniform(T* array_like) {
    Uniform(0, 1, array_like);
  }

  // Draw multiple random samples from a normal distribution.
  template <typename T>
  void NormalDistribution(float mean, float stddev, T* array_like) {
    NormalDistribution(mean, stddev, {array_like->data(), array_like->size()});
  }

  // Draw multiple random samples from a normal distribution.
  void NormalDistribution(float mean, float stddev, absl::Span<float> samples);

  // Draw a single random sample from a normal distribution.
  float NormalDistribution(float mean, float stddev);

  // Returns a uniform random integer in the closed range [mn, mx].
  int UniformInt(int mn, int mx) {
    std::uniform_int_distribution<int> distribution(mn, mx);
    return distribution(impl_);
  }

  uint64_t UniformUint64() {
    uint64_t a = impl_();
    uint64_t b = impl_();
    return (a << 32) | b;
  }

  // Returns a uniform random number in the half-open range [0, 1).
  float operator()() {
    return std::uniform_real_distribution<float>(0, 1)(impl_);
  }

  uint64_t seed() const { return seed_; }

 private:
  uint64_t seed_;
  std::mt19937 impl_;
};

}  // namespace minigo

#endif  // CC_RANDOM_H_
