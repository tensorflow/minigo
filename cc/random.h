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
  static constexpr uint64_t kUniqueSeed = 0;
  static constexpr int kUniqueStream = 0;

  // The implementation supports generating multiple streams of uncorrelated
  // random numbers from a single seed.
  // If seed == Random::kUniqueSeed, a seed will be chosen from the platform's
  // random entropy source.
  // If stream == Random::kUniqueStream, a stream will be chosen from a
  // thread-safe global incrementing ID.
  // It's recommended that for reproducible results (modulo threading timing),
  // all Random instances use a seed speficied by a flag, and
  // Random::kUniqueStream for the stream.
  explicit Random(uint64_t seed, int stream);

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

  // Samples the given CDF at random, returning the index of the element found.
  // Guarantees that elements with zero probability
  int SampleCdf(absl::Span<float> cdf);

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
  int stream() const { return static_cast<int>(impl_.inc >> 1); }

 private:
  // The implementation is based on 32bit PCG Random:
  //   http://www.pcg-random.org/
  struct Impl {
    using result_type = uint32_t;
    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return 0xffffffff; }

    Impl(uint64_t seed, int stream)
        : state(0), inc((static_cast<uint64_t>(stream) << 1) | 1) {
      operator()();
      state += seed;
      operator()();
    }

    result_type operator()() {
      auto old_state = state;
      state = old_state * 6364136223846793005ULL + inc;
      uint32_t xor_shifted = ((old_state >> 18u) ^ old_state) >> 27u;
      uint32_t rot = old_state >> 59u;
      auto result = (xor_shifted >> rot) | (xor_shifted << ((-rot) & 31));
      return result;
    }

    uint64_t state;
    const uint64_t inc;
  };

  uint64_t seed_;
  Impl impl_;
};

}  // namespace minigo

#endif  // CC_RANDOM_H_
