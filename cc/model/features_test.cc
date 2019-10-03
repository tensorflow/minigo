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

#include "cc/model/features.h"

#include <memory>

#include "cc/model/types.h"
#include "gtest/gtest.h"

namespace minigo {
namespace {

// Define some fake features that don't do anything: their `kNumPlanes` are used
// in the compile-time tests.
struct Planes3 {
  static constexpr int kNumPlanes = 3;
  template <typename T>
  static void Set(const ModelInput&, int, T*) {}
};

struct Planes2 {
  static constexpr int kNumPlanes = 2;
  template <typename T>
  static void Set(const ModelInput&, int, T*) {}
};

struct Planes4 {
  static constexpr int kNumPlanes = 4;
  template <typename T>
  static void Set(const ModelInput&, int, T*) {}
};

struct NotRegistered {
  static constexpr int kNumPlanes = 1;
  template <typename T>
  static void Set(const ModelInput&, int, T*) {}
};

// Validate stuff that gets computed at compile-time.
namespace compile_time_tests {
using TestFeatures = Features<Planes3, Planes2, Planes4>;

static_assert(TestFeatures::kNumPlanes == 9, "");
static_assert(TestFeatures::GetPlaneIdx<Planes3>() == 0, "");
static_assert(TestFeatures::GetPlaneIdx<Planes2>() == 3, "");
static_assert(TestFeatures::GetPlaneIdx<Planes4>() == 5, "");
static_assert(TestFeatures::GetPlaneIdx<NotRegistered>() == -1, "");

// Define some lists of features.
using Features2 = Features<Planes2>;
using Features3 = Features<Planes3>;
using Features4 = Features<Planes4>;
using Features5 = Features<Planes2, Planes3>;
using Features6 = Features<Planes4, Planes2>;
using Features7 = Features<Planes4, Planes3>;
using Features9 = Features<Planes4, Planes2, Planes3>;

// Verify the compile-time calculation of the maximum number of required
// feature planes.
// clang-format off
static_assert(
    internal::GetMaxNumFeaturePlanes<Features2, Features3>() == 3,
    "");
static_assert(
    internal::GetMaxNumFeaturePlanes<Features2, Features4, Features3>() == 4,
    "");
static_assert(
    internal::GetMaxNumFeaturePlanes<Features2, Features5, Features4,
                                     Features3>() == 5,
    "");
static_assert(
    internal::GetMaxNumFeaturePlanes<Features2, Features5, Features4,
                                     Features3, Features6>() == 6,
    "");
static_assert(
    internal::GetMaxNumFeaturePlanes<Features2, Features5, Features4,
                                     Features7, Features3, Features6>() == 7,
    "");
static_assert(
    internal::GetMaxNumFeaturePlanes<Features9, Features2, Features5,
                                     Features4, Features7, Features3,
                                     Features6>() == 9,
    "");
// clang-format on
}  // namespace compile_time_tests

// Define some simple feature planes that set easily identifiable features.
struct Set1 {
  static constexpr int kNumPlanes = 1;

  template <typename T>
  static void Set(const ModelInput&, int num_planes, T* dst) {
    for (int i = 0; i < kN * kN; ++i) {
      dst[0] = 1000 + i;
      dst += num_planes;
    }
  }
};

struct Set2 {
  static constexpr int kNumPlanes = 2;

  template <typename T>
  static void Set(const ModelInput&, int num_planes, T* dst) {
    for (int i = 0; i < kN * kN; ++i) {
      dst[0] = 2000 + i;
      dst[1] = 3000 + i;
      dst += num_planes;
    }
  }
};

struct Set3 {
  static constexpr int kNumPlanes = 3;

  template <typename T>
  static void Set(const ModelInput&, int num_planes, T* dst) {
    for (int i = 0; i < kN * kN; ++i) {
      dst[0] = 4000 + i;
      dst[1] = 5000 + i;
      dst[2] = 6000 + i;
      dst += num_planes;
    }
  }
};

// Verify the `Features::Set` sets all of the features listed in its type list.
TEST(FeaturesTest, TestSet) {
  using TestFeatures = Features<Set2, Set3, Set1>;

  ModelInput input;

  constexpr int kBatchSize = 3;

  // The inputs aren't used by any of our test features so we don't need to
  // initialize them to anything meaningful.
  std::vector<const ModelInput*> inputs;
  for (int i = 0; i < kBatchSize; ++i) {
    inputs.push_back(&input);
  }

  // Allocate a feature tensor.
  BackedTensor<int> features;
  features.resize(kBatchSize, kN, kN, TestFeatures::kNumPlanes);

  // Set the test input features.
  TestFeatures::Set(inputs, &features.tensor());

  // Verify the input features.
  for (int j = 0, feature = 0; feature < kBatchSize; ++feature) {
    for (int i = 0; i < kN * kN; ++i) {
      // Features from Set2.
      EXPECT_EQ(2000 + i, features.tensor().data[j++]);
      EXPECT_EQ(3000 + i, features.tensor().data[j++]);

      // Features from Set3.
      EXPECT_EQ(4000 + i, features.tensor().data[j++]);
      EXPECT_EQ(5000 + i, features.tensor().data[j++]);
      EXPECT_EQ(6000 + i, features.tensor().data[j++]);

      // Features from Set1.
      EXPECT_EQ(1000 + i, features.tensor().data[j++]);
    }
  }
}

}  // namespace
}  // namespace minigo
