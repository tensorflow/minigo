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
#include "cc/position.h"
#include "cc/random.h"
#include "cc/symmetries.h"
#include "cc/test_utils.h"
#include "gtest/gtest.h"

namespace minigo {
namespace {

// Define some fake features that don't do anything: their `kNumPlanes` are used
// in the compile-time tests.
struct Planes3 {
  static constexpr int kNumPlanes = 3;
  template <typename T>
  static void SetNhwc(const ModelInput&, int, T*) {}
  template <typename T>
  static void SetNchw(const ModelInput&, T*) {}
};

struct Planes2 {
  static constexpr int kNumPlanes = 2;
  template <typename T>
  static void SetNhwc(const ModelInput&, int, T*) {}
  template <typename T>
  static void SetNchw(const ModelInput&, T*) {}
};

struct Planes4 {
  static constexpr int kNumPlanes = 4;
  template <typename T>
  static void SetNhwc(const ModelInput&, int, T*) {}
  template <typename T>
  static void SetNchw(const ModelInput&, T*) {}
};

struct NotRegistered {
  static constexpr int kNumPlanes = 1;
  template <typename T>
  static void SetNhwc(const ModelInput&, int, T*) {}
  template <typename T>
  static void SetNchw(const ModelInput&, T*) {}
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
  static void SetNhwc(const ModelInput&, int num_planes, T* dst) {
    for (int i = 0; i < kN * kN; ++i) {
      dst[0] = 1000 + i;
      dst += num_planes;
    }
  }

  template <typename T>
  static void SetNchw(const ModelInput&, T* dst) {
    for (int i = 0; i < kN * kN; ++i) {
      *dst++ = 1000 + i;
    }
  }
};

struct Set2 {
  static constexpr int kNumPlanes = 2;

  template <typename T>
  static void SetNhwc(const ModelInput&, int num_planes, T* dst) {
    for (int i = 0; i < kN * kN; ++i) {
      dst[0] = 2000 + i;
      dst[1] = 3000 + i;
      dst += num_planes;
    }
  }

  template <typename T>
  static void SetNchw(const ModelInput&, T* dst) {
    for (int i = 0; i < kN * kN; ++i) {
      *dst++ = 2000 + i;
    }
    for (int i = 0; i < kN * kN; ++i) {
      *dst++ = 3000 + i;
    }
  }
};

struct Set3 {
  static constexpr int kNumPlanes = 3;

  template <typename T>
  static void SetNhwc(const ModelInput&, int num_planes, T* dst) {
    for (int i = 0; i < kN * kN; ++i) {
      dst[0] = 4000 + i;
      dst[1] = 5000 + i;
      dst[2] = 6000 + i;
      dst += num_planes;
    }
  }

  template <typename T>
  static void SetNchw(const ModelInput&, T* dst) {
    for (int i = 0; i < kN * kN; ++i) {
      *dst++ = 4000 + i;
    }
    for (int i = 0; i < kN * kN; ++i) {
      *dst++ = 5000 + i;
    }
    for (int i = 0; i < kN * kN; ++i) {
      *dst++ = 6000 + i;
    }
  }
};

// Verify the `Features::SetNhwc` sets all of the features listed in its type
// list.
TEST(FeaturesTest, TestSetNhwc) {
  using TestFeatures = Features<Set2, Set3, Set1>;

  // Test all symmetries.
  for (int sym_idx = 0; sym_idx < symmetry::kNumSymmetries; ++sym_idx) {
    auto sym = static_cast<symmetry::Symmetry>(sym_idx);

    ModelInput input;
    input.sym = sym;

    constexpr int kBatchSize = 3;

    // The inputs aren't used by any of our test features so we don't need to
    // initialize them to anything meaningful.
    std::vector<const ModelInput*> inputs;
    for (int i = 0; i < kBatchSize; ++i) {
      inputs.push_back(&input);
    }

    // Allocate a feature tensor.
    BackedTensor<int> features;
    features.resize({kBatchSize, kN, kN, TestFeatures::kNumPlanes});

    // Set the test input features.
    TestFeatures::SetNhwc(inputs, &features.tensor());

    // Verify the input features.
    for (int input = 0; input < kBatchSize; ++input) {
      int base = input * kN * kN * TestFeatures::kNumPlanes;
      for (int row = 0; row < kN; ++row) {
        for (int col = 0; col < kN; ++col) {
          // Coordinate before symmetry is applied.
          Coord i = Coord(row, col);

          // Apply symmetry to get the coordinate with which we should read the
          // input feature tensor.
          Coord c = symmetry::ApplySymmetry(sym, i);

          // Calculate the index of the start of the features for this stone.
          int idx = base + c * TestFeatures::kNumPlanes;

          // Features from Set2.
          EXPECT_EQ(2000 + i, features.tensor().data[idx + 0]);
          EXPECT_EQ(3000 + i, features.tensor().data[idx + 1]);

          // Features from Set3.
          EXPECT_EQ(4000 + i, features.tensor().data[idx + 2]);
          EXPECT_EQ(5000 + i, features.tensor().data[idx + 3]);
          EXPECT_EQ(6000 + i, features.tensor().data[idx + 4]);

          // Features from Set1.
          EXPECT_EQ(1000 + i, features.tensor().data[idx + 5]);
        }
      }
    }
    break;
  }
}

// Verify the `Features::SetNchw` sets all of the features listed in its type
// list.
TEST(FeaturesTest, TestSetNchw) {
  using TestFeatures = Features<Set2, Set3, Set1>;

  // Test all symmetries.
  for (int sym_idx = 0; sym_idx < symmetry::kNumSymmetries; ++sym_idx) {
    auto sym = static_cast<symmetry::Symmetry>(sym_idx);

    ModelInput input;
    input.sym = sym;

    constexpr int kBatchSize = 3;

    // The inputs aren't used by any of our test features so we don't need to
    // initialize them to anything meaningful.
    std::vector<const ModelInput*> inputs;
    for (int i = 0; i < kBatchSize; ++i) {
      inputs.push_back(&input);
    }

    // Allocate a feature tensor.
    BackedTensor<int> features;
    features.resize({kBatchSize, TestFeatures::kNumPlanes, kN, kN});

    // Set the test input features.
    TestFeatures::SetNchw(inputs, &features.tensor());

    // Verify the input features.
    for (int input = 0; input < kBatchSize; ++input) {
      int base = input * kN * kN * TestFeatures::kNumPlanes;
      for (int row = 0; row < kN; ++row) {
        for (int col = 0; col < kN; ++col) {
          // Coordinate before symmetry is applied.
          Coord i = Coord(row, col);

          // Apply symmetry to get the coordinate with which we should read the
          // input feature tensor.
          Coord c = symmetry::ApplySymmetry(sym, i);

          // Calculate the index of the start of the features for this stone.
          int idx = base + c;

          // Features from Set2.
          EXPECT_EQ(2000 + i, features.tensor().data[idx + 0]);
          EXPECT_EQ(3000 + i, features.tensor().data[idx + kN * kN]);

          // Features from Set3.
          EXPECT_EQ(4000 + i, features.tensor().data[idx + 2 * kN * kN]);
          EXPECT_EQ(5000 + i, features.tensor().data[idx + 3 * kN * kN]);
          EXPECT_EQ(6000 + i, features.tensor().data[idx + 4 * kN * kN]);

          // Features from Set1.
          EXPECT_EQ(1000 + i, features.tensor().data[idx + 5 * kN * kN]);
        }
      }
    }
    break;
  }
}

// Verify that generated NHWC and NCHW features are equivalent.
TEST(FeaturesTest, CompareNhwcNchw) {
  Random rnd(454, 43263);
  constexpr int kBatchSize = 3;

  // Generate some random positions.
  std::vector<TestablePosition> positions;
  for (int n = 0; n < kBatchSize; ++n) {
    TestablePosition position("");
    for (int i = 0; i < kN * kN; ++i) {
      auto c = GetRandomLegalMove(position, &rnd);
      position.PlayMove(c);
    }
    positions.push_back(std::move(position));
  }

  // Create model inputs.
  std::vector<ModelInput> inputs;
  for (int n = 0; n < kBatchSize; ++n) {
    ModelInput input;
    input.sym = symmetry::kIdentity;
    for (int i = 0; i <= n; ++i) {
      input.position_history.push_back(&positions[i]);
    }
    inputs.push_back(std::move(input));
  }
  std::vector<ModelInput*> input_ptrs;
  for (int n = 0; n < kBatchSize; ++n) {
    input_ptrs.push_back(&inputs[n]);
  }

  // Generate NHWC features.
  auto desc_nhwc = FeatureDescriptor::Create<Mlperf07Features>(
      FeatureDescriptor::Layout::kNhwc);
  auto shape_nhwc = desc_nhwc.GetInputShape(kBatchSize);
  BackedTensor<uint8_t> features_nhwc(shape_nhwc);
  desc_nhwc.set_bytes(input_ptrs, &features_nhwc.tensor());

  // Generate NCHW features.
  auto desc_nchw = FeatureDescriptor::Create<Mlperf07Features>(
      FeatureDescriptor::Layout::kNchw);
  auto shape_nchw = desc_nchw.GetInputShape(kBatchSize);
  BackedTensor<uint8_t> features_nchw(shape_nchw);
  desc_nchw.set_bytes(input_ptrs, &features_nchw.tensor());

  // Verify features are equivalent.
  auto np = Mlperf07Features::kNumPlanes;
  for (int n = 0; n < kBatchSize; ++n) {
    for (int j = 0; j < kN; ++j) {
      for (int i = 0; i < kN; ++i) {
        for (int c = 0; c < np; ++c) {
          auto nhwc_idx = c + np * (i + kN * (j + kN * n));
          auto nchw_idx = i + kN * (j + kN * (c + np * n));
          auto nhwc = features_nhwc.tensor().data[nhwc_idx];
          auto nchw = features_nchw.tensor().data[nchw_idx];
          EXPECT_EQ(nhwc, nchw);
          // MG_LOG(INFO) << n << ":" << j << ":" << i << ":" << c << "  "
          //              << static_cast<int>(nhwc) << ":"
          //              << static_cast<int>(nchw) << "  " << nhwc_idx
          //              << ":" << nchw_idx;
        }
      }
    }
  }
}

}  // namespace
}  // namespace minigo
