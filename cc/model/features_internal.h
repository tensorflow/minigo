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

#ifndef CC_MODEL_FEATURES_INTERNAL_H_
#define CC_MODEL_FEATURES_INTERNAL_H_

#include <type_traits>

#include "cc/model/types.h"

// Internal implementation details of the features code.

namespace minigo {
namespace internal {

// `FeaturesImpl` calls the `Set` static method on each of the feature types in
// `Fs`.
template <typename... Fs>
struct FeaturesImpl;

// Recursive definition of `FeaturesImpl`. The recursion is expanded at compile
// time.
template <typename First, typename... Rest>
struct FeaturesImpl<First, Rest...> {
  static constexpr int kNumPlanes =
      First::kNumPlanes + FeaturesImpl<Rest...>::kNumPlanes;

  template <typename T>
  static void SetAll(const ModelInput& input, int stride, T* dst) {
    First::Set(input, stride, dst);
    dst += First::kNumPlanes;
    FeaturesImpl<Rest...>::SetAll(input, stride, dst);
  }

  template <typename FeatureType>
  static constexpr int GetPlaneIdx(int idx) {
    return std::is_same<FeatureType, First>::value
               ? idx
               : FeaturesImpl<Rest...>::template GetPlaneIdx<FeatureType>(
                     idx + First::kNumPlanes);
  }
};

// `FeaturesImpl` base case that stops the recursive template expansion.
template <>
struct FeaturesImpl<> {
  static constexpr int kNumPlanes = 0;

  template <typename T>
  static void SetAll(const ModelInput& input, int stride, T* dst) {}

  template <typename T>
  static constexpr int GetPlaneIdx(int) {
    return -1;
  }
};

// Helper struct to compute the maximum number of feature planes used by any
// set of input features.
template <typename... Fs>
struct MaxNumFeaturePlanes;

template <typename First, typename... Rest>
struct MaxNumFeaturePlanes<First, Rest...> {
  // std::max wasn't made constexpr until C++14 and Minigo is still on C++11.
  static constexpr int kValue =
      First::kNumPlanes > MaxNumFeaturePlanes<Rest...>::kValue
          ? First::kNumPlanes
          : MaxNumFeaturePlanes<Rest...>::kValue;
};

// `MaxNumFeaturePlanes` recursive base case.
template <>
struct MaxNumFeaturePlanes<> {
  static constexpr int kValue = 0;
};

// Calculates at compile time the maximum number of planes in the set of input
// features `Fs`.
template <typename... Fs>
constexpr int GetMaxNumFeaturePlanes() {
  return MaxNumFeaturePlanes<Fs...>::kValue;
}

}  // namespace internal
}  // namespace minigo

#endif  //  CC_MODEL_FEATURES_INTERNAL_H_
