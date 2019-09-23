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

#ifndef CC_DUAL_NET_DUAL_NET_H_
#define CC_DUAL_NET_DUAL_NET_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "cc/constants.h"
#include "cc/model/model.h"
#include "cc/position.h"
#include "cc/symmetries.h"

namespace minigo {

// The input features to the DualNet neural network have 17 binary feature
// planes. 8 feature planes X_t indicate the presence of the current player's
// stones at time t. A further 8 feature planes Y_t indicate the presence of
// the opposing player's stones at time t. The final feature plane C holds all
// 1s if black is to play, or 0s if white is to play. The planes are
// concatenated together to give input features:
//   [X_t, Y_t, X_t-1, Y_t-1, ..., X_t-7, Y_t-7, C].
class DualNet : public Model {
 public:
  // Size of move history in the stone features.
  static constexpr int kMoveHistory = 8;

  // Index of the per-stone feature that describes whether the black or white
  // player is to play next.
  static constexpr int kPlayerFeature = kMoveHistory * 2;

  // Number of features per stone.
  static constexpr int kNumAgzStoneFeatures = kMoveHistory * 2 + 1;

  static constexpr int kNumLibertyFeatures = 3;

  static constexpr int kNumExtraStoneFeatures =
      kNumAgzStoneFeatures + kNumLibertyFeatures;

  static constexpr int kMaxBoardFeaturesSize = kN * kN * kNumExtraStoneFeatures;

  using Input = Model::Input;
  using Output = Model::Output;

  // A buffer large enough to hold features for all input types.
  template <typename T>
  using BoardFeatureBuffer = std::array<T, kMaxBoardFeaturesSize>;

  static void SetFeatures(const std::vector<const Input*>& model_inputs,
                          FeatureType feature_type, Tensor* features);

  static void GetOutputs(const std::vector<const Input*>& model_inputs,
                         const Tensor& policy, const Tensor& value,
                         std::vector<Output*>* model_outputs);

  DualNet(std::string name, FeatureType feature_type, int buffer_count)
      : Model(std::move(name), feature_type, buffer_count) {}
  ~DualNet() override;
};

}  // namespace minigo

#endif  // CC_DUAL_NET_DUAL_NET_H_
