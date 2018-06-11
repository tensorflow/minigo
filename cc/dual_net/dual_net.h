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
#include "cc/position.h"
#include "cc/random.h"
#include "cc/symmetries.h"

namespace minigo {

// TODO(tommadams): Figure out a better way to handle random symmetries: each
// subclass of DualNet currently must copy-paste the same block of code to
// apply symmetries. Perhaps moving the symmetry code up to the calling code
// might yield cleaner code.

// The input features to the DualNet neural network have 17 binary feature
// planes. 8 feature planes X_t indicate the presence of the current player's
// stones at time t. A further 8 feature planes Y_t indicate the presence of
// the opposing player's stones at time t. The final feature plane C holds all
// 1s if black is to play, or 0s if white is to play. The planes are
// concatenated together to give input features:
//   [X_t, Y_t, X_t-1, Y_t-1, ..., X_t-7, Y_t-7, C].
class DualNet {
 public:
  // Number of features per stone.
  static constexpr int kNumStoneFeatures = 17;

  // Index of the per-stone feature that describes whether the black or white
  // player is to play next.
  static constexpr int kPlayerFeature = 16;

  // Total number of features for the board.
  static constexpr int kNumBoardFeatures = kN * kN * kNumStoneFeatures;

  using StoneFeatures = std::array<float, kNumStoneFeatures>;
  using BoardFeatures = std::array<float, kNumBoardFeatures>;

  // Initializes the input features so that the C feature plane is taken from
  // position.to_play(), and position.stones() are copied into all X and Y
  // feature planes (that is: X_t .. X_t-7 are identical and Y_t .. Y_t-7 are
  // identical).
  static void InitializeFeatures(const Position& position,
                                 BoardFeatures* features);

  // Updates the input features after the move position.previous_move() was
  // played.
  // old_features holds the input features for the network prior to
  // position.previous_move() being played.
  // position.stones() holds the board state after position.previous_move() was
  // played.
  // The updated input features for the network are written to new_features.
  // old_features and new_features are allowed to be the same object.
  static void UpdateFeatures(const BoardFeatures& old_features,
                             const Position& position,
                             BoardFeatures* new_features);

  struct Output {
    std::array<float, kNumMoves> policy;
    float value;
  };

  virtual ~DualNet();

  // Runs the model on a batch of input features.
  // If rnd != nullptr, the features will be randomly rotated and mirrored
  // before running the model, then the inverse transform applied to the
  // returned policy array.
  virtual void RunMany(absl::Span<const BoardFeatures* const> features,
                       absl::Span<Output> outputs, Random* rnd = nullptr) = 0;
};

}  // namespace minigo

#endif  // CC_DUAL_NET_DUAL_NET_H_
