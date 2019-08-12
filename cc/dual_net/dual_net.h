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

  // Number of features per stone.
  static constexpr int kNumStoneFeatures = kMoveHistory * 2 + 1;

  // Index of the per-stone feature that describes whether the black or white
  // player is to play next.
  static constexpr int kPlayerFeature = kMoveHistory * 2;

  // Total number of features for the board.
  static constexpr int kNumBoardFeatures = kN * kN * kNumStoneFeatures;

  // TODO(tommadams): Change features type from float to uint8_t.
  using StoneFeatures = std::array<float, kNumStoneFeatures>;
  using BoardFeatures = std::array<float, kNumBoardFeatures>;

  using Output = Model::Output;

  // Generates the board features from the history of recent moves, where
  // history[0] is the current board position, and history[i] is the board
  // position from i moves ago.
  // history.size() must be <= kMoveHistory.
  // TODO(tommadams): Move Position::Stones out of the Position class so we
  // don't need to depend on position.h.
  static void SetFeatures(absl::Span<const Position::Stones* const> history,
                          Color to_play, BoardFeatures* features);

  explicit DualNet(std::string name) : name_(std::move(name)) {}
  virtual ~DualNet();

  const std::string& name() const { return name_; }

  void RunMany(absl::Span<const Position*> position_history,
               std::vector<Output*> outputs, std::string* model_name) override;

  // Potentially prepares the DualNet to avoid expensive operations during
  // RunMany() calls with up to 'capacity' features.
  virtual void Reserve(size_t capacity);

 private:
  // Runs inference on a batch of input features.
  // TODO(tommadams): rename model -> model_name.
  virtual void RunMany(std::vector<const BoardFeatures*> features,
                       std::vector<Output*> outputs,
                       std::string* model_name) = 0;
  const std::string name_;
};

// Factory that creates DualNet instances.
// All implementations are required to be thread safe.
class DualNetFactory {
 public:
  virtual ~DualNetFactory();

  // Returns the ideal number of inference requests in flight for DualNet
  // instances created by this factory.
  virtual int GetBufferCount() const;

  virtual std::unique_ptr<DualNet> NewDualNet(
      const std::string& model_path) = 0;
};

}  // namespace minigo

#endif  // CC_DUAL_NET_DUAL_NET_H_
