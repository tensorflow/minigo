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

#ifndef CC_MCTS_PLAYER_H_
#define CC_MCTS_PLAYER_H_

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "cc/algorithm.h"
#include "cc/constants.h"
#include "cc/dual_net.h"
#include "cc/mcts_node.h"
#include "cc/position.h"
#include "cc/random.h"

namespace minigo {

class MctsPlayer {
 public:
  struct Options {
    bool inject_noise = true;
    bool soft_pick = true;
    bool random_symmetry = true;

    float resign_threshold = -0.9;

    int batch_size = 8;

    float komi = kDefaultKomi;

    // Seed used from random permutations.
    // If the default value of 0 is used, a time-based seed is used.
    uint64_t random_seed = 0;
  };

  struct History {
    std::array<float, kNumMoves> search_pi;
    Coord c = Coord::kPass;
    std::string comment;
    const MctsNode* node = nullptr;
  };

  enum class GameOverReason {
    kNone,
    kOpponentResigned,
    kBothPassed,
    kMoveLimitReached,
  };

  // If position is non-null, the player will be initilized with that board
  // state. Otherwise, the player is initialized with an empty board with black
  // to play.
  MctsPlayer(DualNet* network, const Options& options);

  void InitializeGame(const Position& position);

  virtual ~MctsPlayer() = default;

  void SelfPlay(int num_readouts);

  const Options& options() const { return options_; }
  float result() const { return result_; }
  const std::string& result_string() const { return result_string_; }
  const std::vector<History>& history() const { return history_; }

  // These methods are protected to facilitate direct testing.
 protected:
  Coord PickMove();

  void TreeSearch(int batch_size);

  void PlayMove(Coord c);

  void SetResult(float result, float score, GameOverReason reason);

  DualNet::Output Run(const DualNet::BoardFeatures* features);

  // Returns the root of the current search tree, i.e. the current board state.
  MctsNode* root() { return root_; }
  const MctsNode* root() const { return root_; }

  // Returns the root of the game tree.
  MctsNode* game_root() { return &game_root_; }
  const MctsNode* game_root() const { return &game_root_; }

  Random* rnd() { return &rnd_; }

 private:
  void PushHistory(Coord c);

  bool ShouldResign() const;

  void RunMany(absl::Span<const DualNet::BoardFeatures* const> features,
               absl::Span<DualNet::Output> outputs);

  DualNet* network_;
  int temperature_cutoff_;

  MctsNode::EdgeStats dummy_stats_;

  MctsNode* root_;
  MctsNode game_root_;

  BoardVisitor bv_;
  GroupVisitor gv_;

  Random rnd_;

  // Game result value:
  //   1: win for black.
  //  -1: win for white.
  //   0: draw.
  float result_ = 0;

  // Score of the final board position using Tromp-Taylor scoring.
  float score_ = 0;

  // Reason that the game ended.
  GameOverReason game_over_reason_ = GameOverReason::kNone;

  // Description of the result.
  std::string result_string_;

  Options options_;

  std::vector<History> history_;
};

}  // namespace minigo

#endif  // CC_MCTS_PLAYER_H_
