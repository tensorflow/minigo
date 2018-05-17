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
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "cc/algorithm.h"
#include "cc/constants.h"
#include "cc/dual_net/dual_net.h"
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
    float resign_threshold = -0.95;
    int batch_size = 8;
    float komi = kDefaultKomi;

    // Seed used from random permutations.
    // If the default value of 0 is used, a time-based seed is used.
    uint64_t random_seed = 0;

    friend std::ostream& operator<<(std::ostream& ios, const Options& options);
  };

  struct History {
    std::array<float, kNumMoves> search_pi;
    Coord c = Coord::kPass;
    std::string comment;
    const MctsNode* node = nullptr;
  };

  // If position is non-null, the player will be initilized with that board
  // state. Otherwise, the player is initialized with an empty board with black
  // to play.
  MctsPlayer(std::unique_ptr<DualNet> network, const Options& options);

  virtual ~MctsPlayer() = default;

  void InitializeGame(const Position& position);

  void NewGame();

  virtual Coord SuggestMove(int num_readouts);

  void PlayMove(Coord c);

  bool ShouldResign() const;

  // Returns the root of the current search tree, i.e. the current board state.
  MctsNode* root() { return root_; }
  const MctsNode* root() const { return root_; }

  // Returns true if the game is over, either because both players passed, one
  // player resigned, or the game reached the maximum number of allowed moves.
  bool game_over() const { return game_over_; }

  // Returns the result of the game:
  //   +1.0 if black won.
  //    0.0 if the game was drawn.
  //   -1.0 if white won.
  // Check fails if the game is not yet over.
  float result() const {
    MG_CHECK(game_over_);
    return result_;
  }

  // Return a text description of the game result, e.g. "B+R", "W+1.5".
  // Check fails if the game is not yet over.
  const std::string& result_string() const {
    MG_CHECK(game_over_);
    return result_string_;
  }

  const Options& options() const { return options_; }
  const std::vector<History>& history() const { return history_; }

  // These methods are protected to facilitate direct testing.
 protected:
  Coord PickMove();

  // Returns the list of nodes that TreeSearch performed inference on.
  // The contents of the returned Span is valid until the next call TreeSearch.
  virtual absl::Span<MctsNode* const> TreeSearch(int batch_size);

  DualNet::Output Run(const DualNet::BoardFeatures* features);

  // Returns the root of the game tree.
  MctsNode* game_root() { return &game_root_; }
  const MctsNode* game_root() const { return &game_root_; }

  Random* rnd() { return &rnd_; }

  std::string FormatScore(float score) const;

 private:
  void PushHistory(Coord c);

  void RunMany(absl::Span<const DualNet::BoardFeatures* const> features,
               absl::Span<DualNet::Output> outputs);

  std::unique_ptr<DualNet> network_;
  int temperature_cutoff_;

  MctsNode::EdgeStats dummy_stats_;

  MctsNode* root_;
  MctsNode game_root_;

  BoardVisitor bv_;
  GroupVisitor gv_;

  Random rnd_;

  Options options_;

  float result_ = 0;
  std::string result_string_;
  bool game_over_ = false;

  std::vector<History> history_;

  // Vectors reused when running TreeSearch.
  std::vector<MctsNode*> leaves_;
  std::vector<const DualNet::BoardFeatures*> features_;
  std::vector<DualNet::Output> outputs_;
};

}  // namespace minigo

#endif  // CC_MCTS_PLAYER_H_
