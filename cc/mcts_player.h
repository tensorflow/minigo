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
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "cc/algorithm.h"
#include "cc/constants.h"
#include "cc/dual_net/dual_net.h"
#include "cc/mcts_node.h"
#include "cc/position.h"
#include "cc/random.h"
#include "cc/symmetries.h"

namespace minigo {

// Exposed for testing.
float TimeRecommendation(int move_num, float seconds_per_move, float time_limit,
                         float decay_factor);

class MctsPlayer {
 public:
  struct Options {
    bool inject_noise = true;
    bool soft_pick = true;
    bool random_symmetry = true;
    float resign_threshold = -0.95;

    // We use a separate resign_enabled flag instead of setting the
    // resign_threshold to -1 for games where resignation is diabled. This
    // enables us to report games where the eventual winner would have
    // incorrectly resigned early, had resignations been enabled.
    bool resign_enabled = true;

    // TODO(tommadams): rename batch_size to virtual_losses.
    int batch_size = 8;
    float komi = kDefaultKomi;
    std::string name = "minigo";

    // Seed used from random permutations.
    // If the default value of 0 is used, a time-based seed is chosen.
    uint64_t random_seed = 0;

    // Number of readouts to perform (ignored if seconds_per_move is non-zero).
    int num_readouts = 0;

    // If non-zero, the number of seconds to spend thinking about each move
    // instead of using a fixed number of readouts.
    float seconds_per_move = 0;

    // If non-zero, the maximum amount of time to spend thinking in a game:
    // we spend seconds_per_move thinking for each move for as many moves as
    // possible before exponentially decaying the amount of time.
    float time_limit = 0;

    // If time_limit is non-zero, the decay factor used to shorten the amount
    // of time spent thinking as the game progresses.
    float decay_factor = 0.98;

    // If true, print debug info to stderr.
    bool verbose = true;

    friend std::ostream& operator<<(std::ostream& ios, const Options& options);
  };

  struct History {
    std::array<float, kNumMoves> search_pi;
    Coord c = Coord::kPass;
    std::string comment;
    const MctsNode* node = nullptr;
  };

  // State that tracks which model is used for each inference.
  struct InferenceInfo {
    InferenceInfo(std::string model, int first_move)
        : model(std::move(model)),
          first_move(first_move),
          last_move(first_move) {}

    // Model name returned from RunMany.
    std::string model;

    // Total number of times a model was used for inference.
    size_t total_count = 0;

    // The first move a model was used for inference.
    int first_move = 0;

    // The last move a model was used for inference.
    // This needs to be tracked separately from first_move because the common
    // case is that the model changes change part-way through a tree search.
    int last_move = 0;
  };

  // If position is non-null, the player will be initilized with that board
  // state. Otherwise, the player is initialized with an empty board with black
  // to play.
  MctsPlayer(std::unique_ptr<DualNet> network, const Options& options);

  virtual ~MctsPlayer();

  void InitializeGame(const Position& position);

  void NewGame();

  virtual Coord SuggestMove();

  void PlayMove(Coord c);

  bool ShouldResign() const;

  void GetNodeFeatures(const MctsNode* node, DualNet::BoardFeatures* features);

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
  const std::string& name() const { return options_.name; }
  const std::vector<InferenceInfo>& inferences() const { return inferences_; }

  // These methods are protected to facilitate direct testing.
 protected:
  Options* mutable_options() { return &options_; }

  Coord PickMove();

  // Returns the list of nodes that TreeSearch performed inference on.
  // The contents of the returned Span is valid until the next call TreeSearch.
  virtual absl::Span<MctsNode* const> TreeSearch();

  // Returns the root of the game tree.
  MctsNode* game_root() { return &game_root_; }
  const MctsNode* game_root() const { return &game_root_; }

  Random* rnd() { return &rnd_; }

  std::string FormatScore(float score) const;

  DualNet* network() { return network_.get(); }

  // Run inference for the given leaf nodes & incorportate the inference output.
  void ProcessLeaves(absl::Span<MctsNode*> leaves);

 private:
  void PushHistory(Coord c);

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

  std::string model_;
  std::vector<InferenceInfo> inferences_;

  // Vectors reused when running TreeSearch.
  std::vector<MctsNode*> leaves_;
  std::vector<DualNet::BoardFeatures> features_;
  std::vector<DualNet::Output> outputs_;
  std::vector<symmetry::Symmetry> symmetries_used_;
  std::vector<const Position::Stones*> recent_positions_;
};

}  // namespace minigo

#endif  // CC_MCTS_PLAYER_H_
