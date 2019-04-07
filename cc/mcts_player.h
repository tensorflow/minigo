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
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/time/time.h"
#include "cc/algorithm.h"
#include "cc/constants.h"
#include "cc/dual_net/dual_net.h"
#include "cc/dual_net/inference_cache.h"
#include "cc/game.h"
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
    // If inject_noise is true, the amount of noise to mix into the root.
    float noise_mix = 0.25;
    bool inject_noise = true;
    bool soft_pick = true;
    bool random_symmetry = true;

    // See mcts_node.cc for details.
    // Default (0.0) is init-to-parent.
    float value_init_penalty = 0.0;

    // For soft-picked moves, the probabilities are exponentiated by
    // policy_softmax_temp to encourage diversity in early play.
    float policy_softmax_temp = 0.98;

    int virtual_losses = 8;

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

    // If true, children of the current root node are pruned when a move is
    // played. Under normal play, only the descendents of the move played ever
    // have a chance of being visited again during tree search. However, when
    // using Minigo to explore different variations and ponder about the best
    // moves, it makes sense to keep the full tree around.
    bool prune_orphaned_nodes = true;

    // If true, the subtree of a played move that was expanded during tree
    // search will be kept.
    // If false, all children of the current root will be deleted before each
    // move is played.
    bool tree_reuse = true;

    friend std::ostream& operator<<(std::ostream& ios, const Options& options);
  };

  struct History {
    std::array<float, kNumMoves> search_pi;
    Coord c = Coord::kPass;
    std::string comment;
    const MctsNode* node = nullptr;
  };

  // Callback invoked on each batch of leaves expanded during tree search.
  using TreeSearchCallback = std::function<void(const std::vector<MctsNode*>&)>;

  // If position is non-null, the player will be initilized with that board
  // state. Otherwise, the player is initialized with an empty board with black
  // to play.
  MctsPlayer(std::unique_ptr<DualNet> network,
             std::shared_ptr<InferenceCache> inference_cache, Game* game,
             const Options& options);

  virtual ~MctsPlayer();

  void InitializeGame(const Position& position);

  virtual void NewGame();

  virtual Coord SuggestMove();

  // Plays the move at point c.
  // If game is non-null, adds a new move to the game's move history and sets
  // the game over state if appropriate.
  virtual bool PlayMove(Coord c);

  // Moves the root_ node up to its parent, popping the last move off the game
  // history but preserving the game tree.
  virtual bool UndoMove();

  bool ShouldResign() const;

  void SetTreeSearchCallback(TreeSearchCallback cb);

  // Returns a string containing the list of all models used for inference, and
  // which moves they were used for.
  std::string GetModelsUsedForInference() const;

  // Returns the root of the current search tree, i.e. the current board state.
  // TODO(tommadams): Remove mutable access to the root once MiniguiGtpPlayer
  // no longer calls SelectLeaves directly.
  MctsNode* root() { return root_; }
  const MctsNode* root() const { return root_; }

  Game* game() { return game_; }

  const Options& options() const { return options_; }
  const std::string& name() const { return network_->name(); }
  DualNet* network() { return network_.get(); }
  uint64_t seed() const { return rnd_.seed(); }

  void SetOptions(const Options& options) { options_ = options; }

  void TreeSearch();

  // TODO(tommadams): Make SelectLeaves and ProcessLeaves private, or remove
  // them entirely.
  void SelectLeaves(MctsNode* root, int num_leaves,
                    std::vector<MctsNode*>* leaves);

  // Run inference for the given leaf nodes & incorportate the inference output.
  void ProcessLeaves(const std::vector<MctsNode*>& leaves,
                     bool random_symmetry);

  // Protected methods that get exposed for testing.
 protected:
  Coord PickMove();

 private:
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

  void UpdateGame(Coord c);

  std::unique_ptr<DualNet> network_;
  int temperature_cutoff_;

  MctsNode::EdgeStats root_stats_;

  MctsNode* root_;
  MctsNode game_root_;

  BoardVisitor bv_;
  GroupVisitor gv_;

  Game* game_;

  Random rnd_;

  Options options_;

  // The name of the model used for inferences. In the case of ReloadingDualNet,
  // this is different from the model's name: the model name is the pattern used
  // to match each generation of model, while the inference model name is the
  // path to the actual serialized model file.
  std::string inference_model_;

  std::vector<InferenceInfo> inferences_;

  std::shared_ptr<InferenceCache> inference_cache_;

  // Vectors reused when running TreeSearch.
  std::vector<MctsNode*> tree_search_leaves_;
  std::vector<DualNet::BoardFeatures> features_;
  std::vector<DualNet::Output> outputs_;
  std::vector<symmetry::Symmetry> symmetries_used_;
  std::vector<const Position::Stones*> recent_positions_;

  TreeSearchCallback tree_search_cb_ = nullptr;
};

}  // namespace minigo

#endif  // CC_MCTS_PLAYER_H_
