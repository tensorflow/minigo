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
#include "cc/game.h"
#include "cc/mcts_node.h"
#include "cc/model/inference_cache.h"
#include "cc/model/model.h"
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

    // See mcts_node.cc for details.
    // Default (0.0) is init-to-parent.
    float value_init_penalty = 0.0;

    // For soft-picked moves, the probabilities are exponentiated by
    // policy_softmax_temp to encourage diversity in early play.
    float policy_softmax_temp = 0.98;

    int virtual_losses = 8;

    // Random seed & stream used for random permutations.
    uint64_t random_seed = Random::kUniqueSeed;

    // If true, flip & rotate the board features when performing inference. The
    // symmetry chosen is psuedo-randomly chosen in a deterministic way based
    // on the position itself and the random_seed.
    bool random_symmetry = true;

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

    // "Playout Cap Oscillation" as per the KataGo paper.
    // If fastplay_frequency > 0, tree search is modified as follows:
    //   - Each move is either a "low-readout" fast move, or a full, slow move.
    //   The percent of fast moves corresponds to "fastplay_frequency"
    //   - A "fast" move will:
    //     - Reuse the tree
    //     - Not mix noise in at root
    //     - Only perform 'fastplay_readouts' readouts.
    //     - Not be used as a training target.
    //   - A "slow" move will:
    //     - Clear the tree (*not* the cache).
    //     - Mix in dirichlet noise
    //     - Perform 'num_readouts' readouts.
    //     - Be noted in the Game object, to be written as a training example.
    float fastplay_frequency = 0;
    int fastplay_readouts = 20;

    // "Target pruning" adjusts the targets after reading to discard reads
    // caused by 'unhelpful' noise & reflect the 'better' understanding of the
    // reward distribution.  "False" == no pruning will be applied.
    bool target_pruning = false;

    // If true, this will prevent play in benson's pass-alive regions after 5
    // passes have been played (by anyone).  It will also zero out any visits
    // the pass-alive points may have gotten.
    bool restrict_in_bensons = false;
    friend std::ostream& operator<<(std::ostream& ios, const Options& options);
  };

  // Callback invoked on each batch of leaves expanded during tree search.
  using TreeSearchCallback =
      std::function<void(const std::vector<const MctsNode*>&)>;

  // If position is non-null, the player will be initilized with that board
  // state. Otherwise, the player is initialized with an empty board with black
  // to play.
  MctsPlayer(std::unique_ptr<Model> model,
             std::shared_ptr<InferenceCache> inference_cache, Game* game,
             const Options& options);

  ~MctsPlayer();

  void InitializeGame(const Position& position);

  void NewGame();

  Coord SuggestMove(int new_readouts, bool inject_noise = false,
                    bool restrict_in_bensons = false);
  // Plays the move at point c.
  // If game is non-null, adds a new move to the game's move history and sets
  // the game over state if appropriate.
  bool PlayMove(Coord c, bool is_trainable = false);

  // Moves the root_ node up to its parent, popping the last move off the game
  // history but preserving the game tree.
  bool UndoMove();

  bool ShouldResign() const;

  void SetTreeSearchCallback(TreeSearchCallback cb);

  void ClearChildren() { root_->ClearChildren(); }

  // Returns a string containing the list of all models used for inference, and
  // which moves they were used for.
  std::string GetModelsUsedForInference() const;

  // Returns the root of the current search tree, i.e. the current board state.
  // TODO(tommadams): Remove mutable access to the root once MiniguiGtpPlayer
  // no longer calls SelectLeaves directly.
  MctsNode* root() { return root_; }
  const MctsNode* root() const { return root_; }

  const Options& options() const { return options_; }
  const std::string& name() const { return model_->name(); }
  Model* model() { return model_.get(); }
  uint64_t seed() const { return rnd_.seed(); }

  void SetOptions(const Options& options) { options_ = options; }

  void TreeSearch(int num_leaves);

  // Protected methods that get exposed for testing.
 protected:
  Coord PickMove(bool restrict_in_bensons = false);

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

  // A position's canonical symmetry is the symmetry that transforms the
  // canonical form of a position into its actual form. For example, one way of
  // defining a canonical symmetry is that the first move must be played in the
  // top-right corner. For the early moves of a game, there will not be a
  // canonical symmetry defined; in these cases, GetCanonicalSymmetry returns
  // symmetry::kIdentity.
  symmetry::Symmetry GetCanonicalSymmetry(const MctsNode* node) const {
    return node->canonical_symmetry;
  }

  // Returns the symmetry that should be applied to this node's position when
  // performing inference. The MctsPlayer picks a symmetry using a pseudo-random
  // but deterministic function so that the same MctsPlayer instance is
  // guaranteed to return the same symmetry for a given position but different
  // MctsPlayer instances may return different symmetries for the same position.
  symmetry::Symmetry GetInferenceSymmetry(const MctsNode* node) const {
    if (options_.random_symmetry) {
      uint64_t bits = Random::MixBits(
          node->position.stone_hash() * Random::kLargePrime + inference_mix_);
      return static_cast<symmetry::Symmetry>(bits % symmetry::kNumSymmetries);
    } else {
      return symmetry::kIdentity;
    }
  }

  // Inject noise into the root node.
  void InjectNoise(float dirichlet_alpha);

  // Expand the root node if necessary.
  // In order to correctly count the number of reads performed or to inject
  // noise, the root node must be expanded. The root will always be expanded
  // unless this is the first time SuggestMove has been called for a game, or
  // PlayMove was called without a prior call to SuggestMove, or the child nodes
  // of the tree have been cleared.
  void MaybeExpandRoot();

  // Select up to `num_inferences` leaves to perform inference on, storing the
  // selected leaves in `tree_search_inferences_`. If the player has an
  // inference cache, this can cause more nodes to be added to the tree when
  // the selected leaves are already in the cache. To limit this, SelectLeaves
  // will add no more than `max_num_leaves` leaves to the tree.
  //
  // In some positions, the model may favor one move so heavily that it
  // overcomes the effects of virtual loss. In this case, SelectLeaves may
  // choose the same leaf multiple times.
  void SelectLeaves(int num_inferences, int max_num_leaves);

  // Run inference on the contents of `inferences_` that was previously
  // populated by a call to SelectLeaves, and propagate the results back up the
  // tree to the root.
  void ProcessLeaves();

  void UpdateGame(Coord c);

  std::unique_ptr<Model> model_;
  int temperature_cutoff_;

  MctsNode::EdgeStats root_stats_;

  MctsNode* root_;
  MctsNode game_root_;

  Game* game_;

  Random rnd_;

  Options options_;

  // The name of the model used for inferences. In the case of ReloadingModel,
  // this is different from the model's name: the model name is the pattern used
  // to match each generation of model, while the inference model name is the
  // path to the actual serialized model file.
  std::string inference_model_;

  std::vector<InferenceInfo> inferences_;

  std::shared_ptr<InferenceCache> inference_cache_;

  struct TreeSearchInference {
    TreeSearchInference(InferenceCache::Key cache_key,
                        symmetry::Symmetry canonical_sym,
                        symmetry::Symmetry inference_sym, MctsNode* leaf)
        : cache_key(cache_key),
          canonical_sym(canonical_sym),
          inference_sym(inference_sym),
          leaf(leaf) {}
    InferenceCache::Key cache_key;
    symmetry::Symmetry canonical_sym;
    symmetry::Symmetry inference_sym;
    MctsNode* leaf;
    ModelInput input;
    ModelOutput output;
  };

  std::vector<TreeSearchInference> tree_search_inferences_;
  std::vector<const ModelInput*> input_ptrs_;
  std::vector<ModelOutput*> output_ptrs_;

  TreeSearchCallback tree_search_cb_ = nullptr;

  // Random number combined with each Position's Zobrist hash in order to
  // deterministically choose the symmetry to apply when performing inference.
  const int64_t inference_mix_;
};

}  // namespace minigo

#endif  // CC_MCTS_PLAYER_H_
