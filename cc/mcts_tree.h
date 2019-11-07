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

#ifndef CC_MCTS_TREE_H_
#define CC_MCTS_TREE_H_

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "cc/constants.h"
#include "cc/inline_vector.h"
#include "cc/padded_array.h"
#include "cc/position.h"
#include "cc/random.h"
#include "cc/symmetries.h"
#include "cc/zobrist.h"

namespace minigo {

class MctsNode {
  friend class MctsTree;

 public:
  // MctsNode::CalculateChildActionScoreSse requires that the arrays in
  // EdgeStats are padded to a multiple of 16 bytes.
  struct EdgeStats {
    PaddedArray<int32_t, kNumMoves> N{};
    PaddedArray<float, kNumMoves> W{};
    PaddedArray<float, kNumMoves> P{};
    PaddedArray<float, kNumMoves> original_P{};
  };

  // Constructor for root node in the tree.
  MctsNode(EdgeStats* stats, const Position& position);

  // Constructor for child nodes.
  MctsNode(MctsNode* parent, Coord move);

  int N() const { return stats->N[stats_idx]; }
  float W() const { return stats->W[stats_idx]; }
  float P() const { return stats->P[stats_idx]; }
  float original_P() const { return stats->original_P[stats_idx]; }
  float Q() const { return W() / (1 + N()); }
  float Q_perspective() const {
    return position.to_play() == Color::kBlack ? Q() : -Q();
  }
  float U_scale() const {
    return 2.0 * (std::log((1.0f + N() + kUct_base) / kUct_base) + kUct_init);
  }

  int child_N(int i) const { return edges.N[i]; }
  float child_W(int i) const { return edges.W[i]; }
  float child_P(int i) const { return edges.P[i]; }
  float child_original_P(int i) const { return edges.original_P[i]; }
  float child_Q(int i) const { return child_W(i) / (1 + child_N(i)); }
  float child_U(int i) const {
    return U_scale() * std::sqrt(std::max<float>(1, N() - 1)) * child_P(i) /
           (1 + child_N(i));
  }

  bool game_over() const {
    return (move == Coord::kResign) ||
           (move == Coord::kPass && parent != nullptr &&
            parent->move == Coord::kPass);
  }

  // Finds the best move by visit count, N. Ties are broken using the child
  // action score.
  Coord GetMostVisitedMove(bool restrict_pass_alive = false) const;
  std::vector<Coord> GetMostVisitedPath() const;
  std::string GetMostVisitedPathString() const;

  // Remove all children from the node except c.
  void PruneChildren(Coord c);

  // Clears all children and stats of this node.
  void ClearChildren();

  std::array<float, kNumMoves> CalculateChildActionScore() const;

  void CalculateChildActionScoreSse(PaddedSpan<float> result) const;

  float CalculateSingleMoveChildActionScore(float to_play, float U_common,
                                            int i) const {
    float Q = child_Q(i);
    float U = U_common * child_P(i) / (1 + child_N(i));
    return Q * to_play + U - 1000.0f * !position.legal_move(i);
  }

  MctsNode* MaybeAddChild(Coord c);

  // Parent node.
  MctsNode* parent;

  // Stats for the edge from parent to this.
  EdgeStats* stats;

  // Index into `stats` for this node's stats.
  // This is the same as `move` for all nodes except the game root node; the
  // game root's `stats_idx` is initiliazed to 0 because its `move` is
  // `Coord::kInvalid`.
  Coord stats_idx;

  // Move that led to this position.
  Coord move;

  uint8_t is_expanded : 1;
  uint8_t has_canonical_symmetry : 1;

  // If HasFlag(Flag::kHasCanonicalSymmetry) == true, canonical_symmetry holds
  // the symmetry that transforms the canonical form of the position to its real
  // one.
  // TODO(tommadams): for now, the canonical symmetry is just the one whose
  // Zobrist hash is the smallest, which is sufficient for use with the
  // inference cache. It would be more generally useful to use the real
  // canonical transform such that the first move is in the top-right corner,
  // etc.
  symmetry::Symmetry canonical_symmetry = symmetry::kIdentity;

  EdgeStats edges;

  // Map from move to resulting MctsNode.
  absl::flat_hash_map<Coord, std::unique_ptr<MctsNode>> children;

  // Current board position.
  Position position;

  // Number of virtual losses on this node.
  int num_virtual_losses_applied = 0;

  // Each position contains a Zobrist hash of its stones, which can be used for
  // superko detection. In order to accelerate superko detection, caches of all
  // ancestor positions are added at regular depths in the search tree. This
  // reduces superko detection time complexity from O(N) to O(1).
  //
  // If non-null, superko_cache contains the Zobrist hash of all positions
  // played to this position, including position.stone_hash().
  // If null, clients should determine whether a position has appeared before
  // during the game by walking up the tree (via the parent pointer), checking
  // the position.stone_hash() of each node visited, until a node is found that
  // contains a non-null superko_cache.
  using SuperkoCache = absl::flat_hash_set<zobrist::Hash>;
  std::unique_ptr<SuperkoCache> superko_cache;
};

class MctsTree {
 public:
  struct Stats {
    int num_nodes = 0;
    int num_leaf_nodes = 0;
    int max_depth = 0;
    int depth_sum = 0;

    std::string ToString() const;
  };

  // Information about a child. Returned by CalculateRankedChildInfo.
  // TODO(tommadams): rename to MoveInfo.
  struct ChildInfo {
    Coord c = Coord::kInvalid;
    float N;
    float P;
    float action_score;
  };

  struct Options {
    // See mcts_node.cc for details.
    // Default (0.0) is init-to-parent.
    float value_init_penalty = 0.0;

    // For soft-picked moves, the probabilities are exponentiated by
    // policy_softmax_temp to encourage diversity in early play.
    float policy_softmax_temp = 0.98;

    bool soft_pick_enabled = true;

    // When to do deterministic move selection: after 30 moves on a 19x19, 6 on
    // 9x9. The divide 2, multiply 2 guarentees that white and black do the same
    // number of softpicks.
    int soft_pick_cutoff = ((kN * kN / 12) / 2) * 2;

    friend std::ostream& operator<<(std::ostream& ios, const Options& options);
  };

  MctsTree(const Position& position, const Options& options);

  const MctsNode* root() const { return root_; }

  Color to_play() const { return root_->position.to_play(); }
  bool is_game_over() const { return root_->game_over(); }
  bool is_legal_move(Coord c) const { return root_->position.legal_move(c); }

  // Selects the next leaf node for inference.
  // If inference is being batched and SelectLeaf chooses a node that has
  // already been added to the batch (IncorporateResults has not yet been
  // called), then SelectLeaf will return that same node.
  MctsNode* SelectLeaf(bool allow_pass);

  // Performs a soft-pick using `rnd` if the number of moves played is
  // < `soft_pick_cutoff`. Picks the most visited legal move otherwise.
  // If `restrict_pass_alive` is true, playing in pass-alive territory is
  // disallowed.
  Coord PickMove(Random* rnd, bool restrict_pass_alive) const;

  void PlayMove(Coord c);

  void AddVirtualLoss(MctsNode* leaf);

  void RevertVirtualLoss(MctsNode* leaf);

  void IncorporateResults(MctsNode* leaf,
                          absl::Span<const float> move_probabilities,
                          float value);

  void IncorporateEndGameResult(MctsNode* leaf, float value);

  // Exposed for testing.
  void BackupValue(MctsNode* leaf, float value);

  // Mix noise into the node's priors:
  //   P_i = (1 - mix) * P_i + mix * noise_i
  void InjectNoise(const std::array<float, kNumMoves>& noise, float mix);

  // Adjust the visit counts via whatever hairbrained scheme.
  // `restrict_pass_alive` should be set to the same value as was passed to
  // `PickMove`.
  void ReshapeFinalVisits(bool restrict_pass_alive);

  // Converts child visit counts to a probability distribution, pi.
  std::array<float, kNumMoves> CalculateSearchPi() const;

  // Calculate and print statistics about the tree.
  Stats CalculateStats() const;

  std::string Describe() const;

  // Sorts the child nodes by visit counts, breaking ties by child action score.
  // TODO(tommadams): rename to CalculateRankedMoveInfo.
  std::array<ChildInfo, kNumMoves> CalculateRankedChildInfo() const;

  // TODO(tommadams): remove this UndoMove. PlayMove is a destructive operation
  // and UndoMove will leave the tree in a bad state.
  bool UndoMove();

  void ClearSubtrees() { root_->ClearChildren(); }

  float CalculateScore(float komi) const {
    return root_->position.CalculateScore(komi);
  }

 private:
  Coord PickMostVisitedMove(bool restrict_pass_alive) const;
  Coord SoftPickMove(Random* rnd) const;

  MctsNode* root_;
  MctsNode game_root_;
  MctsNode::EdgeStats game_root_stats_;
  Options options_;
};

}  // namespace minigo

#endif  // CC_MCTS_NODE_H_
