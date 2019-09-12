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

#ifndef CC_MCTS_NODE_H_
#define CC_MCTS_NODE_H_

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "cc/constants.h"
#include "cc/inline_vector.h"
#include "cc/position.h"
#include "cc/symmetries.h"
#include "cc/zobrist.h"

namespace minigo {

class MctsNode {
 public:
  struct TreeStats {
    int num_nodes = 0;
    int num_leaf_nodes = 0;
    int max_depth = 0;
    int depth_sum = 0;

    std::string ToString() const;
  };

  struct EdgeStats {
    float N = 0;
    float W = 0;
    float P = 0;
    float original_P = 0;
  };

  // Information about a child. Returned by CalculateRankedChildInfo.
  struct ChildInfo {
    Coord c = Coord::kInvalid;
    float N;
    float P;
    float action_score;
  };

  static bool CmpN(const EdgeStats& a, const EdgeStats& b) { return a.N < b.N; }
  static bool CmpW(const EdgeStats& a, const EdgeStats& b) { return a.W < b.W; }
  static bool CmpP(const EdgeStats& a, const EdgeStats& b) { return a.P < b.P; }

  // Constructor for root node in the tree.
  MctsNode(EdgeStats* stats, const Position& position);

  // Constructor for child nodes.
  MctsNode(MctsNode* parent, Coord move);

  float N() const { return stats->N; }
  float W() const { return stats->W; }
  float P() const { return stats->P; }
  float original_P() const { return stats->original_P; }
  float Q() const { return W() / (1 + N()); }
  float Q_perspective() const {
    return position.to_play() == Color::kBlack ? Q() : -Q();
  }
  float U_scale() const {
    return 2.0 * (std::log((1.0f + N() + kUct_base) / kUct_base) + kUct_init);
  }

  float child_N(int i) const { return edges[i].N; }
  float child_W(int i) const { return edges[i].W; }
  float child_P(int i) const { return edges[i].P; }
  float child_original_P(int i) const { return edges[i].original_P; }
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
  bool at_move_limit() const { return position.n() >= kMaxSearchDepth; }

  enum class Flag : uint8_t {
    // Node is expanded.
    kExpanded = (1 << 0),

    // Node has a valid canonical symmetry.
    kHasCanonicalSymmetry = (1 << 1),
  };

  void SetFlag(Flag flag) { flags |= static_cast<uint8_t>(flag); }
  void ClearFlag(Flag flag) { flags &= ~static_cast<uint8_t>(flag); }
  bool HasFlag(Flag flag) const {
    return (flags & static_cast<uint8_t>(flag)) != 0;
  }

  // Finds the best move by visit count, N. Ties are broken using the child
  // action score.
  Coord GetMostVisitedMove() const;

  std::string Describe() const;
  std::string MostVisitedPathString() const;
  std::vector<Coord> MostVisitedPath() const;

  // Sorts the child nodes by visit counts, breaking ties by child action score.
  std::array<ChildInfo, kNumMoves> CalculateRankedChildInfo() const;

  // Mix noise into the node's priors:
  //   P_i = (1 - mix) * P_i + mix * noise_i
  void InjectNoise(const std::array<float, kNumMoves>& noise, float mix);

  // Selects the next leaf node for inference.
  // If inference is being batched and SelectLeaf chooses a node that has
  // already been added to the batch (IncorporateResults has not yet been
  // called), then SelectLeaf will return that same node.
  MctsNode* SelectLeaf();

  void IncorporateResults(float value_init_penalty,
                          absl::Span<const float> move_probabilities,
                          float value, MctsNode* up_to);

  void IncorporateEndGameResult(float value, MctsNode* up_to);

  void BackupValue(float value, MctsNode* up_to);

  void AddVirtualLoss(MctsNode* up_to);

  void RevertVirtualLoss(MctsNode* up_to);

  // Remove all children from the node except c.
  void PruneChildren(Coord c);

  // Clears all children and stats of this node.
  void ClearChildren();

  // Adjust the visit counts via whatever hairbrained scheme.
  void ReshapeFinalVisits();

  std::array<float, kNumMoves> CalculateChildActionScore() const;

  float CalculateSingleMoveChildActionScore(float to_play, float U_common,
                                            int i) const {
    float Q = child_Q(i);
    float U = U_common * child_P(i) / (1 + child_N(i));
    return Q * to_play + U - 1000.0f * !position.legal_move(i);
  }

  MctsNode* MaybeAddChild(Coord c);

  // Calculate and print statistics about the tree.
  TreeStats CalculateTreeStats() const;

  // Parent node.
  MctsNode* parent;

  // Stats for the edge from parent to this.
  EdgeStats* stats;

  // Move that led to this position.
  Coord move;

  uint8_t flags = 0;

  // If HasFlag(Flag::kHasCanonicalSymmetry) == true, canonical_symmetry holds
  // the symmetry that transforms the canonical form of the position to its real
  // one.
  // TODO(tommadams): for now, the canonical symmetry is just the one whose
  // Zobrist hash is the smallest, which is sufficient for use with the
  // inference cache. It would be more generally useful to use the real
  // canonical transform such that the first move is in the top-right corner,
  // etc.
  uint8_t canonical_symmetry = symmetry::kIdentity;

  std::array<EdgeStats, kNumMoves> edges;

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

}  // namespace minigo

#endif  // CC_MCTS_NODE_H_
