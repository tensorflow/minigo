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
#include "cc/position.h"
#include "cc/zobrist.h"

namespace minigo {

class MctsNode {
 public:
  struct EdgeStats {
    // TODO(tom): consider moving N into the MctsNode to save memory.
    float N = 0;
    float W = 0;
    float P = 0;
    float original_P = 0;
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

  float child_N(int i) const { return edges[i].N; }
  float child_W(int i) const { return edges[i].W; }
  float child_P(int i) const { return edges[i].P; }
  float child_original_P(int i) const { return edges[i].original_P; }
  float child_Q(int i) const { return child_W(i) / (1 + child_N(i)); }
  float child_U(int i) const {
    return kPuct * std::sqrt(std::max<float>(1, N() - 1)) * child_P(i) /
           (1 + child_N(i));
  }

  // Finds the best move by visit count, N. Ties are broken using the child
  // action score.
  Coord GetMostVisitedMove() const;

  std::string Describe() const;
  std::string MostVisitedPathString() const;
  std::vector<Coord> MostVisitedPath() const;

  // Returns up to the last num_moves of moves that lead up to this node,
  // including the node itself.
  // After GetMoveHistory returns, history[0] is this MctsNode and history[i] is
  // the MctsNode from i moves ago.
  void GetMoveHistory(int num_moves,
                      std::vector<const Position::Stones*>* history) const;

  void InjectNoise(const std::array<float, kNumMoves>& noise);

  // Selects the next leaf node for inference.
  // If inference is being batched and SelectLeaf chooses a node that has
  // already been added to the batch (IncorporateResults has not yet been
  // called), then SelectLeaf will return that same node.
  MctsNode* SelectLeaf();

  void IncorporateResults(absl::Span<const float> move_probabilities,
                          float value, MctsNode* up_to);

  void IncorporateEndGameResult(float value, MctsNode* up_to);

  void BackupValue(float value, MctsNode* up_to);

  void AddVirtualLoss(MctsNode* up_to);

  void RevertVirtualLoss(MctsNode* up_to);

  // Remove all children from the node except c.
  void PruneChildren(Coord c);

  // TODO(tommadams): Validate returning by value has the same performance as
  // passing a pointer to the output array.
  std::array<float, kNumMoves> CalculateChildActionScore() const;

  bool HasPositionBeenPlayedBefore(zobrist::Hash stone_hash) const;

  float CalculateSingleMoveChildActionScore(float to_play, float U_scale,
                                            int i) const {
    float Q = child_Q(i);
    float U = U_scale * child_P(i) / (1 + child_N(i));
    return Q * to_play + U - 1000.0f * !legal_moves[i];
  }

  MctsNode* MaybeAddChild(Coord c);

  // Parent node.
  MctsNode* parent;

  // Stats for the edge from parent to this.
  EdgeStats* stats;

  // Move that led to this position.
  Coord move;

  std::array<EdgeStats, kNumMoves> edges;

  std::array<bool, kNumMoves> legal_moves;

  // Map from move to resulting MctsNode.
  absl::flat_hash_map<uint64_t, std::unique_ptr<MctsNode>> children;

  bool is_expanded = false;

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
