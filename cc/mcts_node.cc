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

#include "cc/mcts_node.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <tuple>
#include <utility>

#include "absl/strings/str_format.h"
#include "cc/algorithm.h"
#include "cc/logging.h"

namespace minigo {

namespace {

// Superko implementation that uses MctsNode::superko_cache.
class ZobristHistory : public Position::ZobristHistory {
 public:
  explicit ZobristHistory(const MctsNode* node) : node_(node) {}

  bool HasPositionBeenPlayedBefore(zobrist::Hash stone_hash) const {
    for (const auto* node = node_; node != nullptr; node = node->parent) {
      if (node->superko_cache != nullptr) {
        return node->superko_cache->contains(stone_hash);
      } else {
        if (node->position.stone_hash() == stone_hash) {
          return true;
        }
      }
    }
    return false;
  }

 private:
  const MctsNode* node_;
};

constexpr int kSuperKoCacheStride = 8;

}  // namespace

MctsNode::MctsNode(EdgeStats* stats, const Position& position)
    : parent(nullptr),
      stats(stats),
      move(Coord::kInvalid),
      position(position) {}

MctsNode::MctsNode(MctsNode* parent, Coord move)
    : parent(parent),
      stats(&parent->edges[move]),
      move(move),
      position(parent->position) {
  // TODO(tommadams): move this code into the MctsPlayer and only perform it
  // only if we are using an inference cache.
  if (parent->HasFlag(Flag::kHasCanonicalSymmetry)) {
    SetFlag(Flag::kHasCanonicalSymmetry);
    canonical_symmetry = parent->canonical_symmetry;
  } else {
    // TODO(tommadams): skip this check if `move` is kPass or on the diagonal.
    static_assert(symmetry::kIdentity == 0, "kIdentity must be 0");

    // When choosing a canonical symmetry, we consider the "best" symmetry to
    // be the one with the smallest Zobrist hash. The "best" symmetry is only
    // canonical if its hash value is also unique among the hashes from the
    // other possible symmetries.
    auto best_symmetry = symmetry::kIdentity;
    auto best_hash = position.stone_hash();
    bool found_unique_hash = true;
    std::array<Stone, kN * kN> transformed;
    for (int i = 1; i < symmetry::kNumSymmetries; ++i) {
      auto sym = static_cast<symmetry::Symmetry>(i);
      symmetry::ApplySymmetry<kN, 1>(sym, position.stones().data(),
                                     transformed.data());
      auto stone_hash = Position::CalculateStoneHash(transformed);
      if (stone_hash < best_hash) {
        best_symmetry = sym;
        best_hash = stone_hash;
      } else if (stone_hash == best_hash) {
        found_unique_hash = false;
        break;
      }
    }

    if (found_unique_hash) {
      SetFlag(Flag::kHasCanonicalSymmetry);
      canonical_symmetry = symmetry::Inverse(best_symmetry);
    }
  }

  MG_DCHECK(move >= 0);
  MG_DCHECK(move < kNumMoves);

  ZobristHistory zobrist_history(this);
  position.PlayMove(move, position.to_play(), &zobrist_history);

  // Insert a cache of ancestor Zobrist hashes at regular depths in the tree.
  // See the comment for superko_cache in the mcts_node.h for more details.
  if ((position.n() % kSuperKoCacheStride) == 0) {
    superko_cache = absl::make_unique<SuperkoCache>();
    superko_cache->reserve(position.n() + 1);
    superko_cache->insert(position.stone_hash());
    for (auto* node = parent; node != nullptr; node = node->parent) {
      if (node->superko_cache != nullptr) {
        superko_cache->insert(node->superko_cache->begin(),
                              node->superko_cache->end());
        break;
      }
      superko_cache->insert(node->position.stone_hash());
    }
  }
}

Coord MctsNode::GetMostVisitedMove() const {
  // Find the set of moves with the largest N.
  inline_vector<Coord, kNumMoves> moves;
  int best_N = -1;
  for (int i = 0; i < kNumMoves; ++i) {
    int cn = child_N(i);
    if (cn >= best_N) {
      if (cn > best_N) {
        moves.clear();
        best_N = cn;
      }
      moves.push_back(i);
    }
  }

  MG_CHECK(!moves.empty());

  // If there's only one move with the largest N, we're done.
  if (moves.size() == 1) {
    return moves[0];
  }

  // Otherwise, break score using the child action score.
  float to_play = position.to_play() == Color::kBlack ? 1 : -1;
  float U_common = U_scale() * std::sqrt(1.0f + N());

  Coord c = moves[0];
  float best_cas =
      CalculateSingleMoveChildActionScore(to_play, U_common, moves[0]);
  for (int i = 0; i < moves.size(); ++i) {
    float cas =
        CalculateSingleMoveChildActionScore(to_play, U_common, moves[i]);
    if (cas > best_cas) {
      best_cas = cas;
      c = moves[i];
    }
  }

  return c;
}

void MctsNode::ReshapeFinalVisits() {
  Coord best = GetMostVisitedMove();
  float U_common = U_scale() * std::sqrt(1.0f + N());
  float to_play = position.to_play() == Color::kBlack ? 1 : -1;
  float best_cas =
      CalculateSingleMoveChildActionScore(to_play, U_common, uint16_t(best));

  // int total = 0;
  // We explored this child with uncertainty about its value.  Now, after
  // searching, we change the visit count to reflect how many visits we would
  // have given it with our newer understanding of its regret relative to our
  // best move.
  for (int i = 0; i < kNumMoves; ++i) {
    if (i == uint16_t(best)) {
      continue;
    }

    // Change N_child to the smallest value that satisfies the inequality
    // best_cas > Q + (U_scale * P * sqrt(N_parent) / N_child)
    // Solving for N_child, we get:
    int new_N = std::max(
        0, std::min(
               static_cast<int>(child_N(i)),
               static_cast<int>(-1 * (U_scale() * child_P(i) * std::sqrt(N())) /
                                ((child_Q(i) * to_play) - best_cas)) -
                   1));
    // total += edges[i].N - new_N;
    edges[i].N = new_N;
  }
  // MG_LOG(INFO) << "Pruned " << total << " visits.";
}

std::array<MctsNode::ChildInfo, kNumMoves> MctsNode::CalculateRankedChildInfo()
    const {
  auto child_action_score = CalculateChildActionScore();
  std::array<ChildInfo, kNumMoves> child_info;
  for (int i = 0; i < kNumMoves; ++i) {
    child_info[i].c = i;
    child_info[i].N = child_N(i);
    child_info[i].P = child_P(i);
    child_info[i].action_score = child_action_score[i];
  }
  std::sort(child_info.begin(), child_info.end(),
            [](const ChildInfo& a, const ChildInfo& b) {
              if (a.N != b.N) {
                return a.N > b.N;
              }
              if (a.P != b.P) {
                return a.P > b.P;
              }
              return a.action_score > b.action_score;
            });
  return child_info;
}

std::string MctsNode::Describe() const {
  auto sorted_child_info = CalculateRankedChildInfo();

  auto result = absl::StrFormat(
      "%0.4f\n%s\n"
      "move : action    Q     U     P   P-Dir    N  soft-N  p-delta  p-rel",
      Q(), MostVisitedPathString());

  float child_N_sum = 0;
  for (const auto& e : edges) {
    child_N_sum += e.N;
  }
  for (int rank = 0; rank < 15; ++rank) {
    Coord c = sorted_child_info[rank].c;
    float soft_N = child_N(c) / child_N_sum;
    float p_delta = soft_N - child_P(c);
    float p_rel = p_delta / child_P(c);
    absl::StrAppendFormat(
        &result,
        "\n%-5s: % 4.3f % 4.3f %0.3f %0.3f %0.3f %5d %0.4f % 6.5f % 3.2f",
        c.ToGtp(), sorted_child_info[rank].action_score, child_Q(c), child_U(c),
        child_P(c), child_original_P(c), static_cast<int>(child_N(c)), soft_N,
        p_delta, p_rel);
  }
  return result;
}

std::vector<Coord> MctsNode::MostVisitedPath() const {
  std::vector<Coord> path;
  const auto* node = this;
  while (!node->children.empty()) {
    Coord c = node->GetMostVisitedMove();

    if (node->child_N(c) == 0) {
      // In cases where nodes have been added to the tree manually (after the
      // user has played a move, loading an SGF game), it's possible that no
      // children have been visited. Break before adding a spurious node to the
      // path.
      break;
    }

    path.push_back(c);

    auto it = node->children.find(c);
    if (it == node->children.end()) {
      // When we reach the move limit, last node will have children with visit
      // counts but no children.
      break;
    }

    node = it->second.get();
  }
  return path;
}

std::string MctsNode::MostVisitedPathString() const {
  std::string result;
  const auto* node = this;
  for (Coord c : MostVisitedPath()) {
    auto it = node->children.find(c);
    MG_CHECK(it != node->children.end());
    node = it->second.get();
    absl::StrAppendFormat(&result, "%s (%d) ==> ", node->move.ToGtp(),
                          static_cast<int>(node->N()));
  }
  absl::StrAppendFormat(&result, "Q: %0.5f", node->Q());
  return result;
}

void MctsNode::InjectNoise(const std::array<float, kNumMoves>& noise,
                           float mix) {
  // NOTE: our interpretation is to only add dirichlet noise to legal moves.
  // Because dirichlet entries are independent we can simply zero and rescale.

  float scalar = 0;
  for (int i = 0; i < kNumMoves; ++i) {
    if (position.legal_move(i)) {
      scalar += noise[i];
    }
  }

  if (scalar > std::numeric_limits<float>::min()) {
    scalar = 1.0 / scalar;
  }

  for (int i = 0; i < kNumMoves; ++i) {
    float scaled_noise = scalar * (position.legal_move(i) ? noise[i] : 0);
    edges[i].P = (1 - mix) * edges[i].P + mix * scaled_noise;
  }
}

MctsNode* MctsNode::SelectLeaf() {
  auto* node = this;
  for (;;) {
    // If a node has never been evaluated, we have no basis to select a child.
    if (!node->HasFlag(Flag::kExpanded)) {
      return node;
    }
    // HACK: if last move was a pass, always investigate double-pass first
    // to avoid situations where we auto-lose by passing too early.
    if (node->move == Coord::kPass && node->child_N(Coord::kPass) == 0) {
      node = node->MaybeAddChild(Coord::kPass);
      continue;
    }

    auto child_action_score = node->CalculateChildActionScore();
    Coord best_move = ArgMax(child_action_score);
    node = node->MaybeAddChild(best_move);
  }
}

void MctsNode::IncorporateResults(float value_init_penalty,
                                  absl::Span<const float> move_probabilities,
                                  float value, MctsNode* up_to) {
  MG_DCHECK(move_probabilities.size() == kNumMoves);
  // A finished game should not be going through this code path, it should
  // directly call BackupValue on the result of the game.
  MG_DCHECK(!game_over());

  // If the node has already been selected for the next inference batch, we
  // shouldn't 'expand' it again.
  if (HasFlag(Flag::kExpanded)) {
    return;
  }

  float policy_scalar = 0;
  for (int i = 0; i < kNumMoves; ++i) {
    if (position.legal_move(i)) {
      policy_scalar += move_probabilities[i];
    }
  }
  if (policy_scalar > std::numeric_limits<float>::min()) {
    policy_scalar = 1 / policy_scalar;
  }

  // NOTE: Minigo uses value [-1, 1] from black's perspective
  //       Leela uses value [0, 1] from current player's perspective
  //       AlphaGo uses [0, 1] in tree search (see matthew lai's post)
  //
  // The initial value of a child's Q is not perfectly understood.
  // There are a couple of general ideas:
  //   * Init to Parent:
  //      Init a new child to its parent value.
  //      We think of this as saying "The game is probably the same after
  //      *any* move".
  //   * Init to Draw AKA init to zero AKA "position looks even":
  //      Init a new child to 0 for {-1, 1} or 0.5 for LZ.
  //      We tested this in v11, because this is how we interpretted the
  //      original AGZ paper. This doesn't make a lot of sense: The losing
  //      player tends to explore every move before reading a second one
  //      twice.  The winning player tends to read only the top policy move
  //      because it has much higher value than any other move.
  //   * Init to Parent minus a constant AKA FPU (Leela's approach):
  //      This outperformed init to parent in eval matches when LZ tested it.
  //      Leela-Zero uses a value around 0.15-0.25 based on policy of explored
  //      children. LCZero uses a much large value 1.25 (they use {-1 to 1}).
  //   * Init to Loss:
  //      Init all children to losing.
  //      We think of this as saying "Only a small number of moves work don't
  //      get distracted"
  float reduction =
      value_init_penalty * (position.to_play() == Color::kBlack ? 1 : -1);
  float reduced_value = std::min(1.0f, std::max(-1.0f, value - reduction));

  SetFlag(Flag::kExpanded);
  for (int i = 0; i < kNumMoves; ++i) {
    // Zero out illegal moves, and re-normalize move_probabilities.
    float move_prob =
        position.legal_move(i) ? policy_scalar * move_probabilities[i] : 0;

    edges[i].original_P = edges[i].P = move_prob;

    // Note that we accumulate W here, rather than assigning.
    // When performing tree search normally, we could just assign the value to W
    // because the result of value head is known before we expand the node.
    // When running Minigui in study move however, we load the entire game tree
    // before starting background inference. This means that while background
    // inferences are being performed, nodes in the tree may already be expanded
    // and have non-zero W values at the time we need to incorporate a result
    // for the node from the value head.
    edges[i].W += reduced_value;
  }
  BackupValue(value, up_to);
}

void MctsNode::IncorporateEndGameResult(float value, MctsNode* up_to) {
  MG_DCHECK(game_over() || at_move_limit());
  MG_DCHECK(!HasFlag(Flag::kExpanded));
  BackupValue(value, up_to);
}

void MctsNode::BackupValue(float value, MctsNode* up_to) {
  auto* node = this;
  for (;;) {
    node->stats->W += value;
    ++node->stats->N;
    if (node == up_to) {
      return;
    }
    node = node->parent;
  }
}

void MctsNode::AddVirtualLoss(MctsNode* up_to) {
  auto* node = this;
  for (;;) {
    ++node->num_virtual_losses_applied;
    node->stats->W += node->position.to_play() == Color::kBlack ? 1 : -1;
    if (node == up_to) {
      return;
    }
    node = node->parent;
  }
}

void MctsNode::RevertVirtualLoss(MctsNode* up_to) {
  auto* node = this;
  for (;;) {
    --node->num_virtual_losses_applied;
    node->stats->W -= node->position.to_play() == Color::kBlack ? 1 : -1;
    if (node == up_to) {
      return;
    }
    node = node->parent;
  }
}

void MctsNode::PruneChildren(Coord c) {
  auto child = std::move(children[c]);
  children.clear();
  children[c] = std::move(child);
}

std::array<float, kNumMoves> MctsNode::CalculateChildActionScore() const {
  float to_play = position.to_play() == Color::kBlack ? 1 : -1;
  float U_common = U_scale() * std::sqrt(std::max<float>(1, N() - 1));

  std::array<float, kNumMoves> result;
  for (int i = 0; i < kNumMoves; ++i) {
    result[i] = CalculateSingleMoveChildActionScore(to_play, U_common, i);
  }
  return result;
}

MctsNode* MctsNode::MaybeAddChild(Coord c) {
  auto it = children.find(c);
  if (it == children.end()) {
    auto child = absl::make_unique<MctsNode>(this, c);
    MctsNode* result = child.get();
    children[c] = std::move(child);
    return result;
  } else {
    return it->second.get();
  }
}

MctsNode::TreeStats MctsNode::CalculateTreeStats() const {
  TreeStats stats;

  std::function<void(const MctsNode&, int)> traverse = [&](const MctsNode& node,
                                                           int depth) {
    stats.num_nodes += 1;
    stats.num_leaf_nodes += node.N() <= 1;
    stats.max_depth = std::max(depth, stats.max_depth);
    stats.depth_sum += depth;

    for (const auto& child : node.children) {
      traverse(*child.second.get(), depth + 1);
    }
  };

  traverse(*this, 0);

  return stats;
}

std::string MctsNode::TreeStats::ToString() const {
  return absl::StrFormat(
      "%d nodes, %d leaf, %.1f average children\n"
      "%.1f average depth, %d max depth\n",
      num_nodes, num_leaf_nodes,
      1.0f * num_nodes / std::max(1, num_nodes - num_leaf_nodes),
      1.0f * depth_sum / num_nodes, max_depth);
}

}  // namespace minigo
