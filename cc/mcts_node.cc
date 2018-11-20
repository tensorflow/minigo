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
#include "cc/check.h"

namespace minigo {

namespace {

void InitLegalMoves(MctsNode* node) {
  auto& position = node->position;
  auto position_hash = position.stone_hash();
  auto to_play = position.to_play();
  for (int c = 0; c < kN * kN; ++c) {
    switch (position.ClassifyMove(c)) {
      case Position::MoveType::kIllegal: {
        // The move is trivially not legal.
        node->legal_moves[c] = false;
        break;
      }

      case Position::MoveType::kNoCapture: {
        // The move will not capture any stones: we can calculate the new
        // position's stone hash directly.
        auto new_hash = position_hash ^ zobrist::MoveHash(c, to_play);
        node->legal_moves[c] = !node->HasPositionBeenPlayedBefore(new_hash);
        break;
      }

      case Position::MoveType::kCapture: {
        // The move will capture some opponent stones: in order to calculate the
        // stone hash, we actually have to play the move.

        Position new_position(position);
        // It's safe to call AddStoneToBoard instead of PlayMove because:
        //  - we know the move is not kPass.
        //  - the move is legal (modulo superko).
        //  - we only care about new_position's stone_hash and not the rest of
        //    the bookkeeping that PlayMove updates.
        new_position.AddStoneToBoard(c, to_play);
        auto new_hash = new_position.stone_hash();
        node->legal_moves[c] = !node->HasPositionBeenPlayedBefore(new_hash);
        break;
      }
    }
  }
  node->legal_moves[Coord::kPass] = true;
}

constexpr int kSuperKoCacheStride = 8;

}  // namespace

MctsNode::MctsNode(EdgeStats* stats, const Position& position)
    : parent(nullptr), stats(stats), move(Coord::kInvalid), position(position) {
  InitLegalMoves(this);
}

MctsNode::MctsNode(MctsNode* parent, Coord move)
    : parent(parent),
      stats(&parent->edges[move]),
      move(move),
      position(parent->position) {
  position.PlayMove(move);

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

  InitLegalMoves(this);
}

Coord MctsNode::GetMostVisitedMove() const {
  // Find the set of moves with the largest N.
  inline_vector<Coord, kNumMoves> moves;
  int best_N = -1;
  for (int i = 0; i < kNumMoves; ++i) {
    int cn = child_N(i);
    // In cases like Minigui's Study mode, where we can add nodes directly into
    // the tree without calling TreeSearch, and where we can freely change the
    // current root around, it's possible to end up in a situation where none of
    // the children of the game root have been visited (N == 0).
    // TODO(tommadams): consider returning Coord::kInvalid in these cases
    // instead of a node with 0 visits.
    if (cn >= best_N && children.contains(i)) {
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
  float U_scale = kPuct * std::sqrt(1.0f + N());

  Coord c = moves[0];
  float best_cas =
      CalculateSingleMoveChildActionScore(to_play, U_scale, moves[0]);
  for (int i = 0; i < moves.size(); ++i) {
    float cas = CalculateSingleMoveChildActionScore(to_play, U_scale, moves[i]);
    if (cas > best_cas) {
      best_cas = cas;
      c = moves[i];
    }
  }

  return c;
}

std::string MctsNode::Describe() const {
  auto child_action_score = CalculateChildActionScore();
  using SortInfo = std::tuple<float, float, int>;
  std::array<SortInfo, kNumMoves> sort_order;
  for (int i = 0; i < kNumMoves; ++i) {
    sort_order[i] = SortInfo(child_N(i), child_action_score[i], i);
  }
  std::sort(sort_order.begin(), sort_order.end(), std::greater<SortInfo>());

  auto result = absl::StrFormat(
      "%0.4f\n%s\n"
      "move : action    Q     U     P   P-Dir    N  soft-N  p-delta  p-rel",
      Q(), MostVisitedPathString());

  float child_N_sum = 0;
  for (const auto& e : edges) {
    child_N_sum += e.N;
  }
  for (int rank = 0; rank < 15; ++rank) {
    Coord i = std::get<2>(sort_order[rank]);
    float soft_N = child_N(i) / child_N_sum;
    float p_delta = soft_N - child_P(i);
    float p_rel = p_delta / child_P(i);
    absl::StrAppendFormat(
        &result,
        "\n%-5s: % 4.3f % 4.3f %0.3f %0.3f %0.3f %5d %0.4f % 6.5f % 3.2f",
        i.ToKgs(), child_action_score[i], child_Q(i), child_U(i), child_P(i),
        child_original_P(i), static_cast<int>(child_N(i)), soft_N, p_delta,
        p_rel);
  }
  return result;
}

std::vector<Coord> MctsNode::MostVisitedPath() const {
  std::vector<Coord> path;
  const auto* node = this;
  while (!node->children.empty()) {
    Coord next_kid = node->GetMostVisitedMove();
    path.push_back(next_kid);
    auto it = node->children.find(next_kid);
    MG_CHECK(it != node->children.end());
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
    absl::StrAppendFormat(&result, "%s (%d) ==> ", node->move.ToKgs(),
                          static_cast<int>(node->N()));
  }
  absl::StrAppendFormat(&result, "Q: %0.5f", node->Q());
  return result;
}

void MctsNode::GetMoveHistory(
    int num_moves, std::vector<const Position::Stones*>* history) const {
  history->clear();
  history->reserve(num_moves);
  const auto* node = this;
  for (int j = 0; j < num_moves; ++j) {
    history->push_back(&node->position.stones());
    node = node->parent;
    if (node == nullptr) {
      break;
    }
  }
}

void MctsNode::InjectNoise(const std::array<float, kNumMoves>& noise) {
  // NOTE: our interpretation is to only add dirichlet noise to legal moves.
  // Because dirichlet entries are independent we can simply zero and rescale.

  float scalar = 0;
  for (int i = 0; i < kNumMoves; ++i) {
    if (legal_moves[i]) {
      scalar += noise[i];
    }
  }

  if (scalar > std::numeric_limits<float>::min()) {
    scalar = 1.0 / scalar;
  }

  for (int i = 0; i < kNumMoves; ++i) {
    float scaled_noise = scalar * (legal_moves[i] ? noise[i] : 0);
    edges[i].P = 0.75f * edges[i].P + 0.25f * scaled_noise;
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
    if (node->position.previous_move() == Coord::kPass &&
        node->child_N(Coord::kPass) == 0) {
      node = node->MaybeAddChild(Coord::kPass);
      continue;
    }

    auto child_action_score = node->CalculateChildActionScore();
    Coord best_move = ArgMax(child_action_score);
    node = node->MaybeAddChild(best_move);
  }
}

void MctsNode::IncorporateResults(absl::Span<const float> move_probabilities,
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
    if (legal_moves[i]) {
      policy_scalar += move_probabilities[i];
    }
  }
  if (policy_scalar > std::numeric_limits<float>::min()) {
    policy_scalar = 1 / policy_scalar;
  }

  SetFlag(Flag::kExpanded);
  for (int i = 0; i < kNumMoves; ++i) {
    // Zero out illegal moves, and re-normalize move_probabilities.
    float move_prob =
        legal_moves[i] ? policy_scalar * move_probabilities[i] : 0;

    edges[i].original_P = edges[i].P = move_prob;
    // Initialize child Q as current node's value, to prevent dynamics where
    // if B is winning, then B will only ever explore 1 move, because the Q
    // estimation will be so much larger than the 0 of the other moves.
    //
    // Conversely, if W is winning, then B will explore all 362 moves before
    // continuing to explore the most favorable move. This is a waste of
    // search.
    //
    // The value seeded here acts as a prior, and gets averaged into Q
    // calculations.
    edges[i].W += value;
  }
  BackupValue(value, up_to);
}

void MctsNode::IncorporateEndGameResult(float value, MctsNode* up_to) {
  MG_DCHECK(game_over() || position.n() == kMaxSearchDepth);
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
  float U_scale = kPuct * std::sqrt(std::max<float>(1, N() - 1));

  std::array<float, kNumMoves> result;
  for (int i = 0; i < kNumMoves; ++i) {
    result[i] = CalculateSingleMoveChildActionScore(to_play, U_scale, i);
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

bool MctsNode::HasPositionBeenPlayedBefore(zobrist::Hash stone_hash) const {
  for (const auto* node = this; node != nullptr; node = node->parent) {
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

}  // namespace minigo
