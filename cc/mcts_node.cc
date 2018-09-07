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
#include <iomanip>
#include <sstream>
#include <tuple>
#include <utility>

#include "cc/algorithm.h"
#include "cc/check.h"

namespace minigo {

MctsNode::MctsNode(EdgeStats* stats, const Position& position)
    : parent(nullptr), stats(stats), move(Coord::kInvalid), position(position) {
  // TODO(tommadams): Only call IsMoveLegal if we want to select a leaf for
  // expansion.
  for (int i = 0; i < kNumMoves; ++i) {
    illegal_moves[i] = !position.IsMoveLegal(i);
  }
}

MctsNode::MctsNode(MctsNode* parent, Coord move)
    : parent(parent),
      stats(&parent->edges[move]),
      move(move),
      position(parent->position) {
  position.PlayMove(move);
  // TODO(tommadams): Only call IsMoveLegal if we want to select a leaf for
  // expansion.
  for (int i = 0; i < kNumMoves; ++i) {
    illegal_moves[i] = !position.IsMoveLegal(i);
  }
}

Coord MctsNode::GetMostVisitedMove() const {
  // Find the set of moves with the largest N.
  inline_vector<Coord, kNumMoves> moves;
  moves.push_back(0);
  int best_N = child_N(0);
  for (int i = 1; i < kNumMoves; ++i) {
    int cn = child_N(i);
    if (cn > best_N) {
      moves.clear();
      moves.push_back(i);
      best_N = cn;
    } else if (cn == best_N) {
      moves.push_back(i);
    }
  }

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
  using std::setprecision;
  using std::setw;

  auto child_action_score = CalculateChildActionScore();
  using SortInfo = std::tuple<float, float, int>;
  std::array<SortInfo, kNumMoves> sort_order;
  for (int i = 0; i < kNumMoves; ++i) {
    sort_order[i] = SortInfo(child_N(i), child_action_score[i], i);
  }
  std::sort(sort_order.begin(), sort_order.end(), std::greater<SortInfo>());

  std::ostringstream oss;
  oss << std::fixed;
  oss << setprecision(4) << Q() << "\n";
  oss << MostVisitedPathString() << "\n";
  oss << "move : action    Q     U     P   P-Dir    N  soft-N  p-delta  "
         "p-rel";

  float child_N_sum = 0;
  for (const auto& e : edges) {
    child_N_sum += e.N;
  }
  for (int rank = 0; rank < 15; ++rank) {
    Coord i = std::get<2>(sort_order[rank]);
    float soft_N = child_N(i) / child_N_sum;
    float p_delta = soft_N - child_P(i);
    float p_rel = p_delta / child_P(i);
    // clang-format off
    oss << "\n" << std::left << setw(5) << i.ToKgs() << std::right
        << ": " << setw(6) << setprecision(3) << child_action_score[i]
        << " " << setw(6) << child_Q(i)
        << " " << setw(5) << child_U(i)
        << " " << setw(5) << child_P(i)
        << " " << setw(5) << child_original_P(i)
        << " " << setw(5) << static_cast<int>(child_N(i))
        << " " << setw(5) << setprecision(4) << soft_N
        << " " << setw(8) << setprecision(5) << p_delta
        << " " << setw(5) << setprecision(2) << p_rel;
    // clang-format on
  }
  return oss.str();
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
  std::ostringstream oss;
  const auto* node = this;
  for (Coord c : MostVisitedPath()) {
    auto it = node->children.find(c);
    MG_CHECK(it != node->children.end());
    node = it->second.get();
    oss << node->move.ToKgs() << " (" << static_cast<int>(node->N())
        << ") ==> ";
  }
  oss << std::fixed << std::setprecision(5) << "Q: " << node->Q();
  return oss.str();
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
    if (illegal_moves[i] == 0) {
      scalar += noise[i];
    }
  }

  if (scalar > std::numeric_limits<float>::min()) {
    scalar = 1.0 / scalar;
  }

  for (int i = 0; i < kNumMoves; ++i) {
    float scaled_noise = scalar * (illegal_moves[i] ? 0 : noise[i]);
    edges[i].P = 0.75f * edges[i].P + 0.25f * scaled_noise;
  }
}

MctsNode* MctsNode::SelectLeaf() {
  auto* node = this;
  for (;;) {
    // If a node has never been evaluated, we have no basis to select a child.
    if (!node->is_expanded) {
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
  assert(move_probabilities.size() == kNumMoves);
  // A finished game should not be going through this code path, it should
  // directly call BackupValue on the result of the game.
  assert(!position.is_game_over());

  // If the node has already been selected for the next inference batch, we
  // shouldn't 'expand' it again.
  if (is_expanded) {
    return;
  }

  float policy_scalar = 0;
  for (int i = 0; i < kNumMoves; ++i) {
    if (!illegal_moves[i]) {
      policy_scalar += move_probabilities[i];
    }
  }
  if (policy_scalar > std::numeric_limits<float>::min()) {
    policy_scalar = 1 / policy_scalar;
  }

  is_expanded = true;
  for (int i = 0; i < kNumMoves; ++i) {
    // Zero out illegal moves, and re-normalize move_probabilities.
    float move_prob =
        illegal_moves[i] ? 0 : policy_scalar * move_probabilities[i];

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
    edges[i].W = value;
  }
  BackupValue(value, up_to);
}

void MctsNode::IncorporateEndGameResult(float value, MctsNode* up_to) {
  assert(position.is_game_over() || position.n() == kMaxSearchDepth);
  assert(!is_expanded);
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
  do {
    ++node->num_virtual_losses_applied;
    node->stats->W += node->position.to_play() == Color::kBlack ? 1 : -1;
    node = node->parent;
  } while (node != nullptr && node != up_to);
}

void MctsNode::RevertVirtualLoss(MctsNode* up_to) {
  auto* node = this;
  do {
    --node->num_virtual_losses_applied;
    node->stats->W -= node->position.to_play() == Color::kBlack ? 1 : -1;
    node = node->parent;
  } while (node != nullptr && node != up_to);
}

void MctsNode::PruneChildren(Coord c) {
  // TODO(tommadams): Allocate children out of a pool and return them here.
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
    // TODO(tommadams): Allocate children out of a pool.
    auto child = absl::make_unique<MctsNode>(this, c);
    MctsNode* result = child.get();
    children[c] = std::move(child);
    return result;
  } else {
    return it->second.get();
  }
}
}  // namespace minigo
