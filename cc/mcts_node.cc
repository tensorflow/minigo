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

namespace minigo {

constexpr float MctsNode::kPuct;

MctsNode::MctsNode(EdgeStats* stats, const Position& position)
    : parent(nullptr), stats(stats), move(Coord::kInvalid), position(position) {
  for (int i = 0; i < kNumMoves; ++i) {
    illegal_moves[i] = position.IsMoveLegal(i) ? 0 : 1000;
  }
  DualNet::InitializeFeatures(position, &features);
}

MctsNode::MctsNode(MctsNode* parent, Coord move)
    : parent(parent),
      stats(&parent->edges[move]),
      move(move),
      position(parent->position) {
  position.PlayMove(move);
  for (int i = 0; i < kNumMoves; ++i) {
    illegal_moves[i] = position.IsMoveLegal(i) ? 0 : 1000;
  }
  DualNet::UpdateFeatures(parent->features, position, &features);
}

std::string MctsNode::Describe() const {
  using std::setprecision;
  using std::setw;

  auto child_action_score = CalculateChildActionScore();
  using SortInfo = std::tuple<float, float, int>;
  std::array<SortInfo, kNumMoves> sort_order;
  for (int i = 0; i < kNumMoves; ++i) {
    sort_order[i] = {child_N(i), child_action_score[i], i};
  }
  std::sort(sort_order.begin(), sort_order.end(), std::greater<SortInfo>());

  std::ostringstream oss;
  oss << std::fixed;
  oss << setprecision(4) << Q() << "\n";
  oss << MostVisitedPath() << "\n";
  oss << "move : action    Q     U     P   P-Dir    N  soft-N  p-delta  "
         "p-rel\n";

  float child_N_sum = 0;
  for (const auto& e : edges) {
    child_N_sum += e.N;
  }
  for (int rank = 0; rank < 15; ++rank) {
    Coord i = std::get<2>(sort_order[rank]);
    if (child_N(i) == 0) {
      break;
    }
    float soft_N = child_N(i) / child_N_sum;
    float p_delta = soft_N - child_P(i);
    float p_rel = p_delta / child_P(i);
    // clang-format off
    oss << std::left << setw(5) << i.ToKgs() << std::right
        << ": " << setw(6) << setprecision(3) << child_action_score[i]
        << " " << setw(6) << child_Q(i)
        << " " << setw(5) << child_U(i)
        << " " << setw(5) << child_P(i)
        << " " << setw(5) << child_original_P(i)
        << " " << setw(5) << static_cast<int>(child_N(i))
        << " " << setw(5) << setprecision(4) << soft_N
        << " " << setw(8) << setprecision(5) << p_delta
        << " " << setw(5) << setprecision(2) << p_rel << "\n";
    // clang-format on
  }
  return oss.str();
}

std::string MctsNode::MostVisitedPath() const {
  std::ostringstream oss;

  const auto* node = this;
  while (!node->children.empty()) {
    int next_kid = ArgMax(
        node->edges,
        [](const EdgeStats& a, const EdgeStats& b) { return a.N < b.N; });
    auto it = node->children.find(next_kid);
    if (it == node->children.end()) {
      oss << "GAME END";
      break;
    }
    node = it->second.get();
    oss << node->move.ToKgs() << " (" << static_cast<int>(node->N())
        << ") ==> ";
  }
  oss << std::setprecision(5) << "Q: " << node->Q();
  return oss.str();
}

void MctsNode::InjectNoise(const std::array<float, kNumMoves>& noise) {
  for (int i = 0; i < kNumMoves; ++i) {
    edges[i].P = 0.75f * edges[i].P + 0.25f * noise[i];
  }
}

MctsNode* MctsNode::SelectLeaf() {
  auto* node = this;
  for (;;) {
    ++node->stats->N;

    // If a node has never been evaluated, we have no basis to select a child.
    if (node->state == State::kCollapsed) {
      node->state = State::kSelected;
      return node;
    }
    // If the node has already been selected for the next inference batch, we
    // shouldn't select it again.
    if (node->state == State::kSelected) {
      // Revert the visits we just made.
      --node->stats->N;
      while (node != this) {
        node = node->parent;
        --node->stats->N;
      }
      return nullptr;
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
  assert(state == State::kSelected);
  state = State::kExpanded;
  for (int i = 0; i < kNumMoves; ++i) {
    edges[i].original_P = edges[i].P = move_probabilities[i];
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
  assert(state == State::kSelected);
  // We can't expand an end game result, so collapse the node again.
  state = State::kCollapsed;
  BackupValue(value, up_to);
}

void MctsNode::BackupValue(float value, MctsNode* up_to) {
  auto* node = this;
  for (;;) {
    node->stats->W += value;
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
  float U_scale = kPuct * std::sqrt(1.0f + N());

  std::array<float, kNumMoves> result;
  for (int i = 0; i < kNumMoves; ++i) {
    float Q = child_Q(i);
    float U = U_scale * child_P(i) / (1 + child_N(i));
    result[i] = Q * to_play + U - illegal_moves[i];
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
