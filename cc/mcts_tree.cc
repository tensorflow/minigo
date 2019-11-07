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

#include "cc/mcts_tree.h"

#include <emmintrin.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <tuple>
#include <utility>

#include "absl/strings/str_format.h"
#include "absl/types/optional.h"
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

absl::optional<symmetry::Symmetry> CalculateCanonicalSymmetry(
    const Position& position) {
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
    return symmetry::Inverse(best_symmetry);
  }
  return absl::nullopt;
}

constexpr int kSuperKoCacheStride = 8;

}  // namespace

MctsNode::MctsNode(EdgeStats* stats, const Position& position)
    : parent(nullptr),
      stats(stats),
      stats_idx(0),
      move(Coord::kInvalid),
      is_expanded(false),
      has_canonical_symmetry(false),
      position(position) {}

MctsNode::MctsNode(MctsNode* parent, Coord move)
    : parent(parent),
      stats(&parent->edges),
      stats_idx(move),
      move(move),
      is_expanded(false),
      has_canonical_symmetry(parent->has_canonical_symmetry),
      canonical_symmetry(parent->canonical_symmetry),
      position(parent->position) {
  // TODO(tommadams): move this code into the MctsTree and only perform it
  // only if we are using an inference cache.
  if (!has_canonical_symmetry) {
    auto sym = CalculateCanonicalSymmetry(position);
    if (sym.has_value()) {
      has_canonical_symmetry = true;
      canonical_symmetry = sym.value();
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

Coord MctsNode::GetMostVisitedMove(bool restrict_pass_alive) const {
  // Find the set of moves with the largest N.
  inline_vector<Coord, kNumMoves> moves;
  // CalculatePassAlive does not include the kPass point.
  std::array<Color, kN * kN> out_of_bounds;

  if (restrict_pass_alive) {
    out_of_bounds = position.CalculatePassAliveRegions();
  } else {
    for (auto& x : out_of_bounds) {
      x = Color::kEmpty;
    }
  }

  int best_N = 0;
  for (int i = 0; i < kNumMoves; ++i) {
    if ((i != Coord::kPass) && (out_of_bounds[i] != Color::kEmpty)) {
      continue;
    }
    int cn = child_N(i);
    if (cn >= best_N) {
      if (cn > best_N) {
        moves.clear();
        best_N = cn;
      }
      moves.push_back(i);
    }
  }

  if (moves.empty()) {
    return Coord::kPass;
  }

  // If there's only one move with the largest N, we're done.
  if (moves.size() == 1) {
    return moves[0];
  }

  // Otherwise, break tie using the child action score.
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

std::vector<Coord> MctsNode::GetMostVisitedPath() const {
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

std::string MctsNode::GetMostVisitedPathString() const {
  std::string result;
  const auto* node = this;
  for (Coord c : GetMostVisitedPath()) {
    auto it = node->children.find(c);
    MG_CHECK(it != node->children.end());
    node = it->second.get();
    absl::StrAppendFormat(&result, "%s (%d) ==> ", node->move.ToGtp(),
                          node->N());
  }
  absl::StrAppendFormat(&result, "Q: %0.5f", node->Q());
  return result;
}

void MctsNode::PruneChildren(Coord c) {
  auto child = std::move(children[c]);
  children.clear();
  children[c] = std::move(child);
}

void MctsNode::ClearChildren() {
  // I _think_ this is all the state we need to clear...
  children.clear();
  edges = {};
  *stats = {};
  is_expanded = false;
}

// Vectorized version of CalculateChildActionScore.
void MctsNode::CalculateChildActionScoreSse(PaddedSpan<float> result) const {
  __m128 to_play = _mm_set_ps1(position.to_play() == Color::kBlack ? 1 : -1);
  __m128 U_common =
      _mm_set_ps1(U_scale() * std::sqrt(std::max<float>(1, N() - 1)));

  // A couple of useful constants.
  __m128i one = _mm_set1_epi32(1);
  __m128 one_thousand = _mm_set_ps1(1000);

  for (int i = 0; i < kNumMoves; i += 4) {
    // `rcp_N_one = 1 / (1 + child_N(i))`
    // The division is performed using an approximate reciprocal instruction
    // that has a maximum relative error of 1.5 * 2^-12.
    __m128i N =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(edges.N.data() + i));
    __m128 rcp_N_one = _mm_rcp_ps(_mm_cvtepi32_ps(_mm_add_epi32(one, N)));

    // `Q = child_W(i) / (1 + child_N(i))`
    __m128 W = _mm_loadu_ps(edges.W.data() + i);
    __m128 Q = _mm_mul_ps(W, rcp_N_one);

    // `U = U_common * child_P(i) / (1 + child_N(i))`
    __m128 P = _mm_loadu_ps(edges.P.data() + i);
    __m128 U = _mm_mul_ps(_mm_mul_ps(U_common, P), rcp_N_one);

    // `legal_bits = position.legal_move(i)`
    // This requires a few instructions to load the legal move bytes and
    // shuffle them into each of the four vector slots.
    __m128i legal_bits = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(position.legal_moves().data() + i));
    legal_bits = _mm_unpacklo_epi8(legal_bits, _mm_setzero_si128());
    legal_bits = _mm_unpacklo_epi16(legal_bits, _mm_setzero_si128());

    // `legal = legal_bits == 0 ? 1000 : 0`
    __m128 legal =
        _mm_castsi128_ps(_mm_cmpeq_epi32(legal_bits, _mm_setzero_si128()));
    legal = _mm_and_ps(legal, one_thousand);

    // `child_action_score[i] = Q * to_play + U - legal`
    __m128 cas = _mm_sub_ps(_mm_add_ps(_mm_mul_ps(Q, to_play), U), legal);
    _mm_storeu_ps(result.data() + i, cas);
  }
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
    // TODO(tommadams): Allocate children out of a custom block allocator: we
    // spend about 5% of our runtme inside MctsNode::PruneChildren freeing
    // nodes.
    it = children.emplace(c, absl::make_unique<MctsNode>(this, c)).first;
  }
  return it->second.get();
}

std::string MctsTree::Stats::ToString() const {
  return absl::StrFormat(
      "%d nodes, %d leaf, %.1f average children\n"
      "%.1f average depth, %d max depth\n",
      num_nodes, num_leaf_nodes,
      1.0f * num_nodes / std::max(1, num_nodes - num_leaf_nodes),
      1.0f * depth_sum / num_nodes, max_depth);
}

std::ostream& operator<<(std::ostream& os, const MctsTree::Options& options) {
  return os << "value_init_penalty:" << options.value_init_penalty
            << " policy_softmax_temp:" << options.policy_softmax_temp
            << " soft_pick_enabled:" << options.soft_pick_enabled
            << " soft_pick_cutoff:" << options.soft_pick_cutoff;
}

MctsTree::MctsTree(const Position& position, const Options& options)
    : game_root_(&game_root_stats_, position), options_(options) {
  root_ = &game_root_;
}

MctsNode* MctsTree::SelectLeaf(bool allow_pass) {
  auto* node = root_;
  for (;;) {
    // If a node has never been evaluated, we have no basis to select a child.
    if (!node->is_expanded) {
      return node;
    }

    PaddedArray<float, kNumMoves> child_action_score;
    node->CalculateChildActionScoreSse(child_action_score);
    if (!allow_pass) {
      child_action_score[Coord::kPass] = -100000;
    }

    Coord best_move = ArgMaxSse(child_action_score);
    if (!node->position.legal_move(best_move)) {
      best_move = Coord::kPass;
    }

    node = node->MaybeAddChild(best_move);
  }
}

Coord MctsTree::PickMove(Random* rnd, bool restrict_pass_alive) const {
  if (options_.soft_pick_enabled &&
      root_->position.n() < options_.soft_pick_cutoff) {
    return SoftPickMove(rnd);
  } else {
    return PickMostVisitedMove(restrict_pass_alive);
  }
}

void MctsTree::PlayMove(Coord c) {
  MG_CHECK(!is_game_over() && is_legal_move(c))
      << c << " " << is_game_over() << " " << is_legal_move(c);
  root_ = root_->MaybeAddChild(c);
  // Don't need to keep the parent's children around anymore because we'll
  // never revisit them during normal play.
  // TODO(tommadams): we should just delete all ancestors. This will require
  // changes to UndoMove though.
  root_->parent->PruneChildren(c);
}

void MctsTree::AddVirtualLoss(MctsNode* leaf) {
  auto* node = leaf;
  for (;;) {
    ++node->num_virtual_losses_applied;
    node->stats->W[node->stats_idx] +=
        node->position.to_play() == Color::kBlack ? 1 : -1;
    if (node == root_) {
      return;
    }
    node = node->parent;
  }
}

void MctsTree::RevertVirtualLoss(MctsNode* leaf) {
  auto* node = leaf;
  for (;;) {
    --node->num_virtual_losses_applied;
    node->stats->W[node->stats_idx] -=
        node->position.to_play() == Color::kBlack ? 1 : -1;
    if (node == root_) {
      return;
    }
    node = node->parent;
  }
}

void MctsTree::IncorporateResults(MctsNode* leaf,
                                  absl::Span<const float> move_probabilities,
                                  float value) {
  MG_DCHECK(move_probabilities.size() == kNumMoves);
  // A finished game should not be going through this code path, it should
  // directly call BackupValue on the result of the game.
  MG_DCHECK(!leaf->game_over());

  // If the node has already been selected for the next inference batch, we
  // shouldn't 'expand' it again.
  if (leaf->is_expanded) {
    return;
  }

  float policy_scalar = 0;
  for (int i = 0; i < kNumMoves; ++i) {
    if (leaf->position.legal_move(i)) {
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
  float reduction = options_.value_init_penalty *
                    (leaf->position.to_play() == Color::kBlack ? 1 : -1);
  float reduced_value = std::min(1.0f, std::max(-1.0f, value - reduction));

  leaf->is_expanded = true;
  for (int i = 0; i < kNumMoves; ++i) {
    // Zero out illegal moves, and re-normalize move_probabilities.
    float move_prob = leaf->position.legal_move(i)
                          ? policy_scalar * move_probabilities[i]
                          : 0;

    leaf->edges.original_P[i] = leaf->edges.P[i] = move_prob;

    // Note that we accumulate W here, rather than assigning.
    // When performing tree search normally, we could just assign the value to W
    // because the result of value head is known before we expand the node.
    // When running Minigui in study move however, we load the entire game tree
    // before starting background inference. This means that while background
    // inferences are being performed, nodes in the tree may already be expanded
    // and have non-zero W values at the time we need to incorporate a result
    // for the node from the value head.
    // TODO(tommadams): Minigui doesn't work this way any more so we can just
    // assign.
    leaf->edges.W[i] += reduced_value;
  }
  BackupValue(leaf, value);
}

void MctsTree::IncorporateEndGameResult(MctsNode* leaf, float value) {
  MG_DCHECK(leaf->game_over());
  MG_DCHECK(!leaf->is_expanded);
  BackupValue(leaf, value);
}

void MctsTree::BackupValue(MctsNode* leaf, float value) {
  auto* node = leaf;
  for (;;) {
    node->stats->W[node->stats_idx] += value;
    node->stats->N[node->stats_idx] += 1;
    if (node == root_) {
      return;
    }
    node = node->parent;
  }
}

void MctsTree::InjectNoise(const std::array<float, kNumMoves>& noise,
                           float mix) {
  MG_CHECK(root_->is_expanded);

  // NOTE: our interpretation is to only add dirichlet noise to legal moves.
  // Because dirichlet entries are independent we can simply zero and rescale.

  float scalar = 0;
  for (int i = 0; i < kNumMoves; ++i) {
    if (root_->position.legal_move(i)) {
      scalar += noise[i];
    }
  }

  if (scalar > std::numeric_limits<float>::min()) {
    scalar = 1.0 / scalar;
  }

  for (int i = 0; i < kNumMoves; ++i) {
    float scaled_noise =
        scalar * (root_->position.legal_move(i) ? noise[i] : 0);
    root_->edges.P[i] = (1 - mix) * root_->edges.P[i] + mix * scaled_noise;
  }
}

void MctsTree::ReshapeFinalVisits(bool restrict_pass_alive) {
  // Since we aren't actually disallowing *reads* of bensons moves, only their
  // selection, we get the most visited move regardless of bensons status and
  // reshape based on its action score.
  Coord best = root_->GetMostVisitedMove(false);
  MG_CHECK(root_->edges.N[best] > 0);
  auto pass_alive_regions = root_->position.CalculatePassAliveRegions();
  float U_common = root_->U_scale() * std::sqrt(1.0f + root_->N());
  float to_play = root_->position.to_play() == Color::kBlack ? 1 : -1;
  float best_cas = root_->CalculateSingleMoveChildActionScore(to_play, U_common,
                                                              uint16_t(best));

  bool any = false;  // Track if any move has visits after pruning.

  // We explored this child with uncertainty about its value.  Now, after
  // searching, we change the visit count to reflect how many visits we would
  // have given it with our newer understanding of its regret relative to our
  // best move.
  for (int i = 0; i < kNumMoves; ++i) {
    // Remove visits in pass alive areas.
    if (restrict_pass_alive && (i != Coord::kPass) &&
        (pass_alive_regions[i] != Color::kEmpty)) {
      root_->edges.N[i] = 0;
      continue;
    }

    // Skip the best move; it has the highest action score.
    if (i == best) {
      if (root_->edges.N[i] > 0) {
        any = true;
      }
      continue;
    }

    // Change N_child to the smallest value that satisfies the inequality
    // best_cas > Q + (U_scale * P * sqrt(N_parent) / N_child)
    // Solving for N_child, we get:
    int new_N = std::max<int>(
        0, std::min<int>(root_->child_N(i),
                         -1 *
                             (root_->U_scale() * root_->child_P(i) *
                              std::sqrt(root_->N())) /
                             ((root_->child_Q(i) * to_play) - best_cas)));
    root_->edges.N[i] = new_N;

    if (root_->edges.N[i] > 0) {
      any = true;
    }
  }

  // If all visits were in bensons regions, put a visit on pass.
  if (!any) {
    root_->edges.N[Coord::kPass] = 1;
  }
}

std::array<float, kNumMoves> MctsTree::CalculateSearchPi() const {
  std::array<float, kNumMoves> search_pi;
  if (options_.soft_pick_enabled &&
      root_->position.n() < options_.soft_pick_cutoff) {
    // Squash counts before normalizing to match softpick behavior in PickMove.
    for (int i = 0; i < kNumMoves; ++i) {
      search_pi[i] = std::pow(root_->child_N(i), options_.policy_softmax_temp);
    }
  } else {
    for (int i = 0; i < kNumMoves; ++i) {
      search_pi[i] = root_->child_N(i);
    }
  }
  // Normalize counts.
  float sum = 0;
  for (int i = 0; i < kNumMoves; ++i) {
    sum += search_pi[i];
  }
  MG_CHECK(sum > 0);
  for (int i = 0; i < kNumMoves; ++i) {
    search_pi[i] /= sum;
  }
  return search_pi;
}

MctsTree::Stats MctsTree::CalculateStats() const {
  Stats stats;

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

  traverse(*root_, 0);

  return stats;
}

std::string MctsTree::Describe() const {
  auto sorted_child_info = CalculateRankedChildInfo();

  auto result = absl::StrFormat(
      "%0.4f\n%s\n"
      "move : action    Q     U     P   P-Dir    N  soft-N  p-delta  p-rel",
      root_->Q(), root_->GetMostVisitedPathString());

  float child_N_sum = 0;
  for (const auto& N : root_->edges.N) {
    child_N_sum += N;
  }
  for (int rank = 0; rank < 15; ++rank) {
    Coord c = sorted_child_info[rank].c;
    float soft_N = root_->child_N(c) / child_N_sum;
    float p_delta = soft_N - root_->child_P(c);
    float p_rel = p_delta / root_->child_P(c);
    absl::StrAppendFormat(
        &result,
        "\n%-5s: % 4.3f % 4.3f %0.3f %0.3f %0.3f %5d %0.4f % 6.5f % 3.2f",
        c.ToGtp(), sorted_child_info[rank].action_score, root_->child_Q(c),
        root_->child_U(c), root_->child_P(c), root_->child_original_P(c),
        root_->child_N(c), soft_N, p_delta, p_rel);
  }
  return result;
}

std::array<MctsTree::ChildInfo, kNumMoves> MctsTree::CalculateRankedChildInfo()
    const {
  auto child_action_score = root_->CalculateChildActionScore();
  std::array<ChildInfo, kNumMoves> child_info;
  for (int i = 0; i < kNumMoves; ++i) {
    child_info[i].c = i;
    child_info[i].N = root_->child_N(i);
    child_info[i].P = root_->child_P(i);
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

bool MctsTree::UndoMove() {
  if (root_ == &game_root_) {
    return false;
  }
  root_ = root_->parent;
  return true;
}

Coord MctsTree::PickMostVisitedMove(bool restrict_pass_alive) const {
  auto c = root_->GetMostVisitedMove(restrict_pass_alive);
  if (!root_->position.legal_move(c)) {
    c = Coord::kPass;
  }
  return c;
}

// SoftPickMove is only called for the opening moves of the game, so we don't
// bother restricting play in pass-alive territory.
Coord MctsTree::SoftPickMove(Random* rnd) const {
  // Select from the first kN * kN moves (instead of kNumMoves) to avoid
  // randomly choosing to pass early on in the game.
  std::array<float, kN * kN> cdf;

  // For moves before the temperature cutoff, exponentiate the probabilities by
  // a temperature slightly larger than unity to encourage diversity in early
  // play and hopefully to move away from 3-3s.
  for (size_t i = 0; i < cdf.size(); ++i) {
    cdf[i] = std::pow(root_->child_N(i), options_.policy_softmax_temp);
  }
  for (size_t i = 1; i < cdf.size(); ++i) {
    cdf[i] += cdf[i - 1];
  }

  if (cdf.back() == 0) {
    // It's actually possible for an early model to put all its reads into pass,
    // in which case the SearchSorted call below will always return 0. In this
    // case, we'll just let the model have its way and allow a pass.
    return Coord::kPass;
  }

  Coord c = rnd->SampleCdf(absl::MakeSpan(cdf));
  MG_DCHECK(root_->child_N(c) != 0);
  return c;
}

}  // namespace minigo
