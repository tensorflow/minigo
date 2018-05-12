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

#include "cc/mcts_player.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/time/clock.h"
#include "cc/check.h"
#include "cc/random.h"

namespace minigo {

std::ostream& operator<<(std::ostream& os, const MctsPlayer::Options& options) {
  os << "inject_noise:" << options.inject_noise
     << " soft_pick:" << options.soft_pick
     << " random_symmetry:" << options.random_symmetry
     << " resign_threshold:" << options.resign_threshold
     << " batch_size:" << options.batch_size << " komi:" << options.komi
     << " random_seed:" << options.random_seed;
  return os;
}

MctsPlayer::MctsPlayer(std::unique_ptr<DualNet> network, const Options& options)
    : network_(std::move(network)),
      game_root_(&dummy_stats_, {&bv_, &gv_, Color::kBlack}),
      rnd_(options.random_seed),
      options_(options) {
  options_.resign_threshold = -std::abs(options_.resign_threshold);
  // When to do deterministic move selection: 30 moves on a 19x19, 6 on 9x9.
  temperature_cutoff_ = kN * kN / 12;
  root_ = &game_root_;

  InitializeGame({&bv_, &gv_, Color::kBlack});
}

void MctsPlayer::InitializeGame(const Position& position) {
  game_root_ = {&dummy_stats_, Position(&bv_, &gv_, position)};
  root_ = &game_root_;
  game_over_ = false;
}

void MctsPlayer::NewGame() {
  game_root_ = MctsNode(&dummy_stats_, {&bv_, &gv_, Color::kBlack});
  root_ = &game_root_;
  game_over_ = false;
}

Coord MctsPlayer::SuggestMove(int num_readouts) {
  std::array<float, kNumMoves> noise;
  if (options_.inject_noise) {
    // In order to be able to inject noise into the root node, we need to first
    // expand it. This should be the only time when SuggestMove is called when
    // the root isn't expanded.
    if (!root_->is_expanded) {
      MG_CHECK(root_ == &game_root_);
      auto* first_node = root_->SelectLeaf();
      auto output = Run(&first_node->features);
      first_node->IncorporateResults(output.policy, output.value, first_node);
    }

    rnd_.Dirichlet(kDirichletAlpha, &noise);
    root_->InjectNoise(noise);
  }
  int current_readouts = root_->N();

  auto start = absl::Now();
  while (root_->N() < current_readouts + num_readouts) {
    TreeSearch(options_.batch_size);
  }
  auto elapsed = absl::Now() - start;
  elapsed = elapsed * 100 / num_readouts;
  std::cerr << "Milliseconds per 100 reads: "
            << absl::ToInt64Milliseconds(elapsed) << "ms" << std::endl;

  if (ShouldResign()) {
    return Coord::kResign;
  }

  return PickMove();
}

Coord MctsPlayer::PickMove() {
  if (!options_.soft_pick || root_->position.n() >= temperature_cutoff_) {
    // Choose the most visited node.
    Coord c = ArgMax(root_->edges, MctsNode::CmpN);
    std::cerr << "Picked arg_max " << c << "\n";
    return c;
  }

  // Select from the first kN * kN moves (instead of kNumMoves) to avoid
  // randomly choosing to pass early on in the game.
  std::array<float, kN * kN> cdf;

  cdf[0] = root_->child_N(0);
  for (size_t i = 1; i < cdf.size(); ++i) {
    cdf[i] = cdf[i - 1] + root_->child_N(i);
  }
  float norm = 1 / cdf[cdf.size() - 1];
  for (size_t i = 0; i < cdf.size(); ++i) {
    cdf[i] *= norm;
  }
  float e = rnd_();
  Coord c = SearchSorted(cdf, e);
  std::cerr << "Picked rnd(" << e << ") " << c << "\n";
  MG_DCHECK(root_->child_N(c) != 0);
  return c;
}

absl::Span<MctsNode* const> MctsPlayer::TreeSearch(int batch_size) {
  int max_iterations = batch_size * 2;

  // TODO(tommadams): Avoid creating this vector each time.
  leaves_.clear();
  for (int i = 0; i < max_iterations; ++i) {
    auto* leaf = root_->SelectLeaf();
    if (leaf == nullptr) {
      continue;
    }
    if (leaf->position.is_game_over() ||
        leaf->position.n() >= kMaxSearchDepth) {
      float value = leaf->position.CalculateScore(options_.komi) > 0 ? 1 : -1;
      leaf->IncorporateEndGameResult(value, root_);
    } else {
      leaf->AddVirtualLoss(root_);
      leaves_.push_back(leaf);
      if (static_cast<int>(leaves_.size()) == batch_size) {
        break;
      }
    }
  }

  if (!leaves_.empty()) {
    features_.clear();
    features_.reserve(leaves_.size());
    for (auto* leaf : leaves_) {
      features_.push_back(&leaf->features);
    }

    outputs_.resize(leaves_.size());
    RunMany(features_, {outputs_.data(), outputs_.size()});

    for (size_t i = 0; i < leaves_.size(); ++i) {
      MctsNode* leaf = leaves_[i];
      const auto& output = outputs_[i];
      leaf->RevertVirtualLoss(root_);
      leaf->IncorporateResults(output.policy, output.value, root_);
    }
  }

  return absl::MakeConstSpan(leaves_);
}

bool MctsPlayer::ShouldResign() const {
  return root_->Q_perspective() < options_.resign_threshold;
}

void MctsPlayer::PlayMove(Coord c) {
  if (game_over_) {
    std::cerr << "ERROR: can't play move " << c << ", game is over"
              << std::endl;
    return;
  }

  // Handle resignations.
  if (c == Coord::kResign) {
    if (root_->position.to_play() == Color::kBlack) {
      result_ = -1;
      result_string_ = "W+R";
    } else {
      result_ = 1;
      result_string_ = "B+R";
    }
    game_over_ = true;
    return;
  }

  PushHistory(c);

  root_ = root_->MaybeAddChild(c);
  // Don't need to keep the parent's children around anymore because we'll
  // never revisit them.
  root_->parent->PruneChildren(c);

  std::cerr << "Q: " << std::setw(8) << std::setprecision(5) << root_->Q()
            << "\n";
  std::cerr << "Played >>" << c << std::endl;

  // Handle consecutive passing.
  if (root_->position.is_game_over() ||
      root_->position.n() >= kMaxSearchDepth) {
    float score = root_->position.CalculateScore(options_.komi);
    result_string_ = FormatScore(score);
    result_ = score < 0 ? -1 : score > 0 ? 1 : 0;
    game_over_ = true;
  }
}

std::string MctsPlayer::FormatScore(float score) const {
  std::ostringstream oss;
  oss << std::fixed;
  if (score > 0) {
    oss << "B+" << std::setprecision(1) << score;
  } else {
    oss << "W+" << std::setprecision(1) << -score;
  }
  return oss.str();
}

void MctsPlayer::PushHistory(Coord c) {
  history_.emplace_back();
  History& history = history_.back();
  history.c = c;
  history.comment = root_->Describe();
  history.node = root_;

  // Convert child visit counts to a probability distribution, pi.
  // For moves before the temperature cutoff, exponentiate the probabilities by
  // a temperature slightly larger than unity to encourage diversity in early
  // play and hopefully to move away from 3-3s.
  if (root_->position.n() < temperature_cutoff_) {
    // Squash counts before normalizing.
    for (int i = 0; i < kNumMoves; ++i) {
      history.search_pi[i] = std::pow(root_->child_N(i), 0.98);
    }
  } else {
    for (int i = 0; i < kNumMoves; ++i) {
      history.search_pi[i] = root_->child_N(i);
    }
  }
  // Normalize counts.
  float sum = 0;
  for (int i = 0; i < kNumMoves; ++i) {
    sum += history.search_pi[i];
  }
  for (int i = 0; i < kNumMoves; ++i) {
    history.search_pi[i] /= sum;
  }
}

DualNet::Output MctsPlayer::Run(const DualNet::BoardFeatures* features) {
  DualNet::Output output;
  RunMany({&features, 1}, {&output, 1});
  return output;
}

void MctsPlayer::RunMany(
    absl::Span<const DualNet::BoardFeatures* const> features,
    absl::Span<DualNet::Output> outputs) {
  network_->RunMany(features, outputs,
                    options_.random_symmetry ? &rnd_ : nullptr);
}

}  // namespace minigo
