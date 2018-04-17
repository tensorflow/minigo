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

MctsPlayer::MctsPlayer(DualNet* network, const Options& options)
    : network_(std::move(network)),
      game_root_(&dummy_stats_, {&bv_, &gv_, options.komi, Color::kBlack}),
      rnd_(options.random_seed),
      options_(options) {
  options_.resign_threshold = -std::abs(options_.resign_threshold);
  // When to do deterministic move selection: 30 moves on a 19x19, 6 on 9x9.
  temperature_cutoff_ = kN * kN / 12;
  root_ = &game_root_;
  std::cout << "Using random seed " << rnd_.seed() << "\n";
}

void MctsPlayer::InitializeGame(const Position& position) {
  game_root_ = {&dummy_stats_, Position(&bv_, &gv_, position)};
}

void MctsPlayer::SelfPlay(int num_readouts) {
  auto* first_node = root_->SelectLeaf();
  MG_CHECK(first_node != nullptr);
  auto output = Run(&first_node->features);
  first_node->IncorporateResults(output.policy, output.value, first_node);

  std::array<float, kNumMoves> noise;
  for (;;) {
    if (options_.inject_noise) {
      rnd_.Dirichlet(kDirichletAlpha, &noise);
      root_->InjectNoise(noise);
    }
    int current_readouts = root_->N();

    auto start = absl::Now();
    while (root_->N() < current_readouts + num_readouts) {
      TreeSearch(options_.batch_size);
    }
    auto elapsed = absl::Now() - start;
    std::cout << "Seconds per 100 reads: " << elapsed * 100 / num_readouts
              << "\n";

    std::cout << root_->position.ToPrettyString();
    std::cout << root_->Describe() << "\n";

    if (ShouldResign()) {
      SetResult(root_->position.to_play() == Color::kBlack ? -1 : 1,
                root_->position.CalculateScore(),
                GameOverReason::kOpponentResigned);
      break;
    }

    auto move = PickMove();
    PlayMove(move);
    if (root_->position.is_game_over()) {
      float score = root_->position.CalculateScore();
      float result = score < 0 ? -1 : score > 0 ? 1 : 0;
      SetResult(result, score, GameOverReason::kBothPassed);
      break;
    } else if (root_->position.n() >= kMaxSearchDepth) {
      float score = root_->position.CalculateScore();
      float result = score < 0 ? -1 : score > 0 ? 1 : 0;
      SetResult(result, score, GameOverReason::kMoveLimitReached);
      break;
    }

    std::cout << "Q: " << std::setw(8) << std::setprecision(5) << root_->Q()
              << "\n";
    std::cout << "Played >>" << move << std::endl;
  }
}

Coord MctsPlayer::PickMove() {
  if (!options_.soft_pick || root_->position.n() > temperature_cutoff_) {
    // Choose the most visited node.
    Coord c = ArgMax(root_->edges, MctsNode::CmpN);
    std::cout << "Picked arg_max " << c << "\n";
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
  std::cout << "Picked rnd(" << e << ") " << c << "\n";
  MG_DCHECK(root_->child_N(c) != 0);
  return c;
}

void MctsPlayer::TreeSearch(int batch_size) {
  int max_iterations = batch_size * 2;

  // TODO(tommadams): Avoid creating this vector each time.
  std::vector<MctsNode*> leaves;
  for (int i = 0; i < max_iterations; ++i) {
    auto* leaf = root_->SelectLeaf();
    if (leaf == nullptr) {
      continue;
    }
    if (leaf->position.is_game_over() ||
        leaf->position.n() >= kMaxSearchDepth) {
      float value = leaf->position.CalculateScore() > 0 ? 1 : -1;
      leaf->IncorporateEndGameResult(value, root_);
    } else {
      leaf->AddVirtualLoss(root_);
      leaves.push_back(leaf);
      if (static_cast<int>(leaves.size()) == batch_size) {
        break;
      }
    }
  }

  if (!leaves.empty()) {
    // TODO(tommadams): Avoid creating these vectors each time.
    std::vector<const DualNet::BoardFeatures*> features;
    features.reserve(leaves.size());
    for (auto* leaf : leaves) {
      features.push_back(&leaf->features);
    }

    std::vector<DualNet::Output> outputs;
    outputs.resize(leaves.size());

    RunMany(features, {outputs.data(), outputs.size()});

    for (size_t i = 0; i < leaves.size(); ++i) {
      MctsNode* leaf = leaves[i];
      const auto& output = outputs[i];
      leaf->RevertVirtualLoss(root_);
      leaf->IncorporateResults(output.policy, output.value, root_);
    }
  }
}

bool MctsPlayer::ShouldResign() const {
  return root_->Q_perspective() < options_.resign_threshold;
}

void MctsPlayer::PlayMove(Coord c) {
  PushHistory(c);

  root_ = root_->MaybeAddChild(c);
  // Don't need to keep the parent's children around anymore because we'll
  // never revisit them.
  root_->parent->PruneChildren(c);
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

void MctsPlayer::SetResult(float result, float score, GameOverReason reason) {
  result_ = result;
  score_ = score;
  game_over_reason_ = reason;
  if (reason == GameOverReason::kOpponentResigned) {
    result_string_ = result_ == 1 ? "B+R" : "W+R";
  } else if (score == 0) {
    result_string_ = "DRAW";
  } else {
    std::ostringstream oss;
    oss << std::fixed;
    if (result > 0) {
      oss << "B+" << std::setprecision(1) << score;
    } else {
      oss << "W+" << std::setprecision(1) << -score;
    }
    result_string_ = oss.str();
  }

  std::cout << result_string_ << "\n";
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
