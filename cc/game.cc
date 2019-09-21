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

#include "cc/game.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "cc/logging.h"

namespace minigo {

std::ostream& operator<<(std::ostream& os, const Game::Options& options) {
  os << "resign_threshold:" << options.resign_threshold
     << " resign_enabled:" << options.resign_enabled << " komi:" << options.komi
     << " ignore_repeated_moves:" << options.ignore_repeated_moves;
  return os;
}

std::string Game::FormatScore(float score) {
  return absl::StrFormat("%c+%.1f", score > 0 ? 'B' : 'W', std::abs(score));
}

Game::Game(std::string black_name, std::string white_name,
           const Game::Options& options)
    : options_(options),
      black_name_(std::move(black_name)),
      white_name_(std::move(white_name)) {
  MG_CHECK(options_.resign_threshold < 0);
}

void Game::NewGame() {
  game_over_ = false;
  moves_.clear();
  comment_.clear();
}

void Game::AddComment(const std::string& comment) {
  if (comment_.empty()) {
    comment_ = comment;
  } else {
    absl::StrAppend(&comment_, "\n", comment);
  }
}

void Game::AddMove(Color color, Coord c, const Position& position,
                   std::string comment, float Q,
                   const std::array<float, kNumMoves>& search_pi,
                   std::vector<std::string> models) {
  if (!moves_.empty() && moves_.back()->color == color &&
      moves_.back()->c == c) {
    MG_CHECK(options_.ignore_repeated_moves)
        << "Repeated call to AddMove with same (color, coord) (" << color
        << ", " << c << ") and ignore_repeated_moves is false";
    return;
  }

  MG_CHECK(!game_over_);
  moves_.push_back(absl::make_unique<Move>(position));
  auto* move = moves_.back().get();
  move->color = color;
  move->c = c;
  move->Q = Q;
  move->comment = std::move(comment);
  move->models = std::move(models);
  move->search_pi = search_pi;
}

void Game::MarkLastMoveAsTrainable() {
  auto* move = moves_.back().get();
  move->trainable = true;
}

void Game::UndoMove() {
  MG_CHECK(!moves_.empty());
  moves_.pop_back();
  game_over_ = false;
}

void Game::SetGameOverBecauseOfPasses(float score) {
  MG_CHECK(!game_over_);
  game_over_ = true;
  game_over_reason_ = GameOverReason::kBothPassed;
  result_ = score < 0 ? -1 : score > 0 ? 1 : 0;
  result_string_ = FormatScore(score);
}

void Game::SetGameOverBecauseOfResign(Color winner) {
  MG_CHECK(!game_over_);
  game_over_ = true;
  game_over_reason_ = GameOverReason::kOpponentResigned;
  if (winner == Color::kBlack) {
    result_ = 1;
    result_string_ = "B+R";
  } else {
    result_ = -1;
    result_string_ = "W+R";
  }
}

void Game::SetGameOverBecauseMoveLimitReached(float score) {
  MG_CHECK(!game_over_);
  game_over_ = true;
  game_over_reason_ = GameOverReason::kMoveLimitReached;
  result_ = score < 0 ? -1 : score > 0 ? 1 : 0;
  result_string_ = FormatScore(score);
}

bool Game::FindBleakestMove(int* move, float* q) const {
  if (!game_over_) {
    MG_LOG(ERROR) << "game isn't over";
    return false;
  }
  if (options_.resign_enabled || moves_.empty()) {
    return false;
  }

  // Find the move at which the game looked the bleakest from the perspective
  // of the winner.
  float bleakest_eval = moves_[0]->Q * result_;
  size_t bleakest_move = 0;
  for (size_t i = 1; i < moves_.size(); ++i) {
    float eval = moves_[i]->Q * result_;
    if (eval < bleakest_eval) {
      bleakest_eval = eval;
      bleakest_move = i;
    }
  }
  *move = static_cast<int>(bleakest_move);
  *q = bleakest_eval;
  return true;
}

}  // namespace minigo
