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

namespace minigo {

std::string Game::FormatScore(float score) {
  return absl::StrFormat("%c+%.1f", score > 0 ? 'B' : 'W', std::abs(score));
}

Game::Game(std::string black_name, std::string white_name,
           const Game::Options& options)
    : options_(options),
      black_name_(std::move(black_name)),
      white_name_(std::move(white_name)) {}

void Game::AddComment(const std::string& comment) {
  if (comment_.empty()) {
    comment_ = comment;
  } else {
    absl::StrAppend(&comment_, "\n", comment);
  }
}

void Game::AddMove(Color color, Coord c, const Position::Stones& stones,
                   std::string comment, float Q,
                   const std::array<float, kNumMoves>& search_pi,
                   std::vector<std::string> models) {
  moves_.push_back(absl::make_unique<Move>());
  auto* move = moves_.back().get();
  move->color = color;
  move->c = c;
  move->Q = Q;
  move->comment = std::move(comment);
  move->models = std::move(models);
  move->search_pi = search_pi;
  move->stones = stones;
}

void Game::UndoMove() {
  MG_CHECK(!moves_.empty());
  moves_.pop_back();
}

void Game::SetGameOverBecauseOfPasses(float score) {
  MG_CHECK(!game_over_);
  game_over_ = true;
  result_ = score < 0 ? -1 : score > 0 ? 1 : 0;
  result_string_ = FormatScore(score);
}

void Game::SetGameOverBecauseOfResign(Color winner) {
  MG_CHECK(!game_over_);
  game_over_ = true;
  if (winner == Color::kBlack) {
    result_ = 1;
    result_string_ = "B+R";
  } else {
    result_ = -1;
    result_string_ = "W+R";
  }
}

void Game::GetStoneHistory(
    int move, int num_moves,
    std::vector<const Position::Stones*>* history) const {
  history->clear();
  history->reserve(num_moves);

  MG_CHECK(move >= 0);
  MG_CHECK(move < static_cast<int>(moves_.size()));
  for (int i = 0; i < num_moves && move - i >= 0; ++i) {
    history->push_back(&moves_[move - i]->stones);
  }
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
