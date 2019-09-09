// Copyright 2019 Google LLC
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

#ifndef CC_GAME_UTILS_H_
#define CC_GAME_UTILS_H_

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "cc/game.h"

namespace minigo {

// Stats about how one model won its games.
struct WinStats {
  struct ColorStats {
    int both_passed = 0;
    int opponent_resigned = 0;
    int move_limit_reached = 0;

    int total() const {
      return both_passed + opponent_resigned + move_limit_reached;
    }
  };

  void Update(const Game& game) {
    auto& stats = game.result() > 0 ? black_wins : white_wins;
    switch (game.game_over_reason()) {
      case Game::GameOverReason::kBothPassed:
        stats.both_passed += 1;
        break;
      case Game::GameOverReason::kOpponentResigned:
        stats.opponent_resigned += 1;
        break;
      case Game::GameOverReason::kMoveLimitReached:
        stats.move_limit_reached += 1;
        break;
    }
  }

  ColorStats black_wins;
  ColorStats white_wins;
};

// Returns a string-formatted table of win rates & types of multiple games
// between two players.
std::string FormatWinStatsTable(
    const std::vector<std::pair<std::string, WinStats>>& stats);

// Returns the name (specifically the basename stem) for an output game file
// (e.g. SGF, TF example, etc) based on the hostname, process ID and game ID.
std::string GetOutputName(size_t game_id);

// Writes an SGF of the given game.
void WriteSgf(const std::string& output_dir, const std::string& output_name,
              const Game& game, bool write_comments);

}  // namespace minigo

#endif  //  CC_GAME_UTILS_H_
