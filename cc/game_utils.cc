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

#include "cc/game_utils.h"

#include <iostream>
#include <cstring>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "cc/file/path.h"
#include "cc/file/utils.h"
#include "cc/logging.h"
#include "cc/platform/utils.h"
#include "cc/sgf.h"

namespace minigo {

std::string FormatWinStatsTable(
    const std::vector<std::pair<std::string, WinStats>>& stats) {
  size_t name_length = 4;
  for (const auto& name_stats : stats) {
    name_length = std::max(name_length, name_stats.first.size());
  }

  std::string result;

  auto append_header = [&](absl::string_view str) {
    absl::StrAppendFormat(&result, "%*s %s", name_length, "", str);
  };

  auto append_stats = [&](absl::string_view name, const WinStats& stats) {
    const auto& b = stats.black_wins;
    const auto& w = stats.white_wins;
    absl::StrAppendFormat(
        &result, "\n%-*s %7d %7d %7d %7d %7d %7d %7d %7d", name_length, name,
        b.total(), b.both_passed, b.opponent_resigned, b.move_limit_reached,
        w.total(), w.both_passed, w.opponent_resigned, w.move_limit_reached);
  };

  append_header(
      "  Black   Black   Black   Black   White   White   White   White\n");
  append_header(
      "  total   passes  resign  m.lmt.  total   passes  resign  m.lmt.");
  for (const auto& name_stats : stats) {
    append_stats(name_stats.first, name_stats.second);
  }

  return result;
}

std::string GetOutputName(size_t game_id) {
  return absl::StrCat(GetHostname(), "-", GetProcessId(), "-", game_id);
}

void WriteSgf(const std::string& output_dir, const std::string& output_name,
              const Game& game, bool write_comments) {
  MG_CHECK(file::RecursivelyCreateDir(output_dir));

  bool log_names = game.black_name() != game.white_name();

  std::vector<sgf::MoveWithComment> moves;
  moves.reserve(game.moves().size());

  for (size_t i = 0; i < game.moves().size(); ++i) {
    const auto* move = game.moves()[i].get();
    std::string comment;
    if (write_comments) {
      if (i == 0) {
        comment =
            absl::StrCat("Resign Threshold: ", game.options().resign_threshold,
                         "\n", move->comment);
      } else {
        if (log_names) {
          comment =
              absl::StrCat(move->color == Color::kBlack ? game.black_name()
                                                        : game.white_name(),
                           "\n", move->comment);
        } else {
          comment = move->comment;
        }
      }
    }
    moves.emplace_back(move->color, move->c, std::move(comment));
  }

  sgf::CreateSgfOptions options;
  options.komi = game.options().komi;
  options.result = game.result_string();
  options.black_name = game.black_name();
  options.white_name = game.white_name();
  options.game_comment = game.comment();
  auto sgf_str = sgf::CreateSgfString(moves, options);
  auto output_path = file::JoinPath(output_dir, output_name + ".sgf");
  MG_CHECK(file::WriteFile(output_path, sgf_str));
}

void LogEndGameInfo(const Game& game, absl::Duration game_time) {
  std::cout << game.result_string() << std::endl;
  std::cout << "Playing game: " << absl::ToDoubleSeconds(game_time)
            << std::endl;
  std::cout << "Played moves: " << game.moves().size() << std::endl;

  if (game.moves().empty()) {
    return;
  }

  int bleakest_move = 0;
  float q = 0.0;
  if (game.FindBleakestMove(&bleakest_move, &q)) {
    std::cout << "Bleakest eval: move=" << bleakest_move << " Q=" << q
              << std::endl;
  }

  // If resignation is disabled, check to see if the first time Q_perspective
  // crossed the resign_threshold the eventual winner of the game would have
  // resigned. Note that we only check for the first resignation: if the
  // winner would have incorrectly resigned AFTER the loser would have
  // resigned on an earlier move, this is not counted as a bad resignation for
  // the winner (since the game would have ended after the loser's initial
  // resignation).
  if (!game.options().resign_enabled) {
    for (size_t i = 0; i < game.moves().size(); ++i) {
      const auto* move = game.moves()[i].get();
      float Q_perspective = move->color == Color::kBlack ? move->Q : -move->Q;
      if (Q_perspective < game.options().resign_threshold) {
        if ((move->Q < 0) != (game.result() < 0)) {
          std::cout << "Bad resign: move=" << i << " Q=" << move->Q
                    << std::endl;
        }
        break;
      }
    }
  }
}

}  // namespace minigo
