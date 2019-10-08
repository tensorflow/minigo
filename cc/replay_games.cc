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

#include <stdio.h>

#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "cc/color.h"
#include "cc/file/path.h"
#include "cc/file/utils.h"
#include "cc/init.h"
#include "cc/logging.h"
#include "cc/position.h"
#include "cc/sgf.h"
#include "cc/thread.h"
#include "cc/thread_safe_queue.h"
#include "gflags/gflags.h"

DEFINE_string(sgf_dir, "", "Directory to load SGF games from.");
DEFINE_int32(num_threads, 8, "Number of worker threads.");

namespace minigo {
namespace {

enum class GameOverReason {
  kMoveLimit,
  kPassPass,
  kWholeBoardPassAlive,
};

struct GameInfo {
  GameInfo(GameOverReason game_over_reason, int whole_board_pass_alive_move,
           int game_length)
      : game_over_reason(game_over_reason),
        whole_board_pass_alive_move(whole_board_pass_alive_move),
        game_length(game_length) {}

  GameOverReason game_over_reason;
  int whole_board_pass_alive_move;
  int game_length;
};

GameInfo ProcessSgf(const std::string& path) {
  std::string contents;
  MG_CHECK(file::ReadFile(path, &contents));

  sgf::Ast ast;
  MG_CHECK(ast.Parse(contents));

  std::vector<std::unique_ptr<sgf::Node>> trees;
  MG_CHECK(sgf::GetTrees(ast, &trees));

  Position position(Color::kBlack);

  Coord prev_move = Coord::kInvalid;
  const auto& moves = trees[0]->ExtractMainLine();
  auto num_moves = static_cast<int>(moves.size());
  for (int i = 0; i < num_moves; ++i) {
    const auto& move = moves[i];
    MG_CHECK(position.legal_move(move.c));
    position.PlayMove(move.c);
    if (move.c == Coord::kPass && prev_move == Coord::kPass) {
      return GameInfo(GameOverReason::kPassPass, 0, num_moves);
    }
    if (position.CalculateWholeBoardPassAlive()) {
      return GameInfo(GameOverReason::kWholeBoardPassAlive, i, num_moves);
    }
    prev_move = move.c;
  }

  return GameInfo(GameOverReason::kMoveLimit, 0, num_moves);
}

void Run() {
  std::vector<std::string> basenames;
  MG_CHECK(file::ListDir(FLAGS_sgf_dir, &basenames));

  ThreadSafeQueue<std::string> work_queue;
  for (const auto& basename : basenames) {
    work_queue.Push(basename);
  }

  ThreadSafeQueue<GameInfo> game_info_queue;

  std::vector<std::unique_ptr<LambdaThread>> threads;
  for (int i = 0; i < FLAGS_num_threads; ++i) {
    threads.push_back(absl::make_unique<LambdaThread>([&]() {
      std::string basename;
      while (work_queue.TryPop(&basename)) {
        auto path = file::JoinPath(FLAGS_sgf_dir, basename);
        game_info_queue.Push(ProcessSgf(path));
      }
    }));
    threads.back()->Start();
  }

  int num_pass_pass_games = 0;
  int num_move_limit_games = 0;
  int num_whole_board_pass_alive_games = 0;
  int game_length_sum = 0;
  int whole_board_pass_alive_sum = 0;
  int min_whole_board_pass_alive = kN * kN * 2;
  for (size_t i = 0; i < basenames.size(); ++i) {
    auto info = game_info_queue.Pop();
    switch (info.game_over_reason) {
      case GameOverReason::kMoveLimit:
        num_move_limit_games += 1;
        break;

      case GameOverReason::kPassPass:
        num_pass_pass_games += 1;
        break;

      case GameOverReason::kWholeBoardPassAlive:
        num_whole_board_pass_alive_games += 1;
        game_length_sum += info.game_length;
        whole_board_pass_alive_sum += info.whole_board_pass_alive_move;
        if (info.whole_board_pass_alive_move < min_whole_board_pass_alive) {
          min_whole_board_pass_alive = info.whole_board_pass_alive_move;
        }
        break;
    }
  }

  for (auto& t : threads) {
    t->Join();
  }

  MG_LOG(INFO) << "total games: " << basenames.size();
  MG_LOG(INFO) << "num move limit games: " << num_move_limit_games;
  MG_LOG(INFO) << "num whole-board pass-alive games: "
               << num_whole_board_pass_alive_games;
  MG_LOG(INFO) << "num pass-pass games: " << num_pass_pass_games;
  MG_LOG(INFO) << "mean whole-board pass-alive move number: "
               << float(whole_board_pass_alive_sum) /
                      float(num_whole_board_pass_alive_games);
  MG_LOG(INFO) << "mean length of whole-board pass-alive games: "
               << float(game_length_sum) /
                      float(num_whole_board_pass_alive_games);
  MG_LOG(INFO) << "min whole-board pass-alive move number: "
               << min_whole_board_pass_alive;
}

}  // namespace
}  // namespace minigo

int main(int argc, char* argv[]) {
  minigo::Init(&argc, &argv);
  minigo::Run();
  return 0;
}
