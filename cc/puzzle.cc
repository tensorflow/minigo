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

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "cc/constants.h"
#include "cc/dual_net/batching_dual_net.h"
#include "cc/dual_net/factory.h"
#include "cc/file/path.h"
#include "cc/file/utils.h"
#include "cc/init.h"
#include "cc/logging.h"
#include "cc/mcts_player.h"
#include "cc/sgf.h"
#include "cc/zobrist.h"
#include "gflags/gflags.h"

DEFINE_uint64(seed, 1876509377, "Random seed for symmetries.");
DEFINE_int32(num_readouts, 100,
             "Number of readouts to make during tree search for each move.");
DEFINE_int32(virtual_losses, 8,
             "Number of virtual losses when running tree search.");
DEFINE_string(sgf_dir, "", "SGF directory containing puzzles.");
DEFINE_string(model, "",
              "Path to a minigo model. The format of the model depends on the "
              "inference engine.");
DEFINE_double(value_init_penalty, 0.0,
              "New children value initialize penaly.\n"
              "child's value = parent's value - value_init_penalty * color, "
              "clamped to [-1, 1].\n"
              "0 is init-to-parent [default], 2.0 is init-to-loss.\n"
              "This behaves similiarly to leela's FPU \"First Play Urgency\".");

namespace minigo {
namespace {

void Puzzle() {
  auto start_time = absl::Now();

  auto model_desc = minigo::ParseModelDescriptor(FLAGS_model);
  BatchingDualNetFactory batcher(NewDualNetFactory(model_desc.engine));

  Game::Options game_options;
  game_options.resign_enabled = false;

  MctsPlayer::Options player_options;
  player_options.inject_noise = false;
  player_options.soft_pick = false;
  player_options.random_symmetry = true;
  player_options.value_init_penalty = FLAGS_value_init_penalty;
  player_options.virtual_losses = FLAGS_virtual_losses;
  player_options.random_seed = FLAGS_seed;
  player_options.num_readouts = FLAGS_num_readouts;

  std::atomic<size_t> total_moves(0);
  std::atomic<size_t> correct_moves(0);

  std::vector<std::thread> threads;
  std::vector<std::string> basenames;
  MG_CHECK(file::ListDir(FLAGS_sgf_dir, &basenames));
  for (const auto& basename : basenames) {
    if (!absl::EndsWith(basename, ".sgf")) {
      continue;
    }
    threads.emplace_back([&]() {
      // Read the main line from the SGF.
      auto path = file::JoinPath(FLAGS_sgf_dir, basename);
      std::string contents;
      MG_CHECK(file::ReadFile(path, &contents));
      sgf::Ast ast;
      MG_CHECK(ast.Parse(contents));
      std::vector<std::unique_ptr<sgf::Node>> trees;
      MG_CHECK(GetTrees(ast, &trees));
      auto moves = trees[0]->ExtractMainLine();

      total_moves += moves.size();

      auto model = batcher.NewDualNet(model_desc.model);
      Game game(model->name(), model->name(), game_options);

      // Create player.
      auto player = absl::make_unique<MctsPlayer>(std::move(model), nullptr,
                                                  &game, player_options);
      batcher.StartGame(player->network(), player->network());

      // Play through each game. For each position in the game, compare the
      // model's suggested move to the actual move played in the game.
      for (size_t move_to_predict = 0; move_to_predict < moves.size();
           ++move_to_predict) {
        MG_LOG(INFO) << move_to_predict << "/" << moves.size();

        // Reset the game and play up to the position to be tested.
        player->NewGame();
        for (size_t i = 0; i < move_to_predict; ++i) {
          player->PlayMove(moves[i].c);
        }

        // Check if we predict the move that was played.
        auto expected_move = moves[move_to_predict].c;
        auto actual_move = player->SuggestMove();
        if (actual_move == expected_move) {
          ++correct_moves;
        }
      }
      batcher.EndGame(player->network(), player->network());
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  MG_LOG(INFO) << absl::StreamFormat(
      "Solved %d of %d puzzles (%3.1f%%), total time %f sec.", correct_moves,
      total_moves, correct_moves * 100.0f / total_moves,
      absl::ToDoubleSeconds(absl::Now() - start_time));
}

}  // namespace
}  // namespace minigo

int main(int argc, char* argv[]) {
  minigo::Init(&argc, &argv);
  minigo::zobrist::Init(FLAGS_seed * 614944751);
  minigo::Puzzle();
  return 0;
}
