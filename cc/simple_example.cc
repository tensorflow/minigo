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

#include <iostream>

#include "cc/dual_net/factory.h"
#include "cc/game.h"
#include "cc/init.h"
#include "cc/logging.h"
#include "cc/mcts_player.h"
#include "cc/platform/utils.h"
#include "cc/random.h"
#include "cc/zobrist.h"
#include "gflags/gflags.h"

// Inference flags.
DEFINE_string(model, "",
              "Path to a minigo model. The format of the model depends on the "
              "inference engine.");
DEFINE_int32(num_readouts, 100,
             "Number of readouts to make during tree search for each move.");

namespace minigo {
namespace {

// Demonstrates how to perform basic self-play, while eliding the additional
// complexity required by the training pipeline.
void SimpleExample() {
  // Determine whether ANSI color codes are supported (used when printing
  // the board state after each move).
  const bool use_ansi_colors = FdSupportsAnsiColors(fileno(stderr));

  // Load the model specified by the command line arguments.
  auto descriptor = ParseModelDescriptor(FLAGS_model);
  auto model_factory = NewModelFactory(descriptor.engine, true, 0);
  auto model = model_factory->NewModel(descriptor.model);

  // Create a game object that tracks the move history & final score.
  Game::Options game_options;
  Game game("black", "white", game_options);

  // Create the player.
  MctsPlayer::Options player_options;
  player_options.inject_noise = false;
  player_options.soft_pick = false;
  player_options.num_readouts = FLAGS_num_readouts;
  MctsPlayer player(std::move(model), nullptr, &game, player_options);

  // Play the game.
  while (!game.game_over() && !player.root()->at_move_limit()) {
    auto move = player.SuggestMove(player_options.num_readouts);

    const auto& position = player.root()->position;
    std::cout << player.root()->position.ToPrettyString(use_ansi_colors)
              << "\n";
    std::cout << "Move: " << position.n()
              << " Captures X: " << position.num_captures()[0]
              << " O: " << position.num_captures()[1] << "\n";
    std::cout << player.root()->Describe() << "\n";

    MG_CHECK(player.PlayMove(move));
  }

  std::cout << game.result_string() << std::endl;
}

}  // namespace
}  // namespace minigo

int main(int argc, char* argv[]) {
  minigo::Init(&argc, &argv);
  minigo::zobrist::Init(0);
  minigo::SimpleExample();
  return 0;
}
