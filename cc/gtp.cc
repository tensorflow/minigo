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

#include <iostream>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "cc/constants.h"
#include "cc/dual_net/factory.h"
#include "cc/file/path.h"
#include "cc/gtp_player.h"
#include "cc/init.h"
#include "cc/minigui_player.h"
#include "cc/zobrist.h"
#include "gflags/gflags.h"

// GTP flags.
DEFINE_bool(minigui, false, "Enable Minigui GTP extensions");

// Game options flags.
DEFINE_int32(
    ponder_limit, 0,
    "If non-zero and in GTP mode, the number times of times to perform tree "
    "search while waiting for the opponent to play.");
DEFINE_bool(courtesy_pass, false,
            "If true, always pass if the opponent passes.");
DEFINE_double(resign_threshold, -0.999, "Resign threshold.");

// Tree search flags.
DEFINE_int32(num_readouts, 100,
             "Number of readouts to make during tree search for each move.");
DEFINE_int32(virtual_losses, 8,
             "Number of virtual losses when running tree search.");
DEFINE_double(value_init_penalty, 0.0,
              "New children value initialize penaly.\n"
              "child's value = parent's value - value_init_penalty * color, "
              "clamped to [-1, 1].\n"
              "0 is init-to-parent [default], 2.0 is init-to-loss.\n"
              "This behaves similiarly to leela's FPU \"First Play Urgency\".");
DEFINE_bool(tree_reuse, true,
            "Enable reuse of shared subtrees between consecutive moves.");

// Time control flags.
DEFINE_double(seconds_per_move, 0,
              "If non-zero, the number of seconds to spend thinking about each "
              "move instead of using a fixed number of readouts.");
DEFINE_double(
    time_limit, 0,
    "If non-zero, the maximum amount of time to spend thinking in a game: we "
    "spend seconds_per_move thinking for each move for as many moves as "
    "possible before exponentially decaying the amount of time.");
DEFINE_double(decay_factor, 0.98,
              "If time_limit is non-zero, the decay factor used to shorten the "
              "amount of time spent thinking as the game progresses.");

// Inference flags.
DEFINE_string(model, "",
              "Path to a minigo model. The format of the model depends on the "
              "inference engine.");
DEFINE_int32(cache_size_mb, 0,
             "Size of the inference cache in MB. A value of 0 (the default) "
             "disables the cache.");

namespace minigo {
namespace {

void Gtp() {
  Game::Options game_options;
  game_options.resign_threshold = FLAGS_resign_threshold;

  MctsPlayer::Options player_options;
  player_options.inject_noise = false;
  player_options.soft_pick = false;
  player_options.random_symmetry = true;
  player_options.value_init_penalty = FLAGS_value_init_penalty;
  player_options.tree_reuse = FLAGS_tree_reuse;
  player_options.virtual_losses = FLAGS_virtual_losses;
  player_options.num_readouts = FLAGS_num_readouts;
  player_options.seconds_per_move = FLAGS_seconds_per_move;
  player_options.time_limit = FLAGS_time_limit;
  player_options.decay_factor = FLAGS_decay_factor;

  GtpClient::Options client_options;
  client_options.ponder_limit = FLAGS_ponder_limit;
  client_options.courtesy_pass = FLAGS_courtesy_pass;

  MG_LOG(INFO) << game_options << " " << player_options;

  auto model_desc = minigo::ParseModelDescriptor(FLAGS_model);
  auto model_factory = NewDualNetFactory(model_desc.engine);
  auto model = model_factory->NewDualNet(model_desc.model);
  std::unique_ptr<BasicInferenceCache> cache;
  if (FLAGS_cache_size_mb > 0) {
    auto capacity = BasicInferenceCache::CalculateCapacity(FLAGS_cache_size_mb);
    std::cerr << "Will cache up to " << capacity
              << " inferences, using roughly " << FLAGS_cache_size_mb
              << "MB.\n";
    cache = absl::make_unique<BasicInferenceCache>(capacity);
  }

  Game game(model->name(), model->name(), game_options);
  auto player = absl::make_unique<MctsPlayer>(
      std::move(model), std::move(cache), &game, player_options);

  std::unique_ptr<GtpClient> client;
  if (FLAGS_minigui) {
    client =
        absl::make_unique<MiniguiGtpClient>(std::move(player), client_options);
  } else {
    client = absl::make_unique<GtpClient>(std::move(player), client_options);
  }
  client->Run();
}

}  // namespace
}  // namespace minigo

int main(int argc, char* argv[]) {
  minigo::Init(&argc, &argv);
  minigo::zobrist::Init(0);
  minigo::Gtp();
  return 0;
}
