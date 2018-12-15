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

#include "absl/strings/str_cat.h"
#include "cc/constants.h"
#include "cc/dual_net/factory.h"
#include "cc/file/path.h"
#include "cc/gtp_player.h"
#include "cc/init.h"
#include "cc/zobrist.h"
#include "gflags/gflags.h"

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
              "inferece engine. For engine=tf, the model should be a GraphDef "
              "proto. For engine=lite, the model should be .tflite "
              "flatbuffer.");

namespace minigo {
namespace {

void Gtp() {
  GtpPlayer::Options options;
  options.game_options.resign_threshold = FLAGS_resign_threshold;
  options.name = absl::StrCat("minigo-", file::Basename(FLAGS_model));
  options.ponder_limit = FLAGS_ponder_limit;
  options.courtesy_pass = FLAGS_courtesy_pass;
  options.inject_noise = false;
  options.soft_pick = false;
  options.random_symmetry = true;
  options.value_init_penalty = FLAGS_value_init_penalty;
  options.batch_size = FLAGS_virtual_losses;
  options.num_readouts = FLAGS_num_readouts;
  options.seconds_per_move = FLAGS_seconds_per_move;
  options.time_limit = FLAGS_time_limit;
  options.decay_factor = FLAGS_decay_factor;

  auto model_factory = NewDualNetFactory();
  GtpPlayer player(model_factory->NewDualNet(FLAGS_model), options);
  model_factory->StartGame(player.network(), player.network());
  player.Run();
  model_factory->EndGame(player.network(), player.network());
}

}  // namespace
}  // namespace minigo

int main(int argc, char* argv[]) {
  minigo::Init(&argc, &argv);
  minigo::zobrist::Init(0);
  minigo::Gtp();
  return 0;
}
