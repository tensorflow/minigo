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

#include <memory>
#include <vector>

#include "cc/constants.h"
#include "cc/dual_net.h"
#include "cc/mcts_player.h"
#include "gflags/gflags.h"

DEFINE_uint64(seed, 0,
              "Random seed. Use default value of 0 to use a time-based seed.");
DEFINE_double(resign_threshold, -0.9, "Resign threshold.");
DEFINE_double(komi, minigo::kDefaultKomi, "Komi.");
DEFINE_bool(inject_noise, true,
            "If true, inject noise into the root position at the start of "
            "each tree search.");
DEFINE_bool(soft_pick, true,
            "If true, choose moves early in the game with a probability "
            "proportional to the number of times visited during tree search. "
            "If false, always play the best move.");
DEFINE_bool(random_symmetry, true,
            "If true, randomly flip & rotate the board features before running "
            "the model and apply the inverse transform to the results.");
DEFINE_string(model, "",
              "Path to a minigo model serialized as a GraphDef proto.");
DEFINE_int32(num_readouts, 100,
             "Number of readouts to make during tree search for each move.");
DEFINE_int32(batch_size, 8,
             "Number of readouts to run inference on in parallel.");

// Self play flags:
//   --inject_noise=true
//   --soft_pick=true
//   --random_symmetery=true
//
// Two player flags:
//   --inject_noise=false
//   --soft_pick=false
//   --random_symmetry=true

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  minigo::DualNet dual_net;
  dual_net.Initialize(FLAGS_model);

  minigo::MctsPlayer::Options options;
  options.random_seed = FLAGS_seed;
  options.resign_threshold = FLAGS_resign_threshold;
  options.komi = FLAGS_komi;
  options.inject_noise = FLAGS_inject_noise;
  options.soft_pick = FLAGS_soft_pick;
  options.random_symmetry = FLAGS_random_symmetry;
  options.batch_size = FLAGS_batch_size;

  minigo::MctsPlayer player(&dual_net, options);
  player.SelfPlay(FLAGS_num_readouts);

  return 0;
}
