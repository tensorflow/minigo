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

#include <unistd.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "cc/check.h"
#include "cc/constants.h"
#include "cc/dual_net/tf_dual_net.h"
#include "cc/file/filesystem.h"
#include "cc/file/path.h"
#include "cc/gtp_player.h"
#include "cc/init.h"
#include "cc/mcts_player.h"
#include "cc/random.h"
#include "cc/sgf.h"
#include "cc/tf_utils.h"
#include "gflags/gflags.h"

DEFINE_uint64(seed, 0,
              "Random seed. Use default value of 0 to use a time-based seed. "
              "This seed is used to control the moves played, not whether a "
              "game has resignation disabled or is a holdout.");
DEFINE_double(resign_threshold, -0.95, "Resign threshold.");
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
DEFINE_string(model_two, "",
              "When running 'eval' mode, provide a path to a second minigo "
              "model, also serialized as a GraphDef proto.");
DEFINE_string(output_dir, "",
              "Output directory. If empty, no examples are written.");
DEFINE_string(holdout_dir, "",
              "Holdout directory. If empty, no examples are written.");
DEFINE_string(sgf_dir, "", "SGF directory. If empty, no SGF is written.");
DEFINE_double(holdout_pct, 0.05,
              "Fraction of games to hold out for validation.");
DEFINE_double(disable_resign_pct, 0.1,
              "Fraction of games to disable resignation for.");
DEFINE_int32(num_readouts, 100,
             "Number of readouts to make during tree search for each move.");
DEFINE_int32(batch_size, 8,
             "Number of readouts to run inference on in parallel.");
DEFINE_int32(
    ponder_limit, 0,
    "If non-zero and in GTP mode, the number times of times to perform tree "
    "search while waiting for the opponent to play.");
DEFINE_bool(
    courtesy_pass, false,
    "If true and in GTP mode, we will always pass if the opponent passes.");
DEFINE_double(
    seconds_per_move, 0,
    " If non-zero, the number of seconds to spend thinking about each "
    "move instead of using a fixed number of readouts.");
DEFINE_double(
    time_limit, 0,
    "If non-zero, the maximum amount of time to spend thinking in a game: we "
    "spend seconds_per_move thinking for each move for as many moves as "
    "possible before exponentially decaying the amount of time.");
DEFINE_double(decay_factor, 0.98,
              "If time_limit is non-zero, the decay factor used to shorten the "
              "amount of time spent thinking as the game progresses.");

DEFINE_string(mode, "", "Mode to run in: \"selfplay\", \"eval\" or \"gtp\"");

// Self play flags:
//   --inject_noise=true
//   --soft_pick=true
//   --random_symmetery=true
//
// Two player flags:
//   --inject_noise=false
//   --soft_pick=false
//   --random_symmetry=true

namespace minigo {
namespace {

std::string GetOutputName() {
  auto timestamp = absl::ToUnixSeconds(absl::Now());
  std::string output_name;
  char hostname[64];
  if (gethostname(hostname, sizeof(hostname)) != 0) {
    std::strncpy(hostname, "unknown", sizeof(hostname));
  }
  return absl::StrCat(timestamp, "-", hostname);
}

void WriteExample(const std::string& output_dir, const std::string& output_name,
                  const MctsPlayer& player) {
  MG_CHECK(file::RecursivelyCreateDir(output_dir));

  // Write the TensorFlow examples.
  std::vector<tensorflow::Example> examples;
  examples.reserve(player.history().size());
  for (const auto& h : player.history()) {
    examples.push_back(tf_utils::MakeTfExample(h.node->features, h.search_pi,
                                               player.result()));
  }

  auto output_path = file::JoinPath(output_dir, output_name + ".tfrecord.zz");
  tf_utils::WriteTfExamples(output_path, examples);
}

void WriteSgf(const std::string& output_dir, const std::string& output_name,
              const MctsPlayer& player_b, const MctsPlayer& player_w,
              bool write_comments) {
  MG_CHECK(file::RecursivelyCreateDir(output_dir));
  MG_CHECK(player_b.history().size() == player_w.history().size());

  bool log_names = player_b.name() != player_w.name();

  std::vector<sgf::MoveWithComment> moves;
  moves.reserve(player_b.history().size());

  for (size_t i = 0; i < player_b.history().size(); ++i) {
    const auto& h = i % 2 == 0 ? player_b.history()[i] : player_w.history()[i];
    const auto& color = h.node->position.to_play();
    std::string comment;
    if (write_comments) {
      if (i == 0) {
        comment = absl::StrCat(
            "Resign Threshold: ", player_b.options().resign_threshold, "\n",
            h.comment);
      } else {
        if (log_names) {
          comment = absl::StrCat(i % 2 == 0 ? player_b.name() : player_w.name(),
                                 "\n", h.comment);
        } else {
          comment = h.comment;
        }
      }
      moves.emplace_back(color, h.c, std::move(comment));
    } else {
      moves.emplace_back(color, h.c, "");
    }
  }

  std::string player_name(file::Basename(FLAGS_model));
  sgf::CreateSgfOptions options;
  options.komi = player_b.options().komi;
  options.result = player_b.result_string();
  options.black_name = player_b.name();
  options.white_name = player_w.name();
  auto sgf_str = sgf::CreateSgfString(moves, options);

  auto output_path = file::JoinPath(output_dir, output_name + ".sgf");
  TF_CHECK_OK(tf_utils::WriteFile(output_path, sgf_str));
}

void WriteSgf(const std::string& output_dir, const std::string& output_name,
              const MctsPlayer& player, bool write_comments) {
  WriteSgf(output_dir, output_name, player, player, write_comments);
}

void ParseMctsPlayerOptionsFromFlags(MctsPlayer::Options* options) {
  options->inject_noise = FLAGS_inject_noise;
  options->soft_pick = FLAGS_soft_pick;
  options->random_symmetry = FLAGS_random_symmetry;
  options->resign_threshold = FLAGS_resign_threshold;
  options->batch_size = FLAGS_batch_size;
  options->komi = FLAGS_komi;
  options->random_seed = FLAGS_seed;
  options->num_readouts = FLAGS_num_readouts;
  options->seconds_per_move = FLAGS_seconds_per_move;
  options->time_limit = FLAGS_time_limit;
  options->decay_factor = FLAGS_decay_factor;
}

void SelfPlay() {
  auto start_time = absl::Now();

  MctsPlayer::Options options;
  ParseMctsPlayerOptionsFromFlags(&options);
  Random rnd;
  if (rnd() < FLAGS_disable_resign_pct) {
    options.resign_threshold = -1;
  }
  auto player = absl::make_unique<MctsPlayer>(
      absl::make_unique<TfDualNet>(FLAGS_model), options);

  while (!player->game_over()) {
    auto move = player->SuggestMove();
    std::cerr << player->root()->position.ToPrettyString();
    std::cerr << player->root()->Describe() << std::endl;
    player->PlayMove(move);
  }
  std::cerr << player->result_string() << std::endl;
  std::cout << "Playing game: "
            << absl::ToDoubleSeconds(absl::Now() - start_time) << std::endl;

  std::string output_name = GetOutputName();
  std::string output_dir =
      rnd() < FLAGS_holdout_pct ? FLAGS_holdout_dir : FLAGS_output_dir;

  // Write example outputs.
  if (!output_dir.empty()) {
    WriteExample(output_dir, output_name, *player);
  }

  // Write SGF.
  if (!FLAGS_sgf_dir.empty()) {
    WriteSgf(file::JoinPath(FLAGS_sgf_dir, "clean"), output_name, *player,
             false);
    WriteSgf(file::JoinPath(FLAGS_sgf_dir, "full"), output_name, *player, true);
  }
}

void Eval() {
  MctsPlayer::Options options;
  ParseMctsPlayerOptionsFromFlags(&options);
  options.inject_noise = false;
  options.soft_pick = false;
  options.random_symmetry = true;
  auto black_name = std::string(file::Stem(FLAGS_model));
  options.name = black_name;
  auto player = absl::make_unique<MctsPlayer>(
      absl::make_unique<TfDualNet>(FLAGS_model), options);
  options.name = std::string(file::Stem(FLAGS_model_two));
  auto other_player = absl::make_unique<MctsPlayer>(
      absl::make_unique<TfDualNet>(FLAGS_model_two), options);

  while (!player->game_over()) {
    auto move = player->SuggestMove();
    std::cerr << player->root()->Describe() << "\n";
    player->PlayMove(move);
    other_player->PlayMove(move);
    std::cerr << player->root()->position.ToPrettyString();
    player.swap(other_player);
  }
  std::cerr << player->result_string() << "\n";

  // Swap 'player' back to its original value if needed.
  if (player->name() != black_name) {
    player.swap(other_player);
  }
  std::cerr << "Black was: " << player->name() << "\n";

  std::string output_name =
      absl::StrCat(GetOutputName(), "-", player->name(), other_player->name());

  // Write SGF.
  if (!FLAGS_sgf_dir.empty()) {
    WriteSgf(FLAGS_sgf_dir, output_name, *player, *other_player, true);
  }
}

void Gtp() {
  GtpPlayer::Options options;
  ParseMctsPlayerOptionsFromFlags(&options);
  options.name = absl::StrCat("minigo-", file::Basename(FLAGS_model));
  options.ponder_limit = FLAGS_ponder_limit;
  options.courtesy_pass = FLAGS_courtesy_pass;
  auto player = absl::make_unique<GtpPlayer>(
      absl::make_unique<TfDualNet>(FLAGS_model), options);
  player->Run();
}

}  // namespace

}  // namespace minigo

int main(int argc, char* argv[]) {
  minigo::Init(&argc, &argv);

  if (FLAGS_mode == "selfplay") {
    minigo::SelfPlay();
  } else if (FLAGS_mode == "eval") {
    minigo::Eval();
  } else if (FLAGS_mode == "gtp") {
    minigo::Gtp();
  } else {
    std::cerr << "Unrecognized mode \"" << FLAGS_mode << "\"\n";
    return 1;
  }

  return 0;
}
