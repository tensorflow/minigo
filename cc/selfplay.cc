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

#include <stdio.h>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "cc/constants.h"
#include "cc/dual_net/batching_dual_net.h"
#include "cc/dual_net/factory.h"
#include "cc/dual_net/reloading_dual_net.h"
#include "cc/file/path.h"
#include "cc/file/utils.h"
#include "cc/game.h"
#include "cc/game_utils.h"
#include "cc/init.h"
#include "cc/logging.h"
#include "cc/mcts_player.h"
#include "cc/platform/utils.h"
#include "cc/random.h"
#include "cc/tf_utils.h"
#include "cc/zobrist.h"
#include "gflags/gflags.h"

// Game options flags.
DEFINE_double(resign_threshold, -0.999, "Resign threshold.");
DEFINE_double(disable_resign_pct, 0.1,
              "Fraction of games to disable resignation for.");
DEFINE_uint64(seed, 0,
              "Random seed. Use default value of 0 to use a time-based seed. "
              "This seed is used to control the moves played, not whether a "
              "game has resignation disabled or is a holdout.");
DEFINE_double(holdout_pct, 0.03,
              "Fraction of games to hold out for validation.");

// Tree search flags.
DEFINE_int32(num_readouts, 100,
             "Number of readouts to make during tree search for each move.");
DEFINE_int32(virtual_losses, 8,
             "Number of virtual losses when running tree search.");
DEFINE_bool(inject_noise, true,
            "If true, inject noise into the root position at the start of "
            "each tree search.");
DEFINE_double(noise_mix, 0.25,
              "If inject_noise is true, the amount of noise to mix into the "
              "root.");
DEFINE_bool(soft_pick, true,
            "If true, choose moves early in the game with a probability "
            "proportional to the number of times visited during tree search. "
            "If false, always play the best move.");
DEFINE_bool(random_symmetry, true,
            "If true, randomly flip & rotate the board features before running "
            "the model and apply the inverse transform to the results.");
DEFINE_double(value_init_penalty, 2.0,
              "New children value initialization penalty.\n"
              "Child value = parent's value - penalty * color, clamped to "
              "[-1, 1].  Penalty should be in [0.0, 2.0].\n"
              "0 is init-to-parent, 2.0 is init-to-loss [default].\n"
              "This behaves similiarly to Leela's FPU \"First Play Urgency\".");
DEFINE_double(policy_softmax_temp, 0.98,
              "For soft-picked moves, the probabilities are exponentiated by "
              "policy_softmax_temp to encourage diversity in early play.\n");

DEFINE_string(flags_path, "",
              "Optional path to load flags from. Flags specified in this file "
              "take priority over command line flags. When running selfplay "
              "with run_forever=true, the flag file is reloaded periodically. "
              "Note that flags_path is different from gflags flagfile, which "
              "is only parsed once on startup.");

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
DEFINE_bool(run_forever, false,
            "When running 'selfplay' mode, whether to run forever.");

// Inference flags.
DEFINE_string(model, "",
              "Path to a minigo model. The format of the model depends on the "
              "inference engine. For engine=tf, the model should be a GraphDef "
              "proto. For engine=lite, the model should be .tflite "
              "flatbuffer.");
DEFINE_int32(parallel_games, 32, "Number of games to play in parallel.");

// Output flags.
DEFINE_string(output_dir, "",
              "Output directory. If empty, no examples are written.");
DEFINE_string(holdout_dir, "",
              "Holdout directory. If empty, no examples are written.");
DEFINE_string(output_bigtable, "",
              "Output Bigtable specification, of the form: "
              "project,instance,table. "
              "If empty, no examples are written to Bigtable.");
DEFINE_string(sgf_dir, "",
              "SGF directory for selfplay and puzzles. If empty in selfplay "
              "mode, no SGF is written.");
DEFINE_string(bigtable_tag, "", "Used in Bigtable metadata");

namespace minigo {
namespace {

std::string GetOutputDir(absl::Time now, const std::string& root_dir) {
  auto sub_dirs = absl::FormatTime("%Y-%m-%d-%H", now, absl::UTCTimeZone());
  return file::JoinPath(root_dir, sub_dirs);
}

void ParseMctsPlayerOptionsFromFlags(MctsPlayer::Options* options) {
  options->noise_mix = FLAGS_noise_mix;
  options->inject_noise = FLAGS_inject_noise;
  options->soft_pick = FLAGS_soft_pick;
  options->random_symmetry = FLAGS_random_symmetry;
  options->value_init_penalty = FLAGS_value_init_penalty;
  options->policy_softmax_temp = FLAGS_policy_softmax_temp;
  options->game_options.resign_threshold = -std::abs(FLAGS_resign_threshold);
  options->virtual_losses = FLAGS_virtual_losses;
  options->random_seed = FLAGS_seed;
  options->num_readouts = FLAGS_num_readouts;
  options->seconds_per_move = FLAGS_seconds_per_move;
  options->time_limit = FLAGS_time_limit;
  options->decay_factor = FLAGS_decay_factor;
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

class SelfPlayer {
 public:
  void Run() {
    auto start_time = absl::Now();
    {
      absl::MutexLock lock(&mutex_);
      auto model_factory = NewDualNetFactory();
      // If the model path contains a pattern, wrap the implementation factory
      // in a ReloadingDualNetFactory to automatically reload the latest model
      // that matches the pattern.
      if (FLAGS_model.find("%d") != std::string::npos) {
        model_factory = absl::make_unique<ReloadingDualNetFactory>(
            std::move(model_factory), absl::Seconds(3));
      }
      // Note: it's more efficient to perform the reload wrapping before the
      // batch wrapping because this way, we only need to reload the single
      // implementation DualNet when a new model is found. If we performed batch
      // wrapping before reload wrapping, the reload code would need to update
      // all the BatchingDualNet wrappers.
      batcher_ =
          absl::make_unique<BatchingDualNetFactory>(std::move(model_factory));
    }
    for (int i = 0; i < FLAGS_parallel_games; ++i) {
      threads_.emplace_back(std::bind(&SelfPlayer::ThreadRun, this, i));
    }
    for (auto& t : threads_) {
      t.join();
    }
    MG_LOG(INFO) << "Played " << FLAGS_parallel_games << " games, total time "
                 << absl::ToDoubleSeconds(absl::Now() - start_time) << " sec.";
  }

 private:
  // Struct that holds the options for each thread.
  // Initialized with the SelfPlayer's mutex held. This allows us to safely
  // update the command line arguments from a flag file without causing any
  // race conditions.
  struct ThreadOptions {
    void Init(int thread_id, Random* rnd) {
      ParseMctsPlayerOptionsFromFlags(&player_options);
      player_options.verbose = thread_id == 0;
      // If an random seed was explicitly specified, make sure we use a
      // different seed for each thread.
      if (player_options.random_seed != 0) {
        player_options.random_seed += 1299283 * thread_id;
      }
      player_options.game_options.resign_enabled =
          (*rnd)() >= FLAGS_disable_resign_pct;

      run_forever = FLAGS_run_forever;
      holdout_pct = FLAGS_holdout_pct;
      output_dir = FLAGS_output_dir;
      holdout_dir = FLAGS_holdout_dir;
      sgf_dir = FLAGS_sgf_dir;
    }

    MctsPlayer::Options player_options;
    bool run_forever;
    float holdout_pct;
    std::string output_dir;
    std::string holdout_dir;
    std::string sgf_dir;
  };

  void ThreadRun(int thread_id) {
    // Only print the board using ANSI colors if stderr is sent to the
    // terminal.
    const bool use_ansi_colors = FdSupportsAnsiColors(fileno(stderr));

    ThreadOptions thread_options;
    std::vector<std::string> bigtable_spec =
        absl::StrSplit(FLAGS_output_bigtable, ',');
    bool use_bigtable = bigtable_spec.size() == 3;
    if (!FLAGS_output_bigtable.empty() && !use_bigtable) {
      MG_LOG(FATAL)
          << "Bigtable output must be of the form: project,instance,table";
      return;
    }

    do {
      std::unique_ptr<MctsPlayer> player;
      std::unique_ptr<Game> game;

      {
        absl::MutexLock lock(&mutex_);
        auto old_model = FLAGS_model;
        MaybeReloadFlags();
        MG_CHECK(old_model == FLAGS_model)
            << "Manually changing the model during selfplay is not supported.";
        thread_options.Init(thread_id, &rnd_);
        player = absl::make_unique<MctsPlayer>(
            batcher_->NewDualNet(FLAGS_model), nullptr,
            thread_options.player_options);
        game =
            absl::make_unique<Game>(player->name(), player->name(),
                                    thread_options.player_options.game_options);
      }

      // Play the game.
      auto start_time = absl::Now();
      {
        absl::MutexLock lock(&mutex_);
        batcher_->StartGame(player->network(), player->network());
      }
      while (!game->game_over() && !player->root()->at_move_limit()) {
        auto move = player->SuggestMove();
        if (player->options().verbose) {
          const auto& position = player->root()->position;
          MG_LOG(INFO) << player->root()->position.ToPrettyString(
              use_ansi_colors);
          MG_LOG(INFO) << "Move: " << position.n()
                       << " Captures X: " << position.num_captures()[0]
                       << " O: " << position.num_captures()[1];
          MG_LOG(INFO) << player->root()->Describe();
        }
        MG_CHECK(player->PlayMove(move, game.get()));
      }
      {
        absl::MutexLock lock(&mutex_);
        batcher_->EndGame(player->network(), player->network());
      }

      {
        // Log the end game info with the shared mutex held to prevent the
        // outputs from multiple threads being interleaved.
        absl::MutexLock lock(&mutex_);
        LogEndGameInfo(*game, absl::Now() - start_time);
      }

      // Write the outputs.
      auto now = absl::Now();
      auto output_name = GetOutputName(now, thread_id);

      bool is_holdout;
      {
        absl::MutexLock lock(&mutex_);
        is_holdout = rnd_() < thread_options.holdout_pct;
      }
      auto example_dir =
          is_holdout ? thread_options.holdout_dir : thread_options.output_dir;
      if (!example_dir.empty()) {
        tf_utils::WriteGameExamples(GetOutputDir(now, example_dir), output_name,
                                    *game);
      }
      if (use_bigtable) {
        const auto& gcp_project_name = bigtable_spec[0];
        const auto& instance_name = bigtable_spec[1];
        const auto& table_name = bigtable_spec[2];
        tf_utils::WriteGameExamples(gcp_project_name, instance_name, table_name,
                                    *game);
      }

      game->AddComment(
          absl::StrCat("Inferences: ", player->GetModelsUsedForInference()));
      if (!thread_options.sgf_dir.empty()) {
        WriteSgf(
            GetOutputDir(now, file::JoinPath(thread_options.sgf_dir, "clean")),
            output_name, *game, false);
        WriteSgf(
            GetOutputDir(now, file::JoinPath(thread_options.sgf_dir, "full")),
            output_name, *game, true);
      }
    } while (thread_options.run_forever);

    MG_LOG(INFO) << "Thread " << thread_id << " stopping";
  }

  void MaybeReloadFlags() EXCLUSIVE_LOCKS_REQUIRED(&mutex_) {
    if (FLAGS_flags_path.empty()) {
      return;
    }
    uint64_t new_flags_timestamp;
    MG_CHECK(file::GetModTime(FLAGS_flags_path, &new_flags_timestamp));
    bool skip = new_flags_timestamp == flags_timestamp_;
    MG_LOG(INFO) << "flagfile:" << FLAGS_flags_path
                 << " old_ts:" << absl::FromUnixMicros(flags_timestamp_)
                 << " new_ts:" << absl::FromUnixMicros(new_flags_timestamp)
                 << (skip ? " skipping" : "");
    if (skip) {
      return;
    }

    flags_timestamp_ = new_flags_timestamp;
    std::string contents;
    MG_CHECK(file::ReadFile(FLAGS_flags_path, &contents));

    std::vector<std::string> lines =
        absl::StrSplit(contents, '\n', absl::SkipEmpty());
    MG_LOG(INFO) << " loaded flags:" << absl::StrJoin(lines, " ");

    for (absl::string_view line : lines) {
      std::pair<absl::string_view, absl::string_view> line_comment =
          absl::StrSplit(line, absl::MaxSplits('#', 1));
      line = absl::StripAsciiWhitespace(line_comment.first);
      if (line.empty()) {
        continue;
      }
      MG_CHECK(line.length() > 2 && line[0] == '-' && line[1] == '-') << line;
      std::pair<std::string, std::string> flag_value =
          absl::StrSplit(line, absl::MaxSplits('=', 1));
      flag_value.first = flag_value.first.substr(2);
      MG_LOG(INFO) << "Setting command line flag: --" << flag_value.first << "="
                   << flag_value.second;
      gflags::SetCommandLineOption(flag_value.first.c_str(),
                                   flag_value.second.c_str());
    }
  }

  absl::Mutex mutex_;
  std::unique_ptr<BatchingDualNetFactory> batcher_ GUARDED_BY(&mutex_);
  Random rnd_ GUARDED_BY(&mutex_);
  std::vector<std::thread> threads_;
  uint64_t flags_timestamp_ = 0;
};

}  // namespace
}  // namespace minigo

int main(int argc, char* argv[]) {
  minigo::Init(&argc, &argv);
  minigo::zobrist::Init(FLAGS_seed * 614944751);
  minigo::SelfPlayer player;
  player.Run();
  return 0;
}
