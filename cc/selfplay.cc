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
#include "cc/dual_net/factory.h"
#include "cc/file/path.h"
#include "cc/file/utils.h"
#include "cc/game.h"
#include "cc/game_utils.h"
#include "cc/init.h"
#include "cc/logging.h"
#include "cc/mcts_player.h"
#include "cc/model/batching_model.h"
#include "cc/model/inference_cache.h"
#include "cc/model/reloading_model.h"
#include "cc/platform/utils.h"
#include "cc/random.h"
#include "cc/tf_utils.h"
#include "cc/zobrist.h"
#include "gflags/gflags.h"
#include "wtf/macros.h"

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

DEFINE_double(fastplay_frequency, 0.0,
              "The fraction of moves that should use a lower number of "
              "playouts, aka 'playout cap oscillation'.\nIf this is set, "
              "'fastplay_readouts' should also be set.");

DEFINE_int32(fastplay_readouts, 20,
             "The number of readouts to perform on a 'low readout' move, "
             "aka 'playout cap oscillation'.\nIf this is set, "
             "'fastplay_frequency' should be nonzero.");

DEFINE_bool(target_pruning, false,
            "If true, subtract visits from all moves that weren't the best move "
            "until the uncertainty level compensates.");

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
            "When running 'selfplay' mode, whether to run forever. "
            "Only one of run_forever and num_games must be set.");

// Inference flags.
DEFINE_string(model, "",
              "Path to a minigo model. The format of the model depends on the "
              "inference engine. For engine=tf, the model should be a GraphDef "
              "proto. For engine=lite, the model should be .tflite "
              "flatbuffer.");
DEFINE_int32(parallel_games, 32, "Number of games to play in parallel.");
DEFINE_int32(num_games, 0,
             "Total number of games to play. Defaults to parallel_games. "
             "Only one of num_games and run_forever must be set.");
DEFINE_int32(cache_size_mb, 0, "Size of the inference cache in MB.");
DEFINE_int32(cache_shards, 8,
             "Number of ways to shard the inference cache. The cache uses "
             "is locked on a per-shard basis, so more shards means less "
             "contention but each shard is smaller. The number of shards "
             "is clamped such that it's always <= parallel_games.");

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
DEFINE_string(wtf_trace, "/tmp/minigo.wtf-trace",
              "Output path for WTF traces.");

namespace minigo {
namespace {

std::string GetOutputDir(absl::Time now, const std::string& root_dir) {
  auto sub_dirs = absl::FormatTime("%Y-%m-%d-%H", now, absl::UTCTimeZone());
  return file::JoinPath(root_dir, sub_dirs);
}

void ParseOptionsFromFlags(Game::Options* game_options,
                           MctsPlayer::Options* player_options) {
  game_options->resign_threshold = -std::abs(FLAGS_resign_threshold);
  player_options->noise_mix = FLAGS_noise_mix;
  player_options->inject_noise = FLAGS_inject_noise;
  player_options->soft_pick = FLAGS_soft_pick;
  player_options->value_init_penalty = FLAGS_value_init_penalty;
  player_options->policy_softmax_temp = FLAGS_policy_softmax_temp;
  player_options->virtual_losses = FLAGS_virtual_losses;
  player_options->random_seed = FLAGS_seed;
  player_options->random_symmetry = FLAGS_random_symmetry;
  player_options->num_readouts = FLAGS_num_readouts;
  player_options->seconds_per_move = FLAGS_seconds_per_move;
  player_options->time_limit = FLAGS_time_limit;
  player_options->decay_factor = FLAGS_decay_factor;
  player_options->fastplay_frequency = FLAGS_fastplay_frequency;
  player_options->fastplay_readouts = FLAGS_fastplay_readouts;
  player_options->target_pruning = FLAGS_target_pruning;
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
  explicit SelfPlayer(ModelDescriptor desc)
      : rnd_(Random::kUniqueSeed, Random::kUniqueStream),
        engine_(std::move(desc.engine)),
        model_(std::move(desc.model)) {}

  void Run() {
    auto player_start_time = absl::Now();

    if (FLAGS_cache_size_mb > 0) {
      auto capacity =
          BasicInferenceCache::CalculateCapacity(FLAGS_cache_size_mb);
      MG_LOG(INFO) << "Will cache up to " << capacity
                   << " inferences, using roughly " << FLAGS_cache_size_mb
                   << "MB.\n";
      auto num_shards = std::min(FLAGS_parallel_games, FLAGS_cache_shards);
      inference_cache_ =
          std::make_shared<ThreadSafeInferenceCache>(capacity, num_shards);
    }

    // Figure out how many games we should play.
    MG_CHECK(FLAGS_parallel_games >= 1);

    int num_games = 0;
    if (run_forever_) {
      MG_CHECK(FLAGS_num_games == 0)
          << "num_games must not be set if run_forever is true";
    } else {
      if (FLAGS_num_games == 0) {
        num_games = FLAGS_parallel_games;
      } else {
        MG_CHECK(FLAGS_num_games >= FLAGS_parallel_games)
            << "if num_games is set, it must be >= parallel_games";
        num_games = FLAGS_num_games;
      }
    }

    num_remaining_games_ = num_games;
    run_forever_ = FLAGS_run_forever;

    {
      absl::MutexLock lock(&mutex_);
      auto model_factory = NewModelFactory(engine_);
      // If the model path contains a pattern, wrap the implementation factory
      // in a ReloadingDualNetFactory to automatically reload the latest model
      // that matches the pattern.
      if (model_.find("%d") != std::string::npos) {
        model_factory = absl::make_unique<ReloadingModelFactory>(
            std::move(model_factory), absl::Seconds(3));
      }
      // Note: it's more efficient to perform the reload wrapping before the
      // batch wrapping because this way, we only need to reload the single
      // implementation DualNet when a new model is found. If we performed batch
      // wrapping before reload wrapping, the reload code would need to update
      // all the BatchingModel wrappers.
      batcher_ =
          absl::make_unique<BatchingModelFactory>(std::move(model_factory));
    }
    for (int i = 0; i < FLAGS_parallel_games; ++i) {
      threads_.emplace_back(std::bind(&SelfPlayer::ThreadRun, this, i));
    }
    for (auto& t : threads_) {
      t.join();
    }

    MG_LOG(INFO) << "Played " << num_games << " games, total time "
                 << absl::ToDoubleSeconds(absl::Now() - player_start_time)
                 << " sec.";

    {
      absl::MutexLock lock(&mutex_);
      MG_LOG(INFO) << FormatWinStatsTable({{model_name_, win_stats_}});
    }
  }

 private:
  // Struct that holds the options for each thread.
  // Initialized with the SelfPlayer's mutex held. This allows us to safely
  // update the command line arguments from a flag file without causing any
  // race conditions.
  struct ThreadOptions {
    void Init(int thread_id, Random* rnd) {
      ParseOptionsFromFlags(&game_options, &player_options);
      verbose = thread_id == 0;
      // If an random seed was explicitly specified, make sure we use a
      // different seed for each thread.
      game_options.resign_enabled = (*rnd)() >= FLAGS_disable_resign_pct;

      holdout_pct = FLAGS_holdout_pct;
      output_dir = FLAGS_output_dir;
      holdout_dir = FLAGS_holdout_dir;
      sgf_dir = FLAGS_sgf_dir;
    }

    Game::Options game_options;
    MctsPlayer::Options player_options;
    float holdout_pct;
    std::string output_dir;
    std::string holdout_dir;
    std::string sgf_dir;
    bool verbose = false;
  };

  void ThreadRun(int thread_id) {
    WTF_THREAD_ENABLE("SelfPlay");
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

    for (;;) {
      std::unique_ptr<Game> game;
      std::unique_ptr<MctsPlayer> player;

      {
        absl::MutexLock lock(&mutex_);

        // Check if we've finished playing.
        if (!run_forever_) {
          if (num_remaining_games_ == 0) {
            break;
          }
          num_remaining_games_ -= 1;
        }

        auto old_model = FLAGS_model;
        MaybeReloadFlags();
        MG_CHECK(old_model == FLAGS_model)
            << "Manually changing the model during selfplay is not supported.";
        thread_options.Init(thread_id, &rnd_);
        game = absl::make_unique<Game>(model_, model_,
                                       thread_options.game_options);
        player = absl::make_unique<MctsPlayer>(batcher_->NewModel(model_),
                                               inference_cache_, game.get(),
                                               thread_options.player_options);
        if (model_name_.empty()) {
          model_name_ = player->model()->name();
        }
      }

      if (thread_options.verbose) {
        MG_LOG(INFO) << "MctsPlayer options: " << player->options();
        MG_LOG(INFO) << "Game options: " << game->options();
        MG_LOG(INFO) << "Random seed used: " << player->seed();
      }

      // Play the game.
      auto game_start_time = absl::Now();
      {
        absl::MutexLock lock(&mutex_);
        BatchingModelFactory::StartGame(player->model(), player->model());
      }
      int current_readouts = 0;
      absl::Time search_start_time;
      while (!game->game_over() && !player->root()->at_move_limit()) {
        if (player->root()->position.n() >= kMinPassAliveMoves &&
            player->root()->position.CalculateWholeBoardPassAlive()) {
          // Play pass moves to end the game.
          while (!game->game_over()) {
            MG_CHECK(player->PlayMove(Coord::kPass));
          }
          break;
        }

        // Record some information using for printing tree search stats.
        if (thread_options.verbose) {
          current_readouts = player->root()->N();
          search_start_time = absl::Now();
        }

        bool fastplay =
            (rnd_() < thread_options.player_options.fastplay_frequency);
        int readouts =
            (fastplay ? thread_options.player_options.fastplay_readouts
                      : thread_options.player_options.num_readouts);

        // Choose the move to play, optionally adding noise.
        Coord move = Coord::kInvalid;
        {
          WTF_SCOPE0("SuggestMove");
          move = player->SuggestMove(readouts, !fastplay);
        }

        // Log tree search stats.
        if (thread_options.verbose) {
          WTF_SCOPE0("Logging");
          const auto* root = player->root();
          const auto& position = root->position;

          int num_readouts = root->N() - current_readouts;
          auto elapsed = absl::Now() - search_start_time;
          elapsed = elapsed * 100 / num_readouts;

          auto all_stats = batcher_->FlushStats();
          MG_CHECK(all_stats.size() == 1);
          const auto& stats = all_stats[0].second;
          MG_LOG(INFO)
              << absl::FormatTime("%Y-%m-%d %H:%M:%E3S", absl::Now(),
                                  absl::LocalTimeZone())
              << absl::StreamFormat(
                     "  num_inferences: %d  buffer_count: %d  run_batch_total: "
                     "%.3fms  run_many_total : %.3fms  run_batch_per_inf: "
                     "%.3fms  run_many_per_inf: %.3fms",
                     stats.num_inferences, stats.buffer_count,
                     absl::ToDoubleMilliseconds(stats.run_batch_time),
                     absl::ToDoubleMilliseconds(stats.run_many_time),
                     absl::ToDoubleMilliseconds(stats.run_batch_time /
                                                stats.num_inferences),
                     absl::ToDoubleMilliseconds(stats.run_many_time /
                                                stats.num_inferences));
          MG_LOG(INFO) << root->CalculateTreeStats().ToString();

          if (!fastplay) {
            MG_LOG(INFO) << root->position.ToPrettyString(use_ansi_colors);
            MG_LOG(INFO) << "Move: " << position.n()
                         << " Captures X: " << position.num_captures()[0]
                         << " O: " << position.num_captures()[1];
            MG_LOG(INFO) << root->Describe();
            if (inference_cache_ != nullptr) {
              MG_LOG(INFO) << "Inference cache stats: "
                           << inference_cache_->GetStats();
            }
          }
        }

        // Play the chosen move.
        {
          WTF_SCOPE0("PlayMove");
          MG_CHECK(player->PlayMove(move));
        }

        if (!fastplay && move != Coord::kResign) {
          (*game).MarkLastMoveAsTrainable();
        }

        // Log information about the move played.
        if (thread_options.verbose) {
          MG_LOG(INFO) << absl::StreamFormat("%s Q: %0.5f", player->name(),
                                             player->root()->Q());
          MG_LOG(INFO) << "Played >>" << move;
        }
      }
      {
        absl::MutexLock lock(&mutex_);
        BatchingModelFactory::EndGame(player->model(), player->model());
      }

      if (thread_options.verbose) {
        MG_LOG(INFO) << "Inference history: "
                     << player->GetModelsUsedForInference();
      }

      {
        // Log the end game info with the shared mutex held to prevent the
        // outputs from multiple threads being interleaved.
        absl::MutexLock lock(&mutex_);
        LogEndGameInfo(*game, absl::Now() - game_start_time);
        win_stats_.Update(*game);
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
    }

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
  std::unique_ptr<BatchingModelFactory> batcher_ GUARDED_BY(&mutex_);
  Random rnd_ GUARDED_BY(&mutex_);
  std::string model_name_ GUARDED_BY(&mutex_);
  std::vector<std::thread> threads_;
  std::shared_ptr<ThreadSafeInferenceCache> inference_cache_;

  // True if we should run selfplay indefinitely.
  bool run_forever_ GUARDED_BY(&mutex_) = false;

  // If run_forever_ is false, how many games are left to play.
  int num_remaining_games_ GUARDED_BY(&mutex_) = 0;

  // Stats about how every game was won.
  WinStats win_stats_ GUARDED_BY(&mutex_);

  uint64_t flags_timestamp_ = 0;

  const std::string engine_;
  const std::string model_;
};

}  // namespace
}  // namespace minigo

int main(int argc, char* argv[]) {
  minigo::Init(&argc, &argv);
  minigo::zobrist::Init(FLAGS_seed);

  WTF_THREAD_ENABLE("Main");
  {
    WTF_SCOPE0("Selfplay");
    minigo::SelfPlayer player(minigo::ParseModelDescriptor(FLAGS_model));
    player.Run();
  }

#ifdef WTF_ENABLE
  MG_CHECK(wtf::Runtime::GetInstance()->SaveToFile(FLAGS_wtf_trace));
#endif

  return 0;
}
