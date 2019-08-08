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

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "cc/constants.h"
#include "cc/dual_net/batching_dual_net.h"
#include "cc/dual_net/factory.h"
#include "cc/file/path.h"
#include "cc/file/utils.h"
#include "cc/game.h"
#include "cc/game_utils.h"
#include "cc/init.h"
#include "cc/logging.h"
#include "cc/mcts_player.h"
#include "cc/random.h"
#include "cc/tf_utils.h"
#include "cc/zobrist.h"
#include "gflags/gflags.h"

// Game options flags.
DEFINE_double(resign_threshold, -0.999, "Resign threshold.");
DEFINE_uint64(seed, 0,
              "Random seed. Use default value of 0 to use a time-based seed.");

// Tree search flags.
DEFINE_int32(num_readouts, 100,
             "Number of readouts to make during tree search for each move.");
DEFINE_int32(virtual_losses, 8,
             "Number of virtual losses when running tree search.");
DEFINE_double(value_init_penalty, 2.0,
              "New children value initialization penalty.\n"
              "Child value = parent's value - penalty * color, clamped to"
              " [-1, 1].  Penalty should be in [0.0, 2.0].\n"
              "0 is init-to-parent, 2.0 is init-to-loss [default].\n"
              "This behaves similiarly to Leela's FPU \"First Play Urgency\".");

// Inference flags.
DEFINE_string(model, "",
              "Path to a minigo model. The format of the model depends on the "
              "inference engine. If parallel_games=1, this model is used for "
              "black. For engine=tf, the model should be a GraphDef proto. For "
              "engine=lite, the model should be .tflite flatbuffer.");
DEFINE_string(model_two, "", "Descriptor for the second model");
DEFINE_int32(parallel_games, 32, "Number of games to play in parallel.");

// Output flags.
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

void ParseOptionsFromFlags(Game::Options* game_options,
                           MctsPlayer::Options* player_options) {
  game_options->resign_threshold = -std::abs(FLAGS_resign_threshold);
  game_options->ignore_repeated_moves = true;
  player_options->virtual_losses = FLAGS_virtual_losses;
  player_options->random_seed = FLAGS_seed;
  player_options->num_readouts = FLAGS_num_readouts;
  player_options->inject_noise = false;
  player_options->soft_pick = false;
  player_options->random_symmetry = true;
}

class Evaluator {
  class Model {
   public:
    Model(BatchingDualNetFactory* batcher, const std::string& path)
        : batcher_(batcher), path_(path) {}

    BatchingDualNetFactory* batcher() { return batcher_; }
    std::string name() {
      absl::MutexLock lock(&mutex_);
      if (name_.empty()) {
        // The model's name is lazily initialized the first time we create a
        // instance. Make sure it's valid.
        NewDualNetImpl();
      }
      return name_;
    }

    WinStats GetWinStats() const {
      absl::MutexLock lock(&mutex_);
      return win_stats_;
    }

    void UpdateWinStats(const Game& game) {
      absl::MutexLock lock(&mutex_);
      win_stats_.Update(game);
    }

    std::unique_ptr<DualNet> NewDualNet() {
      absl::MutexLock lock(&mutex_);
      return NewDualNetImpl();
    }

   private:
    std::unique_ptr<DualNet> NewDualNetImpl()
        EXCLUSIVE_LOCKS_REQUIRED(&mutex_) {
      auto dual_net = batcher_->NewDualNet(path_);
      if (name_.empty()) {
        name_ = dual_net->name();
      }
      return dual_net;
    }

    mutable absl::Mutex mutex_;
    BatchingDualNetFactory* batcher_ GUARDED_BY(&mutex_);
    const std::string path_;
    std::string name_ GUARDED_BY(&mutex_);
    WinStats win_stats_ GUARDED_BY(&mutex_);
  };

 public:
  Evaluator(ModelDescriptor desc_a, ModelDescriptor desc_b)
      : desc_a_(std::move(desc_a)), desc_b_(std::move(desc_b)) {
    // Create a batcher for the first model.
    batchers_.push_back(absl::make_unique<BatchingDualNetFactory>(
        NewDualNetFactory(desc_a_.engine)));

    // If the second model requires a different factory, create one & a second
    // batcher too.
    if (desc_b_.engine != desc_a_.engine) {
      batchers_.push_back(absl::make_unique<BatchingDualNetFactory>(
          NewDualNetFactory(desc_b_.engine)));
    }
  }

  void Run() {
    auto start_time = absl::Now();

    Model model_a(batchers_.front().get(), desc_a_.model);
    Model model_b(batchers_.back().get(), desc_b_.model);

    MG_LOG(INFO) << "DualNet factories created from " << desc_a_ << "\n  and "
                 << desc_b_ << " in "
                 << absl::ToDoubleSeconds(absl::Now() - start_time) << " sec.";

    ParseOptionsFromFlags(&game_options_, &player_options_);

    int num_games = FLAGS_parallel_games;
    for (int thread_id = 0; thread_id < num_games; ++thread_id) {
      bool swap_models = (thread_id & 1) != 0;
      threads_.emplace_back(std::bind(&Evaluator::ThreadRun, this, thread_id,
                                      swap_models ? &model_b : &model_a,
                                      swap_models ? &model_a : &model_b));
    }
    for (auto& t : threads_) {
      t.join();
    }

    MG_LOG(INFO) << "Evaluated " << num_games << " games, total time "
                 << (absl::Now() - start_time);

    MG_LOG(INFO) << FormatWinStatsTable(
        {{model_a.name(), model_a.GetWinStats()},
         {model_b.name(), model_b.GetWinStats()}});
  }

 private:
  void ThreadRun(int thread_id, Model* black_model, Model* white_model) {
    // Only print the board using ANSI colors if stderr is sent to the
    // terminal.
    const bool use_ansi_colors = FdSupportsAnsiColors(fileno(stderr));

    // The player and other_player reference this pointer.
    std::unique_ptr<DualNet> dual_net;

    std::vector<std::string> bigtable_spec =
        absl::StrSplit(FLAGS_output_bigtable, ',');
    bool use_bigtable = bigtable_spec.size() == 3;
    if (!FLAGS_output_bigtable.empty() && !use_bigtable) {
      MG_LOG(FATAL)
          << "Bigtable output must be of the form: project,instance,table";
      return;
    }

    Game game(black_model->name(), white_model->name(), game_options_);

    auto player_options = player_options_;
    // If an random seed was explicitly specified, make sure we use a
    // different seed for each thread.
    if (player_options.random_seed != 0) {
      player_options.random_seed += 1299283 * thread_id;
    }

    const bool verbose = thread_id == 0;
    auto black = absl::make_unique<MctsPlayer>(black_model->NewDualNet(),
                                               nullptr, &game, player_options);
    auto white = absl::make_unique<MctsPlayer>(white_model->NewDualNet(),
                                               nullptr, &game, player_options);

    BatchingDualNetFactory::StartGame(black->network(), white->network());
    auto* curr_player = black.get();
    auto* next_player = white.get();
    while (!game.game_over() && !curr_player->root()->at_move_limit()) {
      auto move = curr_player->SuggestMove(player_options.num_readouts);
      if (verbose) {
        std::cerr << curr_player->root()->Describe() << "\n";
      }
      curr_player->PlayMove(move);
      if (!game.game_over()) {
        next_player->PlayMove(move);
      }
      if (verbose) {
        MG_LOG(INFO) << absl::StreamFormat(
            "%d: %s by %s\nQ: %0.4f", curr_player->root()->position.n(),
            move.ToGtp(), curr_player->name(), curr_player->root()->Q());
        MG_LOG(INFO) << curr_player->root()->position.ToPrettyString(
            use_ansi_colors);
      }
      std::swap(curr_player, next_player);
    }
    BatchingDualNetFactory::EndGame(black->network(), white->network());

    if (game.result() > 0) {
      black_model->UpdateWinStats(game);
    } else {
      white_model->UpdateWinStats(game);
    }

    if (verbose) {
      MG_LOG(INFO) << game.result_string();
      MG_LOG(INFO) << "Black was: " << game.black_name();
    }

    // Write SGF.
    std::string output_name = "NO_SGF_SAVED";
    if (!FLAGS_sgf_dir.empty()) {
      output_name = absl::StrCat(GetOutputName(absl::Now(), thread_id), "-",
                                 black->name(), "-", white->name());
      game.AddComment(
          absl::StrCat("B inferences: ", black->GetModelsUsedForInference()));
      game.AddComment(
          absl::StrCat("W inferences: ", white->GetModelsUsedForInference()));
      WriteSgf(FLAGS_sgf_dir, output_name, game, true);
    }

    if (use_bigtable) {
      const auto& gcp_project_name = bigtable_spec[0];
      const auto& instance_name = bigtable_spec[1];
      const auto& table_name = bigtable_spec[2];
      tf_utils::WriteEvalRecord(gcp_project_name, instance_name, table_name,
                                game, output_name, FLAGS_bigtable_tag);
    }

    MG_LOG(INFO) << "Thread " << thread_id << " stopping";
  }

  Game::Options game_options_;
  MctsPlayer::Options player_options_;
  std::vector<std::thread> threads_;

  const ModelDescriptor desc_a_;
  const ModelDescriptor desc_b_;
  std::vector<std::unique_ptr<BatchingDualNetFactory>> batchers_;
};

}  // namespace
}  // namespace minigo

int main(int argc, char* argv[]) {
  minigo::Init(&argc, &argv);
  minigo::zobrist::Init(FLAGS_seed * 614944751);
  minigo::Evaluator evaluator(minigo::ParseModelDescriptor(FLAGS_model),
                              minigo::ParseModelDescriptor(FLAGS_model_two));
  evaluator.Run();
  return 0;
}
