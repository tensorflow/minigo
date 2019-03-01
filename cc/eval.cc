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
DEFINE_string(model_two, "",
              "Provide a path to a second minigo model, also serialized as a "
              "GraphDef proto. If parallel_games=1, this model is used for "
              "white.");
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

void ParseMctsPlayerOptionsFromFlags(MctsPlayer::Options* options) {
  options->game_options.resign_threshold = -std::abs(FLAGS_resign_threshold);
  options->virtual_losses = FLAGS_virtual_losses;
  options->random_seed = FLAGS_seed;
  options->num_readouts = FLAGS_num_readouts;
  options->inject_noise = false;
  options->soft_pick = false;
  options->random_symmetry = true;
}

class Evaluator {
  struct Model {
    explicit Model(const std::string& path)
        : path(path), name(file::Stem(path)), black_wins(0), white_wins(0) {}
    const std::string path;
    const std::string name;
    std::atomic<int> black_wins;
    std::atomic<int> white_wins;
  };

 public:
  void Run() {
    auto start_time = absl::Now();
    BatchingDualNetFactory batcher(NewDualNetFactory());

    Model model_a(FLAGS_model);
    Model model_b(FLAGS_model_two);

    MG_LOG(INFO) << "DualNet factories created from " << FLAGS_model
                 << "\n  and " << FLAGS_model_two << " in "
                 << absl::ToDoubleSeconds(absl::Now() - start_time) << " sec.";

    ParseMctsPlayerOptionsFromFlags(&options_);

    int num_games = FLAGS_parallel_games;
    for (int thread_id = 0; thread_id < num_games; ++thread_id) {
      bool swap_models = (thread_id & 1) != 0;
      threads_.emplace_back(std::bind(&Evaluator::ThreadRun, this, thread_id,
                                      &batcher,
                                      swap_models ? &model_a : &model_b,
                                      swap_models ? &model_b : &model_a));
    }
    for (auto& t : threads_) {
      t.join();
    }

    MG_LOG(INFO) << "Evaluated " << num_games << " games, total time "
                 << (absl::Now() - start_time);

    auto name_length = std::max(model_a.name.size(), model_b.name.size());
    auto format_name = [&](const std::string& name) {
      return absl::StrFormat("%-*s", name_length, name);
    };
    auto format_wins = [&](int wins) {
      return absl::StrFormat(" %5d %6.2f%%", wins, wins * 100.0f / num_games);
    };
    auto print_result = [&](const Model& model) {
      MG_LOG(INFO) << format_name(model.name)
                   << format_wins(model.black_wins + model.white_wins)
                   << format_wins(model.black_wins)
                   << format_wins(model.white_wins);
    };

    MG_LOG(INFO) << format_name("Wins")
                 << "        Total         Black         White";
    print_result(model_a);
    print_result(model_b);
    MG_LOG(INFO) << format_name("") << "              "
                 << format_wins(model_a.black_wins + model_b.black_wins)
                 << format_wins(model_a.white_wins + model_b.white_wins);
  }

 private:
  void ThreadRun(int thread_id, BatchingDualNetFactory* batcher,
                 Model* black_model, Model* white_model) {
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

    auto player_options = options_;
    // If an random seed was explicitly specified, make sure we use a
    // different seed for each thread.
    if (player_options.random_seed != 0) {
      player_options.random_seed += 1299283 * thread_id;
    }

    const bool verbose = thread_id == 0;
    player_options.verbose = false;
    player_options.name = black_model->name;
    auto black = absl::make_unique<MctsPlayer>(
        batcher->NewDualNet(black_model->path), nullptr, player_options);

    player_options.verbose = false;
    player_options.name = white_model->name;
    auto white = absl::make_unique<MctsPlayer>(
        batcher->NewDualNet(white_model->path), nullptr, player_options);

    Game game(black->name(), white->name(), player_options.game_options);
    auto* curr_player = black.get();
    auto* next_player = white.get();
    batcher->StartGame(curr_player->network(), next_player->network());
    while (!game.game_over() && !curr_player->root()->at_move_limit()) {
      auto move = curr_player->SuggestMove();
      if (verbose) {
        std::cerr << curr_player->root()->Describe() << "\n";
      }
      curr_player->PlayMove(move, &game);
      next_player->PlayMove(move, nullptr);
      if (verbose) {
        MG_LOG(INFO) << absl::StreamFormat("%d: %s by %s\nQ: %0.4f",
                     curr_player->root()->position.n(),
                     move.ToGtp(), curr_player->name(),
                     curr_player->root()->Q());
        MG_LOG(INFO) << curr_player->root()->position.ToPrettyString();
      }
      std::swap(curr_player, next_player);
    }
    batcher->EndGame(curr_player->network(), next_player->network());

    if (game.result() > 0) {
      ++black_model->black_wins;
    } else if (game.result() < 0) {
      ++white_model->white_wins;
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

  MctsPlayer::Options options_;
  std::vector<std::thread> threads_;
};

}  // namespace
}  // namespace minigo

int main(int argc, char* argv[]) {
  minigo::Init(&argc, &argv);
  minigo::zobrist::Init(FLAGS_seed * 614944751);
  minigo::Evaluator evaluator;
  evaluator.Run();
  return 0;
}
