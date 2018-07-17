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
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "cc/check.h"
#include "cc/constants.h"
#ifndef MINIGO_DISABLE_INFERENCE_SERVER
#include "cc/dual_net/inference_server.h"
#endif
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

// Game options flags.
DEFINE_string(mode, "", "Mode to run in: \"selfplay\", \"eval\" or \"gtp\"");
DEFINE_int32(
    ponder_limit, 0,
    "If non-zero and in GTP mode, the number times of times to perform tree "
    "search while waiting for the opponent to play.");
DEFINE_bool(
    courtesy_pass, false,
    "If true and in GTP mode, we will always pass if the opponent passes.");
DEFINE_double(resign_threshold, -0.95, "Resign threshold.");
DEFINE_double(komi, minigo::kDefaultKomi, "Komi.");
DEFINE_double(disable_resign_pct, 0.1,
              "Fraction of games to disable resignation for.");
DEFINE_uint64(seed, 0,
              "Random seed. Use default value of 0 to use a time-based seed. "
              "This seed is used to control the moves played, not whether a "
              "game has resignation disabled or is a holdout.");

// Tree search flags.
DEFINE_int32(num_readouts, 100,
             "Number of readouts to make during tree search for each move.");
DEFINE_int32(virtual_losses, 8,
             "Number of virtual losses when running tree search.");
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
              "Path to a minigo model. If remote_inference=false, the model "
              "should be a serialized GraphDef proto. If "
              "remote_inference=true, the model should be saved checkpoint.");
DEFINE_string(checkpoint_dir, "",
              "Path to a directory containing TensorFlow model checkpoints. "
              "The inference worker will monitor this when a new checkpoint "
              "is found, load the model and use it for futher inferences. "
              "Only valid when remote inference is enabled.");
DEFINE_string(model_two, "",
              "When running 'eval' mode, provide a path to a second minigo "
              "model, also serialized as a GraphDef proto.");
DEFINE_bool(remote_inference, false,
            "If true, run the model using the InferenceServer. This launches "
            "a Python subprocess that performs the actual inference. Required "
            "when running on Cloud TPU.");
DEFINE_int32(port, 50051, "The port opened by the InferenceService server.");
DEFINE_string(tpu_name, "", "Cloud TPU name, e.g. grpc://10.240.2.2:8470.");
DEFINE_int32(parallel_games, 32,
             "Number of games to play in parallel. For performance reasons, "
             "parallel_games should equal games_per_inference * 2 because this "
             "allows the transfer of inference requests & responses to be "
             "overlapped with model evaluation.");
DEFINE_int32(
    games_per_inference, 16,
    "Number of games to merge together into a single inference batch.");
DEFINE_int32(parallel_tpus, 8,
             "If model=remote, the number of TPU cores to run on in parallel.");
DEFINE_int32(conv_width, 256, "Width of the model's convolution filters.");

// Output flags.
DEFINE_string(output_dir, "",
              "Output directory. If empty, no examples are written.");
DEFINE_string(holdout_dir, "",
              "Holdout directory. If empty, no examples are written.");
DEFINE_string(sgf_dir, "", "SGF directory. If empty, no SGF is written.");
DEFINE_double(holdout_pct, 0.05,
              "Fraction of games to hold out for validation.");

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

class PlayerFactory {
 public:
  PlayerFactory(const MctsPlayer::Options& options, float disable_resign_pct)
      : options_(options), disable_resign_pct_(disable_resign_pct) {
    if (FLAGS_remote_inference) {
      inference_worker_thread_ = std::thread([]() {
        std::vector<std::string> cmd_parts = {
            absl::StrCat("BOARD_SIZE=", kN),
            "python",
            "inference_worker.py",
            absl::StrCat("--model=", FLAGS_model),
            absl::StrCat("--checkpoint_dir=", FLAGS_checkpoint_dir),
            "--use_tpu=true",
            absl::StrCat("--tpu_name=", FLAGS_tpu_name),
            absl::StrCat("--conv_width=", FLAGS_conv_width),
            absl::StrCat("--parallel_tpus=", FLAGS_parallel_tpus),
        };
        auto cmd = absl::StrJoin(cmd_parts, " ");
        FILE* f = popen(cmd.c_str(), "r");
        for (;;) {
          int c = fgetc(f);
          if (c == EOF) {
            break;
          }
          fputc(c, stderr);
        }
        fputc('\n', stderr);
      });
    }
  }

  virtual ~PlayerFactory() {
    if (inference_worker_thread_.joinable()) {
      inference_worker_thread_.join();
    }
  }

  virtual std::unique_ptr<MctsPlayer> New(
      const MctsPlayer::Options& options) = 0;

  // Thread safe random number generator, provided for convenience: when
  // multiple players are running in parallel, they will need access to a thread
  // safe random number generator for decisions like whether or not to holdout
  // a game or disable resign.
  float rnd() {
    absl::MutexLock lock(&mutex_);
    return rnd_();
  }

  MctsPlayer::Options default_options() {
    auto options = options_;
    if (rnd() < disable_resign_pct_) {
      options.resign_threshold = -1;
    }
    return options;
  }

 private:
  absl::Mutex mutex_;
  Random rnd_ GUARDED_BY(&mutex_);

  const MctsPlayer::Options options_;
  const float disable_resign_pct_;
  std::thread inference_worker_thread_;
};

class RemotePlayerFactory : public PlayerFactory {
 public:
  RemotePlayerFactory(const MctsPlayer::Options& options,
                      float disable_resign_pct, int virtual_losses,
                      int games_per_inference, int port)
      : PlayerFactory(options, disable_resign_pct) {
    server_ = absl::make_unique<InferenceServer>(virtual_losses,
                                                 games_per_inference, port);
  }

  std::unique_ptr<MctsPlayer> New(const MctsPlayer::Options& options) override {
    return absl::make_unique<MctsPlayer>(server_->NewDualNet(), options);
  }

  std::unique_ptr<InferenceServer> server_;
};

class LocalPlayerFactory : public PlayerFactory {
 public:
  LocalPlayerFactory(const MctsPlayer::Options& options,
                     float disable_resign_pct, std::string model_path)
      : PlayerFactory(options, disable_resign_pct),
        model_path_(std::move(model_path)) {}

  std::unique_ptr<MctsPlayer> New(const MctsPlayer::Options& options) override {
    return absl::make_unique<MctsPlayer>(
        absl::make_unique<TfDualNet>(model_path_), options);
  }

 private:
  const std::string model_path_;
};

std::string GetOutputName(absl::Time now, size_t i) {
  auto timestamp = absl::ToUnixSeconds(now);
  std::string output_name;
  char hostname[64];
  if (gethostname(hostname, sizeof(hostname)) != 0) {
    std::strncpy(hostname, "unknown", sizeof(hostname));
  }
  return absl::StrCat(timestamp, "-", hostname, "-", i);
}

std::string GetOutputDir(absl::Time now, const std::string& root_dir) {
  auto sub_dirs = absl::FormatTime("%Y-%m-%d-%H", now, absl::UTCTimeZone());
  return file::JoinPath(root_dir, sub_dirs);
}

void WriteExample(const std::string& output_dir, const std::string& output_name,
                  const MctsPlayer& player) {
  MG_CHECK(file::RecursivelyCreateDir(output_dir));

  // Write the TensorFlow examples.
  std::vector<tensorflow::Example> examples;
  examples.reserve(player.history().size());
  DualNet::BoardFeatures features;
  std::vector<const Position::Stones*> recent_positions;
  for (const auto& h : player.history()) {
    h.node->GetMoveHistory(DualNet::kMoveHistory, &recent_positions);
    DualNet::SetFeatures(recent_positions, h.node->position.to_play(),
                         &features);
    examples.push_back(
        tf_utils::MakeTfExample(features, h.search_pi, player.result()));
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
  options->batch_size = FLAGS_virtual_losses;
  options->komi = FLAGS_komi;
  options->random_seed = FLAGS_seed;
  options->num_readouts = FLAGS_num_readouts;
  options->seconds_per_move = FLAGS_seconds_per_move;
  options->time_limit = FLAGS_time_limit;
  options->decay_factor = FLAGS_decay_factor;
}

std::unique_ptr<PlayerFactory> GetPlayerFactory() {
  MctsPlayer::Options default_options;
  ParseMctsPlayerOptionsFromFlags(&default_options);
  if (FLAGS_remote_inference) {
    return absl::make_unique<RemotePlayerFactory>(
        default_options, FLAGS_disable_resign_pct, FLAGS_virtual_losses,
        FLAGS_games_per_inference, FLAGS_port);
  } else {
    return absl::make_unique<LocalPlayerFactory>(
        default_options, FLAGS_disable_resign_pct, FLAGS_model);
  }
}

std::thread SelfPlayThread(int thread_id, PlayerFactory* player_factory,
                           bool run_forever) {
  return std::thread([thread_id, player_factory, run_forever]() {
    do {
      // Create a new player.
      auto options = player_factory->default_options();
      options.verbose = thread_id == 0;
      // If an random seed was explicitly specified, make sure we use a
      // different seed for each thread.
      if (options.random_seed != 0) {
        options.random_seed += 1299283 * thread_id;
      }
      auto player = player_factory->New(options);

      // Play the game.
      auto start_time = absl::Now();
      while (!player->game_over()) {
        auto move = player->SuggestMove();
        if (player->options().verbose) {
          std::cerr << player->root()->Describe() << std::endl;
          std::cerr << player->root()->position.ToPrettyString();
        }
        player->PlayMove(move);
      }
      std::cerr << player->result_string() << std::endl;
      std::cout << "Playing game: "
                << absl::ToDoubleSeconds(absl::Now() - start_time) << std::endl;

      // Write the outputs.
      auto now = absl::Now();
      auto output_name = GetOutputName(now, thread_id);

      bool is_holdout = player_factory->rnd() < FLAGS_holdout_pct;
      auto example_dir = is_holdout ? FLAGS_holdout_dir : FLAGS_output_dir;
      if (!example_dir.empty()) {
        WriteExample(GetOutputDir(now, example_dir), output_name, *player);
      }

      if (!FLAGS_sgf_dir.empty()) {
        WriteSgf(GetOutputDir(now, file::JoinPath(FLAGS_sgf_dir, "clean")),
                 output_name, *player, false);
        WriteSgf(GetOutputDir(now, file::JoinPath(FLAGS_sgf_dir, "full")),
                 output_name, *player, true);
      }
    } while (run_forever);
  });
}

void SelfPlay() {
  auto player_factory = GetPlayerFactory();
  std::vector<std::thread> threads;
  for (int i = 0; i < FLAGS_parallel_games; ++i) {
    threads.push_back(
        SelfPlayThread(i, player_factory.get(), FLAGS_run_forever));
  }
  for (auto& t : threads) {
    t.join();
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
      absl::StrCat(GetOutputName(absl::Now(), 0), "-", player->name(), "-",
                   other_player->name());

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
