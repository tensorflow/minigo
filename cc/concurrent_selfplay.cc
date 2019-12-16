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

#include <cmath>
#include <functional>
#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "cc/async/poll_thread.h"
#include "cc/async/sharded_executor.h"
#include "cc/async/thread.h"
#include "cc/async/thread_safe_queue.h"
#include "cc/file/directory_watcher.h"
#include "cc/file/path.h"
#include "cc/file/utils.h"
#include "cc/game.h"
#include "cc/game_utils.h"
#include "cc/init.h"
#include "cc/logging.h"
#include "cc/mcts_tree.h"
#include "cc/model/inference_cache.h"
#include "cc/model/loader.h"
#include "cc/platform/utils.h"
#include "cc/random.h"
#include "cc/tf_utils.h"
#include "cc/wtf_saver.h"
#include "cc/zobrist.h"
#include "gflags/gflags.h"
#include "wtf/macros.h"

// Plays multiple selfplay games.
// There are several important classes in this binary:
//  - `SelfplayGame` : holds the state for a single game, most importantly an
//    `MctsTree` and a `Game`. The `SelfplayGame` is responsible for selecting
//    leaves in the MCTS tree to run inference on, propagating inference
//    results back up the tree, and playing moves.
//  - `SelfplayThread` : owns multiple `SelfplayGame` instances and uses them
//    to play games concurrently. See SelfplayThread::Run for the sequence of
//    operations performed when playing games. Tree search is carried out in
//    batches on multiple threads in parallel.
//  - `Selfplayer` : owns multiple `SelfplayThread` instances, which lets the
//    binary perform tree search on multiple threads.
//  - `OutputThread` : responsible for writing SGF & training examples to
//    storage. After a game finished, its `SelfplayThread` hands the
//    `SelfplayGame` instance back to the `Selfplayer`, which pushes it onto
//    an output queue for `OutputThread` to consume.

// Inference flags.
DEFINE_string(device, "",
              "Optional ID of the device to run inference on. For TPUs, pass "
              "the gRPC address.");
DEFINE_string(model, "", "Path to a minigo model.");
DEFINE_int32(cache_size_mb, 0, "Size of the inference cache in MB.");
DEFINE_int32(cache_shards, 8,
             "Number of ways to shard the inference cache. The cache uses "
             "is locked on a per-shard basis, so more shards means less "
             "contention but each shard is smaller. The number of shards "
             "is clamped such that it's always <= parallel_games.");

// Tree search flags.
DEFINE_int32(num_readouts, 104,
             "Number of readouts to make during tree search for each move.");
DEFINE_double(fastplay_frequency, 0.0,
              "The fraction of moves that should use a lower number of "
              "playouts, aka 'playout cap oscillation'.\nIf this is set, "
              "'fastplay_readouts' should also be set.");
DEFINE_int32(fastplay_readouts, 20,
             "The number of readouts to perform on a 'low readout' move, "
             "aka 'playout cap oscillation'.\nIf this is set, "
             "'fastplay_frequency' should be nonzero.");
DEFINE_int32(virtual_losses, 8,
             "Number of virtual losses when running tree search.");
DEFINE_double(dirichlet_alpha, 0.03, "Alpha value for Dirichlet noise.");
DEFINE_double(noise_mix, 0.25, "The amount of noise to mix into the root.");
DEFINE_double(value_init_penalty, 2.0,
              "New children value initialization penalty.\n"
              "Child value = parent's value - penalty * color, clamped to "
              "[-1, 1].  Penalty should be in [0.0, 2.0].\n"
              "0 is init-to-parent, 2.0 is init-to-loss [default].\n"
              "This behaves similiarly to Leela's FPU \"First Play Urgency\".");
DEFINE_bool(target_pruning, false,
            "If true, subtract visits from all moves that weren't the best "
            "move until the uncertainty level compensates.");
DEFINE_double(policy_softmax_temp, 0.98,
              "For soft-picked moves, the probabilities are exponentiated by "
              "policy_softmax_temp to encourage diversity in early play.\n");
DEFINE_bool(allow_pass, true,
            "If false, pass moves will only be read and played if there is no "
            "other legal alternative.");
DEFINE_int32(restrict_pass_alive_play_threshold, 4,
             "If the opponent has passed at least "
             "restrict_pass_alive_play_threshold pass moves in a row, playing "
             "moves in pass-alive territory of either player is disallowed.");

// Threading flags.
DEFINE_int32(selfplay_threads, 3,
             "Number of threads to run batches of selfplay games on.");
DEFINE_int32(parallel_search, 3, "Number of threads to run tree search on.");
DEFINE_int32(parallel_inference, 2, "Number of threads to run inference on.");
DEFINE_int32(concurrent_games_per_thread, 1,
             "Number of games to play concurrently on each selfplay thread. "
             "Inferences from a thread's concurrent games are batched up and "
             "evaluated together. Increasing concurrent_games_per_thread can "
             "help improve GPU or TPU utilization, especially for small "
             "models.");

// Game flags.
DEFINE_uint64(seed, 0,
              "Random seed. Use default value of 0 to use a time-based seed. "
              "This seed is used to control the moves played, not whether a "
              "game has resignation disabled or is a holdout.");
DEFINE_double(min_resign_threshold, -1.0,
              "Each game's resign threshold is picked randomly from the range "
              "[min_resign_threshold, max_resign_threshold)");
DEFINE_double(max_resign_threshold, -0.8,
              "Each game's resign threshold is picked randomly from the range "
              "[min_resign_threshold, max_resign_threshold)");
DEFINE_double(disable_resign_pct, 0.1,
              "Fraction of games to disable resignation for.");
DEFINE_int32(num_games, 0,
             "Total number of games to play. Only one of run_forever and "
             "num_games must be set.");
DEFINE_bool(run_forever, false,
            "Whether to run forever. Only one of run_forever and num_games "
            "must be set.");
DEFINE_string(abort_file, "",
              "If non-empty, specifies a path to a file whose presence is "
              "checked for periodically when run_forever=true. If the file "
              "exists the selfplay process will abort immediately.");

// Output flags.
DEFINE_double(holdout_pct, 0.03,
              "Fraction of games to hold out for validation.");
DEFINE_string(output_dir, "",
              "Output directory. If empty, no examples are written. If "
              "output_dir contains the substring \"$MODEL\", the name of "
              "the last models used for inference when playing a game will "
              "be substituded in the path.");
DEFINE_string(holdout_dir, "",
              "Holdout directory. If empty, no examples are written. If "
              "holdout_dir contains the substring \"$MODEL\", the name of "
              "the last models used for inference when playing a game will "
              "be substituded in the path.");
DEFINE_string(sgf_dir, "",
              "Directory to write output SGFs to. If sgf_dir contains the "
              "substring \"$MODEL\", the name of the last models used for "
              "inference when playing a game will be substituded in the "
              "path.");
DEFINE_string(wtf_trace, "/tmp/minigo.wtf-trace",
              "Output path for WTF traces.");
DEFINE_bool(verbose, true, "Whether to log progress.");
DEFINE_int32(output_threads, 1,
             "Number of threads write training examples on.");

namespace minigo {
namespace {

std::string GetOutputDir(absl::Time now, const std::string& model_name,
                         const std::string& root_dir) {
  auto sub_dirs = absl::FormatTime("%Y-%m-%d-%H", now, absl::UTCTimeZone());
  auto clean_model_name =
      absl::StrReplaceAll(model_name, {{":", "_"}, {"/", "_"}, {".", "_"}});
  std::string processed_root_dir =
      absl::StrReplaceAll(root_dir, {{"$MODEL", clean_model_name}});
  return file::JoinPath(processed_root_dir, sub_dirs);
}

// Information required to run a single inference.
struct Inference {
  InferenceCache::Key cache_key;
  MctsNode* leaf;
  ModelInput input;
  ModelOutput output;
};

// Holds all the state for a single selfplay game.
// Each `SelfplayThread` plays multiple games in parallel, calling
// `SelectLeaves`, `ProcessInferences` and `MaybePlayMove` sequentially.
class SelfplayGame {
 public:
  struct Options {
    // Number of virtual losses.
    int num_virtual_losses;

    // Number of positions to read normally.
    int num_readouts;

    // Number of positions to read if playout cap oscillations determines that
    // this should be a "fast" play.
    int fastplay_readouts;

    // Frequency that a move should be a "fast" play.
    float fastplay_frequency;

    // Alpha value for Dirichlet noise.
    float dirichlet_alpha;

    // Fraction of noise to mix into the root node before performing reads.
    // Noise is not injected for "fast" plays.
    float noise_mix;

    // True if this game's data should be written to the `holdout_dir` instead
    // of the `output_dir`.
    bool is_holdout;

    // If true, subtract visits from all moves that weren't the best move until
    // the uncertainty level compensates.
    bool target_pruning;

    // If true, perform verbose logging. Usually restricted to just the first
    // `SelfplayGame` of the first `SelfplayThread`.
    bool verbose;

    // If false, pass is only read and played if there are no other legal
    // alternatives.
    bool allow_pass;

    // Disallow playing in pass-alive territory once the number of passes played
    // during a game is at least `restrict_pass_alive_play_threshold`.
    int restrict_pass_alive_play_threshold;
  };

  // Stats about the nodes visited during SelectLeaves.
  struct SelectLeavesStats {
    int num_leaves_queued = 0;
    int num_nodes_selected = 0;
    int num_cache_hits = 0;
    int num_game_over_leaves = 0;

    SelectLeavesStats& operator+=(const SelectLeavesStats& other) {
      num_leaves_queued += other.num_leaves_queued;
      num_nodes_selected += other.num_nodes_selected;
      num_cache_hits += other.num_cache_hits;
      num_game_over_leaves += other.num_game_over_leaves;
      return *this;
    }
  };

  SelfplayGame(int game_id, const Options& options, std::unique_ptr<Game> game,
               std::unique_ptr<MctsTree> tree);

  int game_id() const { return game_id_; }
  Game* game() { return game_.get(); }
  const Game* game() const { return game_.get(); }
  const MctsTree* tree() const { return tree_.get(); }
  absl::Duration duration() const { return duration_; }
  const Options& options() const { return options_; }
  const std::vector<std::string>& models_used() const { return models_used_; }

  // Selects leaves to perform inference on.
  // Returns the number of leaves selected. It is possible that no leaves will
  // be selected if all desired leaves are already in the inference cache.
  SelectLeavesStats SelectLeaves(InferenceCache* cache,
                                 std::vector<Inference>* inferences);

  // Processes the inferences selected by `SelectedLeaves` that were evaluated
  // by the SelfplayThread.
  void ProcessInferences(const std::string& model_name,
                         absl::Span<const Inference> inferences);

  // Plays a move if the necessary number of nodes have been read.
  // Returns true if `MaybePlayMove` actually played a move.
  // Returns false if the `SeflplayGame` needs to read more positions before it
  // can play a move.
  bool MaybePlayMove();

 private:
  // Randomly choose whether or not to fast play.
  bool ShouldFastplay();

  // Returns true if the predicted win rate is below `resign_threshold`.
  bool ShouldResign() const;

  // Injects noise into the root.
  void InjectNoise();

  // Returns the symmetry that should be when performing inference on this
  // node's position.
  symmetry::Symmetry GetInferenceSymmetry(const MctsNode* node) const;

  // Looks the `leaf` up in the inference cache:
  //  - if found: propagates the cached inference result back up the tree.
  //  - if not found: appends an element to `inferences` to perform inference
  //    on `leaf`.
  // Returns true in an inference was queued.
  bool MaybeQueueInference(MctsNode* leaf, InferenceCache* cache,
                           std::vector<Inference>* inferences);

  const Options options_;
  int target_readouts_;
  std::unique_ptr<Game> game_;
  std::unique_ptr<MctsTree> tree_;
  const bool use_ansi_colors_;
  const absl::Time start_time_;
  absl::Duration duration_;
  std::vector<std::string> models_used_;
  Random rnd_;
  const uint64_t inference_symmetry_mix_;

  // We need to wait until the root is expanded by the first call to
  // SelectLeaves in the game before injecting noise.
  bool inject_noise_before_next_read_ = false;

  // We don't allow fast play for the opening move: fast play relies to some
  // degree on tree reuse from earlier reads but the tree is empty at the start
  // of the game.
  bool fastplay_ = false;

  // Number of consecutive passes played by black and white respectively.
  // Used to determine when to disallow playing in pass-alive territory.
  // `num_consecutive_passes_` latches once it reaches
  // `restrict_pass_alive_play_threshold` is is not reset to 0 when a non-pass
  // move is played.
  int num_consecutive_passes_[2] = {0, 0};

  const int game_id_;
};

// The main application class.
// Manages multiple SelfplayThread objects.
// Each SelfplayThread plays multiple games concurrently, each one is
// represented by a SelfplayGame.
// The Selfplayer also has a OutputThread, which writes the results of completed
// games to disk.
class Selfplayer {
 public:
  Selfplayer();

  void Run() LOCKS_EXCLUDED(&mutex_);

  std::unique_ptr<SelfplayGame> StartNewGame(bool verbose)
      LOCKS_EXCLUDED(&mutex_);

  void EndGame(std::unique_ptr<SelfplayGame> selfplay_game)
      LOCKS_EXCLUDED(&mutex_);

  // Exectutes `fn` on `parallel_search` threads in parallel on a shared
  // `ShardedExecutor`.
  // Concurrent calls to `ExecuteSharded` are executed sequentially, unless
  // `parallel_search == 1`. This blocking property can be used to pipeline
  // CPU tree search and GPU inference.
  void ExecuteSharded(std::function<void(int, int)> fn);

  // Grabs a model from a pool. If `selfplay_threads > parallel_inference`,
  // `AcquireModel` may block if a model isn't immediately available.
  std::unique_ptr<Model> AcquireModel();

  // Gives a previously acquired model back to the pool.
  void ReleaseModel(std::unique_ptr<Model> model);

 private:
  void ParseFlags() EXCLUSIVE_LOCKS_REQUIRED(&mutex_);
  FeatureDescriptor InitializeModels();
  void CreateModels(const std::string& path);
  void CheckAbortFile();

  mutable absl::Mutex mutex_;
  Game::Options game_options_ GUARDED_BY(&mutex_);
  MctsTree::Options tree_options_ GUARDED_BY(&mutex_);
  int num_games_remaining_ GUARDED_BY(&mutex_) = 0;
  Random rnd_ GUARDED_BY(&mutex_);
  WinStats win_stats_ GUARDED_BY(&mutex_);
  ThreadSafeQueue<std::unique_ptr<SelfplayGame>> output_queue_;
  ShardedExecutor executor_;

  ThreadSafeQueue<std::unique_ptr<Model>> models_;

  // The latest path that matches the model pattern.
  std::string latest_model_name_ GUARDED_BY(&mutex_);

  int next_game_id_ GUARDED_BY(&mutex_) = 1;

  std::unique_ptr<DirectoryWatcher> directory_watcher_;
  std::unique_ptr<PollThread> abort_file_watcher_;

  std::unique_ptr<WtfSaver> wtf_saver_;
};

// Plays multiple games concurrently using `SelfplayGame` instances.
class SelfplayThread : public Thread {
 public:
  SelfplayThread(int thread_id, Selfplayer* selfplayer,
                 std::shared_ptr<InferenceCache> cache);

 private:
  void Run() override;

  // Starts new games playing.
  void StartNewGames();

  // Selects leaves to perform inference on for all currently playing games.
  // The selected leaves are stored in `inferences_` and `inference_spans_`
  // maps contents of `inferences_` back to the `SelfplayGames` that they
  // came from.
  void SelectLeaves();

  // Runs inference on the leaves selected by `SelectLeaves`.
  // Runs the name of the model that ran the inferences.
  std::string RunInferences();

  // Calls `SelfplayGame::ProcessInferences` for all inferences performed.
  void ProcessInferences(const std::string& model);

  // Plays moves on all games that have performed sufficient reads.
  void PlayMoves();

  struct TreeSearch {
    // Holds the span of inferences requested for a single `SelfplayGame`:
    // `pos` and `len` index into the `inferences` array.
    struct InferenceSpan {
      SelfplayGame* selfplay_game;
      size_t pos;
      size_t len;
    };

    void Clear() {
      inferences.clear();
      inference_spans.clear();
    }

    std::vector<Inference> inferences;
    std::vector<InferenceSpan> inference_spans;
  };

  Selfplayer* selfplayer_;
  int num_virtual_losses_ = 8;
  std::vector<std::unique_ptr<SelfplayGame>> selfplay_games_;
  std::unique_ptr<Model> model_;
  std::shared_ptr<InferenceCache> cache_;
  std::vector<TreeSearch> searches_;
  int num_games_finished_ = 0;
  const int thread_id_;
};

// Writes SGFs and training examples for completed games to disk.
class OutputThread : public Thread {
 public:
  OutputThread(int thread_id, FeatureDescriptor feature_descriptor,
               ThreadSafeQueue<std::unique_ptr<SelfplayGame>>* output_queue);

 private:
  void Run() override;
  void WriteOutputs(std::unique_ptr<SelfplayGame> selfplay_game);

  ThreadSafeQueue<std::unique_ptr<SelfplayGame>>* output_queue_;
  const std::string output_dir_;
  const std::string holdout_dir_;
  const std::string sgf_dir_;
  const FeatureDescriptor feature_descriptor_;
};

Selfplayer::Selfplayer()
    : rnd_(FLAGS_seed, Random::kUniqueStream),
      executor_(FLAGS_parallel_search) {
  absl::MutexLock lock(&mutex_);
  ParseFlags();
}

SelfplayGame::SelfplayGame(int game_id, const Options& options,
                           std::unique_ptr<Game> game,
                           std::unique_ptr<MctsTree> tree)
    : options_(options),
      game_(std::move(game)),
      tree_(std::move(tree)),
      use_ansi_colors_(FdSupportsAnsiColors(fileno(stderr))),
      start_time_(absl::Now()),
      rnd_(FLAGS_seed, Random::kUniqueStream),
      inference_symmetry_mix_(rnd_.UniformUint64()),
      game_id_(game_id) {
  target_readouts_ = options_.num_readouts;
}

SelfplayGame::SelectLeavesStats SelfplayGame::SelectLeaves(
    InferenceCache* cache, std::vector<Inference>* inferences) {
  // We can only inject noise if the root is expanded. If it isn't expanded
  // yet, the next call to SelectLeaf must by definition select the root (and
  // break out of the loop below). We'll then inject the noise on the subsequent
  // call to SelectLeaves.
  if (inject_noise_before_next_read_ && tree_->root()->is_expanded) {
    inject_noise_before_next_read_ = false;
    InjectNoise();
  }

  const auto* root = tree_->root();
  SelectLeavesStats stats;
  do {
    auto* leaf = tree_->SelectLeaf(options_.allow_pass);
    if (leaf == nullptr) {
      // TODO(tommadams): Breaking here without updating the stats will under
      // count num_nodes_selected. To fix this, we'd have to plumb the count
      // through MctsTree::SelectLeaf, which is probably not worth doing
      // because MctsTree::SelectLeaf rarely returns null.
      break;
    }

    stats.num_nodes_selected += leaf->position.n() - root->position.n();

    if (leaf->game_over()) {
      float value =
          leaf->position.CalculateScore(game_->options().komi) > 0 ? 1 : -1;
      tree_->IncorporateEndGameResult(leaf, value);
      stats.num_game_over_leaves += 1;
      continue;
    }
    if (MaybeQueueInference(leaf, cache, inferences)) {
      stats.num_leaves_queued += 1;
    } else {
      stats.num_cache_hits += 1;
    }
    if (leaf == root) {
      if (!fastplay_) {
        inject_noise_before_next_read_ = true;
      }
      break;
    }
  } while (stats.num_leaves_queued < options_.num_virtual_losses &&
           tree_->root()->N() < target_readouts_);
  return stats;
}

void SelfplayGame::ProcessInferences(const std::string& model_name,
                                     absl::Span<const Inference> inferences) {
  if (!model_name.empty()) {
    if (models_used_.empty() || model_name != models_used_.back()) {
      models_used_.push_back(model_name);
    }
  }

  for (const auto& inference : inferences) {
    tree_->IncorporateResults(inference.leaf, inference.output.policy,
                              inference.output.value);
    tree_->RevertVirtualLoss(inference.leaf);
  }
}

bool SelfplayGame::MaybePlayMove() {
  // Check if this game's tree search has performed enough reads that it
  // should now play a move.
  if (tree_->root()->N() < target_readouts_) {
    return false;
  }

  // Handle resignation.
  if (ShouldResign()) {
    game_->SetGameOverBecauseOfResign(OtherColor(tree_->to_play()));
  } else {
    // Restrict playing in pass-alive territory once the opponent has passed
    // `restrict_pass_alive_play_threshold` times in a row.
    int num_opponent_passes =
        num_consecutive_passes_[tree_->to_play() == Color::kBlack ? 1 : 0];
    bool restrict_pass_alive_moves =
        num_opponent_passes >= options_.restrict_pass_alive_play_threshold;

    Coord c = tree_->PickMove(&rnd_, restrict_pass_alive_moves);
    if (options_.verbose) {
      const auto& position = tree_->root()->position;
      MG_LOG(INFO) << position.ToPrettyString(use_ansi_colors_);
      MG_LOG(INFO) << "Move: " << position.n()
                   << " Captures X: " << position.num_captures()[0]
                   << " O: " << position.num_captures()[1];
      if (!fastplay_) {
        MG_LOG(INFO) << tree_->Describe();
      }
      MG_LOG(INFO) << absl::StreamFormat("Q: %0.5f", tree_->root()->Q());
      MG_LOG(INFO) << "Played >> " << tree_->to_play() << "[" << c << "]";
    }

    std::string model_str;
    if (!models_used_.empty()) {
      model_str = absl::StrCat("model: ", models_used_.back(), "\n");
    }

    if (options_.target_pruning && !fastplay_) {
      tree_->ReshapeFinalVisits(restrict_pass_alive_moves);
    }

    if (!fastplay_ && c != Coord::kResign) {
      auto search_pi = tree_->CalculateSearchPi();
      game_->AddTrainableMove(tree_->to_play(), c, tree_->root()->position,
                              std::move(model_str), tree_->root()->Q(),
                              tree_->root()->N(), search_pi);
    } else {
      game_->AddNonTrainableMove(tree_->to_play(), c, tree_->root()->position,
                                 std::move(model_str), tree_->root()->Q(),
                                 tree_->root()->N());
    }

    // Update the number of consecutive passes.
    // The number of consecutive passes latches when it hits
    // `restrict_pass_alive_play_threshold`.
    int& num_passes =
        num_consecutive_passes_[tree_->to_play() == Color::kBlack ? 0 : 1];
    if (num_passes < options_.restrict_pass_alive_play_threshold) {
      if (c == Coord::kPass) {
        num_passes += 1;
      } else {
        num_passes = 0;
      }
    }

    tree_->PlayMove(c);

    // if (tree_->root()->position.n() > 8) {
    //   while (!tree_->is_game_over()) {
    //     tree_->PlayMove(Coord::kPass);
    //   }
    // }

    // If the whole board is pass-alive, play pass moves to end the game.
    if (tree_->root()->position.n() >= kMinPassAliveMoves &&
        tree_->root()->position.CalculateWholeBoardPassAlive()) {
      while (!tree_->is_game_over()) {
        tree_->PlayMove(Coord::kPass);
      }
    }

    // TODO(tommadams): move game over logic out of MctsTree and into Game.
    if (tree_->is_game_over()) {
      game_->SetGameOverBecauseOfPasses(
          tree_->CalculateScore(game_->options().komi));
    }
  }

  if (!game_->game_over()) {
    fastplay_ = ShouldFastplay();
    inject_noise_before_next_read_ = !fastplay_;
    int num_readouts =
        fastplay_ ? options_.fastplay_readouts : options_.num_readouts;
    target_readouts_ = tree_->root()->N() + num_readouts;
    if (!fastplay_) {
      if (options_.fastplay_frequency > 0) {
        tree_->ClearSubtrees();
      }
    }
  } else {
    duration_ = absl::Now() - start_time_;
  }

  return true;
}

bool SelfplayGame::ShouldFastplay() {
  return options_.fastplay_frequency > 0 &&
         rnd_() < options_.fastplay_frequency;
}

bool SelfplayGame::ShouldResign() const {
  return game_->options().resign_enabled &&
         tree_->root()->Q_perspective() < game_->options().resign_threshold;
}

void SelfplayGame::InjectNoise() {
  tree_->InjectNoise(rnd_.Dirichlet<kNumMoves>(options_.dirichlet_alpha),
                     options_.noise_mix);
}

symmetry::Symmetry SelfplayGame::GetInferenceSymmetry(
    const MctsNode* node) const {
  uint64_t bits =
      Random::MixBits(node->position.stone_hash() * Random::kLargePrime +
                      inference_symmetry_mix_);
  return static_cast<symmetry::Symmetry>(bits % symmetry::kNumSymmetries);
}

bool SelfplayGame::MaybeQueueInference(MctsNode* leaf, InferenceCache* cache,
                                       std::vector<Inference>* inferences) {
  ModelOutput cached_output;

  auto inference_sym = GetInferenceSymmetry(leaf);
  auto cache_key =
      InferenceCache::Key(leaf->move, leaf->canonical_symmetry, leaf->position);
  if (cache->TryGet(cache_key, leaf->canonical_symmetry, inference_sym,
                    &cached_output)) {
    tree_->IncorporateResults(leaf, cached_output.policy, cached_output.value);
    return false;
  }

  inferences->emplace_back();
  auto& inference = inferences->back();
  inference.cache_key = cache_key;
  inference.input.sym = inference_sym;
  inference.leaf = leaf;

  // TODO(tommadams): add a method to FeatureDescriptor that returns the
  // required position history size.
  auto* node = leaf;
  for (int i = 0; i < inference.input.position_history.capacity(); ++i) {
    inference.input.position_history.push_back(&node->position);
    node = node->parent;
    if (node == nullptr) {
      break;
    }
  }

  tree_->AddVirtualLoss(leaf);
  return true;
}

void Selfplayer::Run() {
  // Create the inference cache.
  std::shared_ptr<InferenceCache> inference_cache;
  if (FLAGS_cache_size_mb > 0) {
    auto capacity = BasicInferenceCache::CalculateCapacity(FLAGS_cache_size_mb);
    MG_LOG(INFO) << "Will cache up to " << capacity
                 << " inferences, using roughly " << FLAGS_cache_size_mb
                 << "MB.\n";
    inference_cache = std::make_shared<ThreadSafeInferenceCache>(
        capacity, FLAGS_cache_shards);
  } else {
    inference_cache = std::make_shared<NullInferenceCache>();
  }

  if (FLAGS_run_forever) {
    // Note that we don't ever have to worry about joining this thread because
    // it's only ever created when selfplay runs forever and when it comes time
    // to terminate the process, CheckAbortFile will call abort().
    abort_file_watcher_ = absl::make_unique<PollThread>(
        "AbortWatcher", absl::Seconds(5),
        std::bind(&Selfplayer::CheckAbortFile, this));
    abort_file_watcher_->Start();
  }

  // Load the models.
  auto feature_descriptor = InitializeModels();

  // Initialize the selfplay threads.
  std::vector<std::unique_ptr<SelfplayThread>> selfplay_threads;

  {
    absl::MutexLock lock(&mutex_);
    selfplay_threads.reserve(FLAGS_selfplay_threads);
    for (int i = 0; i < FLAGS_selfplay_threads; ++i) {
      selfplay_threads.push_back(
          absl::make_unique<SelfplayThread>(i, this, inference_cache));
    }
  }

  // Start the output threads.
  std::vector<std::unique_ptr<OutputThread>> output_threads;
  for (int i = 0; i < FLAGS_output_threads; ++i) {
    output_threads.push_back(
        absl::make_unique<OutputThread>(i, feature_descriptor, &output_queue_));
  }
  for (auto& t : output_threads) {
    t->Start();
  }

#ifdef WTF_ENABLE
  // Save WTF in the background periodically.
  wtf_saver_ = absl::make_unique<WtfSaver>(FLAGS_wtf_trace, absl::Seconds(5));
#endif  // WTF_ENABLE

  // Run the selfplay threads.
  for (auto& t : selfplay_threads) {
    t->Start();
  }
  for (auto& t : selfplay_threads) {
    t->Join();
  }

  // Stop the output threads by pushing one null game onto the output queue
  // for each thread, causing the treads to exit when the pop them off.
  for (size_t i = 0; i < output_threads.size(); ++i) {
    output_queue_.Push(nullptr);
  }
  for (auto& t : output_threads) {
    t->Join();
  }
  MG_CHECK(output_queue_.empty());

  if (FLAGS_cache_size_mb > 0) {
    MG_LOG(INFO) << "Inference cache stats: " << inference_cache->GetStats();
  }

  {
    absl::MutexLock lock(&mutex_);
    MG_LOG(INFO) << FormatWinStatsTable({{latest_model_name_, win_stats_}});
  }
}

std::unique_ptr<SelfplayGame> Selfplayer::StartNewGame(bool verbose) {
  WTF_SCOPE0("StartNewGame");

  Game::Options game_options;
  MctsTree::Options tree_options;
  SelfplayGame::Options selfplay_options;

  std::string player_name;
  int game_id;
  {
    absl::MutexLock lock(&mutex_);
    if (!FLAGS_run_forever && num_games_remaining_ == 0) {
      return nullptr;
    }
    if (!FLAGS_run_forever) {
      num_games_remaining_ -= 1;
    }

    player_name = latest_model_name_;
    game_id = next_game_id_++;

    game_options = game_options_;
    game_options.resign_enabled = rnd_() >= FLAGS_disable_resign_pct;

    tree_options = tree_options_;

    selfplay_options.num_virtual_losses = FLAGS_virtual_losses;
    selfplay_options.num_readouts = FLAGS_num_readouts;
    selfplay_options.fastplay_readouts = FLAGS_fastplay_readouts;
    selfplay_options.fastplay_frequency = FLAGS_fastplay_frequency;
    selfplay_options.noise_mix = FLAGS_noise_mix;
    selfplay_options.dirichlet_alpha = FLAGS_dirichlet_alpha;
    selfplay_options.is_holdout = rnd_() < FLAGS_holdout_pct;
    selfplay_options.target_pruning = FLAGS_target_pruning;
    selfplay_options.verbose = verbose;
    selfplay_options.allow_pass = FLAGS_allow_pass;
    selfplay_options.restrict_pass_alive_play_threshold =
        FLAGS_restrict_pass_alive_play_threshold;
  }

  auto game = absl::make_unique<Game>(player_name, player_name, game_options);
  auto tree =
      absl::make_unique<MctsTree>(Position(Color::kBlack), tree_options);

  return absl::make_unique<SelfplayGame>(game_id, selfplay_options,
                                         std::move(game), std::move(tree));
}

void Selfplayer::EndGame(std::unique_ptr<SelfplayGame> selfplay_game) {
  {
    absl::MutexLock lock(&mutex_);
    win_stats_.Update(*selfplay_game->game());
  }
  output_queue_.Push(std::move(selfplay_game));
}

void Selfplayer::ExecuteSharded(std::function<void(int, int)> fn) {
  executor_.Execute(std::move(fn));
}

std::unique_ptr<Model> Selfplayer::AcquireModel() { return models_.Pop(); }

void Selfplayer::ReleaseModel(std::unique_ptr<Model> model) {
  if (model->name() == latest_model_name_) {
    models_.Push(std::move(model));
  }
}

void Selfplayer::ParseFlags() {
  // Check that exactly one of (run_forever and num_games) is set.
  if (FLAGS_run_forever) {
    MG_CHECK(FLAGS_num_games == 0)
        << "num_games must not be set if run_forever is true";
  } else {
    MG_CHECK(FLAGS_num_games > 0)
        << "num_games must be set if run_forever is false";
  }
  MG_CHECK(!FLAGS_model.empty());

  // Clamp num_concurrent_games_per_thread to avoid a situation where a single
  // thread ends up playing considerably more games than the others.
  if (!FLAGS_run_forever) {
    auto max_concurrent_games_per_thread =
        (FLAGS_num_games + FLAGS_selfplay_threads - 1) / FLAGS_selfplay_threads;
    FLAGS_concurrent_games_per_thread = std::min(
        max_concurrent_games_per_thread, FLAGS_concurrent_games_per_thread);
  }
  game_options_.resign_threshold =
      -rnd_.Uniform(std::fabs(FLAGS_min_resign_threshold),
                    std::fabs(FLAGS_max_resign_threshold));

  tree_options_.value_init_penalty = FLAGS_value_init_penalty;
  tree_options_.policy_softmax_temp = FLAGS_policy_softmax_temp;
  tree_options_.soft_pick_enabled = true;
  num_games_remaining_ = FLAGS_num_games;
}

FeatureDescriptor Selfplayer::InitializeModels() {
  if (FLAGS_model.find("%d") != std::string::npos) {
    using namespace std::placeholders;
    directory_watcher_ = absl::make_unique<DirectoryWatcher>(
        FLAGS_model, absl::Seconds(5),
        std::bind(&Selfplayer::CreateModels, this, _1));
    MG_LOG(INFO) << "Waiting for model to match pattern " << FLAGS_model;
  } else {
    CreateModels(FLAGS_model);
  }

  // Get the feature descriptor from the first model loaded.
  // TODO(tommadams): instead of this, specify the model features explicitly on
  // the command line and pass them in to ModelFactory::NewModel, checking that
  // the models input shape matches the expected number of features.
  auto model = models_.Pop();
  auto feature_descriptor = model->feature_descriptor();
  models_.Push(std::move(model));

  return feature_descriptor;
}

void Selfplayer::CreateModels(const std::string& path) {
  MG_LOG(INFO) << "Loading model " << path;

  auto def = LoadModelDefinition(path);
  auto* factory = GetModelFactory(def, FLAGS_device);

  auto model = factory->NewModel(def);
  {
    absl::MutexLock lock(&mutex_);
    latest_model_name_ = model->name();
  }
  models_.Push(std::move(model));
  for (int i = 1; i < FLAGS_parallel_inference; ++i) {
    models_.Push(factory->NewModel(def));
  }
}

void Selfplayer::CheckAbortFile() {
  if (file::FileExists(FLAGS_abort_file)) {
    MG_LOG(FATAL) << "Aborting because " << FLAGS_abort_file << " was found";
  }
}

SelfplayThread::SelfplayThread(int thread_id, Selfplayer* selfplayer,
                               std::shared_ptr<InferenceCache> cache)
    : Thread(absl::StrCat("Selfplay:", thread_id)),
      selfplayer_(selfplayer),
      cache_(std::move(cache)),
      thread_id_(thread_id) {
  selfplay_games_.resize(FLAGS_concurrent_games_per_thread);
}

void SelfplayThread::Run() {
  WTF_THREAD_ENABLE("SelfplayThread");

  searches_.resize(FLAGS_parallel_search);
  while (!selfplay_games_.empty()) {
    StartNewGames();
    SelectLeaves();
    auto model_name = RunInferences();
    ProcessInferences(model_name);
    PlayMoves();
  }

  MG_LOG(INFO) << "SelfplayThread " << thread_id_ << " played "
               << num_games_finished_ << " games";
}

void SelfplayThread::StartNewGames() {
  WTF_SCOPE0("StartNewGames");
  for (size_t i = 0; i < selfplay_games_.size();) {
    if (selfplay_games_[i] == nullptr) {
      // The i'th element is null, either start a new game, or remove the
      // element from the `selfplay_games_` array.
      bool verbose = FLAGS_verbose && thread_id_ == 0 && i == 0;
      auto selfplay_game = selfplayer_->StartNewGame(verbose);
      if (selfplay_game == nullptr) {
        // There are no more games to play remove the empty i'th slot from the
        // array. To do this without having to shuffle all the elements down,
        // we move the last element into position i and pop off the back. After
        // doing this, go round the loop again without incrementing i (otherwise
        // we'd skip over the newly moved element).
        selfplay_games_[i] = std::move(selfplay_games_.back());
        selfplay_games_.pop_back();
        continue;
      } else {
        selfplay_games_[i] = std::move(selfplay_game);
      }
    }
    // We didn't remove an element from the array, iterate as normal.
    i += 1;
  }
}

void SelfplayThread::SelectLeaves() {
  WTF_SCOPE("SelectLeaves: games", size_t)(selfplay_games_.size());

  std::atomic<size_t> game_idx(0);
  selfplayer_->ExecuteSharded([this, &game_idx](int shard_idx, int num_shards) {
    WTF_SCOPE0("SelectLeaf");
    MG_CHECK(static_cast<size_t>(num_shards) == searches_.size());

    SelfplayGame::SelectLeavesStats total_stats;

    auto& search = searches_[shard_idx];
    search.Clear();

    for (;;) {
      auto i = game_idx.fetch_add(1);
      if (i >= selfplay_games_.size()) {
        break;
      }

      TreeSearch::InferenceSpan span;
      span.selfplay_game = selfplay_games_[i].get();
      span.pos = search.inferences.size();
      auto stats = span.selfplay_game->SelectLeaves(cache_.get(),
                                                    &search.inferences);
      span.len = stats.num_leaves_queued;
      if (span.len > 0) {
        search.inference_spans.push_back(span);
      }
      total_stats += stats;
    }

    WTF_APPEND_SCOPE("leaves, nodes, cache_hits, game_over", int, int, int, int)
    (total_stats.num_leaves_queued, total_stats.num_nodes_selected,
     total_stats.num_cache_hits, total_stats.num_game_over_leaves);
  });
}

std::string SelfplayThread::RunInferences() {
  WTF_SCOPE0("RunInferences");

  // TODO(tommadams): stop allocating theses temporary vectors.
  std::vector<const ModelInput*> input_ptrs;
  std::vector<ModelOutput*> output_ptrs;
  for (auto& s : searches_) {
    for (auto& x : s.inferences) {
      input_ptrs.push_back(&x.input);
      output_ptrs.push_back(&x.output);
    }
  }

  if (input_ptrs.empty()) {
    return {};
  }

  std::string model_name;
  auto model = selfplayer_->AcquireModel();
  model->RunMany(input_ptrs, &output_ptrs, nullptr);
  model_name = model->name();
  selfplayer_->ReleaseModel(std::move(model));
  return model_name;
}

void SelfplayThread::ProcessInferences(const std::string& model_name) {
  {
    WTF_SCOPE0("UpdateCache");
    for (auto& s : searches_) {
      for (auto& inference : s.inferences) {
        cache_->Merge(inference.cache_key, inference.leaf->canonical_symmetry,
                      inference.input.sym, &inference.output);
      }
    }
  }

  {
    WTF_SCOPE0("ProcessInferences");
    for (auto& s : searches_) {
      for (const auto& span : s.inference_spans) {
        span.selfplay_game->ProcessInferences(
            model_name,
            absl::MakeSpan(s.inferences).subspan(span.pos, span.len));
      }
    }
  }
}

void SelfplayThread::PlayMoves() {
  WTF_SCOPE0("PlayMoves");

  for (auto& selfplay_game : selfplay_games_) {
    if (!selfplay_game->MaybePlayMove()) {
      continue;
    }
    if (selfplay_game->options().verbose && FLAGS_cache_size_mb > 0) {
      MG_LOG(INFO) << "Inference cache stats: " << cache_->GetStats();
    }
    if (selfplay_game->game()->game_over()) {
      selfplayer_->EndGame(std::move(selfplay_game));
      num_games_finished_ += 1;
      selfplay_game = nullptr;
    }
  }
}

OutputThread::OutputThread(
    int thread_id, FeatureDescriptor feature_descriptor,
    ThreadSafeQueue<std::unique_ptr<SelfplayGame>>* output_queue)
    : Thread(absl::StrCat("Output:", thread_id)),
      output_queue_(output_queue),
      output_dir_(FLAGS_output_dir),
      holdout_dir_(FLAGS_holdout_dir),
      sgf_dir_(FLAGS_sgf_dir),
      feature_descriptor_(std::move(feature_descriptor)) {}

void OutputThread::Run() {
  for (;;) {
    auto selfplay_game = output_queue_->Pop();
    if (selfplay_game == nullptr) {
      break;
    }
    WriteOutputs(std::move(selfplay_game));
  }
}

void OutputThread::WriteOutputs(std::unique_ptr<SelfplayGame> selfplay_game) {
  auto now = absl::Now();
  auto output_name = GetOutputName(selfplay_game->game_id());
  auto* game = selfplay_game->game();
  if (FLAGS_verbose) {
    LogEndGameInfo(*game, selfplay_game->duration());
  }

  // Take the player name from the last model used to play a move. This is
  // done because the ml_perf RL loop waits for a certain number of games to
  // be played by a model before training a new one. By assigned a game to
  // the last model used to play a move rather than the first, training waits
  // for less time and so we produce new models more quickly.
  const auto& models_used = selfplay_game->models_used();
  const auto& player_name =
      !models_used.empty() ? models_used.back() : game->black_name();

  if (!sgf_dir_.empty()) {
    WriteSgf(GetOutputDir(now, player_name, file::JoinPath(sgf_dir_, "clean")),
             output_name, *game, false);
    WriteSgf(GetOutputDir(now, player_name, file::JoinPath(sgf_dir_, "full")),
             output_name, *game, true);
  }

  const auto& example_dir =
      selfplay_game->options().is_holdout ? holdout_dir_ : output_dir_;
  if (!example_dir.empty()) {
    tf_utils::WriteGameExamples(GetOutputDir(now, player_name, example_dir),
                                output_name, feature_descriptor_, *game);
  }
}

}  // namespace
}  // namespace minigo

int main(int argc, char* argv[]) {
  minigo::Init(&argc, &argv);
  minigo::zobrist::Init(FLAGS_seed);

  {
    minigo::Selfplayer selfplayer;
    selfplayer.Run();
  }

  minigo::ShutdownModelFactories();

  return 0;
}
