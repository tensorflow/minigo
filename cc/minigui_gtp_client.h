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

#ifndef CC_MINIGUI_GTP_CLIENT_H_
#define CC_MINIGUI_GTP_CLIENT_H_

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "cc/async/thread.h"
#include "cc/async/thread_safe_queue.h"
#include "cc/color.h"
#include "cc/gtp_client.h"
#include "cc/model/batching_model.h"
#include "cc/model/model.h"

namespace minigo {

class MiniguiGtpClient : public GtpClient {
 public:
  MiniguiGtpClient(std::string device,
                   std::shared_ptr<ThreadSafeInferenceCache> inference_cache,
                   const std::string& model_path,
                   const Game::Options& game_options,
                   const MctsPlayer::Options& player_options,
                   const GtpClient::Options& client_options);
  ~MiniguiGtpClient() override;

  void NewGame() override;

 private:
  // A tree that tracks all variations played during one game.
  // This tree is persistent throughout a game, unlike the tree used for search.
  // The VariationTree maintains a current_node, which is always updated so that
  // it always corresponds to the MctsPlayer's root.
  class VariationTree {
   public:
    struct Node {
      Node(Node* parent, Coord move);

      Node(const Node&) = delete;
      Node& operator=(const Node&) = delete;

      Node(Node&&) = default;
      Node& operator=(Node&&) = default;

      // Returns the sequence of moves required to reach this node, starting
      // from an empty board.
      std::vector<Coord> GetVariation() const;

      Node* parent;
      Coord move;
      std::string id;
      int n = 0;

      // Number of times we have performed tree search for win rate evaluation
      // for this position. This is tracked separately from MctsNode.N to that
      // every position requiring win rate evaluation is evaluated as a tree
      // search root, regardless of what the "real" tree search is doing.
      int num_eval_reads = 0;

      // Children of this position. These are stored in order that the positions
      // were played (MctsNode::children is unordered), so that the chain of
      // descendants from this position formed by children[0] is the position's
      // main line. Children at index 1 and later are variations from the main
      // line.
      std::vector<Node*> children;

      // Any SGF comments associated with this position.
      std::string comment;
    };

    VariationTree();

    // Plays the given move from the current position, updating current_node.
    void PlayMove(Coord c);

    // Update current_node to its parent.
    void GoToParent();

    // Rewind the current_node all the way back to the starting empty board
    // state.
    void GoToStart();

    // Sets current_node to the node with the given id if it exists.
    // Returns true if the node with a matching id was found.
    bool SelectNode(const std::string& id);

    // Returns the current node in the tree.
    Node* current_node() { return current_node_; }

   private:
    Node* current_node_ = nullptr;
    absl::flat_hash_map<std::string, std::unique_ptr<Node>> id_map_;
  };

  // The WinRateEvaluator handles performing win rate evaluation for positions
  // that run in parellel with conventionaly pondering.
  // For more accurate win rate evaluation, the WinRateEvaluator doesn't use
  // virtual losses and runs inference on one leaf at a time. To improve
  // efficiency, the WinRateEvaluator uses multiple MctsPlayers that all perform
  // tree search in parallel, with each player performing win rate evaluation
  // for a different position.
  // It's a little depressing how much extra code had to be written to support
  // background win rate evaluation, but there we are. Things would be a lot
  // simpler if we had a nice fiber library.
  class WinRateEvaluator {
   public:
    WinRateEvaluator(int num_workers, int num_eval_reads,
                     const std::string& device,
                     std::shared_ptr<ThreadSafeInferenceCache> inference_cache,
                     const std::string& model_path,
                     const Game::Options& game_options,
                     const MctsPlayer::Options& player_options);
    ~WinRateEvaluator();

    bool all_nodes_have_at_least_one_read() const {
      return to_eval_.empty() || to_eval_[0]->num_eval_reads > 0;
    }

    void SetNumEvalReads(int num_eval_reads);
    void SetCurrentVariation(std::vector<VariationTree::Node*> nodes);
    void EvalNodes();

    // private:
    void UpdateNodesToEval();

    class Worker : public Thread {
     public:
      Worker(std::unique_ptr<Game> game, std::unique_ptr<MctsPlayer> player,
             ThreadSafeQueue<VariationTree::Node*>* eval_queue);
      ~Worker();

      // Prepare the worker for running evaluation.
      // Prepare should be called on all workers that are about to perform eval
      // before the Eval calls. This tells the inference batcher shared
      // between workers how many inferences to expect.
      void Prepare();

      // Start running evaluation on the given Node.
      // Each time Eval is called, the Worker resets the board and plays a
      // game out to the given node, then performs tree search until a single
      // inference is performed.
      // Because the workers all share an inference cache, repeated Evals of
      // the same node will result in an ever increasing number of nodes being
      // expanded during the tree search, since cached hits don't count towards
      // the number of inferences performed during the search.
      // This does require that the inference cache is large enough to fit all
      // the inferences performed during win rate evaluation for the current
      // variation.
      void EvalAsync(VariationTree::Node* node);

     private:
      void Run() override;

      bool has_pending_value() const EXCLUSIVE_LOCKS_REQUIRED(&mutex_) {
        return pending_.has_value();
      }

      absl::Mutex mutex_;
      absl::optional<VariationTree::Node*> pending_ GUARDED_BY(&mutex_);
      std::unique_ptr<Game> game_;
      std::unique_ptr<MctsPlayer> player_ GUARDED_BY(&mutex_);
      std::vector<MctsNode*> leaves_;
      ThreadSafeQueue<VariationTree::Node*>* eval_queue_;
    };

    int num_eval_reads_ = 8;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::deque<VariationTree::Node*> to_eval_;
    std::vector<VariationTree::Node*> variation_;
    ThreadSafeQueue<VariationTree::Node*> eval_queue_;
    std::unique_ptr<BatchingModelFactory> batcher_;
  };

  void Ponder() override;

  // GTP command handlers.
  Response HandleCmd(const std::string& line) override;
  Response HandleGenmove(CmdArgs args) override;
  Response HandlePlay(CmdArgs args) override;
  Response ReplaySgf(const sgf::Collection& collection) override;

  Response HandleEcho(CmdArgs args);
  Response HandleReportSearchInterval(CmdArgs args);
  Response HandleSelectPosition(CmdArgs args);
  Response HandleWinrateEvals(CmdArgs args);

  // Writes the search data for the tree search currently being performed to
  // stderr. If leaf is non-null, the search path from root to leaf is also
  // written.
  void ReportSearchStatus(const MctsNode* leaf, bool include_tree_stats);

  // Writes the position data for the node to stderr as a JSON object.
  void ReportRootPosition();

  // Clears the to_eval_ win rate evaluation queue and repopulates it.
  void RefreshPendingWinRateEvals();

  // Callback invoked during the main tree search (not any of the
  // WinRateEvaluator's searches). Calls ReportSearchStatus if the time since
  // the time it called ReportSearchStatus is greater than
  // report_search_interval_.
  void TreeSearchCb(const std::vector<const MctsNode*>& leaves);

  absl::Duration report_search_interval_;
  absl::Time last_report_time_;
  std::unique_ptr<VariationTree> variation_tree_;
  std::unique_ptr<WinRateEvaluator> win_rate_evaluator_;
};

}  // namespace minigo

#endif  // CC_MINIGUI_GTP_CLIENT_H_
