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

#ifndef CC_GTP_PLAYER_H_
#define CC_GTP_PLAYER_H_

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
#include "absl/types/span.h"
#include "cc/color.h"
#include "cc/dual_net/dual_net.h"
#include "cc/mcts_player.h"
#include "cc/thread_safe_queue.h"

namespace minigo {

class GtpPlayer : public MctsPlayer {
 public:
  struct Options : public MctsPlayer::Options {
    // If non-zero, GtpPlayer will print the current state of its tree search
    // every report_search_interval to stderr in a format recognized by Minigui.
    absl::Duration report_search_interval;

    // Maximum number of times to perform TreeSearch when pondering is enabled.
    // The engine's ponder count is reset to 0 each time it receives a "ponder"
    // GTP command.
    int ponder_limit = 0;

    // If true, we will always pass if the opponent passes.
    bool courtesy_pass = false;

    // Number of times to perform tree search for each position in an SGF in
    // order to evalute a win rate estimation.
    int num_eval_reads = 4;
  };

  GtpPlayer(std::unique_ptr<DualNet> network, const Options& options);

  void Run();

  void NewGame() override;
  Coord SuggestMove() override;
  bool PlayMove(Coord c) override;

 protected:
  void ProcessLeaves(absl::Span<TreePath> paths, bool random_symmetry) override;

 private:
  // We maintain some auxiliary data structures about nodes in the search tree
  // that correspond to actual positions played.
  struct AuxInfo {
    AuxInfo(AuxInfo* parent, MctsNode* node);

    // Parent in the game tree.
    // This is a shortcut for the following expression:
    //   node->parent != nullptr ? GetAuxInfo(node->parent) : nullptr
    AuxInfo* parent;

    // Tree search node.
    MctsNode* node;

    // Unique ID.
    std::string id;

    // Number of times we have performed tree search for win rate evaluation for
    // this position. This is tracked separately from MctsNode.N to that every
    // position requiring win rate evaluation is evaluated as a tree search
    // root, regardless of what the "real" tree search is doing.
    int num_eval_reads = 0;

    // Children of this position. These are stored in order that the positions
    // were played (MctsNode::children is unordered), so that the chain of
    // descendants from this position formed by children[0] is the position's
    // main line. Children at index 1 and later are variations from the main
    // line.
    std::vector<AuxInfo*> children;

    // Any SGF comments associated with this position.
    std::string comment;
  };

  // Response from the GTP command handler.
  struct Response {
    static Response Ok(std::string str = "") { return {std::move(str), true}; }

    template <typename... Args>
    static Response Error(const Args&... args) {
      return {absl::StrCat(args...), false};
    }

    std::string str;
    bool ok;
  };

  using CmdArgs = const std::vector<absl::string_view>&;
  using CmdHandler = Response (GtpPlayer::*)(absl::string_view, CmdArgs);
  void RegisterCmd(const std::string& cmd, CmdHandler handler);

  // If waiting for the opponent to play, consider thinking for a bit.
  // Returns true if we pondered.
  bool MaybePonder();

  // Handles a GTP command specified by `line`, printing the result to stdout.
  // Returns false if the GtpPlayer should quit.
  bool HandleCmd(const std::string& line);

  Response CheckArgsExact(absl::string_view cmd, size_t expected_num_args,
                          CmdArgs args);
  Response CheckArgsRange(absl::string_view cmd, size_t expected_min_args,
                          size_t expected_max_args, CmdArgs args);

  Response DispatchCmd(const std::string& cmd, CmdArgs args);

  // TODO(tommadams): clearly document these methods w.r.t. the GTP standard and
  // what public methods they call.
  Response HandleBenchmark(absl::string_view cmd, CmdArgs args);
  Response HandleBoardsize(absl::string_view cmd, CmdArgs args);
  Response HandleClearBoard(absl::string_view cmd, CmdArgs args);
  Response HandleEcho(absl::string_view cmd, CmdArgs args);
  Response HandleFinalScore(absl::string_view cmd, CmdArgs args);
  Response HandleGenmove(absl::string_view cmd, CmdArgs args);
  Response HandleInfo(absl::string_view cmd, CmdArgs args);
  Response HandleKnownCommand(absl::string_view cmd, CmdArgs args);
  Response HandleKomi(absl::string_view cmd, CmdArgs args);
  Response HandleListCommands(absl::string_view cmd, CmdArgs args);
  Response HandleLoadsgf(absl::string_view cmd, CmdArgs args);
  Response HandleName(absl::string_view cmd, CmdArgs args);
  Response HandlePlay(absl::string_view cmd, CmdArgs args);
  Response HandlePlaysgf(absl::string_view cmd, CmdArgs args);
  Response HandlePonder(absl::string_view cmd, CmdArgs args);
  Response HandlePruneNodes(absl::string_view cmd, CmdArgs args);
  Response HandleReadouts(absl::string_view cmd, CmdArgs args);
  Response HandleReportSearchInterval(absl::string_view cmd, CmdArgs args);
  Response HandleSelectPosition(absl::string_view cmd, CmdArgs args);
  Response HandleUndo(absl::string_view cmd, CmdArgs args);
  Response HandleVariation(absl::string_view cmd, CmdArgs args);
  Response HandleVerbosity(absl::string_view cmd, CmdArgs args);

  // Shared implementation used by HandleLoadsgf and HandlePlaysgf.
  Response ParseSgf(const std::string& sgf_str);

  // Writes the search data for the tree search being performed at the given
  // root to stderr. If leaf is non-null, the search path from root to leaf
  // is also written.
  void ReportSearchStatus(MctsNode* root, MctsNode* leaf);

  // Writes the position data for the node to stderr as a JSON object.
  void ReportPosition(MctsNode* node);

  // Registers the given node as having been played during the game,
  // assigning the node a unique ID and constructing AuxInfo for it.
  AuxInfo* RegisterNode(MctsNode* node);

  // Gets the AuxInfo for the given node.
  // CHECK fails if there isn't any AuxInfo, which means that RegisterNode
  // hasn't previously been called: this node doesn't correspond to a move
  // played during the game or a variation (it's a node from tree search).
  AuxInfo* GetAuxInfo(MctsNode* node) const;

  // Clears the to_eval_ win rate evaluation queue and repopulates it.
  void RefreshPendingWinRateEvals();

  bool courtesy_pass_;
  absl::Duration report_search_interval_;
  absl::Time last_report_time_;

  // There are two kinds of pondering supported:
  //   kReadLimited: pondering will run for a maximum number of reads.
  //   kTimeLimited: pondering will run for a maximum number of seconds.
  enum class PonderType {
    kOff,
    kReadLimited,
    kTimeLimited,
  };
  PonderType ponder_type_ = PonderType::kOff;
  int ponder_read_count_ = 0;
  int ponder_read_limit_ = 0;
  absl::Duration ponder_duration_ = {};
  absl::Time ponder_time_limit_ = absl::InfinitePast();
  bool ponder_limit_reached_ = false;

  absl::flat_hash_map<std::string, CmdHandler> cmd_handlers_;

  // Controls which variation is reported during tree search.
  // If child_variation_ == Coord::kInvalid, the principle variation from the
  // root is reported. Otherwise, the principle variation of the
  // corresponding child of the root is reported.
  Coord child_variation_ = Coord::kInvalid;

  // Map from MctsNode to auxiliary info about that node used by the GtpPlayer.
  absl::flat_hash_map<MctsNode*, std::unique_ptr<AuxInfo>> node_to_info_;

  // Map from unique ID associated with every position played in a game or
  // variation to the position's auxiliary info and MctsNode.
  absl::flat_hash_map<std::string, AuxInfo*> id_to_info_;

  // Queue of positions that require their win rate to be evaluated.
  std::deque<AuxInfo*> to_eval_;

  // Number of times to perform tree search for each position when evaluating
  // its win rate.
  int num_eval_reads_;

  ThreadSafeQueue<std::string> stdin_queue_;
};

}  // namespace minigo

#endif  // CC_GTP_PLAYER_H_
