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

#ifndef CC_MINIGUI_PLAYER_H_
#define CC_MINIGUI_PLAYER_H_

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
#include "cc/gtp_player.h"
#include "cc/thread_safe_queue.h"

namespace minigo {

class MiniguiPlayer : public GtpPlayer {
 public:
  MiniguiPlayer(std::unique_ptr<DualNet> network,
                std::unique_ptr<InferenceCache> inference_cache, Game* game,
                const Options& options);

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

  // If waiting for the opponent to play, consider thinking for a bit.
  // Returns true if we pondered.
  void Ponder() override;

  Response HandleCmd(const std::string& line) override;
  Response HandleClearBoard(CmdArgs args) override;
  Response HandleGenmove(CmdArgs args) override;
  Response HandleLoadsgf(CmdArgs args) override;
  Response HandlePlay(CmdArgs args) override;

  Response HandleEcho(CmdArgs args);
  Response HandleInfo(CmdArgs args);
  Response HandlePruneNodes(CmdArgs args);
  Response HandleReportSearchInterval(CmdArgs args);
  Response HandleSelectPosition(CmdArgs args);
  Response HandleWinrateEvals(CmdArgs args);

  // Shared implementation used by HandleLoadsgf and HandlePlaysgf.
  Response ProcessSgf(const std::vector<std::unique_ptr<sgf::Node>>& trees);

  // Writes the search data for the tree search being performed at the given
  // root to stderr. If leaf is non-null, the search path from root to leaf
  // is also written.
  void ReportSearchStatus(MctsNode* root, MctsNode* leaf,
                          bool include_tree_stats);

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

  // Map from MctsNode to auxiliary info about that node used by the GtpPlayer.
  absl::flat_hash_map<MctsNode*, std::unique_ptr<AuxInfo>> node_to_info_;

  // Map from unique ID associated with every position played in a game or
  // variation to the position's auxiliary info and MctsNode.
  absl::flat_hash_map<std::string, AuxInfo*> id_to_info_;

  // Queue of positions that require their win rate to be evaluated.
  std::deque<AuxInfo*> to_eval_;

  // Number of times to perform tree search for each position when evaluating
  // its win rate.
  int num_eval_reads_ = 6;

  // Each call to Ponder alternates between performing background win-rate
  // estimation reads (while there are still some remaining) and regular
  // pondering.
  bool do_winrate_eval_reads_ = true;
};

}  // namespace minigo

#endif  // CC_MINIGUI_PLAYER_H_
