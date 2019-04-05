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

#include "cc/minigui_player.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <utility>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/time/clock.h"
#include "cc/constants.h"
#include "cc/file/utils.h"
#include "cc/logging.h"
#include "cc/sgf.h"
#include "nlohmann/json.hpp"

namespace minigo {

MiniguiGtpClient::MiniguiGtpClient(std::unique_ptr<MctsPlayer> player,
                                   const Options& options)
    : GtpClient(std::move(player), options) {
  RegisterCmd("echo", &MiniguiGtpClient::HandleEcho);
  RegisterCmd("genmove", &MiniguiGtpClient::HandleGenmove);
  RegisterCmd("play", &MiniguiGtpClient::HandlePlay);
  RegisterCmd("prune_nodes", &MiniguiGtpClient::HandlePruneNodes);
  RegisterCmd("report_search_interval",
              &MiniguiGtpClient::HandleReportSearchInterval);
  RegisterCmd("select_position", &MiniguiGtpClient::HandleSelectPosition);
  RegisterCmd("winrate_evals", &MiniguiGtpClient::HandleWinrateEvals);

  player_->SetTreeSearchCallback(
      std::bind(&MiniguiGtpClient::TreeSearchCb, this, std::placeholders::_1));
}

MiniguiGtpClient::~MiniguiGtpClient() = default;

void MiniguiGtpClient::NewGame() {
  node_to_info_.clear();
  id_to_info_.clear();
  to_eval_.clear();
  GtpClient::NewGame();
  ReportPosition(player_->root());
}

void MiniguiGtpClient::Ponder() {
  // Decide whether to perform normal pondering or win rate evaluation.
  if (to_eval_.empty()) {
    // Nothing needs win rate evaluation.
    do_winrate_eval_reads_ = false;
  } else if (to_eval_[0]->num_eval_reads == 0) {
    // While there are still nodes in the win rate eval queue that haven't had
    // any reads, always perform win rate evaluation.
    do_winrate_eval_reads_ = true;
  } else {
    // While all nodes have been evaluated at least once, alternate between
    // performing win rate evaluation and normal pondering.
    do_winrate_eval_reads_ = !do_winrate_eval_reads_;
  }

  if (!do_winrate_eval_reads_) {
    GtpClient::Ponder();
    return;
  }

  // Remember the number of reads at the root.
  int n = player_->root()->N();

  // First populate the batch with any nodes that require win rate evaluation.
  std::vector<MctsPlayer::TreePath> paths;
  for (int i = 0; i < player_->options().virtual_losses; ++i) {
    if (to_eval_.empty()) {
      break;
    }
    player_->SelectLeaves(to_eval_.front()->node, 1, &paths);
    to_eval_.pop_front();
  }

  player_->ProcessLeaves(absl::MakeSpan(paths),
                         player_->options().random_symmetry);

  // Send updated visit and Q data for all the nodes we performed win rate
  // evaluation on. This updates Minigui's win rate graph.
  for (const auto& path : paths) {
    auto* root = path.root;
    nlohmann::json j = {
        {"id", GetAuxInfo(root)->id},
        {"n", root->N()},
        {"q", root->Q()},
    };
    MG_LOG(INFO) << "mg-update:" << j.dump();
  }

  // Increment the ponder count by difference new and old reads.
  ponder_read_count_ += player_->root()->N() - n;

  // Increment the number of reads for all the nodes we performed win rate
  // evaluation on, pushing nodes that require more reads onto the back of the
  // queue.
  for (const auto& path : paths) {
    auto* info = GetAuxInfo(path.root);
    if (++info->num_eval_reads < num_eval_reads_) {
      to_eval_.push_back(info);
    }
  }
}

GtpClient::Response MiniguiGtpClient::HandleCmd(const std::string& line) {
  auto response = GtpClient::HandleCmd(line);
  // Write __GTP_CMD_DONE__ to stderr to signify that handling a GTP command is
  // done. The Minigui Python server waits for this magic string before it
  // consumes the output of each GTP command. This keeps the outputs written to
  // stderr and stdout synchronized so that all data written to stderr while
  // processing a GTP command is consumed before the GTP result written to
  // stdout.
  MG_LOG(INFO) << "__GTP_CMD_DONE__";
  return response;
}

GtpClient::Response MiniguiGtpClient::HandleEcho(CmdArgs args) {
  return Response::Ok(absl::StrJoin(args, " "));
}

GtpClient::Response MiniguiGtpClient::HandleGenmove(CmdArgs args) {
  ReportSearchStatus(player_->root(), nullptr, true);
  auto response = GtpClient::HandleGenmove(args);
  if (response.ok) {
    ReportPosition(player_->root());
  }
  RefreshPendingWinRateEvals();
  return response;
}

GtpClient::Response MiniguiGtpClient::HandlePlay(CmdArgs args) {
  auto response = GtpClient::HandlePlay(args);
  if (response.ok) {
    ReportPosition(player_->root());
  }
  return response;
}

GtpClient::Response MiniguiGtpClient::HandlePruneNodes(CmdArgs args) {
  auto response = CheckArgsExact(1, args);
  if (!response.ok) {
    return response;
  }

  int x;
  if (!absl::SimpleAtoi(args[0], &x)) {
    return Response::Error("couldn't parse ", args[0], " as an integer");
  }

  auto options = player_->options();
  options.prune_orphaned_nodes = x != 0;
  player_->SetOptions(options);

  return Response::Ok();
}

GtpClient::Response MiniguiGtpClient::HandleReportSearchInterval(CmdArgs args) {
  auto response = CheckArgsExact(1, args);
  if (!response.ok) {
    return response;
  }

  int x;
  if (!absl::SimpleAtoi(args[0], &x) || x < 0) {
    return Response::Error("couldn't parse ", args[0], " as an integer >= 0");
  }
  report_search_interval_ = absl::Milliseconds(x);

  return Response::Ok();
}

GtpClient::Response MiniguiGtpClient::HandleSelectPosition(CmdArgs args) {
  auto response = CheckArgsExact(1, args);
  if (!response.ok) {
    return response;
  }

  if (args[0] == "root") {
    player_->ResetRoot();
    return Response::Ok();
  }

  auto it = id_to_info_.find(args[0]);
  if (it == id_to_info_.end()) {
    return Response::Error("unknown position id");
  }
  auto* node = it->second->node;

  // Build the sequence of moves the will end up at the requested position.
  std::vector<Coord> moves;
  while (node->parent != nullptr) {
    moves.push_back(node->move);
    node = node->parent;
  }
  std::reverse(moves.begin(), moves.end());

  // Rewind to the start & play the sequence of moves.
  player_->ResetRoot();
  for (const auto& move : moves) {
    MG_CHECK(player_->PlayMove(move));
  }

  RefreshPendingWinRateEvals();

  return Response::Ok();
}

GtpClient::Response MiniguiGtpClient::HandleWinrateEvals(CmdArgs args) {
  int num_reads;
  if (!absl::SimpleAtoi(args[1], &num_reads) || num_reads < 0) {
    return Response::Error("invalid num_reads");
  }
  num_eval_reads_ = num_reads;
  RefreshPendingWinRateEvals();
  return Response::Ok();
}

GtpClient::Response MiniguiGtpClient::ReplaySgf(
    const std::vector<std::unique_ptr<sgf::Node>>& trees) {
  // Traverse the SGF's game trees, loading them into the backend & running
  // inference on the positions in batches.
  std::function<Response(const sgf::Node&)> traverse =
      [&](const sgf::Node& node) {
        if (node.move.color != player_->root()->position.to_play()) {
          // The move color is different than expected. Play a pass move to flip
          // the colors.
          if (player_->root()->move == Coord::kPass) {
            auto expected = ColorToCode(player_->root()->position.to_play());
            auto actual = node.move.ToSgf();
            MG_LOG(ERROR) << "expected move by " << expected << ", got "
                          << actual
                          << " but can't play an intermediate pass because the"
                          << " previous move was also a pass";
            return Response::Error("cannot load file");
          }
          MG_LOG(WARNING) << "Inserting pass move";
          MG_CHECK(player_->PlayMove(Coord::kPass));
          ReportPosition(player_->root());
        }

        if (!player_->PlayMove(node.move.c)) {
          MG_LOG(ERROR) << "error playing " << node.move.ToSgf();
          return Response::Error("cannot load file");
        }

        if (!node.comment.empty()) {
          auto* info = GetAuxInfo(player_->root());
          info->comment = node.comment;
        }

        ReportPosition(player_->root());
        for (const auto& child : node.children) {
          auto response = traverse(*child);
          if (!response.ok) {
            return response;
          }
        }
        player_->UndoMove();
        return Response::Ok();
      };

  for (const auto& tree : trees) {
    auto response = traverse(*tree);
    if (!response.ok) {
      return response;
    }
  }

  // Play the main line.
  player_->ResetRoot();
  if (!trees.empty()) {
    for (const auto& move : trees[0]->ExtractMainLine()) {
      // We already validated that all the moves could be played in traverse(),
      // so if PlayMove fails here, something has gone seriously awry.
      MG_CHECK(player_->PlayMove(move.c));
    }
    ReportPosition(player_->root());
  }

  return Response::Ok();
}

void MiniguiGtpClient::ReportSearchStatus(MctsNode* root, MctsNode* leaf,
                                          bool include_tree_stats) {
  auto sorted_child_info = root->CalculateRankedChildInfo();

  nlohmann::json j = {
      {"id", GetAuxInfo(root)->id},
      {"n", root->N()},
      {"q", root->Q()},
  };

  // TODO(tommadams): Make the number of child variations sent back
  // configurable.
  nlohmann::json variations;
  for (int i = 0; i < 10; ++i) {
    Coord c = sorted_child_info[i].c;
    const auto child_it = root->children.find(c);
    if (child_it == root->children.end() || root->child_N(c) == 0) {
      break;
    }

    nlohmann::json moves = {c.ToGtp()};
    const auto* node = child_it->second.get();
    for (const auto c : node->MostVisitedPath()) {
      moves.push_back(c.ToGtp());
    }
    variations[c.ToGtp()] = {
        {"n", root->child_N(c)},
        {"q", root->child_Q(c)},
        {"moves", std::move(moves)},
    };
  }
  if (!variations.empty()) {
    j["variations"] = std::move(variations);
  }

  // Current live search variation.
  if (leaf != nullptr) {
    std::vector<const MctsNode*> live;
    for (const auto* node = leaf; node != root; node = node->parent) {
      live.push_back(node);
    }
    if (!live.empty()) {
      std::reverse(live.begin(), live.end());
      nlohmann::json moves;
      for (const auto* node : live) {
        moves.push_back(node->move.ToGtp());
      }
      j["variations"]["live"] = {
          {"n", live.front()->N()},
          {"q", live.front()->Q()},
          {"moves", std::move(moves)},
      };
    }
  }

  // Child N.
  auto& child_N = j["childN"];
  for (const auto& edge : root->edges) {
    child_N.push_back(static_cast<int>(edge.N));
  }

  // Child Q.
  auto& child_Q = j["childQ"];
  for (int i = 0; i < kNumMoves; ++i) {
    child_Q.push_back(static_cast<int>(std::round(root->child_Q(i) * 1000)));
  }

  if (include_tree_stats) {
    auto tree_stats = root->CalculateTreeStats();
    j["treeStats"] = {
        {"numNodes", tree_stats.num_nodes},
        {"numLeafNodes", tree_stats.num_leaf_nodes},
        {"maxDepth", tree_stats.max_depth},
    };
  }

  MG_LOG(INFO) << "mg-update:" << j.dump();
}

void MiniguiGtpClient::ReportPosition(MctsNode* node) {
  const auto& position = node->position;

  std::ostringstream oss;
  for (const auto& stone : position.stones()) {
    char ch;
    if (stone.color() == Color::kBlack) {
      ch = 'X';
    } else if (stone.color() == Color::kWhite) {
      ch = 'O';
    } else {
      ch = '.';
    }
    oss << ch;
  }

  auto* info = GetAuxInfo(node);
  nlohmann::json j = {
      {"id", info->id},
      {"toPlay", position.to_play() == Color::kBlack ? "B" : "W"},
      {"moveNum", position.n()},
      {"stones", oss.str()},
      {"gameOver", node->game_over()},
  };

  const auto& captures = node->position.num_captures();
  if (captures[0] != 0 || captures[1] != 0) {
    j["caps"].push_back(captures[0]);
    j["caps"].push_back(captures[1]);
  }
  if (node->parent != nullptr) {
    j["parentId"] = GetAuxInfo(node->parent)->id;
    if (node->N() > 0) {
      // Only send Q if the node has been read at least once.
      j["q"] = node->Q();
    }
  }
  if (node->move != Coord::kInvalid) {
    j["move"] = node->move.ToGtp();
  }
  if (!info->comment.empty()) {
    j["comment"] = info->comment;
  }

  MG_LOG(INFO) << "mg-position: " << j.dump();
}

MiniguiGtpClient::AuxInfo* MiniguiGtpClient::GetAuxInfo(MctsNode* node) {
  auto it = node_to_info_.find(node);
  if (it != node_to_info_.end()) {
    return it->second.get();
  }

  auto* parent = node->parent != nullptr ? GetAuxInfo(node->parent) : nullptr;
  auto info = absl::make_unique<AuxInfo>(parent, node);
  auto raw_info = info.get();
  id_to_info_.emplace(info->id, raw_info);
  node_to_info_.emplace(node, std::move(info));
  return raw_info;
}

MiniguiGtpClient::AuxInfo::AuxInfo(AuxInfo* parent, MctsNode* node)
    : parent(parent), node(node), id(absl::StrFormat("%p", node)) {
  if (parent != nullptr) {
    parent->children.push_back(this);
  }
}

void MiniguiGtpClient::RefreshPendingWinRateEvals() {
  to_eval_.clear();

  // Build a new list of nodes that require win rate evaluation.
  // First, traverse to the leaf node of the current position's main line.
  auto* info = GetAuxInfo(player_->root());
  while (!info->children.empty()) {
    info = info->children[0];
  }

  // Walk back up the tree to the root, enqueing all nodes that have fewer than
  // the num_eval_reads_ win rate evaluations.
  while (info != nullptr) {
    if (info->num_eval_reads < num_eval_reads_) {
      to_eval_.push_back(info);
    }
    info = info->parent;
  }

  // Sort the nodes for eval by number of eval reads, breaking ties by the move
  // number.
  std::sort(to_eval_.begin(), to_eval_.end(), [](AuxInfo* a, AuxInfo* b) {
    if (a->num_eval_reads != b->num_eval_reads) {
      return a->num_eval_reads < b->num_eval_reads;
    }
    return a->node->position.n() < b->node->position.n();
  });
}

void MiniguiGtpClient::TreeSearchCb(absl::Span<MctsPlayer::TreePath> paths) {
  if (!paths.empty() && report_search_interval_ != absl::ZeroDuration()) {
    auto now = absl::Now();
    if (now - last_report_time_ > report_search_interval_) {
      last_report_time_ = now;
      ReportSearchStatus(paths.back().root, paths.back().leaf, false);
    }
  }
}

}  // namespace minigo
