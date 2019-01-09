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

#include "cc/gtp_player.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <thread>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/time/clock.h"
#include "cc/constants.h"
#include "cc/logging.h"
#include "cc/sgf.h"
#include "nlohmann/json.hpp"

namespace minigo {

namespace {
// String written to stderr to signify that handling a GTP command is done.
constexpr auto kGtpCmdDone = "__GTP_CMD_DONE__";
}  // namespace

GtpPlayer::GtpPlayer(std::unique_ptr<DualNet> network, const Options& options)
    : MctsPlayer(std::move(network), options),
      courtesy_pass_(options.courtesy_pass),
      ponder_read_limit_(options.ponder_limit),
      num_eval_reads_(options.num_eval_reads),
      game_(options.name, options.name, options.game_options) {
  if (ponder_read_limit_ > 0) {
    ponder_type_ = PonderType::kReadLimited;
  }
  RegisterCmd("benchmark", &GtpPlayer::HandleBenchmark);
  RegisterCmd("boardsize", &GtpPlayer::HandleBoardsize);
  RegisterCmd("clear_board", &GtpPlayer::HandleClearBoard);
  RegisterCmd("echo", &GtpPlayer::HandleEcho);
  RegisterCmd("final_score", &GtpPlayer::HandleFinalScore);
  RegisterCmd("genmove", &GtpPlayer::HandleGenmove);
  RegisterCmd("info", &GtpPlayer::HandleInfo);
  RegisterCmd("known_command", &GtpPlayer::HandleKnownCommand);
  RegisterCmd("komi", &GtpPlayer::HandleKomi);
  RegisterCmd("list_commands", &GtpPlayer::HandleListCommands);
  RegisterCmd("loadsgf", &GtpPlayer::HandleLoadsgf);
  RegisterCmd("name", &GtpPlayer::HandleName);
  RegisterCmd("play", &GtpPlayer::HandlePlay);
  RegisterCmd("playsgf", &GtpPlayer::HandlePlaysgf);
  RegisterCmd("ponder", &GtpPlayer::HandlePonder);
  RegisterCmd("prune_nodes", &GtpPlayer::HandlePruneNodes);
  RegisterCmd("readouts", &GtpPlayer::HandleReadouts);
  RegisterCmd("report_search_interval", &GtpPlayer::HandleReportSearchInterval);
  RegisterCmd("select_position", &GtpPlayer::HandleSelectPosition);
  RegisterCmd("undo", &GtpPlayer::HandleUndo);
  RegisterCmd("variation", &GtpPlayer::HandleVariation);
  RegisterCmd("verbosity", &GtpPlayer::HandleVerbosity);
  NewGame();
}

void GtpPlayer::Run() {
  MG_LOG(INFO) << "GTP engine ready";

  // Start a background thread that pushes lines read from stdin into the
  // thread safe stdin_queue_. This allows us to ponder when there's nothing
  // to read from stdin.
  std::atomic<bool> running(true);
  std::thread stdin_thread([this, &running]() {
    std::string line;
    while (std::cin) {
      std::getline(std::cin, line);
      stdin_queue_.Push(line);
    }
    running = false;
  });

  while (running) {
    std::string line;

    // If there's a command waiting on stdin, process it.
    if (stdin_queue_.TryPop(&line)) {
      if (!HandleCmd(line)) {
        break;
      }
      continue;
    }

    // Otherwise, ponder if enabled.
    if (!MaybePonder()) {
      // If pondering isn't enabled, try and pop a command from stdin with a
      // short timeout. The timeout gives us a chance to break out of the loop
      // when stdin is closed with ctrl-C.
      if (stdin_queue_.PopWithTimeout(&line, absl::Seconds(1))) {
        if (!HandleCmd(line)) {
          break;
        }
      }
    }
  }

  stdin_thread.join();
}

void GtpPlayer::NewGame() {
  node_to_info_.clear();
  id_to_info_.clear();
  to_eval_.clear();
  MctsPlayer::NewGame();
  RegisterNode(root());
}

Coord GtpPlayer::SuggestMove() {
  if (courtesy_pass_ && root()->move == Coord::kPass) {
    return Coord::kPass;
  }
  return MctsPlayer::SuggestMove();
}

bool GtpPlayer::PlayMove(Coord c, Game* game) {
  if (!MctsPlayer::PlayMove(c, game)) {
    return false;
  }
  RefreshPendingWinRateEvals();
  return true;
}

void GtpPlayer::RegisterCmd(const std::string& cmd, CmdHandler handler) {
  cmd_handlers_[cmd] = handler;
}

bool GtpPlayer::MaybePonder() {
  if (root()->game_over() || ponder_type_ == PonderType::kOff ||
      ponder_limit_reached_) {
    return false;
  }

  // Check if we're finished pondering.
  if ((ponder_type_ == PonderType::kReadLimited &&
       ponder_read_count_ >= ponder_read_limit_) ||
      (ponder_type_ == PonderType::kTimeLimited &&
       absl::Now() >= ponder_time_limit_)) {
    if (!ponder_limit_reached_) {
      MG_LOG(INFO) << "mg-ponder: done";
      ponder_limit_reached_ = true;
    }
    return false;
  }

  // Remember the number of reads at the root.
  int n = root()->N();

  // First populate the batch with any nodes that require win rate evaluation.
  std::vector<TreePath> paths;
  while (!to_eval_.empty()) {
    auto* info = to_eval_.front();
    int eval_limit = options().batch_size;
    // While there are still nodes in the win rate eval queue that haven't had
    // any reads, use all the available reads in the batch to perform win rate
    // evaluation. Otherwise use up to 50% of each batch for win rate
    // evaluation.
    if (info->num_eval_reads > 0) {
      eval_limit /= 2;
    }
    if (static_cast<int>(paths.size()) >= eval_limit) {
      break;
    }
    to_eval_.pop_front();
    SelectLeaves(info->node, 1, &paths);
  }

  // While there is still space left in the batch, perform regular tree search.
  int num_eval_reads = static_cast<int>(paths.size());
  int num_search_reads = options().batch_size - num_eval_reads;
  if (num_search_reads > 0) {
    SelectLeaves(root(), num_search_reads, &paths);
  }

  ProcessLeaves(absl::MakeSpan(paths), options().random_symmetry);

  // Send updated visit and Q data for all the nodes we performed win rate
  // evaluation on. This updates Minigui's win rate graph.
  for (int i = 0; i < num_eval_reads; ++i) {
    auto* root = paths[i].root;
    nlohmann::json j = {
        {"id", GetAuxInfo(root)->id},
        {"n", root->N()},
        {"q", root->Q()},
    };
    MG_LOG(INFO) << "mg-update:" << j.dump();
  }

  // Increment the ponder count by difference new and old reads.
  ponder_read_count_ += root()->N() - n;

  // Increment the number of reads for all the nodes we performed win rate
  // evaluation on, pushing nodes that require more reads onto the back of the
  // queue.
  for (int i = 0; i < num_eval_reads; ++i) {
    auto* info = GetAuxInfo(paths[i].root);
    if (++info->num_eval_reads < num_eval_reads_) {
      to_eval_.push_back(info);
    }
  }

  return true;
}

bool GtpPlayer::HandleCmd(const std::string& line) {
  std::vector<absl::string_view> args =
      absl::StrSplit(line, absl::ByAnyChar(" \t\r\n"), absl::SkipWhitespace());
  if (args.empty()) {
    MG_LOG(INFO) << kGtpCmdDone;
    std::cout << "=\n\n" << std::flush;
    return true;
  }

  // Split the GTP command and its arguments.
  auto cmd = std::string(args[0]);
  args.erase(args.begin());

  if (cmd == "quit") {
    MG_LOG(INFO) << kGtpCmdDone;
    std::cout << "=\n\n" << std::flush;
    return false;
  }

  auto response = DispatchCmd(cmd, args);
  MG_LOG(INFO) << kGtpCmdDone;
  std::cout << (response.ok ? "=" : "?");
  if (!response.str.empty()) {
    std::cout << " " << response.str;
  }
  std::cout << "\n\n" << std::flush;
  return true;
}

void GtpPlayer::ProcessLeaves(absl::Span<TreePath> paths,
                              bool random_symmetry) {
  MctsPlayer::ProcessLeaves(paths, random_symmetry);
  if (!paths.empty() && report_search_interval_ != absl::ZeroDuration()) {
    auto now = absl::Now();
    if (now - last_report_time_ > report_search_interval_) {
      last_report_time_ = now;
      ReportSearchStatus(paths.back().root, paths.back().leaf);
    }
  }
}

GtpPlayer::Response GtpPlayer::CheckArgsExact(size_t expected_num_args,
                                              CmdArgs args) {
  if (args.size() != expected_num_args) {
    return Response::Error("expected ", expected_num_args, " args, got ",
                           args.size(), " args: ", absl::StrJoin(args, " "));
  }
  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::CheckArgsRange(size_t expected_min_args,
                                              size_t expected_max_args,
                                              CmdArgs args) {
  if (args.size() < expected_min_args || args.size() > expected_max_args) {
    return Response::Error("expected between ", expected_min_args, " and ",
                           expected_max_args, " args, got ", args.size(),
                           " args: ", absl::StrJoin(args, " "));
  }
  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::DispatchCmd(const std::string& cmd,
                                           CmdArgs args) {
  auto it = cmd_handlers_.find(cmd);
  if (it == cmd_handlers_.end()) {
    return Response::Error("unknown command");
  }
  auto handler = it->second;
  return (this->*handler)(args);
}

GtpPlayer::Response GtpPlayer::HandleBenchmark(CmdArgs args) {
  // benchmark [readouts] [batch_size]
  // Note: By default use current time_control (readouts or time).
  auto response = CheckArgsRange(0, 2, args);
  if (!response.ok) {
    return response;
  }

  auto saved_options = options();
  MctsPlayer::Options temp_options = options();

  if (args.size() > 0) {
    temp_options.seconds_per_move = 0;
    if (!absl::SimpleAtoi(args[0], &temp_options.num_readouts)) {
      return Response::Error("bad num_readouts");
    }
  }

  if (args.size() == 2) {
    if (!absl::SimpleAtoi(args[1], &temp_options.batch_size)) {
      return Response::Error("bad batch_size");
    }
  }

  // Set options.
  *mutable_options() = temp_options;
  // Run benchmark.
  MctsPlayer::SuggestMove();
  // Reset options.
  *mutable_options() = saved_options;

  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::HandleBoardsize(CmdArgs args) {
  auto response = CheckArgsExact(1, args);
  if (!response.ok) {
    return response;
  }

  int x;
  if (!absl::SimpleAtoi(args[0], &x) || x != kN) {
    return Response::Error("unacceptable size");
  }

  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::HandleClearBoard(CmdArgs args) {
  auto response = CheckArgsExact(0, args);
  if (!response.ok) {
    return response;
  }

  NewGame();
  ReportPosition(root());

  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::HandleEcho(CmdArgs args) {
  return Response::Ok(absl::StrJoin(args, " "));
}

GtpPlayer::Response GtpPlayer::HandleFinalScore(CmdArgs args) {
  auto response = CheckArgsExact(0, args);
  if (!response.ok) {
    return response;
  }
  if (!root()->game_over()) {
    // Game isn't over yet, calculate the current score using Tromp-Taylor
    // scoring.
    return Response::Ok(Game::FormatScore(
        root()->position.CalculateScore(options().game_options.komi)));
  } else {
    // Game is over, we have the result available.
    return Response::Ok(game_.result_string());
  }
}

GtpPlayer::Response GtpPlayer::HandleGenmove(CmdArgs args) {
  auto response = CheckArgsRange(0, 1, args);
  if (!response.ok) {
    return response;
  }

  auto c = SuggestMove();
  MG_LOG(INFO) << root()->Describe();
  MG_CHECK(PlayMove(c, &game_));

  // Begin pondering again if requested.
  if (ponder_type_ != PonderType::kOff) {
    ponder_limit_reached_ = false;
    ponder_read_count_ = 0;
    if (ponder_type_ == PonderType::kTimeLimited) {
      ponder_time_limit_ = absl::Now() + ponder_duration_;
    }
  }

  ReportPosition(root());

  return Response::Ok(c.ToGtp());
}

GtpPlayer::Response GtpPlayer::HandleInfo(CmdArgs args) {
  auto response = CheckArgsExact(0, args);
  if (!response.ok) {
    return response;
  }

  std::ostringstream oss;
  oss << options();
  oss << " report_search_interval:" << report_search_interval_;
  return Response::Ok(oss.str());
}

GtpPlayer::Response GtpPlayer::HandleKnownCommand(CmdArgs args) {
  auto response = CheckArgsExact(1, args);
  if (!response.ok) {
    return response;
  }
  std::string result;
  if (cmd_handlers_.find(std::string(args[0])) != cmd_handlers_.end()) {
    result = "true";
  } else {
    result = "false";
  }
  return Response::Ok(result);
}

GtpPlayer::Response GtpPlayer::HandleKomi(CmdArgs args) {
  auto response = CheckArgsExact(1, args);
  if (!response.ok) {
    return response;
  }

  double x;
  if (!absl::SimpleAtod(args[0], &x) || x != options().game_options.komi) {
    return Response::Error("unacceptable komi");
  }

  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::HandleListCommands(CmdArgs args) {
  auto response = CheckArgsExact(0, args);
  if (!response.ok) {
    return response;
  }
  std::vector<absl::string_view> cmds;
  for (const auto& kv : cmd_handlers_) {
    cmds.push_back(kv.first);
  }
  std::sort(cmds.begin(), cmds.end());

  response.str = absl::StrJoin(cmds, "\n");
  return response;
}

GtpPlayer::Response GtpPlayer::HandleLoadsgf(CmdArgs args) {
  auto response = CheckArgsExact(1, args);
  if (!response.ok) {
    return response;
  }

  std::ifstream f;
  f.open(std::string(args[0]));
  if (!f.is_open()) {
    MG_LOG(ERROR) << "couldn't read \"" << args[0] << "\"";
    return Response::Error("cannot load file");
  }
  std::stringstream buffer;
  buffer << f.rdbuf();

  return ParseSgf(buffer.str());
}

GtpPlayer::Response GtpPlayer::HandleName(CmdArgs args) {
  auto response = CheckArgsExact(0, args);
  if (!response.ok) {
    return response;
  }
  return Response::Ok(options().name);
}

GtpPlayer::Response GtpPlayer::HandlePlay(CmdArgs args) {
  auto response = CheckArgsExact(2, args);
  if (!response.ok) {
    return response;
  }

  Color color;
  if (std::tolower(args[0][0]) == 'b') {
    color = Color::kBlack;
  } else if (std::tolower(args[0][0]) == 'w') {
    color = Color::kWhite;
  } else {
    MG_LOG(ERROR) << "expected b or w for player color, got " << args[0];
    return Response::Error("illegal move");
  }
  if (color != root()->position.to_play()) {
    return Response::Error("out of turn moves are not yet supported");
  }

  Coord c = Coord::FromGtp(args[1], true);
  if (c == Coord::kInvalid) {
    MG_LOG(ERROR) << "expected GTP coord for move, got " << args[1];
    return Response::Error("illegal move");
  }

  if (!PlayMove(c, &game_)) {
    return Response::Error("illegal move");
  }
  ReportPosition(root());

  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::HandlePlaysgf(CmdArgs args) {
  auto sgf_str = absl::StrReplaceAll(absl::StrJoin(args, " "), {{"\\n", "\n"}});
  return ParseSgf(sgf_str);
}

GtpPlayer::Response GtpPlayer::HandlePonder(CmdArgs args) {
  auto response = CheckArgsRange(1, 2, args);
  if (!response.ok) {
    return response;
  }

  if (args[0] == "off") {
    // Disable pondering.
    ponder_type_ = PonderType::kOff;
    ponder_read_count_ = 0;
    ponder_read_limit_ = 0;
    ponder_duration_ = {};
    ponder_time_limit_ = absl::InfinitePast();
    ponder_limit_reached_ = true;
    return Response::Ok();
  }

  // Subsequent sub commands require exactly 2 arguments.
  response = CheckArgsExact(2, args);
  if (!response.ok) {
    return response;
  }

  if (args[0] == "winrate") {
    // Set the number of reads for win rate evaluation.
    int num_reads;
    if (!absl::SimpleAtoi(args[1], &num_reads) || num_reads < 0) {
      return Response::Error("invalid num_reads");
    }
    num_eval_reads_ = num_reads;
    RefreshPendingWinRateEvals();
    return Response::Ok();
  }

  if (args[0] == "reads") {
    // Enable pondering limited by number of reads.
    int read_limit;
    if (!absl::SimpleAtoi(args[1], &read_limit) || read_limit <= 0) {
      return Response::Error("couldn't parse read limit");
    }
    ponder_read_limit_ = read_limit;
    ponder_type_ = PonderType::kReadLimited;
    ponder_read_count_ = 0;
    ponder_limit_reached_ = false;
    return Response::Ok();
  }

  if (args[0] == "time") {
    // Enable pondering limited by time.
    float duration;
    if (!absl::SimpleAtof(args[1], &duration) || duration <= 0) {
      return Response::Error("couldn't parse time limit");
    }
    ponder_type_ = PonderType::kTimeLimited;
    ponder_duration_ = absl::Seconds(duration);
    ponder_time_limit_ = absl::Now() + ponder_duration_;
    ponder_limit_reached_ = false;
    return Response::Ok();
  }

  return Response::Error("unrecognized ponder mode");
}

GtpPlayer::Response GtpPlayer::HandlePruneNodes(CmdArgs args) {
  auto response = CheckArgsExact(1, args);
  if (!response.ok) {
    return response;
  }

  int x;
  if (!absl::SimpleAtoi(args[0], &x)) {
    return Response::Error("couldn't parse ", args[0], " as an integer");
  }

  mutable_options()->prune_orphaned_nodes = x != 0;

  return Response::Ok();
}
GtpPlayer::Response GtpPlayer::HandleReadouts(CmdArgs args) {
  auto response = CheckArgsExact(1, args);
  if (!response.ok) {
    return response;
  }

  int x;
  if (!absl::SimpleAtoi(args[0], &x) || x <= 0) {
    return Response::Error("couldn't parse ", args[0], " as an integer > 0");
  } else {
    mutable_options()->num_readouts = x;
  }

  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::HandleReportSearchInterval(CmdArgs args) {
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

GtpPlayer::Response GtpPlayer::HandleSelectPosition(CmdArgs args) {
  auto response = CheckArgsExact(1, args);
  if (!response.ok) {
    return response;
  }

  if (args[0] == "root") {
    ResetRoot();
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
  ResetRoot();
  for (const auto& move : moves) {
    MG_CHECK(PlayMove(move, &game_));
  }

  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::HandleUndo(CmdArgs args) {
  auto response = CheckArgsExact(0, args);
  if (!response.ok) {
    return response;
  }

  if (!UndoMove(&game_)) {
    return Response::Error("cannot undo");
  }

  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::HandleVariation(CmdArgs args) {
  auto response = CheckArgsRange(0, 1, args);
  if (!response.ok) {
    return response;
  }

  if (args.size() == 0) {
    child_variation_ = Coord::kInvalid;
  } else {
    Coord c = Coord::FromGtp(args[0], true);
    if (c == Coord::kInvalid) {
      MG_LOG(ERROR) << "expected GTP coord for move, got " << args[0];
      return Response::Error("illegal move");
    }
    if (c != child_variation_) {
      child_variation_ = c;
      ReportSearchStatus(root(), nullptr);
    }
  }

  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::HandleVerbosity(CmdArgs args) {
  auto response = CheckArgsRange(0, 1, args);
  if (!response.ok) {
    return response;
  }

  int x;
  if (!absl::SimpleAtoi(args[0], &x)) {
    return Response::Error("bad verbosity");
  }
  mutable_options()->verbose = x != 0;

  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::ParseSgf(const std::string& sgf_str) {
  sgf::Ast ast;
  if (!ast.Parse(sgf_str)) {
    MG_LOG(ERROR) << "couldn't parse SGF";
    return Response::Error("cannot load file");
  }

  // Clear the board before replaying sgf.
  NewGame();

  // Traverse the SGF's game trees, loading them into the backend & running
  // inference on the positions in batches.
  std::function<Response(const sgf::Node&)> traverse =
      [&](const sgf::Node& node) {
        if (node.move.color != root()->position.to_play()) {
          // The move color is different than expected. Play a pass move to flip
          // the colors.
          if (root()->move == Coord::kPass) {
            auto expected = ColorToCode(root()->position.to_play());
            auto actual = node.move.ToSgf();
            MG_LOG(ERROR) << "expected move by " << expected << ", got "
                          << actual
                          << " but can't play an intermediate pass because the"
                          << " previous move was also a pass";
            return Response::Error("cannot load file");
          }
          MG_LOG(WARNING) << "Inserting pass move";
          MG_CHECK(PlayMove(Coord::kPass, &game_));
          ReportPosition(root());
        }

        if (!PlayMove(node.move.c, &game_)) {
          MG_LOG(ERROR) << "error playing " << node.move.ToSgf();
          return Response::Error("cannot load file");
        }

        if (!node.comment.empty()) {
          auto* info = GetAuxInfo(root());
          info->comment = node.comment;
        }

        ReportPosition(root());
        for (const auto& child : node.children) {
          auto response = traverse(*child);
          if (!response.ok) {
            return response;
          }
        }
        UndoMove(&game_);
        return Response::Ok();
      };

  std::vector<std::unique_ptr<sgf::Node>> trees;
  if (!sgf::GetTrees(ast, &trees)) {
    return Response::Error("cannot load file");
  }
  for (const auto& tree : trees) {
    auto response = traverse(*tree);
    if (!response.ok) {
      return response;
    }
  }

  // Play the main line.
  ResetRoot();
  if (!trees.empty()) {
    for (const auto& move : trees[0]->ExtractMainLine()) {
      // We already validated that all the moves could be played in traverse(),
      // so if PlayMove fails here, something has gone seriously awry.
      MG_CHECK(PlayMove(move.c, &game_));
    }
    ReportPosition(root());
  }

  return Response::Ok();
}

void GtpPlayer::ReportSearchStatus(MctsNode* root, MctsNode* leaf) {
  nlohmann::json j = {
      {"id", GetAuxInfo(root)->id},
      {"n", root->N()},
      {"q", root->Q()},
  };

  // Pricipal variation.
  auto src_pv = root->MostVisitedPath();
  if (!src_pv.empty()) {
    auto& dst_pv = j["variations"]["pv"];
    for (Coord c : src_pv) {
      dst_pv.push_back(c.ToGtp());
    }
  }

  // Current tree search variation.
  if (leaf != nullptr) {
    std::vector<const MctsNode*> src_search;
    for (const auto* node = leaf; node != root; node = node->parent) {
      src_search.push_back(node);
    }
    if (!src_search.empty()) {
      std::reverse(src_search.begin(), src_search.end());
      auto& dst_search = j["variations"]["search"];
      for (const auto* node : src_search) {
        dst_search.push_back(node->move.ToGtp());
      }
    }
  }

  // Requested child variation, if any.
  if (child_variation_ != Coord::kInvalid) {
    auto& child_v = j["variations"][child_variation_.ToGtp()];
    child_v.push_back(child_variation_.ToGtp());
    auto it = root->children.find(child_variation_);
    if (it != root->children.end()) {
      for (Coord c : it->second->MostVisitedPath()) {
        child_v.push_back(c.ToGtp());
      }
    }
  }

  // Child N.
  auto& childN = j["childN"];
  for (const auto& edge : root->edges) {
    childN.push_back(static_cast<int>(edge.N));
  }

  // Child Q.
  auto& childQ = j["childQ"];
  for (int i = 0; i < kNumMoves; ++i) {
    childQ.push_back(static_cast<int>(std::round(root->child_Q(i) * 1000)));
  }

  MG_LOG(INFO) << "mg-update:" << j.dump();
}

void GtpPlayer::ReportPosition(MctsNode* node) {
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

GtpPlayer::AuxInfo* GtpPlayer::RegisterNode(MctsNode* node) {
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

GtpPlayer::AuxInfo* GtpPlayer::GetAuxInfo(MctsNode* node) const {
  auto it = node_to_info_.find(node);
  MG_CHECK(it != node_to_info_.end());
  return it->second.get();
}

GtpPlayer::AuxInfo::AuxInfo(AuxInfo* parent, MctsNode* node)
    : parent(parent), node(node), id(absl::StrFormat("%p", node)) {
  if (parent != nullptr) {
    parent->children.push_back(this);
  }
}

void GtpPlayer::RefreshPendingWinRateEvals() {
  to_eval_.clear();

  // Build a new list of nodes that require win rate evaluation.
  // First, traverse to the leaf node of the current position's main line.
  auto* info = RegisterNode(root());
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

}  // namespace minigo
