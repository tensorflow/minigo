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
#include "cc/sgf.h"
#include "nlohmann/json.hpp"

namespace minigo {

namespace {
// String written to stderr to signify that handling a GTP command is done.
const auto kGtpCmdDone = "__GTP_CMD_DONE__";
}  // namespace

GtpPlayer::GtpPlayer(std::unique_ptr<DualNet> network, const Options& options)
    : MctsPlayer(std::move(network), options),
      courtesy_pass_(options.courtesy_pass),
      ponder_read_limit_(options.ponder_limit) {
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
}

void GtpPlayer::Run() {
  std::cerr << "GTP engine ready" << std::endl;

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

Coord GtpPlayer::SuggestMove() {
  if (courtesy_pass_ && root()->move == Coord::kPass) {
    return Coord::kPass;
  }
  return MctsPlayer::SuggestMove();
}

void GtpPlayer::RegisterCmd(const std::string& cmd, CmdHandler handler) {
  cmd_handlers_[cmd] = handler;
}

bool GtpPlayer::MaybePonder() {
  if (root()->game_over() || ponder_type_ == PonderType::kOff ||
      ponder_limit_reached_) {
    return false;
  }

  if ((ponder_type_ == PonderType::kReadLimited &&
       ponder_read_count_ >= ponder_read_limit_) ||
      (ponder_type_ == PonderType::kTimeLimited &&
       absl::Now() >= ponder_time_limit_)) {
    if (!ponder_limit_reached_) {
      std::cerr << "mg-ponder: done" << std::endl;
      ponder_limit_reached_ = true;
    }
    return false;
  }

  if (ponder_read_count_ == 0) {
    std::cerr << "pondering..." << std::endl;
  }

  // Remember the number of reads at the root.
  int n = root()->N();

  TreeSearch();

  // Increment the ponder count by difference new and old reads.
  ponder_read_count_ += root()->N() - n;

  return true;
}

bool GtpPlayer::HandleCmd(const std::string& line) {
  std::vector<absl::string_view> args =
      absl::StrSplit(line, absl::ByAnyChar(" \t\r\n"), absl::SkipWhitespace());
  if (args.empty()) {
    std::cerr << kGtpCmdDone << std::endl;
    std::cout << "=\n\n" << std::flush;
    return true;
  }

  // Split the GTP command and its arguments.
  auto cmd = std::string(args[0]);
  args.erase(args.begin());

  if (cmd == "quit") {
    std::cerr << kGtpCmdDone << std::endl;
    std::cout << "=\n\n" << std::flush;
    return false;
  }

  auto response = DispatchCmd(cmd, args);
  std::cerr << kGtpCmdDone << std::endl;
  std::cout << (response.ok ? "=" : "?");
  if (!response.str.empty()) {
    std::cout << " " << response.str;
  }
  std::cout << "\n\n" << std::flush;
  return true;
}

absl::Span<const MctsPlayer::TreePath> GtpPlayer::TreeSearch() {
  auto paths = MctsPlayer::TreeSearch();
  if (!paths.empty() && report_search_interval_ != absl::ZeroDuration()) {
    auto now = absl::Now();
    if (now - last_report_time_ > report_search_interval_) {
      last_report_time_ = now;
      ReportSearchStatus(paths.back().leaf);
    }
  }
  return paths;
}

GtpPlayer::Response GtpPlayer::CheckArgsExact(absl::string_view cmd,
                                              size_t expected_num_args,
                                              CmdArgs args) {
  if (args.size() != expected_num_args) {
    return Response::Error("expected ", expected_num_args,
                           " args for GTP command ", cmd, ", got ", args.size(),
                           " args: ", absl::StrJoin(args, " "));
  }
  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::CheckArgsRange(absl::string_view cmd,
                                              size_t expected_min_args,
                                              size_t expected_max_args,
                                              CmdArgs args) {
  if (args.size() < expected_min_args || args.size() > expected_max_args) {
    return Response::Error("expected between ", expected_min_args, " and ",
                           expected_max_args, " args for GTP command ", cmd,
                           ", got ", args.size(),
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
  return (this->*handler)(cmd, args);
}

GtpPlayer::Response GtpPlayer::HandleBenchmark(absl::string_view cmd,
                                               CmdArgs args) {
  // benchmark [readouts] [batch_size]
  // Note: By default use current time_control (readouts or time).
  auto response = CheckArgsRange(cmd, 0, 2, args);
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

GtpPlayer::Response GtpPlayer::HandleBoardsize(absl::string_view cmd,
                                               CmdArgs args) {
  auto response = CheckArgsExact(cmd, 1, args);
  if (!response.ok) {
    return response;
  }

  int x;
  if (!absl::SimpleAtoi(args[0], &x) || x != kN) {
    return Response::Error("unacceptable size");
  }

  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::HandleClearBoard(absl::string_view cmd,
                                                CmdArgs args) {
  auto response = CheckArgsExact(cmd, 0, args);
  if (!response.ok) {
    return response;
  }

  game_nodes_.clear();
  NewGame();
  ReportPosition(root());

  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::HandleEcho(absl::string_view cmd, CmdArgs args) {
  return Response::Ok(absl::StrJoin(args, " "));
}

GtpPlayer::Response GtpPlayer::HandleFinalScore(absl::string_view cmd,
                                                CmdArgs args) {
  auto response = CheckArgsExact(cmd, 0, args);
  if (!response.ok) {
    return response;
  }
  if (!root()->game_over()) {
    // Game isn't over yet, calculate the current score using Tromp-Taylor
    // scoring.
    return Response::Ok(
        FormatScore(root()->position.CalculateScore(options().komi)));
  } else {
    // Game is over, we have the result available.
    return Response::Ok(result_string());
  }
}

GtpPlayer::Response GtpPlayer::HandleGenmove(absl::string_view cmd,
                                             CmdArgs args) {
  auto response = CheckArgsRange(cmd, 0, 1, args);
  if (!response.ok) {
    return response;
  }

  auto c = SuggestMove();
  std::cerr << root()->Describe() << std::endl;
  MG_CHECK(PlayMove(c));

  // Begin pondering again if requested.
  if (ponder_type_ != PonderType::kOff) {
    ponder_limit_reached_ = false;
    ponder_read_count_ = 0;
    if (ponder_type_ == PonderType::kTimeLimited) {
      ponder_time_limit_ = absl::Now() + ponder_duration_;
    }
  }

  ReportPosition(root());

  return Response::Ok(c.ToKgs());
}

GtpPlayer::Response GtpPlayer::HandleInfo(absl::string_view cmd, CmdArgs args) {
  auto response = CheckArgsExact(cmd, 0, args);
  if (!response.ok) {
    return response;
  }

  std::ostringstream oss;
  oss << options();
  oss << " report_search_interval:" << report_search_interval_;
  return Response::Ok(oss.str());
}

GtpPlayer::Response GtpPlayer::HandleKnownCommand(absl::string_view cmd,
                                                  CmdArgs args) {
  auto response = CheckArgsExact(cmd, 1, args);
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

GtpPlayer::Response GtpPlayer::HandleKomi(absl::string_view cmd, CmdArgs args) {
  auto response = CheckArgsExact(cmd, 1, args);
  if (!response.ok) {
    return response;
  }

  double x;
  if (!absl::SimpleAtod(args[0], &x) || x != options().komi) {
    return Response::Error("unacceptable komi");
  }

  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::HandleListCommands(absl::string_view cmd,
                                                  CmdArgs args) {
  auto response = CheckArgsExact(cmd, 0, args);
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

GtpPlayer::Response GtpPlayer::HandleLoadsgf(absl::string_view cmd,
                                             CmdArgs args) {
  auto response = CheckArgsExact(cmd, 1, args);
  if (!response.ok) {
    return response;
  }

  std::ifstream f;
  f.open(std::string(args[0]));
  if (!f.is_open()) {
    std::cerr << "Couldn't read \"" << args[0] << "\"" << std::endl;
    return Response::Error("cannot load file");
  }
  std::stringstream buffer;
  buffer << f.rdbuf();

  return ParseSgf(buffer.str());
}

GtpPlayer::Response GtpPlayer::HandleName(absl::string_view cmd, CmdArgs args) {
  auto response = CheckArgsExact(cmd, 0, args);
  if (!response.ok) {
    return response;
  }
  return Response::Ok(options().name);
}

GtpPlayer::Response GtpPlayer::HandlePlay(absl::string_view cmd, CmdArgs args) {
  auto response = CheckArgsExact(cmd, 2, args);
  if (!response.ok) {
    return response;
  }

  Color color;
  if (std::tolower(args[0][0]) == 'b') {
    color = Color::kBlack;
  } else if (std::tolower(args[0][0]) == 'w') {
    color = Color::kWhite;
  } else {
    std::cerr << "ERRROR: expected b or w for player color, got " << args[0]
              << std::endl;
    return Response::Error("illegal move");
  }
  if (color != root()->position.to_play()) {
    return Response::Error("out of turn moves are not yet supported");
  }

  Coord c = Coord::FromKgs(args[1], true);
  if (c == Coord::kInvalid) {
    std::cerr << "ERRROR: expected KGS coord for move, got " << args[1]
              << std::endl;
    return Response::Error("illegal move");
  }

  if (!PlayMove(c)) {
    return Response::Error("illegal move");
  }
  ReportPosition(root());

  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::HandlePlaysgf(absl::string_view cmd,
                                             CmdArgs args) {
  auto sgf_str = absl::StrReplaceAll(absl::StrJoin(args, " "), {{"\\n", "\n"}});
  return ParseSgf(sgf_str);
}

GtpPlayer::Response GtpPlayer::HandlePonder(absl::string_view cmd,
                                            CmdArgs args) {
  auto response = CheckArgsExact(cmd, 2, args);
  if (!response.ok) {
    return response;
  }

  // Default to pondering disabled in case parsing fails.
  ponder_type_ = PonderType::kOff;
  ponder_read_count_ = 0;
  ponder_read_limit_ = 0;
  ponder_duration_ = {};
  ponder_time_limit_ = absl::InfinitePast();
  ponder_limit_reached_ = true;

  if (args[0] == "reads") {
    if (!absl::SimpleAtoi(args[1], &ponder_read_limit_) ||
        ponder_read_limit_ <= 0) {
      return Response::Error("couldn't parse read limit");
    }
    ponder_type_ = PonderType::kReadLimited;
    ponder_read_count_ = 0;
    ponder_limit_reached_ = false;
  } else if (args[0] == "time") {
    float duration;
    if (!absl::SimpleAtof(args[1], &duration) || duration <= 0) {
      return Response::Error("couldn't parse time limit");
    }
    ponder_type_ = PonderType::kTimeLimited;
    ponder_duration_ = absl::Seconds(duration);
    ponder_time_limit_ = absl::Now() + ponder_duration_;
    ponder_limit_reached_ = false;
  } else if (args[0] != "off") {
    return Response::Error("unrecognized ponder mode");
  }

  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::HandlePruneNodes(absl::string_view cmd,
                                                CmdArgs args) {
  auto response = CheckArgsExact(cmd, 1, args);
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
GtpPlayer::Response GtpPlayer::HandleReadouts(absl::string_view cmd,
                                              CmdArgs args) {
  auto response = CheckArgsExact(cmd, 1, args);
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

GtpPlayer::Response GtpPlayer::HandleReportSearchInterval(absl::string_view cmd,
                                                          CmdArgs args) {
  auto response = CheckArgsExact(cmd, 1, args);
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

GtpPlayer::Response GtpPlayer::HandleSelectPosition(absl::string_view cmd,
                                                    CmdArgs args) {
  auto response = CheckArgsExact(cmd, 1, args);
  if (!response.ok) {
    return response;
  }

  if (args[0] == "root") {
    ResetRoot();
    return Response::Ok();
  }

  auto it = game_nodes_.find(args[0]);
  if (it == game_nodes_.end()) {
    return Response::Error("position id not found");
  }

  auto* node = it->second;
  MG_CHECK(node != nullptr);

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
    MG_CHECK(PlayMove(move));
  }

  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::HandleUndo(absl::string_view cmd, CmdArgs args) {
  auto response = CheckArgsExact(cmd, 0, args);
  if (!response.ok) {
    return response;
  }

  if (!UndoMove()) {
    return Response::Error("cannot undo");
  }

  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::HandleVariation(absl::string_view cmd,
                                               CmdArgs args) {
  auto response = CheckArgsRange(cmd, 0, 1, args);
  if (!response.ok) {
    return response;
  }

  if (args.size() == 0) {
    child_variation_ = Coord::kInvalid;
  } else {
    Coord c = Coord::FromKgs(args[0], true);
    if (c == Coord::kInvalid) {
      std::cerr << "ERRROR: expected KGS coord for move, got " << args[0]
                << std::endl;
      return Response::Error("illegal move");
    }
    if (c != child_variation_) {
      child_variation_ = c;
      ReportSearchStatus(nullptr);
    }
  }

  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::HandleVerbosity(absl::string_view cmd,
                                               CmdArgs args) {
  auto response = CheckArgsRange(cmd, 0, 1, args);
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
    std::cerr << "Couldn't parse SGF" << std::endl;
    return Response::Error("cannot parse file");
  }

  // Clear the board before replaying sgf.
  NewGame();

  std::vector<TreePath> paths;

  // Run inference on a batch of positions, then report the results to the
  // frontend.
  auto run_inference = [&]() {
    ProcessLeaves(absl::MakeSpan(paths), false);
    for (const auto& path : paths) {
      ReportPosition(path.leaf);
    }
    paths.clear();
  };

  // Traverse the SGF's game trees, loading them into the backend & running
  // inference on the positions in batches.
  std::function<void(const sgf::Node&)> traverse = [&](const sgf::Node& node) {
    MG_CHECK(node.move.color == root()->position.to_play());
    auto* leaf = root()->MaybeAddChild(node.move.c);
    MG_CHECK(PlayMove(node.move.c));

    paths.emplace_back(root(), leaf);
    if (paths.size() == static_cast<size_t>(options().batch_size)) {
      run_inference();
    }

    for (const auto& child : node.children) {
      traverse(*child);
    }
    UndoMove();
  };

  auto trees = sgf::GetTrees(ast);
  for (const auto& tree : trees) {
    traverse(*tree);
  }

  // Run inference on any stragglers.
  if (!paths.empty()) {
    run_inference();
  }

  // Play the main line.
  ResetRoot();
  if (!trees.empty()) {
    for (const auto& move : trees[0]->ExtractMainLine()) {
      MG_CHECK(PlayMove(move.c));
    }
    ReportPosition(root());
  }

  return Response::Ok();
}

void GtpPlayer::ReportSearchStatus(const MctsNode* last_read) {
  const auto& pos = root()->position;

  nlohmann::json j = {
      {"id", RegisterNode(root())},
      {"moveNum", pos.n()},
      {"toPlay", pos.to_play() == Color::kBlack ? "B" : "W"},
      {"q", root()->Q()},
      {"n", root()->N()},
  };

  // Pricipal variation.
  auto src_pv = root()->MostVisitedPath();
  if (!src_pv.empty()) {
    auto& dst_pv = j["variations"]["pv"];
    for (Coord c : src_pv) {
      dst_pv.push_back(c.ToKgs());
    }
  }

  // Current tree search variation.
  if (last_read != nullptr) {
    std::vector<const MctsNode*> src_search;
    for (const auto* node = last_read; node != root(); node = node->parent) {
      src_search.push_back(node);
    }
    if (!src_search.empty()) {
      std::reverse(src_search.begin(), src_search.end());
      auto& dst_search = j["variations"]["search"];
      for (const auto* node : src_search) {
        dst_search.push_back(node->move.ToKgs());
      }
    }
  }

  // Requested child variation, if any.
  if (child_variation_ != Coord::kInvalid) {
    auto& child_v = j["variations"][child_variation_.ToKgs()];
    child_v.push_back(child_variation_.ToKgs());
    auto it = root()->children.find(child_variation_);
    if (it != root()->children.end()) {
      for (Coord c : it->second->MostVisitedPath()) {
        child_v.push_back(c.ToKgs());
      }
    }
  }

  // Child N.
  auto& childN = j["childN"];
  for (const auto& edge : root()->edges) {
    childN.push_back(static_cast<int>(edge.N));
  }

  // Child Q.
  auto& childQ = j["childQ"];
  for (int i = 0; i < kNumMoves; ++i) {
    childQ.push_back(static_cast<int>(std::round(root()->child_Q(i) * 1000)));
  }

  std::cerr << "mg-update:" << j.dump() << std::endl;
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

  nlohmann::json j = {
      {"id", RegisterNode(node)},
      {"toPlay", position.to_play() == Color::kBlack ? "B" : "W"},
      {"moveNum", position.n()},
      {"stones", oss.str()},
      {"q", node->parent != nullptr ? node->parent->Q() : 0},
      {"gameOver", node->game_over()},
  };
  if (node->parent != nullptr) {
    j["parentId"] = RegisterNode(node->parent);
  }
  if (node->move != Coord::kInvalid) {
    j["move"] = node->move.ToKgs();
  }

  std::cerr << "mg-position: " << j.dump() << std::endl;
}

std::string GtpPlayer::RegisterNode(MctsNode* node) {
  auto id = absl::StrFormat("%p", node);
  game_nodes_.emplace(id, node);
  return id;
}

}  // namespace minigo
