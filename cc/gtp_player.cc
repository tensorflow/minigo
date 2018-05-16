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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/time/clock.h"
#include "cc/constants.h"
#include "cc/sgf.h"

namespace minigo {

GtpPlayer::GtpPlayer(std::unique_ptr<DualNet> network, const Options& options)
    : MctsPlayer(std::move(network), options),
      ponder_limit_(options.ponder_limit),
      courtesy_pass_(options.courtesy_pass),
      name_(options.name),
      num_readouts_(options.num_readouts) {
  RegisterCmd("boardsize", &GtpPlayer::HandleBoardsize);
  RegisterCmd("clear_board", &GtpPlayer::HandleClearBoard);
  RegisterCmd("echo", &GtpPlayer::HandleEcho);
  RegisterCmd("final_score", &GtpPlayer::HandleFinalScore);
  RegisterCmd("gamestate", &GtpPlayer::HandleGamestate);
  RegisterCmd("genmove", &GtpPlayer::HandleGenmove);
  RegisterCmd("info", &GtpPlayer::HandleInfo);
  RegisterCmd("known_command", &GtpPlayer::HandleKnownCommand);
  RegisterCmd("list_commands", &GtpPlayer::HandleListCommands);
  RegisterCmd("loadsgf", &GtpPlayer::HandleLoadsgf);
  RegisterCmd("name", &GtpPlayer::HandleName);
  RegisterCmd("play", &GtpPlayer::HandlePlay);
  RegisterCmd("ponder_limit", &GtpPlayer::HandlePonderLimit);
  RegisterCmd("readouts", &GtpPlayer::HandleReadouts);
  RegisterCmd("report_search_interval", &GtpPlayer::HandleReportSearchInterval);
}

void GtpPlayer::Run() {
  std::cerr << "GTP engine ready" << std::endl;
  std::string line;
  std::ios::sync_with_stdio(false);
  while (!std::cin.eof()) {
    if (MaybePonder()) {
      continue;
    }
    std::getline(std::cin, line);
    if (!HandleCmd(line)) {
      break;
    }
    ponder_count_ = 0;
  }
}

Coord GtpPlayer::SuggestMove(int num_readouts) {
  if (courtesy_pass_ && root()->move == Coord::kPass) {
    return Coord::kPass;
  }
  return MctsPlayer::SuggestMove(num_readouts);
}

void GtpPlayer::RegisterCmd(const std::string& cmd, CmdHandler handler) {
  cmd_handlers_[cmd] = handler;
}

bool GtpPlayer::MaybePonder() {
  // We ponder if all the following conditions are true:
  //  1) Pondering is enabled.
  //  2) We haven't pondered too much.
  //  3) There's no GTP command pending on stdin.
  //  4) It's the opponent's turn.
  bool should_ponder =
      (ponder_limit_ > 0 && ponder_count_ < ponder_limit_ &&
       std::cin.rdbuf()->in_avail() == 0 && last_genmove_ != Color::kEmpty &&
       last_genmove_ != root()->position.to_play());

  if (!should_ponder) {
    ponder_count_ = 0;
    return false;
  }

  if (ponder_count_ == 0) {
    std::cerr << "pondering...\n";
  }

  TreeSearch(options().batch_size);

  ponder_count_ += options().batch_size;
  if (ponder_count_ >= ponder_limit_) {
    std::cerr << root()->Describe() << "\n";
    std::cerr << "finished pondering\n";
  }

  return true;
}

bool GtpPlayer::HandleCmd(const std::string& line) {
  std::vector<absl::string_view> args =
      absl::StrSplit(line, absl::ByAnyChar(" \t\r\n"), absl::SkipWhitespace());
  if (args.empty()) {
    std::cout << "=" << std::endl;
    return true;
  }

  // Split the GTP command and its arguments.
  auto cmd = std::string(args[0]);
  args.erase(args.begin());

  if (cmd == "quit") {
    std::cout << "=\n\n" << std::flush;
    return false;
  }

  auto response = DispatchCmd(cmd, args);
  std::cout << (response.ok ? "=" : "?");
  if (!response.str.empty()) {
    std::cout << " " << response.str;
  }
  std::cout << "\n\n" << std::flush;
  return true;
}

absl::Span<MctsNode* const> GtpPlayer::TreeSearch(int batch_size) {
  auto leaves = MctsPlayer::TreeSearch(batch_size);
  if (!leaves.empty() && report_search_interval_ != absl::ZeroDuration()) {
    auto now = absl::Now();
    if (now - last_report_time_ > report_search_interval_) {
      last_report_time_ = now;
      ReportSearchStatus(leaves.back());
    }
  }
  return leaves;
}

GtpPlayer::Response GtpPlayer::CheckArgsExact(
    absl::string_view cmd, size_t expected_num_args,
    const std::vector<absl::string_view>& args) {
  if (args.size() != expected_num_args) {
    return Response::Error("expected ", expected_num_args,
                           " args for GTP command ", cmd, ", got ", args.size(),
                           " args: ", absl::StrJoin(args, " "));
  }
  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::CheckArgsRange(
    absl::string_view cmd, size_t expected_min_args, size_t expected_max_args,
    const std::vector<absl::string_view>& args) {
  if (args.size() < expected_min_args || args.size() > expected_max_args) {
    return Response::Error("expected between ", expected_min_args, " and ",
                           expected_max_args, " args for GTP command ", cmd,
                           ", got ", args.size(),
                           " args: ", absl::StrJoin(args, " "));
  }
  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::DispatchCmd(
    const std::string& cmd, const std::vector<absl::string_view>& args) {
  auto it = cmd_handlers_.find(cmd);
  if (it == cmd_handlers_.end()) {
    return Response::Error("unknown command");
  }
  auto handler = it->second;
  return (this->*handler)(cmd, args);
}

GtpPlayer::Response GtpPlayer::HandleBoardsize(
    absl::string_view cmd, const std::vector<absl::string_view>& args) {
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

GtpPlayer::Response GtpPlayer::HandleClearBoard(
    absl::string_view cmd, const std::vector<absl::string_view>& args) {
  auto response = CheckArgsExact(cmd, 0, args);
  if (!response.ok) {
    return response;
  }

  last_genmove_ = Color::kEmpty;
  NewGame();

  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::HandleEcho(
    absl::string_view cmd, const std::vector<absl::string_view>& args) {
  return Response::Ok(absl::StrJoin(args, " "));
}

GtpPlayer::Response GtpPlayer::HandleFinalScore(
    absl::string_view cmd, const std::vector<absl::string_view>& args) {
  auto response = CheckArgsExact(cmd, 0, args);
  if (!response.ok) {
    return response;
  }
  if (!game_over()) {
    // Game isn't over yet, calculate the current score using Tromp-Taylor
    // scoring.
    return Response::Ok(
        FormatScore(root()->position.CalculateScore(options().komi)));
  } else {
    // Game is over, we have the result available.
    return Response::Ok(result_string());
  }
}

GtpPlayer::Response GtpPlayer::HandleGamestate(
    absl::string_view cmd, const std::vector<absl::string_view>& args) {
  auto response = CheckArgsExact(cmd, 0, args);
  if (!response.ok) {
    return response;
  }

  const auto& position = root()->position;

  // board field.
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
  std::string board = oss.str();

  // toPlay field.
  std::string to_play = position.to_play() == Color::kBlack ? "Black" : "White";

  // lastMove field.
  std::string last_move;
  if (!history().empty()) {
    last_move = absl::StrCat("\"", history().back().c.ToKgs(), "\"");
  } else {
    last_move = "null";
  }

  // n field.
  auto n = position.n();

  // q field.
  auto q = root()->parent != nullptr ? root()->parent->Q() : 0;

  // clang-format off
  std::cerr << "mg-gamestate: {"
            << "\"board\":\"" << board << "\", "
            << "\"toPlay\":\"" << to_play << "\", "
            << "\"lastMove\":" << last_move << ", "
            << "\"n\":" << n << ", "
            << "\"q\":" << q
            << "}" << std::endl;
  // clang-format on
  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::HandleGenmove(
    absl::string_view cmd, const std::vector<absl::string_view>& args) {
  auto response = CheckArgsRange(cmd, 0, 1, args);
  if (!response.ok) {
    return response;
  }

  auto c = SuggestMove(num_readouts_);
  std::cerr << root()->Describe() << std::endl;
  last_genmove_ = root()->position.to_play();
  PlayMove(c);

  return Response::Ok(c.ToKgs());
}

GtpPlayer::Response GtpPlayer::HandleInfo(
    absl::string_view cmd, const std::vector<absl::string_view>& args) {
  auto response = CheckArgsExact(cmd, 0, args);
  if (!response.ok) {
    return response;
  }

  std::ostringstream oss;
  oss << options();
  oss << " num_readouts: " << num_readouts_
      << " report_search_interval:" << report_search_interval_
      << " name:" << name_;
  return Response::Ok(oss.str());
}

GtpPlayer::Response GtpPlayer::HandleKnownCommand(
    absl::string_view cmd, const std::vector<absl::string_view>& args) {
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

GtpPlayer::Response GtpPlayer::HandleListCommands(
    absl::string_view cmd, const std::vector<absl::string_view>& args) {
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

GtpPlayer::Response GtpPlayer::HandleLoadsgf(
    absl::string_view cmd, const std::vector<absl::string_view>& args) {
  auto response = CheckArgsExact(cmd, 1, args);
  if (!response.ok) {
    return response;
  }

  std::ifstream f;
  f.open(std::string(args[0]));
  if (!f.is_open()) {
    std::cerr << "Couldn't read \"" << args[0] << "\"\n";
    return Response::Error("cannot load file");
  }
  std::stringstream buffer;
  buffer << f.rdbuf();

  sgf::Ast ast;
  if (!ast.Parse(buffer.str())) {
    std::cerr << "Couldn't parse \"" << args[0] << "\"\n";
    return Response::Error("cannot load file");
  }

  for (const auto& move : sgf::GetMainLineMoves(ast)) {
    PlayMove(move.c);
  }

  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::HandleName(
    absl::string_view cmd, const std::vector<absl::string_view>& args) {
  auto response = CheckArgsExact(cmd, 0, args);
  if (!response.ok) {
    return response;
  }
  return Response::Ok(name_);
}

GtpPlayer::Response GtpPlayer::HandlePlay(
    absl::string_view cmd, const std::vector<absl::string_view>& args) {
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
    // TODO(tommadams): Allow out of turn moves.
    return Response::Error("out of turn moves are not yet supported");
  }

  Coord c = Coord::FromKgs(args[1], true);
  if (c == Coord::kInvalid) {
    std::cerr << "ERRROR: expected KGS coord for move, got " << args[1]
              << std::endl;
    return Response::Error("illegal move");
  }

  if (!root()->position.IsMoveLegal(c)) {
    return Response::Error("illegal move");
  }

  PlayMove(c);
  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::HandlePonderLimit(
    absl::string_view cmd, const std::vector<absl::string_view>& args) {
  auto response = CheckArgsExact(cmd, 1, args);
  if (!response.ok) {
    return response;
  }

  int x;
  if (!absl::SimpleAtoi(args[0], &x) || x < 0) {
    return Response::Error("couldn't parse ", args[0], " as an integer >= 0");
  } else {
    ponder_limit_ = x;
  }

  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::HandleReadouts(
    absl::string_view cmd, const std::vector<absl::string_view>& args) {
  auto response = CheckArgsExact(cmd, 1, args);
  if (!response.ok) {
    return response;
  }

  int x;
  if (!absl::SimpleAtoi(args[0], &x) || x <= 0) {
    return Response::Error("couldn't parse ", args[0], " as an integer > 0");
  } else {
    num_readouts_ = x;
  }

  return Response::Ok();
}

GtpPlayer::Response GtpPlayer::HandleReportSearchInterval(
    absl::string_view cmd, const std::vector<absl::string_view>& args) {
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

void GtpPlayer::ReportSearchStatus(const MctsNode* last_read) {
  std::cerr << "mg-search:";
  std::vector<const MctsNode*> path;
  for (const auto* node = last_read; node != root(); node = node->parent) {
    path.push_back(node);
  }
  for (auto it = path.rbegin(); it != path.rend(); ++it) {
    std::cerr << " " << (*it)->move.ToKgs();
  }
  std::cerr << "\n";

  std::cerr << "mg-q:";
  for (int i = 0; i < kNumMoves; ++i) {
    std::cerr << " " << std::fixed << std::setprecision(3)
              << (root()->child_Q(i) - root()->Q());
  }
  std::cerr << "\n";

  std::cerr << "mg-n:";
  for (const auto& edge : root()->edges) {
    std::cerr << std::setprecision(0) << " " << edge.N;
  }
  std::cerr << "\n";

  std::cerr << "mg-pv:";
  for (Coord c : root()->MostVisitedPath()) {
    std::cerr << " " << c.ToKgs();
  }
  std::cerr << "\n";

  std::cerr << std::flush;
}

}  // namespace minigo
