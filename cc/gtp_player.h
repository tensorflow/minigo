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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "cc/color.h"
#include "cc/dual_net/dual_net.h"
#include "cc/mcts_player.h"

namespace minigo {

class GtpPlayer : public MctsPlayer {
 public:
  struct Options : public MctsPlayer::Options {
    // If non-zero, GtpPlayer will print the current state of its tree search
    // every report_search_interval to stderr in a format recognized by Minigui.
    absl::Duration report_search_interval;

    // If non-zero, TreeSearch up to ponder_limit times while waiting for the
    // other player to play.
    int ponder_limit = 0;

    // If true, we will always pass if the opponent passes.
    bool courtesy_pass = false;
  };

  GtpPlayer(std::unique_ptr<DualNet> network, const Options& options);

  void Run();

  Coord SuggestMove() override;

 protected:
  absl::Span<MctsNode* const> TreeSearch(int batch_size) override;

 private:
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

  Response HandleBoardsize(absl::string_view cmd, CmdArgs args);
  Response HandleClearBoard(absl::string_view cmd, CmdArgs args);
  Response HandleEcho(absl::string_view cmd, CmdArgs args);
  Response HandleFinalScore(absl::string_view cmd, CmdArgs args);
  Response HandleGamestate(absl::string_view cmd, CmdArgs args);
  Response HandleGenmove(absl::string_view cmd, CmdArgs args);
  Response HandleInfo(absl::string_view cmd, CmdArgs args);
  Response HandleKnownCommand(absl::string_view cmd, CmdArgs args);
  Response HandleKomi(absl::string_view cmd, CmdArgs args);
  Response HandleListCommands(absl::string_view cmd, CmdArgs args);
  Response HandleLoadsgf(absl::string_view cmd, CmdArgs args);
  Response HandleName(absl::string_view cmd, CmdArgs args);
  Response HandlePlay(absl::string_view cmd, CmdArgs args);
  Response HandlePonderLimit(absl::string_view cmd, CmdArgs args);
  Response HandleReadouts(absl::string_view cmd, CmdArgs args);
  Response HandleReportSearchInterval(absl::string_view cmd, CmdArgs args);

  void ReportSearchStatus(const MctsNode* last_read);

  // The color of the last genmove command, which under normal circumstances
  // is the color we are playing as.
  // Set to Color::kEmpty when the board is cleared.
  Color last_genmove_ = Color::kEmpty;

  int ponder_count_ = 0;
  int ponder_limit_;
  bool courtesy_pass_;
  absl::Duration report_search_interval_;
  absl::Time last_report_time_;

  std::map<std::string, CmdHandler> cmd_handlers_;
};

}  // namespace minigo

#endif  // CC_GTP_PLAYER_H_
