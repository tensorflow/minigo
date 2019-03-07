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
#include <functional>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
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
#include "cc/sgf.h"
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
  };

  GtpPlayer(std::unique_ptr<DualNet> network,
            std::unique_ptr<InferenceCache> inference_cache, Game* game,
            const Options& options);

  void Run();
  void NewGame() override;
  Coord SuggestMove() override;

 protected:
  // Response from the GTP command handler.
  struct Response {
    static Response Ok(std::string str = "") {
      Response response;
      response.str = std::move(str);
      response.ok = true;
      return response;
    }

    template <typename... Args>
    static Response Error(const Args&... args) {
      Response response;
      response.str = absl::StrCat(args...);
      response.ok = false;
      return response;
    }

    static Response Done() {
      Response response;
      response.done = true;
      return response;
    }

    void set_cmd_id(int id) {
      has_cmd_id = true;
      cmd_id = id;
    }

    friend std::ostream& operator<<(std::ostream& os, const Response& r) {
      os << (r.ok ? "=" : "?");
      if (r.has_cmd_id) {
        os << r.cmd_id;
      }
      if (!r.str.empty()) {
        os << " " << r.str;
      }
      return os << "\n\n";
    }

    // Response to print to stdout.
    std::string str;

    // True if the command completed successfully.
    bool ok = false;

    // True if the Run loop should exit.
    bool done = false;

    bool has_cmd_id = false;

    int cmd_id = 0;
  };

  using CmdArgs = const std::vector<absl::string_view>&;

  // Helper to register a GTP command handler.
  // Templated to allow commands from subclasses to be registered.
  template <typename T>
  void RegisterCmd(const std::string& cmd, Response (T::*handler)(CmdArgs)) {
    static_assert(std::is_base_of<GtpPlayer, T>::value,
                  "T must be derived from GtpPlayer");
    cmd_handlers_[cmd] =
        std::bind(handler, static_cast<T*>(this), std::placeholders::_1);
  }

  // If waiting for the opponent to play, consider thinking for a bit.
  // Returns true if we pondered.
  bool MaybePonder();

  virtual void Ponder();

  // Begin pondering again if requested.
  void MaybeStartPondering();

  // Handles a GTP command specified by `line`.
  // Returns a (bool, string) pair containing whether the GtpPlayer should
  // continue running and the result of the command to write to stdout.
  virtual Response HandleCmd(const std::string& line);

  Response CheckArgsExact(size_t expected_num_args, CmdArgs args);
  Response CheckArgsRange(size_t expected_min_args, size_t expected_max_args,
                          CmdArgs args);

  Response DispatchCmd(const std::string& cmd, CmdArgs args);

  // TODO(tommadams): clearly document these methods w.r.t. the GTP standard and
  // what public methods they call.
  virtual Response HandleBenchmark(CmdArgs args);
  virtual Response HandleBoardsize(CmdArgs args);
  virtual Response HandleClearBoard(CmdArgs args);
  virtual Response HandleFinalScore(CmdArgs args);
  virtual Response HandleGenmove(CmdArgs args);
  virtual Response HandleKnownCommand(CmdArgs args);
  virtual Response HandleKomi(CmdArgs args);
  virtual Response HandleListCommands(CmdArgs args);
  virtual Response HandleLoadsgf(CmdArgs args);
  virtual Response HandleName(CmdArgs args);
  virtual Response HandlePlay(CmdArgs args);
  virtual Response HandlePonder(CmdArgs args);
  virtual Response HandleReadouts(CmdArgs args);
  virtual Response HandleShowboard(CmdArgs args);
  virtual Response HandleUndo(CmdArgs args);
  virtual Response HandleVerbosity(CmdArgs args);

  // Utilities for processing SGF files.
  Response ParseSgf(const std::string& sgf_str,
                    std::vector<std::unique_ptr<sgf::Node>>* trees);

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

  absl::flat_hash_map<std::string, std::function<Response(CmdArgs)>>
      cmd_handlers_;

  ThreadSafeQueue<std::string> stdin_queue_;
};

}  // namespace minigo

#endif  // CC_GTP_PLAYER_H_
