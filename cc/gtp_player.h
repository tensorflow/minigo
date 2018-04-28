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

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "cc/dual_net.h"
#include "cc/mcts_player.h"

namespace minigo {

class GtpPlayer : public MctsPlayer {
 public:
  struct Options : public MctsPlayer::Options {
    // Number of readouts to perform.
    int num_readouts = 100;

    // If non-zero, GtpPlayer will print the current state of its tree search
    // every report_search_interval to stderr in a format recognized by Minigui.
    absl::Duration report_search_interval;

    std::string name = "minigo";
  };

  GtpPlayer(std::unique_ptr<DualNet> network, const Options& options);

  // Handles a GTP command specified by `line`, printing the result to stdout.
  // Returns false if the GtpPlayer should quit.
  bool HandleCmd(const std::string& line);

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

  Response CheckArgsExact(absl::string_view cmd, size_t expected_num_args,
                          const std::vector<absl::string_view>& args);
  Response CheckArgsRange(absl::string_view cmd, size_t expected_min_args,
                          size_t expected_max_args,
                          const std::vector<absl::string_view>& args);

  Response DispatchCmd(absl::string_view cmd,
                       const std::vector<absl::string_view>& args);

  Response HandleBoardsize(absl::string_view cmd,
                           const std::vector<absl::string_view>& args);

  Response HandleClearBoard(absl::string_view cmd,
                            const std::vector<absl::string_view>& args);

  Response HandleEcho(absl::string_view cmd,
                      const std::vector<absl::string_view>& args);

  Response HandleFinalScore(absl::string_view cmd,
                            const std::vector<absl::string_view>& args);

  Response HandleGamestate(absl::string_view cmd,
                           const std::vector<absl::string_view>& args);

  Response HandleGenmove(absl::string_view cmd,
                         const std::vector<absl::string_view>& args);

  Response HandleInfo(absl::string_view cmd,
                      const std::vector<absl::string_view>& args);

  Response HandleName(absl::string_view cmd,
                      const std::vector<absl::string_view>& args);

  Response HandlePlay(absl::string_view cmd,
                      const std::vector<absl::string_view>& args);

  Response HandleReadouts(absl::string_view cmd,
                          const std::vector<absl::string_view>& args);

  Response HandleReportSearchInterval(
      absl::string_view cmd, const std::vector<absl::string_view>& args);

  void ReportSearchStatus(const MctsNode* last_read);

  std::string name_;
  int num_readouts_;
  absl::Duration report_search_interval_;
  absl::Time last_report_time_;
};

}  // namespace minigo

#endif  // CC_GTP_PLAYER_H_
