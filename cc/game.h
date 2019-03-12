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

#ifndef CC_GAME_H_
#define CC_GAME_H_

#include <array>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "cc/color.h"
#include "cc/constants.h"
#include "cc/coord.h"
#include "cc/position.h"

namespace minigo {

// Holds game-specific options.
// Holds the full history of a game.
class Game {
 public:
  struct Options {
    float resign_threshold = -0.95;

    // We use a separate resign_enabled flag instead of setting the
    // resign_threshold to -1 for games where resignation is diabled. This
    // enables us to report games where the eventual winner would have
    // incorrectly resigned early, had resignations been enabled.
    bool resign_enabled = true;

    float komi = kDefaultKomi;

    // If true, repeated calls to AddMove with the same Color and Coord will
    // be ignored. This should be set to true when two separate MctsPlayer
    // instances are playing (because they both make AddMove calls to the same
    // Game object), and set to false for self-play where a single MctsPlayer
    // plays both back and white.
    bool ignore_repeated_moves = false;

    friend std::ostream& operator<<(std::ostream& os, const Options& options);
  };

  struct Move {
    Color color;

    Coord c = Coord::kInvalid;

    float Q;

    // Comments associated with the move.
    std::string comment;

    // Models evaluated when performing tree search.
    std::vector<std::string> models;

    std::array<float, kNumMoves> search_pi;

    // Stones on the board before the move was played.
    // This is used to build training features after a selfplay game has
    // finished.
    Position::Stones stones;
  };

  enum class GameOverReason {
    kBothPassed,
    kOpponentResigned,
    kMoveLimitReached,
  };

  static std::string FormatScore(float score);

  Game(std::string black_name, std::string white_name, const Options& options);

  void NewGame();

  void AddComment(const std::string& comment);

  void AddMove(Color color, Coord c, const Position::Stones& stones,
               std::string comment, float Q,
               const std::array<float, kNumMoves>& search_pi,
               std::vector<std::string> models);

  void UndoMove();

  void SetGameOverBecauseOfPasses(float score);

  void SetGameOverBecauseOfResign(Color winner);

  void SetGameOverBecauseMoveLimitReached(float score);

  // Returns up to the last `num_moves` of moves that lead up to the requested
  // `move`, including the move itself.
  // If `move < num_moves`, history will be truncated to the first `move` moves.
  void GetStoneHistory(int move, int num_moves,
                       std::vector<const Position::Stones*>* history) const;

  // Get information on the bleakest move for a completed game, if the game has
  // history and was played with resign disabled. This only makes sense if
  // resign was disabled (if resign was enabled, bleakest-move calculation is
  // not relevant, since quitters don't know how bad it could have been.)
  //
  // Returns true if the bleakest move was found and returned; false otherwise.
  // Q is returned from the winners perspective, which means we don't have to
  // reference the result to transform this into a sortable list of evaluations.
  // &q is the bleakest move from the perspective of the winner, i.e., negative.
  bool FindBleakestMove(int* move, float* q) const;

  const Options& options() const { return options_; }
  const std::string& black_name() const { return black_name_; }
  const std::string& white_name() const { return white_name_; }
  bool game_over() const { return game_over_; }
  GameOverReason game_over_reason() const {
    MG_CHECK(game_over());
    return game_over_reason_;
  }
  float result() const {
    MG_CHECK(game_over());
    return result_;
  }
  const std::string& result_string() const {
    MG_CHECK(game_over());
    return result_string_;
  }
  const std::string& comment() const { return comment_; }

  int num_moves() const { return static_cast<int>(moves_.size()); }
  const Move* GetMove(int i) {
    MG_CHECK(i >= 0 && i < num_moves());
    return moves_[i].get();
  }

  const std::vector<std::unique_ptr<Move>>& moves() const { return moves_; }

 private:
  const Options options_;
  const std::string black_name_;
  const std::string white_name_;
  bool game_over_ = false;
  GameOverReason game_over_reason_;
  float result_;
  std::string result_string_;
  std::string comment_;
  std::vector<std::unique_ptr<Move>> moves_;
};

}  // namespace minigo

#endif  // CC_GAME_H_
