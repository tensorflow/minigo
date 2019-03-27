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

#ifndef CC_TEST_UTILS_H_
#define CC_TEST_UTILS_H_

#include <string>

#include "absl/strings/string_view.h"
#include "cc/color.h"
#include "cc/coord.h"
#include "cc/group.h"
#include "cc/mcts_node.h"
#include "cc/position.h"
#include "cc/random.h"

namespace minigo {

// A version of the Position class that exposes some protected methods as public
// for testing purposes.
class TestablePosition : public Position {
 public:
  TestablePosition(absl::string_view board_str, Color to_play = Color::kBlack);

  using Position::PlayMove;

  // Convenience functions that automatically parse coords.
  void PlayMove(absl::string_view str, Color color = Color::kEmpty) {
    Position::PlayMove(Coord::FromString(str), color);
  }
  Group GroupAt(absl::string_view str) const {
    return Position::GroupAt(Coord::FromString(str));
  }
  Color IsKoish(absl::string_view str) const {
    return Position::IsKoish(Coord::FromString(str));
  }
  MoveType ClassifyMove(absl::string_view str) const {
    return Position::ClassifyMove(Coord::FromString(str));
  }
  using Position::ClassifyMove;

  BoardVisitor board_visitor;
  GroupVisitor group_visitor;
};

// Removes extraneous whitespace from a board string and returns it in the same
// format as Position::ToSimpleString().
std::string CleanBoardString(absl::string_view str);

std::array<Color, kN * kN> ParseBoard(absl::string_view str);

int CountPendingVirtualLosses(const MctsNode* node);

// Get a random legal move.
// Only returns Coord::kPass if no other move is legal.
Coord GetRandomLegalMove(const Position& position, Random* rnd);

}  // namespace minigo

#endif  // CC_TEST_UTILS_H_
