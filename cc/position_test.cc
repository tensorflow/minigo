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

#include "cc/position.h"

#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "cc/constants.h"
#include "gtest/gtest.h"

namespace minigo {
namespace {

// A version of the Position class that exposes some protected methods as public
// for testing purposes.
class TestablePosition : public Position {
 public:
  explicit TestablePosition(float komi = 0)
      : Position(&board_visitor, &group_visitor, komi) {}

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
  bool IsMoveSuicidal(absl::string_view str, Color color) const {
    return Position::IsMoveSuicidal(Coord::FromString(str), color);
  }

  using Position::PlayMove;

  BoardVisitor board_visitor;
  GroupVisitor group_visitor;
};

// Splits a simple board representation into multiple lines, stripping
// whitespace. Lines are padded with '.' to ensure a kN * kN board.
std::vector<std::string> SplitBoardString(absl::string_view str) {
  std::vector<std::string> lines;
  for (const auto& line : absl::StrSplit(str, '\n')) {
    std::string stripped(absl::StripAsciiWhitespace(line));
    if (stripped.empty()) {
      continue;
    }
    assert(stripped.size() <= kN);
    stripped.resize(kN, '.');
    lines.push_back(std::move(stripped));
  }
  assert(lines.size() <= kN);
  while (lines.size() < kN) {
    lines.emplace_back(kN, '.');
  }
  return lines;
}

// Removes extraneous whitespace from a board string and returns it in the same
// format as Position::ToSimpleString().
std::string CleanBoardString(absl::string_view str) {
  return absl::StrJoin(SplitBoardString(str), "\n") + "\n";
}

TestablePosition ParseBoard(absl::string_view str, float komi = 0) {
  auto lines = SplitBoardString(str);

  TestablePosition board(komi);
  for (int row = 0; row < kN; ++row) {
    for (int col = 0; col < kN; ++col) {
      if (lines[row][col] == 'X') {
        board.PlayMove({row, col}, Color::kBlack);
      } else if (lines[row][col] == 'O') {
        board.PlayMove({row, col}, Color::kWhite);
      }
    }
  }
  return board;
}

TEST(PositionTest, TestIsKoish) {
  auto board = ParseBoard(R"(
      .X.O.O.O.
      X...O...O
      ...O.....
      X.O.O...O
      .X.O...X.
      X.O..X...
      .X.OX.X..
      X.O..X..O
      XX.....X.)");

  std::set<std::string> expected_black_kos = {"A9", "A5", "A3", "F3"};
  std::set<std::string> expected_white_kos = {"E9", "J9", "D6"};
  for (int row = 0; row < kN; ++row) {
    for (int col = 0; col < kN; ++col) {
      auto c = Coord(row, col).ToKgs();
      Color expected;
      Color actual = board.IsKoish(c);
      if (expected_white_kos.find(c) != expected_white_kos.end()) {
        expected = Color::kWhite;
      } else if (expected_black_kos.find(c) != expected_black_kos.end()) {
        expected = Color::kBlack;
      } else {
        expected = Color::kEmpty;
      }
      EXPECT_EQ(expected, actual) << c;
    }
  }
}

TEST(PositionTest, TestMergeGroups1) {
  auto board = ParseBoard(R"(
      .X.
      X.X
      .X.)");

  EXPECT_EQ(3, board.GroupAt("B9").num_liberties);
  EXPECT_EQ(3, board.GroupAt("A8").num_liberties);
  EXPECT_EQ(4, board.GroupAt("C8").num_liberties);
  EXPECT_EQ(4, board.GroupAt("B7").num_liberties);

  board.PlayMove("B8", Color::kBlack);

  EXPECT_EQ(CleanBoardString(R"(
      .X.
      XXX
      .X.)"),
            board.ToSimpleString());

  auto group = board.GroupAt("B8");
  EXPECT_EQ(5, group.size);
  EXPECT_EQ(6, group.num_liberties);
}

TEST(PositionTest, TestMergeGroups2) {
  auto board = ParseBoard(R"(
      .........
      .........
      .........
      .........
      .........
      ..O..O...
      ..O..O...
      ..O..O...
      ..OOO....)");

  EXPECT_EQ(10, board.GroupAt("C1").num_liberties);
  EXPECT_EQ(8, board.GroupAt("F2").num_liberties);

  board.PlayMove("F1", Color::kWhite);

  EXPECT_EQ(CleanBoardString(R"(
      .........
      .........
      .........
      .........
      .........
      ..O..O...
      ..O..O...
      ..O..O...
      ..OOOO...)"),
            board.ToSimpleString());

  auto group = board.GroupAt("F4");
  EXPECT_EQ(10, group.size);
  EXPECT_EQ(16, group.num_liberties);
}

TEST(PositionTest, TestCaptureStone) {
  auto board = ParseBoard(R"(
      .........
      .........
      .........
      .........
      .........
      .........
      .......O.
      ......OX.
      .......O.)");

  board.PlayMove("J2", Color::kWhite);

  std::array<int, 2> expected_captures = {0, 1};
  EXPECT_EQ(expected_captures, board.num_captures());

  EXPECT_EQ(CleanBoardString(R"(
      .........
      .........
      .........
      .........
      .........
      .........
      .......O.
      ......O.O
      .......O.)"),
            board.ToSimpleString());
}

TEST(PositionTest, TestCaptureMany1) {
  auto board = ParseBoard(R"(
      .....
      .....
      ..XX.
      .XOO.
      ..XX.)");

  EXPECT_EQ(4, board.GroupAt("C7").num_liberties);
  EXPECT_EQ(3, board.GroupAt("B6").num_liberties);
  EXPECT_EQ(1, board.GroupAt("C6").num_liberties);
  EXPECT_EQ(4, board.GroupAt("D5").num_liberties);

  board.PlayMove("E6", Color::kBlack);

  std::array<int, 2> expected_captures = {2, 0};
  EXPECT_EQ(expected_captures, board.num_captures());

  EXPECT_EQ(CleanBoardString(R"(
      .....
      .....
      ..XX.
      .X..X
      ..XX.)"),
            board.ToSimpleString());

  EXPECT_EQ(6, board.GroupAt("C7").num_liberties);
  EXPECT_EQ(4, board.GroupAt("B6").num_liberties);
  EXPECT_EQ(4, board.GroupAt("E6").num_liberties);
  EXPECT_EQ(6, board.GroupAt("D5").num_liberties);
}

TEST(PositionTest, TestCaptureMany2) {
  auto board = ParseBoard(R"(
      ..X..
      .XOX.
      XO.OX
      .XOX.
      ..X..)");

  board.PlayMove("C7", Color::kBlack);

  std::array<int, 2> expected_captures = {4, 0};
  EXPECT_EQ(expected_captures, board.num_captures());

  EXPECT_EQ(CleanBoardString(R"(
      ..X..
      .X.X.
      X.X.X
      .X.X.
      ..X..)"),
            board.ToSimpleString());

  EXPECT_EQ(3, board.GroupAt("C9").num_liberties);
  EXPECT_EQ(4, board.GroupAt("B8").num_liberties);
  EXPECT_EQ(4, board.GroupAt("D8").num_liberties);
  EXPECT_EQ(3, board.GroupAt("A7").num_liberties);
  EXPECT_EQ(4, board.GroupAt("B6").num_liberties);
  EXPECT_EQ(4, board.GroupAt("D6").num_liberties);
  EXPECT_EQ(4, board.GroupAt("C5").num_liberties);
}

TEST(PositionTest, TestCaptureMultipleGroups) {
  auto board = ParseBoard(R"(
      .OX
      OXX
      XX.)");

  board.PlayMove("A9", Color::kBlack);

  std::array<int, 2> expected_captures = {2, 0};
  EXPECT_EQ(expected_captures, board.num_captures());

  EXPECT_EQ(CleanBoardString(R"(
      X.X
      .XX
      XX.)"),
            board.ToSimpleString());

  EXPECT_EQ(2, board.GroupAt("A9").num_liberties);
  EXPECT_EQ(7, board.GroupAt("B8").num_liberties);
}

TEST(PositionTest, TestSameFriendlyGroupNeighboringTwice) {
  auto board = ParseBoard(R"(
      OO
      O.)");

  board.PlayMove("B8", Color::kWhite);

  EXPECT_EQ(CleanBoardString(R"(
      OO
      OO)"),
            board.ToSimpleString());

  auto group = board.GroupAt("A9");
  EXPECT_EQ(4, group.size);
  EXPECT_EQ(4, group.num_liberties);
}

TEST(PositionTest, TestSameOpponentGroupNeighboringTwice) {
  auto board = ParseBoard(R"(
      OO
      O.)");

  board.PlayMove("B8", Color::kBlack);

  EXPECT_EQ(CleanBoardString(R"(
      OO
      OX)"),
            board.ToSimpleString());

  auto white_group = board.GroupAt("A9");
  EXPECT_EQ(3, white_group.size);
  EXPECT_EQ(2, white_group.num_liberties);

  auto black_group = board.GroupAt("B8");
  EXPECT_EQ(1, black_group.size);
  EXPECT_EQ(2, black_group.num_liberties);
}

TEST(PositionTest, IsMoveSuicidal) {
  auto board = ParseBoard(R"(
      ...O.O...
      ....O....
      XO.....O.
      OXO...OXO
      O.XO.OX.O
      OXOOOOOOX
      XOOX.XO..
      ...OOOXXO
      .....XOO.)");
  std::vector<std::string> suicidal_moves = {"E9", "H5", "E3"};
  for (const auto& c : suicidal_moves) {
    EXPECT_TRUE(board.IsMoveSuicidal(c, Color::kBlack));
  }
  std::vector<std::string> nonsuicidal_moves = {"B5", "J1", "A9"};
  for (const auto& c : nonsuicidal_moves) {
    EXPECT_FALSE(board.IsMoveSuicidal(c, Color::kBlack));
  }
}

// Tests ko tracking.
TEST(PositionTest, TestKoTracking) {
  auto board = ParseBoard(R"(
      XOXO.....
      .........)");

  // Capturing a stone in a non-koish coord shouldn't create a ko.
  board.PlayMove("B8", Color::kBlack);
  EXPECT_EQ(CleanBoardString(R"(
      X.XO.....
      .X.......)"),
            board.ToSimpleString());

  // Capturing a stone in a koish coord should create a ko.
  board.PlayMove("C8", Color::kWhite);
  board.PlayMove("B9", Color::kWhite);
  EXPECT_EQ(CleanBoardString(R"(
      XO*O.....
      .XO......)"),
            board.ToSimpleString());

  // Playing a move should clear the ko.
  board.PlayMove("J9", Color::kBlack);
  EXPECT_EQ(CleanBoardString(R"(
      XO.O....X
      .XO......)"),
            board.ToSimpleString());

  // Test ko when capturing white as well.
  board.PlayMove("C9", Color::kBlack);
  EXPECT_EQ(CleanBoardString(R"(
      X*XO....X
      .XO......)"),
            board.ToSimpleString());

  // Playing a move should clear the ko.
  board.PlayMove("H9", Color::kWhite);
  EXPECT_EQ(CleanBoardString(R"(
      X.XO...OX
      .XO......)"),
            board.ToSimpleString());
}

TEST(PositionTest, TestSeki) {
  auto board = ParseBoard(R"(
    O....XXXX
    .....XOOO
    .....XOX.
    .....XO.X
    .....XOOX
    ......XOO
    ......XXX)");

  // All empty squares are neutral points and black has 5 more stones than
  // white.
  EXPECT_EQ(5, board.CalculateScore());
}

TEST(PositionTest, TestScoring) {
  EXPECT_EQ(0, TestablePosition().CalculateScore());
  EXPECT_EQ(-42, TestablePosition(42).CalculateScore());
  EXPECT_EQ(kN * kN, ParseBoard("X").CalculateScore());
  EXPECT_EQ(-kN * kN, ParseBoard("O").CalculateScore());

  auto board = ParseBoard(R"(
    .X.
    X.O)");
  EXPECT_EQ(2, board.CalculateScore());

  board = ParseBoard(R"(
   .XX......
   OOXX.....
   OOOX...X.
   OXX......
   OOXXXXXX.
   OOOXOXOXX
   .O.OOXOOX
   .O.O.OOXX
   ......OOO)",
                     6.5);
  EXPECT_EQ(1.5, board.CalculateScore());

  board = ParseBoard(R"(
   XXX......
   OOXX.....
   OOOX...X.
   OXX......
   OOXXXXXX.
   OOOXOXOXX
   .O.OOXOOX
   .O.O.OOXX
   ......OOO)",
                     6.5);
  EXPECT_EQ(2.5, board.CalculateScore());
}

// Plays through an example game and verifies that the outcome is as expected.
TEST(PositionTest, PlayGame) {
  std::vector<std::string> moves = {
      "ge", "de", "cg", "ff", "ed", "dd", "ec", "dc", "eg", "df", "dg", "fe",
      "gc", "gd", "fd", "hd", "he", "bf", "bg", "fg", "cf", "be", "ce", "cd",
      "af", "hc", "hb", "gb", "ic", "bd", "fh", "gh", "fi", "eh", "dh", "di",
      "ci", "fc", "id", "ch", "ei", "eb", "gc", "fb", "gg", "gf", "hg", "hf",
      "if", "hh", "ig", "ih", "ie", "ha", "ga", "gd", "hd", "fa", "ib", "ae",
      "ag", "ee", "gd", "cb", "gi", "ga", "hi", "ia", "ii", "ef", "bh", "db",
      "hh", "ba", "ai", "ac", "bi", "da", "di", "ab", "eh", "bc", "gh",
  };

  TestablePosition board(7.5);
  for (const auto& move : moves) {
    board.PlayMove(move);
    // std::cout << board.ToPrettyString() << std::endl;
  }

  EXPECT_EQ(CleanBoardString(R"(
      .O.O.OOOO
      O.OOOOOXX
      OO.OXOX.X
      .OOOXXXXX
      OOXOOOXXX
      XOXOOOOOX
      XXXXXOXXX
      .X.XXXXX.
      XXXXXXXXX)"),
            board.ToSimpleString());

  std::array<int, 2> expected_captures = {10, 2};
  EXPECT_EQ(expected_captures, board.num_captures());
  EXPECT_EQ(-0.5, board.CalculateScore());
}

}  // namespace
}  // namespace minigo
