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
#include "cc/constants.h"
#include "cc/random.h"
#include "cc/test_utils.h"
#include "gtest/gtest.h"

namespace minigo {
namespace {

TEST(BoardVisitorTest, TestEpochRollover) {
  for (int j = 0; j <= 256; ++j) {
    BoardVisitor bv;
    for (int i = 0; i <= j; ++i) {
      bv.Begin();
    }
    for (int i = 0; i < kN * kN; ++i) {
      auto c = Coord(i);
      ASSERT_TRUE(bv.Visit(c));
      ASSERT_EQ(c, bv.Next());
      ASSERT_FALSE(bv.Visit(c));
    }
  }
}

TEST(GroupVisitorTest, TestEpochRollover) {
  for (int j = 0; j <= 256; ++j) {
    GroupVisitor gv;
    for (int i = 0; i <= j; ++i) {
      gv.Begin();
    }
    for (int i = 0; i < Group::kMaxNumGroups; ++i) {
      auto g = GroupId(i);
      ASSERT_TRUE(gv.Visit(g));
      ASSERT_FALSE(gv.Visit(g));
    }
  }
}

TEST(PositionTest, TestIsKoish) {
  auto board = TestablePosition(R"(
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
  auto board = TestablePosition(R"(
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
  auto board = TestablePosition(R"(
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
  auto board = TestablePosition(R"(
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

  std::array<int, 2> expected_captures = {{0, 1}};
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
  auto board = TestablePosition(R"(
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

  std::array<int, 2> expected_captures = {{2, 0}};
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
  auto board = TestablePosition(R"(
      ..X..
      .XOX.
      XO.OX
      .XOX.
      ..X..)");

  board.PlayMove("C7", Color::kBlack);

  std::array<int, 2> expected_captures = {{4, 0}};
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
  auto board = TestablePosition(R"(
      .OX
      OXX
      XX.)");

  board.PlayMove("A9", Color::kBlack);

  std::array<int, 2> expected_captures = {{2, 0}};
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
  auto board = TestablePosition(R"(
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
  auto board = TestablePosition(R"(
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

TEST(PositionTest, TestSuicidalMovesAreIllegal) {
  auto board = TestablePosition(R"(
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
    EXPECT_EQ(Position::MoveType::kIllegal, board.ClassifyMove(c));
  }
  std::vector<std::string> nonsuicidal_moves = {"B5", "J1", "A9"};
  for (const auto& c : nonsuicidal_moves) {
    EXPECT_NE(Position::MoveType::kIllegal, board.ClassifyMove(c));
  }
}

// Tests ko tracking.
TEST(PositionTest, TestKoTracking) {
  auto board = TestablePosition(R"(
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
  auto board = TestablePosition(R"(
    O....XXXX
    .....XOOO
    .....XOX.
    .....XO.X
    .....XOOX
    ......XOO
    ......XXX)");

  // All empty squares are neutral points and black has 5 more stones than
  // white.
  EXPECT_EQ(5, board.CalculateScore(0));
}

TEST(PositionTest, TestScoring) {
  EXPECT_EQ(0, TestablePosition("").CalculateScore(0));
  EXPECT_EQ(-42, TestablePosition("").CalculateScore(42));
  EXPECT_EQ(kN * kN, TestablePosition("X").CalculateScore(0));
  EXPECT_EQ(-kN * kN, TestablePosition("O").CalculateScore(0));

  {
    auto board = TestablePosition(R"(
    .X.
    X.O)");
    EXPECT_EQ(2, board.CalculateScore(0));
  }

  {
    auto board = TestablePosition(R"(
   .XX......
   OOXX.....
   OOOX...X.
   OXX......
   OOXXXXXX.
   OOOXOXOXX
   .O.OOXOOX
   .O.O.OOXX
   ......OOO)");
    EXPECT_EQ(1.5, board.CalculateScore(6.5));
  }

  {
    auto board = TestablePosition(R"(
   XXX......
   OOXX.....
   OOOX...X.
   OXX......
   OOXXXXXX.
   OOOXOXOXX
   .O.OOXOOX
   .O.O.OOXX
   ......OOO)");
    EXPECT_EQ(2.5, board.CalculateScore(6.5));
  }
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

  TestablePosition board("");
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

  std::array<int, 2> expected_captures = {{10, 2}};
  EXPECT_EQ(expected_captures, board.num_captures());
  EXPECT_EQ(-0.5, board.CalculateScore(kDefaultKomi));
}

// A regression test for a bug where Position::RemoveGroup didn't recycle the
// removed group's ID. The test plays repeatedly plays a random legal move (or
// passes if the player has no legal moves). Under these conditions, the game
// will never end.
TEST(PositionTest, PlayRandomLegalMoves) {
  Random rnd(983465983);
  TestablePosition position("");

  for (int i = 0; i < 10000; ++i) {
    std::vector<Coord> legal_moves;
    for (int c = 0; c < kN * kN; ++c) {
      if (position.ClassifyMove(c) != Position::MoveType::kIllegal) {
        legal_moves.push_back(c);
      }
    }
    if (!legal_moves.empty()) {
      auto c = legal_moves[rnd.UniformInt(0, legal_moves.size() - 1)];
      position.PlayMove(c, position.to_play());
    } else {
      position.PlayMove(Coord::kPass, position.to_play());
    }
  }
}

}  // namespace
}  // namespace minigo
