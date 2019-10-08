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
      auto c = Coord(row, col).ToGtp();
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

  {
    // The bottom right corner is a black pass-alive territory and so should
    // count as 12 points for black.
    auto board = TestablePosition(R"(
   .........
   .........
   .........
   .......XX
   ..O....XO
   .......X.
   .......XX
   .......X.
   .......XX)");
    EXPECT_EQ(4.5, board.CalculateScore(6.5));
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

void ValidatePosition(TestablePosition* p) {
  auto calculate_group_info = [p](Coord c) {
    Group group;

    const auto& stones = p->stones();
    auto color = stones[c].color();
    auto expected_group_id = stones[c].group_id();
    MG_CHECK(color != Color::kEmpty);

    auto other_color = OtherColor(color);
    BoardVisitor bv;
    bv.Begin();
    bv.Visit(c);
    while (!bv.Done()) {
      c = bv.Next();
      if (stones[c].color() == Color::kEmpty) {
        group.num_liberties += 1;
      } else {
        MG_CHECK(stones[c].group_id() == expected_group_id);
        group.size += 1;
        for (auto nc : kNeighborCoords[c]) {
          if (stones[nc].color() != other_color) {
            bv.Visit(nc);
          }
        }
      }
    }

    return group;
  };

  for (int i = 0; i < kN * kN; ++i) {
    auto c = static_cast<Coord>(i);
    if (p->stones()[c].empty()) {
      continue;
    }
    auto expected_group = calculate_group_info(c);
    auto actual_group = p->GroupAt(c);
    MG_CHECK(expected_group.size == actual_group.size)
        << c << " : expected_group_size:" << expected_group.size
        << " actual_group_size:" << actual_group.size;
    MG_CHECK(expected_group.num_liberties == actual_group.num_liberties)
        << c << " : expected_num_liberties:" << expected_group.num_liberties
        << " actual_num_liberties:" << actual_group.num_liberties;
  }
}

TEST(PositionTest, UndoMove) {
  // Play a move at point `c` with color `color` on a board generated from
  // `board_str`. Then undo the move again and validate the board state.
  auto test_undo = [](const std::string& c, Color color,
                      const std::string& board_str) {
    TestablePosition board(board_str);
    ValidatePosition(&board);

    std::array<Color, kN * kN> original_stones;
    for (int i = 0; i < kN * kN; ++i) {
      original_stones[i] = board.stones()[i].color();
    }

    auto ko = board.ko();
    auto undo = board.PlayMove(Coord::FromGtp(c), color);
    ValidatePosition(&board);
    board.UndoMove(undo);
    MG_CHECK(board.ko() == ko);
    ValidatePosition(&board);

    std::array<Color, kN * kN> undone_stones;
    for (int i = 0; i < kN * kN; ++i) {
      undone_stones[i] = board.stones()[i].color();
    }
    MG_CHECK(original_stones == undone_stones);
  };

  // Test that undo correctly updates liberty counts.
  test_undo("B9", Color::kBlack, R"(
     X.
     X.)");
  test_undo("D8", Color::kWhite, R"(
     XXX.
     XOO.
     XXX.)");
  test_undo("B8", Color::kWhite, R"(
     XXX
     X..
     XXX)");
  test_undo("C9", Color::kBlack, R"(
     OO.OX
     OXXOX
     .X.X.)");

  // Test that nothing explodes when we undo a pass.
  test_undo("pass", Color::kWhite, R"()");

  // Test undoing a single move.
  test_undo("C3", Color::kWhite, R"()");

  // Test that undo correctly restores a single captured group.
  test_undo("C8", Color::kWhite, R"(
     .O.
     OX.
     .O.)");
  test_undo("C7", Color::kBlack, R"(
     XXXXX
     XOOOX
     XO.OX
     XOOOX
     XXXXX)");

  // Test that undo correctly restores a multiple captured groups.
  test_undo("D6", Color::kBlack, R"(
     ...X...
     ..XOX..
     .XXOXX.
     XOO.OOX
     .XXOXX.
     ..XOX..
     ...X...)");

  // Test that undo splits a group that was joined by the undone move.
  test_undo("B8", Color::kWhite, R"(
     .O.
     O.O
     .O.)");
  test_undo("B9", Color::kBlack, R"(
     X.X
     X.X
     X.X)");

  // Test that undo doesn't split a group that's joined in another location.
  test_undo("B9", Color::kBlack, R"(
     X.X
     X.X
     XXX)");

  // Test that undo handles ko correctly.
  test_undo("C8", Color::kBlack, R"(
     .XO.
     XO.O
     .XO.)");
}

// A regression test for a bug where Position::RemoveGroup didn't recycle the
// removed group's ID. The test plays repeatedly plays a random legal move (or
// passes if the player has no legal moves). Under these conditions, the game
// will never end.
TEST(PositionTest, PlayRandomLegalMoves) {
  Random rnd(983465983, 1);
  TestablePosition position("");

  struct State {
    State(zobrist::Hash stone_hash, const Position::UndoState& undo)
        : stone_hash(stone_hash), undo(undo) {}
    zobrist::Hash stone_hash;
    Position::UndoState undo;
  };

  std::vector<State> states;
  for (int i = 0; i < 10000; ++i) {
    std::vector<Coord> legal_moves;
    for (int c = 0; c < kN * kN; ++c) {
      if (position.legal_move(c)) {
        legal_moves.push_back(c);
      }
    }

    auto stone_hash = position.stone_hash();
    if (!legal_moves.empty()) {
      auto c = legal_moves[rnd.UniformInt(0, legal_moves.size() - 1)];
      states.emplace_back(stone_hash, position.PlayMove(c, position.to_play()));
    } else {
      states.emplace_back(stone_hash,
                          position.PlayMove(Coord::kPass, position.to_play()));
    }
  }

  // Undo all the moves, validating as we go.
  ValidatePosition(&position);
  while (!states.empty()) {
    const auto& state = states.back();
    position.UndoMove(state.undo);
    ValidatePosition(&position);
    MG_CHECK(position.stone_hash() == state.stone_hash);
    states.pop_back();
  }
}

}  // namespace
}  // namespace minigo
