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

#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "cc/constants.h"
#include "cc/logging.h"
#include "cc/position.h"
#include "cc/random.h"
#include "cc/test_utils.h"
#include "gtest/gtest.h"

namespace minigo {
namespace {

// Copied out of Position::ToPrettyString because it operates on an array of
// Stones, not Colors and we can't construct a Position from the results of
// filling pass-alive regions with stones because the filled areas would
// suicide :(
std::string ToPrettyString(const std::array<Color, kN * kN>& stones) {
  std::ostringstream oss;

  auto format_cols = [&oss]() {
    oss << "   ";
    for (int i = 0; i < kN; ++i) {
      oss << Coord::kGtpColumns[i] << " ";
    }
  };

  const char* print_white = "\x1b[0;31;47m";
  const char* print_black = "\x1b[0;31;40m";
  const char* print_empty = "\x1b[0;31;43m";
  const char* print_normal = "\x1b[0m";

  format_cols();
  oss << "\n";
  for (int row = 0; row < kN; ++row) {
    oss << absl::StreamFormat("%2d ", kN - row);
    for (int col = 0; col < kN; ++col) {
      Coord c(row, col);
      auto color = stones[c];
      if (color == Color::kWhite) {
        oss << print_white << "O ";
      } else if (color == Color::kBlack) {
        oss << print_black << "X ";
      } else {
        oss << print_empty << ". ";
      }
    }
    oss << print_normal << absl::StreamFormat("%2d", kN - row);
    oss << "\n";
  }
  format_cols();
  return oss.str();
}

class PassAliveTest : public ::testing::Test {
 protected:
  struct TestCase {
    TestCase(const std::string& board, const std::string& expected)
        : board(board), expected(ParseBoard(expected)) {}

    TestablePosition board;
    std::array<Color, kN * kN> expected;
  };

  void RunTests(absl::Span<const TestCase> tests) {
    for (const auto& test : tests) {
      // MG_LOG(INFO) << "board state:\n" << test.board.ToPrettyString();
      auto black = test.board.CalculatePassAliveRegions(Color::kBlack);
      auto white = test.board.CalculatePassAliveRegions(Color::kWhite);

      // Initialize the result to the input board state.
      std::array<Color, kN * kN> actual;
      for (size_t i = 0; i < kN * kN; ++i) {
        actual[i] = test.board.stones()[i].color();
      }

      // Merge both pass-alive regions into the result.
      for (size_t i = 0; i < kN * kN; ++i) {
        MG_CHECK(black[i] == Color::kEmpty || white[i] == Color::kEmpty)
            << Coord(i).ToGtp()
            << " was marked as belonging to both black & white pass-alive "
               "regions";
        if (black[i] != Color::kEmpty) {
          actual[i] = Color::kBlack;
        } else if (white[i] != Color::kEmpty) {
          actual[i] = Color::kWhite;
        }
      }

      MG_CHECK(test.expected == actual)
          << "\nexpected:\n"
          << ToPrettyString(test.expected) << "\n\nactual:\n"
          << ToPrettyString(actual) << "\n";

      // MG_LOG(INFO) << "expected:\n" << ToPrettyString(test.expected);
      // MG_LOG(INFO) << "actual:\n" << ToPrettyString(actual);
    }
  }
};

TEST_F(PassAliveTest, 9x9) {
  if (kN != 9) {
    return;
  }

  TestCase tests[] = {
      {// board state
       R"(. X . X O . . . .
          X X X X O . . . .
          O O O O O . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .)",
       // expected result
       R"(X X X X O . . . .
          X X X X O . . . .
          O O O O O . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .)"},

      {// board state
       R"(. . . . O . O . .
          . . . . O O O . .
          . . . . . . O O O
          . . . . . . O . .
          . . . . . . O O O
          . . . . . . O X X
          . . . . . . O X .
          . . . . . . O X X
          . . . . . . O X .)",
       // expected result
       R"(. . . . O O O O O
          . . . . O O O O O
          . . . . . . O O O
          . . . . . . O O O
          . . . . . . O O O
          . . . . . . O X X
          . . . . . . O X X
          . . . . . . O X X
          . . . . . . O X X)"},

      {// board state
       R"(. . . X . . . . .
          X X X X . . . . .
          . X . X . . . . .
          X X X X . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .)",
       // expected result
       R"(X X X X . . . . .
          X X X X . . . . .
          X X X X . . . . .
          X X X X . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .)"},

      {// Top-left region is not pass-alive because it has two empty points that
       // aren't adjacent to the enclosing chain.
       // board state
       R"(. . . X . . . . .
          . . . X . . . . .
          X X X X . . . . .
          . X . X . . . . .
          X X X X . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .)",
       // expected result
       R"(. . . X . . . . .
          . . . X . . . . .
          X X X X . . . . .
          X X X X . . . . .
          X X X X . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .)"},

      {// Top-left region is pass-alive because only one empty point is not
       // adjacent to the enclosing chain.
       // board state
       R"(O . . X . . . . .
          . . . X . . . . .
          X X X X . . . . .
          . X . X . . . . .
          X X X X . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .)",
       // expected result
       R"(X X X X . . . . .
          X X X X . . . . .
          X X X X . . . . .
          X X X X . . . . .
          X X X X . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .)"},

      {// board state
       R"(. . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          O O O O O . . . .
          O . O . O . . . .)",
       // expected result
       R"(. . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          O O O O O . . . .
          O O O O O . . . .)"},

      {// board state
       R"(. . . . . . . . .
          . . X X X X X X .
          . . X . . X . X .
          . . X X X X X X .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .)",
       // expected result
       R"(. . . . . . . . .
          . . X X X X X X .
          . . X X X X X X .
          . . X X X X X X .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .)"},

      {// board state
       R"(. . . . . . . . .
          . O O O O O O O .
          . O . O X X X O .
          . O O O X X X O .
          . O X . X X X O .
          . O X X X X O O .
          . O O O O O O . .
          . . . . . . . . .
          . . . . . . . . .)",
       // expected result
       R"(. . . . . . . . .
          . O O O O O O O .
          . O O O O O O O .
          . O O O O O O O .
          . O O O O O O O .
          . O O O O O O O .
          . O O O O O O . .
          . . . . . . . . .
          . . . . . . . . .)"},

      {// board state
       R"(. . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          O O O O O . . . .
          . O . X O O . . .
          O . O . X O . . .)",
       // expected result
       R"(. . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          O O O O O . . . .
          O O O O O O . . .
          O O O O O O . . .)"},

      {// board state
       R"(O X X . X . X . O
          O X . X O X X X X
          O X X O O O O O O
          O O O O . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .)",
       // expected result
       R"(O X X X X X X X X
          O X X X O X X X X
          O X X O O O O O O
          O O O O . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .)"},

      {// board state
       R"(. . O . O . X X .
          . O O O O . X . X
          O O . . . . . X X
          . . . . . . . . .
          . . . . . . . . .
          X . . . . . . O O
          . X X . . . O . O
          X . X . . O O . O
          . X . X . O . O .)",
       // expected result
       R"(. . O . O . X X X
          . O O O O . X X X
          O O . . . . . X X
          . . . . . . . . .
          . . . . . . . . .
          X . . . . . . O O
          . X X . . . O O O
          X . X . . O O O O
          . X . X . O O O O)"},

      {// board state
       R"(. O O O O O O O .
          O O X . . X X . O
          O . . . X O X . O
          O . . . X . X . O
          O X X X . X X . O
          O X O . X . . . O
          O X . X X . X X O
          O . X X . . O O O
          . O O O O O O X .)",
       // expected result
       R"(O O O O O O O O O
          O O X . . X X . O
          O . . . X X X . O
          O . . . X X X . O
          O X X X X X X . O
          O X X X X . . . O
          O X X X X . X X O
          O . X X . . O O O
          O O O O O O O O O)"},

      {// board state
       R"(. . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .)",
       // expected result
       R"(. . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .)"},

      {// board state
       R"(. . . . . . . . .
          . . . . . . . . .
          . . . X X . . . .
          . . X . . X . . .
          . . X . . X . . .
          . . . X X . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .)",
       // expected result
       R"(. . . . . . . . .
          . . . . . . . . .
          . . . X X . . . .
          . . X . . X . . .
          . . X . . X . . .
          . . . X X . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .)"},

      {// board state
       R"(. . . . . . . . .
          . . . . . . . . .
          . X X X X . . . .
          . X . . X . . . .
          . X . . X X X . .
          . X X X X . X . .
          . . . . X X X . .
          . . . . . . . . .
          . . . . . . . . .)",
       // expected result
       R"(. . . . . . . . .
          . . . . . . . . .
          . X X X X . . . .
          . X X X X . . . .
          . X X X X X X . .
          . X X X X X X . .
          . . . . X X X . .
          . . . . . . . . .
          . . . . . . . . .)"},

      {// board state
       R"(. . . . . . . . .
          . X X X X X . . .
          . X . . . X . . .
          . X . . . X . . .
          . X . . . X X X .
          . X X X X X . X .
          . . . . . X X X .
          . . . . . . . . .
          . . . . . . . . .)",
       // expected result
       R"(. . . . . . . . .
          . X X X X X . . .
          . X . . . X . . .
          . X . . . X . . .
          . X . . . X X X .
          . X X X X X . X .
          . . . . . X X X .
          . . . . . . . . .
          . . . . . . . . .)"},

      {// board state
       R"(. . . . . . . . .
          . . . . . . . . .
          . . . . X . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .)",
       // expected result
       R"(. . . . . . . . .
          . . . . . . . . .
          . . . . X . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .)"},

      {// board state
       R"(. . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . O . . . .
          . . . . . . . . .)",
       // expected result
       R"(. . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . . . . . .
          . . . . O . . . .
          . . . . . . . . .)"},

      {// board state
       R"(O . . . . . . . .
            . . . . O O O O.
            . O O O O X X O.
            . O X X X . X O.
            . O X . . X X O.
            . O X X X O O O.
            . O O O O O . ..
            . . . . . . . ..
            . . . . . . . ..)",
       // expected result
       R"(O . . . . . . . .
          . . . . O O O O .
          . O O O O X X O .
          . O X X X . X O .
          . O X . . X X O .
          . O X X X O O O .
          . O O O O O . . .
          . . . . . . . . .
          . . . . . . . . .)"},

      {// Seki
       // board state
       R"(. O X . X O . . .
          O O O X X O . . .
          . O X X O O . . .
          O X X X O . . . .
          X X . X O . . . .
          O O X X O . . . .
          . O O O O . . . .
          . . . . . . . . .
          . . . . . . . . .)",
       // expected result
       R"(. O X . X O . . .
          O O O X X O . . .
          . O X X O O . . .
          O X X X O . . . .
          X X . X O . . . .
          O O X X O . . . .
          . O O O O . . . .
          . . . . . . . . .
          . . . . . . . . .)"},
  };

  RunTests(tests);
}

TEST_F(PassAliveTest, 19x19) {
  if (kN != 19) {
    return;
  }

  TestCase tests[] = {
      {// board state
       R"(X . X . . . . O O . . . . . . . X . X
          . X X . . . . O . O O O . . . . X X .
          . X . . . . . O O . . O . . . . . X O
          X X . O O . . . . O O O . . . . . X X
          . . . O . O O O . . . . . . . . . . .
          . . . O O . X O . . . . . . . . . . .
          . . . . . O O O . . . . . . . . . . .
          . . . . . . . . . . . . . . . . . . .
          . X X X . . . . . . . . . . . . . . .
          . X . X . . . . . . . . . . . . . . .
          . X . X . . . . . . . . . . . . O O O
          X . X . . . . . . . . . . . . . O . O
          . X X . . . . . . . . . . . . . . O .
          X X . . . . . . . . . . . . . . . . O
          . . . . . . . . . . . . . . . . . O .
          O O . . . . . . . . . . . . . . . O O
          . O . . . . . O O . . . . . . . . O .
          . O O O O . O . O . . . . . . . . O O
          O . . O . O . O O . . . . . . . O . .)",
       // expected result
       R"(X . X . . . . O O . . . . . . . X X X
          . X X . . . . O . O O O . . . . X X X
          . X . . . . . O O . . O . . . . . X X
          X X . O O . . . . O O O . . . . . X X
          . . . O O O O O . . . . . . . . . . .
          . . . O O O O O . . . . . . . . . . .
          . . . . . O O O . . . . . . . . . . .
          . . . . . . . . . . . . . . . . . . .
          . X X X . . . . . . . . . . . . . . .
          . X X X . . . . . . . . . . . . . . .
          . X X X . . . . . . . . . . . . O O O
          X X X . . . . . . . . . . . . . O O O
          X X X . . . . . . . . . . . . . . O O
          X X . . . . . . . . . . . . . . . . O
          . . . . . . . . . . . . . . . . . O O
          O O . . . . . . . . . . . . . . . O O
          . O . . . . . O O . . . . . . . . O O
          . O O O O . O . O . . . . . . . . O O
          O . . O . O . O O . . . . . . . O . .)"},

      {// board state
       R"(. X . X . X X X X . X X X X . X . X .
          X . X X X X . . X . X . . X X X X . X
          X X . X . . X . X . X . X . . X . X .
          X . X . X X . X X . X X . X X . X . X
          . X . X X . X X . . . X X . X X . X .
          . X . X . . . . . . . . . . . X . X .
          X . X . . . . . . . . . . . . . X . X
          . X . . . . . . . . . . . . . . . X .
          . . . . . . . . . . . . . . . . . . .
          X X X . . . . . X X X X X . . . . . .
          . . X . . . . . X . . . X . X X X X X
          O X X X X . . . X . . X X . X . . . .
          O . X . X . . . X . O . X X X . . . .
          . . X X X . . . X . . . X . X . . . .
          X X X . . . . . X X X X X X X X X X X
          . . . . . . . . . . . . . . O . . . .
          O O O O O . . . . O O O O O . O O O O
          . O . . O O . . . O . . . O O O . . .
          O . O . X O . . . O . O . O . O . O O)",
       // expected result
       R"(X X X X X X X X X . X X X X . X . X .
          X X X X X X . . X . X . . X X X X . X
          X X . X . . X . X . X . X . . X . X .
          X . X . X X . X X . X X . X X . X . X
          . X . X X . X X . . . X X . X X . X .
          . X . X . . . . . . . . . . . X . X .
          X . X . . . . . . . . . . . . . X . X
          . X . . . . . . . . . . . . . . . X .
          . . . . . . . . . . . . . . . . . . .
          X X X . . . . . X X X X X . . . . . .
          X X X . . . . . X X X X X . X X X X X
          X X X X X . . . X X X X X . X . . . .
          X X X X X . . . X X X X X X X . . . .
          X X X X X . . . X X X X X X X . . . .
          X X X . . . . . X X X X X X X X X X X
          . . . . . . . . . . . . . . O . . . .
          O O O O O . . . . O O O O O . O O O O
          . O . . O O . . . O . . . O O O . . .
          O . O . X O . . . O . O . O . O . O O)"},

      {// board state
       R"(. X . X . X X X X . X X X X . X . X .
          X . X X X X . . X . X . . X X X X . X
          X X . X . . X . X . X . X . . X . X .
          X . X . X X . X X . X X . X X . X . X
          . X . X X . X X . . . X X . X X . X .
          . X . X . . . . . . . . . . . X . X .
          X . X . . . . . . . . . . . . . X . X
          . X . . . . . . . . . . . . . . . X .
          . . . . . . . . . . . . . . . . . . .
          X X X . . . . . X X X X X . . . . . .
          . . X . . . . . X . . . X . X X X X X
          O X X X X . . . X . . X X . X . . . .
          . . X . X . . . X . . . X X X . . . .
          . . X X X . . . X . . . X . X . . . .
          X X X . . . . . X X X X X X X X X X X
          . . . . . . . . . . . . . . O . . . .
          O O O O O . . . . O O O O O . O O O O
          . O . . O O . . . O . . . O O O . . .
          O . O . . O . . . O . O . O . O . O O)",
       // expected result
       R"(X X X X X X X X X . X X X X . X . X .
          X X X X X X . . X . X . . X X X X . X
          X X . X . . X . X . X . X . . X . X .
          X . X . X X . X X . X X . X X . X . X
          . X . X X . X X . . . X X . X X . X .
          . X . X . . . . . . . . . . . X . X .
          X . X . . . . . . . . . . . . . X . X
          . X . . . . . . . . . . . . . . . X .
          . . . . . . . . . . . . . . . . . . .
          X X X . . . . . X X X X X . . . . . .
          . . X . . . . . X . . . X . X X X X X
          O X X X X . . . X . . X X . X . . . .
          . . X . X . . . X . . . X X X . . . .
          . . X X X . . . X . . . X . X . . . .
          X X X . . . . . X X X X X X X X X X X
          . . . . . . . . . . . . . . O . . . .
          O O O O O . . . . O O O O O . O O O O
          . O . . O O . . . O . . . O O O . . .
          O . O . . O . . . O . O . O . O . O O)"},

      {// Whole board seki
       // board state
       R"(O . X X X O O O O . O . X O O O . O .
          . O O X X X X O O O X X X X O . O X X
          X O X X . X O O O O O O O X O O O O X
          X X O O X X O X X O X X O X O X X X X
          X X O O O O X X X X . X O X X . X O X
          . X X O . X X X O O X X O X O X X O O
          O X O O O O O O O X X O O O O O X X O
          O O X X X O X . O O X X O . O O X O O
          O O X X X O O O O X X . X O O X O O X
          O . O X X O X O X X X X X O O X . . X
          O O O X X X X X O X O X X O X X X X X
          X O X . X X O O O O O O X X O O O O O
          X X X X X O O O O X . O O X O . . X O
          X X O O X O O X X X O O X O O X X X X
          X O O . X O O O X X O O X X X O O O O
          X X O X X O X O O X . O O X . O X O .
          X O O X O X X X X X O O X X O O X O O
          X X O O O X X X O X O . O X X O X X X
          X O O . O X . X O O O O O X . X . . .)",
       // expected result
       R"(O . X X X O O O O . O . X O O O . O .
          . O O X X X X O O O X X X X O . O X X
          X O X X . X O O O O O O O X O O O O X
          X X O O X X O X X O X X O X O X X X X
          X X O O O O X X X X . X O X X . X O X
          . X X O . X X X O O X X O X O X X O O
          O X O O O O O O O X X O O O O O X X O
          O O X X X O X . O O X X O . O O X O O
          O O X X X O O O O X X . X O O X O O X
          O . O X X O X O X X X X X O O X . . X
          O O O X X X X X O X O X X O X X X X X
          X O X . X X O O O O O O X X O O O O O
          X X X X X O O O O X . O O X O . . X O
          X X O O X O O X X X O O X O O X X X X
          X O O . X O O O X X O O X X X O O O O
          X X O X X O X O O X . O O X . O X O .
          X O O X O X X X X X O O X X O O X O O
          X X O O O X X X O X O . O X X O X X X
          X O O . O X . X O O O O O X . X . . .)"},
  };

  RunTests(tests);
}

}  // namespace
}  // namespace minigo
