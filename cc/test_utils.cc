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

#include "cc/test_utils.h"

#include <utility>
#include <vector>

#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "cc/constants.h"
#include "cc/logging.h"

namespace minigo {

namespace {

// Splits a simple board representation into multiple lines, stripping
// whitespace. Lines are padded with '.' to ensure a kN * kN board.
std::vector<std::string> SplitBoardString(absl::string_view str) {
  std::vector<std::string> lines;
  for (const auto& line : absl::StrSplit(str, '\n')) {
    std::string stripped(absl::StripAsciiWhitespace(line));
    if (stripped.empty()) {
      continue;
    }
    stripped = absl::StrReplaceAll(stripped, {{" ", ""}});
    MG_CHECK(stripped.size() <= kN);
    stripped.resize(kN, '.');
    lines.push_back(std::move(stripped));
  }
  MG_CHECK(lines.size() <= kN);
  while (lines.size() < kN) {
    lines.emplace_back(kN, '.');
  }
  return lines;
}

}  // namespace

std::string CleanBoardString(absl::string_view str) {
  return absl::StrJoin(SplitBoardString(str), "\n");
}

TestablePosition::TestablePosition(absl::string_view board_str, Color to_play)
    : TestablePosition(ParseBoard(board_str), to_play) {}

TestablePosition::TestablePosition(const std::array<Color, kN * kN>& stones,
                                   Color to_play)
    : Position(&board_visitor, &group_visitor, to_play) {
  for (int i = 0; i < kN * kN; ++i) {
    if (stones[i] != Color::kEmpty) {
      AddStoneToBoard(i, stones[i]);
    }
  }
  UpdateLegalMoves(nullptr);
}

Coord GetRandomLegalMove(const Position& position, Random* rnd) {
  std::vector<Coord> valid_moves;
  for (int i = 0; i < kN * kN; ++i) {
    if (position.legal_move(i)) {
      valid_moves.push_back(i);
    }
  }
  if (valid_moves.empty()) {
    valid_moves.push_back(Coord::kPass);
  }
  return valid_moves[rnd->UniformInt(0, valid_moves.size() - 1)];
}

std::array<Color, kN * kN> ParseBoard(absl::string_view str) {
  std::array<Color, kN * kN> result;
  auto lines = SplitBoardString(str);
  for (int row = 0; row < kN; ++row) {
    for (int col = 0; col < kN; ++col) {
      Coord c(row, col);
      if (lines[row][col] == 'X') {
        result[c] = Color::kBlack;
      } else if (lines[row][col] == 'O') {
        result[c] = Color::kWhite;
      } else {
        result[c] = Color::kEmpty;
      }
    }
  }
  return result;
}

int CountPendingVirtualLosses(const MctsNode* node) {
  int num = 0;
  std::vector<const MctsNode*> pending{node};
  while (!pending.empty()) {
    node = pending.back();
    pending.pop_back();
    MG_CHECK(node->num_virtual_losses_applied >= 0);
    num += node->num_virtual_losses_applied;
    for (const auto& p : node->children) {
      pending.push_back(p.second.get());
    }
  }
  return num;
}

}  // namespace minigo
