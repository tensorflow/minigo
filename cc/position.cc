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

#include <sstream>
#include <utility>

#include "absl/strings/str_format.h"
#include "cc/tiny_set.h"

namespace minigo {

namespace {

constexpr char kPrintWhite[] = "\x1b[0;31;47m";
constexpr char kPrintBlack[] = "\x1b[0;31;40m";
constexpr char kPrintEmpty[] = "\x1b[0;31;43m";
constexpr char kPrintNormal[] = "\x1b[0m";

}  // namespace

const std::array<inline_vector<Coord, 4>, kN* kN> kNeighborCoords = []() {
  std::array<inline_vector<Coord, 4>, kN * kN> result;
  for (int row = 0; row < kN; ++row) {
    for (int col = 0; col < kN; ++col) {
      auto& coords = result[row * kN + col];
      if (col > 0) {
        coords.emplace_back(row, col - 1);
      }
      if (col < kN - 1) {
        coords.emplace_back(row, col + 1);
      }
      if (row > 0) {
        coords.emplace_back(row - 1, col);
      }
      if (row < kN - 1) {
        coords.emplace_back(row + 1, col);
      }
    }
  }
  return result;
}();

Position::Position(BoardVisitor* bv, GroupVisitor* gv, Color to_play, int n)
    : board_visitor_(bv),
      group_visitor_(gv),
      to_play_(to_play),
      n_(n),
      stone_hash_(0) {}

Position::Position(BoardVisitor* bv, GroupVisitor* gv, const Position& position)
    : Position(position) {
  board_visitor_ = bv;
  group_visitor_ = gv;
}

void Position::PlayMove(Coord c, Color color) {
  if (c == Coord::kPass || c == Coord::kResign) {
    n_ += 1;
    ko_ = Coord::kInvalid;
    to_play_ = OtherColor(to_play_);
    return;
  }

  if (color == Color::kEmpty) {
    color = to_play_;
  } else {
    to_play_ = color;
  }

  MG_DCHECK(ClassifyMove(c) != MoveType::kIllegal) << c;

  AddStoneToBoard(c, color);

  n_ += 1;
  to_play_ = OtherColor(to_play_);
}

std::string Position::ToSimpleString() const {
  std::ostringstream oss;
  for (int row = 0; row < kN; ++row) {
    for (int col = 0; col < kN; ++col) {
      Coord c(row, col);
      auto color = stones_[c].color();
      if (color == Color::kWhite) {
        oss << "O";
      } else if (color == Color::kBlack) {
        oss << "X";
      } else {
        oss << (c == ko_ ? "*" : ".");
      }
    }
    if (row + 1 < kN) {
      oss << "\n";
    }
  }
  return oss.str();
}

std::string Position::ToPrettyString(bool use_ansi_colors) const {
  std::ostringstream oss;

  auto format_cols = [&oss]() {
    oss << "   ";
    for (int i = 0; i < kN; ++i) {
      oss << Coord::kGtpColumns[i] << " ";
    }
  };

  const char* print_white = use_ansi_colors ? kPrintWhite : "";
  const char* print_black = use_ansi_colors ? kPrintBlack : "";
  const char* print_empty = use_ansi_colors ? kPrintEmpty : "";
  const char* print_normal = use_ansi_colors ? kPrintNormal : "";

  format_cols();
  oss << "\n";
  for (int row = 0; row < kN; ++row) {
    oss << absl::StreamFormat("%2d ", kN - row);
    for (int col = 0; col < kN; ++col) {
      Coord c(row, col);
      auto color = stones_[c].color();
      if (color == Color::kWhite) {
        oss << print_white << "O ";
      } else if (color == Color::kBlack) {
        oss << print_black << "X ";
      } else {
        oss << print_empty << (c == ko_ ? "* " : ". ");
      }
    }
    oss << print_normal << absl::StreamFormat("%2d", kN - row);
    oss << "\n";
  }
  format_cols();
  return oss.str();
}

void Position::AddStoneToBoard(Coord c, Color color) {
  auto potential_ko = IsKoish(c);
  auto opponent_color = OtherColor(color);

  // Traverse the coord's neighbors, building useful information:
  //  - list of captured groups (if any).
  //  - coordinates of the new stone's liberties.
  //  - set of neighboring groups of each color.
  inline_vector<std::pair<GroupId, Coord>, 4> captured_groups;
  inline_vector<Coord, 4> liberties;
  tiny_set<GroupId, 4> opponent_groups;
  tiny_set<GroupId, 4> neighbor_groups;
  for (auto nc : kNeighborCoords[c]) {
    auto neighbor = stones_[nc];
    auto neighbor_color = neighbor.color();
    auto neighbor_group_id = neighbor.group_id();
    if (neighbor_color == Color::kEmpty) {
      // Remember the coord of this liberty.
      liberties.push_back(nc);
    } else if (neighbor_color == color) {
      // Remember neighboring groups of same color.
      neighbor_groups.insert(neighbor_group_id);
    } else if (neighbor_color == opponent_color) {
      // Decrement neighboring opponent group liberty counts and remember the
      // gorups we have captured. We'll remove them from the board shortly.
      if (opponent_groups.insert(neighbor_group_id)) {
        Group& opponent_group = groups_[neighbor_group_id];
        if (--opponent_group.num_liberties == 0) {
          captured_groups.emplace_back(neighbor_group_id, nc);
        }
      }
    }
  }

  // Place the new stone on the board.
  if (neighbor_groups.empty()) {
    // The stone doesn't connect to any neighboring groups: create a new group.
    stones_[c] = {color, groups_.alloc(1, liberties.size())};
  } else {
    // The stone connects to at least one neighbor: merge it into the first
    // group we found.
    auto group_id = neighbor_groups[0];
    if (neighbor_groups.size() == 1) {
      // Only one neighbor: update the group's size and liberty count, being
      // careful not to add count coords that were already liberties of the
      // group.
      Group& group = groups_[group_id];
      ++group.size;
      --group.num_liberties;
      for (auto nc : liberties) {
        if (!HasNeighboringGroup(nc, group_id)) {
          ++group.num_liberties;
        }
      }
      stones_[c] = {color, group_id};
    } else {
      // The stone joins multiple groups, merge them.
      // Incrementally updating the merged liberty counts is hard, so we just
      // recalculate the merged group's size and liberty count from scratch.
      // This is the relatively infrequent slow path.
      stones_[c] = {color, group_id};
      MergeGroup(c);
      for (int i = 1; i < neighbor_groups.size(); ++i) {
        groups_.free(neighbor_groups[i]);
      }
    }
  }
  stone_hash_ ^= zobrist::MoveHash(c, color);

  // Remove captured groups.
  for (const auto& p : captured_groups) {
    int num_captured_stones = groups_[p.first].size;
    if (color == Color::kBlack) {
      num_captures_[0] += num_captured_stones;
    } else {
      num_captures_[1] += num_captured_stones;
    }
    RemoveGroup(p.second);
  }

  // Update ko.
  if (captured_groups.size() == 1 &&
      groups_[captured_groups[0].first].size == 1 &&
      potential_ko == opponent_color) {
    ko_ = captured_groups[0].second;
  } else {
    ko_ = Coord::kInvalid;
  }
}

void Position::RemoveGroup(Coord c) {
  // Remember the first stone from the group we're about to remove.
  auto removed_color = stones_[c].color();
  auto other_color = OtherColor(removed_color);
  auto removed_group_id = stones_[c].group_id();

  group_visitor_->Begin();
  board_visitor_->Begin();
  board_visitor_->Visit(c);
  while (!board_visitor_->Done()) {
    c = board_visitor_->Next();

    MG_CHECK(stones_[c].group_id() == removed_group_id);
    stones_[c] = {};
    stone_hash_ ^= zobrist::MoveHash(c, removed_color);
    tiny_set<GroupId, 4> other_groups;
    for (auto nc : kNeighborCoords[c]) {
      auto ns = stones_[nc];
      auto neighbor_color = ns.color();
      auto neighbor_group_id = ns.group_id();
      if (neighbor_color == other_color) {
        if (other_groups.insert(neighbor_group_id)) {
          ++groups_[neighbor_group_id].num_liberties;
        }
      } else if (neighbor_color == removed_color) {
        board_visitor_->Visit(nc);
      }
    }
  }

  groups_.free(removed_group_id);
}

void Position::MergeGroup(Coord c) {
  Stone s = stones_[c];
  Color color = s.color();
  Color opponent_color = OtherColor(color);
  Group& group = groups_[s.group_id()];
  group.num_liberties = 0;
  group.size = 0;

  board_visitor_->Begin();
  board_visitor_->Visit(c);
  while (!board_visitor_->Done()) {
    c = board_visitor_->Next();
    if (stones_[c].color() == Color::kEmpty) {
      ++group.num_liberties;
    } else {
      MG_CHECK(stones_[c].color() == color);
      ++group.size;
      stones_[c] = s;
      for (auto nc : kNeighborCoords[c]) {
        if (stones_[nc].color() != opponent_color) {
          // We visit neighboring stones of the same color and empty coords.
          // Visiting empty coords through the BoardVisitor API ensures that
          // each one is only counted as a liberty once, even if it is
          // neighbored by multiple stones in this group.
          board_visitor_->Visit(nc);
        }
      }
    }
  }
}

Color Position::IsKoish(Coord c) const {
  if (!stones_[c].empty()) {
    return Color::kEmpty;
  }

  Color ko_color = Color::kEmpty;
  for (Coord nc : kNeighborCoords[c]) {
    Stone s = stones_[nc];
    if (s.empty()) {
      return Color::kEmpty;
    }
    if (s.color() != ko_color) {
      if (ko_color == Color::kEmpty) {
        ko_color = s.color();
      } else {
        return Color::kEmpty;
      }
    }
  }
  return ko_color;
}

Position::MoveType Position::ClassifyMove(Coord c) const {
  if (c == Coord::kPass || c == Coord::kResign) {
    return MoveType::kNoCapture;
  }
  if (!stones_[c].empty()) {
    return MoveType::kIllegal;
  }
  if (c == ko_) {
    return MoveType::kIllegal;
  }

  auto result = MoveType::kIllegal;
  auto other_color = OtherColor(to_play_);
  for (auto nc : kNeighborCoords[c]) {
    Stone s = stones_[nc];
    if (s.empty()) {
      // At least one liberty at nc after playing at c.
      if (result == MoveType::kIllegal) {
        result = MoveType::kNoCapture;
      }
    } else if (s.color() == other_color) {
      if (groups_[s.group_id()].num_liberties == 1) {
        // Will capture opponent group that has a stone at nc.
        result = MoveType::kCapture;
      }
    } else {
      if (groups_[s.group_id()].num_liberties > 1) {
        // Connecting to a same colored group at nc that has more than one
        // liberty.
        if (result == MoveType::kIllegal) {
          result = MoveType::kNoCapture;
        }
      }
    }
  }
  return result;
}

bool Position::HasNeighboringGroup(Coord c, GroupId group_id) const {
  for (auto nc : kNeighborCoords[c]) {
    Stone s = stones_[nc];
    if (!s.empty() && s.group_id() == group_id) {
      return true;
    }
  }
  return false;
}

float Position::CalculateScore(float komi) {
  int score = 0;

  static_assert(static_cast<int>(Color::kEmpty) == 0, "Color::kEmpty != 0");
  static_assert(static_cast<int>(Color::kBlack) == 1, "Color::kBlack != 1");
  static_assert(static_cast<int>(Color::kWhite) == 2, "Color::kWhite != 2");

  auto score_empty_area = [this](Coord c) {
    int num_visited = 0;
    int found_bits = 0;
    do {
      c = board_visitor_->Next();
      ++num_visited;
      for (auto nc : kNeighborCoords[c]) {
        Color color = stones_[nc].color();
        if (color == Color::kEmpty) {
          board_visitor_->Visit(nc);
        } else {
          found_bits |= static_cast<int>(color);
        }
      }
    } while (!board_visitor_->Done());

    if (found_bits == 1) {
      return num_visited;
    } else if (found_bits == 2) {
      return -num_visited;
    } else {
      return 0;
    }
  };

  group_visitor_->Begin();
  board_visitor_->Begin();
  for (int row = 0; row < kN; ++row) {
    for (int col = 0; col < kN; ++col) {
      Coord c(row, col);
      Stone s = stones_[c];
      if (s.empty()) {
        if (board_visitor_->Visit(c)) {
          // First time visiting this empty coord.
          score += score_empty_area(c);
        }
      } else if (group_visitor_->Visit(s.group_id())) {
        // First time visiting this group of stones.
        auto size = groups_[s.group_id()].size;
        if (s.color() == Color::kBlack) {
          score += size;
        } else {
          score -= size;
        }
      }
    }
  }

  return static_cast<float>(score) - komi;
}

}  // namespace minigo
