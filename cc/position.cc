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

#include <algorithm>
#include <array>
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

Position::Position(BoardVisitor* bv, GroupVisitor* gv, Color to_play)
    : board_visitor_(bv), group_visitor_(gv), to_play_(to_play) {
  // All moves are initially legal.
  std::fill(legal_moves_.begin(), legal_moves_.end(), true);
}

Position::Position(BoardVisitor* bv, GroupVisitor* gv, const Position& position)
    : Position(position) {
  board_visitor_ = bv;
  group_visitor_ = gv;
}

void Position::PlayMove(Coord c, Color color, ZobristHistory* zobrist_history) {
  if (c == Coord::kPass || c == Coord::kResign) {
    ko_ = Coord::kInvalid;
  } else {
    if (color == Color::kEmpty) {
      color = to_play_;
    } else {
      to_play_ = color;
    }
    MG_DCHECK(ClassifyMove(c) != MoveType::kIllegal) << c;
    AddStoneToBoard(c, color);
  }

  n_ += 1;
  to_play_ = OtherColor(to_play_);
  UpdateLegalMoves(zobrist_history);
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
      // groups we have captured. We'll remove them from the board shortly.
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

// A _region_ is a connected set of intersections regardless of color.
// A _black-enclosed region_ is a maximum region containig no black stones.
// A black-enclosed region is _small_ if all of its empty intersections are
// liberties of the enclosing black stones.
// A small black-enclosed region is _vital_ to an enclosing chain if all of its
// empty intersections are liberties of that chain. Note that a small
// black-enclosed region may not be vital to any of the enclosing chains. For
// example:
//   . . . . . .
//   . . X X . .
//   . X . . X .
//   . X . . X .
//   . . X X . .
//   . . . . . .
//
// A set of black chains X is _unconditionally alive_ if each chain in X has at
// least two distinct small black-enclosed regions that are vital to it.
// A region enclosed by set of unconitionally alive black chains is an
// unconditionally alive black region.
//
// Given these definitions, Benson's Algorithm finds the set of unconditionally
// alive black regions as follows:
//  - Let X be the set of all black chains.
//  - Let R be the set of small black-enclosed regions of X.
//  - Iterate the following two steps until neither one removes an item:
//    - Remove from X all black chains with fewer than two vital
//      black-enclosed regions in R.
//    - Remove from R all black-enclosed regions with a surrounding stone in a
//      chain not in X.
//
// Unconditionally alive chains are also called pass-alive because they cannot
// be captured by the opponent even if that player always passes on their turn.
// More details:
//   https://senseis.xmp.net/?BensonsDefinitionOfUnconditionalLife
std::array<Color, kN * kN> Position::CalculatePassAliveRegions(
    Color color) const {
  constexpr auto kMaxNumRegions = (kN * kN + 1) / 2 + 1;
  constexpr auto kMaxNumGroups = kN * kN;  // A safe over-estimate

  struct BensonGroup {
    explicit BensonGroup(int liberties_begin)
        : liberties_begin(static_cast<uint16_t>(liberties_begin)) {}
    // This group's liberties.
    // See the comments for the liberties array below for more details.
    uint16_t liberties_begin;
    uint16_t num_liberties = 0;

    // The number of vital regions that this group encloses. The BensonGroup
    // itself doesn't keep track of which of its neighboring regions are vital,
    // it is sufficient for the BensonGroup to track which of its enclosing
    // groups are vital.
    uint16_t num_vital_regions = 0;

    // Whether the group has been determined to be pass-alive.
    bool is_pass_alive = false;
  };

  struct BensonRegion {
    BensonRegion(int empty_points_begin, int groups_begin)
        : empty_points_begin(static_cast<uint16_t>(empty_points_begin)),
          groups_begin(static_cast<uint16_t>(groups_begin)) {}
    // This region's empty points.
    // See the comments for the regions array below for more details.
    uint16_t empty_points_begin;
    uint16_t num_empty_points = 0;

    // This region's groups.
    // See the comments for the groups array below for more details.
    uint16_t groups_begin;
    uint16_t num_enclosing_groups = 0;
    uint16_t num_vital_groups = 0;

    // Whether the region has been determined to be pass-alive.
    bool is_pass_alive = false;
  };

  // Storage for liberties of all groups.
  // Each BensonGroup has num_liberties liberties. The coordinates of the
  // i'th liberty of a group are stored at:
  //   liberties[group->liberties_begin + i].
  // We over allocate by 4x because during the process of computing liberties,
  // each liberty may be added mulitple times (before deduplication happens
  // later).
  // The list of liberties for each BensonGroup is sorted by coordinate,
  // so that the list of vital regions for a group can be efficiently found.
  inline_vector<Coord, kMaxNumGroups * 4> liberties;

  // Storage for coordinates of empty points in regions.
  // Each BensonRegion has BensonRegion::num_empty_points empty points. The
  // coordinates of the i'th empty point of a region are stored at
  //   empty_points[region->empty_points_begin + i].
  // The list of empty points for each BensonRegion is sorted by coordinate,
  // so that the list of vital regions for a group can be efficiently found.
  inline_vector<Coord, kN * kN> empty_points;

  // The set of groups for which we're trying to find the pass-alive ones.
  inline_vector<BensonGroup, kMaxNumGroups> groups;

  // The set of regions for which we're trying to find the pass-alive ones.
  inline_vector<BensonRegion, kMaxNumRegions> regions;

  // Each BensonRegion keeps track of two lists of groups:
  //  - enclosing group i is stored at:
  //      region_groups[region->groups_begin + i]
  //  - vital group j is stored at:
  //      region_groups[region->groups_begin + region->num_enclosing_groups + j]
  inline_vector<uint16_t, 4 * kMaxNumGroups> region_groups;

  // For each point c on the board:
  //  - if the point is in an enclosed region (i.e. empty or other_color), then
  //    indices[c] is the index into the regions array of that region.
  //  - if the point is in an group of color, then indices[c] is the index into
  //    the groups array of that group.
  std::array<uint16_t, kN * kN> indices;

  // Initialize the set of groups.
  board_visitor_->Begin();
  for (int row = 0; row < kN; ++row) {
    for (int col = 0; col < kN; ++col) {
      Coord c(row, col);
      if (stones_[c].color() != color || !board_visitor_->Visit(c)) {
        continue;
      }

      // We've found a new group.
      // Visit each stone in the group, building the list of liberties for the
      // group and initializing the indices array so that the group can be
      // quickly found by a Coord on the board.
      auto group_idx = static_cast<uint16_t>(groups.size());
      groups.emplace_back(liberties.size());
      auto& g = groups[group_idx];
      while (!board_visitor_->Done()) {
        c = board_visitor_->Next();
        indices[c] = group_idx;

        for (auto nc : kNeighborCoords[c]) {
          auto ns = stones_[nc];
          if (ns.empty()) {
            // This will potentially add the same liberty up to four times; we
            // will remove duplicates shortly.
            liberties.push_back(nc);
            g.num_liberties += 1;
          } else if (ns.color() == color) {
            board_visitor_->Visit(nc);
          }
        }
      }

      if (g.num_liberties > 1) {
        // Sort all liberties and remove duplicates.
        auto* liberties_begin = &liberties[g.liberties_begin];
        auto* liberties_end = liberties_begin + g.num_liberties;
        std::sort(liberties_begin, liberties_end);
        auto unique_end = std::unique(liberties_begin, liberties_end);
        g.num_liberties =
            static_cast<uint16_t>(std::distance(liberties_begin, unique_end));

        // Release the duplicate liberties we just removed back to the pool.
        liberties.resize(g.liberties_begin + g.num_liberties);
      }
    }
  }

  // Build the set of all regions.
  board_visitor_->Begin();
  for (int row = 0; row < kN; ++row) {
    for (int col = 0; col < kN; ++col) {
      Coord c(row, col);
      if (stones_[c].color() == color || !board_visitor_->Visit(c)) {
        continue;
      }

      // We've found a new region.
      // Visit each empty point and stone of the opposite color in the region,
      // initializing the region's list of empty points, its list of enclosing
      // groups, and the indices array.
      auto region_idx = static_cast<uint16_t>(regions.size());
      regions.emplace_back(empty_points.size(), region_groups.size());
      auto& r = regions[region_idx];
      group_visitor_->Begin();
      while (!board_visitor_->Done()) {
        c = board_visitor_->Next();

        indices[c] = region_idx;
        if (stones_[c].empty()) {
          empty_points.push_back(c);
          r.num_empty_points += 1;
        }

        for (auto nc : kNeighborCoords[c]) {
          auto ns = stones_[nc];
          if (ns.color() != color) {
            board_visitor_->Visit(nc);
          } else if (group_visitor_->Visit(ns.group_id())) {
            region_groups.push_back(indices[nc]);
            r.num_enclosing_groups += 1;
          }
        }
      }

      // Sort the region's list of empty points.
      auto* empty_points_begin = &empty_points[r.empty_points_begin];
      auto* empty_points_end = empty_points_begin + r.num_empty_points;
      std::sort(empty_points_begin, empty_points_end);

      // Find the vital groups for this region.
      // A region is vital for a group if all the region's empty points are
      // liberties of that group.
      for (uint32_t i = 0; i < r.num_enclosing_groups; ++i) {
        auto group_idx = region_groups[r.groups_begin + i];
        auto& g = groups[group_idx];
        const auto* liberties_begin = &liberties[g.liberties_begin];
        const auto* liberties_end = liberties_begin + g.num_liberties;
        const auto* empty_points_begin = &empty_points[r.empty_points_begin];
        const auto* empty_points_end = empty_points_begin + r.num_empty_points;
        bool is_vital = std::includes(liberties_begin, liberties_end,
                                      empty_points_begin, empty_points_end);
        if (is_vital) {
          region_groups.push_back(group_idx);
          r.num_vital_groups += 1;
          g.num_vital_regions += 1;
        }
      }
    }
  }

  // Initialization is now done.

  // Initialize the set of candidate pass-alive groups to all the groups on the
  // board, then iteratively remove those that Benson's Algorithm determines
  // aren't pass-alive.
  inline_vector<uint16_t, kMaxNumGroups> candidate_groups;
  for (int i = 0; i < groups.size(); ++i) {
    candidate_groups.push_back(static_cast<uint16_t>(i));
  }

  // List of groups removed each iteration.
  inline_vector<uint16_t, kMaxNumGroups> removed_groups;
  for (;;) {
    removed_groups.clear();

    // Iterate over remaining groups.
    for (int i = 0; i < candidate_groups.size();) {
      auto group_idx = candidate_groups[i];
      auto& g = groups[group_idx];
      if (g.num_vital_regions < 2) {
        // This group has fewer than two vital regions, remove it.
        removed_groups.push_back(group_idx);
        candidate_groups[i] = candidate_groups.back();
        candidate_groups.pop_back();
      } else {
        i += 1;
      }
    }
    if (removed_groups.empty()) {
      // We didn't remove any groups, we're all done!
      break;
    }

    // For each removed group, remove every region it's adjacent to.
    for (auto group_idx : removed_groups) {
      auto& g = groups[group_idx];
      // Since BensonGroup doesn't track which regions are adjacent to it, we
      // iterate over the group's liberties, removing those regions as we go.
      for (uint32_t i = 0; i != g.num_liberties; ++i) {
        auto c = liberties[g.liberties_begin + i];
        auto region_idx = indices[c];
        if (region_idx == 0xffff) {
          // We've already removed this region.
          continue;
        }

        const auto& r = regions[region_idx];
        for (uint32_t i = 0; i < r.num_empty_points; ++i) {
          auto e = empty_points[r.empty_points_begin + i];
          indices[e] = 0xffff;
        }
        for (uint32_t i = 0; i < r.num_vital_groups; ++i) {
          auto group_idx =
              region_groups[r.groups_begin + r.num_enclosing_groups + i];
          groups[group_idx].num_vital_regions -= 1;
        }
      }
    }
  }

  // candidate_groups now contains only pass-alive groups.
  for (auto group_idx : candidate_groups) {
    groups[group_idx].is_pass_alive = true;
  }

  // Now we know which groups are pass-alive, iterate over all the regions,
  // finding which of those are also pass-alive. For a region to be pass-alive,
  // all its enclosing groups must be pass-alive, and all but zero or one empty
  // points must be adjacent to a neighboring group.

  // Initialize the result array to empty.
  std::array<Color, kN * kN> result;
  for (auto& x : result) {
    x = Color::kEmpty;
  }

  board_visitor_->Begin();
  for (auto& r : regions) {
    // All regions must have at least one empty point, otherwise they'd be dead.
    MG_CHECK(r.num_empty_points != 0);
    if (r.num_enclosing_groups == 0) {
      // Skip regions that have no enclosing group (the empty board).
      // Because we consider regions that have one empty point that isn't
      // adjacent to an enclosing group as pass-alive, we don't skip regions
      // that aren't vital to any groups here.
      continue;
    }

    // A region is only pass-alive if all its enclosing groups are pass-alive.
    r.is_pass_alive = true;
    for (uint32_t i = 0; i < r.num_enclosing_groups; ++i) {
      auto group_idx = region_groups[r.groups_begin + i];
      if (!groups[group_idx].is_pass_alive) {
        r.is_pass_alive = false;
        break;
      }
    }

    // A region is only pass-alive if at most one empty point is not adjacent
    // to an enclosing group.
    if (r.is_pass_alive) {
      int num_interior_points = 0;
      for (uint32_t i = 0; i < r.num_empty_points; ++i) {
        auto c = empty_points[r.empty_points_begin + i];
        bool is_interior = true;
        for (auto nc : kNeighborCoords[c]) {
          if (stones_[nc].color() == color) {
            is_interior = false;
            break;
          }
        }
        if (is_interior && ++num_interior_points == 2) {
          r.is_pass_alive = false;
          break;
        }
      }
    }

    if (r.is_pass_alive) {
      // This region is pass-alive, mark all the points in the region in the
      // output array.
      auto c = empty_points[r.empty_points_begin];
      board_visitor_->Visit(c);
      while (!board_visitor_->Done()) {
        c = board_visitor_->Next();
        result[c] = color;
        for (auto nc : kNeighborCoords[c]) {
          if (stones_[nc].color() != color) {
            board_visitor_->Visit(nc);
          }
        }
      }
    }
  }

  return result;
}

void Position::UpdateLegalMoves(ZobristHistory* zobrist_history) {
  legal_moves_[Coord::kPass] = true;

  if (zobrist_history == nullptr) {
    // We're not checking for superko, use the basic result from ClassifyMove to
    // determine whether each move is legal.
    for (int c = 0; c < kN * kN; ++c) {
      legal_moves_[c] = ClassifyMove(c) != MoveType::kIllegal;
    }
  } else {
    // We're using superko, things are a bit trickier.
    for (int c = 0; c < kN * kN; ++c) {
      switch (ClassifyMove(c)) {
        case Position::MoveType::kIllegal: {
          // The move is trivially not legal.
          legal_moves_[c] = false;
          break;
        }

        case Position::MoveType::kNoCapture: {
          // The move will not capture any stones: we can calculate the new
          // position's stone hash directly.
          auto new_hash = stone_hash_ ^ zobrist::MoveHash(c, to_play_);
          legal_moves_[c] =
              !zobrist_history->HasPositionBeenPlayedBefore(new_hash);
          break;
        }

        case Position::MoveType::kCapture: {
          // The move will capture some opponent stones: in order to calculate
          // the stone hash, we actually have to play the move.

          Position new_position(*this);
          // It's safe to call AddStoneToBoard instead of PlayMove because:
          //  - we know the move is not kPass.
          //  - the move is legal (modulo superko).
          //  - we only care about new_position's stone_hash and not the rest of
          //    the bookkeeping that PlayMove updates.
          new_position.AddStoneToBoard(c, to_play_);
          auto new_hash = new_position.stone_hash();
          legal_moves_[c] =
              !zobrist_history->HasPositionBeenPlayedBefore(new_hash);
          break;
        }
      }
    }
  }
}

}  // namespace minigo
