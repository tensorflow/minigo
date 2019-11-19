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

#ifndef CC_POSITION_H_
#define CC_POSITION_H_

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include "cc/color.h"
#include "cc/constants.h"
#include "cc/coord.h"
#include "cc/group.h"
#include "cc/inline_vector.h"
#include "cc/logging.h"
#include "cc/padded_array.h"
#include "cc/stone.h"
#include "cc/zobrist.h"

namespace minigo {

extern const std::array<inline_vector<Coord, 4>, kN * kN> kNeighborCoords;

// A fixed-capacity stack of Coords used when traversing connected points on
// the board.
class CoordStack : private inline_vector<Coord, kN * kN> {
  using Impl = inline_vector<Coord, kN * kN>;

 public:
  using Impl::empty;

  void push(Coord c) { Impl::push_back(c); }

  Coord pop() {
    auto result = Impl::back();
    Impl::pop_back();
    return result;
  }
};

// BoardVisitor visits points on the board only once.
// A simple example that visits all points on the board only once:
//   BoardVisitor bv;
//   bv.Begin()
//   bv.Visit(0);
//   while (!bv.Done()) {
//     Coord coord = bv.Next();
//     std::cout << "Visiting " << coord << "\n";
//     for (auto neighbor_coord : GetNeighbors(coord)) {
//       bv.Visit(neighbor_coord);
//     }
//   }
//
// Points are visited in the order that they are passed to Visit for the first
// time.
class BoardVisitor {
 public:
  BoardVisitor() = default;

  // Starts a new visit around the board.
  void Begin() {
    MG_DCHECK(Done());
    if (++epoch_ == 0) {
      memset(visited_.data(), 0, sizeof(visited_));
      epoch_ = 1;
    }
  }

  // Returns true when there are no more points to visit.
  bool Done() const { return stack_.empty(); }

  // Returns the coordinates of the next point in the stack to visit.
  Coord Next() { return stack_.pop(); }

  // If this is the first time Visit has been passed coordinate c since the
  // most recent call to Begin, Visit pushes the coordinate onto its stack of
  // points to visit and returns true. Otherwise, Visit returns false.
  bool Visit(Coord c) {
    if (visited_[c] != epoch_) {
      visited_[c] = epoch_;
      stack_.push(c);
      return true;
    }
    return false;
  }

 private:
  CoordStack stack_;
  std::array<uint8_t, kN * kN> visited_;

  // Initializing to 0xff means the visited_ array will get initialized on the
  // first call to Begin().
  uint8_t epoch_ = 0xff;
};

// GroupVisitor simply keeps track of which groups have been visited since the
// most recent call to Begin. Unlike BoardVisitor, it does not keep a pending
// stack of groups to visit.
class GroupVisitor {
 public:
  GroupVisitor() = default;

  void Begin() {
    if (++epoch_ == 0) {
      memset(visited_.data(), 0, sizeof(visited_));
      epoch_ = 1;
    }
  }

  bool Visit(GroupId id) {
    if (visited_[id] != epoch_) {
      visited_[id] = epoch_;
      return true;
    }
    return false;
  }

 private:
  std::array<uint8_t, Group::kMaxNumGroups> visited_;

  // Initializing to 0xff means the visited_ array will get initialized on the
  // first call to Begin().
  uint8_t epoch_ = 0xff;
};

// Position represents a single board position.
// It tracks the stones on the board and their groups, and contains the logic
// for removing groups with no remaining liberties and merging neighboring
// groups of the same color.
//
// Since the MCTS code makes a copy of the board position for each expanded
// node in the tree, we aim to keep the data structures as compact as possible.
// This is in tension with our other aim of avoiding heap allocations where
// possible, which means we have to preallocate some pools of memory. In
// particular, the BoardVisitor and GroupVisitor classes that Position uses to
// update its internal state are relatively large compared to the board size
// (even though we're only talking a couple of kB in total. Consequently, the
// caller of the Position code must pass pointers to previously allocated
// instances of BoardVisitor and GroupVisitor. These can then be reused by all
// instances of the Position class.
class Position {
 public:
  using Stones = std::array<Stone, kN * kN>;

  // State required to undo a call to PlayMove.
  struct UndoState {
    UndoState(Coord c, Color to_play, Coord ko)
        : c(c), to_play(to_play), ko(ko) {}
    Coord c;
    Color to_play;
    Coord ko;
    inline_vector<Coord, 4> captures;
  };

  // Calculates the Zobrist hash for an array of stones. Prefer using
  // Position::stone_hash() if possible.
  static zobrist::Hash CalculateStoneHash(const Stones& stones);

  // Interface used to enforce positional superko based on the Zobrist hash of
  // a position.
  class ZobristHistory {
   public:
    virtual ~ZobristHistory() = default;
    virtual bool HasPositionBeenPlayedBefore(
        zobrist::Hash stone_hash) const = 0;
  };

  // Initializes an empty board.
  // All moves are considered legal.
  explicit Position(Color to_play);

  Position(const Position&) = default;
  Position& operator=(const Position&) = default;

  // Plays the given move and updates which moves are legal.
  // If zobrist_history is non-null, move legality considers positional superko.
  // If zobrist_history is null, positional superko is not considered when
  // updating the legal moves, only ko.
  // Returns an UndoState object that allows the move to be undone.
  UndoState PlayMove(Coord c, Color color = Color::kEmpty,
                     ZobristHistory* zobrist_history = nullptr);

  // Undoes the move recent call to PlayMove.
  void UndoMove(const UndoState& undo,
                ZobristHistory* zobrist_history = nullptr);

  // TODO(tommadams): Do we really need to store this on the position? Return
  // the number of captured stones from AddStoneToBoard and track the number of
  // captures in the player.
  const std::array<int, 2>& num_captures() const { return num_captures_; }

  // Calculates the score from B perspective. If W is winning, score is
  // negative.
  float CalculateScore(float komi) const;

  // Calculates all pass-alive region that are enclosed by groups of `color`
  // stones.
  // Elements in the returned array are set to `Color::kBlack` or
  // `Color::kWhite` if they belong to a pass-alive region or `Color::kEmpty`
  // otherwise. Only intersections inside the enclosed region are set,
  // intersections that are part of an enclosing group are set to
  // `Color::kEmpty`. Concretely, given the following position:
  //   X . X . O X .
  //   X X X X X X .
  //   . . . . . . .
  // The returned array will be set to:
  //   . X . X X . .
  //   . . . . . . .
  std::array<Color, kN * kN> CalculatePassAliveRegions() const;

  // Returns true if the whole board is pass-alive.
  bool CalculateWholeBoardPassAlive() const;

  // Returns true if playing this move is legal.
  // Does not check positional superko.
  // legal_move(c) can be used to check for positional superko.
  enum class MoveType {
    // The position is illegal:
    //  - a stone is already at that position.
    //  - the move is ko.
    //  - the move is suicidal.
    kIllegal,

    // The move will not capture an opponent's group.
    // The move is not necessarily legal because of superko.
    kNoCapture,

    // The move will capture an opponent's group.
    // The move is not necessarily legal because of superko.
    kCapture,
  };
  MoveType ClassifyMoveIgnoringSuperko(Coord c) const;

  std::string ToSimpleString() const;
  std::string ToPrettyString(bool use_ansi_colors = true) const;

  Color to_play() const { return to_play_; }
  const Stones& stones() const { return stones_; }
  int n() const { return n_; }
  Coord ko() const { return ko_; }
  zobrist::Hash stone_hash() const { return stone_hash_; }
  uint8_t legal_move(Coord c) const {
    MG_DCHECK(c < kNumMoves);
    return legal_moves_[c];
  }
  const PaddedArray<uint8_t, kNumMoves>& legal_moves() const {
    return legal_moves_;
  }

  // Returns the number of liberties the chain at c has.
  int num_chain_liberties(Coord c) const {
    MG_DCHECK(c <= kN * kN);
    auto s = stones_[c];
    return s.empty() ? 0 : groups_[s.group_id()].num_liberties;
  }

  int chain_size(Coord c) const {
    MG_DCHECK(c <= kN * kN);
    auto s = stones_[c];
    return s.empty() ? 0 : groups_[s.group_id()].size;
  }

  // The following methods are protected to enable direct testing by unit tests.
 protected:
  // Returns the Group of the stone at the given coordinate. Used for testing.
  Group GroupAt(Coord c) const {
    auto s = stones_[c];
    return s.empty() ? Group() : groups_[s.group_id()];
  }

  // Returns color C if the position at idx is empty and surrounded on all
  // sides by stones of color C.
  // Returns Color::kEmpty otherwise.
  Color IsKoish(Coord c) const;

  // Adds the stone to the board.
  // Removes newly surrounded opponent groups.
  // DOES NOT update legal_moves_: callers of AddStoneToBoard must explicitly
  // call UpdateLegalMoves afterwards (this is because UpdateLegalMoves uses
  // AddStoneToBoard internally).
  // Updates liberty counts of remaining groups.
  // Updates num_captures_.
  // If the move captures a single stone, sets ko_ to the coordinate of that
  // stone. Sets ko_ to kInvalid otherwise.
  // Returns a list of the neighbors of c that belonged to groups that were
  // captured by this move.
  inline_vector<Coord, 4> AddStoneToBoard(Coord c, Color color);

  // Updates legal_moves_.
  // If zobrist_history is non-null, this takes into account positional superko.
  void UpdateLegalMoves(ZobristHistory* zobrist_history);

 private:
  // Sets the pass alive regions for the given color in result.
  // The caller is responsible for initializing all elements in `result` to
  // `Color::kEmpty` before calling.
  void CalculatePassAliveRegionsForColor(
      Color color, BoardVisitor* board_visitor, GroupVisitor* group_visitor,
      std::array<Color, kN * kN>* result) const;

  // Removes the group with a stone at the given coordinate from the board,
  // updating the liberty counts of neighboring groups.
  void RemoveGroup(Coord c);

  // Merge neighboring groups of the same color as the stone at coordinate c
  // into that stone's group. Called when a stone is placed on the board that
  // has two or more distinct neighboring groups of the same color.
  void MergeGroup(Coord c);

  // Called as part of UndoMove for the given color at point capture_c.
  // Replaces the previously captured stones at point group_c.
  GroupId UncaptureGroup(Color color, Coord capture_c, Coord group_c);

  // Called as part of UndoMove.
  // Create a new group for the chain of stones at c.
  void AssignNewGroup(Coord c, BoardVisitor* board_visitor);

  // Returns true if the point at coordinate c neighbors the given group.
  bool HasNeighboringGroup(Coord c, GroupId group_id) const;

  Stones stones_;
  GroupPool groups_;

  Color to_play_;
  Coord ko_ = Coord::kInvalid;

  // Number of captures for (B, W).
  // TODO(tommadams): remove this from the Position class and track it in the
  // game instead.
  std::array<int, 2> num_captures_{{0, 0}};

  int n_ = 0;

  // MctsNode::CalculateChildActionScoreSse requires that `legal_moves_` is
  // padded to a multiple of 16 bytes.
  PaddedArray<uint8_t, kNumMoves> legal_moves_;

  // Zobrist hash of the stones. It can be used for positional superko.
  // This has does not include number of consecutive passes or ko, so should not
  // be used for caching inferences.
  zobrist::Hash stone_hash_ = 0;
};

}  // namespace minigo

#endif  // CC_POSITION_H_
