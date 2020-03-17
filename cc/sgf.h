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

#ifndef CC_SGF_H_
#define CC_SGF_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "cc/color.h"
#include "cc/coord.h"
#include "cc/logging.h"
#include "cc/move.h"
#include "cc/platform/utils.h"

namespace minigo {
namespace sgf {

constexpr char kProgramIdentifier[] = "Minigo";

//  Collection = GameTree { GameTree }
//  GameTree   = "(" Sequence { GameTree } ")"
//  Sequence   = Node { Node }
//  Node       = ";" { Property }
//  Property   = PropIdent PropValue { PropValue }
//  PropIdent  = UcLetter { UcLetter }
//  PropValue  = "[" CValueType "]"
//  CValueType = (ValueType | Compose)
//  ValueType  = (None | Number | Real | Double | Color | SimpleText |
//               Text | Point  | Move | Stone)

// An SGF node property.
// Properties created via the minigo::sgf::Parse function are guaranteed to
// have at least one value.
struct Property {
  std::string ToString() const;

  std::string id;
  std::vector<std::string> values;
};

struct Node {
  std::string ToString() const;

  const Property* FindProperty(absl::string_view id) const;

  // Returns the node's comment if it has one or an empty string otherwise.
  const std::string& GetComment() const;

  // Returns the nodes game comment (GC) and comment (C) properties if any,
  // followed by all other properties separated by newlines.
  std::string GetCommentAndProperties() const;

  Move move;
  std::vector<Property> properties;
};

struct Tree {
  std::string ToString() const;

  std::vector<Move> ExtractMainLine() const;

  std::vector<std::unique_ptr<Node>> nodes;
  std::vector<std::unique_ptr<Tree>> sub_trees;
};

struct Collection {
  std::string ToString() const;

  std::vector<std::unique_ptr<Tree>> trees;
};

// Parses an SGF file.
MG_WARN_UNUSED_RESULT bool Parse(absl::string_view contents,
                                 Collection* collection, std::string* error);

// TODO(tommadams): Replace sgf::MoveWithComment with sgf::Node.
// A single move with a (possibly empty) comment.
struct MoveWithComment {
  MoveWithComment() = default;

  MoveWithComment(Move move, std::string comment)
      : move(move), comment(std::move(comment)) {}

  MoveWithComment(Color color, Coord c, std::string comment)
      : move(color, c), comment(std::move(comment)) {}

  // MoveWithComment is convertible to a Move for ease of use.
  operator Move() const { return move; }

  Move move;
  std::string comment;

  bool operator==(const MoveWithComment& other) const {
    return move == other.move && comment == other.comment;
  }
};

std::ostream& operator<<(std::ostream& ios, const MoveWithComment& move);

struct CreateSgfOptions {
  std::string black_name = kProgramIdentifier;
  std::string white_name = kProgramIdentifier;
  std::string ruleset = "Chinese";
  float komi = 7.5;
  std::string result;
  std::string game_comment;
};

// Returns a valid SGF file for the given move sequence.
std::string CreateSgfString(absl::Span<const MoveWithComment> moves,
                            const CreateSgfOptions& options);

}  // namespace sgf
}  // namespace minigo

#endif  // CC_SGF_H_
