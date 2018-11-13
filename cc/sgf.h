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

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "cc/color.h"
#include "cc/coord.h"
#include "cc/move.h"

namespace minigo {
namespace sgf {

constexpr char kProgramIdentifier[] = "Minigo";

// Abstract syntax tree for an SGF file.
// The Ast class just holds the structure and contents of the tree and doesn't
// infer any meaning from the property IDs or values.
class Ast {
 public:
  struct Property {
    std::string ToString() const;

    std::string id;
    std::vector<std::string> values;
  };

  struct Node {
    std::string ToString() const;

    const Property* FindProperty(absl::string_view id) const;

    std::vector<Property> properties;
  };

  struct Tree {
    std::string ToString() const;

    std::vector<Node> nodes;
    std::vector<Tree> children;
  };

  // Parses the SGF file.
  __attribute__((warn_unused_result)) bool Parse(std::string contents);

  // Returns a non-empty string containing error information if the most recent
  // call to Parse returned false.
  const std::string& error() const { return error_; }

  const std::vector<Tree>& trees() const { return trees_; }

 private:
  std::string error_;
  std::vector<Tree> trees_;
  std::string contents_;
};

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

struct Node {
  Node(Move move, std::string comment)
      : move(move), comment(std::move(comment)) {}

  // Returns a flattened copy of the main line moves: the chain of moves formed
  // by this node and its left-most descendants.
  std::vector<Move> ExtractMainLine() const;

  const Move move;
  std::string comment;
  std::vector<std::unique_ptr<Node>> children;
};

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

// Extracts the complete game trees from an SGF AST.
std::vector<std::unique_ptr<Node>> GetTrees(const Ast& ast);

}  // namespace sgf
}  // namespace minigo

#endif  // CC_SGF_H_
