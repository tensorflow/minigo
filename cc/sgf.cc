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

#include "cc/sgf.h"

#include <cctype>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "cc/constants.h"
#include "cc/logging.h"

namespace minigo {
namespace sgf {

namespace {

class Parser {
 public:
  Parser(absl::string_view contents, std::string* error)
      : original_contents_(contents), contents_(contents), error_(error) {}

  bool Parse(Collection* collection) {
    *error_ = "";
    while (SkipWhitespace()) {
      collection->trees.push_back(absl::make_unique<Tree>());
      if (!ParseTree(collection->trees.back().get())) {
        return false;
      }
    }
    return done();
  }

  bool done() const { return contents_.empty(); }
  char peek() const { return contents_[0]; }

 private:
  bool ParseTree(Tree* tree) {
    if (!Read('(')) {
      return false;
    }

    if (!ParseSequence(tree)) {
      return false;
    }

    for (;;) {
      if (!SkipWhitespace()) {
        return Error("reached EOF when parsing tree");
      }
      if (peek() == '(') {
        tree->sub_trees.push_back(absl::make_unique<Tree>());
        if (!ParseTree(tree->sub_trees.back().get())) {
          return false;
        }
      } else {
        return Read(')');
      }
    }
  }

  bool ParseSequence(Tree* tree) {
    for (;;) {
      if (!SkipWhitespace()) {
        return Error("reached EOF when parsing sequence");
      }
      if (peek() != ';') {
        break;
      }
      tree->nodes.push_back(absl::make_unique<Node>());
      if (!ParseNode(tree->nodes.back().get())) {
        return false;
      }
    }
    if (tree->nodes.empty()) {
      return Error("tree has no nodes");
    }
    return true;
  }

  bool ParseNode(Node* node) {
    MG_CHECK(Read(';'));
    for (;;) {
      if (!SkipWhitespace()) {
        return Error("reached EOF when parsing node");
      }
      if (!absl::ascii_isupper(peek())) {
        return true;
      }
      Property prop;
      if (!ParseProperty(&prop)) {
        return false;
      }
      if (prop.id == "B" || prop.id == "W") {
        if (node->move.color != Color::kEmpty) {
          return Error("node already has a move");
        }
        node->move.color = prop.id == "B" ? Color::kBlack : Color::kWhite;
        if (prop.values.size() != 1) {
          return Error("expected exactly one property value, got \"",
                       prop.ToString(), "\"");
        }
        node->move.c = Coord::FromSgf(prop.values[0], true);
        if (node->move.c == Coord::kInvalid) {
          return Error(prop.values[0], " is not a valid SGF coordinate");
        }
      } else {
        node->properties.push_back(std::move(prop));
      }
    }
  }

  bool ParseProperty(Property* prop) {
    if (!ReadTo('[', &prop->id)) {
      return false;
    }
    if (prop->id.empty()) {
      return Error("property has an empty ID");
    }
    if (!SkipWhitespace()) {
      return Error("reached EOF when parsing property ", prop->id);
    }
    for (;;) {
      prop->values.emplace_back();
      if (!ParseValue(&prop->values.back())) {
        return false;
      }
      SkipWhitespace();
      if (peek() != '[') {
        break;
      }
    }
    return true;
  }

  bool ParseValue(std::string* value) {
    return Read('[') && ReadTo(']', value) && Read(']');
  }

  bool Read(char c) {
    if (done()) {
      return Error("expected '", absl::string_view(&c, 1), "', got EOF");
    }
    if (contents_[0] != c) {
      return Error("expected '", absl::string_view(&c, 1), "', got '",
                   contents_.substr(0, 1), "'");
    }
    contents_ = contents_.substr(1);
    return true;
  }

  bool ReadTo(char c, std::string* result) {
    result->clear();
    bool read_escape = false;
    for (size_t i = 0; i < contents_.size(); ++i) {
      char x = contents_[i];
      if (!read_escape) {
        read_escape = x == '\\';
        if (read_escape) {
          continue;
        }
      }

      // Don't check the we're done reading if the current character is an
      // escaped \].
      if ((!read_escape || x != ']') && x == c) {
        contents_ = contents_.substr(i);
        return true;
      }

      absl::StrAppend(result, absl::string_view(&x, 1));
      read_escape = false;
    }
    return Error("reached EOF before finding '", absl::string_view(&c, 1), "'");
  }

  // Skip over whitespace.
  // Updates contents_ and returns true if there are non-whitespace characters
  // remaining. Leaves contents_ alone and returns false if only whitespace
  // characters remain.
  bool SkipWhitespace() {
    contents_ = absl::StripLeadingAsciiWhitespace(contents_);
    return !contents_.empty();
  }

  template <typename... Args>
  bool Error(Args&&... args) {
    // Find the line & column number the error occured at.
    int line = 1;
    int col = 1;
    for (auto* c = original_contents_.data(); c != contents_.data(); ++c) {
      if (*c == '\n') {
        ++line;
        col = 1;
      } else {
        ++col;
      }
    }
    *error_ = absl::StrCat("ERROR at line:", line, " col:", col, ": ", args...);
    return false;
  }

  absl::string_view original_contents_;
  absl::string_view contents_;
  std::string* error_;
};

}  // namespace

std::string Property::ToString() const {
  return absl::StrCat(id, "[", absl::StrJoin(values, "]["), "]");
}

std::string Node::ToString() const {
  std::string str = ";";
  if (move.color != Color::kEmpty) {
    absl::StrAppend(&str, ColorToCode(move.color), "[", move.c.ToSgf(), "]");
  }
  for (const auto& prop : properties) {
    absl::StrAppend(&str, prop.ToString());
  }
  return str;
}

const Property* Node::FindProperty(absl::string_view id) const {
  for (const auto& prop : properties) {
    if (prop.id == id) {
      return &prop;
    }
  }
  return nullptr;
}

const std::string& Node::GetComment() const {
  static const std::string empty;
  const auto* prop = FindProperty("C");
  return prop != nullptr ? prop->values[0] : empty;
}

std::string Node::GetCommentAndProperties() const {
  std::vector<std::string> comments;
  std::vector<std::string> prop_strs;
  for (const auto& prop : properties) {
    if (prop.id == "GC") {
      // The game comment goes first.
      comments.insert(comments.begin(), prop.values[0]);
    } else if (prop.id == "C") {
      // The node comment goes after the game comment.
      comments.push_back(prop.values[0]);
    } else {
      prop_strs.push_back(prop.ToString());
    }
  }

  // If we have both comments and properties, insert a blank line to separate
  // them.
  if (!comments.empty() && !prop_strs.empty()) {
    comments.push_back("");
  }

  comments.insert(comments.end(), prop_strs.begin(), prop_strs.end());
  return absl::StrJoin(comments, "\n");
}

std::string Tree::ToString() const {
  std::vector<std::string> lines;
  for (const auto& node : nodes) {
    lines.push_back(node->ToString());
  }
  for (const auto& sub_tree : sub_trees) {
    lines.push_back(sub_tree->ToString());
  }
  return absl::StrCat("(", absl::StrJoin(lines, "\n"), ")");
}

std::vector<Move> Tree::ExtractMainLine() const {
  std::vector<Move> result;
  const auto* tree = this;
  for (;;) {
    for (const auto& node : tree->nodes) {
      if (node->move.c != Coord::kInvalid) {
        result.push_back(node->move);
      }
    }
    if (tree->sub_trees.empty()) {
      break;
    }
    tree = tree->sub_trees[0].get();
  }
  return result;
}

std::string Collection::ToString() const {
  std::vector<std::string> parts;
  for (const auto& tree : trees) {
    parts.push_back(tree->ToString());
  }
  return absl::StrJoin(parts, "\n");
}

bool Parse(absl::string_view contents, Collection* collection,
           std::string* error) {
  *collection = {};
  *error = "";
  return Parser(contents, error).Parse(collection);
}

std::string CreateSgfString(absl::Span<const MoveWithComment> moves,
                            const CreateSgfOptions& options) {
  auto str = absl::StrFormat(
      "(;GM[1]FF[4]CA[UTF-8]AP[Minigo_sgfgenerator]RU[%s]\n"
      "SZ[%d]KM[%g]PW[%s]PB[%s]RE[%s]\n",
      options.ruleset, kN, options.komi, options.white_name, options.black_name,
      options.result);
  if (!options.game_comment.empty()) {
    absl::StrAppend(&str, "C[", options.game_comment, "]\n");
  }

  for (const auto& move_with_comment : moves) {
    Move move = move_with_comment.move;
    MG_CHECK(move.color == Color::kBlack || move.color == Color::kWhite);
    absl::StrAppendFormat(&str, ";%s[%s]", ColorToCode(move.color),
                          move.c.ToSgf());
    if (!move_with_comment.comment.empty()) {
      absl::StrAppend(&str, "C[", move_with_comment.comment, "]");
    }
  }

  absl::StrAppend(&str, ")\n");

  return str;
}

std::ostream& operator<<(std::ostream& os, const MoveWithComment& move) {
  MG_CHECK(move.move.color == Color::kBlack ||
           move.move.color == Color::kWhite);
  if (move.move.color == Color::kBlack) {
    os << "B";
  } else {
    os << "W";
  }
  os << "[" << move.move.c.ToSgf() << "]";

  if (!move.comment.empty()) {
    os << "C[" << move.comment << "]";
  }
  return os;
}

}  // namespace sgf
}  // namespace minigo
