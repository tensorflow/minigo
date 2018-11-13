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
#include <iostream>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "cc/check.h"
#include "cc/constants.h"

namespace minigo {
namespace sgf {

namespace {

class Parser {
 public:
  Parser(absl::string_view contents, std::string* error)
      : original_contents_(contents), contents_(contents), error_(error) {}

  bool Parse(std::vector<Ast::Tree>* trees) {
    *error_ = "";
    while (SkipWhitespace()) {
      trees->emplace_back();
      if (!ParseTree(&trees->back())) {
        return false;
      }
    }
    return done();
  }

  bool done() const { return contents_.empty(); }
  char peek() const { return contents_[0]; }

 private:
  bool ParseTree(Ast::Tree* tree) {
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
        tree->children.emplace_back();
        if (!ParseTree(&tree->children.back())) {
          return false;
        }
      } else {
        return Read(')');
      }
    }
  }

  bool ParseSequence(Ast::Tree* tree) {
    if (!SkipWhitespace()) {
      return Error("reached EOF when parsing sequence");
    }
    for (;;) {
      if (peek() != ';') {
        // All valid trees must contain at least one node.
        if (tree->nodes.empty()) {
          return Error("tree has no nodes");
        }
        return true;
      }
      Read(';');
      tree->nodes.emplace_back();
      if (!ParseNode(&tree->nodes.back())) {
        return false;
      }
    }
  }

  bool ParseNode(Ast::Node* node) {
    for (;;) {
      if (!SkipWhitespace()) {
        return Error("reached EOF when parsing node");
      }
      if (!absl::ascii_isupper(peek())) {
        return true;
      }
      node->properties.emplace_back();
      if (!ParseProperty(&node->properties.back())) {
        return false;
      }
    }
  }

  bool ParseProperty(Ast::Property* prop) {
    if (!ReadTo('[', &prop->id)) {
      return false;
    }
    if (prop->id.empty()) {
      return Error("property has an empty ID");
    }
    bool read_value = false;
    for (;;) {
      if (!SkipWhitespace()) {
        return Error("reached EOF when parsing property ", prop->id);
      }
      if (peek() != '[') {
        if (!read_value) {
          return Error("property ", prop->id, " has no values");
        }
        return true;
      }
      Read('[');
      read_value = true;
      std::string value;
      if (!ReadTo(']', &value)) {
        return false;
      }
      prop->values.push_back(std::move(value));
      Read(']');
    }
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

void GetTreeImpl(const Ast::Tree& tree,
                 std::vector<std::unique_ptr<Node>>* dst) {
  const auto* src = &tree;
  const Ast::Property* prop;

  // Extract all the nodes out of this tree.
  for (const auto& node : src->nodes) {
    // Parse move.
    Move move;
    if ((prop = node.FindProperty("B")) != nullptr) {
      move.color = Color::kBlack;
    } else if ((prop = node.FindProperty("W")) != nullptr) {
      move.color = Color::kWhite;
    } else {
      continue;
    }
    if (prop->values.empty()) {
      std::cerr << "Skipping node " << node.ToString() << " because property "
                << prop->ToString() << " has no values" << std::endl;
      continue;
    }
    move.c = Coord::FromSgf(prop->values[0]);

    // Parse comment.
    std::string comment;
    if ((prop = node.FindProperty("C")) != nullptr && !prop->values.empty()) {
      comment = prop->values[0];
    }

    dst->push_back(absl::make_unique<Node>(move, std::move(comment)));
    dst = &(dst->back()->children);
  }

  for (const auto& src_child : src->children) {
    GetTreeImpl(src_child, dst);
  }
}

}  // namespace

std::string Ast::Property::ToString() const {
  return absl::StrCat(id, "[", absl::StrJoin(values, "]["), "]");
}

std::string Ast::Node::ToString() const {
  std::string str = ";";
  for (const auto& property : properties) {
    absl::StrAppend(&str, property.ToString());
  }
  return str;
}

const Ast::Property* Ast::Node::FindProperty(absl::string_view id) const {
  for (const auto& property : properties) {
    if (property.id == id) {
      return &property;
    }
  }
  return nullptr;
}

std::string Ast::Tree::ToString() const {
  std::vector<std::string> parts;
  for (const auto& node : nodes) {
    parts.push_back(node.ToString());
  }
  for (const auto& child : children) {
    parts.push_back(child.ToString());
  }
  return absl::StrCat("(", absl::StrJoin(parts, "\n"), ")");
}

bool Ast::Parse(std::string contents) {
  error_ = "";
  trees_.clear();
  contents_ = std::move(contents);
  return Parser(contents_, &error_).Parse(&trees_);
}

std::string CreateSgfString(absl::Span<const MoveWithComment> moves,
                            const CreateSgfOptions& options) {
  std::string str = "(";

  // TODO(tommadams): Clean this up once the Abseil team releases proper
  // formatting functions.
  absl::StrAppend(
      &str, absl::StrCat(";GM[1]FF[4]CA[UTF-8]AP[Minigo_sgfgenerator]RU[",
                         options.ruleset, "]\n", "SZ[", kN, "]KM[",
                         options.komi, "]PW[", options.white_name, "]PB[",
                         options.black_name, "]RE[", options.result, "]\n"));
  if (!options.game_comment.empty()) {
    absl::StrAppend(&str, "C[", options.game_comment, "]\n");
  }

  for (const auto& move_with_comment : moves) {
    Move move = move_with_comment.move;
    MG_CHECK(move.color == Color::kBlack || move.color == Color::kWhite);
    const char* color = move.color == Color::kBlack ? "B" : "W";
    absl::StrAppend(&str, absl::StrCat(";", color, "[", move.c.ToSgf(), "]"));
    if (!move_with_comment.comment.empty()) {
      absl::StrAppend(&str, "C[", move_with_comment.comment, "]");
    }
  }

  absl::StrAppend(&str, ")\n");

  return str;
}

std::vector<Move> Node::ExtractMainLine() const {
  std::vector<Move> result;
  const auto* node = this;
  for (;;) {
    result.push_back(node->move);
    if (node->children.empty()) {
      break;
    }
    node = node->children[0].get();
  }
  return result;
}

std::vector<std::unique_ptr<Node>> GetTrees(const Ast& ast) {
  std::vector<std::unique_ptr<Node>> dst;
  for (const auto& tree : ast.trees()) {
    GetTreeImpl(tree, &dst);
  }
  return dst;
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
