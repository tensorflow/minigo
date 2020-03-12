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

#include <vector>

#include "absl/strings/str_cat.h"
#include "cc/color.h"
#include "cc/coord.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace minigo {
namespace sgf {
namespace {

class SgfTest : public ::testing::Test {
 protected:
  bool Parse(std::string contents) {
    return minigo::sgf::Parse(contents, &collection_, &error_);
  }

  Collection collection_;
  std::string error_;
};

TEST_F(SgfTest, NoTrees) {
  EXPECT_TRUE(Parse("")) << error_;
  EXPECT_TRUE(Parse(" \n ")) << error_;
}

TEST_F(SgfTest, BadTree) { EXPECT_FALSE(Parse("   \n  x")); }

TEST_F(SgfTest, EmptyTree) { EXPECT_FALSE(Parse("()")); }

TEST_F(SgfTest, EmptyNode) {
  EXPECT_TRUE(Parse("(;)")) << error_;
  ASSERT_EQ(1, collection_.trees.size());
  ASSERT_EQ(1, collection_.trees[0]->nodes.size());

  const auto* node = collection_.trees[0]->nodes[0].get();
  EXPECT_EQ(Move(Color::kEmpty, Coord::kInvalid), node->move);
  EXPECT_TRUE(node->properties.empty());

  EXPECT_EQ("(;)", collection_.ToString());
}

TEST_F(SgfTest, MultipleEmptyNodes) {
  EXPECT_TRUE(Parse("(;;;)")) << error_;
  ASSERT_EQ(1, collection_.trees.size());
  EXPECT_EQ(3, collection_.trees[0]->nodes.size());
  EXPECT_EQ("(;\n;\n;)", collection_.trees[0]->ToString());
}

TEST_F(SgfTest, OneNodeTree) {
  EXPECT_TRUE(Parse("(;A[1][hmm])")) << error_;
  ASSERT_EQ(1, collection_.trees.size());
  EXPECT_EQ("(;A[1][hmm])", collection_.ToString());
}

TEST_F(SgfTest, PropertyIdIsMissing) { EXPECT_FALSE(Parse("(;[])")); }

TEST_F(SgfTest, PropertyIdIsNotUpper) { EXPECT_FALSE(Parse("(;a[])")); }

TEST_F(SgfTest, PropertyHasOneEmptyValue) {
  EXPECT_TRUE(Parse("(;A[])")) << error_;
}

TEST_F(SgfTest, PropertyHasMultipleEmptyValues) {
  EXPECT_TRUE(Parse("(;A[][][])")) << error_;
  ASSERT_EQ(1, collection_.trees.size());
  EXPECT_EQ(1, collection_.trees[0]->nodes.size());
  EXPECT_EQ(1, collection_.trees[0]->nodes[0]->properties.size());
  EXPECT_EQ("A", collection_.trees[0]->nodes[0]->properties[0].id);
  EXPECT_EQ(3, collection_.trees[0]->nodes[0]->properties[0].values.size());
  EXPECT_EQ("(;A[][][])", collection_.trees[0]->ToString());
}

TEST_F(SgfTest, NestedTrees) {
  EXPECT_TRUE(Parse("(; (;A[b][c];D[]) (;) (;E[f];G[] (;H[i])))")) << error_;
  ASSERT_EQ(1, collection_.trees.size());
  EXPECT_EQ(R"((;
(;A[b][c]
;D[])
(;)
(;E[f]
;G[]
(;H[i]))))",
            collection_.ToString());
}

TEST_F(SgfTest, MultipleTrees) {
  EXPECT_TRUE(Parse("(;X[])(;Y[a]) (  ;Z[b][c])")) << error_;
  ASSERT_EQ(3, collection_.trees.size());
  EXPECT_EQ("(;X[])", collection_.trees[0]->ToString());
  EXPECT_EQ("(;Y[a])", collection_.trees[1]->ToString());
  EXPECT_EQ("(;Z[b][c])", collection_.trees[2]->ToString());
}

TEST_F(SgfTest, NodesMustComeBeforeChildren) {
  EXPECT_FALSE(Parse("(() ;A[])"));
}

TEST_F(SgfTest, CreateSgfStringDefaults) {
  CreateSgfOptions options;
  options.result = "W+R";
  auto expected = absl::StrCat(
      "(;GM[1]FF[4]CA[UTF-8]AP[Minigo_sgfgenerator]RU[Chinese]\nSZ[", kN,
      "]KM[7.5]PW[Minigo]PB[Minigo]RE[W+R]\n)\n");
  EXPECT_EQ(expected, CreateSgfString({}, options));
}

TEST_F(SgfTest, CreateSgfStringOptions) {
  CreateSgfOptions options;
  options.black_name = "Alice";
  options.white_name = "Bob";
  options.ruleset = "Some rules";
  options.result = "B+5";
  options.komi = 101;

  auto expected = absl::StrCat(
      "(;GM[1]FF[4]CA[UTF-8]AP[Minigo_sgfgenerator]RU[Some rules]\nSZ[", kN,
      "]KM[101]PW[Bob]PB[Alice]RE[B+5]\n)\n");
  EXPECT_EQ(expected, CreateSgfString({}, options));
}

TEST_F(SgfTest, CreateSgfStringMoves) {
  CreateSgfOptions options;
  options.result = "B+R";

  std::vector<MoveWithComment> moves = {
      {Color::kBlack, Coord::FromSgf("be"), ""},
      {Color::kWhite, Coord::FromSgf("aa"), "Hello there"},
      {Color::kBlack, Coord::FromSgf("hb"), ""},
      {Color::kWhite, Coord::FromSgf("ge"), "General Kenobi"},
      {Color::kBlack, Coord::kPass, "You are a bold one"},
  };

  // clang-format off
  auto expected = absl::StrCat(
      "(;GM[1]FF[4]CA[UTF-8]AP[Minigo_sgfgenerator]RU[Chinese]\nSZ[", kN,
      "]KM[7.5]PW[Minigo]PB[Minigo]RE[B+R]\n",
      ";B[be]",
      ";W[aa]C[Hello there]",
      ";B[hb]",
      ";W[ge]C[General Kenobi]",
      ";B[]C[You are a bold one])\n");
  // clang-format on
  EXPECT_EQ(expected, CreateSgfString(moves, options));
}

TEST_F(SgfTest, InvalidCoord) {
  std::string sgf = "(;FF[4](;B[xx]))\n";
  ASSERT_FALSE(Parse(sgf));
}

TEST_F(SgfTest, GetMainLineMoves) {
  /*
     --- B[aa] - W[ab] - B[ac]
     \                 \
      \                  B[ad] - W[ae]
       \
         B[af] - W[ag] - B[ah]
               \
                 W[ai]
   */
  std::string sgf =
      "(;FF[4](;B[aa];W[ab]C[hello!](;B[ac])(;B[ad];W[ae]))"
      "(;B[af](;W[ag];B[ah])(;W[ai])))\n";

  std::vector<Move> expected_main_line = {
      {Color::kBlack, Coord::FromSgf("aa")},
      {Color::kWhite, Coord::FromSgf("ab")},
      {Color::kBlack, Coord::FromSgf("ac")},
  };

  ASSERT_TRUE(Parse(sgf)) << error_;

  auto actual_main_line = collection_.trees[0]->ExtractMainLine();
  EXPECT_THAT(actual_main_line, ::testing::ContainerEq(expected_main_line));
}

TEST_F(SgfTest, CommentEscaping) {
  // Fragment of an SGF that contains escaped characters.
  std::string sgf = "(;FF[4];C[test [?\\]: comment]B[aa];W[bb]C[\\]])";

  EXPECT_TRUE(Parse(sgf)) << error_;
  ASSERT_EQ(1, collection_.trees.size());
  const auto* tree = collection_.trees[0].get();
  ASSERT_EQ(3, tree->nodes.size());

  EXPECT_EQ("aa", tree->nodes[1]->move.c.ToSgf());
  EXPECT_EQ("test [?]: comment", tree->nodes[1]->GetComment());

  EXPECT_EQ("bb", tree->nodes[2]->move.c.ToSgf());
  EXPECT_EQ("]", tree->nodes[2]->GetComment());
}

}  // namespace
}  // namespace sgf
}  // namespace minigo
