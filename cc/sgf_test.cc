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

class AstTest : public ::testing::Test {
 protected:
  Ast ast_;
};

TEST_F(AstTest, NoTrees) {
  EXPECT_TRUE(ast_.Parse("")) << ast_.error();
  EXPECT_TRUE(ast_.Parse(" \n ")) << ast_.error();
}

TEST_F(AstTest, BadTree) { EXPECT_FALSE(ast_.Parse("   \n  x")); }

TEST_F(AstTest, EmptyTree) { EXPECT_FALSE(ast_.Parse("()")); }

TEST_F(AstTest, EmptyNode) {
  EXPECT_TRUE(ast_.Parse("(;)")) << ast_.error();
  ASSERT_EQ(1, ast_.trees().size());
  EXPECT_EQ(1, ast_.trees()[0].nodes.size());
  EXPECT_EQ(0, ast_.trees()[0].nodes[0].properties.size());
  EXPECT_EQ(0, ast_.trees()[0].children.size());
  EXPECT_EQ("(;)", ast_.trees()[0].ToString());
}

TEST_F(AstTest, MultipleEmptyNodes) {
  EXPECT_TRUE(ast_.Parse("(;;;)")) << ast_.error();
  ASSERT_EQ(1, ast_.trees().size());
  EXPECT_EQ(3, ast_.trees()[0].nodes.size());
  EXPECT_EQ("(;\n;\n;)", ast_.trees()[0].ToString());
}

TEST_F(AstTest, OneNodeTree) {
  EXPECT_TRUE(ast_.Parse("(;A[1][hmm])")) << ast_.error();
  ASSERT_EQ(1, ast_.trees().size());
  EXPECT_EQ("(;A[1][hmm])", ast_.trees()[0].ToString());
}

TEST_F(AstTest, PropertyIdIsMissing) { EXPECT_FALSE(ast_.Parse("(;[])")); }

TEST_F(AstTest, PropertyIdIsNotUpper) { EXPECT_FALSE(ast_.Parse("(;a[])")); }

TEST_F(AstTest, PropertyHasOneEmptyValue) {
  EXPECT_TRUE(ast_.Parse("(;A[])")) << ast_.error();
}

TEST_F(AstTest, PropertyHasMultipleEmptyValues) {
  EXPECT_TRUE(ast_.Parse("(;A[][][])")) << ast_.error();
  ASSERT_EQ(1, ast_.trees().size());
  EXPECT_EQ(1, ast_.trees()[0].nodes.size());
  EXPECT_EQ(1, ast_.trees()[0].nodes[0].properties.size());
  EXPECT_EQ("A", ast_.trees()[0].nodes[0].properties[0].id);
  EXPECT_EQ(3, ast_.trees()[0].nodes[0].properties[0].values.size());
  EXPECT_EQ("(;A[][][])", ast_.trees()[0].ToString());
}

TEST_F(AstTest, NestedTrees) {
  EXPECT_TRUE(ast_.Parse("(; (;A[b][c];D[]) (;) (;E[f];G[] (;H[i])))"))
      << ast_.error();
  ASSERT_EQ(1, ast_.trees().size());
  EXPECT_EQ(R"((;
(;A[b][c]
;D[])
(;)
(;E[f]
;G[]
(;H[i]))))",
            ast_.trees()[0].ToString());
}

TEST_F(AstTest, MultipleTrees) {
  EXPECT_TRUE(ast_.Parse("(;A[])(;B[c]) (  ;D[e][f])")) << ast_.error();
  ASSERT_EQ(3, ast_.trees().size());
  EXPECT_EQ("(;A[])", ast_.trees()[0].ToString());
  EXPECT_EQ("(;B[c])", ast_.trees()[1].ToString());
  EXPECT_EQ("(;D[e][f])", ast_.trees()[2].ToString());
}

TEST_F(AstTest, NodesMustComeBeforeChildren) {
  EXPECT_FALSE(ast_.Parse("(() ;A[])"));
}

TEST(SgfTest, CreateSgfStringDefaults) {
  CreateSgfOptions options;
  options.result = "W+R";
  auto expected = absl::StrCat(
      "(;GM[1]FF[4]CA[UTF-8]AP[Minigo_sgfgenerator]RU[Chinese]\nSZ[", kN,
      "]KM[7.5]PW[Minigo]PB[Minigo]RE[W+R]\n)\n");
  EXPECT_EQ(expected, CreateSgfString({}, options));
}

TEST(SgfTest, CreateSgfStringOptions) {
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

TEST(SgfTest, CreateSgfStringMoves) {
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

TEST(SgfTest, GetMainLineMoves) {
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

  Ast ast;
  ASSERT_TRUE(ast.Parse(sgf)) << ast.error();
  EXPECT_THAT(GetMainLineMoves(ast),
              ::testing::ContainerEq(expected_main_line));
}

}  // namespace
}  // namespace sgf
}  // namespace minigo
