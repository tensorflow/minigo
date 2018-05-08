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

#include "cc/mcts_node.h"

#include <array>

#include "cc/position.h"
#include "cc/random.h"
#include "cc/test_utils.h"
#include "gtest/gtest.h"

namespace minigo {
namespace {

static constexpr char kAlmostDoneBoard[] = R"(
    .XO.XO.OO
    X.XXOOOO.
    XXXXXOOOO
    XXXXXOOOO
    .XXXXOOO.
    XXXXXOOOO
    .XXXXOOO.
    XXXXXOOOO
    XXXXOOOOO)";

// Verifies that no matter who is to play, when we know nothing else, the priors
// should be respected, and the same move should be picked.
TEST(MctsNodeTest, ActionFlipping) {
  Random rnd(1);

  std::array<float, kNumMoves> probs;
  std::uniform_real_distribution<float> dist(0.02, 0.021);
  for (float& prob : probs) {
    prob = rnd();
  }

  MctsNode::EdgeStats black_stats, white_stats;
  MctsNode black_root(&black_stats, TestablePosition("", 0, Color::kBlack));
  MctsNode white_root(&white_stats, TestablePosition("", 0, Color::kWhite));

  black_root.SelectLeaf()->IncorporateResults(probs, 0, &black_root);
  white_root.SelectLeaf()->IncorporateResults(probs, 0, &white_root);
  auto* black_leaf = black_root.SelectLeaf();
  auto* white_leaf = white_root.SelectLeaf();
  EXPECT_EQ(black_leaf->move, white_leaf->move);
  EXPECT_EQ(black_root.CalculateChildActionScore(),
            white_root.CalculateChildActionScore());
}

// Verfies that SelectLeaf chooses the child with the highest action score.
TEST(MctsNodeTest, SelectLeaf) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.02;
  }
  Coord c = Coord::FromKgs("D9");
  probs[c] = 0.4;

  MctsNode::EdgeStats root_stats;
  auto board = TestablePosition(kAlmostDoneBoard, 0, Color::kWhite);
  MctsNode root(&root_stats, board);

  root.SelectLeaf()->IncorporateResults(probs, 0, &root);

  EXPECT_EQ(root.position.to_play(), Color::kWhite);
  auto* leaf = root.SelectLeaf();
  EXPECT_EQ(leaf, root.children[c].get());
}

// Verifies IncorporateResults and BackupValue.
TEST(MctsNodeTest, BackupIncorporateResults) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.02;
  }

  MctsNode::EdgeStats root_stats;
  auto board = TestablePosition(kAlmostDoneBoard, 0, Color::kWhite);
  MctsNode root(&root_stats, board);
  root.SelectLeaf()->IncorporateResults(probs, 0, &root);

  auto* leaf = root.SelectLeaf();
  leaf->IncorporateResults(probs, -1, &root);  // white wins!

  // Root was visited twice: first at the root, then at this child.
  EXPECT_EQ(root.N(), 2);
  // Root has 0 as a prior and two visits with value 0, -1.
  EXPECT_FLOAT_EQ(root.Q(), -1.0 / 3);  // average of 0, 0, -1
  // Leaf should have one visit
  EXPECT_EQ(root.child_N(leaf->move), 1);
  EXPECT_EQ(leaf->N(), 1);
  // And that leaf's value had its parent's Q (0) as a prior, so the Q
  // should now be the average of 0, -1
  EXPECT_FLOAT_EQ(root.child_Q(leaf->move), -0.5);
  EXPECT_FLOAT_EQ(leaf->Q(), -0.5);

  // We're assuming that SelectLeaf() returns a leaf like:
  //   root
  //     |
  //     leaf
  //       |
  //       leaf2
  // which happens in this test because root is W to play and leaf was a W win.
  EXPECT_EQ(root.position.to_play(), Color::kWhite);
  auto* leaf2 = root.SelectLeaf();
  leaf2->IncorporateResults(probs, -0.2, &root);  // another white semi-win
  EXPECT_EQ(root.N(), 3);
  // average of 0, 0, -1, -0.2
  EXPECT_FLOAT_EQ(root.Q(), -0.3);

  EXPECT_EQ(leaf->N(), 2);
  EXPECT_EQ(leaf2->N(), 1);
  // average of 0, -1, -0.2
  EXPECT_FLOAT_EQ(leaf->Q(), root.child_Q(leaf->move));
  EXPECT_FLOAT_EQ(leaf->Q(), -0.4);
  // average of -1, -0.2
  EXPECT_FLOAT_EQ(leaf->child_Q(leaf2->move), -0.6);
  EXPECT_FLOAT_EQ(leaf2->Q(), -0.6);
}

TEST(MctsNodeTest, DoNotExplorePastFinish) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.02;
  }

  MctsNode::EdgeStats root_stats;
  auto board = TestablePosition(kAlmostDoneBoard, 0, Color::kWhite);
  MctsNode root(&root_stats, board);
  root.SelectLeaf()->IncorporateResults(probs, 0, &root);

  auto* first_pass = root.MaybeAddChild(Coord::kPass);
  first_pass->IncorporateResults(probs, 0, &root);
  auto* second_pass = first_pass->MaybeAddChild(Coord::kPass);
  EXPECT_DEATH(second_pass->IncorporateResults(probs, 0, &root),
               "is_game_over");
  float value = second_pass->position.CalculateScore() > 0 ? 1 : -1;
  second_pass->IncorporateEndGameResult(value, &root);
  auto* node_to_explore = second_pass->SelectLeaf();
  // should just stop exploring at the end position.
  EXPECT_EQ(node_to_explore, second_pass);
}

TEST(MctsNodeTest, AddChild) {
  MctsNode::EdgeStats root_stats;
  TestablePosition board("");
  MctsNode root(&root_stats, board);

  Coord c = Coord::FromKgs("B9");
  auto* child = root.MaybeAddChild(c);
  EXPECT_EQ(1, root.children.count(c));
  EXPECT_EQ(child->parent, &root);
  EXPECT_EQ(child->move, c);
}

TEST(MctsNodeTest, AddChildIdempotency) {
  MctsNode::EdgeStats root_stats;
  TestablePosition board("");
  MctsNode root(&root_stats, board);

  Coord c = Coord::FromKgs("B9");
  auto* child = root.MaybeAddChild(c);
  EXPECT_EQ(1, root.children.count(c));
  EXPECT_EQ(1, root.children.size());
  auto* child2 = root.MaybeAddChild(c);
  EXPECT_EQ(child, child2);
  EXPECT_EQ(1, root.children.count(c));
  EXPECT_EQ(1, root.children.size());
}

TEST(MctsNodeTest, NeverSelectIllegalMoves) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.02;
  }
  // let's say the NN were to accidentally put a high weight on an illegal move
  probs[1] = 0.99;

  MctsNode::EdgeStats root_stats;
  auto board = TestablePosition(kAlmostDoneBoard, 0, Color::kWhite);
  MctsNode root(&root_stats, board);
  root.SelectLeaf()->IncorporateResults(probs, 0, &root);

  // and let's say the root were visited a lot of times, which pumps up the
  // action score for unvisited moves...
  root.stats->N = 100000;
  for (int i = 0; i < kNumMoves; ++i) {
    if (root.position.IsMoveLegal(i)) {
      root.edges[i].N = 10000;
    }
  }
  // this should not throw an error...
  auto* leaf = root.SelectLeaf();
  // the returned leaf should not be the illegal move
  EXPECT_NE(leaf->move, 1);

  // and even after injecting noise, we should still not select an illegal move
  Random rnd(1);
  for (int i = 0; i < 10; ++i) {
    std::array<float, kNumMoves> noise;
    rnd.Uniform(0, 1, &noise);
    root.InjectNoise(noise);
    leaf = root.SelectLeaf();
    EXPECT_NE(leaf->move, 1);
  }
}

TEST(MctsNodeTest, DontPickUnexpandedChild) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.001;
  }
  // Make one move really likely so that tree search goes down that path twice
  // even with a virtual loss.
  probs[17] = 0.99;

  MctsNode::EdgeStats root_stats;
  auto board = TestablePosition(kAlmostDoneBoard, 0, Color::kWhite);
  MctsNode root(&root_stats, board);
  root.SelectLeaf()->IncorporateResults(probs, 0, &root);

  auto* leaf1 = root.SelectLeaf();
  EXPECT_EQ(17, leaf1->move);
  leaf1->AddVirtualLoss(&root);

  auto* leaf2 = root.SelectLeaf();
  EXPECT_EQ(leaf2, leaf2);
}

// Verifies that even when one move is hugely more likely than all the others,
// SelectLeaf will eventually start exploring other moves given enough
// iterations.
TEST(MctsNodeTest, TestSelectLeaf) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.001;
  }
  probs[17] = 0.99;

  MctsNode::EdgeStats root_stats;
  auto board = TestablePosition(kAlmostDoneBoard, 0, Color::kWhite);
  MctsNode root(&root_stats, board);
  root.SelectLeaf()->IncorporateResults(probs, 0, &root);

  std::set<MctsNode*> leaves;

  auto* leaf = root.SelectLeaf();
  EXPECT_EQ(17, leaf->move);
  leaf->AddVirtualLoss(&root);
  leaves.insert(leaf);

  for (int i = 0; i < 1000; ++i) {
    leaf = root.SelectLeaf();
    leaf->AddVirtualLoss(&root);
    leaves.insert(leaf);
  }

  // We should have selected at least 2 leaves.
  EXPECT_LE(2, leaves.size());
}

}  // namespace
}  // namespace minigo
