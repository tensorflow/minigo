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
#include <set>

#include "absl/memory/memory.h"
#include "cc/position.h"
#include "cc/random.h"
#include "cc/test_utils.h"
#include "cc/zobrist.h"
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

// Test puct and child action score calculation
TEST(MctsNodeTest, UpperConfidenceBound) {
  float epsilon = 1e-7;
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.02;
  }

  MctsNode::EdgeStats root_stats;
  MctsNode root(&root_stats, TestablePosition("", Color::kBlack));
  auto* leaf = root.SelectLeaf();
  EXPECT_EQ(&root, leaf);
  leaf->IncorporateResults(0.0, probs, 0.5, &root);

  // 0.02 are normalized to 1/82
  EXPECT_NEAR(1.0 / 82, root.child_P(0), epsilon);
  EXPECT_NEAR(1.0 / 82, root.child_P(1), epsilon);
  auto puct_policy = [&](const int n) {
    return 2.0 * (std::log((1.0f + n + kUct_base) / kUct_base) + kUct_init) *
           1.0 / 82;
  };
  ASSERT_EQ(1, root.N());
  EXPECT_NEAR(puct_policy(1) * std::sqrt(1) / (1 + 0), root.child_U(0),
              epsilon);

  leaf = root.SelectLeaf();
  leaf->IncorporateResults(0.0, probs, 0.5, &root);
  EXPECT_NE(&root, leaf);
  EXPECT_EQ(&root, leaf->parent);
  EXPECT_EQ(Coord(0), leaf->move);

  // With the first child expanded.
  ASSERT_EQ(2, root.N());
  EXPECT_NEAR(puct_policy(2) * std::sqrt(1) / (1 + 1), root.child_U(0),
              epsilon);
  EXPECT_NEAR(puct_policy(2) * std::sqrt(1) / (1 + 0), root.child_U(1),
              epsilon);

  auto* leaf2 = root.SelectLeaf();
  EXPECT_NE(&root, leaf2);
  EXPECT_EQ(&root, leaf2->parent);
  EXPECT_EQ(Coord(1), leaf2->move);
  leaf2->IncorporateResults(0.0, probs, 0.5, &root);

  // With the 2nd child expanded.
  ASSERT_EQ(3, root.N());
  EXPECT_NEAR(puct_policy(3) * std::sqrt(2) / (1 + 1), root.child_U(0),
              epsilon);
  EXPECT_NEAR(puct_policy(3) * std::sqrt(2) / (1 + 1), root.child_U(1),
              epsilon);
  EXPECT_NEAR(puct_policy(3) * std::sqrt(2) / (1 + 0), root.child_U(2),
              epsilon);
}

// Verifies that no matter who is to play, when we know nothing else, the
// priors should be respected, and the same move should be picked.
TEST(MctsNodeTest, ActionFlipping) {
  Random rnd(1);

  std::array<float, kNumMoves> probs;
  std::uniform_real_distribution<float> dist(0.02, 0.021);
  for (float& prob : probs) {
    prob = rnd();
  }

  MctsNode::EdgeStats black_stats, white_stats;
  MctsNode black_root(&black_stats, TestablePosition("", Color::kBlack));
  MctsNode white_root(&white_stats, TestablePosition("", Color::kWhite));

  black_root.SelectLeaf()->IncorporateResults(0.0, probs, 0, &black_root);
  white_root.SelectLeaf()->IncorporateResults(0.0, probs, 0, &white_root);
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
  Coord c = Coord::FromGtp("D9");
  probs[c] = 0.4;

  MctsNode::EdgeStats root_stats;
  auto board = TestablePosition(kAlmostDoneBoard, Color::kWhite);
  MctsNode root(&root_stats, board);

  root.SelectLeaf()->IncorporateResults(0.0, probs, 0, &root);

  EXPECT_EQ(Color::kWhite, root.position.to_play());
  auto* leaf = root.SelectLeaf();
  EXPECT_EQ(root.children[c].get(), leaf);
}

// Verifies IncorporateResults and BackupValue.
TEST(MctsNodeTest, BackupIncorporateResults) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.02;
  }

  MctsNode::EdgeStats root_stats;
  auto board = TestablePosition(kAlmostDoneBoard, Color::kWhite);
  MctsNode root(&root_stats, board);
  root.SelectLeaf()->IncorporateResults(0.0, probs, 0, &root);

  auto* leaf = root.SelectLeaf();
  leaf->IncorporateResults(0.0, probs, -1, &root);  // white wins!

  // Root was visited twice: first at the root, then at this child.
  EXPECT_EQ(2, root.N());
  // Root has 0 as a prior and two visits with value 0, -1.
  EXPECT_FLOAT_EQ(-1.0 / 3, root.Q());  // average of 0, 0, -1
  // Leaf should have one visit
  EXPECT_EQ(1, root.child_N(leaf->move));
  EXPECT_EQ(1, leaf->N());
  // And that leaf's value had its parent's Q (0) as a prior, so the Q
  // should now be the average of 0, -1
  EXPECT_FLOAT_EQ(-0.5, root.child_Q(leaf->move));
  EXPECT_FLOAT_EQ(-0.5, leaf->Q());

  // We're assuming that SelectLeaf() returns a leaf like:
  //   root
  //     |
  //     leaf
  //       |
  //       leaf2
  // which happens in this test because root is W to play and leaf was a W
  // win.
  EXPECT_EQ(Color::kWhite, root.position.to_play());
  auto* leaf2 = root.SelectLeaf();
  ASSERT_EQ(leaf, leaf2->parent);

  leaf2->IncorporateResults(0.0, probs, -0.2,
                            &root);  // another white semi-win
  EXPECT_EQ(3, root.N());
  // average of 0, 0, -1, -0.2
  EXPECT_FLOAT_EQ(-0.3, root.Q());

  EXPECT_EQ(2, leaf->N());
  EXPECT_EQ(1, leaf2->N());
  // average of 0, -1, -0.2
  EXPECT_FLOAT_EQ(root.child_Q(leaf->move), leaf->Q());
  EXPECT_FLOAT_EQ(-0.4, leaf->Q());
  // average of -1, -0.2
  EXPECT_FLOAT_EQ(-0.6, leaf->child_Q(leaf2->move));
  EXPECT_FLOAT_EQ(-0.6, leaf2->Q());
}

TEST(MctsNodeTest, ExpandChildValueInit) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.02;
  }

  // Any child will do.
  auto board = TestablePosition(kAlmostDoneBoard, Color::kWhite);
  {
    MctsNode::EdgeStats root_stats;
    MctsNode root(&root_stats, board);
    // 0.0 is init-to-parent
    root.IncorporateResults(0.0, probs, 0.1, &root);

    auto* leaf = root.SelectLeaf();
    EXPECT_FLOAT_EQ(0.1, root.child_Q(2));
    EXPECT_FLOAT_EQ(0.1, leaf->Q());

    // 2nd IncorporateResult shouldn't change Q.
    root.IncorporateResults(0.0, probs, 0.9, &root);

    EXPECT_FLOAT_EQ(0.1, root.child_Q(2));
    EXPECT_FLOAT_EQ(0.1, leaf->Q());
  }

  {
    MctsNode::EdgeStats root_stats;
    MctsNode root(&root_stats, board);
    // -2.0 is init-to-loss
    root.IncorporateResults(-2.0, probs, 0.1, &root);

    auto* leaf = root.SelectLeaf();
    EXPECT_FLOAT_EQ(-1.0, root.child_Q(leaf->move));
    EXPECT_FLOAT_EQ(-1.0, leaf->Q());
  }

  {
    MctsNode::EdgeStats root_stats;
    MctsNode root(&root_stats, board);
    // 2.0 is init-to-win (this is silly don't do this)
    root.IncorporateResults(2.0, probs, 0.1, &root);

    auto* leaf = root.SelectLeaf();
    EXPECT_FLOAT_EQ(1.0, root.child_Q(leaf->move));
    EXPECT_FLOAT_EQ(1.0, leaf->Q());
  }

  {
    MctsNode::EdgeStats root_stats;
    MctsNode root(&root_stats, board);
    // 0.25 slightly prefers to explore already visited children.
    root.IncorporateResults(-0.25, probs, 0.1, &root);

    auto* leaf = root.SelectLeaf();
    EXPECT_FLOAT_EQ(-0.15, root.child_Q(leaf->move));
    EXPECT_FLOAT_EQ(-0.15, leaf->Q());
  }
}

TEST(MctsNodeTest, DoNotExplorePastFinish) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.02;
  }

  MctsNode::EdgeStats root_stats;
  auto board = TestablePosition(kAlmostDoneBoard, Color::kWhite);
  MctsNode root(&root_stats, board);
  root.SelectLeaf()->IncorporateResults(0.0, probs, 0, &root);

  auto* first_pass = root.MaybeAddChild(Coord::kPass);
  first_pass->IncorporateResults(0.0, probs, 0, &root);
  auto* second_pass = first_pass->MaybeAddChild(Coord::kPass);
  EXPECT_DEATH(second_pass->IncorporateResults(0.0, probs, 0, &root),
               "game_over");
  float value = second_pass->position.CalculateScore(0) > 0 ? 1 : -1;
  second_pass->IncorporateEndGameResult(value, &root);
  auto* node_to_explore = second_pass->SelectLeaf();
  // should just stop exploring at the end position.
  EXPECT_EQ(second_pass, node_to_explore);
}

TEST(MctsNodeTest, AddChild) {
  MctsNode::EdgeStats root_stats;
  TestablePosition board("");
  MctsNode root(&root_stats, board);

  Coord c = Coord::FromGtp("B9");
  auto* child = root.MaybeAddChild(c);
  EXPECT_EQ(1, root.children.count(c));
  EXPECT_EQ(&root, child->parent);
  EXPECT_EQ(child->move, c);
}

TEST(MctsNodeTest, AddChildIdempotency) {
  MctsNode::EdgeStats root_stats;
  TestablePosition board("");
  MctsNode root(&root_stats, board);

  Coord c = Coord::FromGtp("B9");
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
  // let's say the NN were to accidentally put a high weight on an illegal
  // move
  probs[1] = 0.99;

  MctsNode::EdgeStats root_stats;
  auto board = TestablePosition(kAlmostDoneBoard, Color::kWhite);
  MctsNode root(&root_stats, board);
  root.SelectLeaf()->IncorporateResults(0.0, probs, 0, &root);

  // and let's say the root were visited a lot of times, which pumps up the
  // action score for unvisited moves...
  root.stats->N = 100000;
  for (int i = 0; i < kNumMoves; ++i) {
    if (root.position.ClassifyMove(i) != Position::MoveType::kIllegal) {
      root.edges[i].N = 10000;
    }
  }
  // this should not throw an error...
  auto* leaf = root.SelectLeaf();
  // the returned leaf should not be the illegal move
  EXPECT_NE(1, leaf->move);

  // and even after injecting noise, we should still not select an illegal
  // move
  Random rnd(1);
  for (int i = 0; i < 10; ++i) {
    std::array<float, kNumMoves> noise;
    rnd.Uniform(0, 1, &noise);
    root.InjectNoise(noise, 0.25);
    leaf = root.SelectLeaf();
    EXPECT_NE(1, leaf->move);
  }
}

TEST(MctsNodeTest, DontTraverseUnexpandedChild) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.001;
  }
  // Make one move really likely so that tree search goes down that path twice
  // even with a virtual loss.
  probs[17] = 0.99;

  MctsNode::EdgeStats root_stats;
  auto board = TestablePosition(kAlmostDoneBoard, Color::kWhite);
  MctsNode root(&root_stats, board);
  root_stats.N = 5;
  root.SelectLeaf()->IncorporateResults(0.0, probs, 0, &root);

  auto* leaf1 = root.SelectLeaf();
  EXPECT_EQ(17, leaf1->move);
  leaf1->AddVirtualLoss(&root);

  auto* leaf2 = root.SelectLeaf();
  EXPECT_EQ(leaf1, leaf2);  // assert we didn't go below the first leaf.
}

// Verifies that action score is used as a tie-breaker to choose between moves
// with the same visit count when selecting the best one.
// This test uses raw indices here instead of GTP coords to make it clear that
// without using action score as a tie-breaker, the move with the lower index
// would be selected by GetMostVisitedMove.
TEST(MctsNodeTest, GetMostVisitedPath) {
  // Give two moves a higher probability.
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.001;
  }
  probs[15] = 0.5;
  probs[16] = 0.6;

  MctsNode::EdgeStats root_stats;
  auto board = TestablePosition("", Color::kBlack);
  MctsNode root(&root_stats, board);
  root.SelectLeaf()->IncorporateResults(0.0, probs, 0, &root);

  // We should select the highest probabilty first.
  auto* leaf1 = root.SelectLeaf();
  EXPECT_EQ(Coord(16), leaf1->move);
  leaf1->AddVirtualLoss(&root);

  // Then the second highest probability.
  auto* leaf2 = root.SelectLeaf();
  EXPECT_EQ(Coord(15), leaf2->move);
  leaf1->RevertVirtualLoss(&root);

  // Both Coord(15) and Coord(16) have visit counts of 1.
  // Coord(16) should be selected because of it's higher action score.
  EXPECT_EQ(Coord(16), root.GetMostVisitedMove());
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
  auto board = TestablePosition(kAlmostDoneBoard, Color::kWhite);
  MctsNode root(&root_stats, board);
  root.SelectLeaf()->IncorporateResults(0.0, probs, 0, &root);

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

TEST(MctsNodeTest, NormalizeTest) {
  // Generate probability with sum of policy less than 1
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.001;
  }
  // Five times larger to test normalization
  probs[17] = 0.005;
  probs[18] = 0;

  MctsNode::EdgeStats root_stats;
  auto board = TestablePosition("");
  MctsNode root(&root_stats, board);
  root.IncorporateResults(0.0, probs, 0, &root);

  // Adjust for the one value that is five times larger and one missing value.
  float normalized = 1.0 / (kNumMoves - 1 + 4);
  for (int i = 0; i < kNumMoves; ++i) {
    if (i == 17) {
      EXPECT_FLOAT_EQ(5 * normalized, root.child_P(i));
    } else if (i == 18) {
      EXPECT_FLOAT_EQ(0, root.child_P(i));
    } else {
      EXPECT_FLOAT_EQ(normalized, root.child_P(i));
    }
  }
}

TEST(MctsNodeTest, InjectNoiseOnlyLegalMoves) {
  // Give moves a uniform policy value.
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.02;
  }

  MctsNode::EdgeStats root_stats;
  auto board = TestablePosition(kAlmostDoneBoard, Color::kWhite);
  MctsNode root(&root_stats, board);
  root.IncorporateResults(0.0, probs, 0, &root);

  // kAlmostDoneBoard has 6 legal moves including pass.
  float uniform_policy = 1.0 / 6;

  for (int i = 0; i < kNumMoves; ++i) {
    if (root.legal_moves[i]) {
      EXPECT_FLOAT_EQ(uniform_policy, root.edges[i].P);
    } else {
      EXPECT_FLOAT_EQ(0, root.edges[i].P);
    }
  }

  // and even after injecting noise, we should still not select an illegal
  // move
  Random rnd(1);
  std::array<float, kNumMoves> noise;
  rnd.Uniform(0, 1, &noise);
  root.InjectNoise(noise, 0.25);

  for (int i = 0; i < kNumMoves; ++i) {
    if (root.legal_moves[i]) {
      EXPECT_LT(0.75 * uniform_policy, root.edges[i].P);
      EXPECT_GT(0.75 * uniform_policy + 0.25, root.edges[i].P);
    } else {
      EXPECT_FLOAT_EQ(0, root.edges[i].P);
    }
  }
}

TEST(MctsNodeTest, TestSuperko) {
  // clang-format off
  std::vector<std::string> non_ko_moves = {
    // Some moves at the top edge of the board that don't interfere with the kos
    // at the bottom of the board.
    "A9", "B9", "C9", "D9", "E9", "F9", "G9", "H9", "J9",
    "A8", "B8", "C8", "D8", "E8", "F8", "G8", "H8", "J8",
  };

  std::vector<std::string> ko_moves = {
      // Create two kos threats on the bottom edge of the board:
      // .........
      // .XO...OX.
      // X.XO.O.OX
      "A1", "F1", "B2", "G2", "C1", "H1", "J1", "D1", "H2", "C2",

      // Capture one ko.
      "G1", "B1", "pass", "H1",
  };
  // clang-format on

  // Superko detection inserts caches into the tree at regularly spaced
  // depths. For nodes that don't have a superko dectection cache, a linear
  // search up the tree, comparing the stone hashes at each node is performed
  // until a superko cache is hit. In order to verify that there isn't a bug
  // related to the linear-scan & cache-lookup pair of checks, we run the
  // superko test multiple times, with a different number of moves played at
  // the start each time.
  for (size_t iteration = 0; iteration < non_ko_moves.size(); ++iteration) {
    std::vector<std::unique_ptr<MctsNode>> nodes;
    MctsNode::EdgeStats root_stats;
    BoardVisitor bv;
    GroupVisitor gv;
    nodes.push_back(absl::make_unique<MctsNode>(
        &root_stats, Position(&bv, &gv, Color::kBlack)));

    for (size_t move_idx = 0; move_idx < iteration; ++move_idx) {
      Coord c = Coord::FromGtp(non_ko_moves[move_idx]);
      ASSERT_TRUE(nodes.back()->legal_moves[c]);
      nodes.push_back(absl::make_unique<MctsNode>(nodes.back().get(), c));
    }

    for (const auto& move : ko_moves) {
      Coord c = Coord::FromGtp(move);
      ASSERT_TRUE(nodes.back()->legal_moves[c]);
      nodes.push_back(absl::make_unique<MctsNode>(nodes.back().get(), c));
    }

    // Without superko checking, it should look like capturing the second ko
    // at C1 is valid.
    auto c1 = Coord::FromGtp("C1");
    EXPECT_EQ(Position::MoveType::kCapture,
              nodes.back()->position.ClassifyMove(c1));

    // When checking superko however, playing at C1 is not legal because it
    // repeats a position.
    EXPECT_FALSE(nodes.back()->legal_moves[c1]);
  }
}

}  // namespace
}  // namespace minigo

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::minigo::zobrist::Init(614944751);
  return RUN_ALL_TESTS();
}
