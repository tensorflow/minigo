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

#include "cc/mcts_tree.h"

#include <array>
#include <set>

#include "absl/memory/memory.h"
#include "cc/algorithm.h"
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

static constexpr char kSomeBensonsBoard[] = R"(
    .XO.XO.OO
    X.XXOOOO.
    XXX.XO.OO
    XX..XO.OO
    .X..XO.O.
    XX..XO.OO
    ....XO...
    ....XO...
    ....OO...)";

static constexpr char kOnlyBensonsBoard[] = R"(
    ....X....
    XXXXXXXXX
    X.X.X.X.X
    XXXXXXXXX
    OOOOOOOOO
    OOOOOOOOO
    O.O.O.O.O
    OOOOOOOOO
    ....O....)";

// Test puct and child action score calculation
TEST(MctsTreeTest, UpperConfidenceBound) {
  float epsilon = 1e-7;
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.02;
  }

  MctsTree tree(TestablePosition("", Color::kBlack), {});
  auto* leaf = tree.SelectLeaf(true);
  EXPECT_EQ(tree.root(), leaf);
  tree.IncorporateResults(leaf, probs, 0.5);

  // 0.02 are normalized to 1/82
  EXPECT_NEAR(1.0 / 82, tree.root()->child_P(0), epsilon);
  EXPECT_NEAR(1.0 / 82, tree.root()->child_P(1), epsilon);
  auto puct_policy = [&](const int n) {
    return 2.0 * (std::log((1.0f + n + kUct_base) / kUct_base) + kUct_init) *
           1.0 / 82;
  };
  ASSERT_EQ(1, tree.root()->N());
  EXPECT_NEAR(puct_policy(1) * std::sqrt(1) / (1 + 0), tree.root()->child_U(0),
              epsilon);

  leaf = tree.SelectLeaf(true);
  tree.IncorporateResults(leaf, probs, 0.5);
  EXPECT_NE(tree.root(), leaf);
  EXPECT_EQ(tree.root(), leaf->parent);
  EXPECT_EQ(Coord(0), leaf->move);

  // With the first child expanded.
  ASSERT_EQ(2, tree.root()->N());
  EXPECT_NEAR(puct_policy(2) * std::sqrt(1) / (1 + 1), tree.root()->child_U(0),
              epsilon);
  EXPECT_NEAR(puct_policy(2) * std::sqrt(1) / (1 + 0), tree.root()->child_U(1),
              epsilon);

  auto* leaf2 = tree.SelectLeaf(true);
  EXPECT_NE(tree.root(), leaf2);
  EXPECT_EQ(tree.root(), leaf2->parent);
  EXPECT_EQ(Coord(1), leaf2->move);
  tree.IncorporateResults(leaf2, probs, 0.5);

  // With the 2nd child expanded.
  ASSERT_EQ(3, tree.root()->N());
  EXPECT_NEAR(puct_policy(3) * std::sqrt(2) / (1 + 1), tree.root()->child_U(0),
              epsilon);
  EXPECT_NEAR(puct_policy(3) * std::sqrt(2) / (1 + 1), tree.root()->child_U(1),
              epsilon);
  EXPECT_NEAR(puct_policy(3) * std::sqrt(2) / (1 + 0), tree.root()->child_U(2),
              epsilon);
}

// Verifies that no matter who is to play, when we know nothing else, the
// priors should be respected, and the same move should be picked.
TEST(MctsTreeTest, ActionFlipping) {
  Random rnd(1, 1);

  std::array<float, kNumMoves> probs;
  rnd.Uniform(0.02, 0.021, &probs);

  MctsTree black_tree(TestablePosition("", Color::kBlack), {});
  MctsTree white_tree(TestablePosition("", Color::kWhite), {});

  black_tree.IncorporateResults(black_tree.SelectLeaf(true), probs, 0);
  white_tree.IncorporateResults(white_tree.SelectLeaf(true), probs, 0);
  auto* black_leaf = black_tree.SelectLeaf(true);
  auto* white_leaf = black_tree.SelectLeaf(true);
  EXPECT_EQ(black_leaf->move, white_leaf->move);
  EXPECT_EQ(black_tree.root()->CalculateChildActionScore(),
            white_tree.root()->CalculateChildActionScore());
}

// Verfies that SelectLeaf chooses the child with the highest action score.
TEST(MctsTreeTest, SelectLeaf) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.02;
  }
  Coord c = Coord::FromGtp("D9");
  probs[c] = 0.4;

  auto board = TestablePosition(kAlmostDoneBoard, Color::kWhite);
  MctsTree tree(board, {});

  tree.IncorporateResults(tree.SelectLeaf(true), probs, 0);

  EXPECT_EQ(Color::kWhite, tree.to_play());
  auto* leaf = tree.SelectLeaf(true);
  EXPECT_EQ(tree.root()->children.find(c)->second.get(), leaf);
}

// Verifies IncorporateResults and BackupValue.
TEST(MctsTreeTest, BackupIncorporateResults) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.02;
  }

  auto board = TestablePosition(kAlmostDoneBoard, Color::kWhite);
  MctsTree tree(board, {});
  tree.IncorporateResults(tree.SelectLeaf(true), probs, 0);

  auto* leaf = tree.SelectLeaf(true);
  tree.IncorporateResults(leaf, probs, -1);  // white wins!

  // Root was visited twice: first at the root, then at this child.
  EXPECT_EQ(2, tree.root()->N());
  // Root has 0 as a prior and two visits with value 0, -1.
  EXPECT_FLOAT_EQ(-1.0 / 3, tree.root()->Q());  // average of 0, 0, -1
  // Leaf should have one visit
  EXPECT_EQ(1, tree.root()->child_N(leaf->move));
  EXPECT_EQ(1, leaf->N());
  // And that leaf's value had its parent's Q (0) as a prior, so the Q
  // should now be the average of 0, -1
  EXPECT_FLOAT_EQ(-0.5, tree.root()->child_Q(leaf->move));
  EXPECT_FLOAT_EQ(-0.5, leaf->Q());

  // We're assuming that SelectLeaf() returns a leaf like:
  //   root
  //     |
  //     leaf
  //       |
  //       leaf2
  // which happens in this test because root is W to play and leaf was a W
  // win.
  EXPECT_EQ(Color::kWhite, tree.to_play());
  auto* leaf2 = tree.SelectLeaf(true);
  ASSERT_EQ(leaf, leaf2->parent);

  tree.IncorporateResults(leaf2, probs, -0.2);  // another white semi-win
  EXPECT_EQ(3, tree.root()->N());
  // average of 0, 0, -1, -0.2
  EXPECT_FLOAT_EQ(-0.3, tree.root()->Q());

  EXPECT_EQ(2, leaf->N());
  EXPECT_EQ(1, leaf2->N());
  // average of 0, -1, -0.2
  EXPECT_FLOAT_EQ(tree.root()->child_Q(leaf->move), leaf->Q());
  EXPECT_FLOAT_EQ(-0.4, leaf->Q());
  // average of -1, -0.2
  EXPECT_FLOAT_EQ(-0.6, leaf->child_Q(leaf2->move));
  EXPECT_FLOAT_EQ(-0.6, leaf2->Q());
}

TEST(MctsTreeTest, ExpandChildValueInit) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.02;
  }

  // Any child will do.
  MctsTree::Options options;
  auto board = TestablePosition(kAlmostDoneBoard, Color::kWhite);
  {
    // 0.0 is init-to-parent
    options.value_init_penalty = 0;
    MctsTree tree(board, options);
    auto* root = tree.SelectLeaf(true);
    ASSERT_EQ(tree.root(), root);
    tree.IncorporateResults(root, probs, 0.1);

    auto* leaf = tree.SelectLeaf(true);
    EXPECT_FLOAT_EQ(0.1, root->child_Q(2));
    EXPECT_FLOAT_EQ(0.1, leaf->Q());

    // 2nd IncorporateResult shouldn't change Q.
    tree.IncorporateResults(root, probs, 0.9);

    EXPECT_FLOAT_EQ(0.1, root->child_Q(2));
    EXPECT_FLOAT_EQ(0.1, leaf->Q());
  }

  {
    // -2.0 is init-to-loss
    options.value_init_penalty = -2;
    MctsTree tree(board, options);
    auto* root = tree.SelectLeaf(true);
    ASSERT_EQ(tree.root(), root);
    tree.IncorporateResults(root, probs, 0.1);

    auto* leaf = tree.SelectLeaf(true);
    EXPECT_FLOAT_EQ(-1.0, root->child_Q(leaf->move));
    EXPECT_FLOAT_EQ(-1.0, leaf->Q());
  }

  {
    // 2.0 is init-to-win (this is silly don't do this)
    options.value_init_penalty = 2;
    MctsTree tree(board, options);
    auto* root = tree.SelectLeaf(true);
    ASSERT_EQ(tree.root(), root);
    tree.IncorporateResults(root, probs, 0.1);

    auto* leaf = tree.SelectLeaf(true);
    EXPECT_FLOAT_EQ(1.0, root->child_Q(leaf->move));
    EXPECT_FLOAT_EQ(1.0, leaf->Q());
  }

  {
    // 0.25 slightly prefers to explore already visited children.
    options.value_init_penalty = -0.25;
    MctsTree tree(board, options);
    auto* root = tree.SelectLeaf(true);
    ASSERT_EQ(tree.root(), root);
    tree.IncorporateResults(root, probs, 0.1);

    auto* leaf = tree.SelectLeaf(true);
    EXPECT_FLOAT_EQ(-0.15, root->child_Q(leaf->move));
    EXPECT_FLOAT_EQ(-0.15, leaf->Q());
  }
}

TEST(MctsTreeTest, DoNotExplorePastFinish) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.02;
  }
  probs[Coord::kPass] = 1;

  auto board = TestablePosition(kAlmostDoneBoard, Color::kWhite);
  MctsTree tree(board, {});
  tree.IncorporateResults(tree.SelectLeaf(true), probs, 0);

  auto* first_pass = tree.SelectLeaf(true);
  ASSERT_EQ(Coord::kPass, first_pass->move);
  tree.IncorporateResults(first_pass, probs, 0);

  auto* second_pass = tree.SelectLeaf(true);
  ASSERT_EQ(Coord::kPass, second_pass->move);
  EXPECT_DEATH(tree.IncorporateResults(second_pass, probs, 0), "game_over");
  float value = second_pass->position.CalculateScore(0) > 0 ? 1 : -1;
  tree.IncorporateEndGameResult(second_pass, value);

  // should just stop exploring at the end position.
  tree.PlayMove(Coord::kPass);
  tree.PlayMove(Coord::kPass);
  auto* node_to_explore = tree.SelectLeaf(true);
  EXPECT_EQ(second_pass, node_to_explore);
}

TEST(MctsTreeTest, AddChild) {
  MctsTree tree(Position(Color::kBlack), {});
  auto* root = tree.SelectLeaf(true);

  Coord c = Coord::FromGtp("B9");
  auto* child = root->MaybeAddChild(c);
  EXPECT_EQ(1, root->children.count(c));
  EXPECT_EQ(root, child->parent);
  EXPECT_EQ(child->move, c);
}

TEST(MctsTreeTest, AddChildIdempotency) {
  MctsTree tree(Position(Color::kBlack), {});
  auto* root = tree.SelectLeaf(true);

  Coord c = Coord::FromGtp("B9");
  auto* child = root->MaybeAddChild(c);
  EXPECT_EQ(1, root->children.count(c));
  EXPECT_EQ(1, root->children.size());
  auto* child2 = root->MaybeAddChild(c);
  EXPECT_EQ(child, child2);
  EXPECT_EQ(1, root->children.count(c));
  EXPECT_EQ(1, root->children.size());
}

TEST(MctsTreeTest, NeverSelectIllegalMoves) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.02;
  }
  // let's say the NN were to accidentally put a high weight on an illegal
  // move
  probs[1] = 0.99;

  auto board = TestablePosition(kAlmostDoneBoard, Color::kWhite);
  MctsTree tree(board, {});
  auto* root = tree.SelectLeaf(true);
  ASSERT_EQ(tree.root(), root);
  tree.IncorporateResults(root, probs, 0);

  // and let's say the root were visited a lot of times, which pumps up the
  // action score for unvisited moves...
  root->stats->N = 100000;
  for (int i = 0; i < kNumMoves; ++i) {
    if (root->position.ClassifyMoveIgnoringSuperko(i) !=
        Position::MoveType::kIllegal) {
      root->edges[i].N = 10000;
    }
  }
  // this should not throw an error...
  auto* leaf = tree.SelectLeaf(true);
  // the returned leaf should not be the illegal move
  EXPECT_NE(1, leaf->move);

  // and even after injecting noise, we should still not select an illegal
  // move
  Random rnd(1, 1);
  for (int i = 0; i < 10; ++i) {
    std::array<float, kNumMoves> noise;
    rnd.Uniform(0, 1, &noise);
    tree.InjectNoise(noise, 0.25);
    leaf = tree.SelectLeaf(true);
    EXPECT_NE(1, leaf->move);
  }
}

TEST(MctsTreeTest, DontTraverseUnexpandedChild) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.001;
  }
  // Make one move really likely so that tree search goes down that path twice
  // even with a virtual loss.
  probs[17] = 0.99;

  auto board = TestablePosition(kAlmostDoneBoard, Color::kWhite);
  MctsTree tree(board, {});
  tree.IncorporateResults(tree.SelectLeaf(true), probs, 0);

  auto* leaf1 = tree.SelectLeaf(true);
  EXPECT_EQ(17, leaf1->move);
  tree.AddVirtualLoss(leaf1);

  auto* leaf2 = tree.SelectLeaf(true);
  EXPECT_EQ(leaf1, leaf2);  // assert we didn't go below the first leaf.
}

// Verifies that action score is used as a tie-breaker to choose between moves
// with the same visit count when selecting the best one.
// This test uses raw indices here instead of GTP coords to make it clear that
// without using action score as a tie-breaker, the move with the lower index
// would be selected by GetMostVisitedMove.
TEST(MctsTreeTest, GetMostVisitedPath) {
  // Give two moves a higher probability.
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.001;
  }
  probs[15] = 0.5;
  probs[16] = 0.6;

  auto board = TestablePosition("", Color::kBlack);
  MctsTree tree(board, {});
  tree.IncorporateResults(tree.SelectLeaf(true), probs, 0);

  // We should select the highest probabilty first.
  auto* leaf1 = tree.SelectLeaf(true);
  EXPECT_EQ(Coord(16), leaf1->move);
  tree.AddVirtualLoss(leaf1);
  tree.IncorporateResults(leaf1, probs, 0);

  // Then the second highest probability.
  auto* leaf2 = tree.SelectLeaf(true);
  EXPECT_EQ(Coord(15), leaf2->move);
  tree.RevertVirtualLoss(leaf1);
  tree.IncorporateResults(leaf2, probs, 0);

  // Both Coord(15) and Coord(16) have visit counts of 1.
  // Coord(16) should be selected because of it's higher action score.
  EXPECT_EQ(Coord(16), tree.root()->GetMostVisitedMove());
}

TEST(MctsTreeTest, GetMostVisitedBensonRestriction) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.001;
  }
  probs[0] = 0.002;  // A9, a bensons point, has higher prior.
  auto board = TestablePosition(kSomeBensonsBoard, Color::kBlack);
  MctsTree tree(board, {});
  for (int i = 0; i < 10; i++) {
    tree.IncorporateResults(tree.SelectLeaf(true), probs, 0);
  }

  EXPECT_EQ(Coord(0), tree.root()->GetMostVisitedMove(false));
  EXPECT_NE(Coord(0), tree.root()->GetMostVisitedMove(true));
}

// Pass is still a valid choice, with or without removing pass-alive areas.
TEST(MctsTreeTest, BensonRestrictionStillPasses) {
  auto board = TestablePosition(kAlmostDoneBoard, Color::kWhite);
  MctsTree tree(board, {});

  auto* root = tree.SelectLeaf(true);
  ASSERT_EQ(tree.root(), root);
  for (int i = 0; i < kNumMoves; ++i) {
    if (root->position.ClassifyMoveIgnoringSuperko(i) !=
        Position::MoveType::kIllegal) {
      root->edges[i].N = 10;
    }
  }
  root->edges[Coord::kPass].N = 100;

  EXPECT_EQ(Coord::kPass, root->GetMostVisitedMove(false));
  EXPECT_EQ(Coord::kPass, root->GetMostVisitedMove(true));
}

TEST(MctsTreeTest, ReshapePrunesBensonsVisits) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.001;
  }
  probs[0] = 0.002;  // A9, a bensons point, has higher prior.

  auto board = TestablePosition(kSomeBensonsBoard, Color::kBlack);
  MctsTree::Options options;
  options.restrict_in_bensons = true;
  MctsTree tree(board, options);
  options.restrict_in_bensons = false;
  MctsTree tree2(board, options);
  for (int i = 0; i < 10; i++) {
    tree.IncorporateResults(tree.SelectLeaf(true), probs, 0);
    tree2.IncorporateResults(tree2.SelectLeaf(true), probs, 0);
  }

  EXPECT_NE(tree.root()->edges[0].N, 0);  // A9 should've had visits.
  tree.ReshapeFinalVisits();
  EXPECT_EQ(tree.root()->edges[0].N, 0);  // Reshape should've removed them.

  EXPECT_NE(tree2.root()->edges[0].N, 0);    // A9 should've had visits.
  auto original = tree2.root()->edges[0].N;  // Store them.
  tree2.ReshapeFinalVisits();
  EXPECT_NE(tree2.root()->edges[0].N, 0);  // Reshape shouldn't've removed them.
  EXPECT_EQ(original,
            tree2.root()->edges[0].N);  // And they should be the same.
}

TEST(MctsTreeTest, ReshapeWhenOnlyBensons) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.01;
  }
  // Let's only explore moves in benson's regions.
  probs[Coord::kPass] = 0;

  auto board = TestablePosition(kOnlyBensonsBoard, Color::kBlack);
  MctsTree::Options options;
  options.restrict_in_bensons = true;
  MctsTree tree(board, options);
  options.restrict_in_bensons = false;
  MctsTree tree2(board, options);
  for (int i = 0; i < 10; i++) {
    tree.IncorporateResults(tree.SelectLeaf(true), probs, 0);
    tree2.IncorporateResults(tree2.SelectLeaf(true), probs, 0);
  }

  EXPECT_EQ(tree.root()->edges[Coord::kPass].N,
            0);  // Pass should no visits.
  EXPECT_EQ(tree2.root()->edges[Coord::kPass].N, 0);

  // Reshape with bensons restricted should add one.
  tree.ReshapeFinalVisits();
  EXPECT_EQ(tree.root()->edges[Coord::kPass].N, 1);

  // Reshape with bensons not restricted should NOT add one.
  tree2.ReshapeFinalVisits();
  EXPECT_EQ(tree2.root()->edges[Coord::kPass].N, 0);
}

// Verifies that even when one move is hugely more likely than all the others,
// SelectLeaf will eventually start exploring other moves given enough
// iterations.
TEST(MctsTreeTest, TestSelectLeaf) {
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.001;
  }
  probs[17] = 0.99;

  auto board = TestablePosition(kAlmostDoneBoard, Color::kWhite);
  MctsTree tree(board, {});
  tree.IncorporateResults(tree.SelectLeaf(true), probs, 0);

  std::set<MctsNode*> leaves;

  auto* leaf = tree.SelectLeaf(true);
  EXPECT_EQ(17, leaf->move);
  tree.AddVirtualLoss(leaf);
  leaves.insert(leaf);

  for (int i = 0; i < 1000; ++i) {
    leaf = tree.SelectLeaf(true);
    tree.AddVirtualLoss(leaf);
    leaves.insert(leaf);
  }

  // We should have selected at least 2 leaves.
  EXPECT_LE(2, leaves.size());
}

class ReshapeTargetTest : public ::testing::Test {
 protected:
  void SetUp() override { best_ = -1; }

  void SearchPosition(const Position& p) {
    float to_play = p.to_play() == Color::kBlack ? 1 : -1;

    std::array<float, kNumMoves> probs;
    for (float& prob : probs) {
      prob = 0.001;
    }
    probs[17] = 0.99;

    tree_ = absl::make_unique<MctsTree>(p, MctsTree::Options());
    tree_->IncorporateResults(tree_->SelectLeaf(true), probs, 0);
    const auto* root = tree_->root();

    MctsNode* leaf;

    // We gave one move a high prior and a neutral value.
    // After many reads, U will increase for the other moves, but they're worse
    // than the one with the high prior.
    // As a result, we can prune those away until the uncertainty rises to
    // compensate for their worse reward estimate.
    for (int i = 0; i < 10000; ++i) {
      leaf = tree_->SelectLeaf(true);
      if (leaf->move == 17) {
        tree_->BackupValue(leaf, 0.0);
      } else {
        tree_->BackupValue(leaf, to_play * -0.10);
      }
    }

    // Child_Q(i), as an average, is actually just computed as W/N.
    // Since we're changing N, we'll want to save the Q-values for the children,
    pre_scores_ = root->CalculateChildActionScore();

    std::array<float, kNumMoves> saved_Q;
    for (int i = 0; i < kNumMoves; ++i) {
      saved_Q[i] = root->child_Q(i);
    }
    best_ = root->GetMostVisitedMove();
    tree_->ReshapeFinalVisits();

    float U_common =
        root->U_scale() * std::sqrt(std::max<float>(1, root->N() - 1));

    // Our tests want to verify that we lowered N until the action score
    // (computed using the after-search estimate of Q) was nearly equal to the
    // action score of the best move.
    //
    // Since "reshaping the target distribution" means twiddling the visit
    // counts, the action scores -- based on Q -- will be misleading.  So,
    // compute the action score using the saved values of Q, as outlined above.
    for (int i = 0; i < kNumMoves; ++i) {
      post_scores_[i] = (saved_Q[i] * to_play + (U_common * root->child_P(i) /
                                                 (1 + root->child_N(i))));
      post_scores_[i] -= 1000.0f * !p.legal_move(i);
    }
  }

  std::array<float, kNumMoves> post_scores_;
  std::array<float, kNumMoves> pre_scores_;
  std::unique_ptr<MctsTree> tree_;
  size_t best_;
};

TEST_F(ReshapeTargetTest, TestReshapeTargetsWhite) {
  auto board = TestablePosition("", Color::kWhite);
  SearchPosition(board);
  int tot_N = 0;

  // Scores should never get smaller as a result of visits being deducted.
  for (int i = 0; i < kNumMoves; i++) {
    EXPECT_LE(pre_scores_[i], post_scores_[i]);
    tot_N += tree_->root()->child_N(i);
  }
  // Score for original best move should be the same.
  EXPECT_EQ(pre_scores_[best_], post_scores_[best_]);
  // Root visits is now greater than sum(child visits)
  EXPECT_LT(tot_N, tree_->root()->N());
  // For the default cpuct params, this should only trim ~1% of reads.
  // If we trimmed over 10%, something is probably wrong.
  EXPECT_GT(tot_N, tree_->root()->N() * 0.90);
}

// As above
TEST_F(ReshapeTargetTest, TestReshapeTargetsBlack) {
  auto board = TestablePosition("", Color::kBlack);
  SearchPosition(board);
  int tot_N = 0;
  for (int i = 0; i < kNumMoves; i++) {
    EXPECT_LE(pre_scores_[i], post_scores_[i]);
    tot_N += tree_->root()->child_N(i);
  }
  EXPECT_EQ(pre_scores_[best_], post_scores_[best_]);
  EXPECT_LT(tot_N, tree_->root()->N());
  EXPECT_GT(tot_N, tree_->root()->N() * 0.90);
}

TEST(MctsTreeTest, NormalizeTest) {
  // Generate probability with sum of policy less than 1
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.001;
  }
  // Five times larger to test normalization
  probs[17] = 0.005;
  probs[18] = 0;

  auto board = TestablePosition("");
  MctsTree tree(board, {});
  tree.IncorporateResults(tree.SelectLeaf(true), probs, 0);

  // Adjust for the one value that is five times larger and one missing value.
  float normalized = 1.0 / (kNumMoves - 1 + 4);
  for (int i = 0; i < kNumMoves; ++i) {
    if (i == 17) {
      EXPECT_FLOAT_EQ(5 * normalized, tree.root()->child_P(i));
    } else if (i == 18) {
      EXPECT_FLOAT_EQ(0, tree.root()->child_P(i));
    } else {
      EXPECT_FLOAT_EQ(normalized, tree.root()->child_P(i));
    }
  }
}

TEST(MctsTreeTest, InjectNoise) {
  MctsTree tree(Position(Color::kBlack), {});

  Random rnd(456943875, 1);

  // Generate some uniform priors.
  std::array<float, kNumMoves> policy;
  rnd.Uniform(&policy);
  for (auto& x : policy) {
    x = 1.0 / kNumMoves;
  }
  float value = 0.2;

  tree.IncorporateResults(tree.SelectLeaf(true), policy, value);

  // Check the priors are normalized.
  float sum_P = 0;
  for (int i = 0; i < kNumMoves; ++i) {
    sum_P += tree.root()->child_P(i);
  }
  EXPECT_NEAR(1, sum_P, 0.000001);
  for (int i = 0; i < kNumMoves; ++i) {
    EXPECT_EQ(tree.root()->child_U(0), tree.root()->child_U(i));
  }

  std::array<float, kNumMoves> noise;
  rnd.Dirichlet(kDirichletAlpha, &noise);
  tree.InjectNoise(noise, 0.25);

  // Priors should still be normalized after injecting noise.
  sum_P = 0;
  for (int i = 0; i < kNumMoves; ++i) {
    sum_P += tree.root()->child_P(i);
  }
  EXPECT_NEAR(1, sum_P, 0.000001);

  // With Dirichelet noise, majority of density should be in one node.
  int i = ArgMax(tree.root()->edges, MctsNode::CmpP);
  float max_P = tree.root()->child_P(i);
  EXPECT_GT(max_P, 3.0 / kNumMoves);
}

TEST(MctsTreeTest, InjectNoiseOnlyLegalMoves) {
  // Give moves a uniform policy value.
  std::array<float, kNumMoves> probs;
  for (float& prob : probs) {
    prob = 0.02;
  }

  auto board = TestablePosition(kAlmostDoneBoard, Color::kWhite);
  MctsTree tree(board, {});
  tree.IncorporateResults(tree.SelectLeaf(true), probs, 0);

  // kAlmostDoneBoard has 6 legal moves including pass.
  float uniform_policy = 1.0 / 6;

  for (int i = 0; i < kNumMoves; ++i) {
    if (tree.is_legal_move(i)) {
      EXPECT_FLOAT_EQ(uniform_policy, tree.root()->edges[i].P);
    } else {
      EXPECT_FLOAT_EQ(0, tree.root()->edges[i].P);
    }
  }

  // and even after injecting noise, we should still not select an illegal
  // move
  Random rnd(1, 1);
  std::array<float, kNumMoves> noise;
  rnd.Uniform(0, 1, &noise);
  tree.InjectNoise(noise, 0.25);

  for (int i = 0; i < kNumMoves; ++i) {
    if (tree.is_legal_move(i)) {
      EXPECT_LT(0.75 * uniform_policy, tree.root()->edges[i].P);
      EXPECT_GT(0.75 * uniform_policy + 0.25, tree.root()->edges[i].P);
    } else {
      EXPECT_FLOAT_EQ(0, tree.root()->edges[i].P);
    }
  }
}

TEST(MctsTreeTest, TestSuperko) {
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
    MctsTree tree(Position(Color::kBlack), {});
    for (size_t move_idx = 0; move_idx < iteration; ++move_idx) {
      tree.PlayMove(Coord::FromGtp(non_ko_moves[move_idx]));
    }

    for (const auto& move : ko_moves) {
      tree.PlayMove(Coord::FromGtp(move));
    }

    // Without superko checking, it should look like capturing the second ko
    // at C1 is valid.
    auto c1 = Coord::FromGtp("C1");
    EXPECT_EQ(Position::MoveType::kCapture,
              tree.root()->position.ClassifyMoveIgnoringSuperko(c1));

    // When checking superko however, playing at C1 is not legal because it
    // repeats a position.
    EXPECT_FALSE(tree.is_legal_move(c1));
  }
}

// Verify that with soft pick disabled, the player will always choose the best
// move.
TEST(MctsTreeTest, PickMoveArgMax) {
  MctsTree::Options options;
  options.soft_pick_enabled = false;
  MctsTree tree(Position(Color::kBlack), options);

  auto* root = tree.SelectLeaf(true);
  ASSERT_EQ(tree.root(), root);

  std::vector<std::pair<Coord, int>> child_visits = {
      {{2, 0}, 10},
      {{1, 0}, 5},
      {{3, 0}, 1},
  };
  for (const auto& p : child_visits) {
    root->MaybeAddChild(p.first);
    root->edges[p.first].N = p.second;
  }

  Random rnd(888, 1);
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(Coord(2, 0), tree.PickMove(&rnd));
  }
}

// Verify that with soft pick enabled, the player will choose moves early in the
// game proportionally to their visit count.
TEST(MctsTreeTest, PickMoveSoft) {
  MctsTree::Options options;
  options.soft_pick_enabled = true;
  MctsTree tree(Position(Color::kBlack), options);

  auto* root = tree.SelectLeaf(true);
  ASSERT_EQ(tree.root(), root);

  root->edges[Coord(2, 0)].N = 10;
  root->edges[Coord(1, 0)].N = 5;
  root->edges[Coord(3, 0)].N = 1;

  int count_1_0 = 0;
  int count_2_0 = 0;
  int count_3_0 = 0;

  Random rnd(888, 1);
  for (int i = 0; i < 1600; ++i) {
    auto move = tree.PickMove(&rnd);
    if (move == Coord(1, 0)) {
      ++count_1_0;
    } else if (move == Coord(2, 0)) {
      ++count_2_0;
    } else {
      ASSERT_EQ(Coord(3, 0), move);
      ++count_3_0;
    }
  }
  EXPECT_NEAR(1000, count_2_0, 50);
  EXPECT_NEAR(500, count_1_0, 50);
  EXPECT_NEAR(100, count_3_0, 50);
}

}  // namespace
}  // namespace minigo

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::minigo::zobrist::Init(614944751);
  return RUN_ALL_TESTS();
}
