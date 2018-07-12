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

#include "cc/mcts_player.h"

#include <memory>
#include <string>
#include <utility>
#include "absl/memory/memory.h"
#include "cc/algorithm.h"
#include "cc/color.h"
#include "cc/constants.h"
#include "cc/dual_net/fake_net.h"
#include "cc/position.h"
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

class TestablePlayer : public MctsPlayer {
 public:
  explicit TestablePlayer(const Options& options)
      : MctsPlayer(absl::make_unique<FakeNet>(), options) {}

  explicit TestablePlayer(std::unique_ptr<DualNet> network,
                          const Options& options)
      : MctsPlayer(std::move(network), options) {}

  TestablePlayer(absl::Span<const float> fake_priors, float fake_value,
                 const Options& options)
      : MctsPlayer(absl::make_unique<FakeNet>(fake_priors, fake_value),
                   options) {}

  using MctsPlayer::PickMove;
  using MctsPlayer::PlayMove;
  using MctsPlayer::ProcessLeaves;
  using MctsPlayer::rnd;
  using MctsPlayer::TreeSearch;

  std::array<float, kNumMoves> Noise() {
    std::array<float, kNumMoves> noise;
    rnd()->Dirichlet(kDirichletAlpha, &noise);
    return noise;
  }

  DualNet::Output Run(const DualNet::BoardFeatures& features) {
    return network()->Run(features);
  }
};

std::unique_ptr<TestablePlayer> CreateBasicPlayer(MctsPlayer::Options options) {
  // Always use a deterministic random seed.
  options.random_seed = 17;

  auto player = absl::make_unique<TestablePlayer>(options);
  auto* first_node = player->root()->SelectLeaf();
  DualNet::BoardFeatures features;
  std::vector<const Position::Stones*> positions = {
      &player->root()->position.stones()};
  DualNet::SetFeatures(positions, Color::kBlack, &features);
  auto output = player->Run(features);
  first_node->IncorporateResults(output.policy, output.value, player->root());
  return player;
}

std::unique_ptr<TestablePlayer> CreateAlmostDonePlayer(int n) {
  // Always use a deterministic random seed.
  MctsPlayer::Options options;
  options.random_seed = 17;
  options.komi = 2.5;
  // Don't apply random symmetries. If we did, the probabilities we set in
  // the FakeNet won't be chosen correctly (since the board position will be
  // randomly transformed).
  options.random_symmetry = false;

  std::array<float, kNumMoves> probs;
  for (auto& p : probs) {
    p = 0.001;
  }
  probs[Coord(0, 2)] = 0.2;
  probs[Coord(0, 3)] = 0.2;
  probs[Coord(0, 4)] = 0.2;
  probs[Coord::kPass] = 0.2;

  auto player = absl::make_unique<TestablePlayer>(probs, 0, options);
  auto board = TestablePosition(kAlmostDoneBoard, Color::kBlack, n);
  player->InitializeGame(board);
  return player;
}

TEST(MctsPlayerTest, TimeRecommendation) {
  // Early in the game with plenty of time left, the time recommendation should
  // be the requested number of seconds per move.
  EXPECT_EQ(5, TimeRecommendation(0, 5, 1000, 0.98));
  EXPECT_EQ(5, TimeRecommendation(1, 5, 1000, 0.98));
  EXPECT_EQ(5, TimeRecommendation(10, 5, 1000, 0.98));
  EXPECT_EQ(5, TimeRecommendation(50, 5, 1000, 0.98));

  // With a small time limit, the time recommendation should immediately be less
  // than requested.
  EXPECT_GT(1.0f, TimeRecommendation(0, 5, 10, 0.98));

  // Time recommendations for even and odd moves should be identical.
  EXPECT_EQ(TimeRecommendation(20, 5, 10, 0.98),
            TimeRecommendation(21, 5, 10, 0.98));

  // If we're later into the game than should really be possible, time
  // recommendation should be almost zero.
  EXPECT_GT(0.0001, TimeRecommendation(1000, 5, 100, 0.98));
}

TEST(MctsPlayerTest, InjectNoise) {
  MctsPlayer::Options options;
  auto player = CreateBasicPlayer(options);
  auto* root = player->root();

  // FakeNet should return normalized priors.
  float sum_P = 0;
  for (int i = 0; i < kNumMoves; ++i) {
    sum_P += root->child_P(i);
  }
  EXPECT_NEAR(1, sum_P, 0.000001);
  for (int i = 0; i < kNumMoves; ++i) {
    EXPECT_EQ(root->child_U(0), root->child_U(i));
  }

  root->InjectNoise(player->Noise());

  // Priors should still be normalized after injecting noise.
  sum_P = 0;
  for (int i = 0; i < kNumMoves; ++i) {
    sum_P += root->child_P(i);
  }
  EXPECT_NEAR(1, sum_P, 0.000001);

  // With Dirichelet noise, majority of density should be in one node.
  int i = ArgMax(root->edges, MctsNode::CmpP);
  float max_P = root->child_P(i);
  EXPECT_GT(max_P, 3.0 / kNumMoves);
}

// Verify that with soft pick disabled, the player will always choose the best
// move.
TEST(MctsPlayerTest, PickMoveArgMax) {
  MctsPlayer::Options options;
  options.soft_pick = false;
  auto player = CreateBasicPlayer(options);
  auto* root = player->root();

  root->edges[Coord(2, 0)].N = 10;
  root->edges[Coord(1, 0)].N = 5;
  root->edges[Coord(3, 0)].N = 1;

  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(Coord(2, 0), player->PickMove());
  }
}

// Verify that with soft pick enabled, the player will choose moves early in the
// game proportionally to their visit count.
TEST(MctsPlayerTest, PickMoveSoft) {
  MctsPlayer::Options options;
  options.soft_pick = true;
  auto player = CreateBasicPlayer(options);
  auto* root = player->root();

  root->edges[Coord(2, 0)].N = 10;
  root->edges[Coord(1, 0)].N = 5;
  root->edges[Coord(3, 0)].N = 1;

  int count_1_0 = 0;
  int count_2_0 = 0;
  int count_3_0 = 0;
  for (int i = 0; i < 160; ++i) {
    auto move = player->PickMove();
    if (move == Coord(1, 0)) {
      ++count_1_0;
    } else if (move == Coord(2, 0)) {
      ++count_2_0;
    } else {
      ASSERT_EQ(Coord(3, 0), move);
      ++count_3_0;
    }
  }
  EXPECT_NEAR(100, count_2_0, 5);
  EXPECT_NEAR(50, count_1_0, 5);
  EXPECT_NEAR(10, count_3_0, 5);
}

TEST(MctsPlayerTest, DontPassIfLosing) {
  auto player = CreateAlmostDonePlayer(0);
  auto* root = player->root();
  EXPECT_EQ(-0.5, root->position.CalculateScore(player->options().komi));

  for (int i = 0; i < 20; ++i) {
    player->TreeSearch(1);
  }

  // Search should converge on D9 as only winning move.
  auto best_move = ArgMax(root->edges, MctsNode::CmpN);
  ASSERT_EQ(best_move, Coord::FromKgs("D9"));
  // D9 should have a positive value.
  EXPECT_LT(0, root->child_Q(best_move));
  EXPECT_LE(20, root->N());
  // Passing should be ineffective.
  EXPECT_GT(0, root->child_Q(Coord::kPass));

  // No virtual losses should be pending.
  EXPECT_EQ(0, CountPendingVirtualLosses(root));
}

TEST(MctsPlayerTest, ParallelTreeSearch) {
  auto player = CreateAlmostDonePlayer(0);
  auto* root = player->root();

  // Initialize the tree so that the root node has populated children.
  player->TreeSearch(1);
  // Virtual losses should enable multiple searches to happen simultaneously
  // without throwing an error...
  for (int i = 0; i < 5; ++i) {
    player->TreeSearch(5);
  }

  // Search should converge on D9 as only winning move.
  auto best_move = ArgMax(root->edges, MctsNode::CmpN);
  EXPECT_EQ(Coord::FromString("D9"), best_move);
  // D9 should have a positive value.
  EXPECT_LT(0, root->child_Q(best_move));
  EXPECT_LE(20, root->N());
  // Passing should be ineffective.
  EXPECT_GT(0, root->child_Q(Coord::kPass));

  // No virtual losses should be pending.
  EXPECT_EQ(0, CountPendingVirtualLosses(root));
}

TEST(MctsPlayerTest, RidiculouslyParallelTreeSearch) {
  auto player = CreateAlmostDonePlayer(0);
  auto* root = player->root();

  for (int i = 0; i < 10; ++i) {
    // Test that an almost complete game will tree search with
    // # parallelism > # legal moves.
    player->TreeSearch(50);
  }

  // No virtual losses should be pending.
  EXPECT_EQ(0, CountPendingVirtualLosses(root));
}

TEST(MctsPlayerTest, LongGameTreeSearch) {
  auto player = CreateAlmostDonePlayer(kMaxSearchDepth - 2);
  // Test that an almost complete game.
  for (int i = 0; i < 10; ++i) {
    player->TreeSearch(8);
  }
  EXPECT_EQ(0, CountPendingVirtualLosses(player->root()));
  EXPECT_LT(0, player->root()->Q());
}

TEST(MctsPlayerTest, ColdStartParallelTreeSearch) {
  MctsPlayer::Options options;
  options.random_seed = 17;
  auto player = absl::make_unique<TestablePlayer>(absl::Span<const float>(),
                                                  0.17, options);
  auto* root = player->root();

  // Test that parallel tree search doesn't trip on an empty tree.
  EXPECT_EQ(0, root->N());
  EXPECT_EQ(false, root->is_expanded);
  player->TreeSearch(4);
  EXPECT_EQ(0, CountPendingVirtualLosses(root));

  // Even though we attempted to run 4 parallel searchs, the root should have
  // only been selected once (the subsequent calls to SelectLeaf should have
  // returned null).
  EXPECT_EQ(1, root->N());

  // 0.085 = average(0, 0.17), since 0 is the prior on the root.
  EXPECT_NEAR(0.085, root->Q(), 0.01);
}

TEST(MctsPlayerTest, TreeSearchFailsafe) {
  // Test that the failsafe works correctly. It can trigger if the MCTS
  // repeatedly visits a finished game state.
  std::array<float, kNumMoves> probs;
  for (auto& p : probs) {
    p = 0.001;
  }
  probs[Coord::kPass] = 1;  // Make the dummy net always want to pass.

  MctsPlayer::Options options;
  options.random_seed = 17;
  auto player = absl::make_unique<TestablePlayer>(probs, 0, options);
  auto board = TestablePosition("");
  board.PlayMove("pass");
  player->InitializeGame(board);
  player->TreeSearch(1);
  EXPECT_EQ(0, CountPendingVirtualLosses(player->root()));
}

// When presented with a situation where the last move was a pass, and we have
// to decide whether to pass, it should be the first thing we check, but not
// more than that.
TEST(MctsPlayerTest, OnlyCheckGameEndOnce) {
  BoardVisitor bv;
  GroupVisitor gv;
  Position position(&bv, &gv, Color::kBlack);
  position.PlayMove({3, 3});  // B plays.
  position.PlayMove({3, 4});  // W plays.
  position.PlayMove({4, 3});  // B plays.
  // W passes. If B passes too, B would lose by komi..
  position.PlayMove(Coord::kPass);

  auto player = absl::make_unique<TestablePlayer>(MctsPlayer::Options());
  player->InitializeGame(position);
  auto* root = player->root();

  // Initialize the root
  player->TreeSearch(1);
  // Explore a child - should be a pass move.
  player->TreeSearch(1);
  EXPECT_EQ(1, root->child_N(Coord::kPass));
  player->TreeSearch(1);

  // Check that we didn't visit the pass node any more times.
  EXPECT_EQ(1, root->child_N(Coord::kPass));
}

TEST(MctsPlayerTest, ExtractDataNormalEnd) {
  auto player = absl::make_unique<TestablePlayer>(MctsPlayer::Options());
  player->TreeSearch(1);
  player->PlayMove(Coord::kPass);
  player->TreeSearch(1);
  player->PlayMove(Coord::kPass);

  auto* root = player->root();
  EXPECT_TRUE(root->position.is_game_over());
  EXPECT_EQ(Color::kBlack, root->position.to_play());

  ASSERT_EQ(2, player->history().size());

  // White wins by komi
  EXPECT_EQ(-1, player->result());
  EXPECT_EQ("W+7.5", player->result_string());
}

TEST(MctsPlayerTest, ExtractDataResignEnd) {
  auto player = absl::make_unique<TestablePlayer>(MctsPlayer::Options());
  player->TreeSearch(1);
  player->PlayMove({0, 0});
  player->TreeSearch(1);
  player->PlayMove(Coord::kPass);
  player->TreeSearch(1);
  player->PlayMove(Coord::kResign);

  auto* root = player->root();

  // Black is winning on the board.
  EXPECT_LT(0, root->position.CalculateScore(player->options().komi));

  EXPECT_EQ(-1, player->result());
  EXPECT_EQ("W+R", player->result_string());
}

// Fake DualNet implementation used to verify that MctsPlayer symmetries work
// correctly. For each position on the board, MergeFeaturesNet returns a policy
// value depending on the feature planes of that square, if the square or any
// four connected neighbor is set true the policy is set to 0.01.
class MergeFeaturesNet : public DualNet {
 public:
  void RunMany(absl::Span<const BoardFeatures> features,
               absl::Span<Output> outputs, std::string* model) override {
    for (size_t i = 0; i < features.size(); ++i) {
      Run(features[i], &outputs[i]);
    }
    if (model != nullptr) {
      *model = "MergeFeaturesNet";
    }
  }

 private:
  void Run(const BoardFeatures& features, Output* output) {
    for (int c = 0; c < kN * kN; ++c) {
      bool present = false;
      for (const auto n : kNeighborCoords[c]) {
        const float* src = features.data() + n * DualNet::kNumStoneFeatures;
        for (int f = 0; f < DualNet::kNumStoneFeatures - 1; ++f) {
          if (src[f] != 0) {
            present = true;
          }
        }
      }
      output->policy[c] = 0.01 * present;
    }
    output->policy[Coord::kPass] = 0.0;
    output->value = 0.0;
  }
};

TEST(MctsPlayerTest, SymmetriesTest) {
  MctsPlayer::Options options;
  options.random_seed = 17;
  options.random_symmetry = true;
  TestablePlayer player(absl::make_unique<MergeFeaturesNet>(), options);

  // Without playing a move, all features planes should be zero except the last
  // one (it's black's turn to play).
  auto* root = player.root();
  player.ProcessLeaves({&root, 1});
  for (int i = 0; i < kN * kN; ++i) {
    ASSERT_EQ(0.0, root->child_P(i));
  }

  // Play an odd number of moves.
  // Because it's white to play next, the output of the MergeFeaturesNet should
  // only be non-zero except near locations where we have played.
  std::vector<std::unique_ptr<MctsNode>> nodes;
  std::vector<std::string> moves = {"B3", "F1", "C7"};
  auto* parent = root;
  for (const auto& move : moves) {
    nodes.push_back(absl::make_unique<MctsNode>(parent, Coord::FromKgs(move)));
    parent = nodes.back().get();
  }

  // 4 squares near B3 + 3 next to F1 + 4 near C7 should have equal policy.
  float policy_fraction = 1.0 / (4 + 3 + 4);

  // Run the MergeFeaturesNet many times to have a good chance of exercising all
  // the symmetries.
  for (int i = 0; i < 100; ++i) {
    auto* leaf = nodes.back().get();
    player.ProcessLeaves({&leaf, 1});
    ASSERT_EQ(0.0, leaf->child_P(Coord::FromKgs("pass")));
    for (const auto move : moves) {
      // Playing where stones exist is illegal and should have been marked as 0.
      ASSERT_EQ(0.0, leaf->child_P(Coord::FromKgs(move)));
      for (const auto n : kNeighborCoords[Coord::FromKgs(move)]) {
        ASSERT_NEAR(policy_fraction, leaf->child_P(n), 1e-7);
      }
    }
  }
}

}  // namespace
}  // namespace minigo
