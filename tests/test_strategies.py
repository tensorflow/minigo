# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import mock
import numpy as np

from absl import flags

import coords
import go
from tests import test_utils
from strategies import MCTSPlayer, time_recommendation

ALMOST_DONE_BOARD = test_utils.load_board('''
.XO.XO.OO
X.XXOOOO.
XXXXXOOOO
XXXXXOOOO
.XXXXOOO.
XXXXXOOOO
.XXXXOOO.
XXXXXOOOO
XXXXOOOOO
''')

# Tromp taylor means black can win if we hit the move limit.
TT_FTW_BOARD = test_utils.load_board('''
.XXOOOOOO
X.XOO...O
.XXOO...O
X.XOO...O
.XXOO..OO
X.XOOOOOO
.XXOOOOOO
X.XXXXXXX
XXXXXXXXX
''')

SEND_TWO_RETURN_ONE = go.Position(
    board=ALMOST_DONE_BOARD,
    n=70,
    komi=2.5,
    caps=(1, 4),
    ko=None,
    recent=(go.PlayerMove(go.BLACK, (0, 1)),
            go.PlayerMove(go.WHITE, (0, 8))),
    to_play=go.BLACK
)


class DummyNet():
    def __init__(self, fake_priors=None, fake_value=0):
        if fake_priors is None:
            fake_priors = np.ones((go.N ** 2) + 1) / (go.N ** 2 + 1)
        self.fake_priors = fake_priors
        self.fake_value = fake_value

    def run(self, position):
        return self.fake_priors, self.fake_value

    def run_many(self, positions):
        if not positions:
            raise ValueError(
                "No positions passed! (Tensorflow would have failed here.")
        return [self.fake_priors] * len(positions), [self.fake_value] * len(positions)


def initialize_basic_player(position=None):
    player = MCTSPlayer(DummyNet())
    player.initialize_game(position)
    first_node = player.root.select_leaf()
    first_node.incorporate_results(
        *player.network.run(player.root.position), up_to=player.root)
    return player


def initialize_almost_done_player():
    probs = np.array([.001] * (go.N * go.N + 1))
    probs[2:5] = 0.2  # some legal moves along the top.
    probs[-1] = 0.2  # passing is also ok
    net = DummyNet(fake_priors=probs)
    player = MCTSPlayer(net)
    # root position is white to play with no history == white passed.
    player.initialize_game(SEND_TWO_RETURN_ONE)
    return player


class TestMCTSPlayer(test_utils.MiniGoUnitTest):
    def test_time_controls(self):
        secs_per_move = 5
        for time_limit in (10, 100, 1000):
            # in the worst case imaginable, let's say a game goes 1000 moves long
            move_numbers = range(0, 1000, 2)
            total_time_spent = sum(
                time_recommendation(move_num, secs_per_move,
                                    time_limit=time_limit)
                for move_num in move_numbers)
            # we should not exceed available game time
            self.assertLess(total_time_spent, time_limit)
            # but we should have used at least 95% of our time by the end.
            self.assertGreater(total_time_spent, time_limit * 0.95)

    def test_inject_noise(self):
        player = initialize_basic_player()
        sum_priors = np.sum(player.root.child_prior)
        # dummyNet should return normalized priors.
        self.assertAlmostEqual(1, sum_priors)
        self.assertTrue(np.all(player.root.child_U == player.root.child_U[0]))

        player.root.inject_noise()
        new_sum_priors = np.sum(player.root.child_prior)
        # priors should still be normalized after injecting noise
        self.assertAlmostEqual(sum_priors, new_sum_priors)

        # With dirichelet noise, majority of density should be in one node.
        max_p = np.max(player.root.child_prior)
        self.assertGreater(max_p, 3 / (go.N ** 2 + 1))

    def test_pick_moves(self):
        player = initialize_basic_player()
        root = player.root
        root.child_N[coords.to_flat((2, 0))] = 10
        root.child_N[coords.to_flat((1, 0))] = 5
        root.child_N[coords.to_flat((3, 0))] = 1

        root.position.n = go.N ** 2  # move 81, or 361, or... Endgame.

        # Assert we're picking deterministically
        self.assertTrue(root.position.n > player.temp_threshold)
        move = player.pick_move()
        self.assertEqual((2, 0), move)

        # But if we're in the early part of the game, pick randomly
        root.position.n = 3
        self.assertFalse(player.root.position.n > player.temp_threshold)

        with mock.patch('random.random', lambda: .5):
            move = player.pick_move()
            self.assertEqual((2, 0), move)

        with mock.patch('random.random', lambda: .99):
            move = player.pick_move()
            self.assertEqual(move, (3, 0))

    def test_dont_pass_if_losing(self):
        player = initialize_almost_done_player()

        # check -- white is losing.
        self.assertEqual(-0.5, player.root.position.score())

        for _ in range(20):
            player.tree_search()
        # uncomment to debug this test
        # print(player.root.describe())

        # Search should converge on D9 as only winning move.
        flattened = coords.to_flat(coords.from_kgs('D9'))
        best_move = np.argmax(player.root.child_N)
        self.assertEqual(flattened, best_move)
        # D9 should have a positive value
        self.assertGreater(player.root.children[flattened].Q, 0)
        self.assertGreaterEqual(player.root.N, 20)
        # passing should be ineffective.
        self.assertLess(player.root.child_Q[-1], 0)
        # no virtual losses should be pending
        self.assertNoPendingVirtualLosses(player.root)
        # uncomment to debug this test
        # print(player.root.describe())

    def test_parallel_tree_search(self):
        player = initialize_almost_done_player()
        # check -- white is losing.
        self.assertEqual(-0.5, player.root.position.score())
        # initialize the tree so that the root node has populated children.
        player.tree_search(parallel_readouts=1)
        # virtual losses should enable multiple searches to happen simultaneously
        # without throwing an error...
        for _ in range(5):
            player.tree_search(parallel_readouts=4)
        # uncomment to debug this test
        # print(player.root.describe())

        # Search should converge on D9 as only winning move.
        flattened = coords.to_flat(coords.from_kgs('D9'))
        best_move = np.argmax(player.root.child_N)
        self.assertEqual(flattened, best_move)
        # D9 should have a positive value
        self.assertGreater(player.root.children[flattened].Q, 0)
        self.assertGreaterEqual(player.root.N, 20)
        # passing should be ineffective.
        self.assertLess(player.root.child_Q[-1], 0)
        # no virtual losses should be pending
        self.assertNoPendingVirtualLosses(player.root)

    def test_ridiculously_parallel_tree_search(self):
        player = initialize_almost_done_player()
        # Test that an almost complete game
        # will tree search with # parallelism > # legal moves.
        for _ in range(10):
            player.tree_search(parallel_readouts=50)
        self.assertNoPendingVirtualLosses(player.root)

    def test_long_game_tree_search(self):
        player = MCTSPlayer(DummyNet())
        endgame = go.Position(
            board=TT_FTW_BOARD,
            n=flags.FLAGS.max_game_length - 2,
            komi=2.5,
            ko=None,
            recent=(go.PlayerMove(go.BLACK, (0, 1)),
                    go.PlayerMove(go.WHITE, (0, 8))),
            to_play=go.BLACK
        )
        player.initialize_game(endgame)

        # Test that MCTS can deduce that B wins because of TT-scoring
        # triggered by move limit.
        for _ in range(10):
            player.tree_search(parallel_readouts=8)
        self.assertNoPendingVirtualLosses(player.root)
        self.assertGreater(player.root.Q, 0)

    def test_cold_start_parallel_tree_search(self):
        # Test that parallel tree search doesn't trip on an empty tree
        player = MCTSPlayer(DummyNet(fake_value=0.17))
        player.initialize_game()
        self.assertEqual(0, player.root.N)
        self.assertFalse(player.root.is_expanded)
        leaves = player.tree_search(parallel_readouts=4)
        self.assertEqual(4, len(leaves))
        self.assertEqual(player.root, leaves[0])

        self.assertNoPendingVirtualLosses(player.root)
        # Even though the root gets selected 4 times by tree search, its
        # final visit count should just be 1.
        self.assertEqual(1, player.root.N)
        # 0.085 = average(0, 0.17), since 0 is the prior on the root.
        self.assertAlmostEqual(0.085, player.root.Q)

    def test_tree_search_failsafe(self):
        # Test that the failsafe works correctly. It can trigger if the MCTS
        # repeatedly visits a finished game state.
        probs = np.array([.001] * (go.N * go.N + 1))
        probs[-1] = 1  # Make the dummy net always want to pass
        player = MCTSPlayer(DummyNet(fake_priors=probs))
        pass_position = go.Position().pass_move()
        player.initialize_game(pass_position)
        player.tree_search(parallel_readouts=8)
        self.assertNoPendingVirtualLosses(player.root)

    def test_only_check_game_end_once(self):
        # When presented with a situation where the last move was a pass,
        # and we have to decide whether to pass, it should be the first thing
        # we check, but not more than that.

        white_passed_pos = (go.Position()
            .play_move((3, 3))  # b plays
            .play_move((3, 4))  # w plays
            .play_move((4, 3))  # b plays
            .pass_move())  # w passes - if B passes too, B would lose by komi.

        player = initialize_basic_player(white_passed_pos)
        # explore a child - should be a pass move.
        player.tree_search()
        pass_move = go.N * go.N
        self.assertEqual(1, player.root.children[pass_move].N)
        self.assertEqual(1, player.root.child_N[pass_move])
        player.tree_search()
        # check that we didn't visit the pass node any more times.
        self.assertEqual(player.root.child_N[pass_move], 1)

    def test_extract_data_normal_end(self):
        player = initialize_basic_player()
        player.tree_search()
        player.play_move(None)
        player.tree_search()
        player.play_move(None)
        self.assertTrue(player.root.is_done())
        player.set_result(player.root.position.result(), was_resign=False)

        data = list(player.extract_data())
        self.assertEqual(2, len(data))
        position, _, result = data[0]
        # White wins by komi
        self.assertEqual(go.WHITE, result)
        self.assertEqual("W+{}".format(player.root.position.komi),
                         player.result_string)

    def test_extract_data_resign_end(self):
        player = initialize_basic_player()
        player.tree_search()
        player.play_move((0, 0))
        player.tree_search()
        player.play_move(None)
        player.tree_search()
        # Black is winning on the board
        self.assertEqual(go.BLACK, player.root.position.result())
        # But if Black resigns
        player.set_result(go.WHITE, was_resign=True)

        data = list(player.extract_data())
        position, _, result = data[0]
        # Result should say White is the winner
        self.assertEqual(go.WHITE, result)
        self.assertEqual("W+R", player.result_string)
