import unittest
import unittest.mock as mock
import numpy as np
np.random.seed(0)

import coords
import go
from go import Position
from coords import kgs_to_flat
from test_utils import load_board, GoPositionTestCase, MCTSTestMixin
from mcts import MCTSNode
from strategies import MCTSPlayerMixin, time_recommendation

ALMOST_DONE_BOARD = load_board('''
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

SEND_TWO_RETURN_ONE = go.Position(
    board=ALMOST_DONE_BOARD,
    n=75,
    komi=2.5,
    caps=(1,4),
    ko=None,
    recent=(go.PlayerMove(go.BLACK, (0, 1)),
            go.PlayerMove(go.WHITE, (0, 8))),
    to_play=go.BLACK
)

class DummyNet():
    def __init__(self, fake_priors=None, fake_value=0):
        if fake_priors is None:
            fake_priors = np.ones((go.N **2) +1) / (go.N ** 2 + 1)
        self.fake_priors = fake_priors
        self.fake_value = fake_value

    def run(self, position):
        return self.fake_priors, self.fake_value

    def run_many(self, positions):
        if not positions:
            raise ValueError("No positions passed! (Tensorflow would have failed here.")
        return [self.fake_priors] * len(positions), [self.fake_value] * len(positions)

def initialize_basic_player():
    player = MCTSPlayerMixin(DummyNet())
    player.initialize_game()
    first_node = player.root.select_leaf()
    first_node.incorporate_results(
        *player.network.run(player.root.position), up_to=player.root)
    return player

def initialize_almost_done_player():
    probs = np.array([.001] * (go.N * go.N + 1))
    probs[2:5] = 0.2 # some legal moves along the top.
    probs[-1] = 0.2 # passing is also ok
    net = DummyNet(fake_priors=probs)
    player = MCTSPlayerMixin(net)
    # root position is white to play with no history == white passed.
    player.initialize_game(SEND_TWO_RETURN_ONE)
    return player


class TestMCTSPlayerMixin(GoPositionTestCase, MCTSTestMixin):

    def test_time_controls(self):
        secs_per_move = 5
        for time_limit in (10, 100, 1000):
            # in the worst case imaginable, let's say a game goes 1000 moves long
            move_numbers = range(0, 1000, 2)
            total_time_spent = sum(
                time_recommendation(move_num, secs_per_move, time_limit=time_limit)
                for move_num in move_numbers)
            # we should not exceed available game time
            self.assertLess(total_time_spent, time_limit)
            # but we should have used at least 95% of our time by the end.
            self.assertGreater(total_time_spent, time_limit * 0.95)

    def test_inject_noise(self):
        player = initialize_basic_player()
        sum_priors = np.sum(player.root.child_prior)
        self.assertAlmostEqual(sum_priors, 1) # dummyNet should return normalized priors.
        self.assertTrue(np.all(player.root.child_U == player.root.child_U[0]))

        player.root.inject_noise()
        new_sum_priors = np.sum(player.root.child_prior)
        # priors should still be normalized after injecting noise
        self.assertAlmostEqual(sum_priors, new_sum_priors)

        # With dirichelet noise, majority of density should be in one node.
        max_p = np.max(player.root.child_prior)
        self.assertGreater(max_p, 3/(go.N ** 2 + 1))

    def test_pick_moves(self):
        player = initialize_basic_player()
        root = player.root
        root.child_N[coords.flatten_coords((2, 0))] = 10
        root.child_N[coords.flatten_coords((1, 0))] = 5
        root.child_N[coords.flatten_coords((3, 0))] = 1

        root.position.n = go.N ** 2 #move 81, or 361, or... Endgame.


        # Assert we're picking deterministically
        self.assertTrue(root.position.n > player.temp_threshold)
        move = player.pick_move()
        self.assertEqual(move, (2, 0))

        # But if we're in the early part of the game, pick randomly
        root.position.n = 3
        self.assertFalse(player.root.position.n > player.temp_threshold)

        with mock.patch('random.random', lambda: .5):
            move = player.pick_move()
            self.assertEqual(move, (2, 0))

        with mock.patch('random.random', lambda: .99):
            move = player.pick_move()
            self.assertEqual(move, (3,0))

    def test_dont_pass_if_losing(self):
        player = initialize_almost_done_player()

        # check -- white is losing.
        self.assertEqual(player.root.position.score(), -0.5)

        for i in range(20):
            player.tree_search()
        # uncomment to debug this test
        # print(player.root.describe())

        # Search should converge on D9 as only winning move.
        best_move = np.argmax(player.root.child_N)
        self.assertEqual(best_move, kgs_to_flat('D9'))
        # D9 should have a positive value
        self.assertGreater(player.root.children[kgs_to_flat('D9')].Q, 0)
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
        self.assertEqual(player.root.position.score(), -0.5)
        # initialize the tree so that the root node has populated children.
        player.tree_search(num_parallel=1)
        # virtual losses should enable multiple searches to happen simultaneously
        # without throwing an error...
        for i in range(5):
            player.tree_search(num_parallel=4)
        # uncomment to debug this test
        # print(player.root.describe())

        # Search should converge on D9 as only winning move.
        best_move = np.argmax(player.root.child_N)
        self.assertEqual(best_move, kgs_to_flat('D9'))
        # D9 should have a positive value
        self.assertGreater(player.root.children[kgs_to_flat('D9')].Q, 0)
        self.assertGreaterEqual(player.root.N, 20)
        # passing should be ineffective.
        self.assertLess(player.root.child_Q[-1], 0)
        # no virtual losses should be pending
        self.assertNoPendingVirtualLosses(player.root)

    def test_ridiculously_parallel_tree_search(self):
        player = initialize_almost_done_player()
        # Test that an almost complete game
        # will tree search with # parallelism > # legal moves.
        for i in range(10):
            player.tree_search(num_parallel=50)
        self.assertNoPendingVirtualLosses(player.root)

    def test_cold_start_parallel_tree_search(self):
        # Test that parallel tree search doesn't trip on an empty tree
        player = MCTSPlayerMixin(DummyNet(fake_value=0.17))
        player.initialize_game()
        self.assertEqual(player.root.N, 0)
        self.assertFalse(player.root.is_expanded)
        player.tree_search(num_parallel=4)
        self.assertNoPendingVirtualLosses(player.root)
        # Even though the root gets selected 4 times by tree search, its
        # final visit count should just be 1.
        self.assertEqual(player.root.N, 1)
        # 0.085 = average(0, 0.17), since 0 is the prior on the root.
        self.assertAlmostEqual(player.root.Q, 0.085)

    def test_tree_search_failsafe(self):
        # Test that the failsafe works correctly. It can trigger if the MCTS
        # repeatedly visits a finished game state.
        probs = np.array([.001] * (go.N * go.N + 1))
        probs[-1] = 1 # Make the dummy net always want to pass
        player = MCTSPlayerMixin(DummyNet(fake_priors=probs))
        pass_position = go.Position().pass_move()
        player.initialize_game(pass_position)
        player.tree_search(num_parallel=1)
        self.assertNoPendingVirtualLosses(player.root)

    def test_only_check_game_end_once(self):
        # When presented with a situation where the last move was a pass,
        # and we have to decide whether to pass, it should be the first thing
        # we check, but not more than that.

        white_passed_pos = go.Position(
            ).play_move((3, 3) # b plays
            ).play_move((3, 4) # w plays
            ).play_move((4, 3) # b plays
            ).pass_move() # w passes - if B passes too, B would lose by komi.

        player = MCTSPlayerMixin(DummyNet(fake_priors=np.array([.001] * (go.N * go.N + 1))))
        player.initialize_game(white_passed_pos)
        # initialize the root
        player.tree_search()
        # explore a child - should be a pass move.
        player.tree_search()
        pass_move = go.N * go.N
        self.assertEqual(player.root.children[pass_move].N, 1)
        self.assertEqual(player.root.child_N[pass_move], 1)
        player.tree_search()
        # check that we didn't visit the pass node any more times.
        self.assertEqual(player.root.child_N[pass_move], 1)
