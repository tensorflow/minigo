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

class TestMCTSPlayerMixin(GoPositionTestCase, MCTSTestMixin):
    def setUp(self):
        self.player = MCTSPlayerMixin(DummyNet())
        self.player.initialize_game()
        first_node = self.player.root.select_leaf()
        first_node.add_virtual_loss(self.player.root)
        first_node.incorporate_results(
            *self.player.network.run(self.player.root.position), up_to=self.player.root)

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
        player = self.player
        sum_priors = np.sum(player.root.prior)
        self.assertAlmostEqual(sum_priors, 1) # dummyNet should return normalized priors.
        self.assertTrue(np.all(player.root.child_U == player.root.child_U[0]))

        player.root.inject_noise()
        new_sum_priors = np.sum(player.root.prior)
        # priors should still be normalized after injecting noise
        self.assertAlmostEqual(sum_priors, new_sum_priors)

        # With dirichelet noise, majority of density should be in one node.
        max_p = np.max(player.root.prior)
        self.assertGreater(max_p, 3/(go.N ** 2 + 1))

    def test_pick_moves(self):
        root = self.player.root
        root.child_N[coords.flatten_coords((2, 0))] = 10
        root.child_N[coords.flatten_coords((1, 0))] = 5
        root.child_N[coords.flatten_coords((3, 0))] = 1

        root.position.n = go.N ** 2 #move 81, or 361, or... Endgame.


        # Assert we're picking deterministically
        self.assertTrue(root.position.n > self.player.temp_threshold)
        move = self.player.pick_move()
        self.assertEqual(move, (2, 0))

        # But if we're in the early part of the game, pick randomly
        root.position.n = 3
        self.assertFalse(self.player.root.position.n > self.player.temp_threshold)

        with mock.patch('random.random', lambda: .5):
            move = self.player.pick_move()
            self.assertEqual(move, (2, 0))

        with mock.patch('random.random', lambda: .99):
            move = self.player.pick_move()
            self.assertEqual(move, (3,0))

    def test_dont_pass_if_losing(self):
        # force exploration of our known set of moves by constructing the
        # probabilities passed in when expanding nodes.
        probs = np.array([.001] * (go.N * go.N + 1))
        probs[2:5] = 0.2 # some legal moves along the top.
        probs[-1] = 0.2 # passing is also ok
        net = DummyNet(fake_priors=probs)
        player = MCTSPlayerMixin(net)
        # root position is white to play with no history == white passed.
        player.initialize_game(SEND_TWO_RETURN_ONE)

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
        self.assertEqual(player.root.N, 20)
        # passing should be ineffective.
        self.assertLess(player.root.child_Q[-1], 0)
        # no virtual losses should be pending
        self.assertNoPendingVirtualLosses(player.root)
        # uncomment to debug this test
        # print(player.root.describe())
