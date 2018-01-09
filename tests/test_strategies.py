import unittest
import unittest.mock as mock
import numpy as np
np.random.seed(0)

import go
from go import Position
from test_utils import load_board, GoPositionTestCase
from utils import parse_kgs_coords as pc
from utils import to_human_coord as un_pc
import utils
from mcts import MCTSNode
from strategies import MCTSPlayerMixin, time_recommendation

class DummyNet():
    def run(self, position):
        return (np.ones((go.N **2) +1) / (go.N ** 2 + 1), 0)

class TestMCTSPlayerMixin(GoPositionTestCase):
    def setUp(self):
        self.player = MCTSPlayerMixin(DummyNet())
        self.player.initialize_game()
        self.player.root.incorporate_results(
            *self.player.network.run(self.player.root.position))

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
        root = self.player.root
        root.child_N[utils.flatten_coords((2, 0))] = 10
        root.child_N[utils.flatten_coords((1, 0))] = 5
        root.child_N[utils.flatten_coords((3, 0))] = 1

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

