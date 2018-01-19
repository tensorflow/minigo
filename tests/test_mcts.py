import copy
import unittest
import numpy as np

import go
from go import Position
from test_utils import load_board, GoPositionTestCase, MCTSTestMixin
from coords import parse_kgs_coords
from coords import kgs_to_flat
import coords

from mcts import MCTSNode

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

TEST_POSITION = go.Position(
    board=ALMOST_DONE_BOARD,
    n=75,
    komi=2.5,
    caps=(1,4),
    ko=None,
    recent=(go.PlayerMove(go.BLACK, (0, 1)),
            go.PlayerMove(go.WHITE, (0, 8))),
    to_play=go.BLACK
)

SEND_TWO_RETURN_ONE = go.Position(
    board=ALMOST_DONE_BOARD,
    n=75,
    komi=0.5,
    caps=(0,0),
    ko=None,
    recent=(go.PlayerMove(go.BLACK, (0, 1)),
            go.PlayerMove(go.WHITE, (0, 8)),
            go.PlayerMove(go.BLACK, (1, 0))),
    to_play=go.WHITE
)


class TestMctsNodes(GoPositionTestCase, MCTSTestMixin):
    def test_action_flipping(self):
        np.random.seed(1)
        probs = np.array([.02] * (go.N * go.N + 1))
        probs = probs + np.random.random([go.N * go.N + 1]) * 0.001
        black_root = MCTSNode(go.Position())
        white_root = MCTSNode(go.Position(to_play=go.WHITE))
        black_root.select_leaf().incorporate_results(probs, 0, black_root)
        white_root.select_leaf().incorporate_results(probs, 0, white_root)
        # No matter who is to play, when we know nothing else, the priors
        # should be respected, and the same move should be picked
        black_leaf = black_root.select_leaf()
        white_leaf = white_root.select_leaf()
        self.assertEqual(black_leaf.fmove, white_leaf.fmove)
        self.assertEqualNPArray(black_root.child_action_score, white_root.child_action_score)

    def test_select_leaf(self):
        probs = np.array([.02] * (go.N * go.N + 1))
        probs[kgs_to_flat('D9')] = 0.4
        root = MCTSNode(SEND_TWO_RETURN_ONE)
        root.select_leaf().incorporate_results(probs, 0, root)

        self.assertEqual(root.position.to_play, go.WHITE)
        self.assertEqual(root.select_leaf(), root.children[kgs_to_flat('D9')])

    def test_backup_incorporate_results(self):
        probs = np.array([.02] * (go.N * go.N + 1))
        root = MCTSNode(SEND_TWO_RETURN_ONE)
        root.select_leaf().incorporate_results(probs, 0, root)


        leaf = root.select_leaf()
        leaf.incorporate_results(probs, -1, root) # white wins!

        # Root was visited twice: first at the root, then at this child.
        self.assertEqual(root.N, 2)
        # Root has 0 as a prior and two visits with value 0, -1
        self.assertAlmostEqual(root.Q, -1/3) # average of 0, 0, -1
        # Leaf should have one visit
        self.assertEqual(root.child_N[leaf.fmove], 1)
        self.assertEqual(leaf.N, 1)
        # And that leaf's value had its parent's Q (0) as a prior, so the Q
        # should now be the average of 0, -1
        self.assertAlmostEqual(root.child_Q[leaf.fmove], -0.5)
        self.assertAlmostEqual(leaf.Q, -0.5)

        # We're assuming that select_leaf() returns a leaf like:
        #   root
        #     \
        #     leaf
        #       \
        #       leaf2
        # which happens in this test because root is W to play and leaf was a W win.
        self.assertEqual(root.position.to_play, go.WHITE)
        leaf2 = root.select_leaf()
        leaf2.incorporate_results(probs, -0.2, root) # another white semi-win
        self.assertEqual(root.N, 3)
        # average of 0, 0, -1, -0.2
        self.assertAlmostEqual(root.Q, -0.3)

        self.assertEqual(leaf.N, 2)
        self.assertEqual(leaf2.N, 1)
        # average of 0, -1, -0.2
        self.assertAlmostEqual(leaf.Q, -0.4)
        self.assertAlmostEqual(root.child_Q[leaf.fmove], -0.4)
        # average of -1, -0.2
        self.assertAlmostEqual(leaf.child_Q[leaf2.fmove], -0.6)
        self.assertAlmostEqual(leaf2.Q, -0.6)

    def test_do_not_explore_past_finish(self):
        probs = np.array([0.02] * (go.N * go.N + 1), dtype=np.float32)
        root = MCTSNode(go.Position())
        root.select_leaf().incorporate_results(probs, 0, root)
        first_pass = root.add_child(coords.flatten_coords(None))
        first_pass.incorporate_results(probs, 0, root)
        second_pass = first_pass.add_child(coords.flatten_coords(None))
        with self.assertRaises(AssertionError):
            second_pass.incorporate_results(probs, 0, root)
        node_to_explore = second_pass.select_leaf()
        # should just stop exploring at the end position.
        self.assertEqual(node_to_explore, second_pass)

    def test_add_child(self):
        root = MCTSNode(go.Position())
        child = root.add_child(17)
        self.assertIn(17, root.children)
        self.assertEqual(child.parent, root)
        self.assertEqual(child.fmove, 17)

    def test_add_child_idempotency(self):
        root = MCTSNode(go.Position())
        child = root.add_child(17)
        current_children = copy.copy(root.children)
        child2 = root.add_child(17)
        self.assertEqual(child, child2)
        self.assertEqual(current_children, root.children)

    def never_select_illegal_moves(self):
        probs = np.array([0.02] * (go.N * go.N + 1))
        # let's say the NN were to accidentally put a high weight on an illegal move
        probs[1] = 0.99
        root = MCTSNode(SEND_TWO_RETURN_ONE)
        root.incorporate_results(probs, 0, root)
        # this should not throw an error...
        leaf = root.select_leaf()
        # the returned leaf should not be the illegal move
        self.assertNotEqual(leaf.fmove, 1)

        # and even after injecting noise, we should still not select an illegal move
        for i in range(10):
            root.inject_noise()
            leaf = root.select_leaf()
            self.assertNotEqual(leaf.fmove, 1)

