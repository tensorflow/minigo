import copy
import unittest
import numpy as np

import go
from go import Position
from test_utils import load_board, GoPositionTestCase
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


class TestMctsNodes(GoPositionTestCase):
    def test_action_flipping(self):
        np.random.seed(1)
        probs = np.array([.02] * (go.N * go.N + 1))
        probs = probs + np.random.random([go.N * go.N + 1]) * 0.001
        black_root = MCTSNode(go.Position())
        white_root = MCTSNode(go.Position(to_play=go.WHITE))
        black_root.incorporate_results(probs, 0)
        white_root.incorporate_results(probs, 0)
        # No matter who is to play, when we know nothing else, the priors
        # should be respected, and the same move should be picked
        black_leaf = black_root.select_leaf()
        white_leaf = white_root.select_leaf()
        self.assertEqual(black_leaf.fmove, white_leaf.fmove)
        self.assertEqualNPArray(black_root.child_action_score, -white_root.child_action_score)

    def test_select_leaf(self):
        probs = np.array([.02] * (go.N * go.N + 1))
        probs[kgs_to_flat('D9')] = 0.4
        root = MCTSNode(SEND_TWO_RETURN_ONE)
        root.incorporate_results(probs, 0)

        self.assertEqual(root.position.to_play, go.WHITE)
        self.assertEqual(root.select_leaf(), root.children[kgs_to_flat('D9')])

    def test_backup_incorporate_results(self):
        probs = np.array([.02] * (go.N * go.N + 1))
        root = MCTSNode(SEND_TWO_RETURN_ONE)
        root.incorporate_results(probs, 0)

        move = parse_kgs_coords('D9')
        fmove = coords.flatten_coords(move)
        leaf = MCTSNode(root.position.play_move(move), fmove, root)
        leaf.incorporate_results(probs, -1) # white wins!

        self.assertNotEqual(root.child_U[fmove], 0)
        self.assertEqual(root.N, 2)
        self.assertAlmostEqual(root.Q, -0.5) # average of 0, -1
        self.assertEqual(root.child_N[fmove], 1)
        self.assertEqual(leaf.N, 1)
        self.assertAlmostEqual(root.child_Q[fmove], -1)
        self.assertAlmostEqual(leaf.Q, -1)

        move2 = parse_kgs_coords('J3')
        fmove2 = coords.flatten_coords(move2)
        leaf2 = MCTSNode(root.position.play_move(move), fmove2, root)
        leaf2.incorporate_results(probs, -0.2) # another white semi-win
        self.assertEqual(root.N, 3)
        self.assertAlmostEqual(root.Q, -0.4) # average of 0, -1, -0.2
        self.assertEqual(root.child_N[fmove2], 1)
        self.assertEqual(leaf2.N, 1)
        self.assertAlmostEqual(root.child_Q[fmove2], -0.2)
        self.assertAlmostEqual(leaf2.Q, -0.2)

    def test_do_not_explore_past_finish(self):
        probs = np.array([0.02] * (go.N * go.N + 1), dtype=np.float32)
        root = MCTSNode(go.Position())
        root.incorporate_results(probs, 0)
        first_pass = root.add_child(coords.flatten_coords(None))
        first_pass.incorporate_results(probs, 0)
        second_pass = first_pass.add_child(coords.flatten_coords(None))
        second_pass.incorporate_results(probs, 0)
        self.assertEqual(second_pass.N, 1)
        self.assertTrue(second_pass.position.is_game_over())
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
        root.incorporate_results(probs, 0)
        # this should not throw an error...
        leaf = root.select_leaf()
        # the returned leaf should not be the illegal move
        self.assertNotEqual(leaf.fmove, 1)

        # and even after injecting noise, we should still not select an illegal move
        for i in range(10):
            root.inject_noise()
            leaf = root.select_leaf()
            self.assertNotEqual(leaf.fmove, 1)

    def test_dont_pass_if_losing(self):
        # force exploration of our known set of moves by constructing the
        # probabilities passed in when expanding nodes.
        probs = np.array([.001] * (go.N * go.N + 1))
        probs[2:5] = 0.2 # some legal moves along the top.
        probs[-1] = 0.2 # passing is also ok
        # root position is white to play with no history == white passed.
        root = MCTSNode(TEST_POSITION)

        # check -- white is losing.
        self.assertEqual(root.position.score(), -0.5)
        for i in range(20):
            leaf = root.select_leaf()
            leaf.incorporate_results(probs, 0)

            # uncomment to debug this test
            # pos = leaf.position
            # if len(pos.recent) == 0:
            #     continue
            # moves = list(map(coords.to_human_coord,
            #                  [move.move for move in pos.recent[2:]]))
            # print("From root: ", " <= ".join(moves))

        #Search should converge on D9 as only winning move.
        print(root.child_N)
        best_move_found = np.argmax(root.child_N)
        self.assertEqual(best_move_found, kgs_to_flat('D9'))
        # D9 should have a positive value
        self.assertGreater(root.children[kgs_to_flat('D9')].Q, 0)
        self.assertEqual(root.N, 20)
        # passing should be ineffective.
        self.assertLess(root.child_Q[-1], 0)
        # uncomment to debug this test
        # print(root.describe())
