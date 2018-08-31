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

import copy
import math
import numpy as np

import coords
import go
import mcts
from tests import test_utils

from absl import flags

FLAGS = flags.FLAGS


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

TEST_POSITION = go.Position(
    board=ALMOST_DONE_BOARD,
    n=105,
    komi=2.5,
    caps=(1, 4),
    ko=None,
    recent=(go.PlayerMove(go.BLACK, (0, 1)),
            go.PlayerMove(go.WHITE, (0, 8))),
    to_play=go.BLACK
)

SEND_TWO_RETURN_ONE = go.Position(
    board=ALMOST_DONE_BOARD,
    n=75,
    komi=0.5,
    caps=(0, 0),
    ko=None,
    recent=(go.PlayerMove(go.BLACK, (0, 1)),
            go.PlayerMove(go.WHITE, (0, 8)),
            go.PlayerMove(go.BLACK, (1, 0))),
    to_play=go.WHITE
)


class TestMctsNodes(test_utils.MiniGoUnitTest):
    def test_upper_bound_confidence(self):
        probs = np.array([.02] * (go.N * go.N + 1))
        root = mcts.MCTSNode(go.Position())
        leaf = root.select_leaf()
        self.assertEqual(root, leaf)
        leaf.incorporate_results(probs, 0.5, root)

        # 0.02 are normalized to 1/82
        self.assertAlmostEqual(root.child_prior[0], 1/82)
        self.assertAlmostEqual(root.child_prior[1], 1/82)
        puct_policy = FLAGS.c_puct * 1/82
        self.assertEqual(root.N, 1)
        self.assertAlmostEqual(
            root.child_U[0], puct_policy * math.sqrt(1) / (1 + 0))

        leaf = root.select_leaf()
        self.assertNotEqual(root, leaf)

        # With the first child expanded.
        self.assertEqual(root.N, 1)
        self.assertAlmostEqual(
            root.child_U[0], puct_policy * math.sqrt(1) / (1 + 0))
        self.assertAlmostEqual(
            root.child_U[1], puct_policy * math.sqrt(1) / (1 + 0))

        leaf.add_virtual_loss(up_to=root)
        leaf2 = root.select_leaf()

        self.assertNotIn(leaf2, (root, leaf))

        leaf.revert_virtual_loss(up_to=root)
        leaf.incorporate_results(probs, 0.3, root)
        leaf2.incorporate_results(probs, 0.3, root)

        # With the 2nd child expanded.
        self.assertEqual(root.N, 3)
        self.assertAlmostEqual(
            root.child_U[0], puct_policy * math.sqrt(2) / (1 + 1))
        self.assertAlmostEqual(
            root.child_U[1], puct_policy * math.sqrt(2) / (1 + 1))
        self.assertAlmostEqual(
            root.child_U[2], puct_policy * math.sqrt(2) / (1 + 0))

    def test_action_flipping(self):
        np.random.seed(1)
        probs = np.array([.02] * (go.N * go.N + 1))
        probs = probs + np.random.random([go.N * go.N + 1]) * 0.001
        black_root = mcts.MCTSNode(go.Position())
        white_root = mcts.MCTSNode(go.Position(to_play=go.WHITE))
        black_root.select_leaf().incorporate_results(probs, 0, black_root)
        white_root.select_leaf().incorporate_results(probs, 0, white_root)
        # No matter who is to play, when we know nothing else, the priors
        # should be respected, and the same move should be picked
        black_leaf = black_root.select_leaf()
        white_leaf = white_root.select_leaf()
        self.assertEqual(black_leaf.fmove, white_leaf.fmove)
        self.assertEqualNPArray(
            black_root.child_action_score, white_root.child_action_score)

    def test_select_leaf(self):
        flattened = coords.to_flat(coords.from_kgs('D9'))
        probs = np.array([.02] * (go.N * go.N + 1))
        probs[flattened] = 0.4
        root = mcts.MCTSNode(SEND_TWO_RETURN_ONE)
        root.select_leaf().incorporate_results(probs, 0, root)

        self.assertEqual(root.position.to_play, go.WHITE)
        self.assertEqual(root.select_leaf(), root.children[flattened])

    def test_backup_incorporate_results(self):
        probs = np.array([.02] * (go.N * go.N + 1))
        root = mcts.MCTSNode(SEND_TWO_RETURN_ONE)
        root.select_leaf().incorporate_results(probs, 0, root)

        leaf = root.select_leaf()
        leaf.incorporate_results(probs, -1, root)  # white wins!

        # Root was visited twice: first at the root, then at this child.
        self.assertEqual(root.N, 2)
        # Root has 0 as a prior and two visits with value 0, -1
        self.assertAlmostEqual(-1 / 3, root.Q)  # average of 0, 0, -1
        # Leaf should have one visit
        self.assertEqual(1, root.child_N[leaf.fmove])
        self.assertEqual(1, leaf.N)
        # And that leaf's value had its parent's Q (0) as a prior, so the Q
        # should now be the average of 0, -1
        self.assertAlmostEqual(-0.5, root.child_Q[leaf.fmove])
        self.assertAlmostEqual(-0.5, leaf.Q)

        # We're assuming that select_leaf() returns a leaf like:
        #   root
        #     \
        #     leaf
        #       \
        #       leaf2
        # which happens in this test because root is W to play and leaf was a W win.
        self.assertEqual(go.WHITE, root.position.to_play)
        leaf2 = root.select_leaf()
        leaf2.incorporate_results(probs, -0.2, root)  # another white semi-win
        self.assertEqual(3, root.N)
        # average of 0, 0, -1, -0.2
        self.assertAlmostEqual(-0.3, root.Q)

        self.assertEqual(2, leaf.N)
        self.assertEqual(1, leaf2.N)
        # average of 0, -1, -0.2
        self.assertAlmostEqual(root.child_Q[leaf.fmove], leaf.Q)
        self.assertAlmostEqual(-0.4, leaf.Q)
        # average of -1, -0.2
        self.assertAlmostEqual(-0.6, leaf.child_Q[leaf2.fmove])
        self.assertAlmostEqual(-0.6, leaf2.Q)

    def test_do_not_explore_past_finish(self):
        probs = np.array([0.02] * (go.N * go.N + 1), dtype=np.float32)
        root = mcts.MCTSNode(go.Position())
        root.select_leaf().incorporate_results(probs, 0, root)
        first_pass = root.maybe_add_child(coords.to_flat(None))
        first_pass.incorporate_results(probs, 0, root)
        second_pass = first_pass.maybe_add_child(coords.to_flat(None))
        with self.assertRaises(AssertionError):
            second_pass.incorporate_results(probs, 0, root)
        node_to_explore = second_pass.select_leaf()
        # should just stop exploring at the end position.
        self.assertEqual(second_pass, node_to_explore)

    def test_add_child(self):
        root = mcts.MCTSNode(go.Position())
        child = root.maybe_add_child(17)
        self.assertIn(17, root.children)
        self.assertEqual(root, child.parent)
        self.assertEqual(17, child.fmove)

    def test_add_child_idempotency(self):
        root = mcts.MCTSNode(go.Position())
        child = root.maybe_add_child(17)
        current_children = copy.copy(root.children)
        child2 = root.maybe_add_child(17)
        self.assertEqual(child, child2)
        self.assertEqual(current_children, root.children)

    def test_never_select_illegal_moves(self):
        probs = np.array([0.02] * (go.N * go.N + 1))
        # let's say the NN were to accidentally put a high weight on an illegal move
        probs[1] = 0.99
        root = mcts.MCTSNode(SEND_TWO_RETURN_ONE)
        root.incorporate_results(probs, 0, root)
        # and let's say the root were visited a lot of times, which pumps up the
        # action score for unvisited moves...
        root.N = 100000
        root.child_N[root.position.all_legal_moves()] = 10000
        # this should not throw an error...
        leaf = root.select_leaf()
        # the returned leaf should not be the illegal move
        self.assertNotEqual(1, leaf.fmove)

        # and even after injecting noise, we should still not select an illegal move
        for i in range(10):
            root.inject_noise()
            leaf = root.select_leaf()
            self.assertNotEqual(1, leaf.fmove)

    def test_dont_pick_unexpanded_child(self):
        probs = np.array([0.001] * (go.N * go.N + 1))
        # make one move really likely so that tree search goes down that path twice
        # even with a virtual loss
        probs[17] = 0.999
        root = mcts.MCTSNode(go.Position())
        root.incorporate_results(probs, 0, root)
        root.N = 5
        leaf1 = root.select_leaf()
        self.assertEqual(17, leaf1.fmove)
        leaf1.add_virtual_loss(up_to=root)
        # the second select_leaf pick should return the same thing, since the child
        # hasn't yet been sent to neural net for eval + result incorporation
        leaf2 = root.select_leaf()
        self.assertIs(leaf1, leaf2)

    def test_normalize_policy(self):
        # sum of probs > 1.0
        probs = np.array([2.0] * (go.N * go.N + 1))

        root = mcts.MCTSNode(TEST_POSITION)
        root.incorporate_results(probs, 0, root)
        root.N = 0

        # Policy sums to 1.0, only legal moves have non-zero values.
        self.assertAlmostEqual(1.0, sum(root.child_prior))
        self.assertEqual(6, np.count_nonzero(root.child_prior))
        self.assertEqual(0, sum(root.child_prior * root.illegal_moves))

    def test_inject_noise_only_legal_moves(self):
        probs = np.array([0.02] * (go.N * go.N + 1))
        root = mcts.MCTSNode(TEST_POSITION)
        root.incorporate_results(probs, 0, root)
        root.N = 0

        uniform_policy = 1 / sum(root.illegal_moves == 0)
        expected_policy = uniform_policy * (1 - root.illegal_moves)

        self.assertTrue((root.child_prior == expected_policy).all())

        root.inject_noise()

        # 0.75/0.25 derived from default dirichlet_noise_weight.
        self.assertTrue((0.75 * expected_policy <= root.child_prior).all())
        self.assertTrue(
            (0.75 * expected_policy + 0.25 >= root.child_prior).all())
        # Policy sums to 1.0, only legal moves have non-zero values.
        self.assertAlmostEqual(1.0, sum(root.child_prior))
        self.assertEqual(0, sum(root.child_prior * root.illegal_moves))
