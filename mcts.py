"""Monte Carlo Tree Search implementation.

All terminology here (Q, U, N, p_UCT) uses the same notation as in the
AlphaGo (AG) paper.
"""
import numpy as np
import copy
import sys
import random
import math

import coords
import go

# Exploration constant
c_PUCT = 1.38

# Dirichlet noise, as a function of go.N
D_NOISE_ALPHA = lambda: 0.03 * 19 / go.N

class MCTSNode(object):
    """A node of a MCTS search tree.

    A node knows how to compute the action scores of all of its children,
    so that a decision can be made about which move to explore next. Upon
    selecting a move, the children dictionary is updated with a new node.

    position: A go.Position instance
    fmove: A move (coordinate) that led to this position, a a flattened coord
            (raw number between 0-N^2, with None a pass)
    parent: A parent MCTSNode.
    """

    def __init__(self, position, initial_Q=0, fmove=None, parent=None):
        self.parent = parent # pointer to another MCTSNode
        self.fmove = fmove # move that led to this position, as flattened coords
        self.position = position
        self.is_expanded = False
        # N and Q are duplicated at a node, and in the parent's child_N/Q.
        self.N = 0 # number of times node is visited
        self.W = initial_Q # value estimate
        self.losses_applied = 0 # number of virtual losses on this node
        # duplication allows vectorized computation of action score.
        self.legal_moves = 10 * self.position.all_legal_moves()
        self.child_N = np.zeros([go.N * go.N + 1], dtype=np.float32)
        self.child_W = np.zeros([go.N * go.N + 1], dtype=np.float32)
        self.original_prior = np.zeros([go.N * go.N + 1], dtype=np.float32)
        self.prior = np.zeros([go.N * go.N + 1], dtype=np.float32)
        self.children = {} # map of flattened moves to resulting MCTSNode

    def __repr__(self):
        return "<MCTSNode move=%s, N=%s, to_play=%s>" % (
            self.position.recent[-1:], self.N, self.position.to_play)

    @property
    def child_action_score(self):
        return (self.child_Q * self.position.to_play + self.child_U) + self.legal_moves

    @property
    def child_Q(self):
        return self.child_W / (1 + self.child_N)

    @property
    def child_U(self):
        return (c_PUCT * math.sqrt(self.N) *
            self.prior / (1 + self.child_N))

    @property
    def Q(self):
        return self.W / (1 + self.N)

    @property
    def Q_perspective(self):
        "Return value of position, from perspective of player to play."
        return self.Q * self.position.to_play

    def select_leaf(self):
        current = self
        pass_move = go.N * go.N
        while True:
            current.N += 1
            # if a node has never been evaluated, we have no basis to select a child.
            if not self.is_expanded:
                break

            best_move = np.argmax(current.child_action_score)
            current.child_N[best_move] += 1
            if best_move in current.children:
                current = current.children[best_move]
            else:
                # Reached a leaf node.
                current = current.add_child(best_move)
                current.N += 1
                break
        return current

    def add_child(self, fcoord):
        """ Adds child node for fcoord if it doesn't already exist, and returns it. """
        if fcoord not in self.children:
            new_position = self.position.play_move(coords.unflatten_coords(fcoord))
            self.children[fcoord] = MCTSNode(
                new_position, initial_Q=self.child_W[fcoord], fmove=fcoord, parent=self)
        return self.children[fcoord]

    def add_virtual_loss(self, up_to):
        """Propagate a virtual loss up to the root node.

        Also increment visit counts for each node

        Args:
            up_to: The node to propagate until. (Keep track of this! You'll
                need it to reverse the virtual loss later.)
        """
        self.losses_applied += 1
        if self.parent is None or self is up_to:
            return
        self.parent.add_virtual_loss(up_to)

    def revert_virtual_loss(self, up_to):
        self.losses_applied -= 1
        if self.parent is None or self is up_to:
            return
        self.parent.revert_virtual_loss(up_to)

    def incorporate_results(self, move_probabilities, value, up_to):
        assert move_probabilities.shape == (go.N * go.N + 1,)
        # A finished game should not be going through this code path - should
        # directly call backup_value() on the result of the game.
        assert not self.position.is_game_over()
        self.is_expanded = True
        self.original_prior = self.prior = move_probabilities
        # initialize child Q as current node's value, to prevent dynamics where
        # if B is winning, then B will only ever explore 1 move, because the Q
        # estimation will be so much larger than the 0 of the other moves.
        #
        # Conversely, if W is winning, then B will explore all 362 moves before
        # continuing to explore the most favorable move. This is a waste of search.
        #
        # The value seeded here will stick around; this acts as a prior.
        self.child_W = np.ones([go.N * go.N + 1], dtype=np.float32) * value
        self.backup_value(value, up_to=up_to)

    def backup_value(self, value, up_to):
        """Propagates a value estimation up to the root node.

        Args:
            value: the value to be propagated (1 = black wins, -1 = white wins)
            up_to: the node to propagate until. Must agree with the up_to value
                used while adding virtual losses.
        """
        self.W += value
        if self.parent is None or self is up_to:
            return
        self.parent.child_W[self.fmove] += value
        self.parent.backup_value(value, up_to)

    def inject_noise(self):
        dirch = np.random.dirichlet([D_NOISE_ALPHA()] * ((go.N * go.N) + 1))
        self.prior = self.prior * 0.75 + dirch * 0.25

    def children_as_pi(self, stretch=False):
        probs = self.child_N
        if stretch:
            probs = probs ** 8
        return probs / np.sum(probs)

    def most_visited_path(self):
        node = self
        output = []
        while node.children:
            next_kid = np.argmax(node.child_N)
            node = node.children[next_kid]
            output.append("%s (%d) ==> " % (coords.to_human_coord(
                                            coords.unflatten_coords(node.fmove)),
                                            node.N))
        output.append("Q: {:.5f}\n".format(node.Q))
        return ''.join(output)

    def mvp_gg(self):
        """ Returns most visited path in go-gui VAR format e.g. 'b r3 w c17..."""
        node = self
        output = []
        while node.children and max(node.child_N) > 1:
            next_kid = np.argmax(node.child_N)
            node = node.children[next_kid]
            output.append("%s" % coords.to_human_coord(coords.unflatten_coords(node.fmove)))
        return ' '.join(output)

    def describe(self):
        sort_order = list(range(go.N * go.N + 1))
        sort_order.sort(key=lambda i: self.child_N[i], reverse=True)
        soft_n = self.child_N / sum(self.child_N)
        p_delta = soft_n - self.prior
        p_rel = p_delta / soft_n
        # Dump out some statistics
        output = []
        output.append("{q:.4f}\n".format(q=self.Q))
        output.append(self.most_visited_path())
        output.append("move:  action      Q      U      P    P-Dir    N  soft-N  p-delta  p-rel\n")
        output.append("\n".join(["{!s:6}: {: .3f}, {: .3f}, {:.3f}, {:.3f}, {:.3f}, {:4d} {:.4f} {: .5f} {: .2f}".format(
                coords.to_human_coord(coords.unflatten_coords(key)),
                self.child_action_score[key],
                self.child_Q[key],
                self.child_U[key],
                self.prior[key],
                self.original_prior[key],
                int(self.child_N[key]),
                soft_n[key],
                p_delta[key],
                p_rel[key])
                for key in sort_order if self.child_N[key] > 0][:15]))
        return ''.join(output)
