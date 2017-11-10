import numpy as np
import copy
import sys
import go
import random
import utils
import math

# All terminology here (Q, U, N, p_UCT) uses the same notation as in the
# AlphaGo paper.
# Exploration constant
c_PUCT = 5
# Dirichlet noise, as a function of go.N
D_NOISE_ALPHA = lambda: 0.03 * 19 / go.N

class MCTSNode():
    '''A node of a MCTS search tree.

    A node knows how to compute the action scores of all of its children,
    so that a decision can be made about which move to explore next. Upon
    selecting a move, the children dictionary is updated with a new node.
    '''
    def __init__(self, position, fmove=None, parent=None):
        self.parent = parent # pointer to another MCTSNode
        self.fmove = fmove # move that led to this position, as flattened coords
        self.position = position
        # N and Q are duplicated at a node, and in the parent's child_N/Q.
        self.N = 0 # number of times node is visited
        self.Q = 0 # value estimate
        # duplication allows vectorized computation of action score.
        self.child_N = np.zeros([go.N * go.N + 1], dtype=np.float32)
        self.child_Q = np.zeros([go.N * go.N + 1], dtype=np.float32)
        self.child_prior = np.zeros([go.N * go.N + 1], dtype=np.float32)
        self.children = {} # map of flattened moves to resulting MCTSNode

    def __repr__(self):
        return "<MCTSNode move=%s, N=%s, to_play=%s>" % (
            self.position.recent[-1:], np.sum(self.child_N), self.position.to_play)

    @property
    def child_action_score(self):
        return self.child_Q + self.position.to_play * self.child_U

    @property
    def child_U(self):
        return (c_PUCT * math.sqrt(max(1, self.N)) *
            self.child_prior / (1 + self.child_N))

    @property
    def Q_perspective(self):
        "Return value of position, from perspective of player to play."
        return self.Q * self.position.to_play

    def select_leaf(self):
        current = self
        while True:
            # if a node has never been evaluated, we have no basis to select a child.
            # this conveniently handles the root-node base case, too.
            if current.N == 0:
                return current
            if current.position.is_game_over():
                # do not attempt to explore children of a finished game position
                return current
            possible_choices = current.child_action_score
            if self.position.n < go.N * 8:
                # Exclude passing from consideration at the start of game
                possible_choices = possible_choices[:-1]
            decide_func = np.argmax if current.position.to_play == go.BLACK else np.argmin
            best_move = decide_func(possible_choices)
            if best_move in current.children:
                current = current.children[best_move]
            else:
                # Reached a leaf node.
                return current.add_child(best_move)

    def add_child(self, fcoord):
        if fcoord not in self.children:
            new_position = self.position.play_move(utils.unflatten_coords(fcoord))
            self.children[fcoord] = MCTSNode(new_position, fcoord, self)
        return self.children[fcoord]

    def incorporate_results(self, move_probabilities, value, up_to=None):
        assert move_probabilities.shape == (go.N * go.N + 1,)
        # if game is over, override the value estimate with the true score
        if self.position.is_game_over():
            value = 1 if self.position.score() > 0 else -1
        # heavily downweight illegal moves so they never pop up.
        illegal_moves = 1 - self.position.all_legal_moves()
        self.child_prior = move_probabilities - illegal_moves * 10
        self.backup_value(value, up_to=up_to)

    def backup_value(self, value, up_to=None):
        """Propagates a value estimation up to the root node.

        Args:
            value: the value to be propagated (1 = black wins, -1 = white wins)
            up_to: the node to propagate until. If not set, unnecessary
                computation may be done to propagate back to the start of game.
        """
        self.N += 1
        Q, N = self.Q, self.N
        # Incrementally calculate Q = running average of all descendant Qs, 
        # given the newest value and the previous averaged N-1 values.
        updated_Q = Q + (value - Q) / N
        self.Q = updated_Q
        if self.parent is None or self is up_to:
            return
        self.parent.child_N[self.fmove] = N
        self.parent.child_Q[self.fmove] = updated_Q
        self.parent.backup_value(value, up_to=up_to)

    def inject_noise(self):
        dirch = np.random.dirichlet([D_NOISE_ALPHA()] * ((go.N * go.N) + 1))
        new_prior = self.child_prior * 0.75 + dirch * 0.25
        self.incorporate_results(new_prior, 0, up_to=self)

    def children_as_pi(self, stretch=False):
        probs = self.child_N
        if stretch:
            probs = probs ** 8
        return probs / np.sum(probs)

    def print_stats(self, target=sys.stdout):
        sort_order = list(range(go.N * go.N + 1))
        sort_order.sort(key=lambda i: self.child_N[i], reverse=True)
        # Dump out some statistics
        print("To play: ", self.position.to_play, file=target)
        print("== Top N:   Sc,    Q,    U,    P,    N == ", file=sys.stderr)
        print("\n".join(["{!s:9}: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {}".format(
                utils.to_human_coord(utils.unflatten_coords(key)),
                self.child_action_score[key],
                self.child_Q[key],
                self.child_U[key],
                self.child_prior[key],
                self.child_N[key])
                for key in sort_order if self.child_N[key] > 0]), file=target)
