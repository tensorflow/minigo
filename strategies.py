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

import os
import random
import time

from absl import flags

import coords
import go
import mcts
import sgf_wrapper
from utils import dbg
from player_interface import MCTSPlayerInterface


flags.DEFINE_integer('softpick_move_cutoff', (go.N * go.N // 12) // 2 * 2,
                     'The move number (<) up to which moves are softpicked from MCTS visits.')
# Ensure that both white and black have an equal number of softpicked moves.
flags.register_validator('softpick_move_cutoff', lambda x: x % 2 == 0)

flags.DEFINE_float('resign_threshold', -0.9,
                   'The post-search Q evaluation at which resign should happen.'
                   'A threshold of -1 implies resign is disabled.')
flags.register_validator('resign_threshold', lambda x: -1 <= x < 0)

flags.DEFINE_integer('num_readouts', 800 if go.N == 19 else 200,
                     'Number of searches to add to the MCTS search tree before playing a move.')
flags.register_validator('num_readouts', lambda x: x > 0)

flags.DEFINE_integer('parallel_readouts', 8,
                     'Number of searches to execute in parallel. This is also the batch size'
                     'for neural network evaluation.')

# this should be called "verbosity" but flag name conflicts with absl.logging.
# Should fix this by overhauling this logging system with appropriate logging.info/debug.
flags.DEFINE_integer('verbose', 1, 'How much debug info to print.')

FLAGS = flags.FLAGS


def time_recommendation(move_num, seconds_per_move=5, time_limit=15 * 60,
                        decay_factor=0.98):
    '''Given the current move number and the 'desired' seconds per move, return
    how much time should actually be used. This is intended specifically for
    CGOS time controls, which has an absolute 15-minute time limit.

    The strategy is to spend the maximum possible moves using seconds_per_move,
    and then switch to an exponentially decaying time usage, calibrated so that
    we have enough time for an infinite number of moves.'''

    # Divide by two since you only play half the moves in a game.
    player_move_num = move_num / 2

    # Sum of geometric series maxes out at endgame_time seconds.
    endgame_time = seconds_per_move / (1 - decay_factor)

    if endgame_time > time_limit:
        # There is so little main time that we're already in 'endgame' mode.
        base_time = time_limit * (1 - decay_factor)
        core_moves = 0
    else:
        # Leave over endgame_time seconds for the end, and play at
        # seconds_per_move for as long as possible.
        base_time = seconds_per_move
        core_moves = (time_limit - endgame_time) / seconds_per_move

    return base_time * decay_factor ** max(player_move_num - core_moves, 0)


class MCTSPlayer(MCTSPlayerInterface):
    def __init__(self, network, seconds_per_move=5, num_readouts=0,
                 resign_threshold=None, two_player_mode=False,
                 timed_match=False):
        self.network = network
        self.seconds_per_move = seconds_per_move
        self.num_readouts = num_readouts or FLAGS.num_readouts
        self.verbosity = FLAGS.verbose
        self.two_player_mode = two_player_mode
        if two_player_mode:
            self.temp_threshold = -1
        else:
            self.temp_threshold = FLAGS.softpick_move_cutoff

        self.initialize_game()
        self.root = None
        self.resign_threshold = resign_threshold or FLAGS.resign_threshold
        self.timed_match = timed_match
        assert (self.timed_match and self.seconds_per_move >
                0) or self.num_readouts > 0
        super().__init__()

    def get_position(self):
        return self.root.position if self.root else None

    def get_root(self):
        return self.root

    def get_result_string(self):
        return self.result_string

    def initialize_game(self, position=None):
        if position is None:
            position = go.Position()
        self.root = mcts.MCTSNode(position)
        self.result = 0
        self.result_string = None
        self.comments = []
        self.searches_pi = []

    def suggest_move(self, position):
        ''' Used for playing a single game.
        For parallel play, use initialize_move, select_leaf,
        incorporate_results, and pick_move
        '''
        start = time.time()

        if self.timed_match:
            while time.time() - start < self.seconds_per_move:
                self.tree_search()
        else:
            current_readouts = self.root.N
            while self.root.N < current_readouts + self.num_readouts:
                self.tree_search()
            if self.verbosity > 0:
                dbg("%d: Searched %d times in %.2f seconds\n\n" % (
                    position.n, self.num_readouts, time.time() - start))

        # print some stats on moves considered.
        if self.verbosity > 2:
            dbg(self.root.describe())
            dbg('\n\n')
        if self.verbosity > 3:
            dbg(self.root.position)

        return self.pick_move()

    def play_move(self, c):
        '''
        Notable side effects:
          - finalizes the probability distribution according to
          this roots visit counts into the class' running tally, `searches_pi`
          - Makes the node associated with this move the root, for future
            `inject_noise` calls.
        '''
        if not self.two_player_mode:
            self.searches_pi.append(self.root.children_as_pi(
                self.root.position.n < self.temp_threshold))
        self.comments.append(self.root.describe())
        try:
            self.root = self.root.maybe_add_child(coords.to_flat(c))
        except go.IllegalMove:
            dbg("Illegal move")
            if not self.two_player_mode:
                self.searches_pi.pop()
            self.comments.pop()
            raise

        self.position = self.root.position  # for showboard
        del self.root.parent.children
        return True  # GTP requires positive result.

    def pick_move(self):
        '''Picks a move to play, based on MCTS readout statistics.

        Highest N is most robust indicator. In the early stage of the game, pick
        a move weighted by visit count; later on, pick the absolute max.'''
        if self.root.position.n >= self.temp_threshold:
            fcoord = self.root.best_child()
        else:
            cdf = self.root.children_as_pi(squash=True).cumsum()
            cdf /= cdf[-2]  # Prevents passing via softpick.
            selection = random.random()
            fcoord = cdf.searchsorted(selection)
            assert self.root.child_N[fcoord] != 0
        return coords.from_flat(fcoord)

    def tree_search(self, parallel_readouts=None):
        if parallel_readouts is None:
            parallel_readouts = min(FLAGS.parallel_readouts, self.num_readouts)
        leaves = []
        failsafe = 0
        while len(leaves) < parallel_readouts and failsafe < parallel_readouts * 2:
            failsafe += 1
            leaf = self.root.select_leaf()
            if self.verbosity >= 4:
                dbg(self.show_path_to_root(leaf))
            # if game is over, override the value estimate with the true score
            if leaf.is_done():
                value = 1 if leaf.position.score() > 0 else -1
                leaf.backup_value(value, up_to=self.root)
                continue
            leaf.add_virtual_loss(up_to=self.root)
            leaves.append(leaf)
        if leaves:
            move_probs, values = self.network.run_many(
                [leaf.position for leaf in leaves])
            for leaf, move_prob, value in zip(leaves, move_probs, values):
                leaf.revert_virtual_loss(up_to=self.root)
                leaf.incorporate_results(move_prob, value, up_to=self.root)
        return leaves

    def show_path_to_root(self, node):
        pos = node.position
        diff = node.position.n - self.root.position.n
        if len(pos.recent) == 0:
            return

        def fmt(move):
            return "{}-{}".format('b' if move.color == go.BLACK else 'w',
                                  coords.to_gtp(move.move))

        path = " ".join(fmt(move) for move in pos.recent[-diff:])
        if node.position.n >= FLAGS.max_game_length:
            path += " (depth cutoff reached) %0.1f" % node.position.score()
        elif node.position.is_game_over():
            path += " (game over) %0.1f" % node.position.score()
        return path

    def is_done(self):
        return self.result != 0 or self.root.is_done()

    def should_resign(self):
        '''Returns true if the player resigned.  No further moves should be played'''
        return self.root.Q_perspective < self.resign_threshold

    def set_result(self, winner, was_resign):
        self.result = winner
        if was_resign:
            string = "B+R" if winner == go.BLACK else "W+R"
        else:
            string = self.root.position.result_string()
        self.result_string = string

    def to_sgf(self, use_comments=True):
        assert self.result_string is not None
        pos = self.root.position
        if use_comments:
            comments = self.comments or ['No comments.']
            comments[0] = ("Resign Threshold: %0.3f\n" %
                           self.resign_threshold) + comments[0]
        else:
            comments = []
        return sgf_wrapper.make_sgf(pos.recent, self.result_string,
                                    white_name=os.path.basename(
                                        self.network.save_file) or "Unknown",
                                    black_name=os.path.basename(
                                        self.network.save_file) or "Unknown",
                                    comments=comments)

    def extract_data(self):
        assert len(self.searches_pi) == self.root.position.n
        assert self.result != 0
        for pwc, pi in zip(go.replay_position(self.root.position, self.result),
                           self.searches_pi):
            yield pwc.position, pi, pwc.result

    def get_num_readouts(self):
        return self.num_readouts

    def set_num_readouts(self, readouts):
        self.num_readouts = readouts


class CGOSPlayer(MCTSPlayer):
    def suggest_move(self, position):
        self.seconds_per_move = time_recommendation(position.n)
        return super().suggest_move(position)
