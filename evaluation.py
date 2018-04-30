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

"""Evalation plays games between two neural nets."""

import os
import time
from absl import flags
from tensorflow import gfile

from gtp_wrapper import MCTSPlayer
import sgf_wrapper


def play_match(black_net, white_net, games, sgf_dir, verbosity):
    """Plays matches between two neural nets.

    black_net: Instance of minigo.DualNetwork, a wrapper around a tensorflow
        convolutional network.
    white_net: Instance of the minigo.DualNetwork.
    games: number of games to play. We play all the games at the same time.
    sgf_dir: directory to write the sgf results.
    """
    readouts = flags.FLAGS.num_readouts  # Flag defined in strategies.py

    # For n games, we create lists of n black and n white players
    black = MCTSPlayer(
        black_net, verbosity=verbosity, two_player_mode=True)
    white = MCTSPlayer(
        white_net, verbosity=verbosity, two_player_mode=True)

    black_name = os.path.basename(black_net.save_file)
    white_name = os.path.basename(white_net.save_file)

    for i in range(games):
        num_move = 0  # The move number of the current game

        black.initialize_game()
        white.initialize_game()

        while True:
            start = time.time()
            active = white if num_move % 2 else black
            inactive = black if num_move % 2 else white

            current_readouts = active.root.N
            while active.root.N < current_readouts + readouts:
                active.tree_search()

            # print some stats on the search
            if verbosity >= 3:
                print(active.root.position)

            # First, check the roots for hopeless games.
            if active.should_resign():  # Force resign
                active.set_result(-1 *
                                  active.root.position.to_play, was_resign=True)
                inactive.set_result(
                    active.root.position.to_play, was_resign=True)

            if active.is_done():
                fname = "{:d}-{:s}-vs-{:s}-{:d}.sgf".format(int(time.time()),
                                                            white_name, black_name, i)
                with gfile.GFile(os.path.join(sgf_dir, fname), 'w') as _file:
                    sgfstr = sgf_wrapper.make_sgf(active.position.recent,
                                                  active.result_string, black_name=black_name,
                                                  white_name=white_name)
                    _file.write(sgfstr)
                print("Finished game", i, active.result_string)
                break

            move = active.pick_move()
            active.play_move(move)
            inactive.play_move(move)

            dur = time.time() - start
            num_move += 1

            if (verbosity > 1) or (verbosity == 1 and num_move % 10 == 9):
                timeper = (dur / readouts) * 100.0
                print(active.root.position)
                print("%d: %d readouts, %.3f s/100. (%.2f sec)" % (num_move,
                                                                   readouts,
                                                                   timeper,
                                                                   dur))
