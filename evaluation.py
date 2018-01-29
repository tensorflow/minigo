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

import go
import time
import numpy as np

from dual_net import DualNetwork
from gtp_wrapper import MCTSPlayer

def play_match(black_net, white_net, games, readouts, verbosity):
    black_players = [MCTSPlayer(black_net, verbosity=verbosity, two_player_mode=True) for i in range(games)]
    white_players = [MCTSPlayer(white_net, verbosity=verbosity, two_player_mode=True) for i in range(games)]
    player_pairs = [(b,w) for b,w in zip(black_players, white_players)]

    done_pairs = []
    global_n = 0

    for p1,p2 in player_pairs:
        p1.initialize_game()
        p2.initialize_game()

    while player_pairs:
        start=time.time()

        for i in range(readouts):
            leaves = [pair[global_n % 2].root.select_leaf() for pair in player_pairs]
            probs, vals = (black_net, white_net)[global_n % 2].run_many([leaf.position for leaf in leaves])

            [leaf.incorporate_results(prob, val, up_to=pair[global_n % 2].root)
                 for pair, leaf, prob, val in zip(player_pairs, leaves, probs, vals)]

        # print some stats on the search
        if (verbosity >= 3):
            print(player_pairs[0][0].root.position)

        for black, white in player_pairs:
            active = white if global_n % 2 else black
            inactive = black if global_n % 2 else white
            # First, check the roots for hopeless games.
            if active.should_resign(): # Force resign
                continue
            move = active.pick_move()
            active.play_move(move)
            inactive.play_move(move)


        dur = time.time() - start
        global_n += 1
        if (verbosity > 1) or (verbosity == 1 and global_n % 10 == 9):
            print("%d: %d readouts, %.3f s/100. (%.2f sec)" % (global_n, 
                   readouts * len(player_pairs), dur / (readouts*len(player_pairs) / 100.0), dur))

        done_pairs.extend([p for p in player_pairs if p[0].is_done() or p[1].is_done()])
        player_pairs = [p for p in player_pairs if not (p[0].is_done() or p[1].is_done())]

    return done_pairs
