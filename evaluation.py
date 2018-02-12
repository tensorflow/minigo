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

import time

from gtp_wrapper import MCTSPlayer


def play_match(black_net, white_net, games, readouts, verbosity):
    """Plays matches between two neural nets.

    black_net: Instance of minigo.DualNetwork, a wrapper around a tensorflow
        convolutional network.
    white_net: Instance of the minigo.DualNetwork.
    games: number of games to play. We play all the games at the same time.
    readouts: number of readouts to perform for each step in each game.
    """

    # For n games, we create lists of n black and n white players
    black_players = [MCTSPlayer(
        black_net, verbosity=verbosity, two_player_mode=True) for i in range(games)]
    white_players = [MCTSPlayer(
        white_net, verbosity=verbosity, two_player_mode=True) for i in range(games)]

    # Each player pair represents two players that are going to play a game.
    player_pairs = [(b, w) for b, w in zip(black_players, white_players)]

    done_pairs = []

    # The number of moves that have been played
    num_moves = 0

    for black, white in player_pairs:
        black.initialize_game()
        white.initialize_game()

    # The heart of the game-playing loop. Each iteration through the while loop
    # plays one move for each player. That means we:
    #   - Do a bunch of MTCS readouts (for each active player, for each game)
    #   - Play a move (for each active player, for each game)
    #   - Remove any finished player-pairs
    while player_pairs:
        start = time.time()

        for _ in range(readouts):
            leaves = [pair[num_moves % 2].root.select_leaf()
                      for pair in player_pairs]
            probs, vals = (black_net, white_net)[num_moves % 2].run_many(
                [leaf.position for leaf in leaves])

            for pair, leaf, prob, val in zip(player_pairs, leaves, probs, vals):
                leaf.incorporate_results(
                    prob, val, up_to=pair[num_moves % 2].root)

        # print some stats on the search
        if verbosity >= 3:
            print(player_pairs[0][0].root.position)

        for black, white in player_pairs:
            active = white if num_moves % 2 else black
            inactive = black if num_moves % 2 else white
            # First, check the roots for hopeless games.
            if active.should_resign():  # Force resign
                continue
            move = active.pick_move()
            active.play_move(move)
            inactive.play_move(move)

        dur = time.time() - start
        num_moves += 1
        if (verbosity > 1) or (verbosity == 1 and num_moves % 10 == 9):
            rdcnt = readouts * len(player_pairs)
            timeper = dur / (readouts*len(player_pairs) / 100.0)
            print("%d: %d readouts, %.3f s/100. (%.2f sec)" % (num_moves,
                                                               rdcnt,
                                                               timeper,
                                                               dur))

        done_pairs.extend(
            [p for p in player_pairs if p[0].is_done() or p[1].is_done()])
        player_pairs = [p for p in player_pairs if not (
            p[0].is_done() or p[1].is_done())]

    return done_pairs
