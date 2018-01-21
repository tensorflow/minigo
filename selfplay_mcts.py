import itertools
import numpy as np
import random
import time

import go
from gtp_wrapper import MCTSPlayer

SIMULTANEOUS_LEAVES = 8

def play(network, readouts, resign_threshold, verbosity=0):
    ''' Plays out a self-play match, returning
    - the final position
    - the n x 362 tensor of floats representing the mcts search probabilities
    - the n-ary tensor of floats representing the original value-net estimate
    where n is the number of moves in the game'''
    player = MCTSPlayer(network,
                        resign_threshold=resign_threshold,
                        verbosity=verbosity)
    global_n = 0

    # Disable resign in 5% of games
    if random.random() < 0.05:
        player.resign_threshold = -0.9999

    player.initialize_game()
    
    start = time.time()
    # Must run this once at the start, so that 8 child nodes actually exist
    # for parallel search to bootstrap. Subsequent moves will have tree reuse
    # which solves the cold start problem.
    first_node = player.root.select_leaf()
    prob, val = network.run(first_node.position)
    first_node.incorporate_results(prob, val, first_node)

    for i in itertools.count():
        player.root.inject_noise()
        while player.root.N < readouts:
            leaves = [player.root.select_leaf() for i in range(SIMULTANEOUS_LEAVES)]
            if verbosity > 3:
                player.show_path_to_root(leaves[0])

            probs, vals = network.run_many([leaf.position for leaf in leaves])

            for leaf, prob, val in zip(leaves, probs, vals):
                leaf.incorporate_results(prob, val, up_to=player.root)

        if (verbosity >= 3):
            print(players[0].root.position)
            print(players[0].root.describe())

        # Sets is_done to be True if player.should resign.
        if player.should_resign(): # TODO: make this less side-effecty.            
            break
        move = player.pick_move()
        player.play_move(move)
        if player.is_done():
            # TODO: actually handle the result instead of ferrying it around as a property.
            player.result = player.position.result()

        if (verbosity >= 2) or (verbosity >= 1 and i % 10 == 9):
            print("Q: {}".format(p.root.Q))
            # print Q somewhere
            if verbosity >= 3:
                print("Played >>",
                      coords.to_human_coord(coords.unflatten_coords(players[0].root.fmove)))

            dur = time.time() - start
            print("%d: %d readouts, %.3f s/100. (%.2f sec)" % (
                i, readouts, dur / readouts / 100.0, dur), flush=True)

        # TODO: break when i >= 2 * go.N * go.N (where is this being done now??...)

    return player
