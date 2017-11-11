import go
import utils
import time
import numpy as np

from dual_net import DualNetwork
from gtp_wrapper import MCTSPlayer

def play(network, games, readouts, verbosity=0):
    ''' Plays out a self-play match, returning
    - the final position
    - the n x 362 tensor of floats representing the mcts search probabilities
    - the n-ary tensor of floats representing the original value-net estimate
    where n is the number of moves in the game'''
    players = [MCTSPlayer(network, verbosity=verbosity) for i in range(games)]
    done_players = []
    global_n = 0

    for player in players:
        player.initialize_game()

    while players:
        start = time.time()

        for i in range(readouts):
            leaves = [player.root.select_leaf() for player in players]
            if verbosity > 3:
                players[0].show_path_to_root(leaves[0])

            probs, vals = network.run_many([leaf.position for leaf in leaves])

            [leaf.incorporate_results(prob, val, up_to=player.root)
                 for player, leaf, prob, val in zip(players, leaves, probs, vals)]
            if i == 0:
                for player in players:
                    player.root.inject_noise()


        # print some stats on the search
        if (global_n % 10) == 9 and verbosity >= 1:
            qs = [p.root.Q for p in players]
            print ("Max/min Q: %.3f / %.3f" % (max(qs), min(qs)), flush=True)
            print ("std: %.3f" % np.std(qs), flush=True)

        if (verbosity >= 3):
            players[0].root.print_stats()
            print(players[0].root.position)

        for player in players:
            # First, check the roots for hopeless games.
            if player.should_resign(): # Force resign, causes 'is_done' to be true.
                continue
            move = player.pick_move()
            player.play_move(move)
            if player.is_done(): # If 'move' was pass #2, score the board.
                player.result = 1 if player.position.score() > 0 else -1

        dur = time.time() - start
        global_n += 1
        if (verbosity > 1) or (verbosity == 1 and global_n % 10 == 9):
            print("%d: %d readouts, %.3f s/100. (%.2f sec)" % (global_n, 
                   readouts * len(players), dur / (readouts*len(players) / 100.0), dur), flush=True)

        done_players.extend([p for p in players if p.is_done()])
        players = [p for p in players if not p.is_done()]

    return done_players
