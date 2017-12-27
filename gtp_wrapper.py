import gtp
import gtp_extensions

import go
import random
import utils
import sys
import os
from dual_net import DualNetwork
from strategies import MCTSPlayerMixin

def translate_gtp_colors(gtp_color):
    if gtp_color == gtp.BLACK:
        return go.BLACK
    elif gtp_color == gtp.WHITE:
        return go.WHITE
    else:
        return go.EMPTY

class GtpInterface(object):
    def __init__(self):
        self.size = 9
        self.position = None
        self.komi = 6.5
        self.clear()

    def set_size(self, n):
        self.size = n
        go.set_board_size(n)
        self.clear()

    def set_komi(self, komi):
        self.komi = komi
        self.position.komi = komi

    def clear(self):
        self.position = go.Position(komi=self.komi)
        self.initialize_game()

    def accomodate_out_of_turn(self, color):
        if not translate_gtp_colors(color) == self.position.to_play:
            self.position.flip_playerturn(mutate=True)

    def make_move(self, color, vertex):
        coords = utils.parse_pygtp_coords(vertex)
        # let's assume this never happens for now.
        # self.accomodate_out_of_turn(color)
        return self.play_move(coords)

    def get_move(self, color):
        self.accomodate_out_of_turn(color)
        if self.should_resign():
            return gtp.RESIGN

        if self.should_pass(self.position):
            return gtp.PASS

        move = self.suggest_move(self.position)
        return utils.unparse_pygtp_coords(move)

    def final_score(self):
        return self.position.result()

    def showboard(self):
        print('\n\n' + str(self.position) + '\n\n', file=sys.stderr)

    def should_resign(self):
        raise NotImplementedError

    def should_pass(self, position):
        # Pass if the opponent passes
        return position.n > 100 and position.recent and position.recent[-1].move == None

    def get_score(self):
        return self.position.result()

    def suggest_move(self, position):
        raise NotImplementedError

    def play_move(self, coords):
        raise NotImplementedError

    def initialize_game(self):
        raise NotImplementedError

    def chat(self, msg_type, sender, text):
        raise NotImplementedError


class MCTSPlayer(MCTSPlayerMixin, GtpInterface): pass

def make_gtp_instance(read_file, readouts_per_move=100, verbosity=1):
    n = DualNetwork()
    try:
        n.initialize_variables(read_file)
    except:
        n.initialize_variables()
    instance = MCTSPlayer(n, simulations_per_move=readouts_per_move, verbosity=verbosity, two_player_mode=True)
    name ="Somebot-" + os.path.basename(read_file)
    gtp_engine = gtp_extensions.KgsExtensionsMixin(instance, name=name)
    return gtp_engine
