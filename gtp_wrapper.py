import gtp
import gtp_extensions

import datetime
import go
import random
import utils
import sys
import os
from dual_net import DualNetwork
from strategies import MCTSPlayerMixin, CGOSPlayerMixin

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

    def set_size(self, n):
        self.size = n
        go.set_board_size(n)
        self.clear()

    def set_komi(self, komi):
        self.komi = komi
        self.position.komi = komi

    def clear(self):
        if self.position and len(self.position.recent) > 1:
            try:
                sgf = self.to_sgf()
                with open(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M.sgf"), 'w') as f:
                        f.write(sgf)
            except NotImplementedError:
                pass
            except:
                print("Error saving sgf", file=sys.stderr, flush=True)
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
        move = self.suggest_move(self.position)
        if self.should_resign():
            return gtp.RESIGN
        return utils.unparse_pygtp_coords(move)

    def final_score(self):
        return self.position.result()

    def showboard(self):
        print('\n\n' + str(self.position) + '\n\n', file=sys.stderr)
        return True

    def should_resign(self):
        raise NotImplementedError

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

    def to_sgf(self):
        raise NotImplementedError


class MCTSPlayer(MCTSPlayerMixin, GtpInterface): pass
class CGOSPlayer(CGOSPlayerMixin, GtpInterface): pass

def make_gtp_instance(read_file, readouts_per_move=100, verbosity=1, cgos_mode=False):
    n = DualNetwork()
    try:
        n.initialize_variables(read_file)
    except:
        n.initialize_variables()
    if cgos_mode:
        instance = CGOSPlayer(n, seconds_per_move=5, verbosity=verbosity, two_player_mode=True)
    else:
        instance = MCTSPlayer(n, simulations_per_move=readouts_per_move,
                              verbosity=verbosity, two_player_mode=True)
    name ="Somebot-" + os.path.basename(read_file)
    gtp_engine = gtp_extensions.GTPDeluxe(instance, name=name)
    return gtp_engine
