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

import gtp
import gtp_extensions

import coords
import datetime
import go
import random
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
        if n != go.N:
            raise ValueError("unacceptable size")

    def set_komi(self, komi):
        self.komi = komi
        self.position.komi = komi

    def clear(self):
        if self.position and self.result_string and len(self.position.recent) > 1:
            try:
                sgf = self.to_sgf()
                with open(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M.sgf"), 'w') as f:
                    f.write(sgf)
            except NotImplementedError:
                pass
            except:
                print("Error saving sgf", file=sys.stderr, flush=True)
        self.position = go.Position(komi=self.komi)
        self.initialize_game(self.position)
        return True

    def accomodate_out_of_turn(self, color):
        if not translate_gtp_colors(color) == self.position.to_play:
            self.position.flip_playerturn(mutate=True)

    def make_move(self, color, vertex):
        c = coords.from_pygtp(vertex)
        # let's assume this never happens for now.
        # self.accomodate_out_of_turn(color)
        return self.play_move(c)

    def get_move(self, color):
        self.accomodate_out_of_turn(color)
        move = self.suggest_move(self.position)
        if self.should_resign():
            return gtp.RESIGN
        return coords.to_pygtp(move)

    def final_score(self):
        return self.position.result_string()

    def showboard(self):
        print('\n\n' + str(self.position) + '\n\n', file=sys.stderr)
        return True

    def should_resign(self):
        raise NotImplementedError

    def get_score(self):
        return self.position.result_string()

    def suggest_move(self, position):
        raise NotImplementedError

    def play_move(self, c):
        raise NotImplementedError

    def initialize_game(self):
        raise NotImplementedError

    def chat(self, msg_type, sender, text):
        raise NotImplementedError

    def to_sgf(self):
        raise NotImplementedError


class MCTSPlayer(MCTSPlayerMixin, GtpInterface):
    pass


class CGOSPlayer(CGOSPlayerMixin, GtpInterface):
    pass


class KGSPlayer(MCTSPlayer):
    """ A wrapper for playing on KGS
    Adds 'courtsey pass' -- if the opponent passes, we pass, regardless
    of score or our opinion on the game. """

    def __init__(self, **kwargs):
        self.they_passed = False
        super().__init__(**kwargs)

    def get_move(self, color):
        if self.they_passed:
            return gtp.PASS
        return super().get_move(color)

    def make_move(self, color, vertex):
        if vertex == gtp.PASS:
            self.they_passed = True
        else:
            self.they_passed = False
        return super().make_move(color, vertex)

    def clear(self):
        self.they_passed = False
        return super().clear()


def make_gtp_instance(read_file, readouts_per_move=100, verbosity=1, cgos_mode=False, kgs_mode=False):
    n = DualNetwork(read_file)
    if cgos_mode:
        instance = CGOSPlayer(n, seconds_per_move=5, timed_match=True,
                              verbosity=verbosity, two_player_mode=True)
    elif kgs_mode:
        instance = KGSPlayer(network=n, num_readouts=readouts_per_move,
                             verbosity=verbosity, two_player_mode=True)
    else:
        instance = MCTSPlayer(network=n, num_readouts=readouts_per_move,
                              verbosity=verbosity, two_player_mode=True)
    name = "Minigo-" + os.path.basename(read_file)
    gtp_engine = gtp_extensions.GTPDeluxe(instance, name=name)
    return gtp_engine
