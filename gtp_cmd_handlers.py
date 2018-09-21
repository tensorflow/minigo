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

from datetime import datetime
import itertools
import json
import time
import sgf_wrapper
import go
import coords
from utils import dbg


def translate_gtp_color(gtp_color):
    if gtp_color.lower() in ["b", "black"]:
        return go.BLACK
    if gtp_color.lower() in ["w", "white"]:
        return go.WHITE
    raise ValueError("invalid color {}".format(gtp_color))


class BasicCmdHandler(object):
    """GTP command handler for basic play commands."""

    def __init__(self, player, courtesy_pass=False):
        self._komi = 6.5
        self._player = player
        self._player.initialize_game()
        self._courtesy_pass = courtesy_pass

    def cmd_boardsize(self, n: int):
        if n != go.N:
            raise ValueError("unsupported board size: {}".format(n))

    def cmd_clear_board(self):
        position = self._player.get_position()
        if (self._player.get_result_string() and
                position and len(position.recent) > 1):
            try:
                sgf = self._player.to_sgf()
                with open(datetime.now().strftime("%Y-%m-%d-%H:%M.sgf"), 'w') as f:
                    f.write(sgf)
            except NotImplementedError:
                pass
            except:
                dbg("Error saving sgf")
        self._player.initialize_game(go.Position(komi=self._komi))

    def cmd_komi(self, komi: float):
        self._komi = komi
        self._player.get_position().komi = komi

    def cmd_play(self, arg0: str, arg1=None):
        if arg1 is None:
            move = arg0
        else:
            # let's assume this never happens for now.
            # self._accomodate_out_of_turn(translate_gtp_color(arg0))
            move = arg1
        return self._player.play_move(coords.from_kgs(move))

    def cmd_genmove(self, color=None):
        if color is not None:
            self._accomodate_out_of_turn(color)

        if self._courtesy_pass:
            # If courtesy pass is True and the previous move was a pass, we'll
            # pass too, regardless of score or our opinion on the game.
            position = self._player.get_position()
            if position.recent and position.recent[-1].move is None:
                return "pass"

        move = self._player.suggest_move(self._player.get_position())
        if self._player.should_resign():
            self._player.set_result(-1 * self._player.get_position().to_play,
                                    was_resign=True)
            return "resign"

        self._player.play_move(move)
        if self._player.get_root().is_done():
            self._player.set_result(self._player.get_position().result(),
                                    was_resign=False)
        return coords.to_kgs(move)

    def cmd_undo(self):
        raise NotImplementedError()

    def cmd_showboard(self):
        dbg('\n\n' + str(self._player.get_position()) + '\n\n')
        return True

    def cmd_final_score(self):
        return self._player.get_result_string()

    def _accomodate_out_of_turn(self, color: str):
        position = self._player.get_position()
        if translate_gtp_color(color) != position.to_play:
            position.flip_playerturn(mutate=True)


class KgsCmdHandler(object):
    def __init__(self, player):
        self._player = player

    def cmd_time_left(self, color: str, time: int, stones: int):
        pass

    def cmd_kgs_chat(self, msg_type: str, sender: str, text: str):
        if not hasattr(self._player, 'get_root'):
            return "I have nothing interesting to say."

        root = self._player.get_root()
        default_response = "Supported commands are 'winrate', 'nextplay', 'fortune', and 'help'."
        if root is None or root.position.n == 0:
            return "I'm not playing right now.  " + default_response

        if 'winrate' in text.lower():
            wr = (abs(root.Q) + 1.0) / 2.0
            color = "Black" if root.Q > 0 else "White"
            return "{} {:.2f}%".format(color, wr * 100.0)
        elif 'nextplay' in text.lower():
            return "I'm thinking... " + root.most_visited_path()
        elif 'fortune' in text.lower():
            return "You're feeling lucky!"
        elif 'help' in text.lower():
            return "I can't help much with go -- try ladders!  Otherwise: " + default_response
        else:
            return default_response


class RegressionsCmdHandler(object):
    def __init__(self, player):
        self._player = player

    def cmd_loadsgf(self, filename: str, movenum=0):
        try:
            with open(filename, 'r') as f:
                contents = f.read()
        except:
            raise ValueError("Unreadable file: " + filename)

        # Clear the board before replaying sgf
        # TODO: should this use the sgfs komi?
        self._player.initialize_game(go.Position())

        # This is kinda bad, because replay_sgf is already calling
        # 'play move' on its internal position objects, but we really
        # want to advance the engine along with us rather than try to
        # push in some finished Position object.
        for idx, p in enumerate(sgf_wrapper.replay_sgf(contents)):
            dbg("playing #", idx, p.next_move)
            self._player.play_move(p.next_move)
            if movenum and idx == movenum:
                break


class GoGuiCmdHandler(object):
    """GTP extensions of 'analysis commands' for gogui."""

    def __init__(self, player):
        self._player = player

    def cmd_gogui_analyze_commands(self):
        return "\n".join(["var/Most Read Variation/nextplay",
                          "var/Think a spell/spin",
                          "var/Final score/final_score",
                          "pspairs/Visit Heatmap/visit_heatmap",
                          "pspairs/Q Heatmap/q_heatmap"])

    def cmd_nextplay(self):
        return self._player.get_root().mvp_gg()

    def cmd_visit_heatmap(self):
        root = self._player.get_root()
        sort_order = list(range(go.N * go.N + 1))
        sort_order.sort(key=lambda i: root.child_N[i], reverse=True)
        return self._heatmap(sort_order, root, 'child_N')

    def cmd_spin(self):
        for i in range(50):
            for j in range(100):
                self._player.tree_search()
            moves = self.cmd_nextplay().lower()
            moves = moves.split()
            root = self._player.get_root()
            colors = "bw" if root.position.to_play is go.BLACK else "wb"
            moves_cols = " ".join(['{} {}'.format(*z)
                                   for z in zip(itertools.cycle(colors), moves)])
            dbg("gogui-gfx: TEXT", "{:.3f} after {}".format(root.Q, root.N))
            dbg("gogui-gfx: VAR", moves_cols)
        return self.cmd_nextplay()

    def _heatmap(self, sort_order, node, prop):
        return "\n".join(["{!s:6} {}".format(
            coords.to_kgs(coords.from_flat(key)),
            node.__dict__.get(prop)[key])
            for key in sort_order if node.child_N[key] > 0][: 20])


class MiniguiBasicCmdHandler(BasicCmdHandler):
    def __init__(self, player, courtesy_pass=False):
        super().__init__(player, courtesy_pass)

        # Wrap the game's tree_search method, which allows us to stream the
        # search state back over stderr if requested.
        self._tree_search = self._player.tree_search
        self._player.tree_search = self._tree_search_wrapper

        self._last_report_time = None
        self._report_search_interval = 0.0
        self._last_pv = None

    def cmd_echo(self, *args):
        return " ".join(args)

    def cmd_info(self):
        return ("num_readouts: %d report_search_interval: %.1f n: %d "
                "resign_threshold: %f" % (
                    self._player.get_num_readouts(),
                    self._report_search_interval * 1000, go.N,
                    self._player.resign_threshold))

    def cmd_readouts(self, readouts: int):
        readouts = max(8, readouts)
        self._player.set_num_readouts(readouts)
        return readouts

    def cmd_report_search_interval(self, interval_ms: float):
        self._report_search_interval = interval_ms / 1000.0

    def cmd_play(self, arg0: str, arg1=None):
        super().cmd_play(arg0, arg1)
        root = self._player.get_root()
        if root.is_done():
            self._player.set_result(
                root.position.result(), was_resign=False)

    def cmd_gamestate(self):
        position = self._player.get_position()
        root = self._player.get_root()
        msg = {}
        board = []
        for row in range(go.N):
            for col in range(go.N):
                stone = position.board[row, col]
                if stone == go.BLACK:
                    board.append("X")
                elif stone == go.WHITE:
                    board.append("O")
                else:
                    board.append(".")
        msg["board"] = "".join(board)
        if position.recent:
            msg["lastMove"] = coords.to_kgs(position.recent[-1].move)
        else:
            msg["lastMove"] = None
        msg["toPlay"] = "B" if position.to_play == 1 else "W"
        msg["moveNum"] = position.n
        msg["q"] = root.parent.Q if root.parent and root.parent.parent else 0
        msg["gameOver"] = position.is_game_over()
        dbg("mg-gamestate:%s" % json.dumps(msg, sort_keys=True))

    def cmd_genmove(self, color=None):
        start = time.time()
        result = super().cmd_genmove(color)
        duration = time.time() - start

        root = self._player.get_root()
        if result != "resign":
            dbg("")
            dbg(root.position.__str__(colors=False))
            dbg("%d readouts, %.3f s/100. (%.2f sec)" % (
                self._player.get_num_readouts(),
                duration / self._player.get_num_readouts() * 100.0, duration))
            dbg("")
            if root.is_done():
                self._player.set_result(
                    root.position.result(), was_resign=False)

        return result

    def _tree_search_wrapper(self, parallel_readouts=None):
        leaves = self._tree_search(parallel_readouts)
        if self._report_search_interval:
            now = time.time()
            if (self._last_report_time is None or
                    now - self._last_report_time > self._report_search_interval):
                self._minigui_report_search_status(leaves)
                self._last_report_time = now
        return leaves

    def _minigui_report_search_status(self, leaves):
        """Prints the current MCTS search status to stderr.

        Reports the current search path, root node's child_Q, root node's
        child_N, the most visited path in a format that can be parsed by
        one of the STDERR_HANDLERS in minigui.ts.

        Args:
          leaves: list of leaf MCTSNodes returned by tree_search().
         """

        root = self._player.get_root()

        msg = {
            "moveNum": root.position.n,
            "toPlay": "B" if root.position.to_play == go.BLACK else "W",
        }

        if leaves:
            path = []
            leaf = leaves[0]
            while leaf != root:
                path.append(leaf.fmove)
                leaf = leaf.parent
            msg["search"] = [coords.to_kgs(coords.from_flat(m))
                             for m in reversed(path)]
        else:
            msg["search"] = []

        dq = root.child_Q - root.Q
        msg["dq"] = [int(round(x * 100)) for x in dq]

        msg["n"] = [int(n) for n in root.child_N]

        nodes = root.most_visited_path_nodes()
        pv = [coords.to_kgs(coords.from_flat(m.fmove)) for m in nodes]
        if pv != self._last_pv:
            msg["pv"] = pv
            self._last_pv = pv

        dbg("mg-search:%s" % json.dumps(msg, sort_keys=True))
