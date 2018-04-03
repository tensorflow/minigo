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

# extends gtp.py

import gtp
import sys
import sgf_wrapper
import itertools
import json
import go
import coords
import time


def parse_message(message):
    message = gtp.pre_engine(message).strip()
    first, rest = (message.split(" ", 1) + [None])[:2]
    if first.isdigit():
        message_id = int(first)
        if rest is not None:
            command, arguments = (rest.split(" ", 1) + [None])[:2]
        else:
            command, arguments = None, None
    else:
        message_id = None
        command, arguments = first, rest

    command = command.replace("-", "_")  # for kgs extensions.
    return message_id, command, arguments


def dbg(fmt, *args):
    print(fmt % args, file=sys.stderr, flush=True)


class KgsExtensionsMixin(gtp.Engine):

    def __init__(self, game_obj, name="gtp (python, kgs-chat extensions)", version="0.1"):
        super().__init__(game_obj=game_obj, name=name, version=version)
        self.known_commands += ["kgs-chat"]

    def send(self, message):
        message_id, command, arguments = parse_message(message)
        if command in self.known_commands:
            try:
                retval = getattr(self, "cmd_" + command)(arguments)
                response = gtp.format_success(message_id, retval)
                sys.stderr.flush()
                return response
            except ValueError as exception:
                return gtp.format_error(message_id, exception.args[0])
        else:
            return gtp.format_error(message_id, "unknown command: " + command)

    # Nice to implement this, as KGS sends it each move.
    def cmd_time_left(self, arguments):
        pass

    def cmd_showboard(self, arguments):
        return self._game.showboard()

    def cmd_kgs_chat(self, arguments):
        try:
            msg_type, sender, *text = arguments.split()
            text = " ".join(text)
        except ValueError:
            return "Unparseable message, args: %r" % arguments
        return self._game.chat(msg_type, sender, text)


class RegressionsMixin(gtp.Engine):
    def cmd_loadsgf(self, arguments):
        args = arguments.split()
        if len(args) == 2:
            file_, movenum = args
            movenum = int(movenum)
            print("movenum =", movenum, file=sys.stderr)
        else:
            file_ = args[0]
            movenum = None

        try:
            with open(file_, 'r') as f:
                contents = f.read()
        except:
            raise ValueError("Unreadable file: " + file_)

        try:
            # This is kinda bad, because replay_sgf is already calling
            # 'play move' on its internal position objects, but we really
            # want to advance the engine along with us rather than try to
            # push in some finished Position object.
            for idx, p in enumerate(sgf_wrapper.replay_sgf(contents)):
                print("playing #", idx, p.next_move, file=sys.stderr)
                self._game.play_move(p.next_move)
                if movenum and idx == movenum:
                    break
        except:
            raise

# Should this class blatantly reach into the game_obj and frob its tree?  Sure!
# What are private members?  Python lets you do *anything!*


class GoGuiMixin(gtp.Engine):
    """ GTP extensions of 'analysis commands' for gogui.
    We reach into the game_obj (an instance of the players in strategies.py),
    and extract stuff from its root nodes, etc.  These could be extracted into
    methods on the Player object, but its a little weird to do that on a Player,
    which doesn't really care about GTP commands, etc.  So instead, we just
    violate encapsulation a bit... Suggestions welcome :) """

    def __init__(self, game_obj, name="gtp (python, gogui extensions)", version="0.1"):
        super().__init__(game_obj=game_obj, name=name, version=version)
        self.session_id = ""
        self.known_commands += ["gogui-analyze_commands"]

    def cmd_gogui_analyze_commands(self, arguments):
        return "\n".join(["var/Most Read Variation/nextplay",
                          "var/Think a spell/spin",
                          "var/Final score/final_score",
                          "pspairs/Visit Heatmap/visit_heatmap",
                          "pspairs/Q Heatmap/q_heatmap"])

    def cmd_nextplay(self, arguments):
        return self._game.root.mvp_gg()

    def cmd_visit_heatmap(self, arguments):
        sort_order = list(range(self._game.size * self._game.size + 1))
        sort_order.sort(key=lambda i: self._game.root.child_N[i], reverse=True)
        return self.heatmap(sort_order, self._game.root, 'child_N')

    def cmd_q_heatmap(self, arguments):
        sort_order = list(range(self._game.size * self._game.size + 1))
        reverse = True if self._game.root.position.to_play is go.BLACK else False
        sort_order.sort(
            key=lambda i: self._game.root.child_Q[i], reverse=reverse)
        return self.heatmap(sort_order, self._game.root, 'child_Q')

    def heatmap(self, sort_order, node, prop):
        return "\n".join(["{!s:6} {}".format(
            coords.to_kgs(coords.from_flat(key)),
            node.__dict__.get(prop)[key])
            for key in sort_order if node.child_N[key] > 0][:20])

    def cmd_spin(self, arguments):
        for i in range(50):
            for j in range(100):
                self._game.tree_search()
            moves = self.cmd_nextplay(None).lower()
            moves = moves.split()
            colors = "bw" if self._game.root.position.to_play is go.BLACK else "wb"
            moves_cols = " ".join(['{} {}'.format(*z)
                                   for z in zip(itertools.cycle(colors), moves)])
            print("gogui-gfx: TEXT", "{:.3f} after {}".format(
                self._game.root.Q, self._game.root.N), file=sys.stderr, flush=True)
            print("gogui-gfx: VAR", moves_cols, file=sys.stderr, flush=True)
        return self.cmd_nextplay(None)

    def _minigui_report_search_status(self, leaves):
        """Prints the current MCTS search status to stderr.

        Reports the current search path, root node's child_Q, root node's
        child_N, the most visited path in a format that can be parsed by
        one of the STDERR_HANDLERS in minigui.ts.

        Args:
          leaves: list of leaf MCTSNodes returned by game.tree_search.
         """
        if leaves:
            path = []
            leaf = leaves[0]
            while leaf != self._game.root:
                path.append(leaf.fmove)
                leaf = leaf.parent
            path = [coords.to_kgs(coords.from_flat(m)) for m in reversed(path)]
            dbg("mg-search:%s", " ".join(path))

        q = self._game.root.child_Q - self._game.root.Q
        q = ['%.3f' % x for x in q]
        dbg("mg-q:%s", " ".join(q))

        n = ['%d' % x for x in self._game.root.child_N]
        dbg("mg-n:%s", " ".join(n))

        nodes = self._game.root.most_visited_path_nodes()
        path = [coords.to_kgs(coords.from_flat(m.fmove)) for m in nodes]
        dbg("mg-pv:%s", " ".join(path))

    def _dbg_game_state(self):
        position = self._game.position
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
        msg["toPlay"] = "Black" if position.to_play == 1 else "White"
        if position.recent:
            msg["lastMove"] = coords.to_kgs(position.recent[-1].move)
        else:
            msg["lastMove"] = None
        msg["session"] = self.session_id
        msg["n"] = position.n
        if self._game.root.parent and self._game.root.parent.parent:
            msg["q"] = self._game.root.parent.Q
        else:
            msg["q"] = 0
        dbg("mg-gamestate:%s", json.dumps(msg, sort_keys=True))

    def cmd_echo(self, arguments):
        return arguments

    def cmd_mg_genmove(self, arguments):
        """Like regular genmove but reports the MCTS status periodically.

        Args:
          arguments: A string containing how many calls to tree_search should be
                     made between reporting the MCTS status.
        """

        game = self._game

        start = time.time()
        debug_interval = int(arguments)
        current_readouts = game.root.N
        # This rather strange initial value means that the search status will
        # be reported after the very first call to tree_search, rather than
        # after debug_interval calls.
        last_dbg = -debug_interval
        leaves = None
        num_readouts = game.simulations_per_move
        while game.root.N < current_readouts + num_readouts:
            search_result = game.tree_search()
            if search_result:
                leaves = search_result
            if game.root.N - last_dbg > debug_interval:
                last_dbg = game.root.N
                self._minigui_report_search_status(leaves)

        move = game.pick_move()

        duration = time.time() - start

        self._minigui_report_search_status(leaves)

        dbg("")
        dbg(game.root.describe())
        if game.should_resign():
            game.set_result(-1 * game.root.position.to_play, was_resign=True)
            # Tell the game object that we're passing to update the root node.
            # This is required to ensure that subsequents calls to gamestate
            # return the correct information.
            game.play_move(None)
            return gtp.RESIGN
        game.play_move(move)
        if game.root.is_done():
            game.set_result(game.root.position.result(), was_resign=False)
        dbg("")
        dbg(game.root.position.__str__(colors=False))
        dbg("%d readouts, %.3f s/100. (%.2f sec)",
            num_readouts, duration / num_readouts * 100.0, duration)
        dbg("")
        return gtp.gtp_vertex(coords.to_pygtp(move))

    def cmd_readouts(self, arguments):
        try:
            reads = max(8, int(arguments))
            self._game.simulations_per_move = reads
            return reads
        except:
            return False

    def cmd_mg_gamestate(self, arguments):
        self._dbg_game_state()

    def cmd_play(self, arguments):
        try:
            super().cmd_play(arguments)
            game = self._game
            if game.root.is_done():
                game.set_result(game.root.position.result(), was_resign=False)
            return True
        except:
            dbg("ILLEGAL MOVE: %s", arguments)
            return False

    def cmd_final_score(self, arguments):
        return self._game.result_string


class GTPDeluxe(KgsExtensionsMixin, RegressionsMixin, GoGuiMixin):
    pass
