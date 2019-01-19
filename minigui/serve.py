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

import threading
import subprocess
import select
import logging
import json
import functools
from flask_socketio import SocketIO
import absl.app
from flask import Flask
from absl import flags
import os
import sys
sys.path.insert(0, ".")  # to run from minigo/ dir


# Server flags.
flags.DEFINE_integer("port", 5001, "Port to listen on.")

flags.DEFINE_string("host", "127.0.0.1", "The hostname or IP to listen on.")

flags.DEFINE_string(
    "control", None,
    "Path to a control file used to configure the Go engine(s).")

# Engine flags.
flags.DEFINE_string("model", None, "Model path.")

flags.DEFINE_integer(
    "virtual_losses", 8, "Number of virtual losses when running tree search.")

flags.DEFINE_integer(
    "num_readouts", 400,
    "Number of searches to add to the MCTS search tree before playing a move.")

flags.DEFINE_float("resign_threshold", -0.8, "Resign threshold.")

flags.DEFINE_float(
    "value_init_penalty", 0,
    "New children value initialize penaly.\n"
    "child's value = parent's value - value_init_penalty * color, "
    "clamped to [-1, 1].\n"
    "0 is init-to-parent [default], 2.0 is init-to-loss.\n"
    "This behaves similiarly to leela's FPU \"First Play Urgency\".")

FLAGS = flags.FLAGS

# Suppress Flask's info logging.
log = logging.getLogger("werkzeug")
log.setLevel(logging.WARNING)

app = Flask(__name__, static_url_path="", static_folder="static")
app.config["SECRET_KEY"] = "woo"
socketio = SocketIO(app, logger=log, engineio_logger=log)

players = None


class Player(object):
    def __init__(self, command, startup_gtp_commands=[], cwd=None, environ={}):
        self.name = None
        self.command = command
        self.startup_gtp_commands = startup_gtp_commands
        self.cwd = cwd
        self.environ = environ
        self.gtp_token = ""
        self.process = None
        self.gtp_cmd_done = threading.Semaphore(0)
        self.echo_streams = True

    def __repr__(self):
        return 'Player("%s, startup_gtp_commands=%s, cwd="%s", environ=%s)' % (
            self.command, self.startup_gtp_commands, self.cwd, self.environ)

    def process_line(self, stream_name, line):
        while line and (line[-1] == "\n" or line[-1] == "\r"):
            line = line[:-1]

        if self.echo_streams:
            if stream_name == "stderr" and line.startswith("mg-"):
                self.echo_streams = False
            else:
                print("%s(%s): %s" % (self.name, stream_name, line))

        if line.startswith("= __NEW_TOKEN__ "):
            self.gtp_token = line.split(" ", 3)[2]
        socketio.send(json.dumps(
            {"player": self.name, stream_name: line, "token": self.gtp_token}),
            namespace="/minigui", json=True)

    def wait_for_gtp_cmd_done(self):
        self.gtp_cmd_done.acquire()

    def signal_gtp_cmd_done(self):
        self.gtp_cmd_done.release()


def stderr_thread(player):
    for line in player.process.stderr:
        line = line.decode()
        if line == "__GTP_CMD_DONE__\n":
            player.signal_gtp_cmd_done()
            continue
        player.process_line("stderr", line)
    print("%s: stderr thread died" % player.name)


def stdout_thread(player):
    for line in player.process.stdout:
        line = line.decode()
        if line[0] == "=" or line[0] == "?":
            # We just read the result of a GTP command, Wait for all lines
            # written to stderr while processing that command to be read.
            player.wait_for_gtp_cmd_done()
        player.process_line("stdout", line)
    print("%s: stdout thread died" % player.name)


@socketio.on("gtpcmd", namespace="/minigui")
def stdin_cmd(message):
    player_name = message["player"]
    data = message["data"]
    try:
        player = players[player_name]
    except KeyError:
        print("Unknown player \"%s\"" % player_name)
        return

    print("%s(stdin): %s" % (player_name, data))
    player.process.stdin.write(bytes(data + "\r\n", encoding="utf-8"))
    player.process.stdin.flush()


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/player_list")
def player_list():
    return json.dumps(list(players.keys()))


def main(unused_argv):
    # Compile and execute the control script.
    result = {"Player": Player, "players": None, "FLAGS": FLAGS}
    with open(FLAGS.control, "r") as f:
        source = f.read()
    code = compile(source, FLAGS.control, "exec", 0, True)
    exec(code, result)

    # Start each of the players specified by the control script.
    global players
    players = result["players"]
    for name, player in players.items():
        command = player.command.split()
        if os.path.exists(command[0]):
            command[0] = os.path.abspath(command[0])
        player.name = name
        player.process = subprocess.Popen(
            command,
            cwd=player.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(os.environ, **player.environ))
        socketio.start_background_task(stderr_thread, player)
        socketio.start_background_task(stdout_thread, player)

    socketio.run(app, port=FLAGS.port, host=FLAGS.host)


if __name__ == "__main__":
    flags.mark_flags_as_required(["model"])
    absl.app.run(main)
