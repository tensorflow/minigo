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

import sys
sys.path.insert(0, '.')  # nopep8

import functools
import json
import logging
import os
import select
import shutil
import subprocess
import tempfile
import threading
import time

import absl.app
from absl import flags
from flask import Flask
import flask
from flask_socketio import SocketIO


flags.DEFINE_integer("port", 5001, "Port to listen on.")

flags.DEFINE_string("host", "127.0.0.1", "The hostname or IP to listen on.")

flags.DEFINE_string(
    "control", None,
    "Path to a control file used to configure the Go engine(s).")

FLAGS = flags.FLAGS

# Suppress Flask's info logging.
log = logging.getLogger("werkzeug")
log.setLevel(logging.WARNING)

app = Flask(__name__, static_url_path="", static_folder="static")
app.config["SECRET_KEY"] = "woo"
socketio = SocketIO(app, logger=log, engineio_logger=log)

board_size = None
connections = {}

tmp_dir = os.path.join(tempfile.gettempdir(), '.minigui_tmp')


class Player(object):
    def __init__(self, command, startup_gtp_commands=[], cwd=None, environ={}):
        self.command = command
        self.startup_gtp_commands = startup_gtp_commands
        self.cwd = cwd
        self.environ = environ

    def __repr__(self):
        return 'Player("%s, startup_gtp_commands=%s, cwd="%s", environ=%s)' % (
            self.command, self.startup_gtp_commands, self.cwd, self.environ)


class Connection(object):
    def __init__(self, name, process):
        self.name = name
        self.process = process
        self._gtp_token = ""
        self._gtp_cmd_done = threading.Semaphore(0)
        self._echo_streams = True
        self._lock = threading.Lock()

    def process_line(self, stream_name, line):
        with self._lock:
            self._process_line_locked(stream_name, line)

    def wait_for_gtp_cmd_done(self):
        self._gtp_cmd_done.acquire()

    def signal_gtp_cmd_done(self):
        self._gtp_cmd_done.release()

    def write(self, cmd):
        self.process.stdin.write(bytes("%s\r\n" % cmd, encoding="utf-8"))
        self.process.stdin.flush()

    def _process_line_locked(self, stream_name, line):
        while line and (line[-1] == "\n" or line[-1] == "\r"):
            line = line[:-1]

        if self._echo_streams:
            if stream_name == "stderr" and line.startswith("mg-"):
                self._echo_streams = False
            else:
                print("%s(%s): %s" % (self.name, stream_name, line))

        if line.startswith("= __NEW_TOKEN__ "):
            self._gtp_token = line.split(" ", 3)[2]
        socketio.send(json.dumps(
            {"player": self.name, stream_name: line, "token": self._gtp_token}),
            namespace="/minigui", json=True)


def stderr_thread(connection):
    for line in connection.process.stderr:
        line = line.decode()
        if line == "__GTP_CMD_DONE__\n":
            connection.signal_gtp_cmd_done()
            continue
        connection.process_line("stderr", line)
    print("%s: stderr thread died" % connection.name)


def stdout_thread(connection):
    for line in connection.process.stdout:
        line = line.decode()
        if line[0] == "=" or line[0] == "?":
            # We just read the result of a GTP command, Wait for all lines
            # written to stderr while processing that command to be read.
            connection.wait_for_gtp_cmd_done()
        connection.process_line("stdout", line)
    print("%s: stdout thread died" % connection.name)


@socketio.on("gtpcmd", namespace="/minigui")
def stdin_cmd(message):
    player_name = message["player"]
    data = message["data"]
    try:
        connection = connections[player_name]
    except KeyError:
        print("Unknown player \"%s\"" % player_name)
        return

    print("%s(stdin): %s" % (player_name, data))
    connection.write(data)


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/config")
def player_list():
    return json.dumps({
        "boardSize": board_size,
        "players": sorted(connections.keys()),
    })


@app.route("/write_tmp_file", methods=["POST"])
def write_tmp_file():
    os.makedirs(tmp_dir, exist_ok=True)
    fd, path = tempfile.mkstemp(dir=tmp_dir)
    try:
      os.write(fd, flask.request.data)
    finally:
      os.close(fd)
    return flask.Response(path, mimetype='text/plain')


def main(unused_argv):
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Compile and execute the control script.
    result = {"Player": Player, "players": None, "board_size": 19}
    with open(FLAGS.control, "r") as f:
        source = f.read()
    code = compile(source, FLAGS.control, "exec", 0, True)
    exec(code, result)

    # Read the board size.
    global board_size
    board_size = result["board_size"]

    global connections
    for name, player in result["players"].items():
        print("Starting engine \"%s\"" % name)

        # Split the command string into a list as required by Popen, and make
        # sure the executable path is absolute.
        command = player.command.split()
        if os.path.exists(command[0]):
            command[0] = os.path.abspath(command[0])

        # Start this player subprocess.
        process = subprocess.Popen(
            command,
            cwd=player.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(os.environ, **player.environ))

        connection = Connection(name, process)

        # Start the stderr handling thread immediately: if the engine writes to
        # stderr before we've issued the board_size GTP command, stderr must be
        # drained to avoid possible deadlock.
        socketio.start_background_task(stderr_thread, connection)

        # Verify that the engine supports the requested board size.
        # We do this before starting the stdout handling thread so that we can
        # block on the output of the boardsize GTP command.
        connection.write("boardsize %d" % board_size)
        connection.wait_for_gtp_cmd_done()
        for line in process.stdout:
            line = line.decode().rstrip()
            if line == "=":
                print("Engine \"%s\" supports board size %d" %
                      (name, board_size))
                break
            elif line[0] == "?":
                raise RuntimeError("Engine %s doesn't support boardsize %d" % (
                    name, board_size))

        # Now start the real stdout handling thread.
        socketio.start_background_task(stdout_thread, connection)

        # Process any startup commands for this engine.
        for cmd in player.startup_gtp_commands:
            connection.write(cmd)

        connections[name] = connection

    print("Starting server")
    socketio.run(app, port=FLAGS.port, host=FLAGS.host)


if __name__ == "__main__":
    absl.app.run(main)
