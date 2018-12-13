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

import os
import sys
sys.path.insert(0, ".")  # to run from minigo/ dir


from absl import flags
from flask import Flask
import absl.app

from flask_socketio import SocketIO

import functools
import json
import logging
import select
import subprocess
import threading

flags.DEFINE_string("model", None, "Model path.")

flags.DEFINE_integer(
    "board_size", 19,
    "Board size to use when running Python Minigo engine: either 9 or 19.")

flags.DEFINE_integer("port", 5001, "Port to listen on.")

flags.DEFINE_string("host", "127.0.0.1", "The hostname or IP to listen on.")

flags.DEFINE_string(
    "engine", "py",
    "Which Minigo engine to use: \"py\" for the Python engine, or "
    "one of the C++ engines (run \"cc/main --helpon=factory\" for the "
    "C++ engine list.")

flags.DEFINE_integer(
    "virtual_losses", 8, "Number of virtual losses when running tree search.")

flags.DEFINE_string(
    "python_for_engine", "python",
    "Which python interpreter to use for the engine. "
    "Defaults to `python` and only applies for the when --engine=py")

flags.DEFINE_integer(
    "num_readouts", 400,
    "Number of searches to add to the MCTS search tree before playing a move.")

flags.DEFINE_float("resign_threshold", -0.8, "Resign threshold.")

FLAGS = flags.FLAGS

# Suppress Flask's info logging.
log = logging.getLogger("werkzeug")
log.setLevel(logging.WARNING)

app = Flask(__name__, static_url_path="", static_folder="static")
app.config["SECRET_KEY"] = "woo"
socketio = SocketIO(app, logger=log, engineio_logger=log)


def _open_pipes():
    if FLAGS.engine == "py":
        GTP_COMMAND = [FLAGS.python_for_engine, "-u",  # turn off buffering
                       "gtp.py",
                       "--load_file=%s" % FLAGS.model,
                       "--minigui_mode=true",
                       "--num_readouts=%d" % FLAGS.num_readouts,
                       "--conv_width=128",
                       "--resign_threshold=%f" % FLAGS.resign_threshold,
                       "--verbose=2"]
    else:
        GTP_COMMAND = [
            "bazel-bin/cc/gtp",
            "--model=%s" % FLAGS.model,
            "--num_readouts=%d" % FLAGS.num_readouts,
            "--courtesy_pass=true",
            "--engine=%s" % FLAGS.engine,
            "--virtual_losses=%d" % FLAGS.virtual_losses,
            "--resign_threshold=%f" % FLAGS.resign_threshold]

    return subprocess.Popen(GTP_COMMAND,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            env=dict(os.environ, BOARD_SIZE=str(FLAGS.board_size)))


token = ""
echo_streams = True
p = None
stderr_done_semaphore = threading.Semaphore(0)


def process_line(stream_name, line):
    global token
    global echo_streams

    if echo_streams:
        sys.stdout.write(line)
        if "GTP engine ready" in line:
            echo_streams = False

    # TODO(tommadams): trim newlines in the frontend to make sure we preserve
    # the exact output of the engine.
    if line[-1] == "\n":
        line = line[:-1]

    if line.startswith("= __NEW_TOKEN__ "):
        token = line.split(" ", 3)[2]
    socketio.send(json.dumps({stream_name: line, "token": token}),
                  namespace="/minigui", json=True)


def stderr_thread():
    for line in p.stderr:
        line = line.decode()
        if line == "__GTP_CMD_DONE__\n":
            stderr_done_semaphore.release()
            continue
        process_line('stderr', line)
    print("stderr thread died")


def stdout_thread():
    for line in p.stdout:
        line = line.decode()
        if line[0] == '=' or line[0] == '?':
            # We just read the result of a GTP command, Wait for all lines
            # written to stderr while processing that command to be read.
            stderr_done_semaphore.acquire()
        process_line('stdout', line)
    print("stdout thread died")


@socketio.on("gtpcmd", namespace="/minigui")
def stdin_cmd(message):
    print("C -> E:", message)
    global p
    try:
        p.stdin.write(bytes(message["data"] + "\r\n", encoding="utf-8"))
        p.stdin.flush()
    except BrokenPipeError:
        p = _open_pipes()


@app.route("/")
def index():
    return app.send_static_file("index.html")


def main(unused_argv):
    global p
    p = _open_pipes()
    socketio.start_background_task(stderr_thread)
    socketio.start_background_task(stdout_thread)
    socketio.run(app, port=FLAGS.port, host=FLAGS.host)


if __name__ == "__main__":
    flags.mark_flags_as_required(["model"])
    absl.app.run(main)
