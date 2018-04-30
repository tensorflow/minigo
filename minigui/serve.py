import os
import sys
sys.path.insert(0, ".")  # to run from minigo/ dir


import argparse
from flask import Flask

from flask_socketio import SocketIO

import functools
import json
import logging
import subprocess
from threading import Lock

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    required=True,
    type=str,
    help="Model path.")

parser.add_argument(
    "--board_size",
    default=19,
    type=int,
    help="Board size to use when running Python Minigo engine: either 9 or "
         "19.")

parser.add_argument(
    "--port",
    default=5001,
    type=int,
    help="Port to listen on.")

parser.add_argument(
    "--host",
    default="127.0.0.1",
    type=str,
    help="The hostname or IP to listen on.")

parser.add_argument(
    "--engine",
    default="py",
    type=str,
    help="Which Minigo engine to use: \"py\" for the Python implementation, "
         "\"cc\" for the C++ implementation.")


parser.add_argument(
    "--python_for_engine",
    default="python",
    type=str,
    help="Which python interpreter to use for the engine. "
         "Defaults to `python` and only applies for the when --engine=py")

args = parser.parse_args()


# Suppress Flask's info logging.
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

app = Flask(__name__, static_url_path="", static_folder="static")
app.config["SECRET_KEY"] = "woo"
socketio = SocketIO(app)


python_binary = args.python_for_engine

if args.engine == "py":
    GTP_COMMAND = [python_binary,  "-u",  # turn off buffering
                   "main.py", "gtp",
                   "--load-file", args.model,
                   "--num_readouts", "1000",
                   "-v", "2"]
elif args.engine == "cc":
    GTP_COMMAND = [
        "bazel-bin/cc/main",
        "--model=" + args.model,
        "--num_readouts=1000",
        "--soft_pick=false",
        "--inject_noise=false",
        "--disable_resign_pct=0",
        "--mode=gtp"]
else:
    raise ValueError(
        "Expected \"py\" or \"cc\" for engine, got \"%s\"" % args.engine)


def _open_pipes():
    return subprocess.Popen(GTP_COMMAND,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            env=dict(os.environ, BOARD_SIZE=str(args.board_size)))


token = ''
echo_streams = True


def std_bg_thread(stream):
    global token
    global echo_streams

    for line in p.__getattribute__(stream):
        line = line.decode()
        if echo_streams:
            sys.stdout.write(line)
            if "GTP engine ready" in line:
                echo_streams = False

        if line[-1] == "\n":
            line = line[:-1]

        if line.startswith("= __NEW_TOKEN__ "):
            token = line.split(" ", 3)[2]
        else:
            socketio.send(json.dumps({stream: line, "token": token}),
                          namespace="/minigui", json=True)
    print(stream, "bg_thread died")


@socketio.on("gtpcmd", namespace="/minigui")
def stdin_cmd(message):
    print("C -> E:", message)
    global p
    try:
        p.stdin.write(bytes(message["data"] + "\r\n", encoding="utf-8"))
        p.stdin.flush()
    except BrokenPipeError:
        p = _open_pipes()


p = _open_pipes()
stderr_thread = socketio.start_background_task(
    target=functools.partial(std_bg_thread, "stderr"))
stdout_thread = socketio.start_background_task(
    target=functools.partial(std_bg_thread, "stdout"))


@app.route("/")
def index():
    return app.send_static_file("index.html")


if __name__ == "__main__":
  socketio.run(app, port=args.port, host=args.host)
