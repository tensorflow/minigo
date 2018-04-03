import sys
sys.path.insert(0, ".")  # to run from minigo/ dir


from flask import Flask

from flask_socketio import SocketIO

import json
import subprocess
from threading import Lock
import functools

app = Flask(__name__, static_url_path="", static_folder="static")
app.config["SECRET_KEY"] = "woo"
socketio = SocketIO(app)

#TODO(amj) extract to flag
MODEL_PATH = "saved_models/000496-polite-ray-upgrade"
GTP_COMMAND = ["python",  "-u",  # turn off buffering
               "main.py", "gtp",
               "--load-file", MODEL_PATH,
               "--readouts", "1000",
               "-v", "2"]


def _open_pipes():
    return subprocess.Popen(GTP_COMMAND,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)


p = _open_pipes()

stderr_thread = None
stdout_thread = None
token = ''
thread_lock = Lock()


def std_bg_thread(stream):
    global token

    for line in p.__getattribute__(stream):
        line = line.decode()
        if line[-1] == "\n":
            line = line[:-1]

        if line.startswith("= __NEW_TOKEN__ "):
            token = line.split(" ", 3)[2]
        else:
            socketio.send(json.dumps({stream: line, "token": token}),
                          namespace="/minigui", json=True)
    print(stream, "bg_thread died")


@socketio.on("connect", namespace="/minigui")
def stderr_connected():
    global stderr_thread
    global stdout_thread
    with thread_lock:
        if stderr_thread is None:
            stderr_thread = socketio.start_background_task(
                target=functools.partial(std_bg_thread, "stderr"))
        if stdout_thread is None:
            stdout_thread = socketio.start_background_task(
                target=functools.partial(std_bg_thread, "stdout"))


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


if __name__ == "__main__":
  socketio.run(app)
