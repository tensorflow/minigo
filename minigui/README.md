

### Instructions

1. Install python requirements: `pip install -r minigui/requirements.txt`

1. Download a model from our [public bucket](https://console.cloud.google.com/storage/browser/minigo-pub).

1. Make sure the command at the top of `serve.py` actually runs and prints
   `GTP engine ready`; if not, something is wrong with the rest of the minigo
   setup, like virtualenv or similar.

1. Compile the typescript. (Requires
   [typescript compiler](https://www.typescriptlang.org/#download-links)).
   Running `cd minigui; tsc` should find and compile the relevant files.

1. Set your current working directory to minigo root and start the flask server.
   Replace `$MODEL_PATH` with the path to your downloaded model.
   Replace `$BOARD_SIZE` with either `9` or `19` depending on the size of the
   model you downloaded.
   Replace `$PORT` with a port number, e.g. 5001.

```
python minigui/serve.py --model=$MODEL_PATH --board_size=$BOARD_SIZE --port=$PORT
```

1. open localhost:5001 (or whatever value you used for $PORT).

1. The buttons in the upper right that say 'Human' can be toggled to set which
   color Minigo will play.

1. By default, Minigui will use the Python Minigo engine. You can use the C++
   engine by first compiling it (see the
   [README](https://github.com/tensorflow/minigo/tree/master/cc/README.md)),
   then passing `--engine=cc` when starting `minigui/serve.py`. You will need
   to pass a frozen GraphDef proto as the `--model` command line argument
   instead of the saved parameter data that the Python backend requires.

   Here's an example, note that we specify the board size when compiling the
   C++ engine binary, not when running the Minigui server:

```
bazel build --define=board_size=19 cc:main
python minigui/serve.py --port=8888 --model=saved_models/000483-indus-upgrade.pb --engine=cc
```
