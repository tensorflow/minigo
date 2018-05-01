## Minigui

A UI for Minigo

### Simple Instructions

1. Make sure you have Docker installed

1. Pick a model from [Cloudy Go](http://cloudygo.com). Make note of the Model
   Name and the Board Size.

    ```shell
    export MINIGUI_BOARD_SIZE=9
    export MINIGUI_MODEL=000360-grown-teal
    ```

1. From the root directory, run:

    ```shell
    cluster/minigui/run-local.sh
    ```

1. Navigate to `localhost:5001`

### Advanced Instructions

1. Install the minigo python requirements: `pip install -r requirements.txt` (or
   `pip3 ...` depending how you've set things up).

1. Install TensorFlow (here, we use the CPU install): `pip install "tensorflow>=1.7,<1.8"`

1. Install the **minigui** python requirements: `pip install -r minigui/requirements.txt`

1. Install the [Google Cloud SDK](https://cloud.google.com/sdk/downloads)

1. Pick a model. See http://cloudygo.com/ for the available models.

1. Change the variables you want (these are the defaults):

    ```shell
    export MINIGUI_BUCKET_NAME=minigo-pub
    export MINIGUI_GCS_DIR=v5-19x19/models
    export MINIGUI_MODEL=000363-auriga
    export MINIGUI_MODEL_TMPDIR=/tmp/minigo-models
    export MINIGUI_BOARD_SIZE=19
    ```

1. Run `source minigui-common.sh`

1. Compile the Typescript to JavaSCript. (Requires
   [typescript compiler](https://www.typescriptlang.org/#download-links)).
   From the `minigui` directory run: `tsc`

1. Run `./fetch-and-run.sh`

1. open localhost:5001 (or whatever value you used for $PORT).

1. The buttons in the upper right that say 'Human' can be toggled to set which
   color Minigo will play.

**C++ Engine**

1. By default, Minigui will use the Python Minigo engine. You can use the C++
   engine by first compiling it (see the
   [README](https://github.com/tensorflow/minigo/tree/master/cc/README.md)),
   then passing `--engine=cc` when starting `minigui/serve.py`. You will need
   to pass a frozen GraphDef proto as the `--model` command line argument
   instead of the saved parameter data that the Python backend requires.

   **Note:** Compiling tensorflow from scratch can take 2+ hours if your
   machine is not terribly beefy, So you might want to kick off the build and
   get a coffee.

1. Before running, you'll need to freeze a model so the C++ job can consume it.
   This assumes a converted model from above.

    ```shell
    python main.py freeze-graph $MODEL_DIR/$MODEL.converted
    ```

1. Here's an example, note that we specify the board size when compiling the
   C++ engine binary, not when running the Minigui server:

    ```shell
    bazel build --define=board_size=19 cc:main
    python minigui/serve.py --port=8888 --model=$MODEL_DIR/$MODEL.converted.pb --engine=cc
    ```
