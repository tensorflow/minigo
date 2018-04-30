

### Instructions

1. Install the minigo python requirements: `pip install -r requirements.txt` (or
   `pip3 ...` depending how you've set things up).

1. Install TensorFlow (here, we use the CPU install): `pip install "tensorflow>=1.7,<1.8"`

1. Install the **minigui** python requirements: `pip install -r minigui/requirements.txt`

1. Pick a model. See http://cloudygo.com/ for the available models.

1. Set up some environment variables. Pick the right model name, path, and
   boardsize. Note that each model can only play a specific size of Go board.

    ```shell
    BUCKET_NAME=tensor-go-minigo-v5-19
    GCS_DIR=models
    MODEL=000570-kingfisher
    MODEL_DIR=/tmp/minigo-models
    BOARD_SIZE=19
    ```

1. Download a model from our [public bucket](https://console.cloud.google.com/storage/browser/minigo-pub). For example:

    ```shell
    mkdir -p $MODEL_DIR
    gsutil cp gs://${BUCKET_NAME}/${GCS_DIR}/${MODEL}.data-00000-of-00001 $MODEL_DIR/
    gsutil cp gs://${BUCKET_NAME}/${GCS_DIR}/${MODEL}.index $MODEL_DIR/
    gsutil cp gs://${BUCKET_NAME}/${GCS_DIR}/${MODEL}.meta $MODEL_DIR/
    ```

1. Compile the typescript. (Requires
   [typescript compiler](https://www.typescriptlang.org/#download-links)).
   From the `minigui` directory run: `tsc`

1. Set your current working directory to minigo root and start the flask server.

    ```shell
    python minigui/serve.py --model=$MODEL_DIR/$MODEL --board_size=$BOARD_SIZE --port=5001
    ```

1. If you get *Invalid size in bundle entry: key global_step; stored size 4; expected size 8*,
   you'll need to run the following:

    ```shell
    export BOARD_SIZE=$BOARD_SIZE
    python main.py convert $MODEL_DIR/$MODEL $MODEL_DIR/$MODEL.converted
    ```

    Make sure there isn't
    a directory in `$MODEL_DIR` named `$MODEL` or tensorflow will get
    confused. Not specifying `BOARD_SIZE` can also end up giving you errors like,
    *Assign requires shapes of both tensors to match. lhs shape= [3,3,17,32]
    rhs shape= [3,3,17,128]*


    Now you should be able to run:

    ```shell
    python minigui/serve.py --model=$MODEL_DIR/$MODEL.converted --board_size=$BOARD_SIZE --port=5001
    ```

1. Make sure the command at the top of `serve.py` actually runs and prints
   `GTP engine ready`; if not, something is wrong with the rest of the minigo
   setup, like virtualenv or similar.

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

1. Before running, you'll need to freeze a model so the C++ job can consume it.

    ```shell
    python main.py freeze-graph $MODEL_DIR/$MODEL.converted
    ```

1. Here's an example, note that we specify the board size when compiling the
   C++ engine binary, not when running the Minigui server:

    ```shell
    bazel build --define=board_size=19 cc:main
    python minigui/serve.py --port=8888 --model=$MODEL_DIR/$MODEL.converted.pb --engine=cc
    ```
