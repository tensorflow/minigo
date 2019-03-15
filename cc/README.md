# C++ Minigo

The current self-play and training pipeline is run using a C++ implementation
of Minigo.

## Set up

The C++ Minigo port uses __version 0.17.2__ of the [Bazel](https://bazel.build/)
build system. We have experienced build issues with the latest version of Bazel,
so for now we recommend installing
[bazel-0.17.2-installer-linux-x86\_64.sh](https://github.com/bazelbuild/bazel/releases).

Minigo++ depends on the TensorFlow C++ libraries, but we have not yet set up
Bazel WORKSPACE and BUILD rules to automatically download and configure
TensorFlow so (for now at least) you must perform a manual step to build the
library.  This depends on `zip`, so be sure that package is installed first:

```shell
sudo apt-get install zip
./cc/configure_tensorflow.sh
```

If you want to compile for CPU and not GPU, then execute the following instead:

```shell
sudo apt-get install zip
TF_NEED_CUDA=0 ./cc/configure_tensorflow.sh
```

This will automatically perform the first steps of
[Installing TensorFlow from Sources](https://www.tensorflow.org/install/install_sources)
but instead of installing the TensorFlow package, it extracts the generated C++
headers into the `cc/tensorflow` subdirectory of the repo. The script then
builds the required TensorFlow shared libraries and copies them to the same
directory. The tensorflow cc\_library build target in `cc/BUILD` pulls these
header and library files together into a format the Bazel understands how to
link against.

Although Minigo uses TensorFlow as its inference engine by default, other
engines can be used (e.g. Cloud TPU, TensorRT, TensorFlow Lite). See the
Inferences engines section below for more details.

## Getting a model

The TensorFlow models can found on our public Google Cloud Storage bucket. A
good model to start with is 000990-cormorant from the v15 run. The C++ engine
requires the frozen model in a `GraphDef` proto format, which is the `.pb` file.
You can copy it locally as follows:

```shell
mkdir -p saved_models
gsutil cp gs://minigo-pub/v15-19x19/models/000990-cormorant.pb saved_models/
```

Minigo can also read models directly from Google Cloud Storage but it doesn't
currently perform any local caching, so you're better off copying the model
locally once instead of copying from GCS every time.

## Binaries

C++ Minigo is made up of several different binaries. All binaries can be run
with `--helpshort`, which will display the full list of command line arguments.


#### cc:simple\_example

A very simple example of how to perform self-play using the Minigo engine. Plays
a single game using a fixed number of readouts.

```shell
bazel build -c opt cc:simple_example
bazel-bin/cc/simple_example \
  --model=tf,saved_models/000990-cormorant.pb \
  --num_readouts=160
```

#### cc:selfplay

The self-play binary used in our training pipeline. Has a lot more functionality
that the simple example, including:

 - Play multiple games in parallel, batching their inferences together for
   better GPU/TPU utilization.
 - Automatic loading of the latest trained model.
 - Write SGF games & TensorFlow training examples to Cloud Storage or Cloud
   BigTable.
 - Flag file support with automatic reloading. This is used among other things
   to dynamically adjust the resign threshold to minimize the number of bad
   resigns.


```shell
bazel build -c opt cc:selfplay
bazel-bin/cc/selfplay \
  --model=saved_models/000990-cormorant.pb \
  --num_readouts=160 \
  --parallel_games=1 \
  --output_dir=data/selfplay \
  --holdout_dir=data/holdout \
  --sgf_dir=sgf
```

#### cc:eval

Evaluates the performance of two models by playing them against each other over
multiple games in parallel.

```shell
bazel build -c opt cc:eval
bazel-bin/cc/eval \
  --model=saved_models/000990-cormorant.pb \
  --model_two=saved_models/000990-cormorant.pb \
  --num_readouts=160 \
  --parallel_games=32 \
  --sgf_dir=sgf
```

#### cc:gtp

Play using the GTP protocol. This is also the binary we recommend using as a
backend for Minigui (see `minigui/README.md`).

```shell
bazel build -c opt cc:gtp
bazel-bin/cc/gtp \
  --model=saved_models/000990-cormorant.pb \
  --num_readouts=160
```

#### cc:puzzle

Loads all SGF files found in the given `sgf_dir` and tries to predict the move
made at each position in each game. After all games are processed, prints
summary stats to about how many moves were correctly predicted.

```shell
bazel build -c opt cc:puzzle
bazel-bin/cc/puzzle \
  --model=saved_models/000990-cormorant.pb \
  --num_readouts=160 \
  --sgf_dir=puzzle_sgf
```

## Running the unit tests

Minigo's C++ unit tests operate on both 9x9 and 19x19, and some tests are only
enabled for a particular board size. Consequently, you must run the tests
twice: once for 9x9 boards and once for 19x19 boards.

```shell
bazel test --define=board_size=9 cc/...  &&  bazel test cc/...
```

Note that Minigo is compiled for a 19x19 board by default, which explains the
lack of a `--define=board_size=19` in the second `bazel test` invocation.

## Running with AddressSanitizer

Bazel supports building with AddressSanitizer to check for C++ memory errors:

```shell
bazel build cc:selfplay \
  --copt=-fsanitize=address \
  --linkopt=-fsanitize=address \
  --copt=-fno-omit-frame-pointer \
  --copt=-O1
```

## Inference engines

C++ Minigo currently supports multiple separate engines for performing
inference. Which engines are compiled into the C++ binaries are controlled by
passing Bazel `--define` arguments at compile time. The inference engine to
use is specified as part of the `--model` or `--model_two` command line
arguments:

 - **tf**: peforms inference using the TensorFlow libraries built by
   `cc/configure_tensorflow.sh`. Compiled & used as the inference by default,
   disable the engine with `--define=tf=0`. Use by passing `--model=tf,$PATH`,
   where `$PATH` is the path to a frozen TensorFlow `GraphDef` proto (as
   generated by `freeze_graph.py`).
 - **lite**: performs inference using TensorFlow Lite, which runs in software on
   the CPU.
   Compile by passing `--define=lite=1` to `bazel build`.
   Use by passing `--model=lite,$PATH`, where `$PATH` is the path to a Toco-
   optimized TFLite flat buffer (see below).
 - **tpu**: perform inference on a Cloud TPU. Your code must run on a Cloud
   TPU-equipped VM for this to work.
   Compile by passing `--define=tpu=1` to `bazel build`.
   Use by passed `--model=tpu:$TPU_NAME,$PATH`, where `$TPU_NAME` is the name
   of a Cloud TPU (e.g. `grpc://10.240.2.10:8470`) and `$PATH` is the path to a
   frozen TensorFlow `GraphDef` proto (as generated by `freeze_graph.py`).
 - **trt**: uses NVIDIA TensorRT.
   Compile by passing `--define=trt=1` to `bazel build`.
   Use by passing `--model=trt,$PATH`, where `$PATH` is a TensorRT model.
 - **fake**: a simple fake that is only useful in so much as you can use it to
   test that the tree search code compiles without also having to compile a
   full inference engine. Enabled by default. There's no way to disable the
   fake inference engine because this guaratees that there's always an engine
   available (even if it's a useless one).
   Use by passing `--model=fake`.
 - **random**: a model that returns random samples from a normal distribution,
   which can be useful for bootstrapping the reinforcement learning pipeline.
   Use by passing `--model=random:$SEED,$POLICY_STD_DEV:$VALUE_STD_DEV`, where
   `$SEED` is a random seed (set to `0` to choose one based on the current
   time), `$POLICY_STD_DEV` is the standard deviation of the distribution of
   policy samples (`0.4` is a reasonable choice) and `$VALUE_STD_DEV` is the
   standard deviation for the distribution of value samples (again, `0.4` is
   a reasonable choice). That was a bit of a long-winded explanation, so just
   try `--model=random:0,0.4:0.4` to start with.

## Compiling a TensorFlow Lite model

First, run a frozen graph through Toco, the TensorFlow optimizing compiler:

```
BATCH_SIZE=8
./cc/tensorflow/toco \
  --input_file=saved_models/000256-opossum.pb \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --output_file=saved_models/000256-opossum.tflite \
  --inference_type=FLOAT \
  --input_type=FLOAT \
  --input_arrays=pos_tensor \
  --output_arrays=policy_output,value_output \
  --input_shapes=8,19,19,17
```

## Cloud TPU

Minigo supports running inference on Cloud TPU.
Build `//cc:selfplay` with `--define=tpu=1` and run with `--model=tpu,$PATH`
(see above).

To freeze a model into a GraphDef proto that can be run on Cloud TPU, use
`freeze_graph.py`:

```
python freeze_graph.py \
  --model_path=$MODEL_PATH \
  --use_tpu=true \
  --tpu_name=$TPU_NAME \
  --num_tpu_cores=8
```

Where `$MODEL_PATH` is the path to your model (either a local file or one on
Google Cloud Storage), and `$TPU_NAME` is the gRPC name of your TPU, e.g.
`grpc://10.240.2.10:8470`. This can be found from the output of
`gcloud beta compute tpus list`.

This command **must** be run from a Cloud TPU-ready GCE VM.

This invocation to `freeze_graph.py` will replicate the model 8 times so that
it can run on all eight cores of a Cloud TPU. To take advantage of this
parallelism when running selfplay, `virtual_losses * parallel_games` must be at
least 8, ideally 128 or higher.

## 9x9 boards

The C++ Minigo implementation requires that the board size be defined at compile
time, using the `MINIGO_BOARD_SIZE` preprocessor define. This allows us to
significantly reduce the number of heap allocations performed. The build scripts
are configured to compile with `MINIGO_BOARD_SIZE=19` by default. To compile a
version that works with a 9x9 board, invoke Bazel with `--define=board_size=9`.

## Bigtable

Minigo supports writing eval and selfplay results to
[Bigtable](https://cloud.google.com/bigtable/).

Build with `--define=bt=1` and run with
`--output_bigtable=<PROJECT>,<INSTANCE>,<TABLE>`.

For eval this would look something like
```
bazel-bin/cc/eval \
  --model=<MODEL_1> \
  --model_two=<MODEL_2> \
  --parallel_games=4 \
  --num_readouts=32
  --sgf_dir=sgf \
  --output_bigtable=<PROJECT>,minigo-instance,games
```

See number of eval games with
```
cbt -project=<PROJECT> -instance=minigo-instance \
  read games columns="metadata:eval_game_counter"
```
See eval results with
```
cbt -project=<PROJECT> -instance=minigo-instance \
  read games prefix="e_"
```

## Style guide

The C++ code follows
[Google's C++ style guide](https://github.com/google/styleguide)
and we use cpplint to delint.

