# Preliminary
This repo is a staging ground for the new [MLPerf](http://mlperf.org) reinforcement model.
Eventually this code will replace the code in
[this directory](http://github.com/mlperf/training/tree/master/reinforcement/tensorflow/minigo).

# 1. Problem
This task benchmarks on policy reinforcement learning for the 9x9 version of the boardgame Go.
The model plays games against itself and uses these games to improve play.

# 2. Directions
### Steps to configure machine
To setup the environment on Debian 9 Stretch with 8 GPUs, you can use the
following commands. This may vary on a different operating system or graphics
cards.

```
    # Install dependencies
    apt-get install -y python3 python3-pip rsync git wget pkg-config zip g++ zlib1g-dev unzip

    # Clone repository.
    git clone https://github.com/tensorflow/minigo
    # Note: This will eventually change to:
    # git clone http://github.com/mlperf/training

    cd minigo

    # Create a virtualenv (this step is optional but highly recommended).
    pip3 install virtualenv
    pip3 install virtualenvwrapper
    virtualenv -p /usr/bin/python3 --system-site-packages $HOME/.venvs/minigo
    source $HOME/.venvs/minigo/bin/activate

    # Install Python dependencies
    pip3 install -r requirements.txt

    # Install Python Tensorflow for GPU
    # (alternatively use "tensorflow==1.15.0" for CPU Tensorflow)
    pip3 install "tensorflow-gpu==1.15.0"

    # Install bazel
    BAZEL_VERSION=0.24.1
    wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
    chmod 755 bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
    sudo ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh

    # Compile TensorFlow C++ libraries
    ./cc/configure_tensorflow.sh

    # Download & extract bootstrap checkpoint.
    gsutil cp gs://minigo-pub/ml_perf/0.7/checkpoint.tar.gz .
    tar xfz checkpoint.tar.gz -C ml_perf/

    # Download and freeze the target model.
    mkdir -p ml_perf/target/
    gsutil cp gs://minigo-pub/ml_perf/0.7/target.* ml_perf/target/
    python3 freeze_graph.py --flagfile=ml_perf/flags/19/architecture.flags  --model_path=ml_perf/target/target

    # Set the benchmark output base directory.
    BASE_DIR=$(pwd)/ml_perf/results/$(date +%Y-%m-%d-%H-%M)

    # Bootstrap the training loop from the checkpoint.
    # This step also builds the required C++ binaries.
    # Bootstrapping is not considered part of the benchmark.
    ./ml_perf/scripts/init_from_checkpoint.sh \
        --board_size=19 \
        --base_dir=$BASE_DIR \
        --checkpoint_dir=ml_perf/checkpoints/mlperf07

    # Start the selfplay binaries running on the specified GPU devices.
    # This launches one selfplay binary per GPU.
    # This script can be run on multiple machines if desired, so long as all
    # machines have access to the $BASE_DIR.
    # In this particular example, the machine running the benchmark has 8 GPUs,
    # of whice devices 1-7 are used for selfplay and device 0 is used for
    # training.
    # The selfplay jobs will start, and wait for the training loop (started in
    # the next step) to produce the first model. The selfplay jobs run forever,
    # reloading new generations of models as the training job trains them.
    ./ml_perf/scripts/start_selfplay.sh \
         --board_size=19 \
         --base_dir=$BASE_DIR \
         --devices=1,2,3,4,5,6,7

    # Start the training loop. This is the point when benchmark timing starts.
    # The training loop produces trains the first model generated from the
    # bootstrap, then waits for selfplay to play games from the new model.
    # When enough games have been played (see min_games_per_iteration in
    # ml_perf/flags/19/train_loop.flags), a new model is trained using these
    # games. This process repeats for a preset number of iterations (again,
    # see ml_perf/flags/19/train_loop.flags).
    # The train scripts terminates the selfplay jobs on exit by writing an
    # "abort" file to the $BASE_DIR.
    ./ml_perf/scripts/train.sh \
         --board_size=19 \
         --base_dir=$BASE_DIR

    # Once the training loop has finished, run model evaluation to find the
    # first trained model that's better than the target.
    python3 ml_perf/eval_models.py \
         --start=0 \
         --flags_dir=ml_perf/flags/19 \
         --model_dir=$BASE_DIR/models/ \
         --target=ml_perf/target/target.minigo \
         --devices=0,1,2,3,4,5,6,7
```

### Tunable hyperparameters

The following flags are allowed to be modified by entrants:
Flags that don't directly affect convergence:
 - _all flags related to file paths & device IDs_
 - `bool_features`
 - `input_layout`
 - `summary_steps`
 - `cache_size_mb`
 - `num_read_threads`
 - `num_write_threads`
 - `output_threads`
 - `selfplay_threads`
 - `parallel_search`
 - `parallel_inference`
 - `concurrent_games_per_thread`
 - `validate`
 - `holdout_pct`

Flags that directly affect convergence:
 - `train_batch_size`
 - `lr_rates`
 - `lr_boundaries`

### Selfplay threading model

The selfplay C++ binary (`//cc:concurrent_selfplay`) has multiple flags that control its
threading:

- `selfplay_threads` controls the number of threads that play selfplay games.
- `concurrent_games_per_thread` controls how many games are played on each thread. All games
  on a selfplay thread have their inferences batched together and dispatched at the same time.
- `parallel_search` controls the size of the thread pool shared between all selfplay threads
  that are used to parallelise tree search. Since the selfplay thread also performs tree
  search, the thread pool size is `parallel_search - 1` and a value of `1` disables the thread
  pool entirely.

To get a better understanding of the threading model, we recommend running a trace of the selfplay
code as described below.

### Profiling

The selfplay C++ binary can output traces of the host CPU using Google's
[Tracing Framework](https://google.github.io/tracing-framework/). Compile the
`//cc:concurrent_selfplay` binary with tracing support by passing `--copt=-DWTF_ENABLE` to
`bazel build`. Then run `//cc:concurrent_selfplay`, passing `--wtf_trace=$TRACE_PATH` to specify
the trace output path. The trace is appended to peridically, so the `//cc:concurrent_selfplay`
binary can be killed after 20 or so seconds and the trace file written so far will be valid.

Install the Tracing Framework [Chrome extension](https://google.github.io/tracing-framework/) to
view the trace.

Note that the amount of CPU time spent performing tree search changes over the lifetime of the
benchmark: initially, the models tend to read very deeply, which takes more CPU time.

Here is an example of building and running selfplay with tracing enabled on a single GPU:

```
bazel build -c opt --copt=-O3 --define=tf=1 --copt=-DWTF_ENABLE cc:concurrent_selfplay
CUDA_VISIBLE_DEVICES=0 ./bazel-bin/cc/concurrent_selfplay \
    --flagfile=ml_perf/flags/19/selfplay.flags \
    --wtf_trace=$HOME/mlperf07.wtf-trace \
    --model=$BASE_DIR/models/000001.minigo
```

### Steps to download and verify data
Unlike other benchmarks, there is no data to download. All training data comes from games played
during benchmarking.

# 3. Model
### Publication/Attribution

This benchmark is based on the [Minigo](https://github.com/tensorflow/minigo) project,
which is and inspired by the work done by Deepmind with
["Mastering the Game of Go with Deep Neural Networks and Tree Search"](https://www.nature.com/articles/nature16961),
["Mastering the Game of Go without Human Knowledge"](https://www.nature.com/articles/nature24270), and
["Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"](https://arxiv.org/abs/1712.01815).

Minigo is built on top of Brian Lee's [MuGo](https://github.com/brilee/MuGo), a pure Python
implementation of the first AlphaGo paper.

Note that Minigo is an independent effort from AlphaGo.

### Reinforcement Setup
This benchmark includes both the environment and training for 19x19 Go. There are three primary
parts to this benchmark.

 - Selfplay: the *latest trained* model plays games with itself as both black and white to produce
   board positions for training.
 - Training: waits for selfplay to play a specified number of games with the latest model, then
   trains the next model generation, updating the neural network waits. Selfplay constantly monitors
   the training output directory and loads the new weights when as they are produced by the trainer.
 - Target Evaluation: The training loop runs for a preset number of iterations, producing a new
   model generation each time. Once finished, target evaluation relplays the each trained model
   until it finds the first one that is able to beat a target model in at least 50% of the games.
   The time from training start to when this generation was produced is taken as the benchmark
   execution time.

### Structure
This task has a non-trivial network structure, including a search tree. A good overview of the
structure can be found here: https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0.

### Weight and bias initialization and Loss Function
Network weights are initialized randomly. Initialization and loss are described here;
["Mastering the Game of Go with Deep Neural Networks and Tree Search"](https://www.nature.com/articles/nature16961)

### Optimizer
We use a MomentumOptimizer to train the network.

# 4. Quality
Due to the difficulty of training a highly proficient Go model, our quality metric and termination
criteria is based on winning against a model of only intermediate amateur strength.

### Quality metric
The quality of a model is measured as the number of games won in a playoff (alternating colors)
of 100 games against a previously trained model.

### Quality target
The quality target is to win 50% of the games.

### Quality Progression
Informally, we have observed that quality should improve roughly linearly with time.  We observed
roughly 0.5% improvement in quality per hour of runtime. An example of approximately how we've seen
quality progress over time:

```
    Approx. Hours to Quality
     1h           TDB%
     2h           TDB%
     4h           TDB%
     8h           TDB%
```

Note that quality does not necessarily monotonically increase.

### Target evaluation frequency
Target evaluation only needs to be performed for models which pass model evaluation.
