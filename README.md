Minigo: A minimalist Go engine modeled after AlphaGo Zero, built on MuGo
==================================================

This is an implementation of a neural-network based Go AI, using TensorFlow.
While inspired by DeepMind's AlphaGo algorithm, this project is not
a DeepMind project nor is it affiliated with the official AlphaGo project.

### This is NOT an official version of AlphaGo ###

Repeat, *this is not the official AlphaGo program by DeepMind*.  This is an
independent effort by Go enthusiasts to replicate the results of the AlphaGo
Zero paper ("Mastering the Game of Go without Human Knowledge," *Nature*), with
some resources generously made available by Google.

Minigo is based off of Brian Lee's "[MuGo](https://github.com/brilee/MuGo)"
-- a pure Python implementation of the first AlphaGo paper
["Mastering the Game of Go with Deep Neural Networks and
Tree Search"](https://www.nature.com/articles/nature16961) published in
*Nature*. This implementation adds features and architecture changes present in
the more recent AlphaGo Zero paper, ["Mastering the Game of Go without Human
Knowledge"](https://www.nature.com/articles/nature24270). More recently, this
architecture was extended for Chess and Shogi in ["Mastering Chess and Shogi by
Self-Play with a General Reinforcement Learning
Algorithm"](https://arxiv.org/abs/1712.01815).  These papers will often be
abridged in Minigo documentation as *AG* (for AlphaGo), *AGZ* (for AlphaGo
Zero), and *AZ* (for AlphaZero) respectively.


Goals of the Project
==================================================

1. Provide a clear set of learning examples using Tensorflow, Kubernetes, and
   Google Cloud Platform for establishing Reinforcement Learning pipelines on
   various hardware accelerators.

2. Reproduce the methods of the original DeepMind AlphaGo papers as faithfully
   as possible, through an open-source implementation and open-source pipeline
   tools.

3. Provide our data, results, and discoveries in the open to benefit the Go,
   machine learning, and Kubernetes communities.

An explicit non-goal of the project is to produce a competitive Go program that
establishes itself as the top Go AI. Instead, we strive for a readable,
understandable implementation that can benefit the community, even if that
means our implementation is not as fast or efficient as possible.

While this product might produce such a strong model, we hope to focus on the
process.  Remember, getting there is half the fun. :)

We hope this project is an accessible way for interested developers to have
access to a strong Go model with an easy-to-understand platform of python code
available for extension, adaptation, etc.

If you'd like to read about our experiences training models, see [RESULTS.md](RESULTS.md).

To see our guidelines for contributing, see [CONTRIBUTING.md](CONTRIBUTING.md).

Getting Started
===============

This project assumes you have the following:

- virtualenv / virtualenvwrapper
- Python 3.5+
- [Docker](https://docs.docker.com/install/)
- [Cloud SDK](https://cloud.google.com/sdk/downloads)
- Bazel v0.11 or greater

The [Hitchhiker's guide to
python](http://docs.python-guide.org/en/latest/dev/virtualenvs/) has a good
intro to python development and virtualenv usage. The instructions after this
point haven't been tested in environments that are not using virtualenv.

```shell
pip3 install virtualenv
pip3 install virtualenvwrapper
```

Install TensorFlow
------------------
First set up and enter your virtualenv and then the shared requirements:

```
pip3 install -r requirements.txt
```

Then, you'll need to choose to install the GPU or CPU tensorflow requirements:

- GPU: `pip3 install "tensorflow-gpu>=1.11,<1.12"`.
  - *Note*: You must install [CUDA
    9.0](https://developer.nvidia.com/cuda-90-download-archive). for Tensorflow
    1.5+.
- CPU: `pip3 install "tensorflow>=1.11,<1.12"`.

Setting up the Environment
--------------------------

You may want to use a cloud project for resources. If so set:

```shell
PROJECT=foo-project
```

Then, running

```shell
source cluster/common.sh
```

will set up other environment variables defaults.

Running unit tests
------------------
```
./test.sh
```

To run individual modules

```
BOARD_SIZE=9 python3 tests/run_tests.py test_go
BOARD_SIZE=19 python3 tests/run_tests.py test_mcts
```

Automated Tests
----------------

[Test Dashboard](https://k8s-testgrid.appspot.com/sig-big-data#tf-minigo-presubmit)

To automatically test PRs, Minigo uses
[Prow](https://github.com/kubernetes/test-infra/tree/master/prow), which is a
test framework created by the Kubernetes team for testing changes in a hermetic
environment. We use prow for running unit tests, linting our code, and
launching our test Minigo Kubernetes clusters.

You can see the status of our automated tests by looking at the Prow and
Testgrid UIs:

- Testgrid (Test Results Dashboard): https://k8s-testgrid.appspot.com/sig-big-data
- Prow (Test-runner dashboard): https://prow.k8s.io/?repo=tensorflow%2Fminigo

Basics
======

All commands are compatible with either Google Cloud Storage as a remote file
system, or your local file system. The examples here use GCS, but local file
paths will work just as well.

To use GCS, set the `BUCKET_NAME` variable and authenticate via `gcloud login`.
Otherwise, all commands fetching files from GCS will hang.

For instance, this would set a bucket, authenticate, and then look for the most
recent model.

```shell
# When you first start we recommend using our minigo-pub bucket.
# Later you can setup your own bucket and store data there.
export BUCKET_NAME=minigo-pub/v9-19x19
gcloud auth application-default login
gsutil ls gs://$BUCKET_NAME/models | tail -4
```

Which might look like:

```
gs://$BUCKET_NAME/models/000737-fury.data-00000-of-00001
gs://$BUCKET_NAME/models/000737-fury.index
gs://$BUCKET_NAME/models/000737-fury.meta
gs://$BUCKET_NAME/models/000737-fury.pb
```

These four files comprise the model. Commands that take a model as an
argument usually need the path to the model basename, e.g.
`gs://$BUCKET_NAME/models/000737-fury`

You'll need to copy them to your local disk.  This fragment copies the files
associated with `$MODEL_NAME` to the directory specified by `MINIGO_MODELS`:

```shell
MODEL_NAME=000737-fury
MINIGO_MODELS=$HOME/minigo-models
mkdir -p $MINIGO_MODELS/models
gsutil ls gs://$BUCKET_NAME/models/$MODEL_NAME.* | \
       gsutil cp -I $MINIGO_MODELS/models
```

Selfplay
--------
To watch Minigo play a game, you need to specify a model. Here's an example
to play using the latest model in your bucket

```shell
python3 selfplay.py \
  --verbose=2 \
  --num_readouts=400 \
  --load_file=$MINIGO_MODELS/models/$MODEL_NAME
```

where `READOUTS` is how many searches to make per move.  Timing information and
statistics will be printed at each move.  Setting verbosity to 3 or
higher will print a board at each move.

Playing Against Minigo
----------------------

Minigo uses the
[GTP Protocol](http://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html),
and you can use any gtp-compliant program with it.

```shell
# Latest model should look like: /path/to/models/000123-something
LATEST_MODEL=$(ls -d $MINIGO_MODELS/* | tail -1 | cut -f 1 -d '.')
python3 gtp.py --load_file=$LATEST_MODEL --num_readouts=$READOUTS --verbose=3
```

After some loading messages, it will display `GTP engine ready`, at which point
it can receive commands.  GTP cheatsheet:

```
genmove [color]             # Asks the engine to generate a move for a side
play [color] [coordinate]   # Tells the engine that a move should be played for `color` at `coordinate`
showboard                   # Asks the engine to print the board.
```

One way to play via GTP is to use gogui-display (which implements a UI that
speaks GTP.) You can download the gogui set of tools at
[http://gogui.sourceforge.net/](http://gogui.sourceforge.net/). See also
[documentation on interesting ways to use
GTP](http://gogui.sourceforge.net/doc/reference-twogtp.html).

```shell
gogui-twogtp -black 'python3 gtp.py --load_file=$LATEST_MODEL' -white 'gogui-display' -size 19 -komi 7.5 -verbose -auto
```

Another way to play via GTP is to watch it play against GnuGo, while
spectating the games:

```shell
BLACK="gnugo --mode gtp"
WHITE="python3 gtp.py --load_file=$LATEST_MODEL"
TWOGTP="gogui-twogtp -black \"$BLACK\" -white \"$WHITE\" -games 10 \
  -size 19 -alternate -sgffile gnugo"
gogui -size 19 -program "$TWOGTP" -computer-both -auto
```

Training Minigo
======================

Overview
--------

The following sequence of commands will allow you to do one iteration of
reinforcement learning on 9x9. These are the basic commands used to produce the
models and games referenced above.

The commands are

 - bootstrap: initializes a random model
 - selfplay: plays games with the latest model, producing data used for training
 - train: trains a new model with the selfplay results from the most recent N
   generations.

Training works via tf.Estimator; a working directory manages checkpoints and
training logs, and the latest checkpoint is periodically exported to GCS, where
it gets picked up by selfplay workers.

Configuration for things like "where do debug SGFs get written", "where does
training data get written", "where do the latest models get published" are
managed by the helper scripts in the rl_loop directory. Those helper scripts
execute the same commands as demonstrated below. Configuration for things like
"what size network is being used?" or "how many readouts during selfplay" can
be passed in as flags. The mask_flags.py utility helps ensure all parts of the
pipeline are using the same network configuration.

All local paths in the examples can be replaced with `gs://` GCS paths, and the
Kubernetes-orchestrated version of the reinforcement learning loop uses GCS.

Bootstrap
---------

This command initializes your working directory for the trainer and a random
model. This random model is also exported to `--model-save-path` so that
selfplay can immediately start playing with this random model.

If these directories don't exist, bootstrap will create them for you.

```shell
export MODEL_NAME=000000-bootstrap
python3 bootstrap.py \
  --work_dir=estimator_working_dir \
  --export_path=outputs/models/$MODEL_NAME
```

Self-play
---------

This command starts self-playing, outputting its raw game data as tf.Examples
as well as in SGF form in the directories.


```shell
python3 selfplay.py \
  --load_file=outputs/models/$MODEL_NAME \
  --num_readouts 10 \
  --verbose 3 \
  --selfplay_dir=outputs/data/selfplay \
  --holdout_dir=outputs/data/holdout \
  --sgf_dir=outputs/sgf
```

Training
--------

This command takes a directory of tf.Example files from selfplay and trains a
new model, starting from the latest model weights in the `estimator_working_dir`
parameter.

Run the training job:

```shell
python3 train.py \
  outputs/data/selfplay/* \
  --work_dir=estimator_working_dir \
  --export_path=outputs/models/000001-first_generation
```

At the end of training, the latest checkpoint will be exported to.
Additionally, you can follow along with the training progress with TensorBoard.
If you point TensorBoard at the estimator working directory, it will find the
training log files and display them.

```shell
tensorboard --logdir=estimator_working_dir
```

Validation
----------

It can be useful to set aside some games to use as a 'validation set' for
tracking the model overfitting.  One way to do this is with the `validate`
command.

### Validating on holdout data

By default, MiniGo will hold out 5% of selfplay games for validation. This can
be changed by adjusting the `holdout_pct` flag on the `selfplay` command.

With this setup, `rl_loop/train_and_validate.py` will validate on the same
window of games that were used to train, writing TensorBoard logs to the
estimator working directory.

### Validating on a different set of data

This might be useful if you have some known set of 'good data' to test your
network against, e.g., a set of pro games.
Assuming you've got a set of .sgfs with the proper komi & boardsizes, you'll
want to preprocess them into the .tfrecord files, by running something similar
to

```python
import preprocessing
filenames = [generate a list of filenames here]
for f in filenames:
    try:
        preprocessing.make_dataset_from_sgf(f, f.replace(".sgf", ".tfrecord.zz"))
    except:
        print(f)
```

Once you've collected all the files in a directory, producing validation is as
easy as

```shell
python3 validate.py \
  validation_files/ \
  --work_dir=estimator_working_dir \
  --validation_name=pro_dataset
```

The validate.py will glob all the .tfrecord.zz files under the
directories given as positional arguments and compute the validation error
for the positions from those files.

Running Minigo on a Kubernetes Cluster
==============================

See more at [cluster/README.md](https://github.com/tensorflow/minigo/tree/master/cluster/README.md)
