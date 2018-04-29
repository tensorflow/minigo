Minigo: A minimalist Go engine modeled after AlphaGo Zero, built on MuGo
==================================================

[Test Dashboard](https://k8s-testgrid.appspot.com/sig-big-data#tf-minigo-presubmit)

This is a pure Python implementation of a neural-network based Go AI, using
TensorFlow. While inspired by DeepMind's AlphaGo algorithm, this project is not
a DeepMind project nor is it affiliated with the official AlphaGo project.

### This is NOT an official version of AlphaGo ###

Repeat, *this is not the official AlphaGo program by DeepMind*.  This is an
independent effort by Go enthusiasts to replicate the results of the AlphaGo
Zero paper ("Mastering the Game of Go without Human Knowledge," *Nature*), with
some resources generously made available by Google.

Minigo is based off of Brian Lee's "MuGo" -- a pure Python implementation of the
first AlphaGo paper ["Mastering the Game of Go with Deep Neural Networks and
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

- GPU: `pip3 install "tensorflow-gpu>=1.7,<1.8"`.
  - *Note*: You must install [CUDA
    9.0].(https://developer.nvidia.com/cuda-90-download-archive) for Tensorflow
    1.5+.
- CPU: `pip3 install "tensorflow>=1.7,<1.8"`.

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

Automated Tests
----------------

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
```bash
export BUCKET_NAME=your_bucket;
gcloud auth application-default login
gsutil ls gs://$BUCKET_NAME/models | tail -3
```

Which might look like:

```
gs://$BUCKET_NAME/models/000193-trusty.data-00000-of-00001
gs://$BUCKET_NAME/models/000193-trusty.index
gs://$BUCKET_NAME/models/000193-trusty.meta
```

These three files comprise the model, and commands that take a model as an
argument usually need the path to the model basename, e.g.
`gs://$BUCKET_NAME/models/000193-trusty`

You'll need to copy them to your local disk.  This fragment copies the latest
model to the directory specified by `MINIGO_MODELS`

```shell
MINIGO_MODELS=$HOME/minigo-models
mkdir -p $MINIGO_MODELS
gsutil ls gs://$BUCKET_NAME/models | tail -3 | xargs -I{} gsutil cp "{}" $MINIGO_MODELS
```

Selfplay
--------
To watch Minigo play a game, you need to specify a model. Here's an example
to play using the latest model in your bucket

```shell
python rl_loop.py selfplay --num_readouts=$READOUTS -v 2
```
where `READOUTS` is how many searches to make per move.  Timing information and
statistics will be printed at each move.  Setting verbosity (-v) to 3 or higher
will print a board at each move.

Playing Against Minigo
----------------------

Minigo uses the
[GTP Protocol](http://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html),
and you can use any gtp-compliant program with it.

```
# Latest model should look like: /path/to/models/000123-something
LATEST_MODEL=$(ls -d $MINIGO_MODELS/* | tail -1 | cut -f 1 -d '.')
BOARD_SIZE=19 python3 main.py gtp -l $LATEST_MODEL --num_readouts=$READOUTS -v 3
```

(If no model is provided, it will initialize one with random values)

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
gogui-twogtp -black 'python3 main.py gtp -l gs://$BUCKET_NAME/models/000000-bootstrap' -white 'gogui-display' -size 19 -komi 7.5 -verbose -auto
```

Another way to play via GTP is to watch it play against GnuGo, while spectating the games
```
BLACK="gnugo --mode gtp"
WHITE="python3 main.py gtp -l path/to/model"
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
 - gather: groups games played with the same model into larger files of
   tfexamples.
 - train: trains a new model with the selfplay results from the most recent N
   generations.

Training works via tf.Estimator; a local directory keeps track of training
progress, and the latest checkpoint is periodically exported to GCS, where it
gets picked up by selfplay workers.

Bootstrap
---------

This command initializes your working directory for the trainer and a random
model. This random model is also exported to `--model-save-path` so that 
selfplay can immediately start playing with this random model.

If these directories don't exist, bootstrap will create them for you.

```bash
export MODEL_NAME=000000-bootstrap
python3 main.py bootstrap \
  --working-dir=estimator_working_dir
  --model-save-path="gs://$BUCKET_NAME/models/$MODEL_NAME"
```

Self-play
---------

This command starts self-playing, outputting its raw game data in a
tensorflow-compatible format as well as in SGF form in the directories

```
gs://$BUCKET_NAME/data/selfplay/$MODEL_NAME/local_worker/*.tfrecord.zz
gs://$BUCKET_NAME/sgf/$MODEL_NAME/local_worker/*.sgf
```

```bash
BOARD_SIZE=19 python3 main.py selfplay gs://$BUCKET_NAME/models/$MODEL_NAME \
  --num_readouts 10 \
  -v 3 \
  --output-dir=gs://$BUCKET_NAME/data/selfplay/$MODEL_NAME/local_worker \
  --output-sgf=gs://$BUCKET_NAME/sgf/$MODEL_NAME/local_worker
```

Gather
------

```
python3 main.py gather
```

This command takes multiple tfrecord.zz files (which will probably be KBs in size)
and shuffles them into tfrecord.zz files that are ~100 MB in size.

Gathering is done according to model numbers, so that games generated by
one model stay together.  By default, [rl_loop.py](rl_loop.py) will use directories
specified by the environment variable `BUCKET_NAME`, set at the top of
[rl_loop.py](rl_loop.py).

```
gs://$BUCKET_NAME/data/training_chunks/$MODEL_NAME-{chunk_number}.tfrecord.zz
```

The file `gs://$BUCKET_NAME/data/training_chunks/meta.txt`is used to keep track of
which games have been processed so far. (more about this needed)

```bash
python3 main.py gather \
  --input-directory=gs://$BUCKET_NAME/data/selfplay \
  --output-directory=gs://$BUCKET_NAME/data/training_chunks
```

Training
--------

This command finds the most recent 50 models' training chunks and trains a new
model, starting from the latest model weights.

Run the training job:
```
BOARD_SIZE=19 python3 main.py train\
  estimator_working_dir
  gs://$BUCKET_NAME/data/training_chunks \
  gs://$BUCKET_NAME/models/000001-somename \
  --generation-num=1 \
```

At the end of training, the latest checkpoint will be exported to the named
directory. Additionally, you can follow along with the training progress with
TensorBoard - if you point TensorBoard at the estimator working dir, it will
find the training log files and display them.

```
tensorboard --logdir=estimator_working_dir
```

Validation
----------

It can be useful to set aside some games to use as a 'validation set' for
tracking the model overfitting.  One way to do this is with the `validate`
command.

### Validating on holdout data

By default, Minigo will hold out 5% of selfplay games for validation, and write
them to `gs://$BUCKET_NAME/data/holdout/<model_name>`.  This can be changed by
adjusting the `holdout-pct` flag on the `selfplay` command.

With this setup, `python rl_loop.py validate --logdir=estimator_working_dir --` will figure out
the most recent model, grab the holdout data from the fifty models prior to that
one, and calculate the validation error, writing the tensorboard logs to
`logdir`.


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

```
BOARD_SIZE=19 python main.py validate path/to/validation/files/ --load-file=/path/to/model
--logdir=path/to/tb/logs --num-steps=<number of positions to run validation on>
```

the `main.py validate` command will glob all the .tfrecord.zz files under the
directories given as positional arguments and compute the validation error for
`num_steps * TRAINING_BATCH_SIZE` positions from those files.

Running Minigo on a Kubernetes Cluster
==============================

See more at [cluster/README.md](https://github.com/tensorflow/minigo/tree/master/cluster/README.md)
