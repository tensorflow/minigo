Minigo: A minimalist Go engine modeled after AlphaGo Zero, built on MuGo
==================================================

This is a pure Python implementation of a neural-network based Go AI, using
TensorFlow. While inspired by Deepmind's AlphaGo algorithm, this project is not
a Deepmind project.

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

2. Reproduce the methods of the original Deepmind AlphaGo papers as faithfully
   as possible, through an open-source implementation and open-source pipeline
   tools.

3. Provide our data, results, and discoveries in the open to benefit the Go,
   machine learning, and Kubernetes communities.

An explicit non-goal of the project is to produce a competitive Go program that
establishes itself as the top Go AI. Instead, we strive for a readable,
understandable implementation that can benefit the community, even if that
means our implementation is not as fast or efficient as possible.

Getting Started
===============

This project assumes you're using:

- virtualenv / virtualenvwrapper
- Python 3.6+

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
First set up and enter your virtualenv. Then start by installing TensorFlow and
the dependencies:

```
pip3 install -r requirements.txt
```

If you wish to run on GPU you must install CUDA 8.0 or later (see TensorFlow
documentation).

If you don't want to run on GPU or don't have one, you can downgrade:

```
pip3 uninstall tensorflow-gpu
pip3 install tensorflow
```

Or just install the CPU requirements:

```
pip3 install -r requirements-cpu.txt
```


Running unit tests
------------------
```
python3 -m unittest discover tests
```


Basics
======

All commands are compatible with either Google Cloud Storage as a remote file
system, or your local file system. The examples here use GCS, but local file
paths will work just as well.

To use GCS, set the BUCKET_NAME variable and authenticate. Otherwise, all
commands fetching files from GCS will hang.

```bash
export BUCKET_NAME=foobar;
gcloud auth application-default login
gsutil ls gs://minigo/models | tail -3
```

Which might look like

```
gs://minigo/models/000193-trusty.data-00000-of-00001
gs://minigo/models/000193-trusty.index
gs://minigo/models/000193-trusty.meta
```

You'll need to copy them to your local disk:

```shell
MINIGO_MODELS=$HOME/minigo-models
mkdir -p $MINIGO_MODELS
gsutil ls gs://minigo/models | tail -3 | xargs -I{} gsutil cp "{}" $MINIGO_MODELS
```

Selfplay
--------

To watch Minigo play itself:

```shell
python main.py selfplay --readouts $READOUTS -g 1 -v 2
```

where `READOUTS` is how many searches to make per move, and (-g 1) is how
many games to play simultaneously.  Timing information and statistics will be
printed at each move.  Setting verbosity (-v) to 3 or higher will print a board at each move.

Playing Against Minigo
----------------------

Minigo uses the GTP protocol, and you can use any gtp-compliant program with it.
```
python3 main.py gtp gs://$BUCKET_NAME/models/000000-bootstrap -r READOUTS -v 3
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
gogui-twogtp -black 'python3 main.py gtp gs://$BUCKET_NAME/models/000000-bootstrap' -white 'gogui-display' -size 19 -komi 7.5 -verbose -auto
```

Another way to play via GTP is to play against GnuGo, while spectating the games
```
BLACK="gnugo --mode gtp"
WHITE="python3 main.py gtp path/to/model"
TWOGTP="gogui-twogtp -black \"$BLACK\" -white \"$WHITE\" -games 10 \
  -size 19 -alternate -sgffile gnugo"
gogui -size 19 -program "$TWOGTP" -computer-both -auto
```

Another way to play via GTP is to connect to CGOS, the [Computer Go Online Server](http://yss-aya.com/cgos/). The CGOS server hosted by boardspace.net is actually abandoned; you'll want to connect to the CGOS server at yss-aya.com.

After configuring your cgos.config file, you can connect to CGOS with `cgosGtp -c cgos.config` and spectate your own game with `cgosView yss-aya.com 6819`

Reinforcement Learning
======================

The following sequence of commands will allow you to do one iteration of
reinforcement learning on 9x9. These are the basic commands used in the 
kubernetified version. You'll need GCS object admin permissions all steps.

Bootstrap
---------

This command creates a random model, which appears at .
`gs://$BUCKET_NAME/models/$MODEL_NAME(.index|.meta|.data-00000-of-00001)`

```bash
export MODEL_NAME=000000-bootstrap
python3 main.py bootstrap gs://$BUCKET_NAME/models/$MODEL_NAME -n $BOARD_SIZE
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
python3 main.py selfplay gs://$BUCKET_NAME/models/$MODEL_NAME \
  --readouts 10 \
  --games 8 \
  -v 3 -n 9 \
  --output-dir=gs://$BUCKET_NAME/data/selfplay/$MODEL_NAME/local_worker \
  --output-sgf=gs://$BUCKET_NAME/sgf/$MODEL_NAME/local_worker
```
(-n 9 makes it play 9x9 games)

Gather
------

This command takes multiple tfrecord.zz files (which will probably be KBs in size)
and shuffles them into tfrecord.zz files that are ~100 MB in size.

```
python3 main.py gather
```

Gathering is done according to model numbers, so that games generated by
one model stay together. The output will be in the directories

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

Training a new model
--------------------

This command finds the most recent 50 models' training chunks and trains
starting from the latest model weights and generates a new set of model weights.

Run the training job:
```
python3 main.py train gs://$BUCKET_NAME/data/training_chunks \
    gs://$BUCKET_NAME/models/000001-somename \
    --load-file=gs://$BUCKET_NAME/models/000000-bootstrap \
    --generation-num=1 \
    --logdir=path/to/tensorboard/logs \
    -n 9
```

The updated model weights will be saved at the end. (TODO: implement some sort
of local checkpointing based on global_step that will resume appropriately.)

Additionally, you can follow along with the training progress with TensorBoard - if you give each run a different name (`logs/my_training_run`, `logs/my_training_run2`), you can overlay the runs on top of each other.

```
tensorboard --logdir=path/to/tensorboard/logs/
```

Running Minigo on a Cluster
==============================

As you might notice, playing games is fairly slow.  One way to speed up playing
games is to run Minigo on many computers simultaneously.  Minigo was
originally trained via a pipeline running many selfplay-workers simultaneously.
The worker jobs are built into containers and run on a Kubernetes cluster,
hosted on the Google Cloud Platform (TODO: links for installing GCP SDK,
kubectl, etc.)

*NOTE* These commands will result in VMs being created and will result in
charges to your GCP account!  *Proceed with care!*

You'll want to install the following command line tools
  - [gcloud](https://cloud.google.com/sdk/downloads)
  - gsutil (via `gcloud components install gsutil`)
  - kubectl (via `gcloud components install kubectl`)
  - docker

In order for each step to work, you'll have to have the following permissions:
  - storage.bucket.(create, get, setIamPolicy) ("Storage Admin")
  - storage.objects.(create, delete, get, list, update) ("Storage Object Admin")
  - iam.serviceAccounts.create ("Service Account Admin")
  - iam.serviceAccountKeys.create ("Service Account Key Admin")
  - iam.serviceAccounts.actAs ("Service Account User")
  - resourcemanager.projects.setIamPolicy ("Project IAM Admin")
  - container.clusters.create ("Kubernetes Engine Cluster Admin")
  - container.secrets.create ("Kubernetes Engine Developer")

You'll also want to activate Kubernetes Cluster on your GCP account (just visit
the Kubernetes Engine page and it will automatically activate.)

Brief Overview of Pipeline
--------------------------

A Kubernetes cluster instantiates _nodes_ on a _node pool_, which specifies
what types of host machines are available.  _Jobs_ can be run on the cluster by
specifying details such as what containers to run, how many are needed, what
arguments they take, etc.  _Pods_ are created to run individual containers --
the pods themselves run on the _nodes_.

In our case, we won't let kubernetes resize of our node pool dynamically, we'll
manually specify how many machines we want: this means kubernetes will leave
machines running even if they're not doing anything!  So be sure to clean up
your clusters...

GCS for simple task signalling
------------------------------

The main way these jobs interact is the filesystem -- which is just a GCS
bucket mounted as a fuse filesystem inside their container.

The selfplay jobs will find the newest model in their directory of models and
play games with it, writing the games out to a different directory in the
bucket.

The training job will collect games from that directory and turn it into
chunks, which it will use to play a new model.


Bringing up a cluster
---------------------

0. Switch to the `cluster` directory
1. Set the common environment variables in `common` corresponding to your GCP project and bucket names.
2. Run `deploy`, which will:
  a. Create a bucket
  b. Create a service account
  c. Grant permissions
  d. Fetch the keys.
  If any of the above have already been done, the script will fail.  At a minimum, run step 'd' to create the keyfile.
3. Run `cluster-up` or (`cluster-up-gpu`), which will:
  a. Create a Google Container Engine cluster with some number of VMs
  b. Load its credentials locally
  c. Load those credentials into our `kubectl` environment, which will let us control the cluster from the command line.

    Creating the cluster might take a while... Once its done, you should be able to see something like this:

    ```
    $ kubectl get nodes
    NAME                                  STATUS    ROLES     AGE       VERSION
    gke-minigo-default-pool-b09dcf70-08rp   Ready     <none>    5m        v1.7.8-gke.0
    gke-minigo-default-pool-b09dcf70-0q5w   Ready     <none>    5m        v1.7.8-gke.0
    gke-minigo-default-pool-b09dcf70-1zmm   Ready     <none>    5m        v1.7.8-gke.0
    gke-minigo-default-pool-b09dcf70-50vm   Ready     <none>    5m        v1.7.8-gke.0
    ```

4. (Optional, GPU only).  If you've set up a GPU enabled cluster, you'll need to install the NVIDIA drivers on each of the nodes in your cluster that will have GPU workers.  This is accomplished by running:

```
kubectl apply -f gpu-provision-daemonset.yaml
```


5. Resizing your cluster.

  ```
  gcloud alpha container clusters resize $CLUSTER_NAME --zone=$ZONE --size=8
  ```

Create Docker image
-------------------

You will need a Docker image in order to initialize the pods.

First you need to update the param in the `Makefile`:

```
source common
sed -i "s/tensor-go/$PROJECT/" Makefile
```

Then `make` will produce and push the image!

CPU worker:
```
make image
make push
```

GPU worker:

```
make gpu-image
make gpu-push
```

Launching selfplay workers on a cluster
---------------------------------------

Now that our cluster is set up, lets check some values on our player job before deploying a bunch of copies:

In `cluster/player.yaml` (or `cluster/gpu-player.yaml`) check the 'parallelism'
field.  This is how many pods will try to run at once: if you can run multiple
pods per node (CPU nodes), you can set it accordingly.  For GPUs, which don't
share as well, limit the parallelism to the number of nodes available.

Now launch the job via the launcher.  (it just subs in the environment variable
for the bucket name, neat!)
```
source common
envsubst < player.yaml | kubectl apply -f -
```

Once you've done this, you can verify they're running via

```
kubectl get jobs
```

and get a list of pods with
```
kubectl get pods
```

Tail the logs of an instance:
```
kubectl logs -f <name of pod>
```

To kill the job,
```
envsubst < player.yaml | kubectl delete -f -
```

Preflight checks for a training run.
====================================


Setting up the selfplay cluster
-------------------------------

* Check your gcloud -- authorized?  Correct default zone settings?
* Check the project name, cluster name, & bucket name variables in the
  `cluster/common` script.  Did you change things?
  * If Yes: Grep for the original string.  Depending on what you changed, you may
    need to change the yaml files for the selfplay workers.
* Create the service account and bucket, if needed, by running `cluster/deploy`,
  or the relevant lines therein.
* Check the number of machines and machine types in the `cluster/cluster-up`
   script.
* Set up the cluster as above and start the nvidia driver installation daemonset
* While the nvidia drivers are getting installed on the fleet, check the
  various hyperparameters and operating parameters:
  * `dual_net.py`, check the `get_default_hyperparams` function
  * `player_wrapper.sh`, the invocation of `rl_loop.py selfplay` has the readout
    depth, game parallelism, resign threshold, etc.
  * `strategies.py`, check the move threshold for move 'temperature'
    (affects deterministic play), and the max game depth.
  * `mcts.py`, check the noise density and the tree branching factor (lol good
    luck)
* Seed the model directory with a randomly initialized model. (`python3
  rl_loop.py bootstrap /path/to/where/you/want/new/model`)
* If you're getting various tensorflow RestoreOp shape mismatches, this is
  often caused by mixing up 9x9 vs. 19x19 in the various system parts.

* Build your docker images with the latest version of the code, optionally
  bumping the version number in the Makefile.
* Don't forget to push the images!

* Now you can launch your job on the cluster -- check the parallelism in the
  spec! -- per the instructions above.  You should let the selfplay cluster
  finish up a bunch of games before you need to start running the training job,
  so now's a good time to make sure things are going well.

Useful things for the selfplay cluster
--------------------------------------

* Getting a list of the selfplay games ordered by most recent start
  ```
  kubectl get po --sort-by=.status.startTime
  ```

* Attaching to a running pod (to check e.g. cpu utilization, what actual code is
  in your container, etc)
  ```
  kubectlc exec -it <pod id> /bin/bash
  ```

* Monitoring how long it's taking the daemonset to install the nvidia driver on
  your nodes
  ```
  kubectl get no -w -o yaml | grep -E 'hostname:|nvidia-gpu'
  ```


Setting up logging via stackdriver, plus metrics, bla bla.


If you've run rsync to collect a set of SGF files (cheatsheet: `python3
rl_loop.py smart-rsync --source-dir="gs://$BUCKET_NAME/sgf/" --from-model-num 0
--game-dir=sgf/`), here are some handy
bashisms to run on them:

* Find the proportion of games won by one color:
  ```
  grep -m 1 "B+" **/*.sgf | wc -l
  ```
  or e.g. "B+R", etc to search for how many by resign etc.

* A histogram of game lengths (uses the 'ministat' package)
  ```
  find . -name "*.sgf" -exec /bin/sh -c 'tr -cd \; < {} | wc -c' \; | ministats
  ```

* Get output of the most frequent first moves
  ```
  grep -oh -m 1 '^;B\[[a-s]*\]' **/*.sgf | sort | uniq -c | sort -n
  ```

* Distribution of game-winning margin (ministat, again):
  ```
  find . -name "*.sgf" -exec /bin/sh -c 'grep -o -m 1 "W+[[:digit:]]*" < {} | cut -c3-'
  \; | ministat
  ```


etc...


