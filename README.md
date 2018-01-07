MuGo Zero: A minimalist Go engine modeled after AlphaGo Zero, built on MuGo
==================================================

This is a pure Python implementation of a neural-network based Go AI, using TensorFlow.

This is based off of Brian Lee's "MuGo" -- a pure Python implementation of the
first AlphaGo paper published in Nature, adapted and extended to the
architecture described in the second.

The goal of this project is to reproduce the results of the original paper
through an open-source implementation using the Google Cloud Platform and
open-source pipeline tools.  A secondary goal is to provide a clear set of
learning examples for establishing RL pipelines on various hardware
accelerators available in TensorFlow.

An explicit non-goal is to produce a competitive program, and the original
non-goal of MuGo -- "diving into the fiddly bits of optimizing" -- is preserved
in the service of the secondary goal above :)

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

Install Tensorflow
------------------
Start by installing Tensorflow and the dependencies, optionally into a
virtualenv if you so choose. First:

```
pip3 install -r requirements.txt
```

Then, you must either install CUDA 8.0 (See Tensorflow documentation).

If you don't have a GPU, you can downgrade

```
pip3 uninstall tensorflow-gpu
pip3 install tensorflow
```

or just install the cpu-requirements.

```
pip3 install -r requirements-cpu.txt
```


Running unit tests
------------------
```
python3 -m unittest discover tests
```


MuGo Zero: Selfplay, or GTP
===========================

You'll need to install
[gsutil](https://cloud.google.com/storage/docs/gsutil_install) before
proceeding.

If you just want to get MuGo Zero working, you can download the latest model at
from `gs://mugozero-v1/models`. The model comes in 3 parts; if you want to see
the latest, you can run

```shell
gsutil ls gs://mugozero-v1/models | tail -3
```

Which might look like

```
gs://mugozero-v1/models/000193-trusty.data-00000-of-00001
gs://mugozero-v1/models/000193-trusty.index
gs://mugozero-v1/models/000193-trusty.meta
```

You'll need to copy them to your local disk:

```shell
MUGO_MODELS=$HOME/mugo-models
gsutil ls gs://mugozero-v1/models | tail -3 | xargs -I{} gsutil cp "{}" $MUGO_MODELS
```

Or, you can build a random model with:

```
BUCKET_NAME=foobar python3 rl_loop.py bootstrap models/random_model
```


Selfplay
--------
To watch MuGo Zero play a game, you need to specify a model. Here's an example
to play using the latest model you've downloaded

```shell
MUGO_MODELS=$HOME/mugo-models

# How many searches to make per move
READOUTS=100

# How many games to play simultaneously
GAMES=5

# Latest model should look like: /path/to/models/000123-something
LATEST_MODEL=$(ls -d $MUGO_MODELS/* | tail -1 | cut -f 1 -d '.')

python3 main.py selfplay $LATEST_MODEL --readouts $READOUTS -g $GAMES -v 3
```

Timing information and statistics will be printed at each move.  Setting
verbosity to 3 or higher will print a board at each move. 


MuGo Zero uses the GTP protocol, and you can use any gtp-compliant program with it.

```
python3 main.py gtp $LATEST_MODEL -r $READOUTS -v 3
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
gogui-twogtp -black 'python3 main.py gtp policy --read-file=saved_models/20170718' -white 'gogui-display' -size 19 -komi 7.5 -verbose -auto
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

Training Mugo Zero
==================

Generate training chunks:

```
python3 main.py gather 
```

This will look in `data/selfplay` for games and write chunks to
`data/training_chunks`.  See main.py for description of the other arguments 

Run the training job:

```shell
python3 main.py train train training_data_dir \
    --load-file=path/to/model \
    --save-file=where/to/save/model \
    --logdir=path/to/tensorboard/logs
    --epochs=1
    --batch-size=64
```

(TODO)

...
As the network is trained, the current model will be saved at `--save-file`. If you reexecute the same command, the network will pick up training where it left off.

Additionally, you can follow along with the training progress with TensorBoard - if you give each run a different name (`logs/my_training_run`, `logs/my_training_run2`), you can overlay the runs on top of each other.

```
tensorboard --logdir=path/to/tensorboard/logs/
```

Running MuGo Zero on a Cluster
==============================

As you might notice, playing games is fairly slow.  One way to speed up playing
games is to run MuGo Zero on many computers simultaneously.  MuGo Zero was
originally trained via a pipeline running many selfplay-workers simultaneously.
The worker jobs are built into containers and run on a Kubernetes cluster,
hosted on the Google Cloud Platform (TODO: links for installing GCP SDK,
kubectl, etc.)

*NOTE* These commands will result in VMs being created and will result in
charges to your GCP account!  *Proceed with care!*

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

Now, what the cluster will do:

3 main tasks:
 - selfplay -- all but one of the nodes
 - training -- one node
 - evaluation -- dunno lol ¯\(ツ)/¯

The main way these jobs interact is the filesystem -- which is just a GCS
bucket mounted as a fuse filesystem inside their container.

The selfplay jobs will find the newest model in their directory of models and
play games with it, writing the games out to a different directory in the
bucket.

The training job will collect games from that directory and turn it into
chunks, which it will use to play a new model.

the evaluation job will collect the new model, evaluate it against the old one,
and bless it into the directory of models if it meets expectations.  


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
    gke-mugo-default-pool-b09dcf70-08rp   Ready     <none>    5m        v1.7.8-gke.0
    gke-mugo-default-pool-b09dcf70-0q5w   Ready     <none>    5m        v1.7.8-gke.0
    gke-mugo-default-pool-b09dcf70-1zmm   Ready     <none>    5m        v1.7.8-gke.0
    gke-mugo-default-pool-b09dcf70-50vm   Ready     <none>    5m        v1.7.8-gke.0
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
  * `go.py`, check the board size
  * `dual_net.py`, check the `get_default_hyperparams` function
  * `player_wrapper.sh`, the invocation of `main.py selfplay` has the readout
    depth, game parallelism, etc.
  * `strategies.py`, check the (currently static) resign threshold, the move
    threshold for move 'temperature' (affects deterministic play), and the max
    game depth.
  * `main.py`, check the default params on 'gather'
  * `mcts.py`, check the noise density and the tree branching factor (lol good
    luck)
  * `rl_loop.py`, check the cluster name, directory for tensorflow logs, and
    constants at the top of the file.
* Seed the model directory with a randomly initialized model. (`python3
  rl_loop.py bootstrap /path/to/where/you/want/new/model`)
* Copy the model to the GCS bucket. (`gsutil cp /path/to/model*
  gs://bucket/models/` etc)

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


If you've run rsync to collect a set of SGF files (cheatsheet: `python
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


