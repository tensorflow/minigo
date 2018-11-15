# Kubernetes Cluster Config for self-play containers / clusters

## Creating Docker images

You will need a Docker image in order to initialize the pods.


Make sure to source the common env vars first:

```bash
source cluster/common.sh
```

If you would like to override the GCR Project or image tag, you can set:

```shell
export PROJECT=my-project
export VERSION_TAG=0.12.34
```

Then `make` will produce and push the image!

CPU worker:

```shell
make py-image
make py-push
```

GPU worker (C++ engine):

```shell
make cc-image
make cc-push
```

TPU worker (C++ engine):

```shell
make tpu-image
make tpu-push
```

## Creating the Kubernetes CLuster

### Brief Overview of Pipeline

A Kubernetes cluster instantiates _nodes_ on a _node pool_, which specifies
what types of host machines are available.  _Jobs_ can be run on the cluster by
specifying details such as what containers to run, how many are needed, what
arguments they take, etc.  _Pods_ are created to run individual containers --
the pods themselves run on the _nodes_.

In our case, we won't let Kubernetes resize of our node pool dynamically, we'll
manually specify how many machines we want: this means kubernetes will leave
machines running even if they're not doing anything!  So be sure to clean up
your clusters.

### Setup

1. Run `cluster-up-cpu` or (`cluster-up-gpu-small`), which will:

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


2. [Optional] Resizing your cluster.  Note that the cluster will *not* use
   autoscaling by default, so it's possible to have a lot of idle containers
   running if you're not careful!

  ```
  gcloud beta container clusters resize $CLUSTER_NAME --zone=$ZONE --size=8
  ```

## GCS for simple task signaling

The main way these jobs interact is through GCS, a distributed webservice
intended to behave like a filesystem.

The selfplay jobs will find the newest model in the GCS directory of models and
play games with it, writing the games out to a different directory in the
bucket.

The training job will collect games from that directory and turn it into
chunks, which it will use to train a new model, adding it to the directory of
models, and completing the circle.

## Launching selfplay workers on a cluster

Once the cluster is setup, all you need to do to set-up the selfplay job is to run:

```shell
cluster/deploy-gpu-player.sh
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


### Useful things for the selfplay cluster

* Getting a list of the selfplay games ordered by start time.
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


If you've run rsync to collect a set of SGF files (cheatsheet: `gsutil -m cp -r
gs://$BUCKET_NAME/sgf/$MODEL_NAME sgf/`), here are some handy
bash fragments to run on them:

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

Also check the 'oneoffs' directory for interesting scripts to analyze e.g. the
resignation threshold.


### Setting up the selfplay cluster: Debugging checklist

* Check your gcloud -- authorized?  Correct default zone settings?
* Check the project name, cluster name, & bucket name variables in the
  `cluster/common.sh` script.  Did you change things?
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

