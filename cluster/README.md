# Kubernetes Cluster and Image Management

Playing games on a local machine can be pretty slow.  One way to speed up
playing games is to run Minigo on many computers simultaneously.  Minigo was
originally trained by containerizing these worker jobs and running them on a
Kubernetes cluster, hosted on the Google Cloud Platform.

*NOTE* These commands will result in VMs and other GCP resources being created
and will result in charges to your GCP account!  *Proceed with care!*

## Initial Setup

Make sure you have the following command line tools:

  - [gcloud](https://cloud.google.com/sdk/downloads)
  - gsutil (via `gcloud components install gsutil`)
  - kubectl (via `gcloud components install kubectl`)
  - docker
  
And the Python [kubernetes-client](https://github.com/kubernetes-client/python)

```
pip install kubernetes
```

Next, make sure you have a Google Cloud Project with GKE Enabled

Make sure you have the following permissions:

  - storage.bucket.(create, get, setIamPolicy) ("Storage Admin")
  - storage.objects.(create, delete, get, list, update) ("Storage Object Admin")
  - iam.serviceAccounts.create ("Service Account Admin")
  - iam.serviceAccountKeys.create ("Service Account Key Admin")
  - iam.serviceAccounts.actAs ("Service Account User")
  - resourcemanager.projects.setIamPolicy ("Project IAM Admin")
  - container.clusters.create ("Kubernetes Engine Cluster Admin")
  - container.secrets.create ("Kubernetes Engine Developer")

Before doing anything else, set any environment variables you need by doing:

```shell
export VAR_NAME=blah
```

For example: if you would like to override the GCP Project or image tag, you can set:

```shell
export PROJECT=my-project
export VERSION_TAG=0.12.34
export BOARD_SIZE=19
```

After you've done that, source the defaults:

```shell
source common.sh
```

To reset your environment variables:

```shell
source unset_common.sh
```

### Building the images, an overview

The docker images are built from the Makefiles inside the subdirectories here.
The images have various entrypoints into the varioius minigo binaries depending
on their purpose.  Some of the images only differ in the way the minigo cc/main
binary is called, e.g. enabling TPU or not, etc.

The images used are:

  - `cc-base`: A base image with tensorflow precompiled with CUDA support.
  - `minigo-cc-player`: A selfplay image using the C++ engine, built on the base
    image, using GPUs by default.
  - `minigo-tpu-player`: A selfplay image using the C++ engine, built to use
    TPUs.
  - `minigo-cc-evaluator`: An image built to run a game between two models,
    saving the SGF to GCS, using the C++ engine (GPUs supported)
  - `minigo-gpu-evaluator`: Same, but using the Python engine.
  - `minigo-player`: Python selfplay (with compiled tensorflow wheel in python,
    optimized for a target architecture, *not* using GPUs)

The engines using the C++ engine (`cc-player`, `cc-evaluator`, `tpu-player`) are
all built off of the cc-base image.

In order to build an image, `cd` to the appropriate directory, read the Makefile
to determine the appropriate targets, and note which environment variables are
set.  In particular, `make` will automatically tag the built images so they can
be uploaded to the Google Container Registry, so the environment variables
identifying the project (`$PROJECT`) and the tag (`$VERSION_TAG`) need to be
set.  The 'version tag' can be whatever you want -- use whatever versioning
system keeps things organized.  Docker can run images specified by tag, and
whatever docker orchestration layer (kubernetes, GKE, etc) will also refer to
the tags, e.g., the kubernetes yaml configurations specify the tag to use.

For example, building the base image could be accomplished like so:

```shell
PROJECT=my-project VERSION_TAG=0.1234 make base-image
```

Once the image is built, you can push it to the container registry via the other
targets in the Makefile, e.g. `PROJECT=foo VERSION_TAG=bar make base-push`


Generally, the idea of deriving our various images from a base image is to
minimize the number of times tensorflow has to be compiled; the base image
compiles tensorflow without bringing in any of our source files, so changes to
our source files don't invalidate docker's caching, which would trigger a
rebuild.
