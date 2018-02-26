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

For example:, if you would like to override the CGP Project or image tag, you can set:

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
