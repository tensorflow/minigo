# Minigo Testing

This directory contains test infrastructure for Minigo, largely based of the
work done by https://github.com/kubeflow/kubeflow.

To test out changes to the docker image, run:

```shell
docker run --rm gcr.io/minigo-testing/minigo-prow-harness-v2:latest --repo=github.com/tensorflow/minigo --job=tf-minigo-presubmit
```
