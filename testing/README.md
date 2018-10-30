# Minigo Testing

This directory contains test infrastructure for Minigo, largely based of the
work done by https://github.com/kubeflow/kubeflow.

Our tests are run on the Kubernetes test runner called prow. See the [Prow
docs](https://github.com/kubernetes/test-infra/tree/master/prow) for more
details.

Some UIs to check out:

Testgrid (Test Results Dashboard): https://k8s-testgrid.appspot.com/sig-big-data
Prow (Test-runner dashboard): https://prow.k8s.io/?repo=tensorflow%2Fminigo

## Local testing

To test out changes to the docker image, first build the test-harness image:

```shell
make buildv2
```

And then run the tests.

```shell
docker run --rm gcr.io/minigo-testing/minigo-prow-harness-v2:latest \
  --repo=github.com/tensorflow/minigo --job=tf-minigo-presubmit --scenario=execute -- ./test.sh
```

## Components

- `../test.sh`: the actual tests that are run. TODO(#188): Change this to
  output junit/XML and Prow will split out the tests.
- `Dockerfile`: Run the tests in this container (and pull in test-infra stuff as the runner).
- `Makefile`: Build the Dockerfile
- `bootstrap_v2.sh`: The Prow wrapper. You'll notice that `bootstrap_v2.sh`
  does not actually reference `../test.sh`. That gets linked in via Prow's
  **Job** config (see below).

## Prow configuration


Minigo has some configuration directly in Prow to make all this jazz work:

- **Test configuration**. This configures the specific test-suites that are run on prow
  https://github.com/kubernetes/test-infra/blob/master/config/jobs/tensorflow/minigo/minigo.yaml

- **Test UI Configuration**: What shows up in testgrid, the Prow test-ui?
  https://github.com/kubernetes/test-infra/blob/master/testgrid/config.yaml

- **Bootstrap-jobs-config**: This is what links `../test.sh` with
  `bootstrap_v2.sh`. See:
  https://github.com/kubernetes/test-infra/blob/master/jobs/config.json

- **Other Plugin Config**. We also use the Size and LGTM plugins provided by
  Prow. See
  https://github.com/kubernetes/test-infra/blob/master/prow/plugins.yaml
