# Minigo Testing

This directory contains test infrastructure for Minigo, largely based of the
work done by https://github.com/kubeflow/kubeflow.

Our tests are run on the Kubernetes test runner called prow. See the [Prow
docs](https://github.com/kubernetes/test-infra/tree/master/prow) for more
details.

Some UIs to check out:

Testgrid (Test Results Dashboard): https://k8s-testgrid.appspot.com/sig-big-data
Prow (Test-runner dashboard): https://prow.k8s.io/?repo=tensorflow%2Fminigo

## Updating continuous integration tests

You will need to update the `cc-base` Docker image if you modify the certain
files in the repo (e.g. `WORKSPACE`, `.bazelrc`, `cc/tensorflow/*`) because they
will break the Docker cache before the slow `./cc/configure_tensorflow.sh` step.

```shell
(cd cluster/base && PROJECT=tensor-go VERSION_TAG=latest make base-push)
```

See the list of `COPY` files in `cluster/base/Dockerfile` for the complete list.

The test image may need to be rebuilt occasionally if installed libraries or
tools need updating (e.g. `clang-format`, `tensorflow`):

```shell
(cd testing/ && PROJECT=tensor-go VERSION_TAG=latest make pushv2)
```

## Test a pull request locally

You can test a PR sent to github locally using the following steps. This
enables you to poke around if something's failing.

```shell
export PATH=$HOME/go/bin:$PATH
cd $HOME  # Or somewhere else outside the minigo repo
git clone git@github.com:kubernetes/test-infra.git
cd test-infra/config
./pj-on-kind.sh pull-tf-minigo-cc
```

Then enter the pull request number when prompted.

To interactively debug the `pj-on-kind` test, edit
`config/jobs/tensorflow/minigo/minigo.yaml` in the `test-infra` repository
and change the command run by `pull-tf-minigo-cc` to run `sleep 10000000`
instead of `./cc/test.sh`.

Find the name of the pod name:

```shell
export KUBECONFIG="$(kind get kubeconfig-path --name="mkpod")"
kubectl get pods --all-namespaces
```

The pod name will be a long hex string, something like
`fc19639a-b497-11e9-95be-ecb1d74c871e`.

You can now attach to the running pod using:

```shell
kubectl exec -it --namespace default $POD_NAME bash
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
