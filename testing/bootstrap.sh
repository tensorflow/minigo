#!/bin/bash
#
# This script is used to bootstrap our prow jobs.
# The point of this script is to check out the kubeflow/kubeflow repo
# at the commit corresponding to the Prow job. We can then
# invoke the launcher script at that commit to submit and
# monitor an Argo workflow
set -xe

mkdir -p /src
git clone https://github.com/tensorflow/minigo.git /src/minigo

cd /src/minigo

echo $(ls)

echo Job Name = ${JOB_NAME}

# See https://github.com/kubernetes/test-infra/tree/master/prow#job-evironment-variables
if [ ! -z ${PULL_NUMBER} ]; then
  git fetch origin  pull/${PULL_NUMBER}/head:pr
  git checkout ${PULL_PULL_SHA}
else
  if [ ! -z ${PULL_BASE_SHA} ]; then
    # Its a post submit; checkout the commit to test.
    git checkout ${PULL_BASE_SHA}
  fi
fi

# Print out the commit so we can tell from logs what we checked out.
echo Repo is at `git describe --tags --always --dirty`
git status

export PYTHONPATH=$PYTHONPATH:/src/minigo

# Note: currently, this is hardcoded for python, but it would be easy to modify
# this to add other languages

pip3 install -r requirements-cpu.txt

ls *.py | xargs pylint

BOARD_SIZE=9 python3 -m unittest discover tests
