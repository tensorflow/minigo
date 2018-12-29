# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
sys.path.insert(0, '.')

from cluster.evaluator import launch_eval

import datetime
import json
import os
import time
import re

import fire
from absl import flags
from tensorflow import gfile

LAUNCH_EVAL_VERSION = 1


def launch_eval_job(tag, m1_path, m2_path, job_name, completions):
    """Launches an evaluator job.
    tag: name for this eval job (used as top level folder name)
    m1_path, m2_path: full gs:// paths to the .pb files to match up
    job_name: string, appended to the container, used to differentiate the job
    names (e.g. 'minigo-cc-evaluator-v5-123-v7-456')
    completions: the number of completions desired (each completion is 2 games)
    """
    print()
    if not re.match(r'[a-z0-9-]*$', tag, re.I):
        print("{} is not a valid tag".format(tag))
        return

    # TODO(sethtroisi): Change to minigo-pub
    sgf_bucket_path = "sethtroisi-sandbox/experiments/eval/" + tag
    assert not sgf_bucket_path.startswith("gs://"), bucket_pat
    bucket_path = "gs://" + sgf_bucket_path


    metadata_path = os.path.join(bucket_path, 'metadata')
    assert not gfile.Exists(metadata_path), "Already exists"

    TS=str(int(time.time()))
    metadata = {
        'timestamp': TS,
        'date': datetime.datetime.now().isoformat(' '),
        'model1': os.path.basename(m1_path),
        'model2': os.path.basename(m2_path),
        'model1_path': m1_path,
        'model2_path': m2_path,
        'job_name': job_name,
        'completions': completions,
        'launch_eval_version' : LAUNCH_EVAL_VERSION,
    }

    job_conf, resp_bw, resp_wb = launch_eval.launch_eval_job(
        m1_path, m2_path, job_name, sgf_bucket_path, completions)

    if not (resp_bw and resp_wb):
        print("launch_eval.py failed")
        print(job_conf)
        print(resp_bw)
        print(resp_wb)
        print()
        assert False

    # Jobs were launched, record metadata to GCS.
    with gfile.GFile(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file)

    with gfile.GFile(os.path.join(bucket_path, 'commands'), "w") as f:
        f.write(str(sys.argv) + "\n")


    with gfile.GFile(os.path.join(bucket_path, 'job_conf'), "w") as f:
        f.write(str(job_conf) + "\n")

    # TODO(sethtroisi): Support patching in launch_eval.py


if __name__ == '__main__':
    remaining_argv = flags.FLAGS(sys.argv, known_only=True)
    fire.Fire({
        'launch_eval_job': launch_eval_job,
    }, remaining_argv[1:])
