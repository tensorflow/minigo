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

import datetime
import json
import os
import time
import re

import fire
from absl import flags
from tensorflow import gfile

def launch_eval_job(tag, m1_path, m2_path, job_name, completions=5):
    """Launches an evaluator job.
    tag: name for later reference
    m1_path, m2_path: full gs:// paths to the .pb files to match up
    job_name: string, appended to the container, used to differentiate the job
    names (e.g. 'minigo-cc-evaluator-v5-123-v7-456')
    completions: the number of completions desired
    """
    if not all([m1_path, m2_path, job_name]):
        print("Provide all of m1_path, m2_path, job_name "
              "params")
        return
    if not re.match(tag, r'[a-z0-9-]*$')
        print("{} is not a valid tag".format(tag))
        return

    api_instance = get_api()

    raw_job_conf = open("cluster/evaluator/cc-evaluator.yaml").read()

    bucket_path = "gs://minigo-pub/experiments/eval/" + tag

    os.environ['BUCKET_NAME'] = bucket_path

    os.environ['MODEL_BLACK'] = m1_path
    os.environ['MODEL_WHITE'] = m2_path
    os.environ['JOBNAME'] = job_name + '-bw'
    env_job_conf = os.path.expandvars(raw_job_conf)

    job_conf = yaml.load(env_job_conf)
    job_conf['spec']['completions'] = completions

    resp = api_instance.create_namespaced_job('default', body=job_conf)

    os.environ['MODEL_WHITE'] = m1_path
    os.environ['MODEL_BLACK'] = m2_path
    os.environ['JOBNAME'] = job_name + '-wb'
    env_job_conf = os.path.expandvars(raw_job_conf)
    job_conf = yaml.load(env_job_conf)
    job_conf['spec']['completions'] = completions

    resp = api_instance.create_namespaced_job('default', body=job_conf)

    #####
    TS=str(int(time.time()))
    with gfile.Gfile(os.path.join(bucket_path, 'commands_' + TS)) as commands:
        commands.write(str(sys.argv) + "\n")

    metadata = {
        'timestamp': TS,
        'date': datetime.datetime.now().isoformat(' '),
        'model1': m1_path,
        'model2': m2_path,
        'job_name': job_name,
        'completions': completions,
    }
    with gfile.Gfile(os.path.join(bucket_path, 'metadata')) as metadata_file:
        json.dump(metadata, metadata_file)

    # TODO(sethtroisi): Support patching in launch_eval.py



    return resp


if __name__ == '__main__':
    remaining_argv = flags.FLAGS(sys.argv, known_only=True)
    fire.Fire({
        'launch_eval_job': launch_eval_job,
    }, remaining_argv[1:])
