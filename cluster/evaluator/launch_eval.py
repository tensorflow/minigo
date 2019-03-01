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

import fire
import random
from absl import flags
import kubernetes
from kubernetes.client.rest import ApiException
import yaml
import json
import os
import time
import random
import numpy as np
from rl_loop import fsdb

from ratings import ratings

MAX_TASKS = 150  # Keep < 500, or k8s may not track completions accurately.
MIN_TASKS = 20


def launch_eval_job(m1_path, m2_path, job_name,
        bucket_name, completions=5, flags_path=None):
    """Launches an evaluator job.
    m1_path, m2_path: full gs:// paths to the .pb files to match up
    job_name: string, appended to the container, used to differentiate the job
    names (e.g. 'minigo-cc-evaluator-v5-123-v7-456')
    bucket_name: Where to write the sgfs, passed into the job as $BUCKET_NAME
    completions: the number of completions desired
    flags_path: the path to the eval flagfile to use (if any)
    """
    if not all([m1_path, m2_path, job_name, bucket_name]):
        print("Provide all of m1_path, m2_path, job_name, and bucket_name "
              "params")
        return
    api_instance = get_api()

    raw_job_conf = open("cluster/evaluator/cc-evaluator.yaml").read()

    if flags_path:
        os.environ['EVAL_FLAGS_PATH'] = flags_path
    else:
        os.environ['EVAL_FLAGS_PATH'] = ""
    os.environ['BUCKET_NAME'] = bucket_name
    os.environ['MODEL_BLACK'] = m1_path
    os.environ['MODEL_WHITE'] = m2_path
    os.environ['JOBNAME'] = job_name
    env_job_conf = os.path.expandvars(raw_job_conf)

    job_conf = yaml.load(env_job_conf)
    job_conf['spec']['completions'] = completions

    response = api_instance.create_namespaced_job('default', body=job_conf)
    return job_conf, response


def same_run_eval(black_num=0, white_num=0, completions=4):
    """Shorthand to spawn a job matching up two models from the same run,
    identified by their model number """
    if black_num <= 0 or white_num <= 0:
        print("Need real model numbers")
        return

    b = fsdb.get_model(black_num)
    w = fsdb.get_model(white_num)

    b_model_path = os.path.join(fsdb.models_dir(), b)
    w_model_path = os.path.join(fsdb.models_dir(), w)
    flags_path = fsdb.eval_flags_path()

    obj = launch_eval_job(b_model_path + ".pb",
                           w_model_path + ".pb",
                           "{:d}-{:d}".format(black_num, white_num),
                           bucket_name=flags.FLAGS.bucket_name,
                           flags_path=flags_path,
                           completions=completions)

    # Fire spams the retval to stdout, so...
    return "{} job launched ok".format(obj[1].metadata.name)


def _append_pairs(new_pairs):
    """ Load the pairlist, add new stuff, save it out """
    desired_pairs = restore_pairs() or []
    desired_pairs += new_pairs
    print("Adding {} new pairs, queue has {} pairs".format(len(new_pairs), len(desired_pairs)))
    save_pairs(desired_pairs)


def add_uncertain_pairs(dry_run=False):
    new_pairs = ratings.suggest_pairs(ignore_before=50)
    if dry_run:
        print(new_pairs)
    else:
        _append_pairs(new_pairs)


def add_top_pairs(dry_run=False, pair_now=False):
    """ Pairs up the top twenty models against each other.
    #1 plays 2,3,4,5, #2 plays 3,4,5,6 etc. for a total of 15*4 matches.

    Default behavior is to add the pairs to the working pairlist.
    `pair_now` will immediately create the pairings on the cluster.
    `dry_run` makes it only print the pairings that would be added
    """
    top = ratings.top_n(15)
    new_pairs = []
    for idx, t in enumerate(top[:10]):
        new_pairs += [[t[0], o[0]] for o in top[idx+1:idx+5]]

    if dry_run:
        print(new_pairs)
        return

    if pair_now:
        maybe_enqueue(new_pairs)
    else:
        _append_pairs(new_pairs)


def maybe_enqueue(desired_pairs):
    failed_pairs = []
    while len(desired_pairs) > 0:
        next_pair = desired_pairs.pop()  # take our pair off
        try:
            resp = same_run_eval(*next_pair)
            print(resp)
        except ApiException as err:
            if err.status == 409: # Conflict.  Flip the order and throw it on the pile.
                print("Conflict enqueing {}.  Continuing...".format(next_pair))
                next_pair = [next_pair[1], next_pair[0]]
                failed_pairs.append(next_pair)
                random.shuffle(desired_pairs)
            else:
                failed_pairs.append(next_pair)
        except:
            failed_pairs.append(next_pair)
            print("*** Unknown error attempting to pair {} ***".format(next_pair) )
            raise
    return failed_pairs


def zoo_loop(sgf_dir=None, max_jobs=40):
    """Manages creating and cleaning up match jobs.

    - Load whatever pairs didn't get queued last time, and whatever our most
      recently seen model was.
    - Loop and...
        - If a new model is detected, create and append new pairs to the list
        - Automatically queue models from a list of pairs to keep a cluster
          busy
        - As jobs finish, delete them from the cluster.
        - If we crash, write out the list of pairs we didn't manage to queue

    sgf_dir -- the directory where sgf eval games should be used for computing
      ratings.
    max_jobs -- the maximum number of concurrent jobs.  jobs * completions * 2
      should be around 500 to keep kubernetes from losing track of completions
    """
    desired_pairs = restore_pairs() or []
    random.shuffle(desired_pairs)
    last_model_queued = restore_last_model()

    if sgf_dir:
        sgf_dir = os.path.abspath(sgf_dir)

    api_instance = get_api()
    toggle = True
    try:
        while True:
            last_model = fsdb.get_latest_pb()[0]
            if last_model_queued < last_model:
                print("Adding models {} to {} to be scheduled".format(
                    last_model_queued+1, last_model))
                for m in reversed(range(last_model_queued+1, last_model+1)):
                    desired_pairs += make_pairs_for_model(m)
                last_model_queued = last_model
                save_last_model(last_model)

            cleanup(api_instance)
            random.shuffle(desired_pairs)
            r = api_instance.list_job_for_all_namespaces()
            if r.items:
                tasks = sum([item.spec.completions for item in r.items])
            else:
                tasks = 0
            if tasks < MAX_TASKS:
                if len(desired_pairs) == 0:
                    if sgf_dir:
                        if tasks > MIN_TASKS:
                            time.sleep(60)
                            continue
                        print("Out of pairs!  Syncing new eval games...")
                        ratings.sync(sgf_dir)
                        print("Updating ratings and getting suggestions...")
                        if toggle:
                          print("Pairing the top of the table.")
                          add_top_pairs()
                        else:
                          print("Pairing the least-known models.")
                          add_uncertain_pairs()
                        toggle = not toggle
                        for modelnum, rate in ratings.top_n():
                          print("{:>30}: {:0.3f} ({:0.3f})".format(modelnum, rate[0], rate[1]))
                        desired_pairs = restore_pairs() or []
                    else:
                        print("Out of pairs.  Sleeping ({} remain)".format(len(r.items)))
                        time.sleep(600)
                        continue


                next_pair = desired_pairs.pop()
                failed = maybe_enqueue([next_pair])
                if failed != []:
                    desired_pairs.extend(failed)
                save_pairs(sorted(desired_pairs))
                save_last_model(last_model)
                time.sleep(1)

            else:
                print("{}\t {} finished / {} requested. "
                      "({} jobs, {} pairs to be scheduled)".format(
                      time.strftime("%I:%M:%S %p"),
                      sum([i.status.succeeded or 0 for i in r.items]),
                      tasks, len(r.items), len(desired_pairs)))
                time.sleep(60)
    except:
        print("Unfinished pairs:")
        print(sorted(desired_pairs))
        save_pairs(sorted(desired_pairs))
        save_last_model(last_model)
        raise


def restore_pairs():
    with open('pairlist.json') as f:
        pairs = json.loads(f.read())
    return pairs


def save_pairs(pairs):
    with open('pairlist.json', 'w') as f:
        json.dump(pairs, f)


def save_last_model(model):
    with open('last_model.json', 'w') as f:
        json.dump(model, f)


def restore_last_model():
    with open('last_model.json') as f:
        last_model = json.loads(f.read())
    return last_model


def get_api():
    kubernetes.config.load_kube_config(persist_config=True)
    configuration = kubernetes.client.Configuration()
    return kubernetes.client.BatchV1Api(
        kubernetes.client.ApiClient(configuration))


def cleanup(api_instance=None):
    """ Remove completed jobs from the cluster """
    api = api_instance or get_api()
    r = api.list_job_for_all_namespaces()
    delete_opts = kubernetes.client.V1DeleteOptions(
            propagation_policy="Background")
    for job in r.items:
        if job.status.succeeded == job.spec.completions:
            print(job.metadata.name, "finished!")
            api.delete_namespaced_job(
                job.metadata.name, 'default', body=delete_opts)


def make_pairs_for_model(model_num=0):
    """ Create a list of pairs of model nums; play every model nearby, then
    every other model after that, then every fifth, etc.

    Returns a list like [[N, N-1], [N, N-2], ... , [N, N-12], ... , [N, N-50]]
    """
    if model_num == 0:
        return
    pairs = []

    mm_fib = lambda n: int((np.matrix([[2,1],[1,1]])**(n//2))[0,(n+1)%2])

    for i in range(2,14): # Max is 233
      if mm_fib(i) >= model_num:
        break
      pairs += [[model_num, model_num - mm_fib(i)]]

    return pairs


if __name__ == '__main__':
    remaining_argv = flags.FLAGS(sys.argv, known_only=True)
    fire.Fire({
        'zoo_loop': zoo_loop,
        'same_run_eval': same_run_eval,
        'cleanup': cleanup,
        'add_top_pairs': add_top_pairs,
        'launch_eval_job': launch_eval_job,
    }, remaining_argv[1:])
