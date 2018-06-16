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

from absl import flags
import jinja2
import kubernetes
import argparse
import yaml
import json
import os
import argh
import fsdb
import random
import time


def cross_eval(m1_path, m2_path, jobname, completions=20):
    """
    m1_path, m2_path: full gs:// paths to the .pb files to match up
    jobname: string, appended to the container, used to differentiate the job names
    (e.g. 'minigo-gpu-evaluator-v5-123-v7-456')
    """
    if m1_path is None or m2_path is None or jobname is None:
        print("Provide all of m1_path, m2_path, and jobname params")
        return
    kubernetes.config.load_kube_config()
    configuration = kubernetes.client.Configuration()
    api_instance = kubernetes.client.BatchV1Api(
        kubernetes.client.ApiClient(configuration))

    raw_job_conf = open("cluster/evaluator/cc-evaluator.yaml").read()

    os.environ['MODEL_BLACK'] = m1_path
    os.environ['MODEL_WHITE'] = m2_path
    os.environ['JOBNAME'] = jobname + '-bw'
    env_job_conf = os.path.expandvars(raw_job_conf)

    job_conf = yaml.load(env_job_conf)
    job_conf['spec']['completions'] = completions

    resp = api_instance.create_namespaced_job('default', body=job_conf)

    os.environ['MODEL_WHITE'] = m1_path
    os.environ['MODEL_BLACK'] = m2_path
    os.environ['JOBNAME'] = jobname + '-wb'
    env_job_conf = os.path.expandvars(raw_job_conf)
    job_conf = yaml.load(env_job_conf)
    job_conf['spec']['completions'] = completions

    resp = api_instance.create_namespaced_job('default', body=job_conf)



def launch_eval(black_num=0, white_num=0):
    if black_num <= 0 or white_num <= 0:
        print("Need real model numbers")
        return

    b = fsdb.get_model(black_num)
    w = fsdb.get_model(white_num)

    b_model_path = os.path.join(fsdb.models_dir(), b)
    w_model_path = os.path.join(fsdb.models_dir(), w)

    kubernetes.config.load_kube_config()
    configuration = kubernetes.client.Configuration()
    api_instance = kubernetes.client.BatchV1Api(
        kubernetes.client.ApiClient(configuration))

    raw_job_conf = open("cluster/evaluator/gpu-evaluator.yaml").read()
    env_job_conf = os.path.expandvars(raw_job_conf)

    t = jinja2.Template(env_job_conf)
    job_conf = yaml.load(t.render({'white': w_model_path,
                                   'black': b_model_path,
                                   'wnum': white_num,
                                   'bnum': black_num}))

    resp = api_instance.create_namespaced_job('default', body=job_conf)

    job_conf = yaml.load(t.render({'white': b_model_path,
                                   'black': w_model_path,
                                   'wnum': black_num,
                                   'bnum': white_num}))

    resp = api_instance.create_namespaced_job('default', body=job_conf)


def get_api():
    kubernetes.config.load_kube_config(persist_config=True)
    configuration = kubernetes.client.Configuration()
    return kubernetes.client.BatchV1Api(
        kubernetes.client.ApiClient(configuration))


def zoo_loop():
    desired_pairs = restore_pairs() or []
    last_model_queued = restore_last_model()

    api_instance = get_api()
    try:
        while True:
            last_model = fsdb.get_latest_model()[0]
            if last_model_queued < last_model:
                print("Adding models {} to {} to be scheduled".format(
                    last_model_queued+1, last_model))
                for m in reversed(range(last_model_queued+1, last_model+1)):
                    desired_pairs += make_pairs_for_model(m)
                last_model_queued = last_model
                save_last_model(last_model)

            cleanup(api_instance)
            r = api_instance.list_job_for_all_namespaces()
            if len(r.items) < 10:
                if not desired_pairs:
                    time.sleep(60*5)
                    continue
                next_pair = desired_pairs.pop()  # take our pair off
                print("Enqueuing:", next_pair)
                try:
                    launch_eval(*next_pair)
                except:
                    desired_pairs.append(next_pair)
                    raise
                save_pairs(sorted(desired_pairs))
                save_last_model(last_model)

            else:
                print("{}\t{} jobs outstanding.".format(
                    time.strftime("%I:%M:%S %p"), len(r.items)))
            time.sleep(30)
    except:
        print("Unfinished pairs:")
        print(sorted(desired_pairs))
        save_pairs(sorted(desired_pairs))
        save_last_model(last_model)
        raise


def restore_pairs():
    with open('closest_pairs.json') as f:
        pairs = json.loads(f.read())
    return pairs


def save_pairs(pairs):
    with open('closest_pairs.json', 'w') as f:
        json.dump(pairs, f)


def save_last_model(model):
    with open('last_model.json', 'w') as f:
        json.dump(model, f)


def restore_last_model():
    with open('last_model.json') as f:
        last_model = json.loads(f.read())
    return last_model


def cleanup(api_instance=None):
    api = api_instance or get_api()
    r = api.list_job_for_all_namespaces()
    delete_opts = kubernetes.client.V1DeleteOptions()
    for job in r.items:
        if job.status.succeeded == job.spec.completions:
            print(job.metadata.name, "finished!")
            resp = api.delete_namespaced_job(
                job.metadata.name, 'default', body=delete_opts)


def backpair(model_num=0):
    pairs = make_pairs_for_model(model_num)
    for p in pairs:
        launch_eval(*p)


def make_pairs_for_model(model_num=0):
    if model_num == 0:
        return
    pairs = []
    pairs += [[model_num, model_num - i]
              for i in range(1, 10)if model_num - i > 0]
    pairs += [[model_num, model_num - i]
              for i in range(10, 20, 2)if model_num - i > 0]
    pairs += [[model_num, model_num - i]
              for i in range(20, 51, 5)if model_num - i > 0]
    return pairs


parser = argparse.ArgumentParser()
argh.add_commands(parser, [zoo_loop, launch_eval, backpair, cleanup, cross_eval])

if __name__ == '__main__':
    remaining_argv = flags.FLAGS(sys.argv, known_only=True)
    argh.dispatch(parser, argv=remaining_argv[1:])
    #for i in range(270, 280, 1):
    #    make_pairs(i)
