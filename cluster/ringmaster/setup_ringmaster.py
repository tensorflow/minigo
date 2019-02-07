""" Helps populate the kubernetes conf for our ringmaster job
"""
import sys
sys.path.insert(0, '.')

from absl import flags
from absl import app

import fire
import random
import kubernetes
from kubernetes.client.rest import ApiException
import yaml
import json
import os
import time
import random
from rl_loop import fsdb


"""

TODO:
    after all this, we're left with...

gs://path/to/control.ctl

gs://path/to/output/[POD_NAME1]/m1_vs_m2_0.sgf
gs://path/to/output/[POD_NAME1]/m1_vs_m2_1.sgf
gs://path/to/output/[POD_NAME2]/m1_vs_m2_0.sgf

    collect these into

gs://path/to/output/control.games/m1_vs_m2_0..N.sgf
"""


FLAGS = flags.FLAGS

def get_api():
    kubernetes.config.load_kube_config(persist_config=True)
    configuration = kubernetes.client.Configuration()
    return kubernetes.client.BatchV1Api(
        kubernetes.client.ApiClient(configuration))


def get_mg_path(model_run, model_num):
    """
    model_run = integer, e.g. 15, 16, corresponding to the v-number
    model_num = integer, e.g 939, for the model number in that run
    """
    fsdb.switch_base("minigo-pub/v{:d}-19x19".format(model_run))
    model = fsdb.get_model(model_num)
    return os.path.join(fsdb.models_dir(), model)


def launch_ring_job(job_name, ctl_path, m1_path, m2_path, out_path, completions):
    # we also need $SERVICE_ACCOUNT to be set.
    if not 'SERVICE_ACCOUNT' in os.environ:
        print("$SERVICE_ACCOUNT should be set")
        raise ValueError

    # Navigating the classes made by the kubernetes API is a real pain.
    # So let's just keep the raw string and do variable expansion.

    os.environ['RINGMASTER_CONTROL_PATH'] = ctl_path
    os.environ['JOB_NAME']  = job_name
    os.environ['MODEL_ONE'] = m1_path
    os.environ['MODEL_TWO'] = m2_path
    os.environ['OUT_PATH']  = out_path.strip('/')

    raw_job_conf = open("cluster/ringmaster/ringmaster.yaml").read()
    env_job_conf = os.path.expandvars(raw_job_conf)

    # completions isn't eight levels down so why not just set it here
    job_conf = yaml.load(env_job_conf)
    job_conf['spec']['completions'] = completions
    job_conf['spec']['parallelism'] = min(completions, 10)

    api = get_api()
    resp = api.create_namespaced_job('default', body=job_conf)
    return resp


if __name__ == '__main__':
    remaining_argv = FLAGS(sys.argv, known_only=True)
    fire.Fire({
        'launch_ring_job': launch_ring_job,
    }, remaining_argv[1:])
