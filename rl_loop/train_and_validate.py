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

""" Run train and validate in a loop, as subprocesses.

We run as subprocesses because it gives us some isolation.
"""

import itertools
import os
import sys
import time

sys.path.insert(0, '.')

from absl import app, flags
from tensorflow import gfile
import tensorflow as tf
from rl_loop import fsdb
import mask_flags
from rl_loop import shipname
import utils
import dual_net

flags.DEFINE_string('pro_dataset', None,
                    'Location of preprocessed pro dataset for validation')

# From fsdb.py - must pass one of the two.
flags.declare_key_flag('base_dir')
flags.declare_key_flag('bucket_name')

FLAGS = flags.FLAGS

try:
    if 'KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS' in os.environ:
        TPU_NAME = os.environ['KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS']
    else:
        TPU_NAME = os.environ['TPU_NAME']
except KeyError:
    raise Exception("Must have $TPU_NAME configured")

def train():
    model_num, model_name = fsdb.get_latest_model()
    print("Training on gathered game data, initializing from {}".format(
        model_name))
    new_model_num = model_num + 1
    new_model_name = shipname.generate(new_model_num)
    print("New model will be {}".format(new_model_name))
    save_file = os.path.join(fsdb.models_dir(), new_model_name)

    # TODO(jacksona): Refactor train.py to take the filepath as a flag.
    cmd = ['python3', 'train.py', '__unused_file__',
           '--use_tpu',
           '--use_bt',
           '--work_dir={}'.format(fsdb.working_dir()),
           '--tpu_name={}'.format(TPU_NAME),
           '--flagfile=rl_loop/distributed_flags',
           '--export_path={}'.format(save_file)]

    completed_process = mask_flags.run(cmd)
    if completed_process.returncode > 0:
        print("Training failed!")
        return completed_process

    # Train.py already copies the {data,index,meta} files to $BUCKET/models
    # Persist the checkpoint two ways:
    # Freeze the .ckpt file in the work_dir for the TPU selfplayers
    # Freeze a non-tpu version of the graph for later GPU use.
    latest_checkpoint = tf.train.latest_checkpoint(fsdb.working_dir())
    p = freeze(latest_checkpoint, rewrite_tpu=True)
    if p.returncode > 0:
        print("== TPU freeze failed!")
        return p

    p = freeze(save_file, rewrite_tpu=False)
    if p.returncode > 0:
        print("== Model freeze failed!")
        return p

    return completed_process

def freeze(save_path, rewrite_tpu=False):
    cmd = ['python3', 'freeze_graph.py',
           '--work_dir={}'.format(fsdb.working_dir()),
           '--flagfile=rl_loop/distributed_flags',
           '--model_path={}'.format(save_path)]

    if rewrite_tpu:
        cmd.extend(['--use_tpu',
                    '--tpu_name={}'.format(TPU_NAME)])

    return mask_flags.run(cmd)


def validate_holdout_selfplay():
    """Validate on held-out selfplay data."""
    holdout_dirs = (os.path.join(fsdb.holdout_dir(), d)
                    for d in reversed(gfile.ListDirectory(fsdb.holdout_dir()))
                    if gfile.IsDirectory(os.path.join(fsdb.holdout_dir(), d))
                    for f in gfile.ListDirectory(os.path.join(fsdb.holdout_dir(), d)))

    # This is a roundabout way of computing how many hourly directories we need
    # to read in order to encompass 20,000 holdout games.
    holdout_dirs = set(itertools.islice(holdout_dirs), 20000)
    cmd = ['python3', 'validate.py'] + list(holdout_dirs) + [
        '--use_tpu',
        '--tpu_name={}'.format(TPU_NAME),
        '--flagfile=rl_loop/distributed_flags',
        '--expand_validation_dirs']
    mask_flags.run(cmd)

def validate_pro():
    """Validate on professional data."""
    cmd = ['python3', 'validate.py', FLAGS.pro_dataset,
           '--use_tpu',
           '--tpu_name={}'.format(TPU_NAME),
           '--work_dir={}'.format(fsdb.working_dir()),
           '--flagfile=rl_loop/distributed_flags',
           '--validate_name=pro']
    mask_flags.run(cmd)


def loop(unused_argv):
    if len(fsdb.get_models()) == 0:
        # TODO(amj): Do bootstrap here.
        pass
    while True:
        print("=" * 40, flush=True)
        with utils.timer("Train"):
            completed_process = train()
        if completed_process.returncode > 0:
            print("Training failed, aborting.")
            sys.exit(1)

        with utils.timer("Validate"):
            if not FLAGS.pro_dataset:
                print("*** --pro_dataset not set, skipping pro validation ***")
            else:
                validate_pro()
            #validate_holdout_selfplay()

if __name__ == '__main__':
    app.run(loop)
