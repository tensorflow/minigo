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
import fsdb
import prep_flags
import shipname
import utils

flags.DEFINE_string('pro_dataset', None,
                    'Location of preprocessed pro dataset for validation')

# From fsdb.py - must pass one of the two.
flags.declare_key_flag('base_dir')
flags.declare_key_flag('bucket_name')

FLAGS = flags.FLAGS

try:
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
    training_file = os.path.join(
        fsdb.golden_chunk_dir(), str(new_model_num) + '.tfrecord.zz')
    while not gfile.Exists(training_file):
        print("Waiting for", training_file)
        time.sleep(1 * 60)
    save_file = os.path.join(fsdb.models_dir(), new_model_name)

    cmd = ['python', 'train.py', training_file,
           '--use_tpu',
           '--tpu_name={}'.format(TPU_NAME),
           '--flagfile=rl_loop/distributed_flags',
           '--export_path={}'.format(save_file)]

    return prep_flags.run(cmd)


def validate_holdout_selfplay():
    """Validate on held-out selfplay data."""
    holdout_dirs = (os.path.join(fsdb.holdout_dir(), d)
                    for d in reversed(gfile.ListDirectory(fsdb.holdout_dir()))
                    if gfile.IsDirectory(os.path.join(fsdb.holdout_dir(), d))
                    for f in gfile.ListDirectory(os.path.join(fsdb.holdout_dir(), d)))

    # This is a roundabout way of computing how many hourly directories we need
    # to read in order to encompass 20,000 holdout games.
    holdout_dirs = set(itertools.islice(holdout_dirs), 20000)
    cmd = ['python', 'validate.py'] + list(holdout_dirs) + [
        '--use_tpu',
        '--tpu_name={}'.format(TPU_NAME),
        '--flagfile=rl_loop/distributed_flags',
        '--expand_validation_dirs']
    prep_flags.run(cmd)

def validate_pro():
    """Validate on professional data."""
    cmd = ['python', 'validate.py', FLAGS.pro_dataset,
           '--use_tpu',
           '--tpu_name={}'.format(TPU_NAME),
           '--flagfile=rl_loop/distributed_flags',
           '--validate_name=pro']
    prep_flags.run(cmd)


def loop(unused_argv):
    while True:
        print("=" * 40)
        with utils.timer("Train"):
            completed_process = train()
        if completed_process.returncode > 0:
            print("Training failed! Skipping validation...")
            continue
        with utils.timer("Validate"):
            validate_pro()
            validate_holdout_selfplay()

if __name__ == '__main__':
    flags.mark_flag_as_required('pro_dataset')
    app.run(loop)
