# Copyright 2019 Google LLC
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

"""Bootstraps a reinforcement learning loop from a checkpoint."""

import sys
sys.path.insert(0, '.')  # nopep8

# Hide the GPUs from TF. This makes startup 2x quicker on some machines.
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # nopep8

import subprocess
import tensorflow as tf
from ml_perf.utils import *

from absl import app, flags


N = os.environ.get('BOARD_SIZE', '19')

flags.DEFINE_string('checkpoint_dir', None, 'Source checkpoint directory.')
flags.DEFINE_string('selfplay_dir', None, 'Selfplay example directory.')
flags.DEFINE_string('work_dir', None, 'Training work directory.')
flags.DEFINE_string('flag_dir', None, 'Flag directory.')

FLAGS = flags.FLAGS


def main(unused_argv):
    # Copy the checkpoint data to the correct location.
    copy_tree(os.path.join(FLAGS.checkpoint_dir, 'data/selfplay'),
              FLAGS.selfplay_dir)
    copy_tree(os.path.join(FLAGS.checkpoint_dir, 'work_dir'), FLAGS.work_dir)
    copy_tree(os.path.join(FLAGS.checkpoint_dir, 'flags'), FLAGS.flag_dir)

if __name__ == '__main__':
    app.run(main)
