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
flags.DEFINE_string('model_dir', None, 'Model directory.')
flags.DEFINE_string('flag_dir', None, 'Flag directory.')
flags.DEFINE_string('tpu_name', '', 'TPU name.')

FLAGS = flags.FLAGS


def main(unused_argv):
    # Copy the checkpoint data to the correct location.
    copy_tree(os.path.join(FLAGS.checkpoint_dir, 'data/selfplay'),
              FLAGS.selfplay_dir)
    copy_tree(os.path.join(FLAGS.checkpoint_dir, 'work_dir'), FLAGS.work_dir)
    copy_tree(os.path.join(FLAGS.checkpoint_dir, 'flags'), FLAGS.flag_dir)

    # Selfplay data is stored in the directory hierarchy:
    #   checkpoint_dir/data/selfplay/model_num/
    # Where model_num is the generation of the model that played those games.
    # The weights in the checkpoint have been trained from games model_num and
    # earlier, so we want to freeze a model named after (model_num + 1).
    model_names = sorted(tf.io.gfile.listdir(os.path.join(
        FLAGS.checkpoint_dir, 'data', 'selfplay')))
    print('Latest selfplay games in checkpoint: {}'.format(model_names[-1]))

    model_num = int(model_names[-1])
    model_name = '{:06}'.format(model_num + 1)
    print('Will freeze model: {}'.format(model_name))

    # Find the path to the model weights.
    ckpt_pattern = os.path.join(FLAGS.work_dir, 'model.ckpt-*.index')
    ckpt_paths = tf.io.gfile.glob(ckpt_pattern)
    if len(ckpt_paths) != 1:
        raise RuntimeError(
            'Expected exactly one file to match "{}", got [{}]'.format(
                ', '.join(ckpt_paths)))
    ckpt_path = ckpt_paths[0]
    ckpt_path = os.path.splitext(ckpt_path)[0]

    # Freeze a new model.
    print('Freezing checkpoint {}'.format(ckpt_path))
    subprocess.run([
        'python3', 'freeze_graph.py',
        '--flagfile={}/architecture.flags'.format(FLAGS.flag_dir, N),
        '--model_path={}'.format(ckpt_path),
        '--use_tpu={}'.format(bool(FLAGS.tpu_name)),
        '--tpu_name={}'.format(FLAGS.tpu_name)],
        check=True)

    src_path = ckpt_path + '.minigo'
    dst_path = os.path.join(FLAGS.model_dir, model_name + '.minigo')
    print('Moving {} to {}'.format(src_path, dst_path))
    tf.io.gfile.rename(src_path, dst_path)

if __name__ == '__main__':
    app.run(main)
