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

"""Creates a bootstrap checkpoint from a previous run."""

import sys
sys.path.insert(0, '.')  # nopep8

# Hide the GPUs from TF. This makes startup 2x quicker on some machines.
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # nopep8

import subprocess
import tensorflow as tf
import time
from ml_perf.utils import *

from absl import app, flags


flags.DEFINE_string('selfplay_dir', None, 'Selfplay example directory.')
flags.DEFINE_string('work_dir', None, 'Training work directory.')
flags.DEFINE_string('flag_dir', None, 'Flag directory.')
flags.DEFINE_integer('window_size', 5,
                     'Maximum number of recent selfplay rounds to train on.')
flags.DEFINE_integer('min_games_per_iteration', 4096,
                     'Minimum number of games to play for each training '
                     'iteration.')
flags.DEFINE_integer(
    'ckpt', None,
    'Training checkpoint in the TensorFlow work_dir to use. For example, '
    'setting --checkpoint_num=12345 will copy the files '
    'model.ckpt-12345.data-00000-of-00001, model.ckpt-12345.index and '
    'model.ckpt-12345.meta from work_dir.')
flags.DEFINE_string('checkpoint_dir', None, 'Destination checkpoint directory.')

FLAGS = flags.FLAGS


def get_model_name(model_num):
    return '{:06}'.format(model_num)


def clean_checkpoint_dir():
    for sub_dir in ['data/selfplay', 'work_dir', 'flags']:
        path = os.path.join(FLAGS.checkpoint_dir, sub_dir)
        if tf.io.gfile.exists(path):
            print('Removing {}'.format(path))
            tf.io.gfile.rmtree(path)
        if not FLAGS.checkpoint_dir.startswith('gs://'):
            tf.io.gfile.makedirs(path)


def copy_training_checkpoint():
    dst_dir = os.path.join(FLAGS.checkpoint_dir, 'work_dir')

    name = 'model.ckpt-{}'.format(FLAGS.ckpt)
    checkpoint_path = os.path.join(dst_dir, 'checkpoint')
    print('Writing {}'.format(checkpoint_path))
    with tf.io.gfile.GFile(checkpoint_path, 'w') as f:
        f.write('model_checkpoint_path: "{}"\n'.format(name))
        f.write('all_model_checkpoint_paths: "{}"\n'.format(name))

    for ext in ['data-00000-of-00001', 'index', 'meta']:
        basename = '{}.{}'.format(name, ext)
        src_path = os.path.join(FLAGS.work_dir, basename)
        dst_path = os.path.join(dst_dir, basename)
        print('Copying {} to {}'.format(src_path, dst_path))
        tf.io.gfile.copy(src_path, dst_path)


def find_model_num():
    # Get the modification date of one of the checkpoint files.
    path = os.path.join(FLAGS.work_dir, 'model.ckpt-{}.meta'.format(FLAGS.ckpt))
    checkpoint_mtime = tf.io.gfile.stat(path).mtime_nsec

    # Selfplay data is written under the following directory structure:
    #   selfplay_dir/model/device/time
    # Look for model directories that contain selfplay data that matches the
    # time that this model was trained from.
    time_str = time.strftime(
        '%Y-%m-%d-%H', time.localtime(checkpoint_mtime / 1000 / 1000 / 1000))
    pattern = os.path.join(FLAGS.selfplay_dir, '*', '*', time_str)
    print(
        'Looking for selfplay data directories that match "{}"'.format(pattern))
    paths = sorted(tf.io.gfile.glob(pattern))
    if not paths:
        raise RuntimeError(
            'Couldn\'t find selfplay data for checkpoint {}'.format(FLAGS.ckpt))

    # Extract the model numbers from the directories found.
    models = sorted(set([os.path.normpath(x).split(os.sep)[-3] for x in paths]))

    model_num = int(models[-1])
    while model_num > 0:
        model_name = get_model_name(model_num)
        pattern = os.path.join(
            FLAGS.selfplay_dir, model_name, '*', '*', '*.tfrecord.zz')
        num_older = len([x for x in tf.io.gfile.glob(pattern)
                         if tf.io.gfile.stat(x).mtime_nsec < checkpoint_mtime])
        print('Found {} games older than checkpoint {} for model {}'.format(
              num_older, FLAGS.ckpt, model_name))
        if num_older >= FLAGS.min_games_per_iteration:
            break

        model_num -= 1

    if model_num == 0:
        raise RuntimeError(
            'Couldn\'t a model with at least {} selfplay games older than '
            'checkpoint {}'.format(FLAGS.min_games_per_iteration, FLAGS.ckpt))

    return model_num


def copy_flags():
    copy_tree(FLAGS.flag_dir, os.path.join(FLAGS.checkpoint_dir, 'flags'))


def copy_selfplay_data(checkpoint_model_num):
    # The first thing the RL loop does is start training a new model, so the
    # checkpoint must contain training examples from games played by the
    # checkpoint model.
    latest_model_num = checkpoint_model_num + 1

    # Make sure there are enough examples. This might not be the case if the
    # user is trying to create a checkpoint from the very latest set of weights.
    paths = tf.io.gfile.glob(os.path.join(
        FLAGS.selfplay_dir, get_model_name(latest_model_num), '*', '*',
        '*.tfrecord.zz'))
    # For the latest directory of games, only copy the minimum number
    # of games required to start training.
    if len(paths) < FLAGS.min_games_per_iteration:
        raise RuntimeError(
            'Require at least {} training examples for the latest '
            'model {:06}, got {}. Please choose an earlier checkpoint'.format(
                FLAGS.min_games_per_iteration, latest_model_num, len(paths)))

    # Copy the training examples.
    end = latest_model_num + 1
    begin = max(0, end - FLAGS.window_size)
    for model_num in range(begin, end):
        model_name = get_model_name(model_num)
        copy_tree(
            os.path.join(FLAGS.selfplay_dir, model_name),
            os.path.join(FLAGS.checkpoint_dir, 'data', 'selfplay', model_name))


def main(unused_argv):
    clean_checkpoint_dir()
    copy_flags()
    copy_training_checkpoint()
    checkpoint_model_num = find_model_num()
    copy_selfplay_data(checkpoint_model_num)


if __name__ == '__main__':
    app.run(main)
