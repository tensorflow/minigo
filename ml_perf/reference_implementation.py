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

"""Runs a reinforcement learning loop to train a Go playing model."""

import sys

sys.path.insert(0, '.')

import logging
import numpy
import os
import random
import re
import shutil
import subprocess
import tensorflow
import utils

from absl import app, flags
from rl_loop import example_buffer, fsdb, shipname

flags.DEFINE_string('engine', 'tf', 'Engine to use for inference.')

FLAGS = flags.FLAGS


class State:

  _NAMES = ['bootstrap'] + random.Random(0).sample(shipname.NAMES,
                                                   len(shipname.NAMES))

  def __init__(self):
    self.iter_num = 0
    self.play_model_num = 0
    self.play_model_name = self.play_output_name
    self.train_model_num = 1

  @property
  def play_output_name(self):
    return '%06d-%s' % (self.iter_num, self._NAMES[self.play_model_num])

  @property
  def play_model_path(self):
    return os.path.join(fsdb.models_dir(), self.play_model_name)

  @property
  def train_model_name(self):
    return '%06d-%s' % (self.iter_num, self._NAMES[self.train_model_num])

  @property
  def train_model_path(self):
    return os.path.join(fsdb.models_dir(), self.train_model_name)

  @property
  def seed(self):
    return self.iter_num + 1


def checked_run(cmd, name):
  logging.info('Running %s:\n  %s', name, '\n  '.join(cmd))
  with utils.logged_timer('%s finished' % name.capitalize()):
    completed_process = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed_process.returncode:
      logging.error('Error running %s: %s', name,
                    completed_process.stdout.decode())
      raise RuntimeError('Non-zero return code executing %s' % ' '.join(cmd))
  return completed_process


def get_lines(completed_process, slice):
  return '\n'.join(completed_process.stdout.decode()[:-1].split('\n')[slice])


class MakeSlice(object):

  def __getitem__(self, item):
    return item


make_slice = MakeSlice()


def cc_flags(state):
  return [
      '--engine={}'.format(FLAGS.engine),
      '--virtual_losses=8',
      '--seed={}'.format(state.seed),
  ]


def py_flags(state):
  return [
      '--work_dir={}'.format(fsdb.working_dir()),
      '--trunk_layers=10',
      '--conv_width=64',
      '--value_cost_weight=0.25',
      '--training_seed={}'.format(state.seed),
  ]


# Generate an initial model with random weights.
def bootstrap(state):
  checked_run([
      'python3', 'bootstrap.py', '--export_path={}'.format(
          state.play_model_path)
  ] + py_flags(state), 'bootstrap')


# Freezes a model checkpoint into a TensorFlow GraphDef proto, which is the
# required model format for C++ TensorFlow.
def freeze(state, model_path):
  checked_run([
      'python3', 'freeze_graph.py', '--model_path={}'.format(model_path)
  ] + py_flags(state), 'freeze')


# Self-play a number of games.
def selfplay(state):
  play_output_name = state.play_output_name
  play_output_dir = os.path.join(fsdb.selfplay_dir(), play_output_name)
  play_holdout_dir = os.path.join(fsdb.holdout_dir(), play_output_name)

  result = checked_run([
      'bazel-bin/cc/selfplay', '--parallel_games=2048',
      '--num_readouts=100', '--model={}.pb'.format(
          state.play_model_path), '--output_dir={}'.format(play_output_dir),
      '--holdout_dir={}'.format(play_holdout_dir)
  ] + cc_flags(state), 'selfplay')
  logging.info(get_lines(result, make_slice[-2:]))

  # Write examples to a single record.
  pattern = os.path.join(play_output_dir, '*', '*.zz')
  logging.info('Extracting examples from "{}"'.format(pattern))
  random.seed(state.seed)
  tensorflow.set_random_seed(state.seed)
  numpy.random.seed(state.seed)
  buffer = example_buffer.ExampleBuffer(sampling_frac=1.0)
  buffer.parallel_fill(tensorflow.gfile.Glob(pattern))
  buffer.flush(
      os.path.join(fsdb.golden_chunk_dir(), play_output_name + '.tfrecord.zz'))


# Train a new model.
def train(state, tf_records):
  result = checked_run([
      'python3',
      'train.py',
      *tf_records,
      '--export_path={}'.format(state.train_model_path),
  ] + py_flags(state), 'training')
  logging.info(get_lines(result, make_slice[-8:-8]))


# Validate the trained model against holdout games.
def validate(state, holdout_glob):
  result = checked_run(
      ['python3', 'validate.py', holdout_glob] + py_flags(state),
      'validation')
  logging.info(get_lines(result, make_slice[-4:-3]))


# Evaluate the trained model.
def evaluate(state, args, name, slice):
  sgf_dir = os.path.join(fsdb.eval_dir(), state.train_model_name)
  result = checked_run([
      'bazel-bin/cc/eval', '--parallel_games=100',
      '--model={}.pb'.format(
          state.train_model_path), '--sgf_dir={}'.format(sgf_dir)
  ] + args, name)
  result = get_lines(result, slice)
  logging.info(result)
  pattern = '{}\s+\d+\s+(\d+\.\d+)%'.format(state.train_model_name)
  return float(re.search(pattern, result).group(1)) * 0.01


# Evaluate trained model against previous best.
def evaluate_model(state):
  model_win_rate = evaluate(
      state,
      ['--num_readouts=100', '--model_two={}.pb'.format(state.play_model_path)] +
      cc_flags(state), 'model evaluation', make_slice[-7:])
  logging.info('Win rate %s vs %s: %.3f', state.train_model_name,
               state.play_model_name, model_win_rate)
  return model_win_rate


# Evaluate trained model against a known target model.
def evaluate_target(state):
  target_win_rate = evaluate(
      state,
      ['--num_readouts=100', '--model_two={}'.format('ml_perf/target.pb')] +
      cc_flags(state), 'target evaluation', make_slice[-7:])
  logging.info('Win rate %s vs %s: %.3f', state.train_model_name,
               state.play_model_name, target_win_rate)
  return target_win_rate


def rl_loop():
  state = State()
  bootstrap(state)
  freeze(state, state.play_model_path)
  selfplay(state)

  while state.iter_num < 100:
    holdout_glob = os.path.join(fsdb.holdout_dir(), '%06d-*' % state.iter_num,
                                '*')
    tf_records = os.path.join(fsdb.golden_chunk_dir(), '*.zz')
    tf_records = sorted(tensorflow.gfile.Glob(tf_records), reverse=True)[:5]

    state.iter_num += 1

    # Train on shuffled game data of the last 5 selfplay rounds.
    train(state, tf_records)
    freeze(state, state.train_model_path)

    # These could run in parallel.
    validate(state, holdout_glob)
    model_win_rate = evaluate_model(state)
    target_win_rate = evaluate_target(state)

    # This could run in parallel to the rest.
    selfplay(state)

    if model_win_rate >= 0.55:
      # Promote the trained model to the play model.
      state.play_model_num = state.train_model_num
      state.play_model_name = state.train_model_name
      state.train_model_num += 1
    elif model_win_rate < 0.4:
      # Bury the selfplay games which produced a significantly worse model.
      logging.info('Burying %s.', tf_records[0])
      shutil.move(tf_records[0], tf_records[0] + '.bury')

    yield target_win_rate


def main(unused_argv):
  """Run the reinforcement learning loop."""

  print('Wiping dir %s' % FLAGS.base_dir, flush=True)
  shutil.rmtree(FLAGS.base_dir, ignore_errors=True)

  utils.ensure_dir_exists(fsdb.models_dir())
  utils.ensure_dir_exists(fsdb.selfplay_dir())
  utils.ensure_dir_exists(fsdb.holdout_dir())
  utils.ensure_dir_exists(fsdb.eval_dir())
  utils.ensure_dir_exists(fsdb.golden_chunk_dir())
  utils.ensure_dir_exists(fsdb.working_dir())

  logging.getLogger().addHandler(
      logging.FileHandler(os.path.join(FLAGS.base_dir, 'reinforcement.log')))
  formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                '%Y-%m-%d %H:%M:%S')
  for handler in logging.getLogger().handlers:
    handler.setFormatter(formatter)

  with utils.logged_timer('Total time'):
    for target_win_rate in rl_loop():
      if target_win_rate > 0.5:
        return logging.info('Passed exit criteria.')
    logging.info('Failed to converge.')


if __name__ == '__main__':
  sys.path.insert(0, '.')
  app.run(main)
