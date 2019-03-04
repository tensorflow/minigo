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
sys.path.insert(0, '.')  # nopep8

import logging
import numpy as np
import os
import random
import re
import shutil
import subprocess
import tensorflow as tf
import utils

from absl import app, flags
from rl_loop import example_buffer, fsdb

flags.DEFINE_string('engine', 'tf', 'Engine to use for inference.')

FLAGS = flags.FLAGS


# Models are named with the current reinforcement learning loop iteration number
# and the model generation (how many models have passed gating). For example, a
# model named "000015-000007" was trained on the 15th iteration of the loop and
# is the 7th models that passed gating.
# Note that we rely on the iteration number being the first part of the model
# name so that the training chunks sort correctly.
class State:

  def __init__(self):
    self.iter_num = 0
    self.gen_num = 0

    # We start playing using a random model.
    # After the first round of selfplay has been completed, the engine is
    # updated to FLAGS.engine.
    self.engine = 'random'

    self.best_model_name = 'random'

  @property
  def output_model_name(self):
    return '%06d-%06d' % (self.iter_num, self.gen_num)

  @property
  def train_model_name(self):
    return '%06d-%06d' % (self.iter_num, self.gen_num + 1)

  @property
  def seed(self):
    return self.iter_num + 1


def checked_run(cmd, name):
  logging.info('Running %s:\n  %s', name, '  '.join(cmd))
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


# TODO(tommadams): replace cc_flags and py_flags with a single hyperparams
# flagfile.
def cc_flags(state):
  return [
      '--engine={}'.format(state.engine),
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


# Return up to num_records of golden chunks to train on.
def get_golden_chunk_records(num_records):
  # Sort the list of chunks so that the most recent ones are first and return
  # the requested prefix.
  pattern = os.path.join(fsdb.golden_chunk_dir(), '*.zz')
  return sorted(tf.gfile.Glob(pattern), reverse=True)[:num_records]


# Self-play a number of games.
def selfplay(state):
  output_dir = os.path.join(fsdb.selfplay_dir(), state.output_model_name)
  holdout_dir = os.path.join(fsdb.holdout_dir(), state.output_model_name)
  model_path = os.path.join(fsdb.models_dir(), state.best_model_name)

  result = checked_run([
      'bazel-bin/cc/selfplay', '--parallel_games=2048',
      '--num_readouts=100', '--model={}.pb'.format(model_path),
      '--output_dir={}'.format(output_dir),
      '--holdout_dir={}'.format(holdout_dir)
  ] + cc_flags(state), 'selfplay')
  logging.info(get_lines(result, make_slice[-2:]))

  # Write examples to a single record.
  pattern = os.path.join(output_dir, '*', '*.zz')
  random.seed(state.seed)
  tf.set_random_seed(state.seed)
  np.random.seed(state.seed)
  # TODO(tommadams): This method of generating one golden chunk per generation
  # is sub-optimal because each chunk gets reused multiple times for training,
  # introducing bias. Instead, a fresh dataset should be uniformly sampled out
  # of *all* games in the training window before the start of each training run.
  buffer = example_buffer.ExampleBuffer(sampling_frac=1.0)

  # TODO(tommadams): parallel_fill is currently non-deterministic. Make it not
  # so.
  logging.info('Writing golden chunk from "{}"'.format(pattern))
  buffer.parallel_fill(tf.gfile.Glob(pattern))
  buffer.flush(os.path.join(fsdb.golden_chunk_dir(),
                            state.output_model_name + '.tfrecord.zz'))


# Train a new model.
def train(state, tf_records):
  model_path = os.path.join(fsdb.models_dir(), state.train_model_name)
  result = checked_run([
      'python3', 'train.py', *tf_records, '--export_path={}'.format(model_path),
      '--freeze=true'] + py_flags(state), 'training')


# Validate the trained model against holdout games.
def validate(state, holdout_glob):
  result = checked_run(
      ['python3', 'validate.py', holdout_glob] + py_flags(state), 'validation')


# Evaluate the trained model against some other model (previous best or target).
def evaluate(state, against_model):
  eval_model = state.train_model_name
  eval_model_path = os.path.join(fsdb.models_dir(), eval_model)
  against_model_path = os.path.join(fsdb.models_dir(), against_model)
  sgf_dir = os.path.join(fsdb.eval_dir(), eval_model)
  result = checked_run([
      'bazel-bin/cc/eval',
      '--num_readouts=100', 
      '--parallel_games=100',
      '--model={}.pb'.format(eval_model_path),
      '--model_two={}.pb'.format(against_model_path),
      '--sgf_dir={}'.format(sgf_dir)
  ] + cc_flags(state), 'evaluation against ' + against_model)
  result = get_lines(result, make_slice[-7:])
  logging.info(result)
  pattern = '{}\s+\d+\s+(\d+\.\d+)%'.format(eval_model)
  win_rate = float(re.search(pattern, result).group(1)) * 0.01
  logging.info('Win rate %s vs %s: %.3f', eval_model, against_model, win_rate)
  return win_rate


def rl_loop():
  state = State()

  # Play the first round of selfplay games with a fake model that returns
  # random noise. We do this instead of playing multiple games using a single
  # model bootstrapped with random noise to avoid any initial bias.
  # TODO(tommadams): disable holdout games for first round of selfplay.
  selfplay(state)
  state.engine = FLAGS.engine

  # Train a real model from the random selfplay games.
  tf_records = get_golden_chunk_records(1)
  state.iter_num += 1
  train(state, tf_records)

  # Select the newly trained model as the best.
  state.best_model_name = state.train_model_name
  state.gen_num += 1

  # Run selfplay using the new model.
  selfplay(state)

  # Now start the full training loop.
  while state.iter_num <= 100:
    # Build holdout glob before incrementing the iteration number because we
    # want to run validation on the previous generation.
    holdout_glob = os.path.join(fsdb.holdout_dir(), '%06d-*' % state.iter_num,
                                '*')

    # Train on shuffled game data of the last 5 selfplay rounds, ignoring the
    # random bootstrapping round.
    # TODO(tommadams): potential improvments:
    #   - "slow window": increment number of models in window by 1 every 2
    #     generations.
    #   - uniformly resample the window each iteration (see TODO in selfplay
    #     for more info).
    tf_records = get_golden_chunk_records(min(5, state.iter_num));
    state.iter_num += 1
    train(state, tf_records)

    # These could all run in parallel.
    validate(state, holdout_glob)
    model_win_rate = evaluate(state, state.best_model_name)
    target_win_rate = evaluate(state, 'target')
    selfplay(state)

    # TODO(tommadams): 0.6 is required for 95% confidence at 100 eval games.
    # TODO(tommadams): if a model doesn't get promoted after N iterations,
    # consider deleting the most recent N training checkpoints because training
    # might have got stuck in a local minima.
    if model_win_rate >= 0.55:
      # Promote the trained model to the best model and increment the generation
      # number.
      state.best_model_name = state.train_model_name
      state.gen_num += 1

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

  # Copy the target model to the models directory so we can find it easily.
  shutil.copy('ml_perf/target.pb', fsdb.models_dir())

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
  app.run(main)
