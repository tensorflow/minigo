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

import asyncio
import logging
import numpy as np
import os
import random
import re
import shutil
import subprocess
import tensorflow as tf
import time
import utils

from absl import app, flags
from rl_loop import example_buffer, fsdb
from tensorflow import gfile

flags.DEFINE_integer('iterations', 100, 'Number of iterations of the RL loop.')

flags.DEFINE_float('gating_win_rate', 0.55,
                   'Win-rate against the current best required to promote a '
                   'model to new best.')

flags.DEFINE_string('flags_dir', None,
                    'Directory in which to find the flag files for each stage '
                    'of the RL loop. The directory must contain the following '
                    'files: bootstrap.flags, selfplay.flags, eval.flags, '
                    'train.flags.')

flags.DEFINE_integer('max_window_size', 5,
                     'Maximum number of recent selfplay rounds to train on.')

flags.DEFINE_integer('slow_window_size', 5,
                     'Window size after which the window starts growing by '
                     '1 every slow_window_speed iterations of the RL loop.')

flags.DEFINE_integer('slow_window_speed', 1,
                     'Speed at which the training window increases in size '
                     'once the window size passes slow_window_size.')

flags.DEFINE_boolean('parallel_post_train', False,
                     'If true, run the post-training stages (eval, validation '
                     '& selfplay) in parallel.')

FLAGS = flags.FLAGS


class State:
  """State data used in each iteration of the RL loop.

  Models are named with the current reinforcement learning loop iteration number
  and the model generation (how many models have passed gating). For example, a
  model named "000015-000007" was trained on the 15th iteration of the loop and
  is the 7th models that passed gating.
  Note that we rely on the iteration number being the first part of the model
  name so that the training chunks sort correctly.
  """

  def __init__(self):
    self.start_time = time.time()

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


def expand_cmd_str(cmd):
  return '  '.join(flags.FlagValues().read_flags_from_files(cmd))


def get_cmd_name(cmd):
  if cmd[0] == 'python' or cmd[0] == 'python3':
    return ' '.join(cmd[:2])
  else:
    return cmd[0]


async def checked_run(*cmd):
  """Run the given subprocess command in a coroutine.

  Args:
    *cmd: the command to run and its arguments.

  Returns:
    The output that the command wrote to stdout as a list of strings, one line
    per element (stderr output is piped to stdout).

  Raises:
    RuntimeError: if the command returns a non-zero result.
  """

  # Start the subprocess.
  logging.info('Running: %s', expand_cmd_str(cmd))
  with utils.logged_timer('{} finished'.format(get_cmd_name(cmd))):
    p = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)

    # Stream output from the process stdout.
    chunks = []
    while True:
      chunk = await p.stdout.read(16 * 1024)
      if not chunk:
        break
      chunks.append(chunk)

    # Wait for the process to finish, check it was successful & build stdout.
    await p.wait()
    stdout = b''.join(chunks).decode()[:-1]
    if p.returncode:
      raise RuntimeError('Return code {} from process: {}\n{}'.format(
          p.returncode, expand_cmd_str(cmd), stdout))

    # Split stdout into lines.
    return stdout.split('\n')


def wait(aws):
  """Waits for all of the awaitable objects (e.g. coroutines) in aws to finish.

  All the awaitable objects are waited for, even if one of them raises an
  exception. When one or more awaitable raises an exception, the exception from
  the awaitable with the lowest index in the aws list will be reraised.

  Args:
    aws: a single awaitable, or list awaitables.

  Returns:
    If aws is a single awaitable, its result.
    If aws is a list of awaitables, a list containing the of each awaitable in
    the list.

  Raises:
    Exception: if any of the awaitables raises.
  """

  aws_list = aws if isinstance(aws, list) else [aws]
  results = asyncio.get_event_loop().run_until_complete(asyncio.gather(
      *aws_list, return_exceptions=True))
  # If any of the cmds failed, re-raise the error.
  for result in results:
    if isinstance(result, Exception):
      raise result
  return results if isinstance(aws, list) else results[0]


def get_golden_chunk_records(num_records):
  """Return up to num_records of golden chunks to train on.

  Args:
    num_records: maximum number of records to return.

  Returns:
    A list of golden chunks up to num_records in length, sorted by path.
  """

  pattern = os.path.join(fsdb.golden_chunk_dir(), '*.zz')
  return sorted(tf.gfile.Glob(pattern), reverse=True)[:num_records]


# Self-play a number of games.
async def selfplay(state, flagfile='selfplay'):
  """Run selfplay and write a training chunk to the fsdb golden_chunk_dir.

  Args:
    state: the RL loop State instance.
    flagfile: the name of the flagfile to use for selfplay, either 'selfplay'
        (the default) or 'boostrap'.
  """

  output_dir = os.path.join(fsdb.selfplay_dir(), state.output_model_name)
  holdout_dir = os.path.join(fsdb.holdout_dir(), state.output_model_name)
  model_path = os.path.join(fsdb.models_dir(), state.best_model_name)

  lines = await checked_run(
      'bazel-bin/cc/selfplay',
      '--flagfile={}.flags'.format(os.path.join(FLAGS.flags_dir, flagfile)),
      '--model={}.pb'.format(model_path),
      '--output_dir={}'.format(output_dir),
      '--holdout_dir={}'.format(holdout_dir),
      '--seed={}'.format(state.seed))
  result = '\n'.join(lines[-2:])
  logging.info(result)

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


async def train(state, tf_records):
  """Run training and write a new model to the fsdb models_dir.

  Args:
    state: the RL loop State instance.
    tf_records: a list of paths to TensorFlow records to train on.
  """

  model_path = os.path.join(fsdb.models_dir(), state.train_model_name)
  await checked_run(
      'python3', 'train.py', *tf_records,
      '--flagfile={}'.format(os.path.join(FLAGS.flags_dir, 'train.flags')),
      '--work_dir={}'.format(fsdb.working_dir()),
      '--export_path={}'.format(model_path),
      '--training_seed={}'.format(state.seed),
      '--freeze=true')
  # Append the time elapsed from when the RL was started to when this model
  # was trained.
  elapsed = time.time() - state.start_time
  timestamps_path = os.path.join(fsdb.models_dir(), 'train_times.txt')
  with gfile.Open(timestamps_path, 'a') as f:
    print('{:.3f} {}'.format(elapsed, state.train_model_name), file=f)


async def validate(state, holdout_glob):
  """Validate the trained model against holdout games.

  Args:
    state: the RL loop State instance.
    holdout_glob: a glob that matches holdout games.
  """

  await checked_run(
      'python3', 'validate.py', holdout_glob,
      '--flagfile={}'.format(os.path.join(FLAGS.flags_dir, 'validate.flags')),
      '--work_dir={}'.format(fsdb.working_dir()))


async def evaluate_model(eval_model, target_model, sgf_dir, seed):
  """Evaluate one model against a target.

  Args:
    eval_model: the name of the model to evaluate.
    target_model: the name of the model to compare to.
    sgf_dif: directory path to write SGF output to.
    seed: random seed to use when running eval.

  Returns:
    The win-rate of eval_model against target_model in the range [0, 1].
  """

  eval_model_path = os.path.join(fsdb.models_dir(), eval_model)
  target_model_path = os.path.join(fsdb.models_dir(), target_model)
  lines = await checked_run(
      'bazel-bin/cc/eval',
      '--flagfile={}'.format(os.path.join(FLAGS.flags_dir, 'eval.flags')),
      '--model={}.pb'.format(eval_model_path),
      '--model_two={}.pb'.format(target_model_path),
      '--sgf_dir={}'.format(sgf_dir),
      '--seed={}'.format(seed))
  result = '\n'.join(lines[-7:])
  logging.info(result)
  pattern = '{}\s+\d+\s+(\d+\.\d+)%'.format(eval_model)
  win_rate = float(re.search(pattern, result).group(1)) * 0.01
  logging.info('Win rate %s vs %s: %.3f', eval_model, target_model, win_rate)
  return win_rate


async def evaluate_trained_model(state):
  """Evaluate the most recently trained model against the current best model.

  Args:
    state: the RL loop State instance.
  """

  return await evaluate_model(
      state.train_model_name, state.best_model_name,
      os.path.join(fsdb.eval_dir(), state.train_model_name), state.seed)


def rl_loop():
  """The main reinforcement learning (RL) loop."""

  state = State()

  # Play the first round of selfplay games with a fake model that returns
  # random noise. We do this instead of playing multiple games using a single
  # model bootstrapped with random noise to avoid any initial bias.
  wait(selfplay(state, 'bootstrap'))

  # Train a real model from the random selfplay games.
  tf_records = get_golden_chunk_records(1)
  state.iter_num += 1
  wait(train(state, tf_records))

  # Select the newly trained model as the best.
  state.best_model_name = state.train_model_name
  state.gen_num += 1

  # Run selfplay using the new model.
  wait(selfplay(state))

  # Now start the full training loop.
  while state.iter_num <= FLAGS.iterations:
    # Build holdout glob before incrementing the iteration number because we
    # want to run validation on the previous generation.
    holdout_glob = os.path.join(fsdb.holdout_dir(), '%06d-*' % state.iter_num,
                                '*')

    # Calculate the window size from which we'll select training chunks.
    window = 1 + state.iter_num
    if window >= FLAGS.slow_window_size:
      window = (FLAGS.slow_window_size +
                (window - FLAGS.slow_window_size) // FLAGS.slow_window_speed)
    window = min(window, FLAGS.max_window_size)

    # Train on shuffled game data from recent selfplay rounds.
    tf_records = get_golden_chunk_records(window)
    state.iter_num += 1
    wait(train(state, tf_records))

    if FLAGS.parallel_post_train:
      # Run eval, validation & selfplay in parallel.
      model_win_rate, _, _ = wait([
          evaluate_trained_model(state),
          validate(state, holdout_glob),
          selfplay(state)])
    else:
      # Run eval, validation & selfplay sequentially.
      model_win_rate = wait(evaluate_trained_model(state))
      wait(validate(state, holdout_glob))
      wait(selfplay(state))

    # TODO(tommadams): if a model doesn't get promoted after N iterations,
    # consider deleting the most recent N training checkpoints because training
    # might have got stuck in a local minima.
    if model_win_rate >= FLAGS.gating_win_rate:
      # Promote the trained model to the best model and increment the generation
      # number.
      state.best_model_name = state.train_model_name
      state.gen_num += 1


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
    try:
      rl_loop()
    finally:
      asyncio.get_event_loop().close()


if __name__ == '__main__':
  app.run(main)
