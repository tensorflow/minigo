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
import glob
import logging
import numpy as np
import os
import random
import re
import signal
import shutil
import subprocess
import tensorflow as tf
import time
from ml_perf.utils import *

from absl import app, flags
from rl_loop import fsdb
from tensorflow import gfile

N = int(os.environ.get('BOARD_SIZE', 19))

flags.DEFINE_string('checkpoint_dir', 'ml_perf/checkpoint/{}'.format(N),
                    'The checkpoint directory specify a start model and a set '
                    'of golden chunks used to start training.  If not '
                    'specified, will start from scratch.')

flags.DEFINE_string('target_path', 'ml_perf/target/{}/target.pb'.format(N),
                    'Path to the target model to beat.')

flags.DEFINE_integer('iterations', 100, 'Number of iterations of the RL loop.')

flags.DEFINE_float('gating_win_rate', 0.55,
                   'Win-rate against the current best required to promote a '
                   'model to new best.')

flags.DEFINE_float('bootstrap_target_win_rate', 0.05,
                   'Win-rate against the target required to halt '
                   'bootstrapping and generate the starting training '
                   'checkpoint')

flags.DEFINE_string('flags_dir', None,
                    'Directory in which to find the flag files for each stage '
                    'of the RL loop. The directory must contain the following '
                    'files: bootstrap.flags, selfplay.flags, eval.flags, '
                    'train.flags.')

flags.DEFINE_integer('window_size', 10,
                     'Maximum number of recent selfplay rounds to train on.')

flags.DEFINE_float('train_filter', 1,
                   'Fraction of selfplay games to pass to training.')

flags.DEFINE_boolean('parallel_post_train', False,
                     'If true, run the post-training stages (eval, validation '
                     '& selfplay) in parallel.')

flags.DEFINE_list('train_devices', None, '')
flags.DEFINE_string('eval_device', None, '')
flags.DEFINE_list('selfplay_devices', None, '')

flags.DEFINE_integer('bootstrap_num_models', 8,
                     'Number of random models to use for bootstrapping.')
flags.DEFINE_integer('eval_num_games', 100,
                     'Number of selfplay games to play.')

flags.DEFINE_boolean('bootstrap', False, '')

flags.DEFINE_boolean('validate', False, 'Run validation on holdout games')

flags.DEFINE_boolean('use_extra_features', False,
                     'Use non-Zero input features')

flags.DEFINE_integer('num_games_per_iteration', 4096,
                     'Minimum number of games to play for each training '
                     'iteration.')

FLAGS = flags.FLAGS


class State:
    def __init__(self):
        self.start_time = time.time()
        self.iter_num = 0

    @property
    def train_model_name(self):
        return '%06d' % self.iter_num

    @property
    def train_model_path(self):
        return '{}.pb'.format(
            os.path.join(fsdb.models_dir(), self.train_model_name))


class ColorWinStats:
    """Win-rate stats for a single model & color."""

    def __init__(self, total, both_passed, opponent_resigned,
                 move_limit_reached):
        self.total = total
        self.both_passed = both_passed
        self.opponent_resigned = opponent_resigned
        self.move_limit_reached = move_limit_reached
        # Verify that the total is correct
        assert total == both_passed + opponent_resigned + move_limit_reached


class WinStats:
    """Win-rate stats for a single model."""

    def __init__(self, line):
        pattern = '\s*(\S+)' + '\s+(\d+)' * 8
        match = re.search(pattern, line)
        if match is None:
            raise ValueError('Can\t parse line "{}"'.format(line))
        self.model_name = match.group(1)
        raw_stats = [float(x) for x in match.groups()[1:]]
        self.black_wins = ColorWinStats(*raw_stats[:4])
        self.white_wins = ColorWinStats(*raw_stats[4:])
        self.total_wins = self.black_wins.total + self.white_wins.total


### def initialize_from_checkpoint(state):
###     """Initialize the reinforcement learning loop from a checkpoint."""
###
###     # The checkpoint's work_dir should contain the most recently trained model.
###     model_paths = glob.glob(os.path.join(FLAGS.checkpoint_dir,
###                                          'work_dir/model.ckpt-*.pb'))
###     if len(model_paths) != 1:
###         raise RuntimeError('Expected exactly one model in the checkpoint '
###                            'work_dir, got [{}]'.format(', '.join(model_paths)))
###     start_model_path = model_paths[0]
###
###     # Copy the latest trained model into the models directory and use it on the
###     # first round of selfplay.
###     state.best_model_name = 'checkpoint'
###     shutil.copy(start_model_path,
###                 os.path.join(fsdb.models_dir(), state.best_model_name + '.pb'))
###
###     # Copy the training chunks.
###     golden_chunks_dir = os.path.join(FLAGS.checkpoint_dir, 'golden_chunks')
###     for basename in os.listdir(golden_chunks_dir):
###         path = os.path.join(golden_chunks_dir, basename)
###         shutil.copy(path, fsdb.golden_chunk_dir())
###
###     # Copy the training files.
###     work_dir = os.path.join(FLAGS.checkpoint_dir, 'work_dir')
###     for basename in os.listdir(work_dir):
###         path = os.path.join(work_dir, basename)
###         shutil.copy(path, fsdb.working_dir())


### def create_checkpoint():
###     for sub_dir in ['work_dir', 'golden_chunks']:
###         ensure_dir_exists(os.path.join(FLAGS.checkpoint_dir, sub_dir))
###
###     # List all the training checkpoints.
###     pattern = os.path.join(FLAGS.base_dir, 'work_dir', 'model.ckpt-*.index')
###     model_paths = glob.glob(pattern)
###
###     # Sort the checkpoints by step number.
###     def extract_step(path):
###         name = os.path.splitext(os.path.basename(path))[0]
###         return int(re.match('model.ckpt-(\d+)', name).group(1))
###     model_paths.sort(key=lambda x: extract_step(x))
###
###     # Get the name of the latest checkpoint.
###     step = extract_step(model_paths[-1])
###     name = 'model.ckpt-{}'.format(step)
###
###     # Copy the model to the checkpoint directory.
###     for ext in ['.data-00000-of-00001', '.index', '.meta']:
###         basename = name + ext
###         src_path = os.path.join(FLAGS.base_dir, 'work_dir', basename)
###         dst_path = os.path.join(FLAGS.checkpoint_dir, 'work_dir', basename)
###         print('Copying {} {}'.format(src_path, dst_path))
###         shutil.copy(src_path, dst_path)
###
###     # Write the checkpoint state proto.
###     checkpoint_path = os.path.join(
###         FLAGS.checkpoint_dir, 'work_dir', 'checkpoint')
###     print('Writing {}'.format(checkpoint_path))
###     with gfile.GFile(checkpoint_path, 'w') as f:
###         f.write('model_checkpoint_path: "{}"\n'.format(name))
###         f.write('all_model_checkpoint_paths: "{}"\n'.format(name))
###
###     # Copy the most recent golden chunks.
###     pattern = os.path.join(FLAGS.base_dir, 'data',
###                            'golden_chunks', '*.tfrecord.zz')
###     src_paths = sorted(glob.glob(pattern))[-FLAGS.window_size:]
###     for i, src_path in enumerate(src_paths):
###         dst_path = os.path.join(FLAGS.checkpoint_dir, 'golden_chunks',
###                                 '000000-{:06}.tfrecord.zz'.format(i))
###         print('Copying {} {}'.format(src_path, dst_path))
###         shutil.copy(src_path, dst_path)


def parse_win_stats_table(stats_str, num_lines):
    result = []
    lines = stats_str.split('\n')
    while True:
        # Find the start of the win stats table.
        assert len(lines) > 1
        if 'Black' in lines[0] and 'White' in lines[0] and 'm.lmt.' in lines[1]:
            break
        lines = lines[1:]

    # Parse the expected number of lines from the table.
    for line in lines[2:2 + num_lines]:
        result.append(WinStats(line))

    return result


async def run(*cmd):
    """Run the given subprocess command in a coroutine.

    Args:
        *cmd: the command to run and its arguments.

    Returns:
        The output that the command wrote to stdout as a list of strings, one line
        per element (stderr output is piped to stdout).

    Raises:
        RuntimeError: if the command returns a non-zero result.
    """

    stdout = await checked_run(*cmd)

    log_path = os.path.join(FLAGS.base_dir, get_cmd_name(cmd) + '.log')
    with gfile.Open(log_path, 'a') as f:
        f.write(await expand_cmd_str(cmd))
        f.write('\n')
        f.write(stdout)
        f.write('\n')

    # Split stdout into lines.
    return stdout.split('\n')


def wait_for_training_examples(state, num_games):
    """Wait for training examples to be generated by the latest model.

    Args:
        state: the RL loop State instance.
        num_games: number of games to wait for.
    """

    first_time_around = True
    while True:
        model_dirs = list(os.scandir(fsdb.selfplay_dir()))
        if len(model_dirs) == state.iter_num:
            pattern = os.path.join(model_dirs[-1], '*', '*', '*.tfrecord.zz')
            paths = sorted(tf.gfile.Glob(pattern))
            if len(paths) >= num_games:
                break
        if first_time_around:
            logging.info('Waiting for %d games', num_games)
            first_time_around = False
        time.sleep(1)


async def sample_training_examples(state):
    """Sample training examples from recent selfplay games.

    Args:
        state: the RL loop State instance.

    Returns:
        A list of golden chunks up to num_records in length, sorted by path.
    """

    # Training examples are written out to the following directory hierarchy:
    #   selfplay_dir/device_id/model_name/timestamp/
    # Read examples from the most recent `window_size` models.
    device_dirs = [x.path
                   for x in os.scandir(fsdb.selfplay_dir())
                   if x.is_dir()]
    models = set()
    for d in device_dirs:
        models.update([x.name for x in os.scandir(d) if x.is_dir()])
    models = sorted(models, reverse=True)[:FLAGS.window_size]

    src_patterns = []
    for d in device_dirs:
        for model in models:
            src_patterns.append(os.path.join(d, model, '*', '*.tfrecord.zz'))

    dst_path = os.path.join(fsdb.golden_chunk_dir(),
                            '{}.tfrecord.zz'.format(state.train_model_name))

    logging.info('Writing training chunks to %s', dst_path)
    lines = await sample_records(src_patterns, dst_path,
                                 num_read_threads=8,
                                 num_write_threads=8,
                                 sample_frac=FLAGS.train_filter)
    logging.info('\n'.join(lines))

    chunk_pattern = os.path.join(
        fsdb.golden_chunk_dir(),
        '{}-*-of-*.tfrecord.zz'.format(state.train_model_name))
    chunk_paths = sorted(tf.gfile.Glob(chunk_pattern))
    assert len(chunk_paths) == 8

    return chunk_paths


async def run_commands(commands):
    all_tasks = []
    loop = asyncio.get_event_loop()
    for cmd in commands:
        all_tasks.append(loop.create_task(run(*cmd)))
    all_results = await asyncio.gather(*all_tasks, return_exceptions=True)
    for cmd, result in zip(commands, all_results):
        if isinstance(result, Exception):
            logging.error('error running command:\n  %s',
                          ' '.join([str(x) for x in cmd]))
            raise result
    return all_results


async def bootstrap_selfplay(state):
    output_dir = os.path.join(fsdb.selfplay_dir(), state.train_model_name)
    holdout_dir = os.path.join(fsdb.holdout_dir(), state.train_model_name)

    features = 'extra' if FLAGS.use_extra_features else 'agz'
    lines = await run(
        'bazel-bin/cc/concurrent_selfplay',
        '--flagfile={}'.format(os.path.join(FLAGS.flags_dir,
                                            'bootstrap.flags')),
        '--model={}:0.4:0.4'.format(features),
        '--num_games={}'.format(FLAGS.num_games_per_iteration),
        '--output_dir={}/0'.format(output_dir),
        '--holdout_dir={}/0'.format(holdout_dir))
    logging.info('\n'.join(lines[-6:]))


# Self-play a number of games.
### async def selfplay(state):
###     """Run selfplay and write a training chunk to the fsdb golden_chunk_dir.
###
###     Args:
###         state: the RL loop State instance.
###     """
###
###     output_dir = os.path.join(fsdb.selfplay_dir(), state.output_model_name)
###     holdout_dir = os.path.join(fsdb.holdout_dir(), state.output_model_name)
###
###     commands = []
###     num_selfplay_processes = len(FLAGS.selfplay_devices)
###     n = max(num_selfplay_processes, 1)
###     for i, device in enumerate(FLAGS.selfplay_devices):
###         a = (i * FLAGS.selfplay_num_games) // n
###         b = ((i + 1) * FLAGS.selfplay_num_games) // n
###         num_games = b - a
###
###         commands.append([
###             'bazel-bin/cc/concurrent_selfplay',
###             '--flagfile={}'.format(os.path.join(FLAGS.flags_dir,
###                                                 'selfplay.flags')),
###             '--num_games={}'.format(num_games),
###             '--device={}'.format(device),
###             '--model={}'.format(state.best_model_path),
###             '--output_dir={}/{}'.format(output_dir, i),
###             '--holdout_dir={}/{}'.format(holdout_dir, i)])
###
###     all_lines = await run_commands(commands)
###
###     black_wins_total = white_wins_total = num_games = 0
###     for lines in all_lines:
###         result = '\n'.join(lines[-6:])
###         logging.info(result)
###         stats = parse_win_stats_table(result, 1)[0]
###         num_games += stats.total_wins
###         black_wins_total += stats.black_wins.total
###         white_wins_total += stats.white_wins.total
###
###     logging.info('Black won %0.3f, white won %0.3f',
###                  black_wins_total / num_games,
###                  white_wins_total / num_games)
###
###     ### # Write examples to a single record.
###     ### src_pattern = os.path.join(output_dir, '*/*', '*.tfrecord.zz')
###     ### dst_path = os.path.join(
###     ###     output_dir, state.output_model_name + '.tfrecord.zz')
###     ### logging.info('Writing selfplay records to "{}" from "{}"'.format(
###     ###     dst_path, src_pattern))
###     ### lines = await sample_records(src_pattern, dst_path,
###     ###                              num_read_threads=len(FLAGS.selfplay_devices),
###     ###                              num_write_threads=4, sample_frac=1)
###     ### logging.info('\n'.join(lines))


def model_name(it):
    return '%06d' % it


async def train(state, tf_records):
    """Run training and write a new model to the fsdb models_dir.

    Args:
        state: the RL loop State instance.
        tf_records: a list of paths to TensorFlow records to train on.
    """
    model_path = os.path.join(fsdb.models_dir(), state.train_model_name)

    await run(
        'python3', 'train.py',
        '--gpu_device_list={}'.format(','.join(FLAGS.train_devices)),
        '--flagfile={}'.format(os.path.join(FLAGS.flags_dir, 'train.flags')),
        '--work_dir={}'.format(fsdb.working_dir()),
        '--export_path={}'.format(model_path),
        '--use_extra_features={}'.format(FLAGS.use_extra_features),
        '--freeze=true',
        *tf_records)

    # Append the time elapsed from when the RL was started to when this model
    # was trained.
    elapsed = time.time() - state.start_time
    timestamps_path = os.path.join(fsdb.models_dir(), 'train_times.txt')
    with gfile.Open(timestamps_path, 'a') as f:
        print('{:.3f} {}'.format(elapsed, state.train_model_name), file=f)

    if FLAGS.validate:
      await validate(state)


async def validate(state):
    dirs = [x.path for x in os.scandir(fsdb.holdout_dir()) if x.is_dir()]
    src_dirs = sorted(dirs, reverse=True)[:FLAGS.window_size]

    await run('python3', 'validate.py',
              '--gpu_device_list={}'.format(','.join(FLAGS.train_devices)),
              '--flagfile={}'.format(os.path.join(FLAGS.flags_dir,
                                                  'validate.flags')),
              '--work_dir={}'.format(fsdb.working_dir()),
              '--expand_validation_dirs',
              *src_dirs)


async def evaluate_model(eval_model_path, target_model_path, sgf_dir):
    """Evaluate one model against a target.

    Args:
        eval_model_path: the path to the model to evaluate.
        target_model_path: the path to the model to compare to.
        sgf_dif: directory path to write SGF output to.

    Returns:
        The win-rate of eval_model against target_model in the range [0, 1].
    """

    lines = await run(
        'bazel-bin/cc/eval',
        '--flagfile={}'.format(os.path.join(FLAGS.flags_dir, 'eval.flags')),
        '--eval_device={}'.format(FLAGS.eval_device),
        '--target_device={}'.format(FLAGS.eval_device),
        '--eval_model={}'.format(eval_model_path),
        '--target_model={}'.format(target_model_path),
        '--parallel_games={}'.format(FLAGS.eval_num_games),
        '--sgf_dir={}'.format(sgf_dir))
    result = '\n'.join(lines[-7:])
    logging.info(result)
    eval_stats, target_stats = parse_win_stats_table(result, 2)
    num_games = eval_stats.total_wins + target_stats.total_wins
    win_rate = eval_stats.total_wins / num_games
    logging.info('Win rate %s vs %s: %.3f', eval_stats.model_name,
                 target_stats.model_name, win_rate)
    return win_rate


async def sample_records(src_patterns, dst_path, num_read_threads,
                         num_write_threads, sample_frac=0, num_records=0):
    return await run(
        'bazel-bin/cc/sample_records',
        '--num_read_threads={}'.format(num_read_threads),
        '--num_write_threads={}'.format(num_write_threads),
        '--sample_frac={}'.format(sample_frac),
        '--num_records={}'.format(num_records),
        '--compression=1',
        '--shuffle=true',
        '--dst={}'.format(dst_path),
        *src_patterns)


async def start_selfplay():
    output_dir = os.path.join(fsdb.selfplay_dir(), "$MODEL")
    holdout_dir = os.path.join(fsdb.holdout_dir(), "$MODEL")
    model_pattern = os.path.join(fsdb.models_dir(), '%d.pb')

    logs = []
    processes = []
    loop = asyncio.get_event_loop()
    for i, device in enumerate(FLAGS.selfplay_devices):
        cmd = [
            'bazel-bin/cc/concurrent_selfplay',
            '--flagfile={}'.format(os.path.join(FLAGS.flags_dir,
                                                'selfplay.flags')),
            '--run_forever=1',
            '--device={}'.format(device),
            '--model={}'.format(model_pattern),
            '--output_dir={}/{}'.format(output_dir, i),
            '--holdout_dir={}/{}'.format(holdout_dir, i)]

        cmd_str = await expand_cmd_str(cmd)
        f = open(os.path.join(FLAGS.base_dir, 'selfplay_%d.log' % i), 'w')
        f.write(cmd_str + '\n\n')
        f.flush()
        logging.info('Running: %s', cmd_str)

        processes.append(await asyncio.create_subprocess_exec(
            *cmd, stdout=f, stderr=asyncio.subprocess.STDOUT))
        logs.append(f)

    return (processes, logs)


async def end_selfplay(processes, logs):
    logging.info('Sending SIGINT to selfplay processes')
    for p in processes:
        p.send_signal(signal.SIGINT)
    logging.info('Waiting for selfplay processes')
    for p in processes:
        await p.wait()
    logging.info('Closing selfplay logs')
    for f in logs:
        f.close()


def rl_loop():
    """The main reinforcement learning (RL) loop."""

    state = State()
    prev_win_rate_vs_target = 0

    if FLAGS.bootstrap:
        wait(bootstrap_selfplay(state))
    else:
        initialize_from_checkpoint(state)

    # Start the selfplay workers. They will wait for a model to become available
    # in the training directory before starting to play.
    selfplay_processes, selfplay_logs = wait(start_selfplay())

    try:
        # Now start the full training loop.
        while state.iter_num < FLAGS.iterations:
            state.iter_num += 1

            wait_for_training_examples(state, FLAGS.num_games_per_iteration)
            tf_records = wait(sample_training_examples(state))

            wait(train(state, tf_records))

            # If we're bootstrapping a checkpoint, evaluate the newly trained model
            # against the target.
            # TODO(tommadams): evaluate the previously trained model against the
            # target in parallel with training the next model.
            if FLAGS.bootstrap and state.iter_num > 15:
                target_model_path = os.path.join(
                    fsdb.models_dir(), 'target.pb')
                sgf_dir = os.path.join(
                    fsdb.eval_dir(),
                    '{}-vs-target'.format(state.train_model_name))
                win_rate_vs_target = wait(evaluate_model(
                    state.train_model_path, target_model_path, sgf_dir))
                if (win_rate_vs_target >= FLAGS.bootstrap_target_win_rate and
                        prev_win_rate_vs_target > 0):
                    # The tranined model won a sufficient number of games against
                    # the target. Create the checkpoint that will be used to start
                    # the real benchmark and exit.
                    create_checkpoint()
                    break
                prev_win_rate_vs_target = win_rate_vs_target
    finally:
        wait(end_selfplay(selfplay_processes, selfplay_logs))


def main(unused_argv):
    """Run the reinforcement learning loop."""

    print('Wiping dir %s' % FLAGS.base_dir, flush=True)
    shutil.rmtree(FLAGS.base_dir, ignore_errors=True)
    dirs = [fsdb.models_dir(), fsdb.selfplay_dir(), fsdb.holdout_dir(),
            fsdb.eval_dir(), fsdb.golden_chunk_dir(), fsdb.working_dir()]
    for d in dirs:
        ensure_dir_exists(d)

    # Copy the flag files so there's no chance of them getting accidentally
    # overwritten while the RL loop is running.
    flags_dir = os.path.join(FLAGS.base_dir, 'flags')
    shutil.copytree(FLAGS.flags_dir, flags_dir)
    FLAGS.flags_dir = flags_dir

    # Copy the target model to the models directory so we can find it easily.
    shutil.copy(FLAGS.target_path, os.path.join(
        fsdb.models_dir(), 'target.pb'))

    logging.getLogger().addHandler(
        logging.FileHandler(os.path.join(FLAGS.base_dir, 'rl_loop.log')))
    formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                  '%Y-%m-%d %H:%M:%S')
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)

    with logged_timer('Total time'):
        try:
            rl_loop()
            final_ratings = wait(run('python',
                                     'ratings/rate_subdir.py',
                                     fsdb.eval_dir()))
        finally:
            asyncio.get_event_loop().close()

    for line in final_ratings:
        print(line)


if __name__ == '__main__':
    app.run(main)
