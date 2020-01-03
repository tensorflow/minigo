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

"""Evaluates a directory of models against the target model."""

import sys
sys.path.insert(0, '.')  # nopep8

import tensorflow as tf
import os

from ml_perf.utils import *

from absl import app, flags


flags.DEFINE_integer('start', 0, 'First model generation to evaluate.')
flags.DEFINE_integer('step', 1,
                     'Number of generations to advance each iteration.')
flags.DEFINE_integer('num_games', 128, 'Number of games to run.')
flags.DEFINE_string('flags_dir', '', 'Flags directory.')
flags.DEFINE_string('model_dir', '', 'Model directory.')
flags.DEFINE_string('target', '', 'Path of the target model.')
flags.DEFINE_string('sgf_dir', '', 'Directory to write SGFs to.')
flags.DEFINE_list('devices', '', 'List of devices to run on.')
flags.DEFINE_float('winrate', 0.5,
                   'Fraction of games that a model must beat the target by.')

FLAGS = flags.FLAGS


class ColorWinStats:
    """Win-rate stats for a single model & color."""

    def __init__(self, total, both_passed, opponent_resigned):
        self.total = total
        self.both_passed = both_passed
        self.opponent_resigned = opponent_resigned
        # Verify that the total is correct
        assert total == both_passed + opponent_resigned


class WinStats:
    """Win-rate stats for a single model."""

    def __init__(self, line):
        pattern = '\s*(\S+)' + '\s+(\d+)' * 6
        match = re.search(pattern, line)
        if match is None:
            raise ValueError('Can\t parse line "{}"'.format(line))
        self.model_name = match.group(1)
        raw_stats = [float(x) for x in match.groups()[1:]]
        self.black_wins = ColorWinStats(*raw_stats[:3])
        self.white_wins = ColorWinStats(*raw_stats[3:])
        self.total_wins = self.black_wins.total + self.white_wins.total


def load_train_times():
  models = []
  path = os.path.join(FLAGS.model_dir, 'train_times.txt')
  with tf.io.gfile.GFile(path, 'r') as f:
    for line in f.readlines():
      line = line.strip()
      if line:
        timestamp, name = line.split(' ')
        path = os.path.join(FLAGS.model_dir, name + '.minigo')
        models.append((float(timestamp), name, path))
  return models


def parse_win_stats_table(lines):
    result = []
    while True:
        # Find the start of the win stats table.
        assert len(lines) > 1
        if 'Black' in lines[0] and 'White' in lines[0] and 'passes' in lines[1]:
            break
        lines = lines[1:]

    # Parse the expected number of lines from the table.
    for line in lines[2:4]:
        result.append(WinStats(line))

    return result


def evaluate_model(eval_model_path):
    processes = []
    for i, device in enumerate(FLAGS.devices):
        a = i * FLAGS.num_games // len(FLAGS.devices)
        b = (i + 1) * FLAGS.num_games // len(FLAGS.devices)
        num_games = b - a;
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = device
        processes.append(checked_run([
            'bazel-bin/cc/eval',
            '--flagfile={}'.format(os.path.join(FLAGS.flags_dir, 'eval.flags')),
            '--eval_model={}'.format(eval_model_path),
            '--target_model={}'.format(FLAGS.target),
            '--sgf_dir={}'.format(FLAGS.sgf_dir),
            '--parallel_games={}'.format(num_games),
            '--verbose=false'], env))
    all_output = wait(processes)

    total_wins = 0
    total_num_games = 0
    for output in all_output:
        lines = output.split('\n')

        eval_stats, target_stats = parse_win_stats_table(lines[-7:])
        num_games = eval_stats.total_wins + target_stats.total_wins
        total_wins += eval_stats.total_wins
        total_num_games += num_games

    win_rate = total_wins / total_num_games
    logging.info('Win rate %s vs %s: %.3f', eval_stats.model_name,
                 target_stats.model_name, win_rate)
    return win_rate


def main(unused_argv):
    models = load_train_times()

    # Skip all models earlier than start and apply step.
    models = [x for x in models if int(x[1]) >= FLAGS.start][::FLAGS.step]

    for i, (timestamp, name, path) in enumerate(models):
        winrate = evaluate_model(path)
        if winrate >= FLAGS.winrate:
            print('Model {} beat target after {}s'.format(name, timestamp))
            break


if __name__ == '__main__':
  app.run(main)

