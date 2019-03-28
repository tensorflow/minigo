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

"""Copies & compiles the data required to start the reinforcement loop."""

import sys
sys.path.insert(0, '.')  # nopep8

import asyncio
import os

from absl import app, flags
from ml_perf import utils

N = int(os.environ.get('BOARD_SIZE', 19))

FLAGS = flags.FLAGS

flags.DEFINE_string('src_dir', 'gs://minigo-pub/ml_perf/',
                    'Directory on GCS to copy source data from.')

flags.DEFINE_string('dst_dir', 'ml_perf/', 'Local directory to write to.')

flags.DEFINE_boolean('use_tpu', False,
                     'Set to true to generate models that can run on Cloud TPU')


def freeze_graph(path):
  utils.wait(utils.checked_run(
      'python', 'freeze_graph.py',
      '--model_path={}'.format(path), '--use_tpu={}'.format(FLAGS.use_tpu)))


def main(unused_argv):
  try:
    # Pull the required training checkpoints and models from GCS.
    for d in ['checkpoint', 'target']:
      src = os.path.join(FLAGS.src_dir, d, str(N))
      dst = os.path.join(FLAGS.dst_dir, d)
      utils.ensure_dir_exists(dst)

      utils.wait(utils.checked_run('gsutil', '-m', 'cp', '-r', src, dst))

    # Freeze the models to protos.
    freeze_graph('ml_perf/target/{}/target'.format(N))
    freeze_graph('ml_perf/checkpoint/{}/start'.format(N))
  finally:
    asyncio.get_event_loop().close()


if __name__ == '__main__':
  app.run(main)

