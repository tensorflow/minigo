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

from tensorflow import gfile
import os

from absl import app
from reference_implementation import evaluate_model, wait
from rl_loop import fsdb


def load_train_times():
  models = []
  path = os.path.join(fsdb.models_dir(), 'train_times.txt')
  with gfile.Open(path, 'r') as f:
    for line in f.readlines():
      line = line.strip()
      if line:
        timestamp, name = line.split(' ')
        path = 'tf,' + os.path.join(fsdb.models_dir(), name + '.pb')
        models.append((float(timestamp), name, path))
  return models


def main(unused_argv):
  sgf_dir = os.path.join(fsdb.eval_dir(), 'target')
  target = 'tf,' + os.path.join(fsdb.models_dir(), 'target.pb')
  models = load_train_times()
  for i, (timestamp, name, path) in enumerate(models):
    winrate = wait(evaluate_model(path, target, sgf_dir, i + 1))
    if winrate >= 0.50:
      print('Model {} beat target after {}s'.format(name, timestamp))
      break


if __name__ == '__main__':
  app.run(main)
