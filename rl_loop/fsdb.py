# Copyright 2018 Google LLC
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

"""Filesystem DB: poor man's worker coordination strategy.

This module works equivalently with local filesystems and GCS.
"""
import os
import sys
sys.path.insert(0, '.')

from absl import flags
from tensorflow import gfile
import re

from rl_loop import shipname

flags.DEFINE_string(
    'base_dir', None,
    'Root directory if using local FS as the database. '
    'Leave blank if using bucket_name.')

flags.DEFINE_string(
    'bucket_name', None,
    'Bucket name if using GCS as the filesystem DB. '
    'Leave blank if using base_dir.')

flags.register_multi_flags_validator(
    ['base_dir', 'bucket_name'],
    lambda flags: bool(flags['base_dir']) != bool(flags['bucket_name']),
    'Exactly one of --base_dir, --bucket_name must be set!')

FLAGS = flags.FLAGS

def switch_base(new_base):
    if FLAGS.base_dir:
        FLAGS.base_dir = new_base
    else:
        FLAGS.bucket_name = new_base

def _with_base(*args):
    def inner():
        base_dir = FLAGS.base_dir or 'gs://{}'.format(FLAGS.bucket_name)
        return os.path.join(base_dir, *args)
    return inner


# Functions to compute various important directories, based on FLAGS input.
working_dir = _with_base('work_dir')
models_dir = _with_base('models')
selfplay_dir = _with_base('data', 'selfplay')
holdout_dir = _with_base('data', 'holdout')
sgf_dir = _with_base('sgf')
eval_dir = _with_base('sgf', 'eval')
golden_chunk_dir = _with_base('data', 'golden_chunks')
flags_path = _with_base('flags.txt')
eval_flags_path = _with_base('eval-flags.txt')


def get_pbs():
    all_pbs = gfile.Glob(os.path.join(models_dir(), '*.pb'))
    return all_pbs


def get_models():
    """Finds all models, returning a list of model number and names
    sorted increasing.

    Returns: [(13, 000013-modelname), (17, 000017-modelname), ...etc]
    """
    all_models = gfile.Glob(os.path.join(models_dir(), '*.meta'))
    model_filenames = [os.path.basename(m) for m in all_models]
    model_numbers_names = sorted([
        (shipname.detect_model_num(m), shipname.detect_model_name(m))
        for m in model_filenames])
    return model_numbers_names


def get_latest_model():
    """Finds the latest model, returning its model number and name

    Returns: (17, 000017-modelname)
    """
    return get_models()[-1]


def get_latest_pb():
    pb = os.path.basename(get_pbs()[-1])
    return shipname.detect_model_num(pb), pb


def get_model(model_num):
    """Given a model number 17, returns its full name 000017-modelname."""
    model_names_by_num = dict(get_models())
    return model_names_by_num[model_num]


def get_hour_dirs(root=None):
    """Gets the directories under selfplay_dir that match YYYY-MM-DD-HH."""
    root = root or selfplay_dir()
    return list(filter(lambda s: re.match(r"\d{4}-\d{2}-\d{2}-\d{2}", s),
                       gfile.ListDirectory(root)))


def get_games(model_name):
    return gfile.Glob(os.path.join(selfplay_dir(), model_name, '*.zz'))


def game_counts(n_back=20):
    """Prints statistics for the most recent n_back models"""
    for _, model_name in get_models[-n_back:]:
        games = get_games(model_name)
        print("Model: {}, Games: {}".format(model_name, len(games)))
