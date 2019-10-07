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

"""Validate a network.

Usage:
    python validate.py tfrecord_dir/ tfrecord_dir2/
"""
import os

from absl import app, flags

import dual_net
import preprocessing
import utils

flags.DEFINE_integer('examples_to_validate', 50 * 2048,
                     'Number of examples to run validation on.')

flags.DEFINE_string('validate_name', 'holdout',
                    'Name of validation set (i.e. holdout or human).')

flags.DEFINE_bool('expand_validation_dirs', True,
                  'Whether to expand the input paths by globbing. If false, '
                  'directly read and validate on the given files.')

# From dual_net.py
flags.declare_key_flag('work_dir')
flags.declare_key_flag('use_tpu')
flags.declare_key_flag('num_tpu_cores')

FLAGS = flags.FLAGS


def validate(*tf_records):
    """Validate a model's performance on a set of holdout data."""
    if FLAGS.use_tpu:
        def _input_fn(params):
            return preprocessing.get_tpu_input_tensors(
                params['batch_size'], tf_records, filter_amount=1.0)
    else:
        def _input_fn():
            return preprocessing.get_input_tensors(
                FLAGS.train_batch_size, tf_records, filter_amount=1.0,
                shuffle_examples=False)

    steps = FLAGS.examples_to_validate // FLAGS.train_batch_size
    if FLAGS.use_tpu:
        steps //= FLAGS.num_tpu_cores

    estimator = dual_net.get_estimator()
    with utils.logged_timer("Validating"):
        estimator.evaluate(_input_fn, steps=steps, name=FLAGS.validate_name)


def main(argv):
    """Validate a model's performance on a set of holdout data."""
    _, *validation_paths = argv
    if FLAGS.expand_validation_dirs:
        tf_records = []
        with utils.logged_timer("Building lists of holdout files"):
            dirs = validation_paths
            while dirs:
                d = dirs.pop()
                for path, newdirs, files in os.walk(d):
                    tf_records.extend(os.path.join(path, f) for f in files if f.endswith('.zz'))
                    dirs.extend(os.path.join(path, d) for d in newdirs)

    else:
        tf_records = validation_paths

    if not tf_records:
        print("Validation paths:", validation_paths)
        print(["{}:\n\t{}".format(p, os.listdir(p)) for p in validation_paths])
        raise RuntimeError("Did not find any holdout files for validating!")
    validate(*tf_records)


if __name__ == "__main__":
    app.run(main)
