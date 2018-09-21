"""Validate a network.

Usage:
  BOARD_SIZE=19 python validate.py tfrecord_dir/ tfrecord_dir2/
"""
import os

from absl import app, flags
from tensorflow import gfile

import dual_net
import preprocessing
import utils

flags.DEFINE_integer('examples_to_validate', 50 * 2048,
                     'Number of examples to run validation on.')

flags.DEFINE_string('validate_name', 'selfplay',
                    'Name of validation set (i.e. selfplay or human).')

flags.DEFINE_bool('expand_validation_dirs', True,
                  'Whether to expand the input paths by globbing. If false, '
                  'directly read and validate on the given files.')

# From dual_net.py
flags.declare_key_flag('model_dir')
flags.declare_key_flag('train_batch_size')
flags.declare_key_flag('use_tpu')
flags.declare_key_flag('num_tpu_cores')

FLAGS = flags.FLAGS


def validate(*tf_records):
    """Validate a model's performance on a set of holdout data."""
    if FLAGS.use_tpu:
        def _input_fn(params):
            return preprocessing.get_tpu_input_tensors(
                params['batch_size'],
                tf_records, filter_amount=0.05, shuffle_examples=False)
    else:
        def _input_fn():
            return preprocessing.get_input_tensors(
                FLAGS.train_batch_size, tf_records, filter_amount=0.05,
                shuffle_examples=False)


    steps = FLAGS.examples_to_validate / FLAGS.train_batch_size
    if FLAGS.use_tpu:
        steps /= FLAGS.num_tpu_cores

    estimator = dual_net.get_estimator()
    with utils.logged_timer("Validating"):
        estimator.evaluate(_input_fn, steps=steps, name=FLAGS.validate_name)


def main(argv):
    """Validate a model's performance on a set of holdout data."""
    _, *validation_paths = argv
    if FLAGS.expand_validation_dirs:
        tf_records = []
        with utils.logged_timer("Building lists of holdout files"):
            for record_dir in validation_paths:
                tf_records.extend(gfile.Glob(os.path.join(record_dir, '*.zz')))
    else:
        tf_records = validation_paths

    if not tf_records:
        raise RuntimeError("Did not find any holdout files for validating!")
    validate(*tf_records)

if __name__ == "__main__":
    app.run(main)
