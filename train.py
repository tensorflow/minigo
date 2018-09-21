"""Train a network.

Usage:
  BOARD_SIZE=19 python train.py tfrecord1 tfrecord2 tfrecord3
"""

import logging

from absl import app, flags
import numpy as np
import tensorflow as tf

import dual_net
import preprocessing
import utils

# See www.moderndescartes.com/essays/shuffle_viz for discussion on sizing
flags.DEFINE_integer('shuffle_buffer_size', 2000,
                     'Size of buffer used to shuffle train examples.')

flags.DEFINE_integer('steps_to_train', None,
                     'Number of training steps to take. If not set, iterates '
                     'once over training data.')

flags.DEFINE_string('export_path', None,
                    'Where to export the model after training.')

# From dual_net.py
flags.declare_key_flag('model_dir')
flags.declare_key_flag('train_batch_size')
flags.declare_key_flag('num_tpu_cores')
flags.declare_key_flag('use_tpu')

FLAGS = flags.FLAGS


class EchoStepCounterHook(tf.train.StepCounterHook):
    """A hook that logs steps per second."""
    def _log_and_record(self, elapsed_steps, elapsed_time, global_step):
        s_per_sec = elapsed_steps / elapsed_time
        logging.info("{}: {:.3f} steps per second".format(global_step, s_per_sec))
        super()._log_and_record(elapsed_steps, elapsed_time, global_step)


def compute_update_ratio(weight_tensors, before_weights, after_weights):
    """Compute the ratio of gradient norm to weight norm."""
    deltas = [after - before for after,
              before in zip(after_weights, before_weights)]
    delta_norms = [np.linalg.norm(d.ravel()) for d in deltas]
    weight_norms = [np.linalg.norm(w.ravel()) for w in before_weights]
    ratios = [d / w for d, w in zip(delta_norms, weight_norms)]
    all_summaries = [
        tf.Summary.Value(tag='update_ratios/' +
                         tensor.name, simple_value=ratio)
        for tensor, ratio in zip(weight_tensors, ratios)]
    return tf.Summary(value=all_summaries)


class UpdateRatioSessionHook(tf.train.SessionRunHook):
    """A hook that computes ||grad|| / ||weights|| (using frobenius norm)."""
    def __init__(self, output_dir, every_n_steps=1000):
        self.output_dir = output_dir
        self.every_n_steps = every_n_steps
        self.before_weights = None
        self.file_writer = None
        self.weight_tensors = None
        self.global_step = None

    def begin(self):
        # These calls only works because the SessionRunHook api guarantees this
        # will get called within a graph context containing our model graph.

        self.file_writer = tf.summary.FileWriterCache.get(self.output_dir)
        self.weight_tensors = tf.trainable_variables()
        self.global_step = tf.train.get_or_create_global_step()

    def before_run(self, run_context):
        global_step = run_context.session.run(self.global_step)
        if global_step % self.every_n_steps == 0:
            self.before_weights = run_context.session.run(self.weight_tensors)

    def after_run(self, run_context, run_values):
        global_step = run_context.session.run(self.global_step)
        if self.before_weights is not None:
            after_weights = run_context.session.run(self.weight_tensors)
            weight_update_summaries = compute_update_ratio(
                self.weight_tensors, self.before_weights, after_weights)
            self.file_writer.add_summary(
                weight_update_summaries, global_step)
            self.before_weights = None


def train(*tf_records: "Records to train on"):
    """Train on examples."""
    tf.logging.set_verbosity(tf.logging.INFO)
    estimator = dual_net.get_estimator()

    effective_batch_size = FLAGS.train_batch_size
    if FLAGS.use_tpu:
        effective_batch_size *= FLAGS.num_tpu_cores

    if FLAGS.use_tpu:
        def _input_fn(params):
            return preprocessing.get_tpu_input_tensors(
                params['batch_size'],
                tf_records,
                random_rotation=True)
        # Hooks are broken with TPUestimator at the moment.
        hooks = []
    else:
        def _input_fn():
            return preprocessing.get_input_tensors(
                FLAGS.train_batch_size,
                tf_records,
                filter_amount=1.0,
                shuffle_buffer_size=FLAGS.shuffle_buffer_size,
                random_rotation=True)

        hooks = [UpdateRatioSessionHook(FLAGS.model_dir),
                 EchoStepCounterHook(output_dir=FLAGS.model_dir)]

    steps = FLAGS.steps_to_train
    logging.info("Training, steps = %s, batch = %s -> %s examples",
                 steps or '?', effective_batch_size,
                 (steps * effective_batch_size) if steps else '?')
    estimator.train(_input_fn, steps=steps, hooks=hooks)


def main(argv):
    """Train on examples and export the updated model weights."""
    tf_records = argv[1:]
    logging.info("Training on %s records: %s to %s",
                 len(tf_records), tf_records[0], tf_records[-1])
    with utils.logged_timer("Training"):
        train(*tf_records)
    if FLAGS.export_path:
        dual_net.export_model(FLAGS.export_path)


if __name__ == "__main__":
    app.run(main)
