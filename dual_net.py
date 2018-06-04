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

"""
The policy and value networks share a majority of their architecture.
This helps the intermediate layers extract concepts that are relevant to both
move prediction and score estimation.
"""

from absl import flags
import argparse
import functools
import math
import os.path
import sys

import argh
import numpy as np
import tensorflow as tf
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.contrib.tpu.python.tpu import bfloat16
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer

import features as features_lib
import go
import preprocessing
import symmetries

flags.DEFINE_integer('train_batch_size', 256,
                     'Batch size to use for train/eval evaluation. For GPU '
                     'this is batch size as expected. For TPU, this is the '
                     'batch size used for each TPU core (so that effective '
                     'batch size = train_batch_size * num_tpu_cores).')

flags.DEFINE_integer('conv_width', 128 if go.N == 19 else 32,
                     'The width of each conv layer in the shared trunk.')

flags.DEFINE_integer('fc_width', 256 if go.N == 19 else 64,
                     'The width of the fully connected layer in value head.')

flags.DEFINE_integer('trunk_layers', go.N,
                     'The number of resnet layers in the shared trunk.')

flags.DEFINE_float('l2_strength', 1e-4,
                   'The L2 regularization parameter applied to weights.')

flags.DEFINE_float('sgd_momentum', 0.9,
                   'Momentum parameter for learning rate.')

# See www.moderndescartes.com/essays/shuffle_viz for discussion on sizing
flags.DEFINE_integer('shuffle_buffer_size', 20000,
                     'Size of buffer used to shuffle train examples.')

flags.DEFINE_bool('use_tpu', False, 'Whether to use TPU for training.')

flags.DEFINE_string(
    'tpu_name', None,
    'The Cloud TPU to use for training. This should be either the name used'
    'when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

flags.register_multi_flags_validator(
    ['use_tpu', 'tpu_name'],
    lambda flags: bool(flags['use_tpu']) == bool(flags['tpu_name']),
    'If use_tpu is set, tpu_name must also be set.')

flags.DEFINE_integer(
    'iterations_per_loop', 100,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

flags.DEFINE_integer(
    'num_tpu_cores', default=8,
    help=('Number of TPU cores. For a single TPU device, this is 8 because each'
          ' TPU has 4 chips each with 2 cores.'))

flags.DEFINE_enum(
    'precision', 'float32', ['bfloat16', 'float32'],
    help=('Precision to use for training ; one of: {bfloat16, float32}. '
          'Does not affect the saved model output - bfloat16 training outputs '
          'float32 weights (least significant 16 bits all end up zero)'))

FLAGS = flags.FLAGS


# TODO: Clean up dual_net.EXAMPLES_PER_GENERATION, main.EXAMPLES_PER_RECORD/main.WINDOW_SIZE
# How many positions to look at per generation.
# Per AGZ, 2048 minibatch * 1k = 2M positions/generation
EXAMPLES_PER_GENERATION = 2000000


class DualNetwork():
    def __init__(self, save_file):
        self.save_file = save_file
        self.inference_input = None
        self.inference_output = None
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=tf.Graph(), config=config)
        self.initialize_graph()

    def initialize_graph(self):
        with self.sess.graph.as_default():
            features, labels = get_inference_input()
            estimator_spec = model_fn(features, labels,
                                      tf.estimator.ModeKeys.PREDICT)
            self.inference_input = features
            self.inference_output = estimator_spec.predictions
            if self.save_file is not None:
                self.initialize_weights(self.save_file)
            else:
                self.sess.run(tf.global_variables_initializer())

    def initialize_weights(self, save_file):
        """Initialize the weights from the given save_file.
        Assumes that the graph has been constructed, and the
        save_file contains weights that match the graph. Used
        to set the weights to a different version of the player
        without redifining the entire graph."""
        tf.train.Saver().restore(self.sess, save_file)

    def run(self, position, use_random_symmetry=True):
        probs, values = self.run_many([position],
                                      use_random_symmetry=use_random_symmetry)
        return probs[0], values[0]

    def run_many(self, positions, use_random_symmetry=True):
        processed = list(map(features_lib.extract_features, positions))
        if use_random_symmetry:
            syms_used, processed = symmetries.randomize_symmetries_feat(
                processed)
        outputs = self.sess.run(self.inference_output,
                                feed_dict={self.inference_input: processed})
        probabilities, value = outputs['policy_output'], outputs['value_output']
        if use_random_symmetry:
            probabilities = symmetries.invert_symmetries_pi(
                syms_used, probabilities)
        return probabilities, value


def get_inference_input():
    """Set up placeholders for input features/labels.

    Returns the feature, output tensors that get passed into model_fn."""
    return (tf.placeholder(tf.float32,
                           [None, go.N, go.N, features_lib.NEW_FEATURES_PLANES],
                           name='pos_tensor'),
            {'pi_tensor': tf.placeholder(tf.float32, [None, go.N * go.N + 1]),
             'value_tensor': tf.placeholder(tf.float32, [None])})


def model_inference_fn(features, training):
    my_batchn = functools.partial(
        tf.layers.batch_normalization,
        momentum=.997, epsilon=1e-5, fused=True, center=True, scale=True,
        training=training)

    my_conv2d = functools.partial(
        tf.layers.conv2d,
        filters=FLAGS.conv_width, kernel_size=[3, 3], padding="same")

    def my_res_layer(inputs):
        int_layer1 = my_batchn(my_conv2d(inputs))
        initial_output = tf.nn.relu(int_layer1)
        int_layer2 = my_batchn(my_conv2d(initial_output))
        output = tf.nn.relu(inputs + int_layer2)
        return output

    initial_output = tf.nn.relu(my_batchn(my_conv2d(features)))

    # the shared stack
    shared_output = initial_output
    for _ in range(FLAGS.trunk_layers):
        shared_output = my_res_layer(shared_output)

    # policy head
    policy_conv = tf.nn.relu(my_batchn(
        my_conv2d(shared_output, filters=2, kernel_size=[1, 1]),
        center=False, scale=False))
    logits = tf.layers.dense(
        tf.reshape(policy_conv, [-1, go.N * go.N * 2]),
        go.N * go.N + 1)

    policy_output = tf.nn.softmax(logits, name='policy_output')

    # value head
    value_conv = tf.nn.relu(my_batchn(
        my_conv2d(shared_output, filters=1, kernel_size=[1, 1]),
        center=False, scale=False))
    value_fc_hidden = tf.nn.relu(tf.layers.dense(
        tf.reshape(value_conv, [-1, go.N * go.N]),
        FLAGS.fc_width))
    value_output = tf.nn.tanh(
        tf.reshape(tf.layers.dense(value_fc_hidden, 1), [-1]),
        name='value_output')

    return policy_output, value_output, logits


def model_fn(features, labels, mode, params=None):
    '''
    Args:
        features: tensor with shape
            [BATCH_SIZE, go.N, go.N, features_lib.NEW_FEATURES_PLANES]
        labels: dict from string to tensor with shape
            'pi_tensor': [BATCH_SIZE, go.N * go.N + 1]
            'value_tensor': [BATCH_SIZE]
        mode: a tf.estimator.ModeKeys (batchnorm params update for TRAIN only)
        params: (Ignored; needed for compat with TPUEstimator)
    Returns: tf.estimator.EstimatorSpec with props
        mode: same as mode arg
        predictions: dict of tensors
            'policy': [BATCH_SIZE, go.N * go.N + 1]
            'value': [BATCH_SIZE]
        loss: a single value tensor
        train_op: train op
        eval_metric_ops
    return dict of tensors
        logits: [BATCH_SIZE, go.N * go.N + 1]
    '''

    policy_output, value_output, logits = model_inference_fn(
        features, mode == tf.estimator.ModeKeys.TRAIN)

    # train ops
    global_step = tf.train.get_or_create_global_step()
    policy_cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=tf.stop_gradient(labels['pi_tensor'])))
    value_cost = tf.reduce_mean(
        tf.square(value_output - labels['value_tensor']))
    l2_cost = FLAGS.l2_strength * tf.add_n([
        tf.nn.l2_loss(v)
        for v in tf.trainable_variables() if not 'bias' in v.name])
    combined_cost = policy_cost + value_cost + l2_cost
    boundaries = [40 * int(1e6), 80 * int(1e6)]
    values = [1e-3, 1e-4, 1e-5]
    learning_rate = tf.train.piecewise_constant(
        global_step, boundaries, values)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, FLAGS.sgd_momentum)
    if FLAGS.use_tpu:
        optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(combined_cost, global_step=global_step)

    # Computations to be executed on CPU, outside of the main TPU queues.
    def host_call_fn(policy_output, value_output, pi_tensor, policy_cost,
                     value_cost, l2_cost, combined_cost):
        policy_entropy = -tf.reduce_mean(tf.reduce_sum(
            policy_output * tf.log(policy_output), axis=1))
        # pi_tensor is one_hot when generated from sgfs (for supervised learning)
        # and soft-max when using self-play records. argmax normalizes the two.
        policy_target_top_1 = tf.argmax(pi_tensor, axis=1)
        policy_output_top_1 = tf.argmax(policy_output, axis=1)

        policy_output_in_top3 = tf.to_float(
            tf.nn.in_top_k(policy_output, policy_target_top_1, k=3))

        policy_top_1_confidence = tf.reduce_max(policy_output, axis=1)
        policy_target_top_1_confidence = tf.boolean_mask(
            policy_output,
            tf.one_hot(policy_target_top_1, tf.shape(policy_output)[1]))

        metric_ops = {
            'policy_cost': tf.metrics.mean(policy_cost),
            'value_cost': tf.metrics.mean(value_cost),
            'l2_cost': tf.metrics.mean(l2_cost),
            'policy_entropy': tf.metrics.mean(policy_entropy),
            'combined_cost': tf.metrics.mean(combined_cost),

            'policy_accuracy_top_1': tf.metrics.accuracy(
                labels=policy_target_top_1, predictions=policy_output_top_1),
            'policy_accuracy_top_3': tf.metrics.mean(policy_output_in_top3),
            'policy_top_1_confidence': tf.metrics.mean(policy_top_1_confidence),
            'policy_target_top_1_confidence': tf.metrics.mean(
                policy_target_top_1_confidence),
            'value_confidence': tf.metrics.mean(tf.abs(value_output)),
        }
        # Create summary ops so that they show up in SUMMARIES collection
        # That way, they get logged automatically during training
        for metric_name, metric_op in metric_ops.items():
            tf.summary.scalar(metric_name, metric_op[1])
        return metric_ops

    if FLAGS.use_tpu:
        return tpu_estimator.TPUEstimatorSpec(
            mode=mode,
            loss=combined_cost,
            train_op=train_op,
            host_call=(host_call_fn, [
                policy_output,
                value_output,
                labels['pi_tensor'],
                tf.reshape(policy_cost, [1]),
                tf.reshape(value_cost, [1]),
                tf.reshape(l2_cost, [1]),
                tf.reshape(combined_cost, [1])]))
    else:
        metric_ops = host_call_fn(policy_output, value_output, labels['pi_tensor'],
                                  policy_cost, value_cost, l2_cost, combined_cost)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'policy_output': policy_output,
                'value_output': value_output,
            },
            loss=combined_cost,
            train_op=train_op,
            eval_metric_ops=metric_ops,
        )


def get_estimator(working_dir):
    run_config = tf.estimator.RunConfig(save_summary_steps=500)
    return tf.estimator.Estimator(
        model_fn,
        model_dir=working_dir,
        config=run_config)


def bootstrap(working_dir):
    """Initialize a tf.Estimator run with random initial weights.

    Args:
        working_dir: The directory where tf.estimator will drop logs,
            checkpoints, and so on
    """
    # a bit hacky - forge an initial checkpoint with the name that subsequent
    # Estimator runs will expect to find.
    #
    # Estimator will do this automatically when you call train(), but calling
    # train() requires data, and I didn't feel like creating training data in
    # order to run the full train pipeline for 1 step.
    estimator_initial_checkpoint_name = 'model.ckpt-1'
    save_file = os.path.join(working_dir, estimator_initial_checkpoint_name)
    sess = tf.Session(graph=tf.Graph())
    with sess.graph.as_default():
        features, labels = get_inference_input()
        model_fn(features, labels, tf.estimator.ModeKeys.PREDICT)
        sess.run(tf.global_variables_initializer())
        tf.train.Saver().save(sess, save_file)


def export_model(working_dir, model_path):
    """Take the latest checkpoint and export it to model_path for selfplay.

    Assumes that all relevant model files are prefixed by the same name.
    (For example, foo.index, foo.meta and foo.data-00000-of-00001).

    Args:
        working_dir: The directory where tf.estimator keeps its checkpoints
        model_path: The path (can be a gs:// path) to export model to
    """
    estimator = tf.estimator.Estimator(model_fn, model_dir=working_dir)
    latest_checkpoint = estimator.latest_checkpoint()
    all_checkpoint_files = tf.gfile.Glob(latest_checkpoint + '*')
    for filename in all_checkpoint_files:
        suffix = filename.partition(latest_checkpoint)[2]
        destination_path = model_path + suffix
        print("Copying {} to {}".format(filename, destination_path))
        tf.gfile.Copy(filename, destination_path)


def train(working_dir, *tf_records, steps=None):
    if FLAGS.use_tpu:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=None, project=None)

        config = tpu_config.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=working_dir,
            save_checkpoints_steps=max(600, FLAGS.iterations_per_loop),
            tpu_config=tpu_config.TPUConfig(
                iterations_per_loop=FLAGS.iterations_per_loop,
                num_shards=FLAGS.num_tpu_cores,
                per_host_input_for_training=tpu_config.InputPipelineConfig.PER_HOST_V2))  # pylint: disable=line-too-long

        estimator = tpu_estimator.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=config,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.train_batch_size)

        def input_fn(params):
            return preprocessing.get_tpu_input_tensors(params['batch_size'], tf_records)
        # TODO: get hooks working again with TPUestimator.
        hooks = []
    else:
        estimator = get_estimator(working_dir)

        def input_fn():
            return preprocessing.get_input_tensors(
                FLAGS.train_batch_size, tf_records, filter_amount=1.0,
                shuffle_buffer_size=FLAGS.shuffle_buffer_size)

        hooks = [UpdateRatioSessionHook(working_dir),
                 EchoStepCounterHook(output_dir=working_dir)]

    if steps is None:
        steps = EXAMPLES_PER_GENERATION // FLAGS.train_batch_size
    print("Training, steps = {}".format(steps))
    estimator.train(input_fn, steps=int(steps), hooks=hooks)


def validate(working_dir, tf_records, checkpoint_name=None, validate_name=None):
    estimator = get_estimator(working_dir)
    validate_name = validate_name or "selfplay"
    checkpoint_name = checkpoint_name or estimator.latest_checkpoint()

    def input_fn():
        return preprocessing.get_input_tensors(
            FLAGS.train_batch_size, tf_records, filter_amount=0.05,
            shuffle_buffer_size=20000)

    estimator.evaluate(input_fn, steps=500, name=validate_name)


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


class EchoStepCounterHook(tf.train.StepCounterHook):
    def _log_and_record(self, elapsed_steps, elapsed_time, global_step):
        s_per_sec = elapsed_steps / elapsed_time
        print("{}: {:.3f} steps per second".format(global_step, s_per_sec))
        super()._log_and_record(elapsed_steps, elapsed_time, global_step)


class UpdateRatioSessionHook(tf.train.SessionRunHook):
    def __init__(self, working_dir, every_n_steps=1000):
        self.working_dir = working_dir
        self.every_n_steps = every_n_steps
        self.before_weights = None

    def begin(self):
        # These calls only works because the SessionRunHook api guarantees this
        # will get called within a graph context containing our model graph.

        self.summary_writer = SummaryWriterCache.get(self.working_dir)
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
            self.summary_writer.add_summary(
                weight_update_summaries, global_step)
            self.before_weights = None


parser = argparse.ArgumentParser()
argh.add_commands(parser, [train])

if __name__ == '__main__':
    # Let absl.flags parse known flags from argv, then pass the remaining flags
    # into argh for dispatching.
    remaining_argv = flags.FLAGS(sys.argv, known_only=True)
    argh.dispatch(parser, argv=remaining_argv[1:])
