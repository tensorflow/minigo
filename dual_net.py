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
import os.path
import sys

import argh
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.contrib import summary
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer

import features as features_lib
import go
import preprocessing
import symmetries

flags.DEFINE_integer('train_batch_size', 256,
                     'Batch size to use for train/eval evaluation. For GPU '
                     'this is batch size as expected. If \"use_tpu\" is set,'
                     'final batch size will be = train_batch_size * num_tpu_cores')

flags.DEFINE_integer('conv_width', 256 if go.N == 19 else 32,
                     'The width of each conv layer in the shared trunk.')

flags.DEFINE_integer('policy_conv_width', 2,
                     'The width of the policy conv layer.')

flags.DEFINE_integer('value_conv_width', 1,
                     'The width of the value conv layer.')

flags.DEFINE_integer('fc_width', 256 if go.N == 19 else 64,
                     'The width of the fully connected layer in value head.')

flags.DEFINE_integer('trunk_layers', go.N,
                     'The number of resnet layers in the shared trunk.')

flags.DEFINE_multi_integer('lr_boundaries', [400000, 600000],
                           'The number of steps at which the learning rate will decay')

flags.DEFINE_multi_float('lr_rates', [0.01, 0.001, 0.0001],
                         'The different learning rates')

flags.DEFINE_float('l2_strength', 1e-4,
                   'The L2 regularization parameter applied to weights.')

flags.DEFINE_float('value_cost_weight', 1.0,
                   'Scalar for value_cost, AGZ paper suggests 1/100 for '
                   'supervised learning')

flags.DEFINE_float('sgd_momentum', 0.9,
                   'Momentum parameter for learning rate.')

flags.DEFINE_string('model_dir', None,
                    'The working directory of the model')

# See www.moderndescartes.com/essays/shuffle_viz for discussion on sizing
flags.DEFINE_integer('shuffle_buffer_size', 2000,
                     'Size of buffer used to shuffle train examples.')

flags.DEFINE_bool('use_tpu', False, 'Whether to use TPU for training.')

flags.DEFINE_bool('quantize', False,
                  'Whether create a quantized model. When loading a model for '
                  'inference, this must match how the model was trained.')

flags.DEFINE_integer('quant_delay', 700 * 1024,
                     'Number of training steps after which weights and '
                     'activations are quantized.')

flags.DEFINE_string(
    'tpu_name', None,
    'The Cloud TPU to use for training. This should be either the name used'
    'when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

flags.register_multi_flags_validator(
    ['lr_boundaries', 'lr_rates'],
    lambda flags: len(flags['lr_boundaries']) == len(flags['lr_rates']) - 1,
    'Number of learning rates must be exactly one greater than the number of boundaries')

flags.DEFINE_integer(
    'iterations_per_loop', 128,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

flags.DEFINE_integer(
    'num_tpu_cores', default=8,
    help=('Number of TPU cores. For a single TPU device, this is 8 because each'
          ' TPU has 4 chips each with 2 cores.'))

flags.DEFINE_integer(
    'summary_steps', default=256,
    help='Number of steps between logging summary scalars.')

flags.DEFINE_integer(
    'keep_checkpoint_max', default=5, help='Number of checkpoints to keep.')

flags.DEFINE_bool(
    'use_random_symmetry', True,
    help='If true random symmetries be used when doing inference.')

flags.register_multi_flags_validator(
    ['use_tpu', 'iterations_per_loop', 'summary_steps'],
    lambda flags: (not flags['use_tpu'] or
                   flags['summary_steps'] % flags['iterations_per_loop'] == 0),
    'If use_tpu, summary_steps must be a multiple of iterations_per_loop')

FLAGS = flags.FLAGS


# How many positions to look at per generation.
# Per AGZ, 2048 minibatch * 1k = 2M positions/generation
EXAMPLES_PER_GENERATION = 2 ** 21


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

    def run(self, position):
        probs, values = self.run_many([position])
        return probs[0], values[0]

    def run_many(self, positions):
        processed = list(map(features_lib.extract_features, positions))
        if FLAGS.use_random_symmetry:
            syms_used, processed = symmetries.randomize_symmetries_feat(
                processed)
        outputs = self.sess.run(self.inference_output,
                                feed_dict={self.inference_input: processed})
        probabilities, value = outputs['policy_output'], outputs['value_output']
        if FLAGS.use_random_symmetry:
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
    policy_cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=tf.stop_gradient(labels['pi_tensor'])))

    value_cost = FLAGS.value_cost_weight * tf.reduce_mean(
        tf.square(value_output - labels['value_tensor']))

    reg_vars = [v for v in tf.trainable_variables()
                if 'bias' not in v.name and 'beta' not in v.name]
    l2_cost = FLAGS.l2_strength * \
        tf.add_n([tf.nn.l2_loss(v) for v in reg_vars])

    combined_cost = policy_cost + value_cost + l2_cost

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.piecewise_constant(
        global_step, FLAGS.lr_boundaries, FLAGS.lr_rates)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # Insert quantization ops if requested
    if FLAGS.quantize:
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.contrib.quantize.create_training_graph(
                quant_delay=FLAGS.quant_delay)
        else:
            tf.contrib.quantize.create_eval_graph()

    optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.sgd_momentum)
    if FLAGS.use_tpu:
        optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(combined_cost, global_step=global_step)

    # Computations to be executed on CPU, outside of the main TPU queues.
    def eval_metrics_host_call_fn(policy_output, value_output, pi_tensor, policy_cost,
                                  value_cost, l2_cost, combined_cost, step,
                                  est_mode=tf.estimator.ModeKeys.TRAIN):
        policy_entropy = -tf.reduce_mean(tf.reduce_sum(
            policy_output * tf.log(policy_output), axis=1))
        # pi_tensor is one_hot when generated from sgfs (for supervised learning)
        # and soft-max when using self-play records. argmax normalizes the two.
        policy_target_top_1 = tf.argmax(pi_tensor, axis=1)

        policy_output_in_top1 = tf.to_float(
            tf.nn.in_top_k(policy_output, policy_target_top_1, k=1))
        policy_output_in_top3 = tf.to_float(
            tf.nn.in_top_k(policy_output, policy_target_top_1, k=3))

        policy_top_1_confidence = tf.reduce_max(policy_output, axis=1)
        policy_target_top_1_confidence = tf.boolean_mask(
            policy_output,
            tf.one_hot(policy_target_top_1, tf.shape(policy_output)[1]))

        with tf.variable_scope("metrics"):
            metric_ops = {
                'policy_cost': tf.metrics.mean(policy_cost),
                'value_cost': tf.metrics.mean(value_cost),
                'l2_cost': tf.metrics.mean(l2_cost),
                'policy_entropy': tf.metrics.mean(policy_entropy),
                'combined_cost': tf.metrics.mean(combined_cost),

                'policy_accuracy_top_1': tf.metrics.mean(policy_output_in_top1),
                'policy_accuracy_top_3': tf.metrics.mean(policy_output_in_top3),
                'policy_top_1_confidence': tf.metrics.mean(policy_top_1_confidence),
                'policy_target_top_1_confidence': tf.metrics.mean(
                    policy_target_top_1_confidence),
                'value_confidence': tf.metrics.mean(tf.abs(value_output)),
            }

        if est_mode == tf.estimator.ModeKeys.EVAL:
            return metric_ops

        # Create summary ops so that they show up in SUMMARIES collection
        # That way, they get logged automatically during training
        summary_writer = summary.create_file_writer(FLAGS.model_dir)
        with summary_writer.as_default(), \
                summary.always_record_summaries():
            for metric_name, metric_op in metric_ops.items():
                summary.scalar(metric_name, metric_op[1])

        # Reset metrics occasionally so that they are mean of recent batches.
        reset_op = tf.variables_initializer(tf.local_variables("metrics"))
        cond_reset_op = tf.cond(
            tf.equal(tf.mod(tf.reduce_min(step),
                            FLAGS.summary_steps), tf.to_int64(1)),
            lambda: reset_op,
            lambda: tf.no_op())

        return summary.all_summary_ops() + [cond_reset_op]

    metric_args = [
        policy_output,
        value_output,
        labels['pi_tensor'],
        tf.reshape(policy_cost, [1]),
        tf.reshape(value_cost, [1]),
        tf.reshape(l2_cost, [1]),
        tf.reshape(combined_cost, [1]),
        tf.reshape(global_step, [1]),
    ]

    predictions = {
        'policy_output': policy_output,
        'value_output': value_output,
    }

    eval_metrics_only_fn = functools.partial(
        eval_metrics_host_call_fn, est_mode=tf.estimator.ModeKeys.EVAL)
    host_call_fn = functools.partial(
        eval_metrics_host_call_fn, est_mode=tf.estimator.ModeKeys.TRAIN)

    tpu_estimator_spec = tpu_estimator.TPUEstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=combined_cost,
        train_op=train_op,
        eval_metrics=(eval_metrics_only_fn, metric_args),
        host_call=(host_call_fn, metric_args)
    )
    if FLAGS.use_tpu:
        return tpu_estimator_spec
    else:
        return tpu_estimator_spec.as_estimator_spec()


def model_inference_fn(features, training):
    """Builds just the inference part of the model graph.

    Args:
        features: input features tensor.
        training: True if the model is training.

    Returns:
        (policy_output, value_output, logits) tuple of tensors.
    """

    my_batchn = functools.partial(
        tf.layers.batch_normalization,
        axis=-1,
        momentum=.95,
        epsilon=1e-5,
        center=True,
        scale=True,
        fused=True,
        training=training)

    my_conv2d = functools.partial(
        tf.layers.conv2d,
        filters=FLAGS.conv_width,
        kernel_size=3,
        padding="same",
        data_format="channels_last",
        use_bias=False)

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
    policy_conv = my_conv2d(
        shared_output, filters=FLAGS.policy_conv_width, kernel_size=1)
    policy_conv = tf.nn.relu(my_batchn(policy_conv, center=False, scale=False))
    logits = tf.layers.dense(
        tf.reshape(policy_conv, [-1, FLAGS.policy_conv_width * go.N * go.N]),
        go.N * go.N + 1)

    policy_output = tf.nn.softmax(logits, name='policy_output')

    # value head
    value_conv = my_conv2d(
        shared_output, filters=FLAGS.value_conv_width, kernel_size=1)
    value_conv = tf.nn.relu(my_batchn(value_conv, center=False, scale=False))

    value_fc_hidden = tf.nn.relu(tf.layers.dense(
        tf.reshape(value_conv, [-1, FLAGS.value_conv_width * go.N * go.N]),
        FLAGS.fc_width))
    value_output = tf.nn.tanh(
        tf.reshape(tf.layers.dense(value_fc_hidden, 1), [-1]),
        name='value_output')

    return policy_output, value_output, logits


def get_estimator(working_dir):
    if FLAGS.use_tpu:
        return get_tpu_estimator(working_dir)

    run_config = tf.estimator.RunConfig(
        save_summary_steps=FLAGS.summary_steps,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max)
    return tf.estimator.Estimator(
        model_fn,
        model_dir=working_dir,
        config=run_config)


def get_tpu_estimator(working_dir):
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=None, project=None)
    tpu_grpc_url = tpu_cluster_resolver.get_master()

    run_config = tpu_config.RunConfig(
        master=tpu_grpc_url,
        evaluation_master=tpu_grpc_url,
        model_dir=working_dir,
        save_checkpoints_steps=max(1000, FLAGS.iterations_per_loop),
        save_summary_steps=FLAGS.summary_steps,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
        session_config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True),
        tpu_config=tpu_config.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=tpu_config.InputPipelineConfig.PER_HOST_V2))

    return tpu_estimator.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size * FLAGS.num_tpu_cores,
        eval_batch_size=FLAGS.train_batch_size * FLAGS.num_tpu_cores)


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


def train(
        *tf_records: "Records to train on",
        steps: "Number of steps to train. If not set iterates over "
               "tf_records and sets steps to examples / batch_size"=-1):
    tf.logging.set_verbosity(tf.logging.INFO)
    estimator = get_estimator(FLAGS.model_dir)

    effective_batch_size = FLAGS.train_batch_size
    if FLAGS.use_tpu:
        effective_batch_size *= FLAGS.num_tpu_cores

    if steps == -1:
        def count_examples(tf_record):
            opts = preprocessing.TF_RECORD_CONFIG
            return sum(1 for _ in tqdm(
                tf.python_io.tf_record_iterator(tf_record, opts),
                desc=tf_record))

        total_examples = sum(map(count_examples, tf_records))
        steps = total_examples // effective_batch_size

    if FLAGS.use_tpu:
        def input_fn(params):
            return preprocessing.get_tpu_input_tensors(
                params['batch_size'],
                tf_records,
                random_rotation=True)
        # TODO: get hooks working again with TPUestimator.
        hooks = []
    else:
        def input_fn():
            return preprocessing.get_input_tensors(
                FLAGS.train_batch_size,
                tf_records,
                filter_amount=1.0,
                shuffle_buffer_size=FLAGS.shuffle_buffer_size,
                random_rotation=True)

        hooks = [UpdateRatioSessionHook(FLAGS.model_dir),
                 EchoStepCounterHook(output_dir=FLAGS.model_dir)]

    print("Training, steps = {} x{} = {} examples".format(
        steps, effective_batch_size, steps * effective_batch_size))
    estimator.train(input_fn, steps=steps, hooks=hooks)


def validate(tf_records, validate_name=None):
    validate_name = validate_name or "selfplay"

    if FLAGS.use_tpu:
        def input_fn(params):
            return preprocessing.get_tpu_input_tensors(
                params['batch_size'],
                tf_records, filter_amount=0.05)
    else:
        def input_fn():
            return preprocessing.get_input_tensors(
                FLAGS.train_batch_size, tf_records, filter_amount=0.05,
                shuffle_buffer_size=20000)

    estimator = get_estimator(FLAGS.model_dir)
    estimator.evaluate(input_fn, steps=50, name=validate_name)


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
argh.add_commands(parser, [train, export_model, validate])

if __name__ == '__main__':
    # Let absl.flags parse known flags from argv, then pass the remaining flags
    # into argh for dispatching.
    remaining_argv = flags.FLAGS(sys.argv, known_only=True)
    argh.dispatch(parser, argv=remaining_argv[1:])
