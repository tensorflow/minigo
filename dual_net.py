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

import collections
import functools
import math
import numpy as np
import os.path
import itertools
import sys
import tensorflow as tf
from tqdm import tqdm
from typing import Dict

import features
import preprocessing
import symmetries
import go

# How many positions to look at per generation.
# Per AGZ, 2048 minibatch * 1k = 2M positions/generation
EXAMPLES_PER_GENERATION = 2000

# How many positions can fit on a graphics card. 256 for 9s, 16 or 32 for 19s.
TRAIN_BATCH_SIZE = 16


class DualNetworkTrainer():
    def __init__(self, save_file=None, logdir=None, **hparams):
        self.hparams = get_default_hyperparams(**hparams)
        self.save_file = save_file
        self.logdir = logdir
        self.sess = tf.Session(graph=tf.Graph())

    def initialize_weights(self, init_from=None):
        """Initialize weights from model checkpoint.

        If model checkpoint does not exist, fall back to init_from.
        If that doesn't exist either, bootstrap with random weights.

        This priority order prevents the mistake where the latest saved
        model exists, but you accidentally init from an older model checkpoint
        and then overwrite the newer model weights.
        """
        tf.logging.set_verbosity(tf.logging.WARN)  # Hide startup spam
        if self.save_file is not None and os.path.exists(self.save_file + '.meta'):
            tf.train.Saver().restore(self.sess, self.save_file)
            return
        if init_from is not None:
            print("Initializing from {}".format(init_from), file=sys.stderr)
            tf.train.Saver().restore(self.sess, init_from)
        else:
            print("Bootstrapping with random weights", file=sys.stderr)
            self.sess.run(tf.global_variables_initializer())
        tf.logging.set_verbosity(tf.logging.INFO)

    def save_weights(self):
        with self.sess.graph.as_default():
            tf.train.Saver().save(self.sess, self.save_file)

    def bootstrap(self):
        'Create a save file with random initial weights.'
        sess = tf.Session(graph=tf.Graph())
        with sess.graph.as_default():
            input_tensors = get_inference_input()
            output_tensors = model_fn(input_tensors, TRAIN_BATCH_SIZE,
                                      train_mode=True, **self.hparams)
            train_tensors = train_ops(
                input_tensors, output_tensors, **self.hparams)
            sess.run(tf.global_variables_initializer())
            tf.train.Saver().save(sess, self.save_file)

    def train(self, tf_records, init_from=None, num_steps=None,
              logging_freq=100, verbosity=1):
        logdir = os.path.join(
            self.logdir, 'train') if self.logdir is not None else None

        def should_log(i):
            return logdir is not None and i % logging_freq == 0
        if num_steps is None:
            num_steps = EXAMPLES_PER_GENERATION // TRAIN_BATCH_SIZE
        with self.sess.graph.as_default():
            input_tensors = preprocessing.get_input_tensors(
                TRAIN_BATCH_SIZE, tf_records)
            output_tensors = model_fn(input_tensors, TRAIN_BATCH_SIZE,
                                      train_mode=True, **self.hparams)
            train_tensors = train_ops(
                input_tensors, output_tensors, **self.hparams)
            weight_summary_op = logging_ops()
            weight_tensors = tf.trainable_variables()
            self.initialize_weights(init_from)
            if logdir is not None:
                training_stats = StatisticsCollector()
                logger = tf.summary.FileWriter(logdir, self.sess.graph)
            for i in tqdm(range(num_steps)):
                if should_log(i):
                    before_weights = self.sess.run(weight_tensors)
                try:
                    tensor_values = self.sess.run(train_tensors)
                except tf.errors.OutOfRangeError:
                    break

                if verbosity > 1 and i % logging_freq == 0:
                    print(tensor_values)
                if logdir is not None:
                    training_stats.report(
                        {k: tensor_values[k] for k in (
                            'policy_cost', 'value_cost', 'l2_cost',
                            'combined_cost', 'policy_entropy')})
                if should_log(i):
                    after_weights = self.sess.run(weight_tensors)
                    weight_update_summaries = compute_update_ratio(
                        weight_tensors, before_weights, after_weights)
                    accuracy_summaries = training_stats.collect()
                    weight_summaries = self.sess.run(weight_summary_op)
                    global_step = tensor_values['global_step']
                    logger.add_summary(weight_update_summaries, global_step)
                    logger.add_summary(accuracy_summaries, global_step)
                    logger.add_summary(weight_summaries, global_step)
            self.save_weights()

    def validate(self, tf_records, batch_size=128, init_from=None, num_steps=1000):
        """Compute only the error terms for a set of tf_records, ideally a
        holdout set, and report them to an 'test' subdirectory of the logs.
        """
        cost_tensor_names = ['policy_cost', 'value_cost', 'l2_cost',
                             'combined_cost']
        if self.logdir is None:
            print("Error, trainer not initialized with a logdir.", file=sys.stderr)
            return

        logdir = os.path.join(self.logdir, 'test')

        with self.sess.graph.as_default():
            input_tensors = preprocessing.get_input_tensors(
                batch_size, tf_records, shuffle_buffer_size=1000, filter_amount=0.05)
            output_tensors = model_fn(input_tensors, TRAIN_BATCH_SIZE,
                                      train_mode=False, **self.hparams)
            train_tensors = train_ops(
                input_tensors, output_tensors, **self.hparams)

            # just run our cost tensors
            validate_tensors = {k: train_tensors[k]
                                for k in cost_tensor_names}
            self.initialize_weights(init_from)
            training_stats = StatisticsCollector()
            logger = tf.summary.FileWriter(logdir, None)  # No graph needed.

            for i in tqdm(range(num_steps)):
                try:
                    tensor_values = self.sess.run(validate_tensors)
                except tf.errors.OutOfRangeError:
                    break
                training_stats.report(tensor_values)

            accuracy_summaries = training_stats.collect()
            global_step = self.sess.run(train_tensors['global_step'])
            logger.add_summary(accuracy_summaries, global_step)
            logger.flush()
            print(accuracy_summaries)


class DualNetwork():
    def __init__(self, save_file, **hparams):
        self.save_file = save_file
        self.hparams = get_default_hyperparams(**hparams)
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
                tf.estimator.ModeKeys.PREDICT, self.hparams, None)
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
        processed = list(map(features.extract_features, positions))
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
    return (tf.placeholder(tf.float32,
                [None, go.N, go.N, features.NEW_FEATURES_PLANES],
                name='pos_tensor'),
            {'pi_tensor': tf.placeholder(tf.float32, [None, go.N * go.N + 1]),
             'value_tensor': tf.placeholder(tf.float32, [None])})


def _round_power_of_two(n):
    """Finds the nearest power of 2 to a number.

    Thus 84 -> 64, 120 -> 128, etc.
    """
    return 2 ** int(round(math.log(n, 2)))


def get_default_hyperparams(**overrides):
    """Returns the hyperparams for the neural net.

    In other words, returns a dict whose parameters come from the AGZ
    paper:
      k: number of filters (AlphaGoZero used 256). We use 128 by
        default for a 19x19 go board.
      fc_width: Dimensionality of the fully connected linear layer
      num_shared_layers: number of shared residual blocks.  AGZ used both 19
        and 39. Here we use 19 because it's faster to train.
      l2_strength: The L2 regularization parameter.
      momentum: The momentum parameter for training
    """
    k = _round_power_of_two(go.N ** 2 / 3)  # width of each layer
    hparams = {
        'k': k,  # Width of each conv layer
        'fc_width': 2 * k,  # Width of each fully connected layer
        'num_shared_layers': go.N,  # Number of shared trunk layers
        'l2_strength': 1e-4,  # Regularization strength
        'momentum': 0.9,  # Momentum used in SGD
    }
    hparams.update(**overrides)
    return hparams


def model_fn(features, labels, mode, params, config):
    '''
    Args:
        features: tensor with shape
            [BATCH_SIZE, go.N, go.N, features.NEW_FEATURES_PLANES]
        labels: dict from string to tensor with shape
            'pi_tensor': [BATCH_SIZE, go.N * go.N + 1]
            'value_tensor': [BATCH_SIZE]
        mode: a tf.estimator.ModeKeys
        params: a dict of hyperparams
        config: ignored
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
    my_batchn = functools.partial(
        tf.layers.batch_normalization,
        momentum=.997, epsilon=1e-5, fused=True, center=True, scale=True,
        training=(mode == tf.estimator.ModeKeys.TRAIN))

    my_conv2d = functools.partial(
        tf.layers.conv2d,
        filters=params['k'], kernel_size=[3, 3], padding="same")

    def my_res_layer(inputs):
        int_layer1 = my_batchn(my_conv2d(inputs))
        initial_output = tf.nn.relu(int_layer1)
        int_layer2 = my_batchn(my_conv2d(initial_output))
        output = tf.nn.relu(inputs + int_layer2)
        return output

    initial_output = tf.nn.relu(my_batchn(my_conv2d(features)))

    # the shared stack
    shared_output = initial_output
    for i in range(params['num_shared_layers']):
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
        params['fc_width']))
    value_output = tf.nn.tanh(
        tf.reshape(tf.layers.dense(value_fc_hidden, 1), [-1]),
        name='value_output')

    # train ops
    global_step = tf.train.get_or_create_global_step()
    policy_cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels['pi_tensor']))
    value_cost = tf.reduce_mean(
        tf.square(value_output - labels['value_tensor']))
    l2_cost = params['l2_strength'] * tf.add_n([tf.nn.l2_loss(v)
        for v in tf.trainable_variables() if not 'bias' in v.name])
    combined_cost = policy_cost + value_cost + l2_cost
    policy_entropy = -tf.reduce_mean(tf.reduce_sum(
        policy_output * tf.log(policy_output), axis=1))
    boundaries = [int(1e6), int(2e6)]
    values = [1e-2, 1e-3, 1e-4]
    learning_rate = tf.train.piecewise_constant(
        global_step, boundaries, values)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.MomentumOptimizer(
            learning_rate, params['momentum']).minimize(
                combined_cost, global_step=global_step)


    metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels['pi_tensor'],
                                        predictions=policy_output,
                                        name='accuracy_op'),
        'policy_cost': tf.metrics.mean(policy_cost),
        'value_cost': tf.metrics.mean(value_cost),
        'l2_cost': tf.metrics.mean(l2_cost),
        'policy_entropy': tf.metrics.mean(policy_entropy),
        'combined_cost': tf.metrics.mean(combined_cost),
    }
    # Register metrics as tf.summary to make them show up during training
    for metric_name, metric_op in metric_ops.items():
        tf.summary.scalar(metric_name, metric_op[1])

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

def bootstrap(training_dir, **hparams):
    """Initialize a tf.Estimator run with random initial weights.

    Args:
        training_dir: The directory where tf.estimator will drop logs,
            checkpoints, and so on
        hparams: hyperparams of the model.
    """
    hparams = get_default_hyperparams(**hparams)
    # a bit hacky - forge an initial checkpoint with the name that subsequent
    # Estimator runs will expect to find.
    #
    # Estimator will do this automatically when you call train(), but calling
    # train() requires data, and I didn't feel like creating training data in
    # order to run the full train pipeline for 1 step.
    estimator_initial_checkpoint_name = 'model.ckpt-1'
    save_file = os.path.join(training_dir, estimator_initial_checkpoint_name)
    sess = tf.Session(graph=tf.Graph())
    with sess.graph.as_default():
        features, labels = get_inference_input()
        model_fn(features, labels, tf.estimator.ModeKeys.PREDICT, hparams, None)
        sess.run(tf.global_variables_initializer())
        tf.train.Saver().save(sess, save_file)


def export_model(training_dir, model_path):
    """Take the latest checkpoint and export it to model_path for selfplay.

    Assumes that all relevant model files are prefixed by the same name.
    (For example, foo.index, foo.meta and foo.data-00000-of-00001).

    Args:
        training_dir: The directory where tf.estimator keeps its checkpoints
        model_path: The path (can be a gs:// path) to export model to
    """
    estimator = tf.estimator.Estimator(model_fn, model_dir=training_dir,
        params='ignored')
    latest_checkpoint = estimator.latest_checkpoint()
    all_checkpoint_files = tf.gfile.Glob(latest_checkpoint + '*')
    for filename in all_checkpoint_files:
        suffix = filename.partition(latest_checkpoint)[2]
        destination_path = model_path + suffix
        print("Copying {} to {}".format(filename, destination_path))
        tf.gfile.Copy(filename, destination_path)


def train(training_dir, tf_records, generation_num, **hparams):
    assert generation_num > 0, "Model 0 is random weights"
    hparams = get_default_hyperparams(**hparams)
    estimator = tf.estimator.Estimator(
        model_fn,
        model_dir=training_dir,
        params=hparams
    )
    current_step = estimator.get_variable_value('global_step')
    max_steps = generation_num * EXAMPLES_PER_GENERATION // TRAIN_BATCH_SIZE
    if current_step > max_steps:
        raise ValueError("Generation {} would have happened at step {}, "
            "but training is already at step {}".format(
                generation_num, max_steps, current_step))
    print(current_step, max_steps)
    input_fn = lambda: preprocessing.get_input_tensors(
        TRAIN_BATCH_SIZE, tf_records)
    estimator.train(input_fn, max_steps=max_steps)


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


class StatisticsCollector(object):
    """Collect statistics on the runs and create graphs.

    Accuracy and cost cannot be calculated with the full test dataset
    in one pass, so they must be computed in batches. Unfortunately,
    the built-in TF summary nodes cannot be told to aggregate multiple
    executions. Therefore, we aggregate the accuracy/cost ourselves at
    the python level, and then generate the summary protobufs for writing.
    """

    def __init__(self):
        self.accums = collections.defaultdict(list)

    def report(self, values):
        """Take a dict of scalar names to scalars, and aggregate by name."""
        for key, val in values.items():
            self.accums[key].append(val)

    def collect(self):
        all_summaries = []
        for summary_name, summary_values in self.accums.items():
            avg_value = sum(summary_values) / len(summary_values)
            self.accums[summary_name] = []
            all_summaries.append(tf.Summary.Value(
                tag=summary_name, simple_value=avg_value))
        return tf.Summary(value=all_summaries)

if __name__ == '__main__':
    train('training', ['1516423825-minigo-gpu-player-7v901.tfrecord.zz'], 0)