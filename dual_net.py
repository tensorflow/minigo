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
import functools
import logging
import os.path
import time
import numpy as np
import random

import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.contrib import summary
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer

import features as features_lib
import go
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

flags.DEFINE_integer('training_seed', 0,
                     'Random seed to use for training and validation')

flags.register_multi_flags_validator(
    ['lr_boundaries', 'lr_rates'],
    lambda flags: len(flags['lr_boundaries']) == len(flags['lr_rates']) - 1,
    'Number of learning rates must be exactly one greater than the number of boundaries')

flags.DEFINE_float('l2_strength', 1e-4,
                   'The L2 regularization parameter applied to weights.')

flags.DEFINE_float('value_cost_weight', 1.0,
                   'Scalar for value_cost, AGZ paper suggests 1/100 for '
                   'supervised learning')

flags.DEFINE_float('sgd_momentum', 0.9,
                   'Momentum parameter for learning rate.')

flags.DEFINE_string('work_dir', None,
                    'The Estimator working directory. Used to dump: '
                    'checkpoints, tensorboard logs, etc..')

flags.DEFINE_bool('use_tpu', False, 'Whether to use TPU for training.')

flags.DEFINE_string(
    'tpu_name', None,
    'The Cloud TPU to use for training. This should be either the name used'
    'when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

flags.DEFINE_integer(
    'num_tpu_cores', default=8,
    help=('Number of TPU cores. For a single TPU device, this is 8 because each'
          ' TPU has 4 chips each with 2 cores.'))

flags.DEFINE_bool('quantize', False,
                  'Whether create a quantized model. When loading a model for '
                  'inference, this must match how the model was trained.')

flags.DEFINE_integer('quant_delay', 700 * 1024,
                     'Number of training steps after which weights and '
                     'activations are quantized.')

flags.DEFINE_integer(
    'iterations_per_loop', 128,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

flags.DEFINE_integer(
    'summary_steps', default=256,
    help='Number of steps between logging summary scalars.')

flags.DEFINE_integer(
    'keep_checkpoint_max', default=5, help='Number of checkpoints to keep.')

flags.DEFINE_bool(
    'use_random_symmetry', True,
    help='If true random symmetries be used when doing inference.')

flags.DEFINE_bool(
    'use_SE', False,
    help='Use Squeeze and Excitation.')

flags.DEFINE_bool(
    'use_SE_bias', False,
    help='Use Squeeze and Excitation with bias.')

flags.DEFINE_integer(
    'SE_ratio', 2,
    help='Squeeze and Excitation ratio.')

flags.DEFINE_bool(
    'use_swish', False,
    help=('Use Swish activation function inplace of ReLu. '
          'https://arxiv.org/pdf/1710.05941.pdf'))


# TODO(seth): Verify if this is still required.
flags.register_multi_flags_validator(
    ['use_tpu', 'iterations_per_loop', 'summary_steps'],
    lambda flags: (not flags['use_tpu'] or
                   flags['summary_steps'] % flags['iterations_per_loop'] == 0),
    'If use_tpu, summary_steps must be a multiple of iterations_per_loop')

FLAGS = flags.FLAGS


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
            params = FLAGS.flag_values_dict()
            logging.info('TPU inference is supported on C++ only. '
                         'DualNetwork will ignore use_tpu=True')
            params['use_tpu'] = False
            estimator_spec = model_fn(features, labels,
                                      tf.estimator.ModeKeys.PREDICT,
                                      params=params)
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


def model_fn(features, labels, mode, params):
    """
    Create the model for estimator api

    Args:
        features: tensor with shape
            [BATCH_SIZE, go.N, go.N, features_lib.NEW_FEATURES_PLANES]
        labels: dict from string to tensor with shape
            'pi_tensor': [BATCH_SIZE, go.N * go.N + 1]
            'value_tensor': [BATCH_SIZE]
        mode: a tf.estimator.ModeKeys (batchnorm params update for TRAIN only)
        params: A dictionary (Typically derived from the FLAGS object.)
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
    """

    policy_output, value_output, logits = model_inference_fn(
        features, mode == tf.estimator.ModeKeys.TRAIN, params)

    # train ops
    policy_cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=tf.stop_gradient(labels['pi_tensor'])))

    value_cost = params['value_cost_weight'] * tf.reduce_mean(
        tf.square(value_output - labels['value_tensor']))

    reg_vars = [v for v in tf.trainable_variables()
                if 'bias' not in v.name and 'beta' not in v.name]
    l2_cost = params['l2_strength'] * \
        tf.add_n([tf.nn.l2_loss(v) for v in reg_vars])

    combined_cost = policy_cost + value_cost + l2_cost

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.piecewise_constant(
        global_step, params['lr_boundaries'], params['lr_rates'])
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # Insert quantization ops if requested
    if params['quantize']:
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.contrib.quantize.create_training_graph(
                quant_delay=params['quant_delay'])
        else:
            tf.contrib.quantize.create_eval_graph()

    optimizer = tf.train.MomentumOptimizer(
        learning_rate, params['sgd_momentum'])
    if params['use_tpu']:
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

        value_cost_normalized = value_cost / params['value_cost_weight']

        with tf.variable_scope('metrics'):
            metric_ops = {
                'policy_cost': tf.metrics.mean(policy_cost),
                'value_cost': tf.metrics.mean(value_cost),
                'value_cost_normalized': tf.metrics.mean(value_cost_normalized),
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

        # NOTE: global_step is rounded to a multiple of FLAGS.summary_steps.
        eval_step = tf.reduce_min(step)

        # Create summary ops so that they show up in SUMMARIES collection
        # That way, they get logged automatically during training
        summary_writer = summary.create_file_writer(FLAGS.work_dir)
        with summary_writer.as_default(), \
                summary.record_summaries_every_n_global_steps(
                    params['summary_steps'], eval_step):
            for metric_name, metric_op in metric_ops.items():
                summary.scalar(metric_name, metric_op[1], step=eval_step)

        # Reset metrics occasionally so that they are mean of recent batches.
        reset_op = tf.variables_initializer(tf.local_variables('metrics'))
        cond_reset_op = tf.cond(
            tf.equal(eval_step % params['summary_steps'], tf.to_int64(1)),
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
    if params['use_tpu']:
        return tpu_estimator_spec
    else:
        return tpu_estimator_spec.as_estimator_spec()


def model_inference_fn(features, training, params):
    """Builds just the inference part of the model graph.

    Args:
        features: input features tensor.
        training: True if the model is training.
        params: A dictionary

    Returns:
        (policy_output, value_output, logits) tuple of tensors.
    """

    mg_batchn = functools.partial(
        tf.layers.batch_normalization,
        axis=-1,
        momentum=.95,
        epsilon=1e-5,
        center=True,
        scale=True,
        fused=True,
        training=training)

    mg_conv2d = functools.partial(
        tf.layers.conv2d,
        filters=params['conv_width'],
        kernel_size=3,
        padding='same',
        data_format='channels_last',
        use_bias=False)

    mg_global_avgpool2d = functools.partial(
        tf.layers.average_pooling2d,
        pool_size=go.N,
        strides=1,
        padding='valid',
        data_format='channels_last')

    def mg_activation(inputs):
        if FLAGS.use_swish:
            return tf.nn.swish(inputs)

        return tf.nn.relu(inputs)

    def residual_inner(inputs):
        conv_layer1 = mg_batchn(mg_conv2d(inputs))
        initial_output = mg_activation(conv_layer1)
        conv_layer2 = mg_batchn(mg_conv2d(initial_output))
        return conv_layer2

    def mg_res_layer(inputs):
        residual = residual_inner(inputs)
        output = mg_activation(inputs + residual)
        return output

    def mg_squeeze_excitation_layer(inputs):
        # Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks.
        # 2018 IEEE/CVF Conference on Computer Vision, 7132-7141.
        # arXiv:1709.01507 [cs.CV]

        channels = params['conv_width']
        ratio = FLAGS.SE_ratio
        assert channels % ratio == 0

        residual = residual_inner(inputs)
        pool = mg_global_avgpool2d(residual)
        fc1 = tf.layers.dense(pool, units=channels // ratio)
        squeeze = mg_activation(fc1)

        if FLAGS.use_SE_bias:
            fc2 = tf.layers.dense(squeeze, units=2*channels)
            # Channels_last so axis = 3 = -1
            gamma, bias = tf.split(fc2, 2, axis=3)
        else:
            gamma = tf.layers.dense(squeeze, units=channels)
            bias = 0

        sig = tf.nn.sigmoid(gamma)
        # Explicitly signal the broadcast.
        scale = tf.reshape(sig, [-1, 1, 1, channels])

        excitation = tf.multiply(scale, residual) + bias
        return mg_activation(inputs + excitation)

    initial_block = mg_activation(mg_batchn(mg_conv2d(features)))

    # the shared stack
    shared_output = initial_block
    for _ in range(params['trunk_layers']):
        if FLAGS.use_SE or FLAGS.use_SE_bias:
            shared_output = mg_squeeze_excitation_layer(shared_output)
        else:
            shared_output = mg_res_layer(shared_output)

    # Policy head
    policy_conv = mg_conv2d(
        shared_output, filters=params['policy_conv_width'], kernel_size=1)
    policy_conv = mg_activation(
        mg_batchn(policy_conv, center=False, scale=False))
    logits = tf.layers.dense(
        tf.reshape(
            policy_conv, [-1, params['policy_conv_width'] * go.N * go.N]),
        go.N * go.N + 1)

    policy_output = tf.nn.softmax(logits, name='policy_output')

    # Value head
    value_conv = mg_conv2d(
        shared_output, filters=params['value_conv_width'], kernel_size=1)
    value_conv = mg_activation(
        mg_batchn(value_conv, center=False, scale=False))

    value_fc_hidden = mg_activation(tf.layers.dense(
        tf.reshape(value_conv, [-1, params['value_conv_width'] * go.N * go.N]),
        params['fc_width']))
    value_output = tf.nn.tanh(
        tf.reshape(tf.layers.dense(value_fc_hidden, 1), [-1]),
        name='value_output')

    return policy_output, value_output, logits


def tpu_model_inference_fn(features):
    """Builds the model graph suitable for running on TPU.

    It does two things:
     1) Mark all weights as constant, which improves TPU inference performance
        because it prevents the weights being transferred to the TPU every call
        to Session.run().
     2) Adds constant to the graph with a unique value and marks it as a
        dependency on the rest of the model. This works around a TensorFlow bug
        that prevents multiple models being run on a single TPU.

    Returns:
        (policy_output, value_output, logits) tuple of tensors.
    """
    def custom_getter(getter, name, *args, **kwargs):
        with tf.control_dependencies(None):
            return tf.guarantee_const(
                getter(name, *args, **kwargs), name=name + '/GuaranteeConst')
    with tf.variable_scope('', custom_getter=custom_getter):
        # TODO(tommadams): remove the tf.control_dependencies context manager
        # when a fixed version of TensorFlow is released.
        t = int(time.time())
        epoch_time = tf.constant(t, name='epoch_time_%d' % t)
        with tf.control_dependencies([epoch_time]):
            return model_inference_fn(features, False, FLAGS.flag_values_dict())


def maybe_set_seed():
    if FLAGS.training_seed != 0:
        random.seed(FLAGS.training_seed)
        tf.set_random_seed(FLAGS.training_seed)
        np.random.seed(FLAGS.training_seed)


def get_estimator():
    if FLAGS.use_tpu:
        return _get_tpu_estimator()
    else:
        return _get_nontpu_estimator()


def _get_nontpu_estimator():
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        save_summary_steps=FLAGS.summary_steps,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
        session_config=session_config)
    return tf.estimator.Estimator(
        model_fn,
        model_dir=FLAGS.work_dir,
        config=run_config,
        params=FLAGS.flag_values_dict())


def _get_tpu_estimator():
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=None, project=None)
    tpu_grpc_url = tpu_cluster_resolver.get_master()

    run_config = tpu_config.RunConfig(
        master=tpu_grpc_url,
        evaluation_master=tpu_grpc_url,
        model_dir=FLAGS.work_dir,
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
        eval_batch_size=FLAGS.train_batch_size * FLAGS.num_tpu_cores,
        params=FLAGS.flag_values_dict())


def bootstrap():
    """Initialize a tf.Estimator run with random initial weights."""
    # a bit hacky - forge an initial checkpoint with the name that subsequent
    # Estimator runs will expect to find.
    #
    # Estimator will do this automatically when you call train(), but calling
    # train() requires data, and I didn't feel like creating training data in
    # order to run the full train pipeline for 1 step.
    maybe_set_seed()
    initial_checkpoint_name = 'model.ckpt-1'
    save_file = os.path.join(FLAGS.work_dir, initial_checkpoint_name)
    sess = tf.Session(graph=tf.Graph())
    with sess.graph.as_default():
        features, labels = get_inference_input()
        model_fn(features, labels, tf.estimator.ModeKeys.PREDICT,
                 params=FLAGS.flag_values_dict())
        sess.run(tf.global_variables_initializer())
        tf.train.Saver().save(sess, save_file)


def export_model(model_path):
    """Take the latest checkpoint and copy it to model_path.

    Assumes that all relevant model files are prefixed by the same name.
    (For example, foo.index, foo.meta and foo.data-00000-of-00001).

    Args:
        model_path: The path (can be a gs:// path) to export model
    """
    estimator = tf.estimator.Estimator(model_fn, model_dir=FLAGS.work_dir,
                                       params=FLAGS.flag_values_dict())
    latest_checkpoint = estimator.latest_checkpoint()
    all_checkpoint_files = tf.gfile.Glob(latest_checkpoint + '*')
    for filename in all_checkpoint_files:
        suffix = filename.partition(latest_checkpoint)[2]
        destination_path = model_path + suffix
        print('Copying {} to {}'.format(filename, destination_path))
        tf.gfile.Copy(filename, destination_path)


def freeze_graph(model_path, use_trt=False, trt_max_batch_size=8,
                 trt_precision='fp32'):
    output_names = ['policy_output', 'value_output']

    n = DualNetwork(model_path)
    out_graph = tf.graph_util.convert_variables_to_constants(
        n.sess, n.sess.graph.as_graph_def(), output_names)

    if use_trt:
        out_graph = trt.create_inference_graph(
            input_graph_def=out_graph,
            outputs=output_names,
            max_batch_size=trt_max_batch_size,
            max_workspace_size_bytes=1 << 29,
            precision_mode=trt_precision)
        with tf.gfile.GFile(model_path + '.trt.pb', 'wb') as f:
            f.write(out_graph.SerializeToString())
    else:
        with tf.gfile.GFile(model_path + '.pb', 'wb') as f:
            f.write(out_graph.SerializeToString())


def freeze_graph_tpu(model_path):
    """Custom freeze_graph implementation for Cloud TPU."""

    assert model_path
    assert FLAGS.tpu_name
    if FLAGS.tpu_name.startswith('grpc://'):
        tpu_grpc_url = FLAGS.tpu_name
    else:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=None, project=None)
        tpu_grpc_url = tpu_cluster_resolver.get_master()
    sess = tf.Session(tpu_grpc_url)

    output_names = []
    with sess.graph.as_default():
        # Replicate the inference function for each TPU core.
        replicated_features = []
        for i in range(FLAGS.num_tpu_cores):
            features = tf.placeholder(
                tf.float32, [None, go.N, go.N,
                             features_lib.NEW_FEATURES_PLANES],
                name='pos_tensor_%d' % i)
            replicated_features.append((features,))
        outputs = tf.contrib.tpu.replicate(
            tpu_model_inference_fn, replicated_features)

        # The replicate op assigns names like output_0_shard_0 to the output
        # names. Give them human readable names.
        for i, (policy_output, value_output, _) in enumerate(outputs):
            policy_name = 'policy_output_%d' % i
            value_name = 'value_output_%d' % i
            output_names.extend([policy_name, value_name])
            tf.identity(policy_output, policy_name)
            tf.identity(value_output, value_name)

        tf.train.Saver().restore(sess, model_path)

    # Freeze the graph.
    model_def = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), output_names)
    with tf.gfile.GFile(model_path + '.pb', 'wb') as f:
        f.write(model_def.SerializeToString())
