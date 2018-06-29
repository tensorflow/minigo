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
Worker process for running remote inference.
The worker wraps the inference model in an infinte loop: input features are
fetched via RPC at the top of the loop, and inference output is written back
at the bottom (again, via RPC).
"""

import sys
import time
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib.proto.python.ops import decode_proto_op
from tensorflow.contrib.proto.python.ops import encode_proto_op
import threading
import numpy as np
from absl import flags
import grpc
from proto import inference_service_pb2
from proto import inference_service_pb2_grpc
import dual_net
import features as features_lib
import go

flags.DEFINE_string("model", "", "Path to the TensorFlow model.")
flags.DEFINE_string("local_address", "localhost:50051",
                    "Inference server local address.")
flags.DEFINE_string("remote_address", "10.128.0.2:50051",
                    "Inference server remote address.")
flags.DEFINE_string("descriptor",
                    "proto/inference_service_py_pb2.pb.descriptor_set",
                    "Path to the InferenceService proto descriptor.")
flags.DEFINE_integer("batch_size", 8, "Batch size.")

FLAGS = flags.FLAGS


GRPC_OPTIONS = [
    ("grpc.max_message_length", 50 *  1024 * 1024),
    ("grpc.max_receive_message_length", 50 *  1024 * 1024),
]

def get_server_config():
    """Connects to the inference server and fetches its configuration.
    Returns:
        Server's configuration as a inference_service_pb2.GetConfigResponse
        proto.
    """
    while True:
        try:
            # Fetch the server config, used to set batch size.
            channel = grpc.insecure_channel(FLAGS.local_address)
            stub = inference_service_pb2_grpc.InferenceServiceStub(channel)
            return stub.GetConfig(inference_service_pb2.GetConfigRequest())
        except grpc.RpcError:
            print("Waiting for server")
            time.sleep(1)


def const_model_inference_fn(features):
    def custom_getter(getter, name, *args, **kwargs):
        with tf.control_dependencies(None):
            return tf.guarantee_const(
                getter(name, *args, **kwargs), name=name+"/GuaranteeConst")
    with tf.variable_scope("", custom_getter=custom_getter):
        return dual_net.model_inference_fn(features, False)


def wrapped_model_inference_fn():
    """Wraps dual_net.model_inference_fn in a loop & RPC ops.
    The loop runs forever: the top of the loop issues a GetFeatures RPC to
    fetch input features, the bottom of the loop isses a PutOutputs RPC to
    write the inference output back to the server.

    Returns:
        A tensor op that drives the model's infinite loop.
    """

    value_output_size = FLAGS.batch_size
    policy_output_size = FLAGS.batch_size * (go.N * go.N + 1)

    def loop_condition(a, unused_b):
        """Loop condition for the tf.while_loop op.
        Args:
            a: a constant 0
            unused_b: a string placeholder (to satisfy the requirement that a
                      while_loop's condition and body accept the same args as
                      the loop returns).
        Returns:
            A TensorFlow subgraph that returns true.
        """

        # TensorFlow will reject a loop unless its condition contains at least
        # one comparison. So we use (a < 1), initialize a to 0, and never
        # increment it.
        return tf.less(a, 1)

    def loop_body(a, unused_b):
        """Loop body for the tf.while_loop op.
        Args:
            a: a constant 0
            unused_b: a string placeholder (to satisfy the requirement that a
                      while_loop's condition and body accept the same args as
                      the loop returns).
        Returns:
            A TensorFlow subgraph.
        """

        # Request features features.
        raw_response = tf.contrib.rpc.rpc(
            address=FLAGS.remote_address,
            method="/minigo.InferenceService/GetFeatures",
            request="",
            protocol="grpc",
            fail_fast=True,
            timeout_in_ms=0,
            name="get_features")

        # Decode features from a proto to a flat tensor.
        _, (batch_id, flat_features) = decode_proto_op.decode_proto(
            bytes=raw_response,
            message_type='minigo.GetFeaturesResponse',
            field_names=['batch_id', 'features'],
            output_types=[dtypes.int32, dtypes.float32],
            descriptor_source=FLAGS.descriptor,
            name="decode_raw_features_response")

        # Reshape flat features.
        features = tf.reshape(
            flat_features, [-1, go.N, go.N, features_lib.NEW_FEATURES_PLANES],
            name="unflatten_features")

        # Run inference.
        policy_output, value_output, _ = const_model_inference_fn(features)

        # Flatten model outputs.
        flat_policy = tf.reshape(policy_output, [-1], name="flatten_policy")
        flat_value = value_output  # value_output is already flat.

        # Encode outputs from flat tensors to a proto.
        request_tensors = encode_proto_op.encode_proto(
            message_type='minigo.PutOutputsRequest',
            field_names=['batch_id', 'policy', 'value'],
            sizes=[[1, policy_output_size, value_output_size]],
            values=[[batch_id], [flat_policy], [flat_value]],
            descriptor_source=FLAGS.descriptor,
            name="encode_outputs")

        # Send outputs.
        raw_response = tf.contrib.rpc.rpc(
            address=FLAGS.remote_address,
            method="/minigo.InferenceService/GetFeatures",
            request=request_tensors,
            protocol="grpc",
            fail_fast=True,
            timeout_in_ms=0,
            name="put_outputs")

        _, batch_id = decode_proto_op.decode_proto(
            bytes=raw_response,
            message_type='minigo.PutOutputsResponse',
            field_names=['batch_id'],
            output_types=[dtypes.int32],
            descriptor_source=FLAGS.descriptor,
            name="decode_put_outputs_response")

        return a, tf.reshape(batch_id, (1,))

    loop_vars = [0, tf.constant(0, shape=[1], dtype=dtypes.int32)]
    return tf.while_loop(loop_condition, loop_body, loop_vars,
                         name="inference_worker_loop")


def main():
    """Runs the inference worker."""

    tf.logging.set_verbosity(tf.logging.DEBUG)

    server_config = get_server_config()
    print(server_config)
    if server_config.board_size != go.N:
        raise RuntimeError("Board size mismatch: server=%d, worker=%d" % (
            server_config.board_size, go.N))
    if server_config.batch_size != 8 * FLAGS.batch_size:
        raise RuntimeError("Batch size mismatch: server=%d, worker=%d" % (
            server_config.batch_size, FLAGS.batch_size))

    tpu_init = tf.contrib.tpu.initialize_system()
    tpu_shutdown = tf.contrib.tpu.shutdown_system()

    tpu_grpc_url = tf.contrib.cluster_resolver.TPUClusterResolver(
        tpu=[FLAGS.tpu_name]).get_master()


    ##################################################


    """
    # sess = tf.Session(tpu_grpc_url, config=tf.ConfigProto(allow_soft_placement=True))
    sess = tf.Session(tpu_grpc_url)
    with sess.graph.as_default():
        loop = tf.contrib.tpu.rewrite(wrapped_model_inference_fn, [])

        tf.train.Saver().restore(sess, FLAGS.model)

    sess.run(tpu_init)
    outputs = sess.run(loop)
    """


    sess = tf.Session(tpu_grpc_url)
    features_list = []
    with sess.graph.as_default():
        for i in range(8):
            features = tf.placeholder(
                tf.float32, [None, go.N, go.N, features_lib.NEW_FEATURES_PLANES],
                name='pos_tensor')
            features_list.append((features,))

        replicate_outputs = tf.contrib.tpu.replicate(
            const_model_inference_fn, features_list)

        tf.train.Saver().restore(sess, FLAGS.model)

    print(replicate_outputs)

    print("initializing tpu")
    sess.run(tpu_init)

    print("warming up")
    warm_up = []
    for i in range(8):
        warm_up.append(np.random.rand(FLAGS.batch_size, go.N, go.N, features_lib.NEW_FEATURES_PLANES))

    key = tuple(features_list)
    print("------------------")
    print(key)
    print("------------------")
    outputs = sess.run(replicate_outputs, {key: warm_up})

    channel = grpc.insecure_channel(FLAGS.local_address, GRPC_OPTIONS)
    stub = inference_service_pb2_grpc.InferenceServiceStub(channel)


    def Loop():
        while True:
            N = FLAGS.batch_size * go.N * go.N * features_lib.NEW_FEATURES_PLANES
            f = []
            features_response = stub.GetFeatures(
                inference_service_pb2.GetFeaturesRequest())

            all_features = features_response.byte_features

            for i in range(8):
                begin = i * N
                end = begin + N
                x = np.frombuffer(all_features, dtype=np.int8, count=N, offset=begin)
                x = x.reshape(
                    [FLAGS.batch_size, go.N, go.N, features_lib.NEW_FEATURES_PLANES])
                f.append(x)

            outputs = sess.run(replicate_outputs, {tuple(features_list): f})

            flattened_policy_outputs = [x[0].reshape(-1) for x in outputs]
            flattened_value_outputs = [x[1].reshape(-1) for x in outputs]

            put_outputs_request = inference_service_pb2.PutOutputsRequest(
                 batch_id=features_response.batch_id,
                 policy=np.concatenate(flattened_policy_outputs),
                 value=np.concatenate(flattened_value_outputs))

            stub.PutOutputs(put_outputs_request)


    num_threads = 2
    threads = []
    for i in range(num_threads):
        threads.append(threading.Thread(target=Loop))
        threads[i].start()
    for i in range(num_threads):
        threads[i].join()

    print("shutting down TPU")
    sess.run(tpu_shutdown)
    print("all done!")


if __name__ == "__main__":
    flags.FLAGS(sys.argv, known_only=True)
    main()
