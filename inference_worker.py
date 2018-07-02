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

flags.DEFINE_integer("parallel_tpus", 8,
                     "Number of TPU cores to run on in parallel.")

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


def main():
    """Runs the inference worker."""

    tf.logging.set_verbosity(tf.logging.DEBUG)

    config = get_server_config()
    print(config)
    if config.board_size != go.N:
        raise RuntimeError("Board size mismatch: server=%d, worker=%d" % (
            config.board_size, go.N))

    positions_per_inference = config.games_per_inference * config.virtual_losses
    if positions_per_inference % FLAGS.parallel_tpus != 0:
        raise RuntimeError(
            "games_per_inference * virtual_losses must be divisible by "
            "parallel_tpus")
    batch_size = positions_per_inference // FLAGS.parallel_tpus

    print("parallel_tpus = %d" % FLAGS.parallel_tpus)
    print("games_per_inference = %d" % config.games_per_inference)
    print("virtual_losses = %d" % config.virtual_losses)
    print("positions_per_inference = %d" % positions_per_inference)
    print("batch_size = %d" % batch_size)
    sys.stdout.flush()

    num_board_features = go.N * go.N * features_lib.NEW_FEATURES_PLANES

    tpu_init = tf.contrib.tpu.initialize_system()
    tpu_shutdown = tf.contrib.tpu.shutdown_system()

    tpu_grpc_url = tf.contrib.cluster_resolver.TPUClusterResolver(
        tpu=[FLAGS.tpu_name]).get_master()

    sess = tf.Session(tpu_grpc_url)
    features_list = []
    with sess.graph.as_default():
        for i in range(FLAGS.parallel_tpus):
            features = tf.placeholder(
                tf.float32, [None, go.N, go.N, features_lib.NEW_FEATURES_PLANES],
                name='pos_tensor')
            features_list.append((features,))

        replicate_outputs = tf.contrib.tpu.replicate(
            const_model_inference_fn, features_list)

        tf.train.Saver().restore(sess, FLAGS.model)

    print("initializing tpu")
    sess.run(tpu_init)

    print("warming up")
    warm_up = []
    for i in range(FLAGS.parallel_tpus):
        warm_up.append(np.random.rand(batch_size, go.N, go.N,
                                      features_lib.NEW_FEATURES_PLANES))

    outputs = sess.run(replicate_outputs, {tuple(features_list): warm_up})

    channel = grpc.insecure_channel(FLAGS.local_address, GRPC_OPTIONS)
    stub = inference_service_pb2_grpc.InferenceServiceStub(channel)

    def Loop():
        while True:
            features_response = stub.GetFeatures(
                inference_service_pb2.GetFeaturesRequest())
            all_features = features_response.byte_features

            features = []
            num_features = batch_size * num_board_features
            for i in range(FLAGS.parallel_tpus):
                begin = i * num_features
                end = begin + num_features
                x = np.frombuffer(
                    all_features, dtype=np.int8, count=num_features, offset=begin)
                x = x.reshape(
                    [batch_size, go.N, go.N, features_lib.NEW_FEATURES_PLANES])
                features.append(x)

            outputs = sess.run(replicate_outputs, {tuple(features_list): features})

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
