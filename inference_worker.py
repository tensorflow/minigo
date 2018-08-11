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

import abc
from contextlib import contextmanager
import sys
import time
import threading

from absl import flags
import grpc
import numpy as np
from proto import inference_service_pb2
from proto import inference_service_pb2_grpc
import tensorflow as tf
from tensorflow.python.training import saver

import dual_net
import features as features_lib
import go
from utils import dbg


flags.DEFINE_string("model", "", "Path to the TensorFlow model.")

flags.DEFINE_string("checkpoint_dir", "",
                    "Path to a directory containing TensorFlow model "
                    "checkpoints. The inference worker will monitor this "
                    "when a new checkpoint is found, load the model and use it "
                    "for futher inferences.")

flags.DEFINE_string("server_address", "localhost:50051",
                    "Inference server local address.")

flags.DEFINE_string("descriptor",
                    "proto/inference_service_py_pb2.pb.descriptor_set",
                    "Path to the InferenceService proto descriptor.")

flags.DEFINE_integer("parallel_tpus", 8,
                     "Number of TPU cores to run on in parallel.")

FLAGS = flags.FLAGS


# The default maximum receive RPC size is only 4MB, which isn't large enough
# for our messages.
GRPC_OPTIONS = [
    ("grpc.max_message_length", 50 * 1024 * 1024),
    ("grpc.max_receive_message_length", 50 * 1024 * 1024),
]

NUM_WORKER_THREADS = 2


class RwMutex(object):
    """A simple read/write mutex.

    I'm surprised Python doesn't provide one of these by default.
    """

    def __init__(self):
        self._resource_lock = threading.Semaphore()
        self._read_lock = threading.Semaphore()
        self._read_count = 0

    @contextmanager
    def write_lock(self):
        self._acquire_write()
        try:
            yield
        finally:
            self._release_write()

    @contextmanager
    def read_lock(self):
        self._acquire_read()
        try:
            yield
        finally:
            self._release_read()

    def _acquire_write(self):
        self._resource_lock.acquire()

    def _release_write(self):
        self._resource_lock.release()

    def _acquire_read(self):
        with self._read_lock:
            self._read_count += 1
            if self._read_count == 1:
                self._resource_lock.acquire()

    def _release_read(self):
        with self._read_lock:
            self._read_count -= 1
            if self._read_count == 0:
                self._resource_lock.release()


def const_model_inference_fn(features):
    """Builds the model graph with weights marked as constant.

    This improves TPU inference performance because it prevents the weights
    being transferred to the TPU every call to Session.run().

    Returns:
        (policy_output, value_output, logits) tuple of tensors.
    """
    def custom_getter(getter, name, *args, **kwargs):
        with tf.control_dependencies(None):
            return tf.guarantee_const(
                getter(name, *args, **kwargs), name=name + "/GuaranteeConst")
    with tf.variable_scope("", custom_getter=custom_getter):
        return dual_net.model_inference_fn(features, False)


class Session(abc.ABC):
    def __init__(self, sess):
        self._sess = sess

        # Event that gets set after a model is loaded.
        # The worker threads wait for this event before starting inference.
        self.model_available = threading.Event()
        self._model_path = None
        self._mutex = RwMutex()

    def maybe_load_model(self, path):
        """Loads the given model if it's different from the current one."""
        with self._mutex.read_lock():
            if path == self._model_path:
                return

        with self._mutex.write_lock():
            dbg(time.time(), "loading %s" % path)
            self._locked_load_model(path)
            self._model_path = path
            dbg(time.time(), "loaded %s" % path)
        self.model_available.set()

    def run(self, raw_features):
        """Performs inference on the given raw features."""
        features = self._prepare_features(raw_features)
        with self._mutex.read_lock():
            policy, value = self._locked_run(features)
            local_model_path = self._model_path

        return policy, value, local_model_path

    def shutdown(self):
        """Shuts down the session."""
        with self._mutex.write_lock():
            self._locked_shutdown()

    @abc.abstractmethod
    def _locked_load_model(self, path):
        """Device-specific wrapper around a call to _load_graph.

        Must be called with self._lock held for write.
        """
        pass

    @abc.abstractmethod
    def _locked_run(self, raw_features):
        """Device-specific evaluation of the model with the given raw features.

        Must be called with self._lock held for read.
        """
        pass

    @abc.abstractmethod
    def _locked_shutdown(self, raw_features):
        """Device-specific shutdown.

        Must be called with self._lock held for write.
        """
        pass

    @abc.abstractmethod
    def _prepare_features(self, raw_features):
        """Device-specific preparation of raw features.

        Does not require a lock to be held.
        """
        pass


class BasicSession(Session):
    def __init__(self):
        Session.__init__(self, tf.Session(graph=tf.Graph()))

        with self._sess.graph.as_default():
            self._feature_placeholder = tf.placeholder(
                tf.float32, [None, go.N, go.N,
                             features_lib.NEW_FEATURES_PLANES],
                name='pos_tensor')

    def _locked_shutdown(self):
        pass

    def _locked_load_model(self, path):
        tf.reset_default_graph()

        if path[-3:] == ".pb":
            graph_def = tf.GraphDef()
            with tf.gfile.FastGFile(path, 'rb') as f:
                graph_def.ParseFromString(f.read())
            with self._sess.graph.as_default():
                self._outputs = tf.import_graph_def(
                    graph_def,
                    input_map={'pos_tensor': self._feature_placeholder},
                    return_elements=['policy_output:0', 'value_output:0'])
        else:
            with self._sess.graph.as_default():
                self._outputs = dual_net.model_inference_fn(
                    self._feature_placeholder, training=False)
                tf.train.Saver().restore(self._sess, path)

    def _locked_run(self, features):
        outputs = self._sess.run(self._outputs,
                                 {self._feature_placeholder: features})
        return outputs[0], outputs[1]

    def _prepare_features(self, raw_features):
        features = np.frombuffer(raw_features, dtype=np.int8)
        features = features.reshape([-1, go.N, go.N,
                                     features_lib.NEW_FEATURES_PLANES])
        return features


class TpuSession(Session):
    def __init__(self, tpu_name, parallel_tpus, batch_size):
        tpu = [tpu_name] if tpu_name else None
        tpu_grpc_url = tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu=tpu).get_master()
        sess = tf.Session(tpu_grpc_url)
        Session.__init__(self, sess)

        self._parallel_tpus = parallel_tpus
        self._batch_size = batch_size

        # Create init & shutdown ops up front. This is probably not really
        # necessary but it's what the sample code does.
        self._tpu_init = tf.contrib.tpu.initialize_system()
        self._tpu_shutdown = tf.contrib.tpu.shutdown_system()

        self._feature_placeholders = []
        with self._sess.graph.as_default():
            for i in range(parallel_tpus):
                features = tf.placeholder(
                    tf.float32, [None, go.N, go.N,
                                 features_lib.NEW_FEATURES_PLANES],
                    name='pos_tensor')
                self._feature_placeholders.append((features,))

            self._outputs = tf.contrib.tpu.replicate(
                const_model_inference_fn, self._feature_placeholders)

            # tpu.replicate requires a list, but sess.run requires a tuple...
            self._feature_placeholders = tuple(self._feature_placeholders)

    def _locked_shutdown(self):
        self._sess.run(self._tpu_shutdown)

    def _locked_load_model(self, path):
        if self._model_path:
            dbg("shutting down tpu")
            self._sess.run(self._tpu_shutdown)

        with self._sess.graph.as_default():
            tf.train.Saver().restore(self._sess, path)

        dbg("initializing tpu")
        self._sess.run(self._tpu_init)

    def _locked_run(self, features):
        outputs = self._sess.run(self._outputs,
                                 {self._feature_placeholders: features})
        policy = []
        value = []
        for x in outputs:
            policy.extend(x[0])
            value.extend(x[1])
        return policy, value

    def _prepare_features(self, raw_features):
        num_board_features = go.N * go.N * features_lib.NEW_FEATURES_PLANES
        num_features = self._batch_size * num_board_features
        assert len(raw_features) == num_features * self._parallel_tpus

        features = []
        for i in range(self._parallel_tpus):
            begin = i * num_features
            x = np.frombuffer(
                raw_features, dtype=np.int8, count=num_features, offset=begin)
            x = x.reshape([self._batch_size, go.N, go.N,
                           features_lib.NEW_FEATURES_PLANES])
            features.append(x)
        return features


class Worker(object):
    def __init__(self):
        self.parallel_inferences = FLAGS.parallel_tpus if FLAGS.use_tpu else 1

        self._get_server_config()

        if FLAGS.use_tpu:
            self.sess = TpuSession(
                FLAGS.tpu_name, self.parallel_inferences, self.batch_size)
        else:
            self.sess = BasicSession()

        if FLAGS.model:
            self.sess.maybe_load_model(FLAGS.model)

    def run(self):
        self._running = True
        try:
            self._run_threads()
        finally:
            self._running = False
            dbg("shutting down session")
            self.sess.shutdown()
            dbg("all done!")

    def _get_server_config(self):
        while True:
            try:
                channel = grpc.insecure_channel(FLAGS.server_address)
                self.stub = inference_service_pb2_grpc.InferenceServiceStub(
                    channel)
                config = self.stub.GetConfig(
                    inference_service_pb2.GetConfigRequest())
                break
            except grpc.RpcError:
                dbg("Waiting for server")
                time.sleep(1)

        if config.board_size != go.N:
            raise RuntimeError("Board size mismatch: server=%d, worker=%d" % (
                config.board_size, go.N))

        positions_per_inference = (config.games_per_inference *
                                   config.virtual_losses)
        if positions_per_inference % self.parallel_inferences != 0:
            raise RuntimeError(
                "games_per_inference * virtual_losses must be divisible by "
                "parallel_tpus")
        self.batch_size = positions_per_inference // self.parallel_inferences

        dbg("parallel_inferences = %d" % self.parallel_inferences)
        dbg("games_per_inference = %d" % config.games_per_inference)
        dbg("virtual_losses = %d" % config.virtual_losses)
        dbg("positions_per_inference = %d" % positions_per_inference)
        dbg("batch_size = %d" % self.batch_size)

    def _run_threads(self):
        """Run inference threads and optionally a thread that updates the model.

        Synchronization between the inference threads and the model update
        thread is performed using a RwLock that protects access to self.sess.
        The inference threads enter the critical section using a read lock, so
        they can both run inference concurrently. The model update thread enters
        the critical section using a write lock for exclusive access.
        """
        threads = []
        # Start the worker threads before the checkpoint thread: if the parent
        # process dies, the worker thread RPCs will fail and the thread will
        # exit. This gives us a chance below to set self._running to False,
        # telling the checkpoint thread to exit.
        for i in range(NUM_WORKER_THREADS):
            threads.append(threading.Thread(
                target=self._worker_thread, args=[i]))
        if FLAGS.checkpoint_dir:
            threads.append(threading.Thread(target=self._checkpoint_thread))

        for t in threads:
            t.start()
        for i, t in enumerate(threads):
            t.join()
            dbg("joined thread %d" % i)
            # Once the first thread has joined, tell the remaining ones to stop.
            self._running = False

    def _checkpoint_thread(self):
        dbg("starting model loader thread")
        while self._running:
            freshest = saver.latest_checkpoint(FLAGS.checkpoint_dir)
            if freshest:
                self.sess.maybe_load_model(freshest)
            # Wait a few seconds before checking again.
            time.sleep(5)

    def _worker_thread(self, thread_id):
        dbg("waiting for model")
        while self._running and not self.sess.model_available.wait(1):
            pass

        dbg("running worker", thread_id)
        while self._running:
            features_response = self.stub.GetFeatures(
                inference_service_pb2.GetFeaturesRequest())

            policy, value, model_path = self.sess.run(
                features_response.features)

            put_outputs_request = inference_service_pb2.PutOutputsRequest(
                batch_id=features_response.batch_id,
                policy=np.concatenate(policy), value=value,
                model_path=model_path)

            self.stub.PutOutputs(put_outputs_request)

        dbg("stopping worker", thread_id)


def main():
    tf.logging.set_verbosity(tf.logging.DEBUG)
    worker = Worker()
    worker.run()


if __name__ == "__main__":
    flags.FLAGS(sys.argv, known_only=True)
    main()
