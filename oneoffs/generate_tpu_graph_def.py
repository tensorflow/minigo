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

"""Generates a graph def containing TPU initialization and shutdown ops."""

from absl import app, flags
import tensorflow as tf

flags.DEFINE_string(
    'tpu_name', None,
    'The name of a Cloud TPU. This should be either the name used when creating '
    'the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

FLAGS = flags.FLAGS

def main(unused_argv):
    assert FLAGS.tpu_name
    if FLAGS.tpu_name.startswith('grpc://'):
        tpu_grpc_url = FLAGS.tpu_name
    else:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=None, project=None)
        tpu_grpc_url = tpu_cluster_resolver.get_master()

    sess = tf.Session(tpu_grpc_url)
    with sess.graph.as_default():
      tf.contrib.tpu.initialize_system()
      tf.contrib.tpu.shutdown_system()

    output_names = ['ConfigureDistributedTPU', 'ShutdownDistributedTPU']
    model_def = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), output_names)
    print(model_def)



if __name__ == "__main__":
    app.run(main)

