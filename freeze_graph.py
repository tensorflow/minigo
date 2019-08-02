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

"""Freeze a model to a GraphDef proto."""

from absl import app, flags

import dual_net

flags.DEFINE_string('model_path', None, 'Path to model to freeze')

flags.mark_flag_as_required('model_path')

flags.DEFINE_boolean(
    'use_trt', False, 'True to write a GraphDef that uses the TRT runtime')
flags.DEFINE_integer('trt_max_batch_size', None,
                     'Maximum TRT batch size')
flags.DEFINE_string('trt_precision', 'fp32',
                    'Precision for TRT runtime: fp16, fp32 or int8')
flags.register_multi_flags_validator(
    ['use_trt', 'trt_max_batch_size'],
    lambda flags: not flags['use_trt'] or flags['trt_max_batch_size'],
    'trt_max_batch_size must be set if use_trt is true')

FLAGS = flags.FLAGS


def main(unused_argv):
    """Freeze a model to a GraphDef proto."""
    if FLAGS.use_tpu:
        dual_net.freeze_graph_tpu(FLAGS.model_path)
    else:
        dual_net.freeze_graph(FLAGS.model_path, FLAGS.use_trt,
                              FLAGS.trt_max_batch_size, FLAGS.trt_precision)


if __name__ == "__main__":
    app.run(main)
