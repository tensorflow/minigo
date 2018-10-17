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

FLAGS = flags.FLAGS


def main(unused_argv):
    """Freeze a model to a GraphDef proto."""
    if FLAGS.use_tpu:
        dual_net.freeze_graph_tpu(FLAGS.model_path)
    else:
        dual_net.freeze_graph(FLAGS.model_path)


if __name__ == "__main__":
    app.run(main)
