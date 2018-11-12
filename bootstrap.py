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
"""Bootstrap network weights.

Usage:
    BOARD_SIZE=19 python bootstrap.py \
    --work_dir=/tmp/estimator_working_dir \
    --export_path=/tmp/published_models_dir
"""

import os

from absl import app, flags
import dual_net
import utils

flags.DEFINE_string('export_path', None,
                    'Where to export the model after training.')

flags.DEFINE_bool('create_bootstrap', True,
                  'Whether to create a bootstrap model before exporting')

flags.declare_key_flag('work_dir')

FLAGS = flags.FLAGS


def main(unused_argv):
    """Bootstrap random weights."""
    utils.ensure_dir_exists(os.path.dirname(FLAGS.export_path))
    if FLAGS.create_bootstrap:
        dual_net.bootstrap()
    dual_net.export_model(FLAGS.export_path)


if __name__ == '__main__':
    flags.mark_flags_as_required(['work_dir', 'export_path'])
    app.run(main)
