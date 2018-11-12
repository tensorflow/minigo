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

import os
import sys

sys.path.insert(0, '.')

from absl import app, flags
import mask_flags
from rl_loop import fsdb
from rl_loop import shipname

# From rl_loop/fsdb.py
# Must pass one or the other in.
flags.declare_key_flag('bucket_name')
flags.declare_key_flag('base_dir')


def bootstrap(unused_argv):
    bootstrap_name = shipname.generate(0)
    bootstrap_model_path = os.path.join(fsdb.models_dir(), bootstrap_name)
    mask_flags.checked_run([
        'python', 'bootstrap.py',
        '--export_path={}'.format(bootstrap_model_path),
        '--flagfile=rl_loop/distributed_flags'])


if __name__ == '__main__':
    app.run(bootstrap)
