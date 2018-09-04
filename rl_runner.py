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

""" Run train and validate in a loop, as subprocesses.

We run as subprocesses because it gives us some isolation.
"""

import datetime as dt
import os
import subprocess
import sys

import argh
from utils import timer

BUCKET_NAME = os.environ['BUCKET_NAME']


def loop(working_dir='estimator_working_dir', tpu_name=None):
    """Run train and validate as subprocesses."""
    flags = [
        working_dir,
        '--bucket_name', BUCKET_NAME,
        '--model_dir', working_dir,
        '--use_tpu',
        '--tpu_name', tpu_name
    ]
    while True:
        print("=" * 40)
        with timer("Train"):
            train = subprocess.call(['python', 'rl_loop.py', 'train'] + flags)
            if train != 0:
                print("Skipping validation")
                print("!!!")
                print("=== Training failed at ", dt.datetime.utcnow())
                print("!!!")
                sys.exit(1)
                continue

        with timer("validate"):
            subprocess.call(['python', 'rl_loop.py', 'validate-hourly'] + flags)
            subprocess.call(['python', 'main.py', 'validate',
                             'gs://jacksona-sandbox/data/validate',
                             '--validate-name=pro'] + flags[3:])


if __name__ == '__main__':
    argh.dispatch_command(loop)
