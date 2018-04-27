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

import argh
import subprocess
from utils import timer


def loop(working_dir=None):
    """Run train and validate as subprocesses."""
    while True:
        print("==================================")
        with timer("Train"):
            train = subprocess.call(
                ("python rl_loop.py train --working-dir=%s" % working_dir).split())
            if train != 0:
                print("Skipping validation")
                continue

        with timer("validate"):
            subprocess.call("python rl_loop.py validate", shell=True)


if __name__ == '__main__':
    argh.dispatch_command(loop)
