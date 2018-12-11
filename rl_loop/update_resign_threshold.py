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

"""Updates a flagfile containing a resign threshold flag

Reads the bigtables defined by env vars in PROJECT,CBT_INSTANCE,CBT_TABLE to
compute the 95 percentile of the bleakest-evaluations found in calibration games,
then updates the flagfile on the default bucket path, resetting that value
"""

import sys
import re
import os
import time
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

sys.path.insert(0, '.')

import bigtable_input
import rl_loop.fsdb as fsdb


FLAGS = flags.FLAGS
RESIGN_FLAG_REGEX = re.compile(r'--resign_threshold=([-\d.]+)')

def get_95_percentile_bleak(n_back=500):
    """Gets the 95th percentile of bleakest_eval from bigtable"""
    end_game = int(bigtable_input._games_nr.latest_game_number())
    start_game = end_game - n_back if end_game >= n_back else 0
    moves = bigtable_input._games_nr.bleakest_moves(start_game, end_game)
    evals = np.array([m[2] for m in moves])
    return np.percentile(evals, 5)

def update_flagfile(flags_path, new_threshold):
    """Updates the flagfile at `flags_path`, changing the value for
    `resign_threshold` to `new_threshold` 
    """
    if abs(new_threshold) > 1:
        raise ValueError("Invalid new percentile for resign threshold")
    with tf.gfile.GFile(flags_path) as f:
        lines = f.read()
    if new_threshold > 0:
        new_threshold *= -1
    if not RESIGN_FLAG_REGEX.search(lines):
        print("Resign threshold flag not found in flagfile {}!  Aborting.".format(flags_path))
        sys.exit(1)
    old_threshold = RESIGN_FLAG_REGEX.search(lines).groups(1)
    lines = re.sub(RESIGN_FLAG_REGEX, "--resign_threshold={:.3f}".format(new_threshold), lines)

    print("Updated percentile from {} to {:.3f}".format(old_threshold, new_threshold))
    with tf.gfile.GFile(flags_path, 'w') as f:
        f.write(lines)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    if not 'CBT_TABLE' in os.environ:
        raise app.UsageError('CBT_TABLE not set')

    while True:
        new_pct = get_95_percentile_bleak()
        update_flagfile(fsdb.flags_path(), new_pct)
        time.sleep(60 * 3)

if __name__ == '__main__':
    app.run(main)
