# Copyright 2019 Google LLC
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
"""Replay a single game from an SGF.

Usage: python dump_game.py ${SGF_PATH}
"""

import sys
sys.path.insert(0, '.')  # nopep8

from absl import app
from absl import flags
import os
import re
import time

FLAGS = flags.FLAGS
flags.DEFINE_float('sleep_time', 0.0,
                   'The time to sleep between printing moves.')



def main(argv):
    # It takes a couple of seconds to import anything from tensorflow, so only
    # do it if we need to read from GCS.
    path = argv[1]
    if path.startswith('gs://'):
        from tensorflow import gfile
        f = gfile.GFile(path, 'r')
    else:
        f = open(path, 'r')
    contents = f.read()
    f.close()

    # Determine the board size before importing any Minigo libraries because
    # require that the BOARD_SIZE environment variable is set correctly before
    # import.
    m = re.search(r'SZ\[([^]]+)', contents)
    if not m:
        print('Couldn\'t find SZ node, assuming 19x19 board')
        board_size = 19
    else:
        board_size = int(m.group(1))

    m = re.search(r'RE\[([^]]+)', contents)
    if not m:
        print('No game result found')
    else:
        result = m.group(1)

    m = re.search(r'PB\[([^]]+)', contents)
    if not m:
        print('Couldn\'t find PB node')
        player='unknown'
    else:
        player = m.group(1)

    # Set the board size and import the Minigo libs.
    os.environ['BOARD_SIZE'] = str(board_size)
    import coords
    import go
    import sgf_wrapper

    # Replay the game.
    for x in sgf_wrapper.replay_sgf(contents):
        to_play = 'B' if x.position.to_play == 1 else 'W'
        print('{}>> {}: {}\n'.format(
            x.position, to_play, coords.to_gtp(x.next_move)))
        time.sleep(FLAGS.sleep_time)

    print(result)
    print("Black was: ", player)


if __name__ == '__main__':
    app.run(main)
