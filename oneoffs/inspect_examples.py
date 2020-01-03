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

"""
Inspect the contents training examples.

The script loads a record of TF examples from selfplay games and allows the
user to inspect them using a simple command line interface.

Usage:
  python3 oneoffs/inspect_examples.py --path=examples.tfrecord.zz

Once the script has loaded the examples, you can examine them using the
following commands:
    n [steps]: Move to the next example (or skip forward `steps` examples).
               Pressing enter will also step forward to the next example.
    p [steps]: Move to the previous example (or skip back `steps` examples).
    q: quit.
    [gtp_coord]: Entering a GTP coordinate of a point will print the
                 corresponding point's features.
"""

import sys
sys.path.insert(0, '.')  # nopep8

# Hide the GPUs from TF. This makes startup 2x quicker on some machines.
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # nopep8

from absl import app, flags
import numpy as np
import tensorflow as tf

import coords
import features as features_lib
import go


tf.enable_eager_execution()


flags.DEFINE_string('path', '', 'Path to a TF example file.')
flags.DEFINE_integer('to_play_feature', 16,
                     'Index of the "to play" feature.')
flags.DEFINE_string('feature_layout', 'nhwc',
                    'Feature layout: "nhwc" or "nchw".')

FLAGS = flags.FLAGS


class ParsedExample(object):
    def __init__(self, features, pi, value, q, n, c):
        self.features = features
        self.pi = pi
        self.value = value
        self.q = q
        self.n = n
        self.c = c


def read_examples(path):
    records = list(tf.data.TFRecordDataset([path], 'ZLIB'))
    num_records = len(records)

    # n, q, c have default_values because they're optional.
    features = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'pi': tf.io.FixedLenFeature([], tf.string),
        'outcome': tf.io.FixedLenFeature([], tf.float32),
        'n': tf.io.FixedLenFeature([], tf.int64, default_value=[-1]),
        'q': tf.io.FixedLenFeature([], tf.float32, default_value=[-1]),
        'c': tf.io.FixedLenFeature([], tf.int64, default_value=[-1]),
    }

    parsed = tf.io.parse_example(records, features)

    x = tf.decode_raw(parsed['x'], tf.uint8)
    x = tf.cast(x, tf.float32)
    if FLAGS.feature_layout == 'nhwc':
        x = tf.reshape(x, [num_records, go.N, go.N, -1])
    elif FLAGS.feature_layout == 'nchw':
        x = tf.reshape(x, [num_records, -1, go.N, go.N])
        x = tf.transpose(x, [0, 2, 3, 1])
    else:
        raise ValueError('invalid feature_layout "%s"' % FLAGS.feature_layout)
    x = x.numpy()

    pi = tf.decode_raw(parsed['pi'], tf.float32)
    pi = tf.reshape(pi, [num_records, go.N * go.N + 1])
    pi = pi.numpy()

    outcome = parsed['outcome'].numpy()
    n = parsed['n'].numpy()
    q = parsed['q'].numpy()
    c = parsed['c'].numpy()

    return [ParsedExample(*args) for args in zip(x, pi, outcome, q, n, c)]


def parse_board(example):
    """Parses a go board from a TF example.

    Args:
      example: a ParsedExample.

    Returns:
      A go.Position parsed from the input example.
    """

    to_play_feature = example.features[0, 0, FLAGS.to_play_feature]

    to_play = go.BLACK if to_play_feature else go.WHITE
    other = go.WHITE if to_play_feature else go.BLACK

    board = np.zeros([go.N, go.N], dtype=np.int8)
    for row in range(go.N):
        for col in range(go.N):
            f = example.features[row, col]
            if f[0]:
                board[row, col] = to_play
            elif f[1]:
                board[row, col] = other
            else:
                board[row, col] = go.EMPTY

    return go.Position(board=board, to_play=to_play)


def format_pi(pi, stone, mean, mx, picked):
    # Start of the ANSI color code for this point: gray if this point was picked
    # as the next move, black otherwise.
    col = '\x1b[48;5;8;' if picked else '\x1b[0;'

    GREEN = '32m'
    BRIGHT_YELLOW = '33;1m'
    BRIGHT_WHITE = '37;1m'
    BLUE = '34m'
    RED = '31m'

    NORMAL = '\x1b[0m'

    if stone != go.EMPTY:
        return '\x1b[0;31m  %s %s' % ('X' if stone == go.BLACK else 'O', NORMAL)

    if pi < 1:
        s = ('%.3f' % pi)[1:]
    else:
        s = '%.2f' % pi

    if s == '.000':
        col += BLUE
    elif pi < mean:
        col += GREEN
    elif pi < mx:
        col += BRIGHT_YELLOW
    else:
        col += BRIGHT_WHITE
    s = '%s%s%s' % (col, s, NORMAL)
    return s


def print_example(examples, i):
    example = examples[i]
    p = parse_board(example)
    print('\nExample %d of %d, %s to play, winner is %s' % (
        i + 1, len(examples), 'Black' if p.to_play == 1 else 'White',
        'Black' if example.value > 0 else 'White'))

    if example.n != -1:
        print('N:%d  Q:%.3f  picked:%s' % (
            example.n, example.q, coords.to_gtp(coords.from_flat(example.c))))
    board_lines = str(p).split('\n')[:-2]

    mean = np.mean(example.pi[example.pi > 0])
    mx = np.max(example.pi)

    pi_lines = ['PI']
    for row in range(go.N):
        pi = []
        for col in range(go.N):
            stone = p.board[row, col]
            idx = row * go.N + col
            if example.c != -1:
                picked = example.c == row * go.N + col
            else:
                picked = False
            pi.append(format_pi(example.pi[idx], stone, mean, mx, picked))
        pi_lines.append(' '.join(pi))

    pi_lines.append(format_pi(example.pi[-1], go.EMPTY, mean, mx,
                              example.c == go.N * go.N))

    for b, p in zip(board_lines, pi_lines):
        print('%s  |  %s' % (b, p))


def main(unused_argv):
    examples = read_examples(FLAGS.path)

    i = 0
    while i < len(examples):
        example = examples[i]
        print_example(examples, i)
        sys.stdout.write('>> ')
        sys.stdout.flush()

        try:
            cmd = sys.stdin.readline().split()
        except KeyboardInterrupt:
            print()
            break

        if not cmd or cmd[0] == 'n':
            if len(cmd) == 2:
                try:
                    i += int(cmd[1])
                except:
                    print('ERROR: "%s" isn\'t an int' % cmd[1])
                    continue
            else:
                i += 1
            i = min(i, len(examples) - 1)
        elif cmd[0] == 'p':
            if len(cmd) == 2:
                try:
                    i -= int(cmd[1])
                except:
                    print('ERROR: "%s" isn\'t an int' % cmd[1])
                    continue
            else:
                i -= 1
            i = max(i, 0)
        elif cmd[0] == 'q':
            break
        else:
            try:
                c = coords.from_gtp(cmd[0].upper())
                f = example.features[c[0], c[1]]
            except:
                print('ERROR: "%s" isn\'t a valid GTP coord' % cmd[0])
                continue
            print('  plane:', ' '.join(['%2d' % i for i in range(len(f))]))
            print('feature:', ' '.join(['%2d' % x for x in f]))
    print('Bye!')


if __name__ == "__main__":
    app.run(main)
