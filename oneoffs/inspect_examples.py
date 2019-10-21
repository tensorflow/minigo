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

from absl import app, flags
import numpy as np
import tensorflow as tf

import coords
import features as features_lib
import go

flags.DEFINE_string("path", "", "Path to a TF example file.")
flags.DEFINE_integer("to_play_feature", 16,
                     "Index of the 'to play' feature.")

FLAGS = flags.FLAGS

TF_RECORD_CONFIG = tf.python_io.TFRecordOptions(
    tf.python_io.TFRecordCompressionType.ZLIB)


class ParsedExample(object):
    def __init__(self, features, pi, value):
        self.features = features
        self.pi = pi
        self.value = value


def ReadExamples(path):
    features = {
        'x': tf.FixedLenFeature([], tf.string),
        'pi': tf.FixedLenFeature([], tf.string),
        'outcome': tf.FixedLenFeature([], tf.float32),
    }

    result = []
    for record in tf.python_io.tf_record_iterator(path, TF_RECORD_CONFIG):
        example = tf.train.Example()
        example.ParseFromString(record)

        parsed = tf.parse_example([record], features)

        x = tf.decode_raw(parsed['x'], tf.uint8)
        x = tf.cast(x, tf.float32)
        x = tf.reshape(x, [go.N, go.N, -1])

        pi = tf.decode_raw(parsed['pi'], tf.float32)
        pi = tf.reshape(pi, [go.N * go.N + 1])

        outcome = parsed['outcome']
        assert outcome.shape == (1,)

        result.append(ParsedExample(x.eval(), pi.eval(), outcome.eval()))
    return result


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


def format_pi(pi, mean, mx):
    GREEN = '\x1b[0;32m'
    BRIGHT_YELLOW = '\x1b[0;33;1m'
    BRIGHT_WHITE = '\x1b[0;37;1m'
    BLUE = '\x1b[0;34m'
    NORMAL = '\x1b[0m'

    if pi < 1:
        s = ('%.3f' % pi)[1:]
    else:
        s = '%.2f' % pi

    if s == '.000':
        col = BLUE
    elif pi < mean:
        col = GREEN
    elif pi < mx:
        col = BRIGHT_YELLOW
    else:
        col = BRIGHT_WHITE
    s = '%s%s%s' % (col, s, NORMAL)
    return s


def print_board_and_pi(examples, i):
    example = examples[i]
    p = parse_board(example)
    print('\nExample %d of %d, %s to play, winner is %s' % (
        i + 1, len(examples), 'Black' if p.to_play == 1 else 'White',
        'Black' if example.value > 0 else 'White'))
    board_lines = str(p).split('\n')[:-2]

    mean = np.mean(example.pi[example.pi > 0])
    mx = np.max(example.pi)

    pi_lines = ['PI']
    for row in range(go.N):
        pi_lines.append(
            ' '.join([format_pi(x, mean, mx)
                      for x in example.pi[row * go.N: (row + 1) * go.N]]))
    pi_lines.append(format_pi(example.pi[-1], mean, mx))

    for b, p in zip(board_lines, pi_lines):
        print('%s  |  %s' % (b, p))


def main(unused_argv):
    with tf.Session():
        examples = ReadExamples(FLAGS.path)

    i = 0
    while i < len(examples):
        example = examples[i]
        print_board_and_pi(examples, i)
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
