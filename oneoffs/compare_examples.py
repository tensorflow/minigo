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

import sys
sys.path.insert(0, '.')  # nopep8

from absl import app, flags
import numpy as np
import tensorflow as tf

import go


flags.DEFINE_string("a", "", "Path to first example")
flags.DEFINE_string("b", "", "Path to second example")

FLAGS = flags.FLAGS

TF_RECORD_CONFIG = tf.python_io.TFRecordOptions(
    tf.python_io.TFRecordCompressionType.ZLIB)


class ParsedExample(object):
    def __init__(self, features, pi, value, q, n, c):
        self.features = features
        self.pi = pi
        self.value = value
        self.q = q;
        self.n = n
        self.c = c


def ReadExamples(path):
    print("Reading", path)

    records = list(tf.python_io.tf_record_iterator(path, TF_RECORD_CONFIG))
    num_records = len(records)

    features = {
        'x': tf.FixedLenFeature([], tf.string),
        'pi': tf.FixedLenFeature([], tf.string),
        'outcome': tf.FixedLenFeature([], tf.float32),
        'n': tf.FixedLenFeature([], tf.int64, default_value=[-1]),
        'q': tf.FixedLenFeature([], tf.float32, default_value=[-1]),
        'c': tf.FixedLenFeature([], tf.int64, default_value=[-1]),
    }

    parsed = tf.parse_example(records, features)

    x = tf.decode_raw(parsed['x'], tf.uint8)
    x = tf.cast(x, tf.float32)
    x = tf.reshape(x, [num_records, go.N, go.N, -1])
    x = x.eval()

    pi = tf.decode_raw(parsed['pi'], tf.float32)
    pi = tf.reshape(pi, [num_records, go.N * go.N + 1])
    pi = pi.eval()

    outcome = parsed['outcome'].eval()
    n = parsed['n'].eval()
    q = parsed['q'].eval()
    c = parsed['c'].eval()

    return [ParsedExample(*args) for args in zip(x, pi, outcome, q, n, c)]


def main(unused_argv):
    with tf.Session():
        examples_a = ReadExamples(FLAGS.a)
        examples_b = ReadExamples(FLAGS.b)
    print(len(examples_a), len(examples_b))

    #assert len(examples_a) == len(examples_b)
    for i, (a, b) in enumerate(zip(examples_a, examples_b)):
        print(i, a.value, b.value)
        #assert a.value == b.value
        np.testing.assert_array_equal(a.features, b.features)
        np.testing.assert_array_almost_equal(a.pi, b.pi, decimal=4)


if __name__ == "__main__":
    app.run(main)
