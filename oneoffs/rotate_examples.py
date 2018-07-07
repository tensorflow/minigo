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
"""Randomly rotate the examples in a tfrecords.zz file."""

import sys
sys.path.insert(0, '.')

import itertools
import os.path
import multiprocessing as mp

from absl import app, flags
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import dual_net
import preprocessing
import symmetries

# This file produces a lot of logging, supress most of it
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags.DEFINE_string("in_dir", None, "tfrecord.zz in this dir are converted.")
flags.DEFINE_string("out_dir", None, "Records are writen to this dir.")
flags.DEFINE_bool("compare", False, "Whether to run compare after rotation.")
flags.DEFINE_integer("threads", None, "number of threads, default: num cpus.")
flags.DEFINE_integer("batch_size", 100, "batch_size for rotating.")

FLAGS = flags.FLAGS
OPTS = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)


def grouper(n, iterable):
    """Itertools recipe
    >>> list(grouper(3, iter('ABCDEFG')))
    [['A', 'B', 'C'], ['D', 'E', 'F'], ['G']]
    """
    return iter(lambda: list(itertools.islice(iterable, n)), [])


def batched_reader(file_path):
    reader = tf.python_io.tf_record_iterator(file_path, OPTS)
    return grouper(FLAGS.batch_size, reader)


def get_size(path):
    return tf.gfile.Stat(path).length


def convert(paths):
    position, in_path, out_path = paths
    assert tf.gfile.Exists(in_path)
    assert tf.gfile.Exists(os.path.dirname(out_path))

    in_size = get_size(in_path)
    if tf.gfile.Exists(out_path):
        # Make sure out_path is about the size of in_path
        size = get_size(out_path)
        error = (size - good_size) / (in_size + 1)
        # 5% smaller to 20% larger
        if -0.05 < error < 0.20:
            return out_path + " already existed"
        return "ERROR on file size ({:.1f}% diff) {}".format(
            100 * error, out_path)

    #assert abs(in_size/2**20 - 670) <= 80, in_size
    num_batches = dual_net.EXAMPLES_PER_GENERATION // FLAGS.batch_size + 1

    with tf.python_io.TFRecordWriter(out_path, OPTS) as writer:
        record_iter = tqdm(
            batched_reader(in_path),
            desc=os.path.basename(in_path),
            position=position,
            total=num_batches)
        for record in record_iter:
            xs, rs = preprocessing.batch_parse_tf_example(len(record), record)
            # Undo cast in batch_parse_tf_example.
            xs = tf.cast(xs, tf.uint8)

            # map the rotation function.
            x_rot, r_rot = preprocessing._random_rotation(xs, rs)

            with tf.Session() as sess:
                x_rot, r_rot = sess.run([x_rot, r_rot])
            tf.reset_default_graph()

            pi_rot = r_rot['pi_tensor']
            val_rot = r_rot['value_tensor']
            for r, x, pi, val in zip(record, x_rot, pi_rot, val_rot):
                record_out = preprocessing.make_tf_example(x, pi, val)
                serialized = record_out.SerializeToString()
                writer.write(serialized)
                assert len(r) == len(serialized), (len(r), len(serialized))


def compare(pair):
    position, in_path, out_path = pair
    num_batches = dual_net.EXAMPLES_PER_GENERATION // FLAGS.batch_size + 1
    compare_iter = tqdm(
        zip(batched_reader(in_path), batched_reader(out_path)),
        desc=os.path.basename(in_path),
        position=position,
        total=num_batches)

    count = 0
    equal = 0
    results = {}
    for a, b in compare_iter:
        # a, b are batched records
        xa, ra = preprocessing.batch_parse_tf_example(len(a), a)
        xb, rb = preprocessing.batch_parse_tf_example(len(b), b)
        xa, xb, ra, rb = tf.Session().run([xa, xb, ra, rb])

        # NOTE: This relies on python3 deterministic dictionaries.
        values = [xa] + list(ra.values()) + [xb] + list(rb.values())
        for xa, pa, va, xb, pb, vb in zip(*values):
            count += 1
            assert va == vb
            equal += (xa == xb).all() + (pa == pb).all()
        results['equal'] = "{}/{} = {:.3f}".format(equal, count, equal / count)
        compare_iter.set_postfix(results)


def main(remaining_argv):
    paths = sorted(tf.gfile.ListDirectory(FLAGS.in_dir))
    total = len(paths)
    pairs = []
    for i, path in enumerate(paths):
      ext = '.tfrecord.zz'
      out_path = path.replace(ext, '_rot' + ext)
      pairs.append((
          -total + i,
          os.path.join(FLAGS.in_dir, path),
          os.path.join(FLAGS.out_dir, out_path)))

    with mp.Pool(FLAGS.threads) as p:
        # NOTE: this keeps tqdm progress bars visible.
        print ("\n" * (total+1))
        list(tqdm(p.imap(convert, pairs), desc="converting", total=total))

        if FLAGS.compare:
            print ("\n" * (total+1))
            list(tqdm(p.imap(compare, pairs), desc="comparing", total=total))


if __name__ == "__main__":
    app.run(main)
