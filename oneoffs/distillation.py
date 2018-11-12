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

"""
Used to create distillation training examples.

Distillation is training on the value from a teacher network (z=0.7)
instead of the supervised labels (z=1). In theory this can help because
the values are expressable by some network.

This script replaces the policy and game results in a file of TFExamples
(-1 or 1) with a model's evaluation of that position.

Usage:
BOARD_SIZE=19 python3 oneoffs/distillation.py \
    --model data/000721-eagle \
    --in_path data/300.tfrecord.zz \
    --out_path data/300.dist.tfrecord.zz \
    --batch_size=32
"""

import sys
sys.path.insert(0, '.')

import itertools
import os.path

import tensorflow as tf
import matplotlib.pyplot as plt
from absl import app, flags
from tqdm import tqdm

import preprocessing
import dual_net

flags.DEFINE_integer("batch_size", 64, "batch_size for rotating.")
flags.DEFINE_string("in_path", None, "tfrecord.zz to distill.")
flags.DEFINE_string("out_path", None, "Records are writen to this file.")
flags.DEFINE_string('model', 'saved_models/000721-eagle', 'Minigo Model')

FLAGS = flags.FLAGS
OPTS = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)


def test():
    records = list(tqdm(tf.python_io.tf_record_iterator(FLAGS.out_path, OPTS)))
    print()
    print("Values:")
    for i, r in enumerate(records[:10]):
        e = tf.train.Example.FromString(r)
        print("\t", i, e.features.feature['outcome'].float_list.value)


def main(unused_argv):
    in_path = FLAGS.in_path
    out_path = FLAGS.out_path

    assert tf.gfile.Exists(in_path)
    # TODO(amj): Why does ensure_dir_exists skip gs paths?
    #tf.gfile.MakeDirs(os.path.dirname(out_path))
    #assert tf.gfile.Exists(os.path.dirname(out_path))

    policy_err = []
    value_err = []

    print()
    with tf.python_io.TFRecordWriter(out_path, OPTS) as writer:
        ds_iter = preprocessing.get_input_tensors(
            FLAGS.batch_size,
            [in_path],
            shuffle_examples=False,
            random_rotation=False,
            filter_amount=1.0)

        with tf.Session() as sess:
            features, labels = ds_iter
            p_in = labels['pi_tensor']
            v_in = labels['value_tensor']

            p_out, v_out, logits = dual_net.model_inference_fn(
                features, False, FLAGS.flag_values_dict())
            tf.train.Saver().restore(sess, FLAGS.model)

            # TODO(seth): Add policy entropy.

            p_err = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits,
                labels=tf.stop_gradient(p_in))
            v_err = tf.square(v_out - v_in)

            for _ in tqdm(itertools.count(1)):
                try:
                    # Undo cast in batch_parse_tf_example.
                    x_in = tf.cast(features, tf.int8)

                    x, pi, val, pi_err, val_err = sess.run(
                        [x_in, p_out, v_out, p_err, v_err])

                    for i, (x_i, pi_i, val_i) in enumerate(zip(x, pi, val)):
                        # NOTE: The teacher's policy has much higher entropy
                        # Than the Self-play policy labels which are mostly 0
                        # expect that resulting file is 3-5x larger.

                        r = preprocessing.make_tf_example(x_i, pi_i, val_i)
                        serialized = r.SerializeToString()
                        writer.write(serialized)

                    policy_err.extend(pi_err)
                    value_err.extend(val_err)

                except tf.errors.OutOfRangeError:
                    print()
                    print("Breaking OutOfRangeError")
                    break

    print("Counts", len(policy_err), len(value_err))
    test()

    plt.subplot(121)
    n, bins, patches = plt.hist(policy_err, 40)
    plt.title('Policy Error histogram')

    plt.subplot(122)
    n, bins, patches = plt.hist(value_err, 40)
    plt.title('Value Error')

    plt.show()


if __name__ == "__main__":
    app.run(main)
