import sys
sys.path.insert(0, '.')

import tensorflow as tf
import go
import numpy as np
import features as features_lib

tf.app.flags.DEFINE_string("a", "", "Path to first example")
tf.app.flags.DEFINE_string("b", "", "Path to second example")

FLAGS = tf.app.flags.FLAGS

TF_RECORD_CONFIG = tf.python_io.TFRecordOptions(
    tf.python_io.TFRecordCompressionType.ZLIB)


class ParsedExample(object):
    def __init__(self, features, pi, value):
        self.features = features
        self.pi = pi
        self.value = value


def ReadExamples(path):
    print("Reading", path)

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
        x = tf.reshape(x, [go.N, go.N, features_lib.NEW_FEATURES_PLANES])

        pi = tf.decode_raw(parsed['pi'], tf.float32)
        pi = tf.reshape(pi, [go.N * go.N + 1])

        outcome = parsed['outcome']
        assert outcome.shape == (1,)

        result.append(ParsedExample(x.eval(), pi.eval(), outcome.eval()))
    return result


def main(unused_argv):
    with tf.Session() as _:
        examples_a = ReadExamples(FLAGS.a)
        examples_b = ReadExamples(FLAGS.b)
    print(len(examples_a), len(examples_b))

    assert len(examples_a) == len(examples_b)
    for i, (a, b) in enumerate(zip(examples_a, examples_b)):
        assert a.value == b.value
        np.testing.assert_array_equal(a.features, b.features)
        np.testing.assert_array_almost_equal(a.pi, b.pi, decimal=4)


if __name__ == "__main__":
    tf.app.run(main)
