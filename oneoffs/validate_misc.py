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

# A collection of misc scripts to validate various defensive programming.

import tensorflow as tf
import numpy as np
from tqdm import tqdm


def validate_examples(example_file):
    """Validate that examples are well formed.

    Pi should sum to 1.0
    value should be {-1,1}

    Usage:
        validate_examples("../data/300.tfrecord.zz")
    """

    def test_example(raw):
        example = tf.train.Example()
        example.ParseFromString(raw)

        pi = np.frombuffer(example.features.feature['pi'].bytes_list.value[0], np.float32)
        value = example.features.feature['outcome'].float_list.value[0]
        assert abs(pi.sum() - 1) < 1e-4, pi.sum()
        assert value in (-1, 1), value

    opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    for record in tqdm(tf.python_io.tf_record_iterator(example_file, opts)):
        test_example(record)

