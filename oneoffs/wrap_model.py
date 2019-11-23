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
Wraps an existing model with the Minigo format.

Usage:
  python3 oneoffs/wrap_model.py \
      --src_path "$SRC_PATH" \
      --dst_path "$DST_PATH" \
      <metadata>

Where <metadata> is a list of key-value pairs to be written as metadata in the
destination file, formatted as key=value (i.e. without a leading `--`).

For example to wrap a TensorFlow GPU/CPU GraphDef that uses AlphaGo Zero input
features:
  python3 oneoffs/wrap_model.py \
      --src_path model.pb \
      --dst_path model.minigo \
      --metadata=engine=tf,input_features=agz,input_layout=nhwc

TPU models should use engine=tpu.

The following metadata is required:
  engine: 'tf', 'lite', 'tpu'.
  input_features: 'agz', 'mlperf07', etc.
  input_layout: 'nhwc' or 'nchw'.
  board_size: 9 or 19

Any other metadata can be added as desired.
"""

import sys
sys.path.insert(0, '.')  # nopep8

# Hide the GPUs from TF. This makes startup 2x quicker on some machines.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # nopep8

from absl import app, flags

import json
import tensorflow as tf

import minigo_model

flags.DEFINE_string('src_path', None, 'Source model path.')
flags.DEFINE_string('dst_path', None, 'Destination model path.')
flags.DEFINE_list('metadata', None, 'Comma-separated list of metadata.')

FLAGS = flags.FLAGS


def main(argv):
    # Parse the model metadata from non-flag command line arguments.
    metadata = {}
    for m in FLAGS.metadata:
        key, value = m.split('=', 1)
        # Attempt to coerce each value to a numeric type.
        for t in [int, float]:
            try:
                value = t(value)
                break
            except:
                pass
        metadata[key] = value

    print('metadata: %s' % json.dumps(metadata, sort_keys=True, indent=2))
    for m in ['engine', 'input_features', 'input_layout', 'board_size']:
        assert m in metadata, 'Missing required metadata: "%s"' % m
    assert metadata['input_layout'] in ['nhwc', 'nchw']

    with tf.io.gfile.GFile(FLAGS.src_path, 'rb') as f:
        model_bytes = f.read()
    minigo_model.write_model_bytes(model_bytes, metadata, FLAGS.dst_path)


if __name__ == "__main__":
    app.run(main)
