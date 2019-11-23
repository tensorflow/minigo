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
Unwraps a Minigo format model to a file and prints its metadata to stdout.

Models must be unwrapped if you want to load them outside the Minigo engine.

Usage:
  python3 oneoffs/unwrap_model.py \
      --src_path "$SRC_PATH" \
      --dst_path "$DST_PATH"
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
flags.DEFINE_string(
    'dst_path', None,
    'Optional destination model path to write the unwrapped model to.')

FLAGS = flags.FLAGS


def main(argv):
    metadata, model_bytes = minigo_model.read_model(FLAGS.src_path)
    print('metadata: %s' % json.dumps(metadata, sort_keys=True, indent=2))

    if FLAGS.dst_path:
        with tf.io.gfile.GFile(FLAGS.dst_path, 'wb') as f:
            f.write(model_bytes)


if __name__ == "__main__":
    app.run(main)
