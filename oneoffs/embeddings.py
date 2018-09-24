#!/usr/bin/env python3

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
sys.path.insert(0, '.')

import os
import pickle

from absl import app, flags
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import dual_net
import features as features_lib
import sgf_wrapper

flags.DEFINE_string('sgf_root', None, 'root directory for eval games')
flags.DEFINE_string('embedding_file', None, 'Where to save the embeddings.')

flags.DEFINE_string('model', 'saved_models/000721-eagle', 'Minigo Model')

flags.DEFINE_integer('first', 20, 'first move in game to consider')
flags.DEFINE_integer('last', 150, 'last move in game to consider')
flags.DEFINE_integer('every', 10, 'choose every X position from game')

flags.DEFINE_integer('embedding_size', 361, 'Size of embedding')

flags.mark_flags_as_required(['sgf_root', 'embedding_file'])

flags.register_validator(
    'sgf_root',
    lambda root: os.path.isdir(root),
    'sgf_root must be an existing directory')

FLAGS = flags.FLAGS


def get_files():
  files = []
  for d in os.listdir(FLAGS.sgf_root):
    for f in os.listdir(os.path.join(FLAGS.sgf_root, d))[:2000]:
        if f.endswith('.sgf'):
            files.append(os.path.join(FLAGS.sgf_root, d, f))
  return files


def main(argv):
    features, labels = dual_net.get_inference_input()
    tf_tensors = dual_net.model_inference_fn(features, False)
    if len(tf_tensors) != 4:
        print("oneoffs/embeddings.py requires you modify")
        print("dual_net.model_inference_fn and add a fourth param")
        sys.exit(1)

    p_out, v_out, logits, shared = tf_tensors
    predictions = { 'shared': shared }

    sess = tf.Session()
    tf.train.Saver().restore(sess, FLAGS.model)

    try:
      progress = tqdm(get_files())
      embeddings = np.empty([len(progress), FLAGS.embedding_size])
      metadata = []
      for i, f in enumerate(progress):
        short_f = os.path.basename(f)
        progress.set_description('Processing %s' % short_f)

        processed = []
        for idx, p in enumerate(sgf_wrapper.replay_sgf_file(f)):
          if idx < FLAGS.first: continue
          if idx > FLAGS.last: break
          if idx % FLAGS.every != 0: continue

          processed.append(features_lib.extract_features(p.position))
          metadata.append((f, idx))

        if len(processed) > 0:
          # If len(processed) gets too large may have to chunk.
          res = sess.run(predictions, feed_dict={features: processed})
          for r in res['shared']:
            assert np.size(r) == FLAGS.embedding_size, np.size(r)
            embeddings[i] = r.flatten()
    except:
      # Raise shows us the error but only after the finally block executes.
      raise
    finally:
      with open(FLAGS.embedding_file, 'wb') as pickle_file:
        pickle.dump([metadata, embeddings], pickle_file)

if __name__ == "__main__":
    app.run(main)
