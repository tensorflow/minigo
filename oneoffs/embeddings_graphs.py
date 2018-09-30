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

import os
import pickle
import subprocess
import time

from absl import app, flags
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

flags.DEFINE_string('embedding_file', None, 'Where to save the embeddings.')

flags.DEFINE_integer(
    'pca_dims', None,
    help='None to skip PCA, else number of dimensions to reduce to')

flags.DEFINE_bool(
    'produce_pngs', False,
    help='if true call sgftopng for all positions')

flags.mark_flag_as_required('embedding_file')

flags.register_validator(
    'embedding_file',
    lambda ef: ef.endswith('.pickle') and os.path.isfile(ef),
    'embedding_file must be an existing .pickle file')

FLAGS = flags.FLAGS


def main(argv):
    t0 = time.time()

    embedding_file = FLAGS.embedding_file
    with open(embedding_file, 'rb') as pickle_file:
      metadata, embeddings = pickle.load(pickle_file)

    t1 = time.time()

    reduced = embeddings
    if FLAGS.pca_dims:
        pca = PCA(n_components=FLAGS.pca_dims)
        pca_result = pca.fit_transform(embeddings)

        print('Explained variation per principal component:')
        print(pca.explained_variance_ratio_)
        print('Total Explained: {:.4f}'.format(
            sum(pca.explained_variance_ratio_)))
        print()
        reduced = pca_result

    t2 = time.time()

    #reduced = reduced[:150]

    print('Shape:', reduced.shape)
    tsne = TSNE(
        n_components=2,
        verbose=4,
        perplexity=40,
        n_iter=2000,
        min_grad_norm=5e-5)
    coords = tsne.fit_transform(reduced)

    assert len(coords.shape) == 2, coords.shape[1] == 2

    # scale coords to be [0,1] in both dims
    coords -= [min(coords[:,0]), min(coords[:,1])]
    coords /= max(coords.flatten())

    t3 = time.time()

    for i, (path, move) in enumerate(tqdm(metadata)):
        assert path.endswith('.sgf'), path
        png = '{}_{}.png'.format(path[:-4], move)
        assert '/eval/' in png, png
        png = png.replace('/eval/', '/thumbnails/')
        if FLAGS.produce_pngs and not os.path.exists(png):
            # NOTE: sgftopng is a pain to install, sorry.
            with open(path) as sgf_file:
                subprocess.run(
                    ['sgftopng', png, '-'+str(move+1)],
                    stdin=sgf_file)
        metadata[i] = (path, move, png)

    t4 = time.time()

    print('Read {:.2f}s, PCA {:.2f}s t-SNE {:.2f}s, PNGs {:.2f}s'.format(
        t1 - t0, t2 - t1, t3 - t2, t4 - t3))

    new_file = embedding_file.replace('.pickle', '.graph.pickle')
    assert new_file != embedding_file, (new_file, embedding_file)
    with open(new_file, 'wb') as pickle_file:
        pickle.dump([metadata, embeddings, coords], pickle_file)

    print('TSNE cords added to', new_file)



if __name__ == '__main__':
    app.run(main)
