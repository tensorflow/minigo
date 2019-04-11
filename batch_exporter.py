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

"""Copy Minigo training sets from table to GCS..
"""


import bisect
import math
import multiprocessing
import os
import tensorflow as tf
from absl import flags
from absl import app
from tqdm import tqdm
import bigtable_input
import utils


flags.DEFINE_bool('dry_run', False,
                  'If true, generate and print the windows, rather than export.')

flags.DEFINE_integer('starting_game', None,
                     'Export beginning with the window that follows this regular game')

flags.DEFINE_integer('training_games', 500000,
                     'Number of games to include in training window')

flags.DEFINE_integer('training_moves', 2**21,
                     'Number of moves to select from training window')

flags.DEFINE_float('training_fresh', 0.05,
                   'Fraction of fresh games in each new training window')

flags.DEFINE_integer('batch_size', 1024,
                     'How many TFRecords to pull through tf.Session at a time')

flags.DEFINE_string('output_prefix', 'gs://dtj-minigo-us-central1/tryit_',
                    'Name of output file to receive TFRecords')

flags.DEFINE_integer('concurrency', 4,
                     'Number of parallel subprocesses')

flags.DEFINE_integer('max_trainings', None,
                     'Process no more than this many training brackets')

FLAGS = flags.FLAGS


def training_series(cursor_r, cursor_c, mix, increment_fraction=0.05):
    """Given two end-cursors and a mix of games, produce a series of bounds.
    """
    while (cursor_r - mix.games_r) >= 0 and (cursor_c - mix.games_c) >= 0:
        yield (cursor_r - mix.games_r), cursor_r, (cursor_c - mix.games_c), cursor_c
        cursor_r -= math.ceil(mix.games_r * increment_fraction)
        cursor_c -= math.ceil(mix.games_c * increment_fraction)


def _export_training_set(args):
    spec, start_r, start_c, mix, batch_size, output_url = args
    gq_r = bigtable_input.GameQueue(spec.project, spec.instance, spec.table)
    gq_c = bigtable_input.GameQueue(spec.project, spec.instance, spec.table + '-nr')
    total_moves = mix.moves_r + mix.moves_c

    with tf.Session() as sess:
        ds = bigtable_input.get_unparsed_moves_from_games(gq_r, gq_c,
                                                          start_r, start_c,
                                                          mix)
        ds = ds.batch(batch_size)
        iterator = ds.make_initializable_iterator()
        sess.run(iterator.initializer)
        get_next = iterator.get_next()
        writes = 0
        print('Writing to', output_url)
        with tf.io.TFRecordWriter(
                output_url,
                options=tf.io.TFRecordCompressionType.ZLIB) as wr:
            log_filename = '/tmp/{}_{}.log'.format(start_r, start_c)
            with open(log_filename, 'w') as progress_file:
                with tqdm(desc='Records', unit_scale=2, total=total_moves,
                          file=progress_file) as pbar:
                    while True:
                        try:
                            batch = sess.run(get_next)
                            pbar.update(len(batch))
                            for b in batch:
                                wr.write(b)
                            writes += 1
                            if (writes % 10000) == 0:
                                wr.flush()
                        except tf.errors.OutOfRangeError:
                            break
            os.unlink(log_filename)


def main(argv):
    """Main program.
    """
    del argv  # Unused
    total_games = FLAGS.training_games
    total_moves = FLAGS.training_moves
    fresh = FLAGS.training_fresh
    batch_size = FLAGS.batch_size
    output_prefix = FLAGS.output_prefix

    spec = bigtable_input.BigtableSpec(
        FLAGS.cbt_project,
        FLAGS.cbt_instance,
        FLAGS.cbt_table)
    gq_r = bigtable_input.GameQueue(spec.project, spec.instance, spec.table)
    gq_c = bigtable_input.GameQueue(spec.project, spec.instance, spec.table + '-nr')

    mix = bigtable_input.mix_by_decile(total_games, total_moves, 9)
    trainings = [(spec, start_r, start_c,
                  mix, batch_size,
                  '{}{:0>10}_{:0>10}.tfrecord.zz'.format(output_prefix, start_r, start_c))
                 for start_r, finish_r, start_c, finish_c
                 in reversed(list(training_series(gq_r.latest_game_number,
                                                  gq_c.latest_game_number,
                                                  mix,
                                                  fresh)))]

    if FLAGS.starting_game:
        game = FLAGS.starting_game
        starts = [t[1] for t in trainings]
        where = bisect.bisect_left(starts, game)
        trainings = trainings[where:]

    if FLAGS.max_trainings:
        trainings = trainings[:FLAGS.max_trainings]

    # TODO:  have a --dry_run to review
    if FLAGS.dry_run:
        for t in trainings:
            print(t)
        raise SystemExit

    concurrency = min(FLAGS.concurrency, multiprocessing.cpu_count() * 2)
    with tqdm(desc='Training Sets', unit_scale=2, total=len(trainings)) as pbar:
        for b in utils.iter_chunks(concurrency, trainings):
            with multiprocessing.Pool(processes=concurrency) as pool:
                pool.map(_export_training_set, b)
                pbar.update(len(b))


if __name__ == '__main__':
    app.run(main)
