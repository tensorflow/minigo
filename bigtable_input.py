"""Read Minigo game examples from a Bigtable.
"""

import collections
import operator
import os
import re
import struct
import time
import numpy as np
from google.cloud import bigtable
from google.cloud.bigtable import row_filters
import tensorflow as tf
import utils


# Constants

ROW_PREFIX = 'g_{:0>10}_'
ROWCOUNT_PREFIX = 'ct_{:0>10}_'

## Column family and qualifier constants.

#### Column Families

METADATA = 'metadata'

#### Column Qualifiers

#### Note that in CBT, families are strings and qualifiers are bytes.

TABLE_STATE = b'table_state'
WAIT_CELL = b'wait_for_game_number'
MOVE_COUNT = b'move_count'

# Patterns

_game_row_key = re.compile(r'g_(\d+)_m_(\d+)')


def cbt_intvalue(value):
    """Decode a big-endian uint64.

    Cloud Bigtable stores integers as big-endian uint64,
    and performs this translation when integers are being
    set.  But when being read, the values need to be
    decoded.
    """
    return int(struct.unpack('>q', value)[0])


def make_single_array(ds, batch_size=8*1024):
    """Create a single numpy array from a dataset.

    Args:
      ds:  a TF Dataset.
      batch_size:  how many elements to read per pass

    Returns:
      a single numpy array.
    """
    batches = []
    with tf.Session() as sess:
        ds = ds.batch(batch_size)
        iterator = ds.make_initializable_iterator()
        sess.run(iterator.initializer)
        get_next = iterator.get_next()
        try:
            while True:
                batches.append(sess.run(get_next))
        except tf.errors.OutOfRangeError:
            pass
    return np.concatenate(batches)


def _histogram_move_keys_by_game(sess, ds, batch_size=8*1024):
    """Given dataset of key names, return histogram of moves/game.

    Move counts are written by the game players, so
    this is mostly useful for repair or backfill.

    Args:
      sess:  TF session
      ds:  TF dataset containing game move keys.
      batch_size:  performance tuning parameter
    """
    ds = ds.batch(batch_size)
    # Turns 'g_0000001234_m_133' into 'g_0000001234'
    ds = ds.map(lambda x: tf.strings.substr(x, 0, 12))
    iterator = ds.make_initializable_iterator()
    sess.run(iterator.initializer)
    get_next = iterator.get_next()
    h = collections.Counter()
    try:
        while True:
            h.update(sess.run(get_next))
    except tf.errors.OutOfRangeError:
        pass
    # NOTE:  Cannot be truly sure the count is right till the end.
    return h


class GameQueue:
    """Queue of games stored in a Cloud Bigtable.

    The state of the table is stored in the `table_state`
    row, which includes the columns `metadata:game_counter`.
    """

    def __init__(self, project_name, instance_name, table_name):
        """Constructor.

        Args:
          project_name:  string name of GCP project having table.
          instance_name:  string name of CBT instance in project.
          table_name:  string name of CBT table in instance.
        """
        self.bt_table = bigtable.Client(
            project_name).instance(
                instance_name).table(
                    table_name)
        self.tf_table = tf.contrib.cloud.BigtableClient(
            project_name,
            instance_name).table(
                table_name)

    def latest_game_number(self):
        """Return the number of the next game to be written."""
        game_counter = b'game_counter'
        table_state = self.bt_table.read_row(
            TABLE_STATE,
            filter_=row_filters.ColumnRangeFilter(
                METADATA, game_counter, game_counter))
        return cbt_intvalue(table_state.cell_value(METADATA, game_counter))

    def bleakest_moves(self, start_game, end_game):
        """Given a range of games, return the bleakest moves.

        Returns a list of (game, move, q) sorted by q.
        """
        bleak = b'bleakest_q'
        rows = self.bt_table.read_rows(
            ROW_PREFIX.format(start_game),
            ROW_PREFIX.format(end_game),
            filter_=row_filters.ColumnRangeFilter(
                METADATA, bleak, bleak))
        def parse(r):
            rk = str(r.row_key, 'utf-8')
            g, m = _game_row_key.match(rk).groups()
            q = r.cell_value(METADATA, bleak)
            return int(g), int(m), float(q)
        return sorted([parse(r) for r in rows], key=operator.itemgetter(2))

    def require_fresh_games(self, number_fresh):
        """Require a given number of fresh games to be played.

        Args:
          number_fresh:  integer, number of new fresh games needed

        Increments the cell `table_state=metadata:wait_for_game_number`
        by the given number of games.  This will cause
        `self.wait_for_fresh_games()` to block until the game
        counter has reached this number.
        """
        latest = self.latest_game_number()
        table_state = self.bt_table.row(TABLE_STATE)
        table_state.set_cell(METADATA, WAIT_CELL, int(latest + number_fresh))
        table_state.commit()

    def wait_for_fresh_games(self, poll_interval=15.0):
        """Block caller until required new games have been played.

        Args:
          poll_interval:  number of seconds to wait between checks

        If the cell `table_state=metadata:wait_for_game_number` exists,
        then block the caller, checking every `poll_interval` seconds,
        until `table_state=metadata:game_counter is at least the value
        in that cell.
        """
        table_state = self.bt_table.read_row(
            TABLE_STATE,
            filter_=row_filters.ColumnRangeFilter(
                METADATA, WAIT_CELL, WAIT_CELL))
        if table_state is None:
            utils.dbg('No waiting for new games needed; '
                      'wait_for_game_number column not in table_state')
            return
        value = table_state.cell_value(METADATA, WAIT_CELL)
        if not value:
            utils.dbg('No waiting for new games needed; '
                      'no value in wait_for_game_number cell '
                      'in table_state')
            return
        wait_until_game = cbt_intvalue(value)
        latest_game = self.latest_game_number()
        while latest_game < wait_until_game:
            utils.dbg('Latest game {} not yet at required game {}'.
                      format(latest_game, wait_until_game))
            time.sleep(poll_interval)
            latest_game = self.latest_game_number()

    def count_moves_in_game_range(self, game_begin, game_end):
        """Count the total moves in a game range.

        Args:
          game_begin:  integer, starting game
          game_end:  integer, ending game

        Uses the `ct_` keyspace for rapid move summary.
        """
        rows = self.bt_table.read_rows(
            ROWCOUNT_PREFIX.format(game_begin),
            ROWCOUNT_PREFIX.format(game_end),
            filter_=row_filters.ColumnRangeFilter(
                METADATA, MOVE_COUNT, MOVE_COUNT))
        return sum([int(r.cell_value(METADATA, MOVE_COUNT)) for r in rows])

    def moves_from_games(self, start_game, end_game, moves, shuffle,
                         column_family, column):
        """Dataset of samples and/or shuffled moves from game range.

        Args:
          n:  an integer indicating how many past games should be sourced.
          moves:  an integer indicating how many moves should be sampled
            from those N games.
          column_family:  name of the column family containing move examples.
          column:  name of the column containing move examples.
          shuffle:  if True, shuffle the selected move examples.

        Returns:
          A dataset containing no more than `moves` examples, sampled
            randomly from the last `n` games in the table.
        """
        start_row = ROW_PREFIX.format(start_game)
        end_row = ROW_PREFIX.format(end_game)
        # NOTE:  Choose a probability high enough to guarantee at least the
        # required number of moves, by using a slightly lower estimate
        # of the total moves, then trimming the result.
        total_moves = self.count_moves_in_game_range(start_game, end_game)
        probability = moves / (total_moves * 0.99)
        utils.dbg('Row range: %s - %s; total moves: %d; probability %.3f' % (
            start_row, end_row, total_moves, probability))
        shards = 8
        ds = self.tf_table.parallel_scan_range(start_row, end_row,
                                               probability=probability,
                                               num_parallel_scans=shards,
                                               columns=[(column_family, column)])
        if shuffle:
            rds = tf.data.Dataset.from_tensor_slices(
                tf.random_shuffle(tf.range(0, shards, dtype=tf.int64)))
            ds = rds.apply(
                tf.contrib.data.parallel_interleave(
                    lambda x: ds.shard(shards, x),
                    cycle_length=shards, block_length=1024))
            ds = ds.shuffle(shards * 1024 * 2)
        ds = ds.take(moves)
        return ds

    def _write_move_counts(self, sess, h):
        """Add move counts from the given histogram to the table.

        Used to update the move counts in an existing table.  Should
        not be needed except for backfill or repair.

        Args:
          sess:  TF session to use for doing a Bigtable write.
          tf_table:  TF Cloud Bigtable to use for writing.
          h:  a dictionary keyed by game row prefix ("g_0023561") whose values
             are the move counts for each game.
        """
        def gen():
            for k, v in h.items():
                # The keys in the histogram may be of type 'bytes'
                k = str(k, 'utf-8')
                vs = str(v)
                yield (k.replace('g_', 'ct_') + '_%d' % v, vs)
                yield (k + '_m_000', vs)
        mc = tf.data.Dataset.from_generator(gen, (tf.string, tf.string))
        wr_op = self.tf_table.write(mc,
                                    column_families=[METADATA],
                                    columns=[MOVE_COUNT])
        sess.run(wr_op)

    def update_move_counts(self, start_game, end_game, interval=1000):
        """Used to update the move_count cell for older games.

        Should not be needed except for backfill or repair.

        move_count cells will be updated in both g_<game_id>_m_000 rows
        and ct_<game_id>_<move_count> rows.
        """
        for g in range(start_game, end_game, interval):
            with tf.Session() as sess:
                start_row = ROW_PREFIX.format(g)
                end_row = ROW_PREFIX.format(g + interval)
                print('Range:', start_row, end_row)
                start_time = time.time()
                ds = self.tf_table.keys_by_range_dataset(start_row, end_row)
                h = _histogram_move_keys_by_game(sess, ds)
                self._write_move_counts(sess, h)
                end_time = time.time()
                elapsed = end_time - start_time
                print('  games/sec:', len(h)/elapsed)


# TODO(president.jackson): document these
_games = GameQueue(os.environ['PROJECT'],
                   os.environ['CBT_INSTANCE'],
                   os.environ['CBT_TABLE'])


def get_unparsed_moves_from_last_n_games(n,
                                         moves=2**21,
                                         shuffle=True,
                                         column_family='tfexample',
                                         column='example'):
    """Get a dataset of serialized TFExamples from the last N games.

    Args:
      n:  an integer indicating how many past games should be sourced.
      moves:  an integer indicating how many moves should be sampled
        from those N games.
      column_family:  name of the column family containing move examples.
      column:  name of the column containing move examples.
      shuffle:  if True, shuffle the selected move examples.

    Returns:
      A dataset containing no more than `moves` examples, sampled
        randomly from the last `n` games in the table.
    """
    _games.wait_for_fresh_games()
    latest_game = int(_games.latest_game_number())
    utils.dbg('Latest game: %s' % latest_game)
    if latest_game == 0:
        raise ValueError('Cannot find a latest game in the table')

    start = int(max(0, latest_game - n))
    ds = _games.moves_from_games(start, latest_game, moves, shuffle,
                                 column_family, column)
    return ds.map(lambda row_name, s: s)


def count_elements_in_dataset(ds, batch_size=1*1024, parallel_batch=8):
    """Count and return all the elements in the given dataset.

    Debugging function.  The elements in a dataset cannot be counted
    without enumerating all of them.  By counting in batch and in
    parallel, this method allows rapid traversal of the dataset.

    Args:
      ds:  The dataset whose elements should be counted.
      batch_size:  the number of elements to count a a time.
      parallel_batch:  how many batches to count in parallel.

    Returns:
      The number of elements in the dataset.
    """
    with tf.Session() as sess:
        dsc = ds.apply(tf.contrib.data.enumerate_dataset())
        dsc = dsc.apply(
            tf.contrib.data.map_and_batch(lambda c, v: c, batch_size,
                                          num_parallel_batches=parallel_batch))
        iterator = dsc.make_initializable_iterator()
        sess.run(iterator.initializer)
        get_next = iterator.get_next()
        counted = 0
        try:
            while True:
                # The numbers in the tensors are 0-based indicies,
                # so add 1 to get the number counted.
                counted = sess.run(tf.reduce_max(get_next)) + 1
                utils.dbg('Counted so far: %d' % counted)
        except tf.errors.OutOfRangeError:
            pass
        utils.dbg('Counted total: %d' % counted)
        return counted
