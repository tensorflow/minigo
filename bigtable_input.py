"""Read Minigo game examples from a Bigtable.
"""

import collections
import numpy as np
import os
import struct
import time
from google.cloud import bigtable
from google.cloud.bigtable import row_filters
import tensorflow as tf
import utils


# TODO(president.jackson): document these
_project_name = os.environ['PROJECT']
_instance_name = os.environ['CBT_INSTANCE']
_table_name = os.environ['CBT_TABLE']

_bt_table = bigtable.Client(
    _project_name).instance(
        _instance_name).table(
            _table_name)

_tf_table = tf.contrib.cloud.BigtableClient(
    _project_name,
    _instance_name).table(
        _table_name)


def get_latest_game_number():
    """Return the number of the next game to be stored.

    The state of the table is stored in `metadata:game_counter`
    in the `table_state` row only.
    """
    table_state = _bt_table.read_row(
        b'table_state',
        filter_=row_filters.ColumnRangeFilter(
            'metadata', b'game_counter', b'game_counter'))
    value = table_state.cell_value('metadata', b'game_counter')
    return struct.unpack('>q', value)[0]


def get_game_range_row_names(game_begin, game_end):
    """Get the row range containing the given games.

    Sample row name:
      g_0000000001_m001

    Args:
      game_begin:  an integer of the beginning game number.
      game_end:  an integer of the ending game number, exclusive.

    Returns:
      The two string row numbers to pass to Bigtable as the row range.
    """
    row_fmt = 'g_{:0>10}_'
    return row_fmt.format(game_begin), row_fmt.format(game_end)


def make_single_array(ds, batch_size=8*1024):
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


def require_fresh_games(number_fresh):
    """Require a given number of fresh games to be played.

    Updates the cell `table_state=metadata:wait_for_game_number`
    by the given number of games.
    """
    latest = get_latest_game_number()
    wait_cell = b'wait_for_game_number'
    table_state = _bt_table.row(b'table_state')
    table_state.set_cell('metadata', wait_cell, latest + number_fresh)
    table_state.commit()


def wait_for_fresh_games(poll_interval=15.0):
    """Block caller until required new games have been played.

    If the cell `table_state=metadata:wait_for_game_number` exists,
    then block the caller, checking every `poll_interval` seconds,
    until `table_state=metadata:game_counter is at least the value
    in that cell.
    """
    wait_cell = b'wait_for_game_number'
    table_state = _bt_table.read_row(
        b'table_state',
        filter_=row_filters.ColumnRangeFilter(
            'metadata', wait_cell, wait_cell))
    if table_state is None:
        utils.dbg('No waiting for new games needed; '
                  'wait_for_game_number column not in table_state')
        return
    value = table_state.cell_value('metadata', wait_cell)
    if not value:
        utils.dbg('No waiting for new games needed; '
                  'no value in wait_for_game_number cell '
                  'in table_state')
        return
    wait_until_game = struct.unpack('>q', value)[0]
    latest_game = get_latest_game_number()
    while latest_game < wait_until_game:
        utils.dbg('Latest game {} not yet at required game {}'.
                  format(latest_game, wait_until_game))
        time.sleep(poll_interval)
        latest_game = get_latest_game_number()


def get_moves_from_games(start_game, end_game, moves, shuffle,
                         column_family, column):
    start_row, end_row = get_game_range_row_names(start_game, end_game)
    # NOTE:  Choose a probability high enough to guarantee at least the
    # required number of moves, by using a slightly lower estimate
    # of the total moves, then trimming the result.
    total_moves = count_moves_in_game_range(start_game, end_game)
    probability = moves / (total_moves * 0.99)
    utils.dbg('Row range: %s - %s; total moves: %d; probability %.3f' % (
        start_row, end_row, total_moves, probability))
    shards = 8
    ds = _tf_table.parallel_scan_range(start_row, end_row,
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
    wait_for_fresh_games()
    latest_game = int(get_latest_game_number())
    utils.dbg('Latest game: %s' % latest_game)
    if latest_game == 0:
        raise ValueError('Cannot find a latest game in the table')

    start = int(max(0, latest_game - n))
    ds = get_moves_from_games(start, latest_game, moves, shuffle,
                              column_family, column)
    return ds.map(lambda row_name, s: s)


def histogram_move_keys_by_game(sess, ds, batch_size=8*1024):
    """Given dataset of key names, return histogram of moves/game."""
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


def write_move_counts(sess, h):
    """Add move counts from the given histogram to the table.

    Args:
      sess:  TF session to use for doing a Bigtable write.
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
    wr_op = _tf_table.write(mc, column_families=['metadata'], columns=['move_count'])
    sess.run(wr_op)


def update_move_count_in_games(start_game, end_game, interval=1000):
    """Used to update the move_count cell for older games.

    move_count cells will be updated in both g_<game_id>_m_000 rows
    and ct_<game_id>_<move_count> rows.
    """
    for g in range(start_game, end_game, interval):
        with tf.Session() as sess:
            g_range = get_game_range_row_names(g, g + interval)
            print('Range:', g_range)
            start_time = time.time()
            ds = _tf_table.keys_by_range_dataset(*g_range)
            h = histogram_move_keys_by_game(sess, ds)
            write_move_counts(sess, h)
            end_time = time.time()
            elapsed = end_time - start_time
            print('  games/sec:', len(h)/elapsed)


def count_moves_in_game_range(game_begin, game_end):
    """Use the ct_ rows for rapid move summary.
    """
    row_fmt = 'ct_{:0>10}_'
    start_row = row_fmt.format(game_begin)
    end_row = row_fmt.format(game_end)
    rows = _bt_table.read_rows(start_row, end_row,
                               filter_=row_filters.ColumnRangeFilter(
                                   'metadata', b'move_count', b'move_count'))
    moves = sum([int(r.cell_value('metadata', b'move_count')) for r in rows])
    return moves


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
