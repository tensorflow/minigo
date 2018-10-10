"""Read Minigo game examples from a Bigtable.
"""

import os
import struct
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
        'table_state',
        filter_=row_filters.ColumnRangeFilter(
            'metadata', b'game_counter', b'game_counter'))
    value = table_state.cells['metadata'][b'game_counter'][0].value
    return struct.unpack('>q', value)[0]


def get_game_range_row_names(game_begin, game_end):
    """Get the row range containing the given games.

    Sample row name:
      g_0000000001_m001

    To capture the range of all moves in the two given games,
    the end row will need to go up to g_00..(N+1).

    Args:
      game_begin:  an integer of the beginning game number.
      game_end:  an integer of the ending game number, inclusive.

    Returns:
      The two string row numbers to pass to Bigtable as the row range.
    """
    row_fmt = 'g_{:0>10}_'
    return row_fmt.format(game_begin), row_fmt.format(game_end + 1)


def get_unparsed_moves_from_last_n_games(n,
                                         moves=2**21,
                                         column_family='tfexample',
                                         column='example'):
    """Get a dataset of serialized TFExamples from the last N games.

    Args:
      n:  an integer indicating how many past games should be sourced.
      moves:  an integer indicating how many moves should be sampled
        from those N games.
      column_family:  name of the column family containing move examples.
      column:  name of the column containing move examples.

    Returns:
      A dataset containing no more than `moves` examples, sampled
        randomly from the last `n` games in the table.
    """
    latest_game = int(get_latest_game_number())
    utils.dbg('Latest game: %s' % latest_game)
    if latest_game == 0:
        raise ValueError('Cannot find a latest game in the table')

    start = int(max(0, latest_game - n))
    start_row, end_row = get_game_range_row_names(start, latest_game)
    # NOTE:  Choose a probability high enough to guarantee at least the
    # required number of moves, by using a low estimate of the number
    # of moves per game.  Some experiments show that it's right around
    # 250 moves/game on average, so 100 is a conservative lower bound.
    estimate = 100
    probability = moves / (n * estimate)
    utils.dbg('Row range: %s - %s; probability %.3f' % (
        start_row, end_row, probability))
    ds = _tf_table.parallel_scan_range(start_row, end_row,
                                       probability=probability,
                                       columns=[(column_family, column)])
    # Clip the dataset to the desired number of moves, counting on
    # the fact that the probability chosen will produce at least the
    # number of moves required.
    # TODO: Randomize ordering if it isn't already.
    ds = ds.take(moves)
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
