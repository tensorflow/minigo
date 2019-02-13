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

"""Read Minigo game examples from a Bigtable.
"""

import bisect
import collections
import datetime
import math
import multiprocessing
import operator
import re
import struct
import time
import numpy as np

from tqdm import tqdm
from absl import flags
from google.cloud import bigtable
from google.cloud.bigtable import row_filters as bigtable_row_filters
from google.cloud.bigtable import column_family as bigtable_column_family
import tensorflow as tf

import utils


flags.DEFINE_string('cbt_project', None,
                    'The project used to connect to the cloud bigtable ')

# cbt_instance:  identifier of Cloud Bigtable instance in cbt_project.
flags.DEFINE_string('cbt_instance', None,
                    'The identifier of the cloud bigtable instance in cbt_project')

# cbt_table:  identifier of Cloud Bigtable table in cbt_instance.
# The cbt_table is expected to be accompanied by one with an "-nr"
# suffix, for "no-resign".
flags.DEFINE_string('cbt_table', None,
                    'The table within the cloud bigtable instance to use')

FLAGS = flags.FLAGS


# Constants

ROW_PREFIX = 'g_{:0>10}_'
ROWCOUNT_PREFIX = 'ct_{:0>10}_'

# Maximum number of concurrent processes to use when issuing requests against
# Bigtable.  Value taken from default in the load-testing tool described here:
#
# https://github.com/googleapis/google-cloud-go/blob/master/bigtable/cmd/loadtest/loadtest.go

MAX_BT_CONCURRENCY = 100

# Column family and qualifier constants.

# Column Families

METADATA = 'metadata'
TFEXAMPLE = 'tfexample'

# Column Qualifiers

# Note that in CBT, families are strings and qualifiers are bytes.

TABLE_STATE = b'table_state'
WAIT_CELL = b'wait_for_game_number'
GAME_COUNTER = b'game_counter'
MOVE_COUNT = b'move_count'

# Patterns

_game_row_key = re.compile(r'g_(\d+)_m_(\d+)')
_game_from_counter = re.compile(r'ct_(\d+)_')


# The string information needed to construct a client of a Bigtable table.
BigtableSpec = collections.namedtuple(
    'BigtableSpec',
    ['project', 'instance', 'table'])


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

    The dataset must have only one dimension, that is,
    the length of its `output_shapes` and `output_types`
    is 1, and its output shape must be `[]`, that is,
    every tensor in the dataset must be a scalar.

    Args:
      ds:  a TF Dataset.
      batch_size:  how many elements to read per pass

    Returns:
      a single numpy array.
    """
    if isinstance(ds.output_types, tuple) or isinstance(ds.output_shapes, tuple):
        raise ValueError('Dataset must have a single type and shape')
    nshapes = len(ds.output_shapes)
    if nshapes > 0:
        raise ValueError('Dataset must be comprised of scalars (TensorShape=[])')
    batches = []
    with tf.Session() as sess:
        ds = ds.batch(batch_size)
        iterator = ds.make_initializable_iterator()
        sess.run(iterator.initializer)
        get_next = iterator.get_next()
        with tqdm(desc='Elements', unit_scale=1) as pbar:
            try:
                while True:
                    batches.append(sess.run(get_next))
                    pbar.update(len(batches[-1]))
            except tf.errors.OutOfRangeError:
                pass
    if batches:
        return np.concatenate(batches)
    return np.array([], dtype=ds.output_types.as_numpy_dtype)


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


def _game_keys_as_array(ds):
    """Turn keys of a Bigtable dataset into an array.

    Take g_GGG_m_MMM and create GGG.MMM numbers.

    Valuable when visualizing the distribution of a given dataset in
    the game keyspace.
    """
    ds = ds.map(lambda row_key, cell: row_key)
    # want 'g_0000001234_m_133' is '0000001234.133' and so forth
    ds = ds.map(lambda x:
                tf.strings.to_number(tf.strings.substr(x, 2, 10) +
                                     '.' +
                                     tf.strings.substr(x, 15, 3),
                                     out_type=tf.float64))
    return make_single_array(ds)


def _delete_rows(args):
    """Delete the given row keys from the given Bigtable.

    The args are (BigtableSpec, row_keys), but are passed
    as a single argument in order to work with
    multiprocessing.Pool.map.  This is also the reason why this is a
    top-level function instead of a method.

    """
    btspec, row_keys = args
    bt_table = bigtable.Client(btspec.project).instance(
        btspec.instance).table(btspec.table)
    rows = [bt_table.row(k) for k in row_keys]
    for r in rows:
        r.delete()
    bt_table.mutate_rows(rows)
    return row_keys


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
        self.btspec = BigtableSpec(project_name, instance_name, table_name)
        self.bt_table = bigtable.Client(
            self.btspec.project, admin=True).instance(
                self.btspec.instance).table(self.btspec.table)
        self.tf_table = tf.contrib.cloud.BigtableClient(
            self.btspec.project,
            self.btspec.instance).table(self.btspec.table)

    def create(self):
        """Create the table underlying the queue.

        Create the 'metadata' and 'tfexample' column families
        and their properties.
        """
        if self.bt_table.exists():
            utils.dbg('Table already exists')
            return

        max_versions_rule = bigtable_column_family.MaxVersionsGCRule(1)
        self.bt_table.create(column_families={
            METADATA: max_versions_rule,
            TFEXAMPLE: max_versions_rule})

    @property
    def latest_game_number(self):
        """Return the number of the next game to be written."""
        table_state = self.bt_table.read_row(
            TABLE_STATE,
            filter_=bigtable_row_filters.ColumnRangeFilter(
                METADATA, GAME_COUNTER, GAME_COUNTER))
        if table_state is None:
            return 0
        return cbt_intvalue(table_state.cell_value(METADATA, GAME_COUNTER))

    @latest_game_number.setter
    def latest_game_number(self, latest):
        table_state = self.bt_table.row(TABLE_STATE)
        table_state.set_cell(METADATA, GAME_COUNTER, int(latest))
        table_state.commit()

    def games_by_time(self, start_game, end_game):
        """Given a range of games, return the games sorted by time.

        Returns [(time, game_number), ...]

        The time will be a `datetime.datetime` and the game
        number is the integer used as the basis of the row ID.

        Note that when a cluster of self-play nodes are writing
        concurrently, the game numbers may be out of order.
        """
        move_count = b'move_count'
        rows = self.bt_table.read_rows(
            ROWCOUNT_PREFIX.format(start_game),
            ROWCOUNT_PREFIX.format(end_game),
            filter_=bigtable_row_filters.ColumnRangeFilter(
                METADATA, move_count, move_count))

        def parse(r):
            rk = str(r.row_key, 'utf-8')
            game = _game_from_counter.match(rk).groups()[0]
            return (r.cells[METADATA][move_count][0].timestamp, game)
        return sorted([parse(r) for r in rows], key=operator.itemgetter(0))

    def delete_row_range(self, format_str, start_game, end_game):
        """Delete rows related to the given game range.

        Args:
          format_str:  a string to `.format()` by the game numbers
            in order to create the row prefixes.
          start_game:  the starting game number of the deletion.
          end_game:  the ending game number of the deletion.
        """
        row_keys = make_single_array(
            self.tf_table.keys_by_range_dataset(
                format_str.format(start_game),
                format_str.format(end_game)))
        row_keys = list(row_keys)
        if not row_keys:
            utils.dbg('No rows left for games %d..%d' % (
                start_game, end_game))
            return
        utils.dbg('Deleting %d rows:  %s..%s' % (
            len(row_keys), row_keys[0], row_keys[-1]))

        # Reverse the keys so that the queue is left in a more
        # sensible end state if you change your mind (say, due to a
        # mistake in the timestamp) and abort the process: there will
        # be a bit trimmed from the end, rather than a bit
        # trimmed out of the middle.
        row_keys.reverse()
        total_keys = len(row_keys)
        utils.dbg('Deleting total of %d keys' % total_keys)
        concurrency = min(MAX_BT_CONCURRENCY,
                          multiprocessing.cpu_count() * 2)
        with multiprocessing.Pool(processes=concurrency) as pool:
            batches = []
            with tqdm(desc='Keys', unit_scale=2, total=total_keys) as pbar:
                for b in utils.iter_chunks(bigtable.row.MAX_MUTATIONS,
                                           row_keys):
                    pbar.update(len(b))
                    batches.append((self.btspec, b))
                    if len(batches) >= concurrency:
                        pool.map(_delete_rows, batches)
                        batches = []
                pool.map(_delete_rows, batches)
                batches = []

    def trim_games_since(self, t, max_games=500000):
        """Trim off the games since the given time.

        Search back no more than max_games for this time point, locate
        the game there, and remove all games since that game,
        resetting the latest game counter.

        If `t` is a `datetime.timedelta`, then the target time will be
        found by subtracting that delta from the time of the last
        game.  Otherwise, it will be the target time.
        """
        latest = self.latest_game_number
        earliest = int(latest - max_games)
        gbt = self.games_by_time(earliest, latest)
        if not gbt:
            utils.dbg('No games between %d and %d' % (earliest, latest))
            return
        most_recent = gbt[-1]
        if isinstance(t, datetime.timedelta):
            target = most_recent[0] - t
        else:
            target = t
        i = bisect.bisect_right(gbt, (target,))
        if i >= len(gbt):
            utils.dbg('Last game is already at %s' % gbt[-1][0])
            return
        when, which = gbt[i]
        utils.dbg('Most recent:  %s  %s' % most_recent)
        utils.dbg('     Target:  %s  %s' % (when, which))
        which = int(which)
        self.delete_row_range(ROW_PREFIX, which, latest)
        self.delete_row_range(ROWCOUNT_PREFIX, which, latest)
        self.latest_game_number = which

    def bleakest_moves(self, start_game, end_game):
        """Given a range of games, return the bleakest moves.

        Returns a list of (game, move, q) sorted by q.
        """
        bleak = b'bleakest_q'
        rows = self.bt_table.read_rows(
            ROW_PREFIX.format(start_game),
            ROW_PREFIX.format(end_game),
            filter_=bigtable_row_filters.ColumnRangeFilter(
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
        latest = self.latest_game_number
        table_state = self.bt_table.row(TABLE_STATE)
        table_state.set_cell(METADATA, WAIT_CELL, int(latest + number_fresh))
        table_state.commit()
        print("== Setting wait cell to ", int(latest + number_fresh), flush=True)

    def wait_for_fresh_games(self, poll_interval=15.0):
        """Block caller until required new games have been played.

        Args:
          poll_interval:  number of seconds to wait between checks

        If the cell `table_state=metadata:wait_for_game_number` exists,
        then block the caller, checking every `poll_interval` seconds,
        until `table_state=metadata:game_counter is at least the value
        in that cell.
        """
        wait_until_game = self.read_wait_cell()
        if not wait_until_game:
            return
        latest_game = self.latest_game_number
        last_latest = latest_game
        while latest_game < wait_until_game:
            utils.dbg('Latest game {} not yet at required game {} '
                      '(+{}, {:0.3f} games/sec)'.format(
                          latest_game,
                          wait_until_game,
                          latest_game - last_latest,
                          (latest_game - last_latest) / poll_interval
                      ))
            time.sleep(poll_interval)
            last_latest = latest_game
            latest_game = self.latest_game_number

    def read_wait_cell(self):
        """Read the value of the cell holding the 'wait' value,

        Returns the int value of whatever it has, or None if the cell doesn't
        exist.
        """

        table_state = self.bt_table.read_row(
            TABLE_STATE,
            filter_=bigtable_row_filters.ColumnRangeFilter(
                METADATA, WAIT_CELL, WAIT_CELL))
        if table_state is None:
            utils.dbg('No waiting for new games needed; '
                      'wait_for_game_number column not in table_state')
            return None
        value = table_state.cell_value(METADATA, WAIT_CELL)
        if not value:
            utils.dbg('No waiting for new games needed; '
                      'no value in wait_for_game_number cell '
                      'in table_state')
            return None
        return cbt_intvalue(value)

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
            filter_=bigtable_row_filters.ColumnRangeFilter(
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
        utils.dbg('Row range: %s - %s; total moves: %d; probability %.3f; moves %d' % (
            start_row, end_row, total_moves, probability, moves))
        ds = self.tf_table.parallel_scan_range(start_row, end_row,
                                               probability=probability,
                                               columns=[(column_family, column)])
        if shuffle:
            utils.dbg('Doing a complete shuffle of %d moves' % moves)
            ds = ds.shuffle(moves)
        ds = ds.take(moves)
        return ds

    def moves_from_last_n_games(self, n, moves, shuffle,
                                column_family, column):
        """Randomly choose a given number of moves from the last n games.

        Args:
          n:  number of games at the end of this GameQueue to source.
          moves:  number of moves to be sampled from `n` games.
          shuffle:  if True, shuffle the selected moves.
          column_family:  name of the column family containing move examples.
          column:  name of the column containing move examples.

        Returns:
          a dataset containing the selected moves.
        """
        self.wait_for_fresh_games()
        latest_game = self.latest_game_number
        utils.dbg('Latest game in %s: %s' % (self.btspec.table, latest_game))
        if latest_game == 0:
            raise ValueError('Cannot find a latest game in the table')

        start = int(max(0, latest_game - n))
        ds = self.moves_from_games(start, latest_game, moves, shuffle,
                                   column_family, column)
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


def set_fresh_watermark(game_queue, count_from, window_size,
                        fresh_fraction=0.05, minimum_fresh=20000):
    """Sets the metadata cell used to block until some quantity of games have been played.

    This sets the 'freshness mark' on the `game_queue`, used to block training
    until enough new games have been played.  The number of fresh games required
    is the larger of:
       - The fraction of the total window size
       - The `minimum_fresh` parameter
    The number of games required can be indexed from the 'count_from' parameter.
    Args:
      game_queue: A GameQueue object, on whose backing table will be modified.
      count_from: the index of the game to compute the increment from
      window_size:  an integer indicating how many past games are considered
      fresh_fraction: a float in (0,1] indicating the fraction of games to wait for
      minimum_fresh:  an integer indicating the lower bound on the number of new
      games.
    """
    already_played = game_queue.latest_game_number - count_from
    print("== already_played: ", already_played, flush=True)
    if window_size > count_from:  # How to handle the case when the window is not yet 'full'
        game_queue.require_fresh_games(int(minimum_fresh * .9))
    else:
        num_to_play = max(0, math.ceil(window_size * .9 * fresh_fraction) - already_played)
        print("== Num to play: ", num_to_play, flush=True)
        game_queue.require_fresh_games(num_to_play)


def get_unparsed_moves_from_last_n_games(games, games_nr, n,
                                         moves=2**21,
                                         shuffle=True,
                                         column_family=TFEXAMPLE,
                                         column='example',
                                         values_only=True):
    """Get a dataset of serialized TFExamples from the last N games.

    Args:
      games, games_nr: GameQueues of the regular selfplay and calibration
        (aka 'no resign') games to sample from.
      n:  an integer indicating how many past games should be sourced.
      moves:  an integer indicating how many moves should be sampled
        from those N games.
      column_family:  name of the column family containing move examples.
      column:  name of the column containing move examples.
      shuffle:  if True, shuffle the selected move examples.
      values_only: if True, return only column values, no row keys.

    Returns:
      A dataset containing no more than `moves` examples, sampled
        randomly from the last `n` games in the table.
    """
    # The prefixes and suffixes below have the following meanings:
    #   ct_: count
    #   fr_: fraction
    #    _r: resign (ordinary)
    #   _nr: no-resign
    ct_r, ct_nr = 9, 1
    ct_total = ct_r + ct_nr
    fr_r = ct_r / ct_total
    fr_nr = ct_nr / ct_total
    resign = games.moves_from_last_n_games(
        math.ceil(n * fr_r),
        math.ceil(moves * fr_r),
        shuffle,
        column_family, column)
    no_resign = games_nr.moves_from_last_n_games(
        math.floor(n * fr_nr),
        math.floor(moves * fr_nr),
        shuffle,
        column_family, column)
    selection = np.array([0] * ct_r + [1] * ct_nr, dtype=np.int64)
    choice = tf.data.Dataset.from_tensor_slices(selection).repeat().take(moves)
    ds = tf.contrib.data.choose_from_datasets([resign, no_resign], choice)
    if shuffle:
        ds = ds.shuffle(len(selection) * 2)
    if values_only:
        ds = ds.map(lambda row_name, s: s)
    return ds


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
