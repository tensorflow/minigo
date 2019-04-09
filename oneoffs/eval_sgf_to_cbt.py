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

"""Write Minigo eval_game records to Bigtable.

This is used to backfill eval games from before they were written by
cc-evaluator as part of https://github.com/tensorflow/minigo/pull/709

Usage:
python3 oneoffs/eval_sgf_to_cbt.py \
    --cbt_project "$PROJECT" \
    --cbt_instance "$CBT_INSTANCE" \
    --cbt_table    "$CBT_EVAL_TABLE" \
    --sgf_glob     "gs://<path>/eval/*/*.sgf"
"""

import sys
sys.path.insert(0, '.')

import itertools
import multiprocessing
import os
import re
from collections import Counter

from absl import app, flags
from google.cloud import bigtable
from google.cloud.bigtable import row_filters
from tqdm import tqdm
from tensorflow import gfile

import sgf_wrapper
from bigtable_input import METADATA, TABLE_STATE


flags.DEFINE_string(
    'sgf_glob', None,
    'Glob for SGFs to backfill into eval_games bigtable.')

flags.mark_flags_as_required([
    'sgf_glob', 'cbt_project', 'cbt_instance', 'cbt_table'
])

FLAGS = flags.FLAGS

# Constants

EVAL_PREFIX = 'e_{:0>10}'
EVAL_GAME_COUNTER = b'eval_game_counter'
SGF_FILENAME = b'sgf'

#### Common Filters

EVAL_COUNT_FILTER = row_filters.ColumnRangeFilter(
    METADATA, EVAL_GAME_COUNTER, EVAL_GAME_COUNTER)

#### START ####


def grouper(iterable, n):
    iterator = iter(iterable)
    group = tuple(itertools.islice(iterator, n))
    while group:
        yield group
        group = tuple(itertools.islice(iterator, n))


def latest_game_number(bt_table):
    """Return the number of the last game to be written."""
    # TODO(amj): Update documentation on latest_game_number (last game or next game)?
    table_state = bt_table.read_row(TABLE_STATE, filter_=EVAL_COUNT_FILTER)
    if table_state is None:
        return 0

    value = table_state.cell_value(METADATA, EVAL_GAME_COUNTER)
    # see bigtable_input.py cbt_intvalue(...)
    return int.from_bytes(value, byteorder='big')


def read_existing_paths(bt_table):
    """Return the SGF filename for each existing eval record."""
    rows = bt_table.read_rows(
        filter_=row_filters.ColumnRangeFilter(
            METADATA, SGF_FILENAME, SGF_FILENAME))
    reader = tqdm(rows, desc="eval_game", unit=" rows")
    names = (row.cell_value(METADATA, SGF_FILENAME).decode() for row in reader)
    processed = [os.path.splitext(os.path.basename(r))[0] for r in names]
    return processed


def canonical_name(sgf_name):
    """Keep filename and some date folders"""
    sgf_name = os.path.normpath(sgf_name)
    assert sgf_name.endswith('.sgf'), sgf_name
    # Strip off '.sgf'
    sgf_name = sgf_name[:-4]

    # Often eval is inside a folder with the run name.
    # include from folder before /eval/ if part of path.
    with_folder = re.search(r'(?:^|/)([^/]*/eval/.*)', sgf_name)
    if with_folder:
        return with_folder.group(1)

    # Return the filename
    return os.path.basename(sgf_name)


def process_game(path):
    with open(path) as f:
        sgf_contents = f.read()

    root_node = sgf_wrapper.get_sgf_root_node(sgf_contents)
    assert root_node.properties['FF'] == ['4'], ("Bad game record", path)

    result = root_node.properties['RE'][0]
    assert result.lower()[0] in 'bw', result
    assert result.lower()[1] == '+', result
    black_won = result.lower()[0] == 'b'

    length = 0
    node = root_node.next
    while node:
        props = node.properties
        length += 1 if props.get('B') or props.get('W') else 0
        node = node.next

    sgf_path = canonical_name(path)

    return (
        (b"black", root_node.properties['PB'][0]),
        (b"white", root_node.properties['PW'][0]),
        # All values are strings, "1" for true and "0" for false here
        (b"black_won", '1' if black_won else '0'),
        (b"white_won", '0' if black_won else '1'),
        (b"result", result),
        (b"length", str(length)),
        (b"sgf", sgf_path),
        (b"tag", ""),
        (b"tool", "eval_sgf_to_cbt"),
    )


def read_games(glob, existing_paths):
    """Read all SGFs that match glob

    Parse each game and extract relevant metadata for eval games table.
    """

    globbed = sorted(gfile.Glob(glob))

    skipped = 0
    to_parse = []
    for sgf_name in tqdm(globbed):
        assert sgf_name.lower().endswith('.sgf'), sgf_name
        sgf_path = canonical_name(sgf_name)
        sgf_filename = os.path.basename(sgf_path)

        if sgf_path in existing_paths or sgf_filename in existing_paths:
            skipped += 1
            continue

        to_parse.append(sgf_name)

    game_data = []
    with multiprocessing.Pool() as pool:
        game_data = pool.map(process_game, tqdm(to_parse), 100)

    print("Read {} SGFs, {} new, {} existing".format(
        len(globbed), len(game_data), skipped))
    return game_data


def write_eval_records(bt_table, game_data, last_game):
    """Write all eval_records to eval_table

    In addition to writing new rows table_state must be updated in
    row `table_state` columns `metadata:eval_game_counter`

    Args:
      bt_table: bigtable table to add rows to.
      game_data:  metadata pairs (column name, value) for each eval record.
      last_game:  last_game in metadata:table_state
    """
    eval_num = last_game

    # Each column counts as a mutation so max rows is ~10000
    GAMES_PER_COMMIT = 2000
    for games in grouper(tqdm(game_data), GAMES_PER_COMMIT):
        assert bt_table.read_row(EVAL_PREFIX.format(eval_num)), "Prev row doesn't exists"
        assert bt_table.read_row(EVAL_PREFIX.format(eval_num+1)) is None, "Row already exists"

        rows = []
        for i, metadata in enumerate(games):
            eval_num += 1
            row_name = EVAL_PREFIX.format(eval_num)
            row = bt_table.row(row_name)
            for column, value in metadata:
                row.set_cell(METADATA, column, value)
            rows.append(row)
            # For each batch of games print a couple of the rows being added.
            if i < 5 or i + 5 > len(games):
                print("\t", i, row_name, metadata[6][1])

        if eval_num == last_game + len(games):
            test = input("Commit ('y'/'yes' required): ")
            if test.lower() not in ('y', 'yes'):
                break

        # TODO(derek): Figure out how to condition on atomic counter update.
        # Condition all updates on the current value of last_game

        game_num_update = bt_table.row(TABLE_STATE)
        game_num_update.set_cell(METADATA, EVAL_GAME_COUNTER, eval_num)
        print(TABLE_STATE, eval_num)

        response = bt_table.mutate_rows(rows)

        # validate that all rows written successfully
        any_bad = False
        for i, status in enumerate(response):
            if status.code is not 0:
                print("Row number {} failed to write {}".format(i, status))
                any_bad = True
        if any_bad:
            break

        game_num_update.commit()


def main(unusedargv):
    """All of the magic together."""
    del unusedargv

    bt_table = (bigtable
                .Client(FLAGS.cbt_project, admin=True)
                .instance(FLAGS.cbt_instance)
                .table(FLAGS.cbt_table))
    assert bt_table.exists(), "Table doesn't exist"

    # Get current game counter, updates are conditioned on this matching.
    last_game = latest_game_number(bt_table)
    print("eval_game_counter:", last_game)
    print()

    # Get existing SGF paths so we avoid uploading duplicates
    existing_paths = read_existing_paths(bt_table)
    print("Found {} existing".format(len(existing_paths)))
    if existing_paths:
        duplicates = Counter(existing_paths)
        existing_paths = set(existing_paths)

        for k, v in duplicates.most_common():
            if v == 1:
                break
            print("{}x{}".format(v, k))

        print("\tmin:", min(existing_paths))
        print("\tmax:", max(existing_paths))
        print()

    # Get all SGFs that match glob, skipping SGFs with existing records.
    data = read_games(FLAGS.sgf_glob, existing_paths)
    if data:
        write_eval_records(bt_table, data, last_game)


if __name__ == "__main__":
    app.run(main)
