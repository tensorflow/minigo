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

"""Write Minigo model records to Bigtable.

This is used to insert models data into cbt model table from gcs training dir.

Usage:
python3 oneoffs/cbt_models.py \
    --cbt_project "$PROJECT" \
    --cbt_instance "$CBT_INSTANCE" \
    --cbt_table    "$CBT_MODEL_TABLE" \
    --model_glob "gs://<path>/models/*.meta"
"""

import sys
sys.path.insert(0, ".")

import re

from absl import app, flags
from google.cloud import bigtable
from google.cloud.bigtable import row_filters
from tensorflow import gfile
from tqdm import tqdm

from bigtable_input import METADATA


flags.DEFINE_string(
    "model_glob", None,
    "Glob for .meta model files to backfill into models bigtable.")

flags.mark_flags_as_required([
    "model_glob", "cbt_project", "cbt_instance", "cbt_table"
])

FLAGS = flags.FLAGS

# Constants

# TODO(sethtroisi): Move to cluster or somewhere.
MODEL_ROW = "m_{}_{:0>10}"
MODEL_NAME = b"model"

MODEL_RUN_NAME_RE = re.compile(
    r"^gs://.*(v[1-9][0-9]?)-(19|9)/models/(([0-9]{6})-([a-z-]*)).meta$")

#### START ####


def read_existing_models(bt_table):
    """Return model names from each existing record."""
    # TODO(dtj): Is there a clean way to read just row_keys.
    rows = bt_table.read_rows(filter_=row_filters.ColumnRangeFilter(
        METADATA, MODEL_NAME, MODEL_NAME))
    return [row.row_key.decode() for row in rows]


def parse_model_components(model_path):
    """Return model run, number, full-name

    model_path = "gs://tensor-go-minigo-v16-19/models/000002-medusa.meta"
    returns ("v16", "000002", "000002-medusa")
    """
    match = MODEL_RUN_NAME_RE.match(model_path)
    assert match, model_path
    return match.group(1), match.group(4), match.group(3)


def get_model_data(glob, existing):
    """Read all model meta filenames and extract per model metadata."""

    globbed = sorted(gfile.Glob(glob))

    skipped = 0
    model_data = []
    for model_path in tqdm(globbed):
        assert model_path.lower().endswith(".meta"), model_path
        model_run, model_num, model_name = parse_model_components(model_path)
        row_name = MODEL_ROW.format(model_run, model_name)

        if row_name in existing:
            skipped += 1
            continue

        metadata = (
            (MODEL_NAME, model_name),
            (b"model_num", model_num),
            (b"run", model_run),
            (b"parent", ""),
            (b"tag", ""),
            (b"tool", "cbt_models_backfill_to_cbt"),
            (b"trained_date", ""),
        )
        model_data.append((row_name, metadata))

    print("Read {} Models, {} new, {} existing".format(
        len(globbed), len(model_data), skipped))
    return model_data


def write_records(bt_table, model_data):
    """Write all new models to models table.

    Args:
      bt_table: bigtable table to add rows to.
      game_data:  metadata pairs (column name, value) for each eval record.
      last_game:  last_game in metadata:table_state
    """

    rows = []
    for i, (row_name, metadata) in enumerate(model_data):
        row = bt_table.row(row_name)
        for column, value in metadata:
            row.set_cell(METADATA, column, value)
        rows.append(row)
        # Print a couple of the row name.
        if i < 5 or i + 5 > len(model_data):
            print("\t{}\t{}".format(i, row_name))

    test = input("Commit ('y'/'yes' required): ")
    if test.lower() not in ("y", "yes"):
        return

    response = bt_table.mutate_rows(rows)

    # validate that all rows written successfully
    for i, status in enumerate(response):
        if status.code is not 0:
            print("Row number {} failed to write {}".format(i, status))


def main(unusedargv):
    """Read, dedup, and write models."""
    del unusedargv

    bt_table = (bigtable
                .Client(FLAGS.cbt_project, admin=True)
                .instance(FLAGS.cbt_instance)
                .table(FLAGS.cbt_table))
    assert bt_table.exists(), "Table doesn't exist"

    # Get existing SGF paths so we avoid uploading duplicates
    existing = set(read_existing_models(bt_table))
    print("Found {} existing".format(len(existing)))

    if existing:
        print("\tmin:", min(existing))
        print("\tmax:", max(existing))
        print()

    # Get all SGFs that match glob, skipping SGFs with existing records.
    data = get_model_data(FLAGS.model_glob, existing)
    if data:
        write_records(bt_table, data)


if __name__ == "__main__":
    app.run(main)
