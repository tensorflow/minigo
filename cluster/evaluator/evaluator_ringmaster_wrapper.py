#!/usr/bin/env python
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
evaluator_ringmaster_wrapper: Plays some eval games between two players

players are rows in the cloud bigtable (normally models_for_eval)
each row is a unique pair of (binary, flags) along with name.

The sh script has already:
1. Downloaded both players (binaries and models)

This script is responsible for:
2. Setup ringmaster control file
3. Call ringmaster
4. Record results into CBT
5. upload games, log, report to GCS
"""

import sys
sys.path.insert(0, "..")

import os
import shutil
import subprocess
import time
from collections import namedtuple

from google.cloud import bigtable
from tensorflow import gfile

# Each import must be added to the dockerfile (via Makefile)
import bigtable_output
from bigtable_input import METADATA


CTL_NAME = "ringmaster_evals"
CTL_FILENAME = CTL_NAME + ".ctl"
CTL_GAME_DIR = CTL_NAME + ".games"
CTL_LOG = CTL_NAME + ".log"
CTL_REPORT = CTL_NAME + ".report"

MODEL_ROW_FMT = "m_eval_{}"

CTL_FILE = '''
competition_type = "playoff"
description = " Testing models_for_eval "
board_size = 19
komi = 7.5

record_games = True
stderr_to_log = True

def MinigoPlayer(path, model_pb, flags):
    return Player(
        "bin/bazel-bin/cc/gtp --model='tf,{{}}' {{}}".format(model_pb, flags),
        cwd=path,
        environ={{"LD_LIBRARY_PATH": "bin/cc/tensorflow/lib"}},
        sgf_player_name_from_gtp=False)

p_a = "{m_a.name}_{m_a.hash}"
p_b = "{m_b.name}_{m_b.hash}"
players = {{
    p_a: MinigoPlayer("{m_a.hash}", "{m_a.model_pb}", "{m_a.flags}"),
    p_b: MinigoPlayer("{m_b.hash}", "{m_b.model_pb}", "{m_b.flags}"),
}}

matchups = [
    Matchup(p_a, p_b, id="{m_a.hash}_vs_{m_b.hash}_t_{start_time}",
            alternating=True, number_of_games={num_games})
]
'''

Model = namedtuple("Model", ["hash", "name", "model_pb", "flags"])

def setup_ringmaster(model_a, model_b, start_time, num_games):
    with open(CTL_FILENAME, "w") as ctl_f:
        ctl_f.write(CTL_FILE.format(
            m_a=model_a,
            m_b=model_b,
            start_time=start_time,
            num_games=num_games))


def call_ringmaster(num_games):
    process = subprocess.run(
        ["ringmaster", CTL_FILENAME, "run"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=num_games*20*60)

    # log errors somewhere
    if process.returncode != 0:
        def print_err(*args):
          print(*args, file=stderr)

        print_err("Error ringmaster return=", process.returncode)
        print_err("Stdout:")
        print_err(process.stdout)
        print_err("Stderr:")
        print_err(process.stderr)

    return process.returncode == 0


def copy_to_gcs(src, dst):
    assert gfile.Exists(src), src
    assert not gfile.Exists(dst), dst

    print("Saving to", dst)
    with gfile.GFile(src, "rb") as src_f, gfile.GFile(dst, "wb") as dst_f:
        shutil.copyfileobj(src_f, dst_f)


def record_results(bt_table, sgf_base, num_games, start_time):
    games = os.listdir(CTL_GAME_DIR)
    failed = len(games) != num_games

    if failed:
        # Upload something? log error somewhere?
        assert False, (len(games), num_games)

    # Upload .log and .report along side all .games
    copy_to_gcs(CTL_LOG, os.path.join(sgf_base, CTL_LOG))
    copy_to_gcs(CTL_REPORT, os.path.join(sgf_base, CTL_REPORT))

    rows = []
    for game_fn in games:
        game_path = os.path.join(CTL_GAME_DIR, game_fn)
        copy_to_gcs(game_path, os.path.join(sgf_base, game_fn))

        metadata = bigtable_output.process_game(game_path)
        metadata["sgf"] = game_fn
        metadata["tool"] = "evaluator_ringmaster"

        # game_fn, which contains timestamp and game number, is unique.
        row = bt_table.row(game_fn)
        for column, value in metadata.items():
            row.set_cell(METADATA, column, value)

        rows.append(row)

    response = bt_table.mutate_rows(rows)

    # validate that all rows were written successfully
    all_good = True
    for i, status in enumerate(response):
        if status.code is not 0:
            print("Row number {} failed to write {}".format(i, status))
            all_good = False

    return all_good


def get_cbt_model(bt_table, model_hash):
    model_row = bt_table.read_row(MODEL_ROW_FMT.format(model_hash))
    def get_cell(cell):
        return model_row.cell_value(METADATA, cell.encode()).decode()

    model_flags = get_cell("model_flags")
    model_pb = get_cell("model_path")
    model_name = get_cell("model")
    return Model(
        model_hash,
        model_name,
        os.path.basename(model_pb),
        model_flags.replace("flags: ", ""),
    )


if __name__ == "__main__":
    ENV_VARS = [
        "PROJECT",
        "CBT_INSTANCE",
        "CBT_TABLE",
        "MODEL_A",
        "MODEL_B",
        "SGF_BUCKET_NAME",
    ]
    ENV = {}
    for env_var in ENV_VARS:
        value = os.environ.get(env_var)
        assert value, (env_var, os.environ.keys())
        ENV[env_var] = value

    print("bigtable: ", ENV["PROJECT"], ENV["CBT_INSTANCE"], ENV["CBT_TABLE"])
    bt_table = (bigtable
                .Client(ENV["PROJECT"], admin=True)
                .instance(ENV["CBT_INSTANCE"])
                .table(ENV["CBT_TABLE"]))
    assert bt_table.exists(), "Table doesn't exist"

    m_a_name = ENV["MODEL_A"]
    m_b_name = ENV["MODEL_B"]
    if m_a_name > m_b_name:
        # Sort models so a <= b alphabetically
        m_a_name, m_b_name = m_b_name, m_a_name

    model_a = get_cbt_model(bt_table, m_a_name)
    model_b = get_cbt_model(bt_table, m_b_name)
    start_time = int(time.time())

    print(model_a)
    print(model_b)

    # TODO(amj): Pass from dockerfile.
    num_games = 4
    setup_ringmaster(model_a, model_b, start_time, num_games)

    success = call_ringmaster(num_games)
    assert success

    SGF_BASE = "gs://{}/eval_server/games/{}_vs_{}/{}/".format(
        ENV["SGF_BUCKET_NAME"], m_a_name, m_b_name, start_time)
    print("Saving to", SGF_BASE)
    success = record_results(bt_table, SGF_BASE,
        num_games, start_time=start_time)
    assert success

