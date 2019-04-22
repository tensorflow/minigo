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

"""
Build local database of eval games using Google Cloud Bigtable as the source of
truth.

Inspiration from Seth's experience with CloudyGo db.
Replacing the much slower and clunkier ratings.py

Usage:

$ sqlite3 cbt_ratings.db < ratings/schema.sql
$ python3 ratings/cbt_ratings.py  \
    --cbt_project "$PROJECT" \
    --cbt_instance "$CBT_INSTANCE"
"""

import sys
sys.path.insert(0, '.')

import math
import random
import re
import sqlite3
from collections import defaultdict, Counter

import choix
import numpy as np
from absl import flags
from tqdm import tqdm
from google.cloud import bigtable

from bigtable_input import METADATA, TABLE_STATE


FLAGS = flags.FLAGS

flags.DEFINE_bool('sync_ratings', False, 'Synchronize files before computing ratings.')

flags.mark_flags_as_required([
    "cbt_project", "cbt_instance"
])

MODEL_REGEX = re.compile(r"(\d*)-(.*)")
CROSS_EVAL_REGEX = re.compile(r"(v\d+)-(\d+)-vs-(v\d+)-(\d+)")
MODELS_FROM_FN = re.compile(r"(\d{6}-[a-z-]+)-(\d{6}-[a-z-]+)$")


def assert_pair_matches(pb, pw, m1, m2, sgf_file):
    if pb == m1 and pw == m2:
        return False
    if pb == m2 and pw == m1:
        return True
    assert False, ((pb, pw), (m1, m2), sgf_file)

def determine_model_id(sgf_file, pb, pw, model_runs):
    """Determine which run+model PB and PW are."""

    # Get number ("123") of model with no leading zeros.
    num_b = str(int(pb.split('-')[0]))
    num_w = str(int(pw.split('-')[0]))

    # Possible runs for black and white player.
    runs_pb = model_runs[pb]
    runs_pw = model_runs[pw]
    assert runs_pb and runs_pw, (pb, pw)

    # Validation that pb/pw (from cbt) match the filename.
    models = MODELS_FROM_FN.search(sgf_file)
    assert models, sgf_file
    m_1, m_2 = models.groups()
    assert_pair_matches(pb, pw, m_1, m_2, sgf_file)

    simple = CROSS_EVAL_REGEX.search(sgf_file)
    if not simple:
        # After #812, All cross evals are as vX-XXX-vs-vZ-ZZZ,
        # so this must be an inter-run eval games.

        # Check if both models (num + name) are unique to a single run.
        if len(runs_pb & runs_pw) == 1:
            run = min(runs_pb & runs_pw)
            return run, run

        # Both v12 and v15 have the same name for both these models!
        # 2018-09-30/1538304073-minigo-cc-evaluator-357-332-wb-9bjnq-0-000332-zealous-000357-arachne
        return None

    run_1, num_1, run_2, num_2 = simple.groups()

    # We have to unravel a mystery here.
    # filename tells up which number goes with which run,
    # filename also tells us model_name (number + name),
    # And pb,pw are model_name from PB[] and PW[].
    #
    # The easy case is num_1 != num_2:
    #      int(PB) => model_number which identifies PB run (same for PW)
    # The hard case is num_1 == num_2:
    #       this requires us checking if either PB/PW's model number + name
    #       is unique to only one of the two runs.

    to_consider = {run_1, run_2}
    runs_pb = runs_pb & to_consider
    runs_pw = runs_pw & to_consider
    assert runs_pb and runs_pw

    # To simply code assume run_1 goes with pb.
    # If not set swap = True

    if num_1 != num_2:
        swap = assert_pair_matches(num_b, num_w, num_1, num_2, sgf_file)
    else:
        # Imagine v12-80-vs-v10-80-bw-nhh5x-0-000080-duke-000080-duke
        # No way to tell which 80-duke is PB or PW
        assert pw != pb, (sgf_file)

        if len(runs_pb) == 1:
            swap = run_2 in runs_pb
        elif len(runs_pw) == 1:
            swap = run_1 in runs_pw
        else:
            # This would be very unlucky, both runs would have to have the same
            # model names for both numbers.
            assert False, (sgf_file, runs_pb, runs_pw)

    if swap:
        run_1, num_1, run_2, num_2 = \
            run_2, num_2, run_1, num_1

    assert num_b == num_1 and run_1 in runs_pb
    assert num_w == num_2 and run_2 in runs_pw

    # Verify the inverse isn't also valid.
    assert not (run_1 != run_2 and
                num_b != num_2 and run_2 in runs_pb and
                num_w != num_1 and run_1 in runs_pw), sgf_file

    # (run_b, run_b)
    return run_1, run_2


def read_models(db):
    """
    Read model names and runs from db.

    Returns:
      {(<run>,<model_name>): db_model_id}, {model_name: [run_a, run_b]}
    """
    model_ids = {}
    model_runs = defaultdict(set)

    cur = db.execute("SELECT id, model_name, bucket FROM models")
    for model_id, name, run in cur.fetchall():
        assert model_id not in model_ids
        model_ids[(run, name)] = model_id
        model_runs[name].add(run)

    return model_ids, model_runs


def setup_models(models_table):
    """
    Read all (~10k) models from cbt and db
    Merge both lists and write any new models to db.

    Returns:
      {(<run>,<model_name>): db_model_id}, {model_name: [run_a, run_b]}
    """

    with sqlite3.connect("cbt_ratings.db") as db:
        model_ids, model_runs = read_models(db)

        cbt_models = 0
        new_models = []
        for row in tqdm(models_table.read_rows()):
            cbt_models += 1
            name = row.cell_value(METADATA, b'model').decode()
            run = row.cell_value(METADATA, b'run').decode()
            num = int(row.cell_value(METADATA, b'model_num').decode())
            if (run, name) not in model_ids:
                new_models.append((name, run, num))

        print("Existing models(cbt):", cbt_models)

        if new_models:
            print("New models insertted into db:", len(new_models))

            db.executemany(
                """INSERT INTO models VALUES (
                null, ?, ?, ?, 0, 0, 0, 0, 0, 0, 0, 0)""",
                new_models)

            # Read from db to pick up new model_ids.
            model_ids, model_runs = read_models(db)

        assert len(model_ids) == cbt_models
        return model_ids, model_runs


def sync():
    # TODO(djk): table.exists() without admin=True, read_only=False.
    models_table = (bigtable
                    .Client(FLAGS.cbt_project, read_only=True)
                    .instance(FLAGS.cbt_instance)
                    .table("models"))

    eval_games_table = (bigtable
                        .Client(FLAGS.cbt_project, read_only=True)
                        .instance(FLAGS.cbt_instance)
                        .table("eval_games"))

    model_ids, model_runs = setup_models(models_table)
    # TODO(sethtroisi): Potentially only update from a starting rows.

    status = Counter()
    game_records = []

    reader = tqdm(eval_games_table.read_rows(), desc="eval_game", unit=" rows")
    for row in reader:
        row_key = row.row_key

        if row_key == TABLE_STATE:
            continue

        sgf_file = row.cell_value(METADATA, b'sgf').decode()
        timestamp = sgf_file.split('-')[0]
        pb = row.cell_value(METADATA, b'black').decode()
        pw = row.cell_value(METADATA, b'white').decode()
        result = row.cell_value(METADATA, b'result').decode()
        black_won = result.lower().startswith('b')

        assert pw and pb and result, row_key

        status['considered'] += 1

        # TODO(sethtroisi): At somepoint it would be nice to store this
        # during evaluation and backfill cbt.

        test = determine_model_id(sgf_file, pb, pw, model_runs)
        if test is None:
            status['determine failed'] += 1
            continue

        run_b, run_w = test
        b_model_id = model_ids[(run_b, pb)]
        w_model_id = model_ids[(run_w, pw)]

        game_records.append([
            timestamp, sgf_file,
            b_model_id, w_model_id,
            black_won, result
        ])

    print()
    with sqlite3.connect("cbt_ratings.db") as db:
        c = db.cursor()

        # Most of these games will not be new.
        c.executemany(
            "INSERT OR IGNORE INTO games VALUES (null, ?, ?, ?, ?, ?, ?)",
            game_records)

        inserted = c.rowcount
        if inserted > 0:
            print("Inserted {} new games from {} rows".format(
                inserted, len(game_records)))

        c.executescript("""
            DELETE FROM wins;
            INSERT INTO wins
                SELECT game_id, b_id, w_id FROM games WHERE black_won
                UNION
                SELECT game_id, w_id, b_id FROM games WHERE NOT black_won;
        """)
        print("Wins({}) updated".format(c.rowcount))

        # Do all the calculations here with maps instead of in SQL.
        # num_games, num_wins, black_games, black_wins, white_games, white_wins
        model_stats = defaultdict(lambda: [0, 0, 0, 0, 0, 0])

        cur = c.execute("select b_id, w_id, black_won from games")
        for b_id, w_id, black_won in cur.fetchall():
            model_stats[b_id][0] += 1
            model_stats[b_id][1] += black_won
            model_stats[b_id][2] += 1
            model_stats[b_id][3] += black_won

            model_stats[w_id][0] += 1
            model_stats[w_id][1] += not black_won
            model_stats[w_id][4] += 1
            model_stats[w_id][5] += not black_won

        c.executemany(
            """
            UPDATE models set
                num_games = ?, num_wins = ?,
                black_games = ?, black_wins = ?,
                white_games = ?, white_wins = ?
            WHERE id = ?
            """,
            (tuple(stats) + (m_id,) for m_id, stats in model_stats.items()))
        print("Models({}) updated".format(c.rowcount))

    print()
    for s, count in status.most_common():
        print("{:<10}".format(count), s)


def current_run(model_ids):
    """Return the largest number run.

    If you want something different, feel free to extend with fsdb or something.
    """
    all_runs = set(run for run, name in model_ids)

    # TODO fix before submit
    return max(all_runs)

    # Assumption of the run format.
    assert all(re.match("v[0-9]+", run) for run in all_runs), all_runs

    # "Current" is defined as largest numbered run.
    max_num = max(int(r[1:] for r in all_runs))
    return 'v' + str(max_num)


def run_win_records(run=None):
    """Return all win records between two models both in run

    If run is None, models must be in different runs.
    """

    with sqlite3.connect('cbt_ratings.db') as db:
        if run:
            data = db.execute(
                """
                SELECT model_winner, model_loser FROM wins
                JOIN models m1, models m2 WHERE
                    m1.bucket = ? AND m1.id = model_winner AND
                    m2.bucket = ? AND m2.id = model_loser
                """,
                (run, run))
        else:
            # run=None is for cross eval, don't allow games from same run
            data = db.execute("""
                SELECT model_winner, model_loser FROM wins
                JOIN models m1, models m2 WHERE
                    m1.id = model_winner AND
                    m2.id = model_loser AND
                    m1.bucket != m2.bucket
                """)

    return data.fetchall()


def compute_ratings(model_ids, data=None):
    """ Calculate ratings from win records

    Args:
      model_ids: dictionary of {(run, model_name): model_id}
      data: list of tuples of (winner_id, loser_id)

    Returns:
      dictionary {(run, model_name): (rating, variance)}
    """
    if data is None:
        with sqlite3.connect("cbt_ratings.db") as db:
            query = "select model_winner, model_loser from wins"
            data = db.execute(query).fetchall()

    data_ids = sorted(set(np.array(data).flatten()))

    # Map data_ids to a contiguous range.
    new_id = {}
    for i, m in enumerate(data_ids):
        new_id[m] = i

    # Create inverse model_ids lookup
    model_names = {v: k for k, v in model_ids.items()}

    # A function to rewrite the data_ids in our pairs
    def ilsr_data(d):
        p1, p2 = d
        p1 = new_id[p1]
        p2 = new_id[p2]
        return (p1, p2)

    pairs = list(map(ilsr_data, data))
    ilsr_param = choix.ilsr_pairwise(
        len(data_ids),
        pairs,
        alpha=0.0001,
        max_iter=800)

    hessian = choix.opt.PairwiseFcts(pairs, penalty=.1).hessian(ilsr_param)
    std_err = np.sqrt(np.diagonal(np.linalg.inv(hessian)))

    # Elo conversion
    elo_mult = 400 / math.log(10)

    # Used to make all ratings positive.
    min_rating = min(ilsr_param)

    ratings = {}
    for m_id, param, err in zip(data_ids, ilsr_param, std_err):
        model_rating = (elo_mult * (param - min_rating), elo_mult * err)
        ratings[model_names[m_id]] = model_rating

    return ratings


def top_n(n=10):
    with sqlite3.connect('cbt_ratings.db') as db:
        model_ids, _ = read_models(db)

    data = run_win_records()
    ratings = compute_ratings(model_ids, data)
    top_models = sorted(ratings.items(), key=lambda k: k[::-1])
    return top_models[-n:][::-1]


def suggest_pairs(top_n=10, per_n=3, ignore_before=300):
    """ Find the maximally interesting pairs of models to match up
    First, sort the ratings by uncertainty.
    Then, take the ten highest models with the highest uncertainty
    For each of them, call them `m1`
    Sort all the models by their distance from m1's rating and take the 20
    nearest rated models. ('candidate_m2s')
    Choose `per_n` random pairings, (m1, m2), from this list

    `top_n` will pair the top n models by uncertainty.
    `per_n` will give each of the top_n models this many opponents
    `ignore_before` is the model number to `filter` off, i.e., the early models.
    Returns a list of *model numbers*, not model ids.
    """

    with sqlite3.connect('cbt_ratings.db') as db:
        model_ids, model_runs = read_models(db)

    run = current_run(model_ids)
    data = run_win_records(run)

    ratings = compute_ratings(model_ids, data)
    ratings = [(name,) + rating for (run, name), rating in ratings.items()]
    ratings.sort()

    # Filter off the first X models, which improve quickly but are chaotic.
    ratings = ratings[ignore_before:]

    # Sort by rating variance
    ratings.sort(key=lambda r: r[2], reverse=True)

    pairs = []
    for m1 in ratings[:top_n]:
        print("Pairing {}, sigma {:.1f} (Rating {:.1f})".format(
            m1[0], m1[2], m1[1]))

        def rating_diff(m2):
            """Rating difference between two models"""
            return abs(m1[1] - m2[1])

        candidate_m2s = sorted(ratings, key=rating_diff)[1:20]
        for m2 in random.sample(candidate_m2s, per_n):
            pairs.append((m1[0], m2[0]))
            print("    {}, rating delta {:.1f}".format(m2[0], rating_diff(m2)))

    return pairs


def main():
    if FLAGS.sync_ratings:
        sync()

    with sqlite3.connect('cbt_ratings.db') as db:
        model_ids, model_runs = read_models(db)

    data = run_win_records()

    print("DB has", len(data), "games")
    if not data:
        return

    ratings = compute_ratings(model_ids, data)
    top_models = sorted(ratings.items(), key=lambda k: k[::-1])
    print()
    print("Best models")
    for k, v in top_models[-20:][::-1]:
        print("{:>30}: {}".format("/".join(k), v))

    # Stats on recent models
    run = 'v' + str(max(int(r[1:]) for r, m in model_ids))
    print()
    print("Recent ratings for", run)
    for m in sorted(m for r, m in model_ids if r == run)[-20:]:
        name = run + "/" + m
        rating = ratings.get((run, m))
        if rating:
            rating, sigma = rating
            print("{:>30}:  {:.2f} ({:.3f})".format(name, rating, sigma))
        else:
            print("{:>30}:  not found".format(name))

    random.seed(5)
    # Suggest some pairs
    print()
    suggest_pairs(5, 2)


if __name__ == '__main__':
    remaining_argv = flags.FLAGS(sys.argv, known_only=True)
    main()
