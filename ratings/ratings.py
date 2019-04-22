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

import sys
sys.path.insert(0, '.')

from absl import flags

import choix
import numpy as np
import sqlite3
import os
import re
from rl_loop import fsdb
import random
import subprocess
import math
from tqdm import tqdm
import datetime as dt


flags.DEFINE_bool('sync_ratings', False, 'Synchronize files before computing ratings.')

FLAGS = flags.FLAGS

EVAL_REGEX = "(\d*)-minigo-cc-evaluator-"
MODEL_REGEX = "(\d*)-(.*)"
PW_REGEX = "PW\[([^]]*)\]"
PB_REGEX = "PB\[([^]]*)\]"
RESULT_REGEX = "RE\[([^]]*)\]"


def maybe_insert_model(db, bucket, name, num):
    with db:
        db.execute("""insert or ignore into models(
                      model_name, model_num, bucket,
                      num_games, num_wins, black_games,
                      black_wins, white_games, white_wins) values(
                      ?, ?, ?,
                      0, 0, 0,
                      0, 0, 0)""", [name, num, bucket])


def model_id_of(name_or_num):
    db = sqlite3.connect("ratings.db")
    bucket = fsdb.models_dir()
    if not isinstance(name_or_num, str):
        name_or_num = fsdb.get_model(name_or_num)
    return rowid_for(db, bucket, name_or_num)


def model_num_for(model_id):
    try:
        db = sqlite3.connect("ratings.db")
        return db.execute("select model_num from models where id = ?",
                          (model_id,)).fetchone()[0]
    except:
        print("No model found for id: {}".format(model_id))
        raise

def rowid_for(db, bucket, name):
    try:
        return db.execute("select id from models where bucket = ? "
                          "and model_name = ?",
                          [bucket, name]).fetchone()[0]
    except:
        print("No row found for bucket: {} name: {}".format(bucket, name))
        return None


def import_files(files, bucket=None):
    if bucket is None:
        bucket = fsdb.models_dir()

    print("Importing for bucket:", bucket)
    db = sqlite3.connect("ratings.db")
    new_games = 0
    with db:
        c = db.cursor()
        for _file in tqdm(files):
            match = re.match(EVAL_REGEX, os.path.basename(_file))
            if not match:
                print("Bad file: ", _file)
                continue
            timestamp = match.groups(1)[0]
            with open(_file) as f:
                text = f.read()
            pw = re.search(PW_REGEX, text)
            pb = re.search(PB_REGEX, text)
            result = re.search(RESULT_REGEX, text)
            if not (pw and pb and result):
                print("Fields not found: ", _file)

            pw = pw.group(1)
            pb = pb.group(1)
            result = result.group(1)

            m_num_w = re.match(MODEL_REGEX, pw).group(1)
            m_num_b = re.match(MODEL_REGEX, pb).group(1)

            try:
                # create models or ignore.
                maybe_insert_model(db, bucket, pb, m_num_b)
                maybe_insert_model(db, bucket, pw, m_num_w)

                b_id = rowid_for(db, bucket, pb)
                w_id = rowid_for(db, bucket, pw)

                # insert into games or bail
                game_id = None
                try:
                    with db:
                        c = db.cursor()
                        c.execute("""insert into games(timestamp, filename, b_id, w_id, black_won, result)
                                        values(?, ?, ?, ?, ?, ?)
                        """, [timestamp, os.path.relpath(_file), b_id, w_id, result.lower().startswith('b'), result])
                        game_id = c.lastrowid
                except sqlite3.IntegrityError:
                    # print("Duplicate game: {}".format(_file))
                    continue

                if game_id is None:
                    print("Somehow, game_id was None")

                # update wins/game counts on model, and wins table.
                c.execute("update models set num_games = num_games + 1 where id in (?, ?)", [b_id, w_id])
                if result.lower().startswith('b'):
                    c.execute("update models set black_games = black_games + 1, black_wins = black_wins + 1 where id = ?", (b_id,))
                    c.execute("update models set white_games = white_games + 1 where id = ?", (w_id,))
                    c.execute("insert into wins(game_id, model_winner, model_loser) values(?, ?, ?)",
                              [game_id, b_id, w_id])
                elif result.lower().startswith('w'):
                    c.execute("update models set black_games = black_games + 1 where id = ?", (b_id,))
                    c.execute("update models set white_games = white_games + 1, white_wins = white_wins + 1 where id = ?", (w_id,))
                    c.execute("insert into wins(game_id, model_winner, model_loser) values(?, ?, ?)",
                              [game_id, w_id, b_id])
                new_games += 1
                if new_games % 1000 == 0:
                    print("committing", new_games)
                    db.commit()
            except:
                print("Bailed!")
                db.rollback()
                raise
        print("Added {} new games to database".format(new_games))


def compute_ratings(data=None):
    """ Returns the tuples of (model_id, rating, sigma)
    N.B. that `model_id` here is NOT the model number in the run

    'data' is tuples of (winner, loser) model_ids (not model numbers)
    """
    if data is None:
        with sqlite3.connect("ratings.db") as db:
            data = db.execute("select model_winner, model_loser from wins").fetchall()
    model_ids = set([d[0] for d in data]).union(set([d[1] for d in data]))

    # Map model_ids to a contiguous range.
    ordered = sorted(model_ids)
    new_id = {}
    for i, m in enumerate(ordered):
        new_id[m] = i

    # A function to rewrite the model_ids in our pairs
    def ilsr_data(d):
        p1, p2 = d
        p1 = new_id[p1]
        p2 = new_id[p2]
        return (p1, p2)

    pairs = list(map(ilsr_data, data))
    ilsr_param = choix.ilsr_pairwise(
        len(ordered),
        pairs,
        alpha=0.0001,
        max_iter=800)

    hessian = choix.opt.PairwiseFcts(pairs, penalty=.1).hessian(ilsr_param)
    std_err = np.sqrt(np.diagonal(np.linalg.inv(hessian)))

    # Elo conversion
    elo_mult = 400 / math.log(10)

    min_rating = min(ilsr_param)
    ratings = {}

    for model_id, param, err in zip(ordered, ilsr_param, std_err):
        ratings[model_id] = (elo_mult * (param - min_rating), elo_mult * err)

    return ratings


def top_n(n=10):
    data = wins_subset(fsdb.models_dir())
    r = compute_ratings(data)
    return [(model_num_for(k), v) for v, k in
            sorted([(v, k) for k, v in r.items()])[-n:][::-1]]


def ingest_dirs(root, dirs):
    for d in dirs:
        if os.path.isdir(os.path.join(root, d)):
            fs = [os.path.join(root, d, f) for f in os.listdir(os.path.join(root, d))]
            print("Importing({}) from {}/{}".format(len(fs), root,d))
            import_files(fs)


def last_timestamp():
    db = sqlite3.connect("ratings.db")
    with db:
        ts = db.execute("select timestamp from games order by timestamp desc limit 1").fetchone()
    return ts[0] if ts else None


def suggest_pairs(top_n=10, per_n=3, ignore_before=300):
    """ Find the maximally interesting pairs of players to match up
    First, sort the ratings by uncertainty.
    Then, take the ten highest players with the highest uncertainty
    For each of them, call them `p1`
    Sort all the models by their distance from p1's rating and take the 20
    nearest rated models. ('candidate_p2s')
    Choose pairings, (p1, p2), randomly from this list.

    `top_n` will pair the top n models by uncertainty.
    `per_n` will give each of the top_n models this many opponents
    `ignore_before` is the model number to `filter` off, i.e., the early models.
    Returns a list of *model numbers*, not model ids.
    """
    db = sqlite3.connect("ratings.db")
    data = db.execute("select model_winner, model_loser from wins").fetchall()
    bucket_ids = [id[0] for id in db.execute(
        "select id from models where bucket = ?", (fsdb.models_dir(),)).fetchall()]
    bucket_ids.sort()
    data = [d for d in data if d[0] in bucket_ids and d[1] in bucket_ids]

    ratings = [(model_num_for(k),) + v for k, v in compute_ratings(data).items()]
    ratings.sort()
    ratings = ratings[ignore_before:]  # Filter off the first 100 models, which improve too fast.

    ratings.sort(key=lambda r: r[2], reverse=True)

    res = []
    for p1 in ratings[:top_n]:
        candidate_p2s = sorted(ratings, key=lambda p2_tup: abs(p1[1] - p2_tup[1]))[1:20]
        choices = random.sample(candidate_p2s, per_n)
        print("Pairing {}, sigma {:.2f} (Rating {:.2f})".format(p1[0], p1[2], p1[1]))
        for p2 in choices:
            res.append([p1[0], p2[0]])
            print("   {}, ratings delta {:.2f}".format(p2[0], abs(p1[1] - p2[1])))
    return res


def sync(root, force_all=False):
    last_ts = last_timestamp()
    if last_ts and not force_all:
        # Build a list of days from the day before our last timestamp to today
        num_days = (dt.datetime.utcnow() -
                    dt.datetime.utcfromtimestamp(last_ts) +
                    dt.timedelta(days=1)).days
        ds = [(dt.datetime.utcnow() - dt.timedelta(days=d)).strftime("%Y-%m-%d") for d in range(num_days + 1)]
        for d in ds:
            if not os.path.isdir(os.path.join(root, d)):
                os.mkdir(os.path.join(root, d))
            cmd = ["gsutil", "-m", "rsync", "-r", os.path.join(fsdb.eval_dir(), d), os.path.join(root, d)]
            print(" ".join(cmd))
            subprocess.call(cmd)
        ingest_dirs(root, ds)
    else:
        cmd = ["gsutil", "-m", "rsync", "-r", fsdb.eval_dir(), root]
        print(" ".join(cmd))
        subprocess.call(cmd)
        dirs = os.listdir(root)
        ingest_dirs(root, dirs)


def wins_subset(bucket):
    with sqlite3.connect('ratings.db') as db:
        data = db.execute(
            "SELECT model_winner, model_loser FROM wins WHERE"
            "  (model_winner in (SELECT id FROM models where bucket = ?)) or "
            "  (model_loser  in (SELECT id FROM models where bucket = ?)) ",
            (bucket, bucket)).fetchall()
    return data


def main():
    root = os.path.abspath(os.path.join("sgf", fsdb.FLAGS.bucket_name, "sgf/eval"))
    if FLAGS.sync_ratings:
        sync(root)

    for k, v in top_n(20):
        print("Top model {}: {}".format(k, v))

    db = sqlite3.connect("ratings.db")
    print("db has", db.execute("select count(*) from wins").fetchone()[0], "games")
    models = fsdb.get_models()
    for m in models[-10:]:
        m_id = model_id_of(m[0])
        if m_id in r:
            rat, sigma = r[m_id]
            print("{:>30}:  {:.2f} ({:.3f})".format(m[1], rat, sigma))
        else:
            print("{}, Model id not found({})".format(m[1], m_id))

    # Suggest some pairs
    random.seed(5)
    print()
    suggest_pairs(5, 2)


if __name__ == '__main__':
    remaining_argv = flags.FLAGS(sys.argv, known_only=True)
    main()
