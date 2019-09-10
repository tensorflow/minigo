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


import sys
sys.path.insert(0, '.')  # nopep8
import logging
import sqlite3
import collections
import datetime as dt

from flask import Flask, g, jsonify
from timeit import default_timer as timer

import os
import flask

import oneoffs.joseki.opening_freqs as openings

# static_folder is location of npm build
app = Flask(__name__, static_url_path="", static_folder="./build")

# Path to database relative to joseki_query.py, with pre-extracted joseki
# information (see 'opening_frequencies.py')
DATABASE = ''
assert os.path.exists(DATABASE)

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def get_sequence_hour_counts(seq):
    cur = get_db().execute('''
                     select run, hour, count from joseki_counts
                           where seq_id in (select id from joseki where seq = ?);
                     ''', (seq, ))
    return list(cur.fetchall())

def seq_id_or_none(db, seq):
    s_id = db.execute("select id from joseki where seq = ?", (seq,)).fetchone()
    return s_id[0] if s_id else None

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/nexts", methods=["POST"])
def next_moves():
    d = flask.request.get_json()
    prefix = d['params']['prefix']
    run = d['params']['run']

    db = get_db()

    # If error, return this default result.
    res = {'count': 0, 'next_moves': {}}

    # For the blank board, there's no 'prefix' string sent, and so there's no
    # 'next_move' information to be had.  This selects and counts all the unique
    # opening moves from the joseki table.

    # TODO: Once the db is rebuilt with empty opening stats, this special case
    # could be removed.
    if not prefix:
        nexts = db.execute("select distinct(substr(seq, 0, 7)), count(*) from joseki group  by 1;").fetchall()
        total = sum([n[1] for n in nexts])
    else:
        s_id = seq_id_or_none(db, prefix)
        if not s_id:
            return jsonify(res)

        if run is None:
            nexts = db.execute("""
                               select next_move, sum(count) from next_moves where seq_id = ? group by 1
                               """, (s_id,)).fetchall()
            total = db.execute("select sum(count) from joseki_counts where seq_id = ?",
                               (s_id,)).fetchone()[0]
        else:
            start = timer()
            nexts = db.execute("""
                               select next_move, sum(nm.count)
                               from next_moves as nm join joseki_counts as jc
                               on jc.id == nm.joseki_hour_id
                               where nm.seq_id = ? and jc.run = ? group by 1
                               """, (s_id, run)).fetchall()
            end = timer()
            print('%.4f seconds for fancy join.' % (end-start,))
            total = db.execute("select sum(count) from joseki_counts where seq_id = ? and run = ?",
                               (s_id, run)).fetchone()[0]

    if not nexts:
        print("No next moves found, post params:", d['params'])
        return jsonify(res)

    next_moves = dict(nexts)
    tot = sum(next_moves.values())

    for k in next_moves:
        next_moves[k] /= tot

    max_v = max(next_moves.values())

    next_moves = {k:v / max_v for k,v in next_moves.items() if v > 0.001}
    res = {'count': total,
           'next_moves': next_moves}

    return jsonify(res)


@app.route("/games", methods=["POST"])
def games():
    d = flask.request.get_json()
    prefix = d['params']['sgf']
    sort_hour = d['params']['sort']
    run = d['params']['run']
    # "page" is 1-indexed, so subtract 1 to get the proper OFFSET.
    page = d['params']['page'] - 1
    db = get_db()

    if sort_hour.lower() not in ('desc', 'asc'):
        print("Invalid input for sort_hour param: ", sort_hour)
        return jsonify({'rows': []})

    s_id = seq_id_or_none(db, prefix)
    if not s_id:
        return jsonify({'rows': []})

    q = """select example_sgf, hour, run, b_wins*1.0/count from joseki_counts
        where seq_id = ? {} order by hour {} limit 30 offset ?""".format(
            "and run = ?" if run else "", sort_hour)

    if run:
        rows = db.execute(q, (s_id, run, page * 30)).fetchall()
    else:
        rows = db.execute(q, (s_id, page * 30)).fetchall()

    res = [ {'game': os.path.basename(r[0]), 'hour': r[1],
             'run': r[2], 'winrate': r[3]} for r in rows]
    return jsonify({'rows': res})


@app.route("/search", methods=["POST"])
def search():
    d = flask.request.get_json()
    print(d)
    query = d['params']['sgf']

    ts = lambda hr: int(dt.datetime.strptime(hr, "%Y-%m-%d-%H").timestamp())
    ranges = openings.run_time_ranges(get_db())
    interps = openings.build_run_time_transformers(ranges)

    runs = sorted(ranges.keys())

    cols = []
    cols.append({'id': 'time', 'label': '% of Training', 'type': 'number'})
    for run in runs:
        cols.append({'id': run + ' count', 'label': run + ' times seen', 'type': 'number'})

    sequence_counts = get_sequence_hour_counts(query)

    rows = collections.defaultdict(lambda: [0 for i in range(len(runs))])

    for r, hr, ct in sequence_counts:
        key = interps[r](ts(hr))
        idx = runs.index(r)
        rows[key][idx] = ct

    row_data = [ {'c': [ {'v': key} ] + [{'v': v if v else None} for v in value ] }
                for key,value in rows.items()]
    obj = {'cols': cols, "rows": row_data, "sequence": query}
    return jsonify(obj)
