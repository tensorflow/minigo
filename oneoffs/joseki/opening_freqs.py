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

"""opening_freqs.py -- Analyze SGFs for opening pattern frequency information

Opening_freqs.py looks at subdirectories of SGFs and extracts joseki frequency
information to a sqlite DB. It tracks the first MAX_JOSEKI_LENGTH stones played
in each corner, moving to the next file when all four corners are 'finished'.
The 'sequences' are stored in SGF notation, i.e. "B[qq];W[pd]; ... ".   These
"SGF fragments" are not valid sgf files, but are easily embedded in various SGF
viewing tools, see opening_freqs_export.py for more.
"""

import sys
sys.path.insert(0, '.')

from tqdm import tqdm
from sgfmill import sgf, sgf_moves
from absl import flags
from absl import app
from hashlib import sha256
import sqlite3
import random
import multiprocessing as mp
from collections import Counter, defaultdict
import datetime as dt
import re
import functools
import os
import coords

FLAGS = flags.FLAGS

MIN_JOSEKI_LENGTH = 5
MAX_JOSEKI_LENGTH = 20

flags.DEFINE_string("in_dir", None, "sgfs here are parsed.")
flags.DEFINE_string("db_path", 'joseki.db', "Path to josekidb")
flags.DEFINE_string("run_name", '',
                    "If the games should be associated with a 'run' key (e.g., 'v17')")
flags.DEFINE_integer("most_common", 3000,
                     "Number of most-common-joseki to save per hour")
flags.DEFINE_integer("threads", 12, "Number of threads to use.")
flags.DEFINE_float("sample_frac", 1.0,
                   "Fraction of sgfs in a directory to parse")


def sha(sequence):
    return sha256(bytes(sequence)).hexdigest()


SCHEMA = """

CREATE TABLE IF NOT EXISTS joseki (
  id integer primary key,
  seq text,
  length integer,
  num_tenukis integer
);

CREATE TABLE IF NOT EXISTS joseki_counts (
  id integer primary key,
  seq_id integer,
  hour text,
  run text,
  count integer,
  b_wins integer,
  example_sgf text,

  UNIQUE(seq_id, hour, run),
  FOREIGN KEY(seq_id) REFERENCES joseki(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS next_moves (
  id integer primary key,
  seq_id integer,         -- select distinct(next_move), sum(count) from next_moves where seq_id='...' and 
  joseki_hour_id integer, -- select jc.hour, next_move, count from next_moves where seq='...' join on joseki_counts as jc where jc.id
  next_move text,         -- e.g. 'B[jj];'
  count integer,

  FOREIGN KEY(seq_id) REFERENCES joseki(id) ON DELETE CASCADE,
  FOREIGN KEY(joseki_hour_id) REFERENCES joseki_counts(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS game_counts_by_hour(
  id integer primary key,
  hour text,
  count integer
);

CREATE INDEX IF NOT EXISTS seq_index on joseki (seq);
CREATE INDEX IF NOT EXISTS nm_index on next_moves (seq_id);
CREATE INDEX IF NOT EXISTS jc_index on next_moves (joseki_hour_id);
CREATE INDEX IF NOT EXISTS jc_seq_index on joseki_counts (seq_id);
"""

COMPACT_NEXT_MOVES_TABLE_QUERY="""
DELETE FROM next_moves WHERE id IN (SELECT nm.id FROM next_moves AS nm JOIN joseki AS j ON nm.seq_id = j.id WHERE (j.seq || nm.next_
move) NOT IN (SELECT seq FROM joseki));
"""


def run_time_ranges(db):
    ts = lambda hr: int(dt.datetime.strptime(hr, "%Y-%m-%d-%H").timestamp())
    runs = {r[0]: (ts(r[1]), ts(r[2])) for r in db.execute('''
        select run, min(hour), max(hour) from joseki_counts group by 1;
        ''').fetchall()}
    return runs


def build_run_time_transformers(ranges, buckets=250):
    """ Build a dict of functions to transform from a timestamp into a relative
    offset.  E.g.
    input: {'v17': (1234567890, 1235567879) ... }
    output: {'v17': lambda t: (t - min) * (1/max) ... }
    """

    funcs = {}
    def f(t, min_, max_):
        #return "%0.2f" % ((t-min_) * (1/(max_-min_)))
        key = (t-min_) * (1/(max_-min_))
        return "%0.3f" % (int(buckets*key) / (buckets / 100.0))

    for run, range_ in ranges.items():
        funcs[run] = functools.partial(f, min_=range_[0], max_=range_[1])

    return funcs


def move_to_corner(move):
    """
    Take a move ('color', (row,col)).
    where row, col are 0-indexed (from sgfmill)
    Figure out which corner it's in
    Transform it to be in the upper right corner;
    Returns [corners, move]
      corners: A list of the indices of the corners it is in (0-3)
      move: The transformed move
    """

    color, m = move
    if not m:
        return (None, move)
    y, x = m
    corners = []

    if (x >= 9 and y >= 9):
        corners.append(0)
    elif (x >= 9 and y <= 9):
        corners.append(1)
    elif (x <= 9 and y <= 9):
        corners.append(2)
    elif (x <= 9 and y >= 9):
        corners.append(3)

    y -= 9
    x -= 9
    y = abs(y)
    x = abs(x)
    y += 9
    x += 9

    return [corners, (color, (y, x))]


def extract_from_game(game_path):
    """
    In addition to the corner information returned, we also add the winner,
    returning a tuple of ((sequence_counts, nextmove_counts), winner)
    where `winner` == 1 for B win, 0 otherwise.
    """
    with open(game_path) as sgf_file:
        game_data = sgf_file.read().encode('utf-8')

    try:
        g = sgf.Sgf_game.from_bytes(game_data)
        _, moves = sgf_moves.get_setup_and_moves(g)
    except BaseException:
        print("bad file: ", game_path)
        return Counter(), {}

    return (extract_corners(moves), 1 if g.get_winner().lower() == 'b' else 0)


def extract_corners(moves):
    '''
    Takes a list of moves ('color', (row,col)) (e.g. from sgfmill)
    returns a tuple of sequence_counts, nextmove_counts:
      sequence_counts: a Counter("sequence", count) of all sequences/subsequences in all corners
                       of a given game_path
      nextmove_counts: a dict of {"sequence": Counter} objects of the next moves of given sequences.
    '''
    corner_trackers = [[], [], [], []]
    stop = []

    for m in moves:
        corners, m = move_to_corner(m)
        if not corners:
            continue
        for c in corners:
            if c in stop:
                continue
            corner_trackers[c].append(m)
            if len(corner_trackers[c]) > MAX_JOSEKI_LENGTH:
                stop.append(c)
        if len(stop) == 4:
            break

    # we now have the sequences of the four corners of the board extracted, but
    # not 'canonical', i.e., the reflections across y=x would be distinct.
    # so, canonicalize them
    # TODO: this doesn't really work for when the position returns to a
    # symmetrical one, e.g.  4-4, knight approach, tenuki, knight approach the
    # other side = two sets of sequences even though the position is symmetric.

    sequence_counts = Counter()
    next_moves = defaultdict(Counter)
    for c in corner_trackers:
        seq = ""
        canonical = None
        for idx, (color, m) in enumerate(c):
            y, x = m
            if y < x and canonical is None:
                canonical = True
            elif y > x and canonical is None:
                x, y = y, x
                canonical = False
            elif canonical is False:
                x, y = y, x
            next_move = sgf_format((color, (18 - y, x)))
            next_moves[seq][next_move] += 1
            seq += next_move
            sequence_counts[seq] += 1

    return sequence_counts, next_moves


def sgf_format(move):
    return "{}[{}];".format(move[0].upper(), coords.to_sgf(move[1]))


def analyze_dir(directory):
    """
    Parses all .sgfs in `directory` and updates database accordingly.

    `directory` is assumed to be a path ending in 'YYYY-mm-dd-HH'.
    e.g. /path/to/games/2019-07-01-00/
    """
    counts = Counter()
    next_moves = defaultdict(Counter)
    example_sgfs = {}
    b_wins = Counter()

    sgf_files = [os.path.join(directory, p)
                 for p in os.listdir(directory) if p.endswith('.sgf')]
    amt = int(len(sgf_files) * FLAGS.sample_frac)
    if FLAGS.sample_frac < 1:
        random.shuffle(sgf_files)

    hr = os.path.basename(directory.rstrip('/'))

    for path in sgf_files[:amt]:
        (corners, seq_nexts), b_win = extract_from_game(path)
        counts.update(corners)
        for seq, next_cts in seq_nexts.items():
            next_moves[seq].update(next_cts)
        for seq in corners:
            b_wins[seq] += b_win
            if not seq in example_sgfs:
                example_sgfs[seq] = os.path.join(hr, os.path.basename(path))

    db = sqlite3.connect(FLAGS.db_path, check_same_thread=False)
    with db:
        for seq, count in counts.most_common(FLAGS.most_common):
            cur = db.cursor()

            s_id = cur.execute('select id from joseki where seq=?', (seq,)).fetchone()
            if s_id:
                s_id = s_id[0]
            else:
                cur.execute(""" INSERT INTO joseki(seq, length, num_tenukis) VALUES(?, ?, ?) """,
                            (seq, seq.count(';'), count_tenukis(seq)))
                s_id = cur.lastrowid

            cur.execute("""
                 INSERT INTO joseki_counts(seq_id, hour, count, run, b_wins, example_sgf) VALUES (?, ?, ?, ?, ?, ?)
                 """, (s_id, hr, count, FLAGS.run_name, b_wins[seq], example_sgfs[seq]))
            jc_id = cur.lastrowid

            for next_move, next_count in next_moves[seq].items():
                cur.execute("""
                     INSERT INTO next_moves(seq_id, joseki_hour_id, next_move, count) VALUES (?, ?, ?, ?)
                     """, (s_id, jc_id, next_move, next_count))

        db.commit()
    db.close()


def count_tenukis(seq):
    colors = ''.join([c for c in seq if c in 'BW'])
    # string.count only finds non-overlapping, and 'BWWW' is 2 tenukis
    # thus this weird regex expr.
    return (sum(1 for _ in re.finditer('(?=BB)', colors)) +
            sum(1 for _ in re.finditer('(?=WW)', colors)))

def main(_):
    """Entrypoint for absl.app"""
    db = sqlite3.connect(FLAGS.db_path)
    db.executescript(SCHEMA)
    db.close()

    root = FLAGS.in_dir
    dirs = [os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))]

    total = len(dirs)
    with mp.Pool(FLAGS.threads) as p:
        list(
            tqdm(
                p.imap_unordered(
                    analyze_dir,
                    dirs),
                desc="Extracting Joseki",
                total=total))


if __name__ == '__main__':
    app.run(main)
