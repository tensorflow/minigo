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
from collections import Counter
import re
import os
import coords

FLAGS = flags.FLAGS

MIN_JOSEKI_LENGTH = 5
MAX_JOSEKI_LENGTH = 20

flags.DEFINE_string("in_dir", None, "sgfs here are parsed.")
flags.DEFINE_string("db_path", 'joseki.db', "Path to josekidb")
flags.DEFINE_string("run_name", '',
                    "If the games should be associated with a 'run' key (e.g., 'v17')")
flags.DEFINE_integer("max_patterns", 5000,
                     "Number of most-common-joseki to save per hour")
flags.DEFINE_integer("threads", 8, "Number of threads to use.")
flags.DEFINE_float("sample_frac", 1.0,
                   "Fraction of sgfs in a directory to parse")


def sha(sequence):
    return sha256(bytes(sequence)).hexdigest()


SCHEMA = """

CREATE TABLE IF NOT EXISTS joseki (
  seq text primary key,
  length integer,
  num_tenukis integer
);

CREATE TABLE IF NOT EXISTS joseki_counts (
  id integer primary key,
  seq text,
  hour text,
  count integer,
  run text,
  example_sgf text,

  UNIQUE(seq, hour)
  FOREIGN KEY(seq) REFERENCES joseki(seq) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS game_counts_by_hour(
  id integer primary key,
  hour text,
  count integer
);
CREATE INDEX IF NOT EXISTS seq_index on joseki (seq);
"""


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


def extract_corners(game_path):
    '''
    returns a Counter("sequence", count) of all sequences/subsequences in all corners
    of a given game_path
    '''
    with open(game_path) as sgf_file:
        game_data = sgf_file.read().encode('utf-8')

    try:
        g = sgf.Sgf_game.from_bytes(game_data)
        _, moves = sgf_moves.get_setup_and_moves(g)
    except BaseException:
        print("bad file: ", game_path)
        return Counter()

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

    sequence_counts = Counter()
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
            seq += sgf_format((color, (18 - y, x)))

            if idx > MIN_JOSEKI_LENGTH:
                sequence_counts[seq] += 1

    return sequence_counts


def sgf_format(move):
    return "{}[{}];".format(move[0].upper(), coords.to_sgf(move[1]))


def analyze_dir(directory):
    """
    Parses all .sgfs in `directory` and updates database accordingly.

    `directory` is assumed to be a path ending in 'YYYY-mm-dd-HH'.
    e.g. /path/to/games/2019-07-01-00/
    """
    counts = Counter()
    example_sgfs = {}

    sgf_files = [os.path.join(directory, p)
                 for p in os.listdir(directory) if p.endswith('.sgf')]
    amt = int(len(sgf_files) * FLAGS.sample_frac)
    if FLAGS.sample_frac < 1:
        random.shuffle(sgf_files)

    hr = os.path.basename(directory.rstrip('/'))

    for path in sgf_files[:amt]:
        corners = extract_corners(path)
        counts.update(corners)
        for seq in corners:
            example_sgfs[seq] = os.path.join(hr, os.path.basename(path))

    db = sqlite3.connect(FLAGS.db_path, check_same_thread=False)
    with db:
        for c in counts.most_common(1000):
            cur = db.cursor()
            cur.execute(""" INSERT INTO joseki(seq, length, num_tenukis) VALUES(?, ?, ?)
                 ON CONFLICT DO NOTHING """,
                        (c[0], c[0].find(';'), count_tenukis(c[0])))
            cur.execute("""
                 INSERT INTO joseki_counts(seq, hour, count, run, example_sgf) VALUES (?, ?, ?, ?, ?) ON CONFLICT(seq,hour) DO UPDATE SET count=count + ?
                 """,
                        (c[0], hr, c[1], FLAGS.run_name, example_sgfs[c[0]], c[1]))
            cur.execute("""
                  INSERT INTO game_counts_by_hour(hour, count) VALUES (?, ?) ON CONFLICT DO NOTHING 
                        """,
                        (hr, c[1]))
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
