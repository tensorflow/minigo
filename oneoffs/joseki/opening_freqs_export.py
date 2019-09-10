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

"""Opening_freqs_export -- Write out static views of opening frequency data.

Once opening_freqs.py has created a sqlite db with data on joseki information,
this file can be used to write some html which visualizes that data.


create_top_report -- loads the most popular 300 joseki (across all times and
runs), and then attempts to prune "uninteresting" subsequences via the logic in
"collapse_common_prefix".  The remaining joseki and their count information is
rendered into a javascript object inside a jinja template, including their
frequency histograms.

This was a static proof of concept which was the precursor for the React based
joseki explorer.


create_hourly_reports -- This renders a much simpler html fragment with just an
SGF viewer showing the most popular joseki for a given hour.  Dead code, but
useful for unfinished runs.

"""

import os
import sqlite3
import json
import datetime as dt
import collections
from hashlib import sha256
from jinja2 import Template

from absl import app
from absl import flags

import oneoffs.joseki.opening_freqs as openings

FLAGS = flags.FLAGS

# See also flags in opening_freqs
flags.DEFINE_string("template", 'joseki_static_report.html', "Path to template to render")
flags.DEFINE_string("out_file", 'openings.html', "Where to write the report")
flags.DEFINE_integer("top_n", 100, "Number of top openings to use per run")


def sha(sequence):
    """ Simple sha helper """
    return sha256(bytes(sequence)).hexdigest()


class MoveTrieNode():
    """
    Unlike regular tries, We don't really care about valid standalone prefixes
    (i.e., prefixes that are complete sequences themselves).  So no leaf
    field needed.  Nor do we update a count.
    The atom used in the nodes is a "move", i.e., anything between ';' 's
    """
    def __init__(self, move, count=0, parent=None):
        self.move = move
        self.children = []
        self.parent = parent
        self.count = count

    def add(self, sequence, count):
        """ adds a new sequence under this node, with its count. """
        node = self
        for move in sequence.split(';'):
            found = False
            for kid in node.children:
                if kid.move == move:
                    found = True
                    node = kid
                    break
            if not found and move != "":
                new_node = MoveTrieNode(move, count, node)
                node.children.append(new_node)
                node = new_node
        return node

    def sequence_to(self):
        """ Renders the prefix up to this node
        i.e.
        n = root.add("A;B;C;D;")
        n.sequence_to() == "A;B;C;D;"
        """
        s = []
        node = self
        while node.parent:
            s = [node.move,] + s
            node = node.parent
        return ';'.join(s) + ';'

    def __str__(self):
        return self.sequence_to()


def collapse_common_prefix(seq_cts, threshold=0.9):
    """
    'seq_cts' is the list of ("B[];W[];...", count) sequences to prune.
    Suppose they contain
    ("A;B;C;D;", 1000),
    ("A;B;C;D;E;", 500),
    ("A;B;C;D;E;F;", 490),
    ("A;B;C;D;X;", 500),
    ("A;B;C;D;X;Y;", 490)
    , etc.
    We would want the longest sequences (ABCDXY, ABCDEF), and their common root,
    (ABCD), and we would remove the intermediate sequences.

    Returns a list formatted like seq_cts
    """

    # Build the Trie.
    root = MoveTrieNode("", 0)
    for seq, ct in seq_cts:
        root.add(seq, ct)

    node = root
    keepers = []
    to_visit = []

    # We want to keep a node if:
    # a. it has no parent OR no kids
    # b. it has no child with > threshold frac of our own kids.
    while True:
        to_visit.extend(list(node.children))

        if not node.parent or not node.children:
            keepers.append((node.sequence_to(), node.count))
        else:
            all_one_kid = False
            for k in node.children:
                if k.count > (node.count * threshold):
                    all_one_kid = True
                    break # We're not checking that freq(parent) >= sum(freq(kids))
            if not all_one_kid:
                keepers.append((node.sequence_to(), node.count))
        if to_visit:
            node = to_visit.pop(0)
        else:
            break

    #truncate the first [empty] node.
    return keepers[1:]


def make_wgo_diagram(sequence, count):
    """
    Makes an html fragment showing a wgo board for the given sequence/count
    """
    return '''
<h3> Joseki {hash_id}, seen {count} times </h3>
<div id="{hash_id}">
  <div data-wgo="(;SZ[19];{sequence})"
       data-wgo-move="999"
       style="width: 300px"
       data-wgo-layout="bottom: ['Control']">
  </div>
  <div id="{hash_id}-chart">
  </div>
</div>'''.format(hash_id=sha(sequence), count=count, sequence=sequence)


def create_hourly_reports(hour_directory):
    """
    Creates an html page showing the most common sequences for the given hour.
    """
    hr = os.path.basename(hour_directory.rstrip('/'))

    db = sqlite3.connect(FLAGS.db_path)
    cur = db.execute('''
                     select seq, sum(count) from joseki_counts where hour = ? group by seq order by 2 desc limit 300;
                     ''', (hr,))
    sequences = list(cur.fetchall())

    dias = [make_wgo_diagram(seq, c) for (seq, c) in sequences]
    with open(os.path.join(FLAGS.in_dir, '{}-out.html'.format(hr)), 'w') as out:
        out.write(HTML_SHELL.format("\n".join(dias)))


HTML_SHELL = """
<!DOCTYPE HTML>
<html>
  <head>
    <title>My page</title>
    <script type="text/javascript" src="wgo/wgo.min.js"></script>
    <script type="text/javascript" src="wgo/wgo.player.min.js"></script>
    <link type="text/css" href="wgo/wgo.player.css" rel="stylesheet" />
  </head>
  <body>
{}
  </body>
</html>
"""


def get_runs(db):
    return [r[0] for r in db.execute('''
               select distinct(run) from joseki_counts;'''
                                     ).fetchall()]

def top_seqs_by_run(db, topN=300):
    runs = get_runs(db)
    top_seqs_by_run = {}
    for r in runs:
        cur = db.execute('''
                         select seq, sum(count) from joseki_counts
                         where run = ?
                         group by seq order by 2 desc limit ?;
                         ''', (r, topN))
        sequences = list(cur.fetchall())
        sequences = collapse_common_prefix(sequences)
        print("Pruned to ", len(sequences))
        sequences.sort(key=lambda s: s[1], reverse=True)
        top_seqs_by_run[r] = sequences
    return top_seqs_by_run


def create_top_report(top_n=100):
    """
    Creates an html page showing the most common sequences in the database, and
    charting their popularity over time.
    """
    db = sqlite3.connect(FLAGS.db_path)
    ts = lambda hr: int(dt.datetime.strptime(hr, "%Y-%m-%d-%H").timestamp())
    print('querying')
    ranges = openings.run_time_ranges(db)
    interps = openings.build_run_time_transformers(ranges)
    seqs_by_run = top_seqs_by_run(db, top_n)
    runs = sorted(seqs_by_run.keys())

    cols = []
    cols.append({'id': 'time', 'label': '% of Training', 'type': 'number'})
    for run in runs:
        cols.append({'id': run + 'count', 'label': run + ' times seen', 'type': 'number'})

    for run in runs:
        data = []
        sequences = seqs_by_run[run]
        for seq, count in sequences:
            print(run, seq, count)
            rows = collections.defaultdict(lambda: [0 for i in range(len(runs))])

            for idx, r in enumerate(runs):
                cur = db.execute('''
                                 SELECT hour, count from joseki_counts where seq = ? and run = ?;
                                 ''', (seq, r))

                for hr, ct in cur.fetchall():
                    key = interps[r](ts(hr))
                    rows[key][idx] = ct

            row_data = [ {'c': [ {'v': key} ] + [{'v': v if v else None} for v in value ] }
                        for key,value in rows.items()]
            obj = {'run': run, "count": count, 'cols': cols, "rows": row_data, "sequence": seq}
            data.append(obj)

        print('saving')
        tmpl = Template(open('oneoffs/joseki.html').read())
        with open(run + FLAGS.out_file, 'w') as out:
            out.write(tmpl.render(giant_blob=json.dumps(data), run=run, time_ranges=json.dumps(ranges)))


def main(_):
    """ Entrypoint for absl.app """
    create_top_report(FLAGS.top_n)

if __name__ == '__main__':
    app.run(main)
