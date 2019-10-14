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
sys.path.insert(0, '.')

from absl import app

import fnmatch
import os
import re
from tqdm import tqdm
from collections import defaultdict
from ratings import math_ratings


PW_REGEX = r"PW\[([^]]*)\]"
PB_REGEX = r"PB\[([^]]*)\]"
RESULT_REGEX = r"RE\[([^]]*)\]"

def extract_pairwise(files):
    """Given a set of paths to sgf files (`files`), extract the PW, PB, & RE
    tags from the file, build a lookup table for {player: id}, and a list of
    (winner, loser) id pairs.

    Returns `ids, results`, the id lookup dictionary and the list of pairs as
    tuples.
    """

    # this defaultdict gives each new key referenced a unique increasing int
    # starting from zero.  Since the choix library wants a continuous set of
    # ids, this is works to transform the data into (N, M) pairs.
    ids = defaultdict(lambda: len(ids))
    results = []
    for _file in tqdm(files):
        with open(_file) as f:
            text = f.read()
        pw = re.search(PW_REGEX, text)
        pb = re.search(PB_REGEX, text)
        result = re.search(RESULT_REGEX, text)
        if not (pw and pb and result):
            print("Player or result fields not found: ", _file)
            continue

        pw = pw.group(1)
        pb = pb.group(1)
        result = result.group(1).lower()

        if pw == pb:
            print("Players were the same: ", _file)
            continue
        if result.startswith('b'):
            results.append((ids[pb], ids[pw]))
        if result.startswith('w'):
            results.append((ids[pw], ids[pb]))

    return ids, results


def fancyprint_ratings(ids, ratings, results=None):
    """Prints the dictionary given in `ratings` with fancy headings.

    Optional arg `results` is the individual pairs of (winner_id, loser_id).
    If passed, this function will also print the W/L records of each player.
    """
    player_lookup = {v:k for k, v in ids.items()}
    HEADER = "\n{:<25s}{:>8s}{:>8s}{:>8}{:>7}-{:<8}"
    ROW = "{:<25.23s} {:6.0f}  {:6.0f}  {:>6d}  {:>6d}-{:<6d}"

    sorted_ratings = sorted(
        ratings.items(), key=lambda i: i[1][0], reverse=True)

    # If we don't have win-loss results, just summarize the ratings of the
    # players and bail.
    if not results:
        for pid, (rating, sigma) in sorted_ratings:
            print("{:25s}\t{:5.1f}\t{:5.1f}".format(player_lookup[pid], rating, sigma))
        return

    wins = {pid : sum([1 for r in results if r[0] == pid]) for pid in ids.values()}
    losses = {pid : sum([1 for r in results if r[1] == pid]) for pid in ids.values()}

    print("\n{} games played among {} players\n".format(len(results), len(ids)))
    print(HEADER.format("Name", "Rating", "Error", "Games", "Win", "Loss"))
    max_r = max(v[0] for v in ratings.values())
    for pid, (rating, sigma) in sorted_ratings:
        if rating != max_r:
            rating -= max_r
        print(ROW.format(player_lookup[pid], rating, sigma,
                         wins[pid] + losses[pid], wins[pid], losses[pid]))
    print("\n")



def main(argv):
    """Get the directory from argv, build a list of sgfs found under the
    directory, extract the pairings, compute the ratings, print them out"""
    if len(argv) < 2:
        print("Usage: rate_subdir.py <directory of sgfs to rate>")
        return 1
    sgfs = []
    for root, _, filenames in os.walk(argv[1]):
        for filename in fnmatch.filter(filenames, '*.sgf'):
            sgfs.append(os.path.join(root, filename))

    if not sgfs:
        print("No SGFs found in", argv)
        return 1

    print("Found {} sgfs".format(len(sgfs)))
    ids, results = extract_pairwise(sgfs)
    if not results:
        for m in sgfs:
            print(m)
        print("No SGFs with valid results were found")
        return 1
    rs = math_ratings.compute_ratings(results)

    fancyprint_ratings(ids, rs, results)
    return 0

if __name__ == '__main__':
    app.run(main)
