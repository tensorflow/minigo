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
Detect and count number of ladders in a directory of SGF files

Afterwards it maybe useful to inspect the files or thumbnails
find . -iname '*.sgf' | xargs -P4 -I{} gogui-thumbnailer -size 256 {} {}.png

"""

import sys
sys.path.insert(0, '.')

import os

from collections import Counter, defaultdict

from absl import app
from tqdm import tqdm
from sgfmill import sgf, sgf_moves

import oneoff_utils


ADJACENT_FOR_LADDER = 10


def subtract(a, b):
    return (b[0] - a[0], b[1] - a[1])

def manhattanDistance(a, b):
    d = subtract(a, b)
    return abs(d[0]) + abs(d[1])

def isLadderIsh(game_path):
    with open(game_path) as sgf_file:
        game_data = sgf_file.read().encode('utf-8')

    g = sgf.Sgf_game.from_bytes(game_data)
    _, moves = sgf_moves.get_setup_and_moves(g)

    mostAdjacent = 0
    mostStart = 0

    # colorStart, moveStart
    cS, mS = -1, (-2, -2)
    adjacent = 0

    # colorLast, moveLast
    cL, mL = cS, mS

    for i, (c, m) in enumerate(moves, 1):
        if m is None:
            continue

        newColor = c != cL
        dS = subtract(mS, m)
        dL = manhattanDistance(mL, m)

        diagonalDistance = abs(abs(dS[0]) - abs(dS[1]))
        isLadder = ((c == cS and diagonalDistance <= 1 and dL == 2) or
                    (c != cS and diagonalDistance <= 2 and dL == 1))

        if newColor and isLadder:
            adjacent += 1
            if adjacent > mostAdjacent:
                mostAdjacent = adjacent
                mostStart = i - adjacent
        else:
            cS  = c
            mS = m
            adjacent = 0

        cL, mL = c, m

    if mostAdjacent >= ADJACENT_FOR_LADDER:
        return (mostAdjacent, mostStart)
    return None

def main(unused_argv):
    assert len(unused_argv) == 2, unused_argv
    sgf_dir = unused_argv[1]
    sgf_dir += '/' * (sgf_dir[-1] != '/')

    sgf_files = oneoff_utils.find_and_filter_sgf_files(sgf_dir)

    per_folder = defaultdict(lambda: [0,0])
    lengths = Counter()
    ladders = []
    for name in tqdm(sorted(sgf_files)):
        folder = os.path.dirname(name[len(sgf_dir):])
        per_folder[folder][0] += 1

        ladderAt = isLadderIsh(name)
        if ladderAt:
            ladders.append((name, ladderAt))
            lengths[ladderAt[0]] += 1
            per_folder[folder][1] += 1
            print("Ladderish({}): {}, {}".format(len(ladders), ladderAt, name))

        from shutil import copyfile
        replace = '/ladder/' + ('yes' if ladderAt else 'no') + '/'
        copyfile(name, name.replace('/ladder/', replace))

    print()

    stars_per = max(max(lengths.values()) / 50, 1)
    for length, c in sorted(lengths.items()):
        print("{:2d} ({:<4d}): {}".format(length, c, "*" * int(c / stars_per)))
    print()

    if len(per_folder) > 1:
        for folder, counts in sorted(per_folder.items()):
            if not folder.endswith('/'): folder += "/"
            print("{}/{} ({:.1f}%) {}".format(
                counts[1], counts[0], 100 * counts[1] / counts[0], folder))

    count = len(ladders)
    print("{:3d}/{:<4d} ({:.1f}%) overall".format(
         count, len(sgf_files), 100 * count / len(sgf_files)))


if __name__ == "__main__":
    app.run(main)
