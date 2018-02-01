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

"""Logic for dealing with coordinates.

This introduces some helpers and terminology that are used throughout MiniGo.

MiniGo Coordinate: This is a tuple of the form (row, column) that is indexed
    starting out at (0, 0) from the upper-left.
Flattened Coordinate: this is a number ranging from 0 - N^2 (so N^2+1
    possible values). The extra value N^2 is used to mark a 'pass' move.
SGF Coordinate: Coordinate used for SGF serialization format. Coordinates use
    two-letter pairs having the form (column, row) indexed from the upper-left
    where 0,0 == 'aa'.
KGS Coordinate: Human-readable coordinate string indexed from bottom left, with
    the first character a capital letter and the second a number from 1-19.
PYGTP Coordinate: Tuple coordinate indexed starting at 1,1 from bottom-left
    in the format (column, row)

So, for a 19x19,

Coord Type      upper_left      upper_right     pass
-------------------------------------------------------
minigo coord    (0, 0)          (0, 18)         None
flat            0               18              361
SGF             'aa'            'sa'            ''
KGS             'A19'           'T19'           'pass'
pygtp           (1, 19)         (19, 19)        (0, 0)
"""

import gtp

import go

KGS_COLUMNS = 'ABCDEFGHJKLMNOPQRST'
SGF_COLUMNS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def sgf_to_flat(sgf):
    """Transforms an SGF coordinate directly into a flattened coordinate."""
    return flatten_coords(parse_sgf_coords(sgf))


def kgs_to_flat(sgf):
    """Transforms a KGS coordinate directly into a flattened coordinate."""
    return flatten_coords(parse_kgs_coords(sgf))


def flatten_coords(coord):
    """Flattens a coordinate tuple from a MiniGo coordinate"""
    if coord is None:
        return go.N * go.N
    return go.N * coord[0] + coord[1]


def unflatten_coords(flat):
    """Unflattens a flattened coordinate into a Minigo coordinate"""
    if flat == go.N * go.N:
        return None
    return divmod(flat, go.N)


def parse_sgf_coords(sgfc):
    """Transform a SGF coordinate into a coordinate-tuple"""
    if sgfc is None or sgfc == '':
        return None
    return SGF_COLUMNS.index(sgfc[1]), SGF_COLUMNS.index(sgfc[0])


def unparse_sgf_coords(coord):
    """Turns a MiniGo coordinate tuple into a SGF coordinate."""
    if coord is None:
        return ''
    return SGF_COLUMNS[coord[1]] + SGF_COLUMNS[coord[0]]


def parse_kgs_coords(kgsc):
    """Interprets KGS coordinates returning a minigo coordinate tuple."""
    if kgsc == 'pass':
        return None
    kgsc = kgsc.upper()
    col = KGS_COLUMNS.index(kgsc[0])
    row_from_bottom = int(kgsc[1:]) - 1
    return go.N - row_from_bottom - 1, col


def to_human_coord(coord):
    """Converts from a MiniGo coord to a human readable string.

    This is equivalent to a KGS coordinate.
    """
    if coord is None:
        return "pass"
    y, x = coord
    return "{}{}".format("ABCDEFGHJKLMNOPQRSTYVWYZ"[x], go.N-y)


def parse_pygtp_coords(vertex):
    """Transforms a GTP coordinate into a standard MiniGo coordinate.

    GTP has a notion of both a Pass and a Resign, both of which
    are mapped to None so the conversion is not precisely bijective.
    """
    if vertex in (gtp.PASS, gtp.RESIGN):
        return None
    return go.N - vertex[1], vertex[0] - 1


def unparse_pygtp_coords(coord):
    """Transforms a MiniGo Coordinate back into a GTP coordinate."""
    if coord is None:
        return gtp.PASS
    return coord[1] + 1, go.N - coord[0]
