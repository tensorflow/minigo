"""Logic for dealing with coordinates.

This introduces some helpers and terminology that are used throughout MiniGo.

MiniGo Coord/Coordinate: This is a tuple of the form (column, row) that's
    indexed from (0,0) from the upper-left.
Flattened Coordinate: this is a number ranging from 0 - N^2 (so N^2+1
    possible values). The extra value N^2 is used to mark a 'pass' move.
KGS Coordinate: Human-readable coordinate indexed from bottom left.
GTP Coordinate: Tuple coordinate indexed starting at 1,1 from top-left (r, c)

So, for a 19x19,

Coord Type      upper_left      upper_right     pass
-------------------------------------------------------
minigo coord    (0, 0)          (18, 0)         None
flat            0               342             361
KGS             'A19'           'T19'           'pass'
GTP             (1, 1)          (1,19)          'pass'
"""

import go
import gtp

KGS_COLUMNS = 'ABCDEFGHJKLMNOPQRST'
SGF_COLUMNS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def sgf_to_flat(sgf):
    """Transforms an SGF coordinate directly into a flattened coordinate."""
    return flatten_coords(parse_sgf_coords(sgf))

def kgs_to_flat(sgf):
    """Transforms a KGS coordinate directly into a flattened coordinate."""
    return flatten_coords(parse_kgs_coords(sgf))

def flatten_coords(c):
    """Flattens a coordinate tuple from a MiniGo Coordinate"""
    if c is None:
        return go.N * go.N
    return go.N * c[0] + c[1]

def unflatten_coords(f):
    """Unflattens a flattened coordinate into a Minigo Coordinate"""
    if f == go.N * go.N:
        return None
    return divmod(f, go.N)

def parse_sgf_coords(s):
    """Transform a SGF coordinate into a coordinate-tuple

    SGF coordinates have the form '<letter><letter>', where aa is top left
    corner; sa (18, 1) is top right corner of a 19x19.

    An SGF coordinate of '' is interpreted as a pass-move.
    """
    if s is None or s == '':
        return None
    return SGF_COLUMNS.index(s[1]), SGF_COLUMNS.index(s[0])

def unparse_sgf_coords(c):
    """Turns a MiniGo coordinate tuple into a SGF coordinate."""
    if c is None:
        return ''
    return SGF_COLUMNS[c[1]] + SGF_COLUMNS[c[0]]

def parse_kgs_coords(s):
    """Interprets KGS coordinates returning a coordinate tuple (c, r)."""
    if s == 'pass':
        return None
    s = s.upper()
    col = KGS_COLUMNS.index(s[0])
    row_from_bottom = int(s[1:]) - 1
    return go.N - row_from_bottom - 1, col

def to_human_coord(coord):
    """Converts from a MiniGo coord to a human readable string.

    This is equivalent to a KGS coordinate.
    """
    if coord == None:
        return "pass"
    else:
        y, x = coord
        return "{}{}".format("ABCDEFGHJKLMNOPQRSTYVWYZ"[x], go.N-y) 

def parse_pygtp_coords(vertex):
    """Transforms a GTP coordinate into a standard MiniGo coordinate (c, r)

    GTP has a notion of both a Pass and a Resign, both of which
    are mapped to None so the conversion is not precisely bijective.
    """
    if vertex in (gtp.PASS, gtp.RESIGN):
        return None
    return go.N - vertex[1], vertex[0] - 1

def unparse_pygtp_coords(c):
    """Transforms a MiniGo Coordinate back into a GTP coordinate."""
    if c is None:
        return gtp.PASS
    return c[1] + 1, go.N - c[0]
