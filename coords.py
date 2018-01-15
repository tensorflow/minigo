"""Logic for dealing with coordinates.

This introduces some helpers and terminology that are used throughout MiniGo.

Coord/Coordinate: This is a tuple of the form (column, row) that's indexed from
    the upper-left.
Flattened Coordinate: this is a number ranging from 0 - N^2 (so N^2+1
    possible values). The extra value N^2 is used to mark a 'pass' move.
KGS Coordinate: Human-readable coordinate indexed from bottom left.
GTP Coordinate: Tuple coordinate indexed starting at 1,1 from top-left (r, c)
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
    """Flattens a coordinate tuple.

    Takes coordinates of the form (col, row) indexed from the upper-Left and
    returns a number between 0 and go-board-size^2-1 (i.e., N^2-1), unless the
    row/column is unspecified, in which case, N^2 is returned.

    Note here that None coordinates represent pass moves, so N^2 is a
    pass-move.
    """
    if c is None:
        return go.N * go.N
    return go.N * c[0] + c[1]

def unflatten_coords(f):
    """Unflattens a flattened coordinate into a tuple.

    Takes a flattened coordinate (integer) and returns a coordinate tuple.

    Note that a flattened coordinate of N^2 becomes None, to represent
    pass-moves.
    """
    if f == go.N * go.N:
        return None
    return divmod(f, go.N)

def parse_sgf_coords(s):
    """Transform a SGF coordinate into a coordinate-tuple

    SGF coordinates have the form '<letter><letter>', where aa is top left
    corner; sa (18, 1) is top right corner of a 19x19. The resulting coordinate 

    An SGF coordinate of '' is interpreted as a pass-move.
    """
    if s is None or s == '':
        return None
    return SGF_COLUMNS.index(s[1]), SGF_COLUMNS.index(s[0])

def unparse_sgf_coords(c):
    """Turns a coordinate tuple into a SGF coordinate."""
    if c is None:
        return ''
    return SGF_COLUMNS[c[1]] + SGF_COLUMNS[c[0]]

def parse_kgs_coords(s):
    """Interprets KGS coordinates returning a coordinate tuple (c, r).

    In KGS terminology, A1 is bottom left; A19 is top left. So, to transform to
    a standard coordinate tuple (c,r), we need to know the board size.
    """
    if s == 'pass':
        return None
    s = s.upper()
    col = KGS_COLUMNS.index(s[0])
    row_from_bottom = int(s[1:]) - 1
    return go.N - row_from_bottom - 1, col

def to_human_coord(coord):
    """From a MiniGo coord to a human readable string

    This is equivalent to a KGS coordinate.
    """
    if coord == None:
        return "pass"
    else:
        y, x = coord
        return "{}{}".format("ABCDEFGHJKLMNOPQRSTYVWYZ"[x], go.N-y) 

def parse_pygtp_coords(vertex):
    """Transform a GTP coordinate into a standard MiniGo coordinate (c, r)

    GTP coordinates are interpeted as (1, 1) being the bottom left and (1, 19)
    the top left. GTP has a notion of both a Pass and a Resign, both of which
    are mapped to None so the conversion is not precisely bijective.
    """
    if vertex in (gtp.PASS, gtp.RESIGN):
        return None
    return go.N - vertex[1], vertex[0] - 1

def unparse_pygtp_coords(c):
    """Transform a MiniGo Coordinate back into a GTP coordinate

    As in other coordinates, None is interpreted as a Pass.
    """
    if c is None:
        return gtp.PASS
    return c[1] + 1, go.N - c[0]
