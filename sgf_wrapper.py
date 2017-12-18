'''
Code to extract a series of positions + their next moves from an SGF.

Most of the complexity here is dealing with two features of SGF:
- Stones can be added via "play move" or "add move", the latter being used
  to configure L+D puzzles, but also for initial handicap placement.
- Plays don't necessarily alternate colors; they can be repeated B or W moves
  This feature is used to handle free handicap placement.
'''
from collections import namedtuple
import numpy as np
import itertools

import go
from go import Position, GameMetadata, PositionWithContext
from utils import parse_sgf_coords as pc, unparse_sgf_coords as upc
import sgf

SGF_TEMPLATE = '''(;GM[1]FF[4]CA[UTF-8]AP[MuGo_sgfgenerator]RU[{ruleset}]
SZ[{boardsize}]KM[{komi}]PW[{white_name}]PB[{black_name}]RE[{result}]
{game_moves})'''

PROGRAM_IDENTIFIER = "MuGo"

def translate_sgf_move_qs(player_move,q):
  return "{move}C[{q:.4f}]".format(
      move=translate_sgf_move(player_move), q=q)

def translate_sgf_move(player_move, comment):
    if player_move.color not in (go.BLACK, go.WHITE):
        raise ValueError("Can't translate color %s to sgf" % player_move.color)
    coords = upc(player_move.move)
    color = 'B' if player_move.color == go.BLACK else 'W'
    if comment is not None:
        comment = comment.replace(']', r'\]')
        comment_node = "C[{}]".format(comment)
    else:
        comment_node = ""
    return ";{color}[{coords}]{comment_node}".format(
        color=color, coords=coords, comment_node=comment_node)

def make_sgf(
    move_history,
    result_string,
    ruleset="Chinese",
    boardsize=go.N,
    komi=7.5,
    white_name=PROGRAM_IDENTIFIER,
    black_name=PROGRAM_IDENTIFIER,
    comments=[]
    ):
    '''Turn a game into SGF.

    Doesn't handle handicap games or positions with incomplete history.

    Args:
        move_history: iterable of PlayerMoves
        result_string: "B+R", "W+0.5", etc.
        comments: iterable of string/None. Will be zipped with move_history.
    '''
    game_moves = ''.join(translate_sgf_move(*z)
        for z in itertools.zip_longest(move_history, comments))
    result = result_string
    return SGF_TEMPLATE.format(**locals()) 

def sgf_prop(value_list):
    'Converts raw sgf library output to sensible value'
    if value_list is None:
        return None
    if len(value_list) == 1:
        return value_list[0]
    else:
        return value_list

def sgf_prop_get(props, key, default):
    return sgf_prop(props.get(key, default))

def handle_node(pos, node):
    'A node can either add B+W stones, play as B, or play as W.'
    props = node.properties
    black_stones_added = [pc(coords) for coords in props.get('AB', [])]
    white_stones_added = [pc(coords) for coords in props.get('AW', [])]
    if black_stones_added or white_stones_added:
        return add_stones(pos, black_stones_added, white_stones_added)
    # If B/W props are not present, then there is no move. But if it is present and equal to the empty string, then the move was a pass.
    elif 'B' in props:
        black_move = pc(props.get('B', [''])[0])
        return pos.play_move(black_move, color=go.BLACK)
    elif 'W' in props:
        white_move = pc(props.get('W', [''])[0])
        return pos.play_move(white_move, color=go.WHITE)
    else:
        return pos

def add_stones(pos, black_stones_added, white_stones_added):
    working_board = np.copy(pos.board)
    go.place_stones(working_board, go.BLACK, black_stones_added)
    go.place_stones(working_board, go.WHITE, white_stones_added)
    new_position = Position(board=working_board, n=pos.n, komi=pos.komi, caps=pos.caps, ko=pos.ko, recent=pos.recent, to_play=pos.to_play)
    return new_position

def get_next_move(node):
    if not node.next:
        return None
    props = node.next.properties
    if 'W' in props:
        return pc(props['W'][0])
    else:
        return pc(props['B'][0])

def maybe_correct_next(pos, next_node):
    if next_node is None:
        return
    if (('B' in next_node.properties and not pos.to_play == go.BLACK) or
        ('W' in next_node.properties and not pos.to_play == go.WHITE)):
        pos.flip_playerturn(mutate=True)

def replay_sgf(sgf_contents):
    '''
    Wrapper for sgf files, exposing contents as position_w_context instances
    with open(filename) as f:
        for position_w_context in replay_sgf(f.read()):
            print(position_w_context.position)
    '''
    collection = sgf.parse(sgf_contents)
    game = collection.children[0]
    props = game.root.properties
    assert int(sgf_prop(props.get('GM', ['1']))) == 1, "Not a Go SGF!"

    komi = 0
    if props.get('KM') != None:
        komi = float(sgf_prop(props.get('KM')))
    metadata = GameMetadata(
        result=sgf_prop(props.get('RE')),
        handicap=int(sgf_prop(props.get('HA', [0]))),
        board_size=int(sgf_prop(props.get('SZ'))))
    go.set_board_size(metadata.board_size)

    pos = Position(komi=komi)
    current_node = game.root
    while pos is not None and current_node is not None:
        pos = handle_node(pos, current_node)
        maybe_correct_next(pos, current_node.next)
        next_move = get_next_move(current_node)
        yield PositionWithContext(pos, next_move, metadata)
        current_node = current_node.next 
