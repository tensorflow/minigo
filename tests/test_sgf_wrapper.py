import go
from sgf_wrapper import replay_sgf, translate_sgf_move, make_sgf
import unittest

from utils import parse_kgs_coords as pc
from test_utils import GoPositionTestCase, load_board

JAPANESE_HANDICAP_SGF = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Japanese]SZ[9]HA[2]KM[5.50]PW[test_white]PB[test_black]AB[gc][cg];W[ee];B[dg])"

CHINESE_HANDICAP_SGF = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Chinese]SZ[9]HA[2]KM[5.50]PW[test_white]PB[test_black]RE[B+39.50];B[gc];B[cg];W[ee];B[gg];W[eg];B[ge];W[ce];B[ec];W[cc];B[dd];W[de];B[cd];W[bd];B[bc];W[bb];B[be];W[ac];B[bf];W[dh];B[ch];W[ci];B[bi];W[di];B[ah];W[gh];B[hh];W[fh];B[hg];W[gi];B[fg];W[dg];B[ei];W[cf];B[ef];W[ff];B[fe];W[bg];B[bh];W[af];B[ag];W[ae];B[ad];W[ae];B[ed];W[db];B[df];W[eb];B[fb];W[ea];B[fa])"

NO_HANDICAP_SGF = "(;CA[UTF-8]SZ[9]PB[Murakawa Daisuke]PW[Iyama Yuta]KM[6.5]HA[0]RE[W+1.5]GM[1];B[fd];W[cf];B[eg];W[dd];B[dc];W[cc];B[de];W[cd];B[ed];W[he];B[ce];W[be];B[df];W[bf];B[hd];W[ge];B[gd];W[gg];B[db];W[cb];B[cg];W[bg];B[gh];W[fh];B[hh];W[fg];B[eh];W[ei];B[di];W[fi];B[hg];W[dh];B[ch];W[ci];B[bh];W[ff];B[fe];W[hf];B[id];W[bi];B[ah];W[ef];B[dg];W[ee];B[di];W[ig];B[ai];W[ih];B[fb];W[hi];B[ag];W[ab];B[bd];W[bc];B[ae];W[ad];B[af];W[bd];B[ca];W[ba];B[da];W[ie])"

class TestSgfGeneration(GoPositionTestCase):
    def test_translate_sgf_move(self):
        self.assertEqual(
            ";B[db]",
            translate_sgf_move(go.PlayerMove(go.BLACK, (1, 3))))
        self.assertEqual(
            ";W[aa]",
            translate_sgf_move(go.PlayerMove(go.WHITE, (0, 0))))
        self.assertEqual(
            ";W[]",
            translate_sgf_move(go.PlayerMove(go.WHITE, None)))

    def test_make_sgf(self):
        all_positions = list(replay_sgf(NO_HANDICAP_SGF))
        last_position, _, metadata = all_positions[-1]
        back_to_sgf = make_sgf(
            last_position.recent,
            last_position.score(),
            boardsize=metadata.board_size,
            komi=last_position.komi,
        )
        reconstructed_positions = list(replay_sgf(back_to_sgf))
        last_position2, _, _ = reconstructed_positions[-1]

        self.assertEqualPositions(last_position, last_position2)


class TestSgfWrapper(GoPositionTestCase):
    def test_sgf_props(self):
        sgf_replayer = replay_sgf(CHINESE_HANDICAP_SGF)
        initial = next(sgf_replayer)
        self.assertEqual(initial.metadata.result, 'B+39.50')
        self.assertEqual(initial.metadata.board_size, 9)
        self.assertEqual(initial.position.komi, 5.5)

    def test_japanese_handicap_handling(self):
        intermediate_board = load_board('''
            .........
            .........
            ......X..
            .........
            ....O....
            .........
            ..X......
            .........
            .........
        ''')
        intermediate_position = go.Position(
            intermediate_board,
            n=1,
            komi=5.5,
            caps=(0, 0),
            recent=(go.PlayerMove(go.WHITE, pc('E5')),),
            to_play=go.BLACK,
        )
        final_board = load_board('''
            .........
            .........
            ......X..
            .........
            ....O....
            .........
            ..XX.....
            .........
            .........
        ''')
        final_position = go.Position(
            final_board,
            n=2,
            komi=5.5,
            caps=(0, 0),
            recent=(go.PlayerMove(go.WHITE, pc('E5')),
                    go.PlayerMove(go.BLACK, pc('D3')),),
            to_play=go.WHITE,
        )

        positions_w_context = list(replay_sgf(JAPANESE_HANDICAP_SGF))
        self.assertEqualPositions(intermediate_position, positions_w_context[1].position)
        self.assertEqualPositions(final_position, positions_w_context[-1].position)

    def test_chinese_handicap_handling(self):
        intermediate_board = load_board('''
            .........
            .........
            ......X..
            .........
            .........
            .........
            .........
            .........
            .........
        ''')
        intermediate_position = go.Position(
            intermediate_board,
            n=1,
            komi=5.5,
            caps=(0, 0),
            recent=(go.PlayerMove(go.BLACK, pc('G7')),),
            to_play=go.BLACK,
        )
        final_board = load_board('''
            ....OX...
            .O.OOX...
            O.O.X.X..
            .OXXX....
            OX...XX..
            .X.XXO...
            X.XOOXXX.
            XXXO.OOX.
            .XOOX.O..
        ''')
        final_position = go.Position(
            final_board,
            n=50,
            komi=5.5,
            caps=(7, 2),
            ko=None,
            recent=(go.PlayerMove(go.WHITE, pc('E9')),
                    go.PlayerMove(go.BLACK, pc('F9')),),
            to_play=go.WHITE
        )
        positions_w_context = list(replay_sgf(CHINESE_HANDICAP_SGF))
        self.assertEqualPositions(intermediate_position, positions_w_context[1].position)
        self.assertEqual(positions_w_context[1].next_move, pc('C3'))
        self.assertEqualPositions(final_position, positions_w_context[-1].position)
        self.assertFalse(positions_w_context[-1].is_usable())
        self.assertTrue(positions_w_context[-2].is_usable())

class TestPositionReplay(GoPositionTestCase):
    def test_replay_position(self):
        sgf_positions = list(replay_sgf(NO_HANDICAP_SGF))
        initial = sgf_positions[0]
        self.assertEqual(initial.metadata.result, 'W+1.5')
        self.assertEqual(initial.metadata.board_size, 9)
        self.assertEqual(initial.position.komi, 6.5)

        final = sgf_positions[-1].position

        # sanity check to ensure we're working with the right position
        final_board = load_board('''
            .OXX.....
            O.OX.X...
            .OOX.....
            OOOOXXXXX
            XOXXOXOOO
            XOOXOO.O.
            XOXXXOOXO
            XXX.XOXXO
            X..XOO.O.
        ''')
        expected_final_position = go.Position(
            final_board,
            n=62,
            komi=6.5,
            caps=(3, 2),
            ko=None,
            recent=tuple(),
            to_play=go.BLACK
        )
        self.assertEqualPositions(expected_final_position, final)
        self.assertEqual(final.n, len(final.recent))

        replayed_positions = list(go.replay_position(final))
        for sgf_pos, replay_pos in zip(sgf_positions, replayed_positions):
            self.assertEqualPositions(sgf_pos.position, replay_pos.position)

if __name__ == '__main__':
    unittest.main()
