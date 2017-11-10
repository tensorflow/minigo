import itertools
import numpy as np

import features
from features import apply_symmetry_feat as apply_f
from features import apply_symmetry_pi as apply_p
import go
import utils
from test_utils import load_board, GoPositionTestCase

go.set_board_size(9)
EMPTY_ROW = '.' * go.N + '\n'
TEST_BOARD = load_board('''
.X.....OO
X........
XXXXXXXXX
''' + EMPTY_ROW * 6)

TEST_POSITION = go.Position(
    board=TEST_BOARD,
    n=3,
    komi=6.5,
    caps=(1,2),
    ko=None,
    recent=(go.PlayerMove(go.BLACK, (0, 1)),
            go.PlayerMove(go.WHITE, (0, 8)),
            go.PlayerMove(go.BLACK, (1, 0))),
    to_play=go.BLACK,
)

TEST_BOARD2 = load_board('''
.XOXXOO..
XO.OXOX..
XXO..X...
''' + EMPTY_ROW * 6)

TEST_POSITION2 = go.Position(
    board=TEST_BOARD2,
    n=0,
    komi=6.5,
    caps=(0, 0),
    ko=None,
    recent=tuple(),
    to_play=go.BLACK,
)


TEST_POSITION3= go.Position()
for coord in ((0, 0), (0, 1), (0, 2), (0, 3), (1, 1)):
    TEST_POSITION3.play_move(coord, mutate=True)
# resulting position should look like this:
# X.XO.....
# .X.......
# .........

class TestFeatureExtraction(GoPositionTestCase):
    def test_stone_features(self):
        f = features.stone_features(TEST_POSITION3)
        self.assertEqual(TEST_POSITION3.to_play, go.WHITE)
        self.assertEqual(f.shape, (9, 9, 16))
        self.assertEqualNPArray(f[:, :, 0], load_board('''
            ...X.....
            .........''' + EMPTY_ROW * 7))

        self.assertEqualNPArray(f[:, :, 1], load_board('''
            X.X......
            .X.......''' + EMPTY_ROW * 7))

        self.assertEqualNPArray(f[:, :, 2], load_board('''
            .X.X.....
            .........''' + EMPTY_ROW * 7))

        self.assertEqualNPArray(f[:, :, 3], load_board('''
            X.X......
            .........''' + EMPTY_ROW * 7))

        self.assertEqualNPArray(f[:, :, 4], load_board('''
            .X.......
            .........''' + EMPTY_ROW * 7))

        self.assertEqualNPArray(f[:, :, 5], load_board('''
            X.X......
            .........''' + EMPTY_ROW * 7))

        for i in range(10, 16):
            self.assertEqualNPArray(f[:, :, i], np.zeros([go.N, go.N]))


    def test_stone_color_feature(self):
        f = features.stone_color_feature(TEST_POSITION)
        self.assertEqual(f.shape, (9, 9, 3))
        # plane 0 is B
        self.assertEqual(f[0, 1, 0], 1)
        self.assertEqual(f[0, 1, 1], 0)
        # plane 1 is W
        self.assertEqual(f[0, 8, 1], 1)
        self.assertEqual(f[0, 8, 0], 0)
        # plane 2 is empty
        self.assertEqual(f[0, 5, 2], 1)
        self.assertEqual(f[0, 5, 1], 0)

    def test_liberty_feature(self):
        f = features.liberty_feature(TEST_POSITION)
        self.assertEqual(f.shape, (9, 9, features.liberty_feature.planes))

        self.assertEqual(f[0, 0, 0], 0)
        # the stone at 0, 1 has 3 liberties.
        self.assertEqual(f[0, 1, 2], 1)
        self.assertEqual(f[0, 1, 4], 0)
        # the group at 0, 7 has 3 liberties
        self.assertEqual(f[0, 7, 2], 1)
        self.assertEqual(f[0, 8, 2], 1)
        # the group at 1, 0 has 18 liberties
        self.assertEqual(f[1, 0, 7], 1)

    def test_recent_moves_feature(self):
        f = features.recent_move_feature(TEST_POSITION)
        self.assertEqual(f.shape, (9, 9, features.recent_move_feature.planes))
        # most recent move at (1, 0)
        self.assertEqual(f[1, 0, 0], 1)
        self.assertEqual(f[1, 0, 3], 0)
        # second most recent move at (0, 8)
        self.assertEqual(f[0, 8, 1], 1)
        self.assertEqual(f[0, 8, 0], 0)
        # third most recent move at (0, 1)
        self.assertEqual(f[0, 1, 2], 1)
        # no more older moves
        self.assertEqualNPArray(f[:, :, 3], np.zeros([9, 9]))
        self.assertEqualNPArray(f[:, :, features.recent_move_feature.planes - 1], np.zeros([9, 9]))

    def test_would_capture_feature(self):
        f = features.would_capture_feature(TEST_POSITION2)
        self.assertEqual(f.shape, (9, 9, features.would_capture_feature.planes))
        # move at (1, 2) would capture 2 stones
        self.assertEqual(f[1, 2, 1], 1)
        # move at (0, 0) should not capture stones because it's B's move.
        self.assertEqual(f[0, 0, 0], 0)
        # move at (0, 7) would capture 3 stones
        self.assertEqual(f[0, 7, 2], 1)
        self.assertEqual(f[0, 7, 1], 0)

class TestSymmetryOperations(GoPositionTestCase):
    def setUp(self):
        np.random.seed(1)
        self.feat = np.random.random([go.N, go.N, 3])
        self.pi = np.random.random([go.N ** 2 + 1])
        super().setUp()

    def test_inversions(self):
        for s in features.SYMMETRIES:
            with self.subTest(symmetry=s):
                self.assertEqualNPArray(self.feat,
                    apply_f(s, apply_f(features.invert_symmetry(s), self.feat)))
                self.assertEqualNPArray(self.feat,
                    apply_f(features.invert_symmetry(s), apply_f(s, self.feat)))

                self.assertEqualNPArray(self.pi,
                    apply_p(s, apply_p(features.invert_symmetry(s), self.pi)))
                self.assertEqualNPArray(self.pi,
                    apply_p(features.invert_symmetry(s), apply_p(s, self.pi)))

    def test_compositions(self):
        test_cases = [
            ('rot90', 'rot90', 'rot180'),
            ('rot90', 'rot180', 'rot270'),
            ('identity', 'rot90', 'rot90'),
            ('fliprot90', 'rot90', 'fliprot180'),
            ('rot90', 'rot270', 'identity'),
        ]
        for s1, s2, composed in test_cases:
            with self.subTest(s1=s1, s2=s2, composed=composed):
                self.assertEqualNPArray(apply_f(composed, self.feat),
                    apply_f(s2, apply_f(s1, self.feat)))
                self.assertEqualNPArray(apply_p(composed, self.pi),
                    apply_p(s2, apply_p(s1, self.pi)))

    def test_uniqueness(self):
        all_symmetries_f = [
            apply_f(s, self.feat) for s in features.SYMMETRIES
        ]
        all_symmetries_pi = [
            apply_p(s, self.pi) for s in features.SYMMETRIES
        ]
        for f1, f2 in itertools.combinations(all_symmetries_f, 2):
            self.assertNotEqualNPArray(f1, f2)
        for pi1, pi2 in itertools.combinations(all_symmetries_pi, 2):
            self.assertNotEqualNPArray(pi1, pi2)

    def test_proper_move_transform(self):
        # Check that the reinterpretation of 362 = 19*19 + 1 during symmetry
        # application is consistent with utils.unflatten_coords
        move_array = np.arange(go.N ** 2 + 1)
        coord_array = np.zeros([go.N, go.N])
        for c in range(go.N ** 2):
            coord_array[utils.unflatten_coords(c)] = c
        for s in features.SYMMETRIES:
            with self.subTest(symmetry=s):
                transformed_moves = apply_p(s, move_array)
                transformed_board = apply_f(s, coord_array)
                for new_coord, old_coord in enumerate(transformed_moves[:-1]):
                    self.assertEqual(
                        old_coord,
                        transformed_board[utils.unflatten_coords(new_coord)])
