import numpy as np
import random
import re
import time
import unittest

import go
import utils

go.set_board_size(9)

def load_board(string):
    reverse_map = {
        'X': go.BLACK,
        'O': go.WHITE,
        '.': go.EMPTY,
        '#': go.FILL,
        '*': go.KO,
        '?': go.UNKNOWN
    }

    string = re.sub(r'[^XO\.#]+', '', string)
    assert len(string) == go.N ** 2, "Board to load didn't have right dimensions"
    board = np.zeros([go.N, go.N], dtype=np.int8)
    for i, char in enumerate(string):
        np.ravel(board)[i] = reverse_map[char]
    return board

class TestUtils(unittest.TestCase):
    def test_parsing(self):
        self.assertEqual(utils.parse_sgf_coords('aa'), (0, 0))
        self.assertEqual(utils.parse_sgf_coords('ac'), (2, 0))
        self.assertEqual(utils.parse_sgf_coords('ca'), (0, 2))
        self.assertEqual(utils.parse_sgf_coords(''), None)
        self.assertEqual(utils.unparse_sgf_coords(None), '')
        self.assertEqual(
            'aa',
            utils.unparse_sgf_coords(utils.parse_sgf_coords('aa')))
        self.assertEqual(
            'sa',
            utils.unparse_sgf_coords(utils.parse_sgf_coords('sa')))
        self.assertEqual(
            (1, 17),
            utils.parse_sgf_coords(utils.unparse_sgf_coords((1, 17))))
        self.assertEqual(utils.parse_kgs_coords('A1'), (8, 0))
        self.assertEqual(utils.parse_kgs_coords('A9'), (0, 0))
        self.assertEqual(utils.parse_kgs_coords('C2'), (7, 2))
        self.assertEqual(utils.parse_kgs_coords('J2'), (7, 8))
        self.assertEqual(utils.parse_pygtp_coords((1, 1)), (8, 0))
        self.assertEqual(utils.parse_pygtp_coords((1, 9)), (0, 0))
        self.assertEqual(utils.parse_pygtp_coords((3, 2)), (7, 2))
        self.assertEqual(utils.unparse_pygtp_coords((8, 0)), (1, 1))
        self.assertEqual(utils.unparse_pygtp_coords((0, 0)), (1, 9))
        self.assertEqual(utils.unparse_pygtp_coords((7, 2)), (3, 2))

        self.assertEqual(utils.to_human_coord((0,8)), 'J9')
        self.assertEqual(utils.to_human_coord((8,0)), 'A1')

    def test_flatten(self):
        self.assertEqual(utils.flatten_coords((0, 0)), 0)
        self.assertEqual(utils.flatten_coords((0, 3)), 3)
        self.assertEqual(utils.flatten_coords((3, 0)), 27)
        self.assertEqual(utils.unflatten_coords(27), (3, 0))
        self.assertEqual(utils.unflatten_coords(10), (1, 1))
        self.assertEqual(utils.unflatten_coords(80), (8, 8))
        self.assertEqual(utils.flatten_coords(utils.unflatten_coords(10)), 10)
        self.assertEqual(utils.unflatten_coords(utils.flatten_coords((5, 4))), (5, 4))

    def test_unflatten_coords_ndindex_equivalence(self):
        ndindices = list(np.ndindex(go.N, go.N))
        flat_coords = list(range(go.N * go.N))
        self.assertEqual(list(map(utils.unflatten_coords, flat_coords)), ndindices)

    def test_shuffler(self):
        random.seed(1)
        dataset = (i for i in range(10))
        shuffled = list(utils.shuffler(
            dataset, pool_size=5, refill_threshold=0.8))
        self.assertEqual(len(shuffled), 10)
        self.assertNotEqual(shuffled, list(range(10)))

    def test_parse_game_result(self):
        self.assertEqual(utils.parse_game_result('B+3.5'), go.BLACK)
        self.assertEqual(utils.parse_game_result('W+T'), go.WHITE)
        self.assertEqual(utils.parse_game_result('Void'), 0)



class GoPositionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.start_time = time.time()

    @classmethod
    def tearDownClass(cls):
        print("\n%s.%s: %.3f seconds" % (cls.__module__, cls.__name__, time.time() - cls.start_time))

    def setUp(self):
        go.set_board_size(9)

    def assertEqualNPArray(self, array1, array2):
        if not np.all(array1 == array2):
            raise AssertionError("Arrays differed in one or more locations:\n%s\n%s" % (array1, array2))

    def assertNotEqualNPArray(self, array1, array2):
        if np.all(array1 == array2):
            raise AssertionError("Arrays were identical:\n%s" % array1)

    def assertEqualLibTracker(self, lib_tracker1, lib_tracker2):
        # A lib tracker may have differently numbered groups yet still
        # represent the same set of groups.
        # "Sort" the group_ids to ensure they are the same.
        def find_group_mapping(lib_tracker):
            current_gid = 0
            mapping = {}
            for group_id in lib_tracker.group_index.ravel().tolist():
                if group_id == go.MISSING_GROUP_ID:
                    continue
                if group_id not in mapping:
                    mapping[group_id] = current_gid
                    current_gid += 1
            return mapping

        lt1_mapping = find_group_mapping(lib_tracker1)
        lt2_mapping = find_group_mapping(lib_tracker2)

        remapped_group_index1 = [lt1_mapping.get(gid, go.MISSING_GROUP_ID) for gid in lib_tracker1.group_index.ravel().tolist()]
        remapped_group_index2 = [lt2_mapping.get(gid, go.MISSING_GROUP_ID) for gid in lib_tracker2.group_index.ravel().tolist()]
        self.assertEqual(remapped_group_index1, remapped_group_index2)

        remapped_groups1 = {lt1_mapping.get(gid): group for gid, group in lib_tracker1.groups.items()}
        remapped_groups2 = {lt2_mapping.get(gid): group for gid, group in lib_tracker2.groups.items()}
        self.assertEqual(remapped_groups1, remapped_groups2)

        self.assertEqualNPArray(lib_tracker1.liberty_cache, lib_tracker2.liberty_cache)

    def assertEqualPositions(self, pos1, pos2):
        self.assertEqualNPArray(pos1.board, pos2.board)
        self.assertEqualLibTracker(pos1.lib_tracker, pos2.lib_tracker)
        self.assertEqual(pos1.n, pos2.n)
        self.assertEqual(pos1.caps, pos2.caps)
        self.assertEqual(pos1.ko, pos2.ko)
        r_len = min(len(pos1.recent), len(pos2.recent))
        if r_len > 0: # if a position has no history, then don't bother testing
            self.assertEqual(pos1.recent[-r_len:], pos2.recent[-r_len:])
        self.assertEqual(pos1.to_play, pos2.to_play)
