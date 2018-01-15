import unittest
import numpy

import coords
import go


class TestCoords(unittest.TestCase):
    def test_upperleft(self):
        go.set_board_size(19)
        self.assertEqual(coords.parse_sgf_coords('aa'), (0, 0))
        self.assertEqual(coords.unflatten_coords(0), (0, 0))
        self.assertEqual(coords.parse_kgs_coords('A19'), (0, 0))
        self.assertEqual(coords.parse_pygtp_coords((1,19)), (0, 0))

        self.assertEqual(coords.unparse_sgf_coords((0, 0)), 'aa')
        self.assertEqual(coords.flatten_coords((0, 0)), 0)
        self.assertEqual(coords.to_human_coord((0, 0)), 'A19')
        self.assertEqual(coords.unparse_pygtp_coords((0, 0)), (1, 19))

    def test_topleft(self):
        go.set_board_size(19)
        self.assertEqual(coords.parse_sgf_coords('sa'), (0, 18))
        self.assertEqual(coords.unflatten_coords(18), (0, 18))
        self.assertEqual(coords.parse_kgs_coords('T19'), (0, 18))
        self.assertEqual(coords.parse_pygtp_coords((19,19)), (0, 18))

        self.assertEqual(coords.unparse_sgf_coords((0, 18)), 'sa')
        self.assertEqual(coords.flatten_coords((0, 18)), 18)
        self.assertEqual(coords.to_human_coord((0, 18)), 'T19')
        self.assertEqual(coords.unparse_pygtp_coords((0, 18)), (19, 19))

    def test_pass(self):
        go.set_board_size(19)
        self.assertEqual(coords.parse_sgf_coords(''), None)
        self.assertEqual(coords.unflatten_coords(361), None)
        self.assertEqual(coords.parse_kgs_coords('pass'), None)
        self.assertEqual(coords.parse_pygtp_coords((0,0)), None)

        self.assertEqual(coords.unparse_sgf_coords(None), '')
        self.assertEqual(coords.flatten_coords(None), 361)
        self.assertEqual(coords.to_human_coord(None), 'pass')
        self.assertEqual(coords.unparse_pygtp_coords(None), (0, 0))

    def test_parsing_9x9(self):
        go.set_board_size(9)
        self.assertEqual(coords.parse_sgf_coords('aa'), (0, 0))
        self.assertEqual(coords.parse_sgf_coords('ac'), (2, 0))
        self.assertEqual(coords.parse_sgf_coords('ca'), (0, 2))
        self.assertEqual(coords.parse_sgf_coords(''), None)
        self.assertEqual(coords.unparse_sgf_coords(None), '')
        self.assertEqual(
            'aa',
            coords.unparse_sgf_coords(coords.parse_sgf_coords('aa')))
        self.assertEqual(
            'sa',
            coords.unparse_sgf_coords(coords.parse_sgf_coords('sa')))
        self.assertEqual(
            (1, 17),
            coords.parse_sgf_coords(coords.unparse_sgf_coords((1, 17))))
        self.assertEqual(coords.parse_kgs_coords('A1'), (8, 0))
        self.assertEqual(coords.parse_kgs_coords('A9'), (0, 0))
        self.assertEqual(coords.parse_kgs_coords('C2'), (7, 2))
        self.assertEqual(coords.parse_kgs_coords('J2'), (7, 8))
        self.assertEqual(coords.parse_pygtp_coords((1, 1)), (8, 0))
        self.assertEqual(coords.parse_pygtp_coords((1, 9)), (0, 0))
        self.assertEqual(coords.parse_pygtp_coords((3, 2)), (7, 2))
        self.assertEqual(coords.unparse_pygtp_coords((8, 0)), (1, 1))
        self.assertEqual(coords.unparse_pygtp_coords((0, 0)), (1, 9))
        self.assertEqual(coords.unparse_pygtp_coords((7, 2)), (3, 2))

        self.assertEqual(coords.to_human_coord((0,8)), 'J9')
        self.assertEqual(coords.to_human_coord((8,0)), 'A1')

    def test_flatten(self):
        go.set_board_size(9)
        self.assertEqual(coords.flatten_coords((0, 0)), 0)
        self.assertEqual(coords.flatten_coords((0, 3)), 3)
        self.assertEqual(coords.flatten_coords((3, 0)), 27)
        self.assertEqual(coords.unflatten_coords(27), (3, 0))
        self.assertEqual(coords.unflatten_coords(10), (1, 1))
        self.assertEqual(coords.unflatten_coords(80), (8, 8))
        self.assertEqual(coords.flatten_coords(coords.unflatten_coords(10)), 10)
        self.assertEqual(coords.unflatten_coords(coords.flatten_coords((5, 4))), (5, 4))

    def test_unflatten_coords_ndindex_equivalence(self):
        go.set_board_size(9)
        ndindices = list(numpy.ndindex(go.N, go.N))
        flat_coords = list(range(go.N * go.N))
        self.assertEqual(list(map(coords.unflatten_coords, flat_coords)), ndindices)

