import unittest
import numpy

import coords
import go
import test_utils

class TestCoords(test_utils.MiniGoUnitTest):
    def test_upperleft(self):
        self.assertEqual(coords.parse_sgf_coords('aa'), (0, 0))
        self.assertEqual(coords.unflatten_coords(0), (0, 0))
        self.assertEqual(coords.parse_kgs_coords('A9'), (0, 0))
        self.assertEqual(coords.parse_pygtp_coords((1,9)), (0, 0))

        self.assertEqual(coords.unparse_sgf_coords((0, 0)), 'aa')
        self.assertEqual(coords.flatten_coords((0, 0)), 0)
        self.assertEqual(coords.to_human_coord((0, 0)), 'A9')
        self.assertEqual(coords.unparse_pygtp_coords((0, 0)), (1, 9))

    def test_topleft(self):
        self.assertEqual(coords.parse_sgf_coords('ia'), (0, 8))
        self.assertEqual(coords.unflatten_coords(8), (0, 8))
        self.assertEqual(coords.parse_kgs_coords('J9'), (0, 8))
        self.assertEqual(coords.parse_pygtp_coords((9,9)), (0, 8))

        self.assertEqual(coords.unparse_sgf_coords((0, 8)), 'ia')
        self.assertEqual(coords.flatten_coords((0, 8)), 8)
        self.assertEqual(coords.to_human_coord((0, 8)), 'J9')
        self.assertEqual(coords.unparse_pygtp_coords((0, 8)), (9, 9))

    def test_pass(self):
        self.assertEqual(coords.parse_sgf_coords(''), None)
        self.assertEqual(coords.unflatten_coords(81), None)
        self.assertEqual(coords.parse_kgs_coords('pass'), None)
        self.assertEqual(coords.parse_pygtp_coords((0,0)), None)

        self.assertEqual(coords.unparse_sgf_coords(None), '')
        self.assertEqual(coords.flatten_coords(None), 81)
        self.assertEqual(coords.to_human_coord(None), 'pass')
        self.assertEqual(coords.unparse_pygtp_coords(None), (0, 0))

    def test_parsing_9x9(self):
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
        self.assertEqual(coords.flatten_coords((0, 0)), 0)
        self.assertEqual(coords.flatten_coords((0, 3)), 3)
        self.assertEqual(coords.flatten_coords((3, 0)), 27)
        self.assertEqual(coords.unflatten_coords(27), (3, 0))
        self.assertEqual(coords.unflatten_coords(10), (1, 1))
        self.assertEqual(coords.unflatten_coords(80), (8, 8))
        self.assertEqual(coords.flatten_coords(coords.unflatten_coords(10)), 10)
        self.assertEqual(coords.unflatten_coords(coords.flatten_coords((5, 4))), (5, 4))

    def test_unflatten_coords_ndindex_equivalence(self):
        ndindices = list(numpy.ndindex(go.N, go.N))
        flat_coords = list(range(go.N * go.N))
        self.assertEqual(list(map(coords.unflatten_coords, flat_coords)), ndindices)

