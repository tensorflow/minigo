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

import numpy

import coords
import go
from tests import test_utils


class TestCoords(test_utils.MiniGoUnitTest):
    def test_upperleft(self):
        self.assertEqual((0, 0), coords.from_sgf('aa'))
        self.assertEqual((0, 0), coords.from_flat(0))
        self.assertEqual((0, 0), coords.from_kgs('A9'))

        self.assertEqual('aa', coords.to_sgf((0, 0)))
        self.assertEqual(0, coords.to_flat((0, 0)))
        self.assertEqual('A9', coords.to_kgs((0, 0)))

    def test_topleft(self):
        self.assertEqual((0, 8), coords.from_sgf('ia'))
        self.assertEqual((0, 8), coords.from_flat(8))
        self.assertEqual((0, 8), coords.from_kgs('J9'))

        self.assertEqual('ia', coords.to_sgf((0, 8)))
        self.assertEqual(8, coords.to_flat((0, 8)))
        self.assertEqual('J9', coords.to_kgs((0, 8)))

    def test_pass(self):
        self.assertEqual(None, coords.from_sgf(''))
        self.assertEqual(None, coords.from_flat(81))
        self.assertEqual(None, coords.from_kgs('pass'))

        self.assertEqual('', coords.to_sgf(None))
        self.assertEqual(81, coords.to_flat(None))
        self.assertEqual('pass', coords.to_kgs(None))

    def test_parsing_9x9(self):
        self.assertEqual((0, 0), coords.from_sgf('aa'))
        self.assertEqual((2, 0), coords.from_sgf('ac'))
        self.assertEqual((0, 2), coords.from_sgf('ca'))
        self.assertEqual(None, coords.from_sgf(''))
        self.assertEqual('', coords.to_sgf(None))
        self.assertEqual('aa', coords.to_sgf(coords.from_sgf('aa')))
        self.assertEqual('sa', coords.to_sgf(coords.from_sgf('sa')))
        self.assertEqual((1, 17), coords.from_sgf(coords.to_sgf((1, 17))))
        self.assertEqual((8, 0), coords.from_kgs('A1'))
        self.assertEqual((0, 0), coords.from_kgs('A9'))
        self.assertEqual((7, 2), coords.from_kgs('C2'))
        self.assertEqual((7, 8), coords.from_kgs('J2'))

        self.assertEqual('J9', coords.to_kgs((0, 8)))
        self.assertEqual('A1', coords.to_kgs((8, 0)))

    def test_flatten(self):
        self.assertEqual(0, coords.to_flat((0, 0)))
        self.assertEqual(3, coords.to_flat((0, 3)))
        self.assertEqual(27, coords.to_flat((3, 0)))
        self.assertEqual((3, 0), coords.from_flat(27))
        self.assertEqual((1, 1), coords.from_flat(10))
        self.assertEqual((8, 8), coords.from_flat(80))
        self.assertEqual(10, coords.to_flat(coords.from_flat(10)))
        self.assertEqual((5, 4), coords.from_flat(coords.to_flat((5, 4))))

    def test_from_flat_ndindex_equivalence(self):
        ndindices = list(numpy.ndindex(go.N, go.N))
        flat_coords = list(range(go.N * go.N))
        self.assertEqual(ndindices, list(map(coords.from_flat, flat_coords)))
