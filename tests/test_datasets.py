import numpy as np
import os
from test_utils import GoPositionTestCase
import go
import load_data_sets
import random

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
TEMP_FILE_NAME = "dataset_unittest_tempfile"

class TestDataSetV2(GoPositionTestCase):
    def tearDown(self):
        if os.path.isfile(TEMP_FILE_NAME):
            os.remove(TEMP_FILE_NAME)

    def test_datasetv2_serialization(self):
        sgf_files = list(load_data_sets.find_sgf_files(TEST_DIR))
        positions_w_context = list(load_data_sets.get_positions_from_sgf(sgf_files[0]))

        randos = np.random.random(go.N**2 + 1).astype(np.float32)
        randos /= np.sum(randos)
        searches = np.array([randos] * len(positions_w_context), dtype=np.float32)

        results = np.ones(len(positions_w_context), dtype=np.int8)
        dataset = load_data_sets.DataSetV2.from_positions_w_context(positions_w_context,
                                                                    searches,
                                                                    results)
        dataset.write(TEMP_FILE_NAME)
        recovered = load_data_sets.DataSetV2.read(TEMP_FILE_NAME)
        self.assertEqual(dataset.is_test, recovered.is_test)
        self.assertEqual(dataset.data_size, recovered.data_size)
        self.assertEqual(dataset.board_size, recovered.board_size)
        self.assertEqual(dataset.input_planes, recovered.input_planes)
        self.assertEqual(dataset.is_test, recovered.is_test)
        self.assertEqual(dataset.pos_features.shape, recovered.pos_features.shape)
        self.assertEqual(dataset.next_moves.shape, recovered.next_moves.shape)
        self.assertEqual(dataset.results.shape, recovered.results.shape)
        self.assertEqualNPArray(dataset.next_moves, recovered.next_moves)
        self.assertEqualNPArray(dataset.pos_features, recovered.pos_features)
        self.assertEqualNPArray(dataset.results, recovered.results)

