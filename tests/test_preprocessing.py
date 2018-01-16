import itertools
import tensorflow as tf
import numpy as np
import tempfile

import coords
import preprocessing
import features
import go
from test_utils import GoPositionTestCase

go.set_board_size(9)
TEST_SGF = "(;CA[UTF-8]SZ[9]PB[Murakawa Daisuke]PW[Iyama Yuta]KM[6.5]HA[0]RE[W+1.5]GM[1];B[fd];W[cf])"

class TestPreprocessing(GoPositionTestCase):
    def create_random_data(self, num_examples):
        raw_data = []
        for i in range(num_examples):
            feature = np.random.random([
                go.N, go.N, features.NEW_FEATURES_PLANES]).astype(np.uint8)
            pi = np.random.random([go.N * go.N + 1]).astype(np.float32)
            value = np.random.random()
            raw_data.append((feature, pi, value))
        return raw_data

    def extract_data(self, tf_record):
        tf_example_tensor = preprocessing.get_input_tensors(
            1, [tf_record], num_repeats=1, shuffle_records=False, shuffle_examples=False)
        recovered_data = []
        with tf.Session() as sess:
            while True:
                try:
                    values = sess.run(tf_example_tensor)
                    recovered_data.append((
                        values['pos_tensor'],
                        values['pi_tensor'],
                        values['value_tensor']))
                except tf.errors.OutOfRangeError:
                    break
        return recovered_data

    def assertEqualData(self, data1, data2):
        '''Assert that two data are equal, where both are of form:
        data = List<Tuple<feature_array, pi_array, value>>
        '''
        self.assertEqual(len(data1), len(data2))
        for datum1, datum2 in zip(data1, data2):
            # feature
            self.assertEqualNPArray(datum1[0], datum2[0])
            # pi
            self.assertEqualNPArray(datum1[1], datum2[1])
            # value
            self.assertEqual(datum1[2], datum2[2])

    def test_serialize_round_trip(self):
        np.random.seed(1)
        raw_data = self.create_random_data(10)
        tfexamples = list(map(preprocessing.make_tf_example, *zip(*raw_data)))

        with tempfile.NamedTemporaryFile() as f:
            preprocessing.write_tf_examples(f.name, tfexamples)
            recovered_data = self.extract_data(f.name)

        self.assertEqualData(raw_data, recovered_data)

    def test_serialize_round_trip_no_parse(self):
        np.random.seed(1)
        raw_data = self.create_random_data(10)
        tfexamples = list(map(preprocessing.make_tf_example, *zip(*raw_data)))

        with tempfile.NamedTemporaryFile() as start_file, \
            tempfile.NamedTemporaryFile() as rewritten_file:
            preprocessing.write_tf_examples(start_file.name, tfexamples)
            # We want to test that the rewritten, shuffled file contains correctly
            # serialized tf.Examples.
            batch_size = 4
            batches = list(preprocessing.shuffle_tf_examples(batch_size, [start_file.name]))
            self.assertEqual(len(batches), 3) # 2 batches of 4, 1 incomplete batch of 2.

            # concatenate list of lists into one list
            all_batches = list(itertools.chain.from_iterable(batches))

            for batch in batches:
                preprocessing.write_tf_examples(rewritten_file.name, all_batches, serialize=False)

            original_data = self.extract_data(start_file.name)
            recovered_data = self.extract_data(rewritten_file.name)

        # stuff is shuffled, so sort before checking equality
        sort_key = lambda nparray_tuple: nparray_tuple[2]
        original_data = sorted(original_data, key=sort_key)
        recovered_data = sorted(recovered_data, key=sort_key)

        self.assertEqualData(original_data, recovered_data)

    def test_make_dataset_from_sgf(self):
        with tempfile.NamedTemporaryFile() as sgf_file, \
            tempfile.NamedTemporaryFile() as record_file:
            sgf_file.write(TEST_SGF.encode('utf8'))
            sgf_file.seek(0)
            preprocessing.make_dataset_from_sgf(sgf_file.name, record_file.name)
            recovered_data = self.extract_data(record_file.name)
        start_pos = go.Position()
        first_move = coords.parse_sgf_coords('fd')
        next_pos = start_pos.play_move(first_move)
        second_move = coords.parse_sgf_coords('cf')
        expected_data = [
            (
                features.extract_features(start_pos),
                preprocessing._one_hot(coords.flatten_coords(first_move)),
                -1
            ), (
                features.extract_features(next_pos),
                preprocessing._one_hot(coords.flatten_coords(second_move)),
                -1
            )]
        self.assertEqualData(expected_data, recovered_data)
