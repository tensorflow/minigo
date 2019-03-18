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

import random
import tempfile

import coords
import preprocessing
import features
import go
import symmetries
from tests import test_utils

import numpy as np
import tensorflow as tf


TEST_SGF = "(;CA[UTF-8]SZ[9]PB[Seth the best]PW[Andrew opens H4]KM[6.5]HA[0]RE[W+1.5]GM[1];B[fd];W[cf])"


class TestPreprocessing(test_utils.MinigoUnitTest):
    def create_random_data(self, num_examples):
        raw_data = []
        for _ in range(num_examples):
            feature = (256 * np.random.random([
                go.N, go.N, features.NEW_FEATURES_PLANES])).astype(np.uint8)
            pi = np.random.random([go.N * go.N + 1]).astype(np.float32)
            value = np.random.random()
            raw_data.append((feature, pi, value))
        return raw_data

    def get_data_tensors(self, pos_tensor, label_tensors):
        recovered_data = []
        with tf.Session() as sess:
            while True:
                try:
                    pos_value, label_values = sess.run(
                        [pos_tensor, label_tensors])
                    recovered_data.append((
                        pos_value,
                        label_values['pi_tensor'],
                        label_values['value_tensor']))
                except tf.errors.OutOfRangeError:
                    break
        return recovered_data

    def extract_data(self, tf_record, filter_amount=1, random_rotation=False):
        pos_tensor, label_tensors = preprocessing.get_input_tensors(
            1, [tf_record], num_repeats=1, shuffle_records=False,
            shuffle_examples=False, filter_amount=filter_amount,
            random_rotation=random_rotation)
        return self.get_data_tensors(pos_tensor, label_tensors)

    def extract_tpu_data(self, tf_record, random_rotation=False):
        dataset = preprocessing.get_tpu_input_tensors(
            1, [tf_record], num_repeats=1, filter_amount=1,
            random_rotation=random_rotation)
        pos_tensor, label_tensors = dataset.make_one_shot_iterator().get_next()
        return self.get_data_tensors(pos_tensor, label_tensors)

    def assertEqualData(self, data1, data2):
        """Assert that the two datas are equal.

        Args:
            data1: List<Tuple<feature_array, pi_array, value>>
            data2: Same form as data1
        """
        self.assertEqual(len(data1), len(data2))
        for datum1, datum2 in zip(data1, data2):
            # feature
            self.assertEqualNPArray(datum1[0], datum2[0])
            # pi
            self.assertEqualNPArray(datum1[1], datum2[1])
            # value
            self.assertEqual(datum1[2], datum2[2])

    def reset_random(self):
        random.seed(1)
        tf.set_random_seed(1)

    def find_symmetry(self, x, pi, x2, pi2):
        for sym in symmetries.SYMMETRIES:
            x_equal = (x == symmetries.apply_symmetry_feat(sym, x2)).all()
            pi_equal = (pi == symmetries.apply_symmetry_pi(sym, pi2)).all()
            if x_equal and pi_equal:
                return sym

        self.assertTrue(False, "No rotation makes {} equal {}".format(pi, pi2))

    def x_and_pi_same(self, run_a, run_b):
        x_a, pi_a, values_a = zip(*run_a)
        x_b, pi_b, values_b = zip(*run_b)
        self.assertEqual(values_a, values_b, "Values are not same")
        return np.array_equal(x_a, x_b) and np.array_equal(pi_a, pi_b)

    def assert_rotate_data(self, run_one, run_two, run_three):
        """Verify run_one is rotated and run_two is identical to run_three"""
        self.assertTrue(
            self.x_and_pi_same(run_two, run_three),
            "Not deterministic")
        self.assertFalse(
            self.x_and_pi_same(run_one, run_two),
            "Not randomly rotated")

        syms = []
        for (x, pi, v), (x2, pi2, v2) in zip(run_one, run_two):
            self.assertEqual(v, v2, "values not the same")
            # For each record find the symmetry that makes them equal
            syms.extend(
                map(lambda r: self.find_symmetry(*r), zip(x, pi, x2, pi2)))

        difference = set(symmetries.SYMMETRIES) - set(syms)
        self.assertEqual(len(run_one), len(syms), "Not same number of records")
        self.assertEqual(set(), difference, "Didn't find these rotations")

    def test_serialize_round_trip(self):
        np.random.seed(1)
        raw_data = self.create_random_data(10)
        tfexamples = list(map(preprocessing.make_tf_example, *zip(*raw_data)))

        with tempfile.NamedTemporaryFile() as f:
            preprocessing.write_tf_examples(f.name, tfexamples)
            recovered_data = self.extract_data(f.name)

        self.assertEqualData(raw_data, recovered_data)

    def test_filter(self):
        raw_data = self.create_random_data(100)
        tfexamples = list(map(preprocessing.make_tf_example, *zip(*raw_data)))

        with tempfile.NamedTemporaryFile() as f:
            preprocessing.write_tf_examples(f.name, tfexamples)
            recovered_data = self.extract_data(f.name, filter_amount=.05)

        # TODO: this will flake out very infrequently.  Use set_random_seed
        self.assertLess(len(recovered_data), 50)

    def test_make_dataset_from_sgf(self):
        with tempfile.NamedTemporaryFile() as sgf_file, \
                tempfile.NamedTemporaryFile() as record_file:
            sgf_file.write(TEST_SGF.encode('utf8'))
            sgf_file.seek(0)
            preprocessing.make_dataset_from_sgf(
                sgf_file.name, record_file.name)
            recovered_data = self.extract_data(record_file.name)
        start_pos = go.Position()
        first_move = coords.from_sgf('fd')
        next_pos = start_pos.play_move(first_move)
        second_move = coords.from_sgf('cf')
        expected_data = [
            (
                features.extract_features(start_pos),
                preprocessing._one_hot(coords.to_flat(first_move)),
                -1
            ), (
                features.extract_features(next_pos),
                preprocessing._one_hot(coords.to_flat(second_move)),
                -1
            )]
        self.assertEqualData(expected_data, recovered_data)

    def test_rotate_pyfunc(self):
        num_records = 20
        raw_data = self.create_random_data(num_records)
        tfexamples = list(map(preprocessing.make_tf_example, *zip(*raw_data)))

        with tempfile.NamedTemporaryFile() as f:
            preprocessing.write_tf_examples(f.name, tfexamples)

            self.reset_random()
            run_one = self.extract_data(f.name, random_rotation=False)

            self.reset_random()
            run_two = self.extract_data(f.name, random_rotation=True)

            self.reset_random()
            run_three = self.extract_data(f.name, random_rotation=True)

        self.assert_rotate_data(run_one, run_two, run_three)

    def test_tpu_rotate(self):
        num_records = 100
        raw_data = self.create_random_data(num_records)
        tfexamples = list(map(preprocessing.make_tf_example, *zip(*raw_data)))

        with tempfile.NamedTemporaryFile() as f:
            preprocessing.write_tf_examples(f.name, tfexamples)

            self.reset_random()
            run_one = self.extract_tpu_data(f.name, random_rotation=False)

            self.reset_random()
            run_two = self.extract_tpu_data(f.name, random_rotation=True)

            self.reset_random()
            run_three = self.extract_tpu_data(f.name, random_rotation=True)

        self.assert_rotate_data(run_one, run_two, run_three)
