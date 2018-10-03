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

import unittest

import rl_loop.shipname


class TestShipname(unittest.TestCase):
    def test_bootstrap_gen(self):
        name = rl_loop.shipname.generate(0)
        self.assertIn('bootstrap', name)

    def test_detect_name(self):
        string = '000017-model.index'
        detected_name = rl_loop.shipname.detect_model_name(string)
        self.assertEqual('000017-model', detected_name)

    def test_detect_num(self):
        string = '000017-model.index'
        detected_name = rl_loop.shipname.detect_model_num(string)
        self.assertEqual(17, detected_name)
