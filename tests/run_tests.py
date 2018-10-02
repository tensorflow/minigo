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

import sys
sys.path.insert(0, '.')
import unittest

# Importing all of these modules causes all the relevant flags to get defined.
# They thus become overrideable, either with cmd line args to run_tests or via
# the test_flags file.
import test_coords
import test_dual_net
import test_features
import test_go
import test_mcts
import test_preprocessing
import test_sgf_wrapper
import test_shipname
import test_strategies
import test_symmetries
import test_utils

from absl import flags

if __name__ == '__main__':
    # Parse test flags and initialize default flags
    flags.FLAGS(['ignore', '--flagfile=tests/test_flags'])
    # Replicate the behavior of `python -m unittest discover tests`
    unittest.main(module=None, argv=['run_tests.py', 'discover', '.'])
