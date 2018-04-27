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
