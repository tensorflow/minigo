import os
import tempfile
import unittest

import dual_net
import go
import preprocessing
from tests import test_utils

fast_hparams = {'k': 1, 'fc_width': 2, 'num_shared_layers': 1}

class TestDualNet(test_utils.MiniGoUnitTest):
    def test_train(self):
        with tempfile.TemporaryDirectory() as model_dir, \
            tempfile.NamedTemporaryFile() as tf_record:
            preprocessing.make_dataset_from_sgf(
                'tests/example_game.sgf', tf_record.name)
            model_save = os.path.join(model_dir, 'test_model')
            n = dual_net.DualNetworkTrainer(model_save, **fast_hparams)
            n.train([tf_record.name], num_steps=1)

    def test_inference(self):
        with tempfile.TemporaryDirectory() as model_dir:
            model_path = os.path.join(model_dir, 'blah')
            n = dual_net.DualNetworkTrainer(model_path, **fast_hparams)
            n.bootstrap()

            n1 = dual_net.DualNetwork(model_path, **fast_hparams)
            n1.run(go.Position())

            # In the past we've had issues initializing two separate NNs
            # in the same process... just double check that two DualNetwork
            # instances can live side by side.
            n2 = dual_net.DualNetwork(model_path, **fast_hparams)
            n2.run(go.Position())
