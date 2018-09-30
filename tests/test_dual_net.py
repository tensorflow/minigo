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

import os
import tempfile

from absl import flags

import dual_net
import go
import preprocessing
import train
from tests import test_utils


class TestDualNet(test_utils.MiniGoUnitTest):
    def test_train(self):
        with tempfile.TemporaryDirectory() as working_dir, \
                tempfile.NamedTemporaryFile() as tf_record:
            flags.FLAGS.work_dir = working_dir
            preprocessing.make_dataset_from_sgf(
                'tests/example_game.sgf', tf_record.name)
            train.train([tf_record.name])

    def test_inference(self):
        with tempfile.TemporaryDirectory() as working_dir, \
                tempfile.TemporaryDirectory() as export_dir:
            flags.FLAGS.work_dir = working_dir
            dual_net.bootstrap()
            exported_model = os.path.join(export_dir, 'bootstrap-model')
            dual_net.export_model(exported_model)

            n1 = dual_net.DualNetwork(exported_model)
            n1.run(go.Position())

            # In the past we've had issues initializing two separate NNs
            # in the same process... just double check that two DualNetwork
            # instances can live side by side.
            n2 = dual_net.DualNetwork(exported_model)
            n2.run(go.Position())
