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

"""
This file contains an implementation of dual_net.py for Google's EdgeTPU.
It can only be used for inference and requires a specially quantized and
compiled model file. For more information see https://coral.withgoogle.com
"""

import numpy as np
import features as features_lib
from edgetpu.basic.basic_engine import BasicEngine

class DualNetworkEdgeTpu(object):
  def __init__(self, save_file, board_size=19):
    self.engine = BasicEngine(save_file)
    self.board_size = board_size
    self.output_policy_size = self.board_size**2 + 1

    input_tensor_shape = self.engine.get_input_tensor_shape()
    expected_input_shape = [1,self.board_size,self.board_size,17]
    if not np.array_equal(input_tensor_shape, expected_input_shape):
      raise RuntimeError(
          'Invalid input tensor shape {}. Expected: {}'.format(
              input_tensor_shape, expected_input_shape))
    output_tensors_sizes = self.engine.get_all_output_tensors_sizes()
    expected_output_tensor_sizes = [self.output_policy_size, 1]
    if not np.array_equal(output_tensors_sizes, expected_output_tensor_sizes):
      raise RuntimeError(
          'Invalid output tensor sizes {}. Expected: {}'.format(
              output_tensors_sizes, expected_output_tensor_sizes))

  def run(self, position):
    probs, values = self.run_many([position])
    return probs[0], values[0]

  def run_many(self, positions):
    processed = list(map(features_lib.extract_features, positions))
    probabilities = []
    values = []
    for state in processed:
      assert state.shape  == (self.board_size, self.board_size, 17), str(state.shape)
      result = self.engine.RunInference(state.flatten())
      inference_time = result[0] # ms
      policy_output = result[1][0:self.output_policy_size]
      value_output = result[1][-1]
      probabilities.append(policy_output)
      values.append(value_output)
    return probabilities, values

