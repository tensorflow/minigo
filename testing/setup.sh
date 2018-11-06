#!/bin/bash
#
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

# Simple shell script to lint files and run the tests. Could be helpful for
# users, but largely for automation.
#
# NOTE! If this file changes/moves, please change
# https://github.com/kubernetes/test-infra/blob/master/config/jobs/tensorflow/minigo/minigo.yaml

# Copy prebuilt TensorFlow from image.
rsync -ah /app/cc/tensorflow/ ./cc/tensorflow/

