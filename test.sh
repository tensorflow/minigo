#!/bin/bash

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

#
# Simple shell script to lint files and run the tests. Could be helpful for
# users, but largely for automation.
#
# NOTE! If this file changes/moves, please change
# https://github.com/kubernetes/test-infra/blob/master/jobs/config.json

# Ensure we're running from this directory to ensure PYTHONPATH is set
# correctly.
cd "$(dirname "$0")"

lint_fail=0
ls *.py | xargs pylint || {
  lint_fail=1
}

test_fail=0
PYTHONPATH= BOARD_SIZE=9 python3 -m unittest discover tests || {
  test_fail=1
}

if [ "${lint_fail}" -eq "1" ]; then
  echo >&2 "--------------------------------------"
  echo >&2 "Py linting did not pass successfully!"
fi

if [ "${test_fail}" -eq "1" ]; then
  echo >&2 "--------------------------------------"
  echo >&2 "The tests did not pass successfully!"
fi

if [ "${test_fail}" -eq "1" ] || [ "${lint_fail}" -eq "1" ]; then
  exit 1
fi
