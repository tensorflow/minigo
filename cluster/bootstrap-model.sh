#!/bin/sh
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

# Create bootstrap data

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source ${SCRIPT_DIR}/common.sh
source ${SCRIPT_DIR}/utils.sh

echo "Bootstrapping a Minigo model!"
echo "Bucket name:      ${BUCKET_NAME}"
echo "Bucket location:  ${MODEL_NAME}"
echo "Board Size:       ${BOARD_SIZE}"

MODEL_NAME=000000-bootstrap
PYTHONPATH=$SCRIPT_DIR/..

python3 ../main.py bootstrap gs://$BUCKET_NAME/models/$MODEL_NAME
