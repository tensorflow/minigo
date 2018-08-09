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

# To use this file, run: source minigui-defaults.sh

export MINIGUI_PYTHON=${MINIGUI_PYTHON:-"python3"}
export MINIGUI_BUCKET_NAME=${MINIGUI_BUCKET_NAME:-"minigo-pub"}
export MINIGUI_GCS_DIR=${MINIGUI_GCS_DIR:-"v7-19x19/models"}
export MINIGUI_MODEL=${MINIGUI_MODEL:-"000303-olympus"}
export MINIGUI_MODEL_TMPDIR=${MINIGUI_MODEL_TMPDIR:-"/tmp/minigo-models"}
export MINIGUI_BOARD_SIZE=${MINIGUI_BOARD_SIZE:-"19"}
export MINIGUI_PORT=${MINIGUI_PORT:-"5001"}
export MINIGUI_HOST=${MINIGUI_HOST:-"127.0.0.1"}
