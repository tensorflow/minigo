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


set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source ${SCRIPT_DIR}/../../minigui/minigui-common.sh

echo "Using: the following defaults for run-local:"
echo "--------------------------------------------------"
echo "MINIGUI_MODEL:        ${MINIGUI_MODEL}"
echo "MINIGUI_MODEL_TMPDIR: ${MINIGUI_MODEL_TMPDIR}"
echo "MINIGUI_PORT:         ${MINIGUI_PORT}"

echo "PROJECT:              ${PROJECT}"
echo "VERSION_TAG:          ${VERSION_TAG}"
echo "MINIGUI CONTAINER:    ${MINIGUI_PY_CPU_CONTAINER}"
echo

docker run \
-p $MINIGUI_PORT:$MINIGUI_PORT \
-ti \
--mount type=bind,source="${MINIGUI_MODEL_TMPDIR}",target="${MINIGUI_MODEL_TMPDIR}" \
--rm gcr.io/${PROJECT}/${MINIGUI_PY_CPU_CONTAINER}:${VERSION_TAG}
