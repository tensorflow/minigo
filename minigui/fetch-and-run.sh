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
source ${SCRIPT_DIR}/minigui-common.sh
cd $SCRIPT_DIR

echo "Using: the following defaults:"
echo "--------------------------------------------------"
echo "MINIGUI_PYTHON:       ${MINIGUI_PYTHON}"
echo "MINIGUI_BUCKET_NAME:  ${MINIGUI_BUCKET_NAME}"
echo "MINIGUI_GCS_DIR:      ${MINIGUI_GCS_DIR}"
echo "MINIGUI_MODEL:        ${MINIGUI_MODEL}"
echo "MINIGUI_MODEL_TMPDIR: ${MINIGUI_MODEL_TMPDIR}"
echo "MINIGUI_BOARD_SIZE:   ${MINIGUI_BOARD_SIZE}"
echo "MINIGUI_PORT:         ${MINIGUI_PORT}"

pyversion=$($MINIGUI_PYTHON --version)
echo "Python Version:       ${pyversion}"

echo
echo "Downloading Model files:"
echo "--------------------------------------------------"

MODEL_SUFFIXES=( "data-00000-of-00001" "index" "meta" )
mkdir -p $MINIGUI_MODEL_TMPDIR

for suffix in "${MODEL_SUFFIXES[@]}"
do
  file_to_check="${MINIGUI_MODEL_TMPDIR}/${MINIGUI_MODEL}.${suffix}"
  echo "Checking for: ${file_to_check}"
  if [[ ! -f "${file_to_check}" ]]; then
    gsutil cp gs://${MINIGUI_BUCKET_NAME}/${MINIGUI_GCS_DIR}/${MINIGUI_MODEL}.${suffix} $MINIGUI_MODEL_TMPDIR/
  fi
done

# Assume models need to be converted if .converted.meta doesn't exist.
# TODO(amj): Backfill all the models and remove this.
cd ..
if [[ ! -f "${MINIGUI_MODEL_TMPDIR}/${MINIGUI_MODEL}.converted.meta" ]]; then
  echo
  echo "Converting model-data type"
  echo "--------------------------------------------------"
  (export BOARD_SIZE=$MINIGUI_BOARD_SIZE; $MINIGUI_PYTHON main.py convert $MINIGUI_MODEL_TMPDIR/$MINIGUI_MODEL $MINIGUI_MODEL_TMPDIR/$MINIGUI_MODEL.converted)
fi

echo
echo "Running Minigui!"
echo "--------------------------------------------------"
echo "Model: $MINIGUI_MODEL_TMPDIR/$MINIGUI_MODEL.converted"
echo "Size:  $MINIGUI_BOARD_SIZE"

$MINIGUI_PYTHON minigui/serve.py \
--model="$MINIGUI_MODEL_TMPDIR/$MINIGUI_MODEL.converted" \
--board_size="$MINIGUI_BOARD_SIZE" \
--port=$MINIGUI_PORT \
--host=0.0.0.0 \
--python_for_engine=${MINIGUI_PYTHON}
