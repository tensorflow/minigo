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

# Run in a sub-shell so we don't unexpectedly set new variables.
{
  source ${SCRIPT_DIR}/minigui-common.sh
  cd $SCRIPT_DIR

  # TODO(Kashomon): Use common utils library instead of copy-pasta.
  command -v gsutil >/dev/null 2>&1 || {
    echo >&2 "gsutil command is not defined"
    echo >&2 "Install Google Cloud SDK first:"
    echo >&2 "   https://cloud.google.com/sdk/downloads"
    exit 1
  }

  echo "Using: the following options for Minigui launch:"
  echo "--------------------------------------------------"
  echo "MINIGUI_PYTHON:      ${MINIGUI_PYTHON}"
  echo "MINIGUI_BUCKET_NAME: ${MINIGUI_BUCKET_NAME}"
  echo "MINIGUI_GCS_DIR:     ${MINIGUI_GCS_DIR}"
  echo "MINIGUI_MODEL:       ${MINIGUI_MODEL}"
  echo "MINIGUI_TMPDIR:      ${MINIGUI_TMPDIR}"
  echo "MINIGUI_BOARD_SIZE:  ${MINIGUI_BOARD_SIZE}"
  echo "MINIGUI_PORT:        ${MINIGUI_PORT}"
  echo "MINIGUI_HOST:        ${MINIGUI_HOST}"
  echo "MINIGUI_CONV_WIDTH:  ${MINIGUI_CONV_WIDTH}"
  echo "MINIGUI_NUM_READS:   ${MINIGUI_NUM_READS}"

  pyversion=$($MINIGUI_PYTHON --version)
  echo "Python Version:       ${pyversion}"

  echo
  echo "Downloading Model files:"
  echo "--------------------------------------------------"

  model_tmpdir="${MINIGUI_TMPDIR}/models"
  control_tmpdir="${MINIGUI_TMPDIR}/control"
  mkdir -p ${model_tmpdir}
  mkdir -p ${control_tmpdir}

  MODEL_SUFFIXES=( "data-00000-of-00001" "index" "meta" )
  for suffix in "${MODEL_SUFFIXES[@]}"
  do
    file_to_check="${model_tmpdir}/${MINIGUI_MODEL}.${suffix}"
    echo "Checking for: ${file_to_check}"
    if [[ ! -f "${file_to_check}" ]]; then
      gsutil cp gs://${MINIGUI_BUCKET_NAME}/${MINIGUI_GCS_DIR}/${MINIGUI_MODEL}.${suffix} ${model_tmpdir}/
    fi
  done

  # Assume models need to be frozen if .pb doesn't exist.
  cd ..
  model_path="${model_tmpdir}/${MINIGUI_MODEL}"
  if [[ ! -f "${model_path}.pb" ]]; then
    echo
    echo "Freezing model"
    echo "--------------------------------------------------"

    BOARD_SIZE=$MINIGUI_BOARD_SIZE $MINIGUI_PYTHON freeze_graph.py \
        --model_path=${model_path} --conv_width=$MINIGUI_CONV_WIDTH
  fi

  echo
  echo "Running Minigui!"
  echo "--------------------------------------------------"
  echo "Model: ${model_path}"
  echo "Size:  ${MINIGUI_BOARD_SIZE}"

  control_path="${control_tmpdir}/${MINIGUI_MODEL}.ctl"
  cat > ${control_path} << EOL
board_size=19
players = {
  "${MINIGUI_MODEL}" : Player("${MINIGUI_PYTHON}"
                       " -u"
                       " gtp.py"
                       " --load_file=${model_path}"
                       " --minigui_mode=true"
                       " --num_readouts=${MINIGUI_NUM_READS}"
                       " --conv_width=${MINIGUI_CONV_WIDTH}"
                       " --resign_threshold=-0.8"
                       " --verbose=2",
                       startup_gtp_commands=[],
                       environ={"BOARD_SIZE": str(board_size)}),
}
EOL

  ${MINIGUI_PYTHON} minigui/serve.py \
      --control="${control_path}" \
      --port "${MINIGUI_PORT}" \
      --host "${MINIGUI_HOST}"
}
