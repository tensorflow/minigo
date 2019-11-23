# Copyright 2019 Google LLC
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

# Sets up common state used by other MLPerf scripts.


set -euo pipefail

# Parse the command line arguments, converting each one into a bash variable.
while getopts “-:” opt; do
  case $opt in
    -)
      arg="${OPTARG%=*}"
      val="${OPTARG#*=}"
      case $arg in
        board_size | base_dir | abort_file | flag_dir | golden_chunk_dir | \
        holdout_dir | log_dir | model_dir | selfplay_dir | sgf_dir | work_dir | \
        tpu_name | devices)
          eval "${arg}=${val}" ;;
      esac ;;
  esac
done

# Set default values for any missing command line arguments.
if [ -z "${devices-}" ]; then devices="0"; fi
if [ -z "${board_size-}" ]; then board_size="19"; fi
if [ -z "${abort_file-}" ]; then abort_file="${base_dir}/abort"; fi
if [ -z "${flag_dir-}" ]; then flag_dir="${base_dir}/flags"; fi
if [ -z "${golden_chunk_dir-}" ]; then golden_chunk_dir="${base_dir}/data/golden_chunks"; fi
if [ -z "${holdout_dir-}" ]; then holdout_dir="${base_dir}/data/holdout"; fi
if [ -z "${log_dir-}" ]; then log_dir="${base_dir}/logs"; fi
if [ -z "${model_dir-}" ]; then model_dir="${base_dir}/models"; fi
if [ -z "${selfplay_dir-}" ]; then selfplay_dir="${base_dir}/data/selfplay"; fi
if [ -z "${sgf_dir-}" ]; then sgf_dir="${base_dir}/sgf"; fi
if [ -z "${work_dir-}" ]; then work_dir="${base_dir}/work_dir"; fi
if [ -z "${tpu_name-}" ]; then tpu_name=""; fi

# Preserve the arguments the script was called with.
script_args=("$@")

set -x
