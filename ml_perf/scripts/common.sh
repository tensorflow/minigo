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


set -eu

while getopts “-:” opt; do
  case $opt in
    -)
      arg="${OPTARG%=*}"
      val="${OPTARG#*=}"
      case $arg in
        board_size) board_size="${val}" ;;
        base_dir) base_dir="${val}" ;;
      esac ;;
  esac
done

abort_file="${base_dir}/abort"

set -x
