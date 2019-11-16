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

# Starts selfplay processes running.
# In this example, 8 selfplay processes are started, with each one running
# inference on a different GPU.
# Example usage:
#  ./ml_perf/scripts/start_selfplay.sh \
#    --board_size=19 \
#    --base_dir="${OUTPUT_DIR}"


source ml_perf/scripts/common.sh


log_dir="${base_dir}/logs/selfplay/`hostname`"
mkdir -p "${log_dir}"


# Run selfplay workers.
for device in {0..7}; do
  CUDA_VISIBLE_DEVICES="${device}" \
  ./bazel-bin/cc/concurrent_selfplay \
    --flagfile="${flag_dir}/selfplay.flags" \
    --output_dir="${data_dir}/selfplay/\$MODEL/${device}" \
    --holdout_dir="${data_dir}/holdout/\$MODEL/${device}" \
    --model="${model_dir}/%d.pb" \
    --run_forever=1 \
    --abort_file=${abort_file} \
    > "${log_dir}/`hostname`_selfplay_${device}.log" 2>&1 &
done
