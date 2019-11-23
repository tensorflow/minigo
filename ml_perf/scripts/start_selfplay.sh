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
#
# Example usage that starts 8 processes running on 8 GPUs:
#  ./ml_perf/scripts/start_selfplay.sh \
#    --board_size=19 \
#    --devices=0,1,2,3,4,5,6,7 \
#    --base_dir="${OUTPUT_DIR}"


source ml_perf/scripts/common.sh


log_dir="${base_dir}/logs/selfplay/`hostname`"
mkdir -p "${log_dir}"


# Run selfplay workers.
for device in ${devices//,/ }; do
  CUDA_VISIBLE_DEVICES="${device}" \
  ./bazel-bin/cc/concurrent_selfplay \
    --flagfile="${flag_dir}/selfplay.flags" \
    --output_dir="${selfplay_dir}/\$MODEL/${device}" \
    --holdout_dir="${holdout_dir}/\$MODEL/${device}" \
    --model="${model_dir}/%d.minigo" \
    --run_forever=1 \
    --abort_file=${abort_file} \
    > "${log_dir}/`hostname`_selfplay_${device}.log" 2>&1 &
done
