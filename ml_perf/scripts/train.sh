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

# Starts the training loop.
# Example usage:
#  ./ml_perf/scripts/train.sh \
#    --board_size=19 \
#    --base_dir="${OUTPUT_DIR}"


source ml_perf/scripts/common.sh


# Set up an exit handler that stops the selfplay workers.
function stop_selfplay {
  ./ml_perf/scripts/stop_selfplay.sh "${script_args[@]}"
}
trap stop_selfplay EXIT


# Run the training loop.
BOARD_SIZE="${board_size}" \
CUDA_VISIBLE_DEVICES="0" \
python3 ml_perf/train_loop.py \
  --flags_dir="${flag_dir}" \
  --golden_chunk_dir="${golden_chunk_dir}" \
  --holdout_dir="${holdout_dir}" \
  --log_dir="${log_dir}" \
  --model_dir="${model_dir}" \
  --selfplay_dir="${selfplay_dir}" \
  --work_dir="${work_dir}" \
  --flagfile="${flag_dir}/train_loop.flags" \
  --tpu_name="${tpu_name}" \
  2>&1 | tee "${log_dir}/train_loop.log"
