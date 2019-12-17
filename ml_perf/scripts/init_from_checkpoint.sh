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

# Bootstraps a reinforcement learning loop from a checkpoint generated from
# a previous run.
#
# Example usage:
#  ./ml_perf/scripts/init_from_checkpoint.sh \
#    --board_size=19 \
#    --base_dir="${BASE_DIR}" \
#    --checkpoint_dir="${SOURCE_CHECKPOINT_DIR}"


source ml_perf/scripts/common.sh


# Initialize a clean directory structure.
for var_name in flag_dir golden_chunk_dir holdout_dir log_dir model_dir \
                selfplay_dir sgf_dir work_dir; do
  clean_dir "${!var_name}"
done
rm -f "${abort_file}"


BOARD_SIZE="${board_size}" \
python3 ml_perf/init_from_checkpoint.py \
  --checkpoint_dir="${checkpoint_dir}" \
  --selfplay_dir="${selfplay_dir}" \
  --work_dir="${work_dir}" \
  --model_dir="${model_dir}" \
  --flag_dir="${flag_dir}" \
  --tpu_name="${tpu_name}"

