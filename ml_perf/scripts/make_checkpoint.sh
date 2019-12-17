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

# Creates a bootstrap checkpoint from a previous run.
#
# Example usage:
#  ./ml_perf/scripts/make_checkpoint.sh \
#    --model_num=24 \
#    --checkpoint_num=10193 \
#    --base_dir="${PREVIOUS_RUN_BASE_DIR}" \
#    --dst_dir="${DESTINATION_CHECKPOINT_DIR}"


source ml_perf/scripts/common.sh


bazel build -c opt --copt=-O3 cc:sample_records


python3 ml_perf/make_checkpoint.py \
  --selfplay_dir="${selfplay_dir}" \
  --work_dir="${work_dir}" \
  --flag_dir="${flag_dir}" \
  --window_size="${window_size}" \
  --min_games_per_iteration="${min_games_per_iteration}" \
  --ckpt="${ckpt}" \
  --checkpoint_dir="${checkpoint_dir}"
