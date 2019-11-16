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


# Plays selfplay games using a random model in order to bootstrap the
# reinforcement learning training loop.
# Example usage:
#  ./ml_perf/scripts/bootstrap.sh \
#    --board_size=19 \
#    --base_dir="${OUTPUT_DIR}"


source ml_perf/scripts/common.sh


# Build the C++ binaries
bazel build  -c opt \
  --define=board_size="${board_size}" \
  --define=tf=1 \
  cc:concurrent_selfplay cc:sample_records


# Initialize a clean directory structure.
for var_name in flag_dir golden_chunk_dir holdout_dir log_dir model_dir \
                selfplay_dir sgf_dir work_dir; do
  dir="${!var_name}"
  if [[ "${dir}" == gs://* ]]; then
    gsutil -m rm -rf "${dir}"/*
  else
    mkdir -p "${dir}"
    rm -rf "${dir}"/*
  fi
done
rm -f "${abort_file}"

echo "Copying flags to ${flag_dir}"
cp -r "ml_perf/flags/${board_size}"/* "${flag_dir}/"


# Run bootstrap selfplay.
./bazel-bin/cc/concurrent_selfplay \
  --flagfile="${flag_dir}/bootstrap.flags" \
  --output_dir="${selfplay_dir}/000000/0" \
  --holdout_dir="${holdout_dir}/000000/0" \
  --verbose=1 \
  | tee "${log_dir}/bootstrap.log"
