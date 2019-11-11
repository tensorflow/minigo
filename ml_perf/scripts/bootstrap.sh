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
for sub_dir in flags logs work_dir models data/selfplay data/holdout data/golden_chunks; do
  mkdir -p "${base_dir}/${sub_dir}"
  rm -rf "${base_dir}/${sub_dir}"/*
done
rm -f "${abort_file}"

echo "Copying flags to ${base_dir}/flags"
cp -r "ml_perf/flags/${board_size}"/* "${base_dir}/flags/"



# Run bootstrap selfplay.
./bazel-bin/cc/concurrent_selfplay \
  --flagfile="${base_dir}/flags/bootstrap.flags" \
  --output_dir="${base_dir}/data/selfplay/000000/0" \
  --holdout_dir="${base_dir}/data/holdout/000000/0" \
  --verbose=1 \
  | tee "${base_dir}/logs/bootstrap.log"
