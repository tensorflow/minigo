#!/bin/bash
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

# Usage:  `repeat_run.sh bucket_name` where 'bucket-name' is the path to a gcs
#         bucket without the gs:// prefix

set -e
: ${1?"Need to call with 'repeat_run.sh BUCKET_NAME' (no gs:// prefix needed)"}

BUCKET_NAME=$1

while true
do
  BASE_DIR=$(pwd)/results/$(hostname)-$(date +%Y-%m-%d-%H-%M)
  BOARD_SIZE=9 python ml_perf/reference_implementation.py \
      --base_dir=$BASE_DIR \
      --flagfile=ml_perf/flags/9/rl_loop.flags \
      --parallel_post_train

  BOARD_SIZE=9 python ml_perf/eval_models.py \
      --base_dir=$BASE_DIR \
      --flags_dir=ml_perf/flags/9/

  gsutil -m rsync -r -x "data" $BASE_DIR gs://$BUCKET_NAME/$(basename $BASE_DIR)
done


