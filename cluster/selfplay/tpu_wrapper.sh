#!/bin/sh
# Copyright 2018 Google LLC
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

# Added to the player image.
# Wraps our call to cc/main

set -e

: ${BUCKET_NAME?"Bucket name must be set!"}
: ${WORK_DIR?"Working directory must be set!"}

bazel-bin/cc/main \
  --engine=remote \
  --checkpoint-dir=${WORK_DIR}\
  --inject_noise=true \
  --soft_pick=true \
  --random_symmetry=true \
  --virtual_losses=8 \
  --parallel_games=32 \
  --num_readouts=800 \
  --resign_threshold=-0.999 \
  --disable_resign_pct=0.10 \
  --output_dir=gs://${BUCKET_NAME}/data/selfplay \
  --holdout_dir=gs://${BUCKET_NAME}/data/holdout \
  --sgf_dir=gs://${BUCKET_NAME}/sgf \
  --mode=selfplay \
  --flags_path=gs://${BUCKET_NAME}/flags.txt
  --run_forever=true
