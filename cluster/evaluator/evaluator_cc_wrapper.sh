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

# Added to the evaluator image.
# Wraps our call to cc/main

# MODEL_BLACK and MODEL_WHITE should be full gs:// paths to .pb files

set -e

env
echo creds: $GOOGLE_APPLICATION_CREDENTIALS
echo bucket: $SGF_BUCKET_NAME
echo black:  ${MODEL_BLACK}
echo white:  ${MODEL_WHITE}

# TODO(amj) Check that cc/main runs with perms to read a gs:// path directly
echo Retrieiving Models
gsutil cp ${MODEL_BLACK} .
gsutil cp ${MODEL_WHITE} .

BASENAME_BLACK=`basename $MODEL_BLACK`
BASENAME_WHITE=`basename $MODEL_WHITE`
DATE=`date +%Y-%m-%d`

bazel-bin/cc/eval \
  --model=$BASENAME_BLACK \
  --model_two=$BASENAME_WHITE \
  --sgf_dir "gs://$SGF_BUCKET_NAME/sgf/eval/$DATE" \
  --num_readouts=1000 \
  --parallel_games=1 \
  --value_init_penalty=2.00 \
  --virtual_losses=8 \
  --resign_threshold=0.70

echo Finished an evaluation game!
