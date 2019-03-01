#!/bin/bash
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

: ${MODEL_WHITE?"Need to set MODEL_WHITE"}
: ${MODEL_BLACK?"Need to set MODEL_BLACK"}
: ${SGF_BUCKET_NAME?"Need to set SGF_BUCKET_NAME"}

echo Creds: $GOOGLE_APPLICATION_CREDENTIALS
echo Bucket: $SGF_BUCKET_NAME
echo Black:  ${MODEL_BLACK}
echo White:  ${MODEL_WHITE}
echo Flags:  ${EVAL_FLAGS_PATH}

gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS

if [[ -z ${EVAL_FLAGS_PATH} ]]; then

echo "No flags found, using default flags"
cat <<EOF > flags.txt
  --num_readouts=1000
  --parallel_games=2
  --value_init_penalty=2.00
  --virtual_losses=8
  --resign_threshold=0.70
EOF

else
  echo "Using flags from ${EVAL_FLAGS_PATH}"
  gsutil cp ${EVAL_FLAGS_PATH} flags.txt
fi


# TODO(amj) Check that cc/main runs with perms to read a gs:// path directly
echo Retrieiving Models
gsutil cp ${MODEL_BLACK} .
gsutil cp ${MODEL_WHITE} .

BASENAME_BLACK=`basename $MODEL_BLACK`
BASENAME_WHITE=`basename $MODEL_WHITE`
DATE=`date +%Y-%m-%d`

python3 mask_flags.py bazel-bin/cc/eval \
  --model=$BASENAME_BLACK \
  --model_two=$BASENAME_WHITE \
  --sgf_dir="gs://$SGF_BUCKET_NAME/sgf/eval/$DATE" \
  --output_bigtable="tensor-go,minigo-instance,eval_games" \
  --bigtable_tag="$JOBNAME" \
  --flagfile=flags.txt

echo Finished an evaluation game!  Cleaning up...
rm $BASENAME_BLACK
rm $BASENAME_WHITE
rm flags.txt
