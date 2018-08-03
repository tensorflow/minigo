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

echo creds: $GOOGLE_APPLICATION_CREDENTIALS
echo bucket: $BUCKET_NAME
echo board_size: $BOARD_SIZE

#gcloud auth activate-service-account --key-file=/etc/credentials/service-account.json
echo Retrieiving Model
MODEL_FILE=`gsutil ls "gs://$BUCKET_NAME/models/*.pb" | sort | tail -n 1`
echo Retrieiving games
NAME=`echo $MODEL_FILE  | rev | cut -d/ -f1 | rev`
BASENAME=`echo $NAME | cut -d. -f1`
echo "gsutil ls 'gs://$BUCKET_NAME/data/selfplay/$BASENAME/*.zz'"
GAMES=`gsutil ls "gs://$BUCKET_NAME/data/selfplay/$BASENAME/*.zz" | wc -l`

gsutil cp $MODEL_FILE .

if [ $GAMES -lt 25000 ];
then
  echo Playing $NAME
  bazel-bin/cc/main \
    --model=$NAME \
    --num_readouts=800 \
    --mode=selfplay \
    --resign_threshold=0.999 \
    --disable_resign_pct=0.1 \
    --holdout_dir="gs://$BUCKET_NAME/data/holdout/$BASENAME" \
    --output_dir="gs://$BUCKET_NAME/data/selfplay/$BASENAME" \
    --sgf_dir="gs://$BUCKET_NAME/sgf/$BASENAME"
  echo Finished a set of games!
else
  echo "$NAME has enough games ($GAMES)"
fi

