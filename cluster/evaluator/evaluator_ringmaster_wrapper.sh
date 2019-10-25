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

set -e

: ${PROJECT?"Need to set Project"}
: ${MODEL_A?"Need to set MODEL_A"}
: ${MODEL_B?"Need to set MODEL_B"}
: ${SGF_BUCKET_NAME?"Need to set SGF_BUCKET_NAME"}

set -u

#echo Creds: $GOOGLE_APPLICATION_CREDENTIALS
echo Bucket: $SGF_BUCKET_NAME
echo Black:  ${MODEL_A}
echo White:  ${MODEL_B}

#gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS

# Verify gs://$SGF_BUCKET_NAME exists
gsutil ls "gs://${SGF_BUCKET_NAME}" &> /dev/null

# Download models
MODEL_PATH_A=$(cbt -project tensor-go -instance minigo-instance lookup models_for_eval "m_eval_${MODEL_A}" columns="metadata:model_path" | sed -n 's#^\s*"\(gs://.*\)"#\1#p')
MODEL_PATH_B=$(cbt -project tensor-go -instance minigo-instance lookup models_for_eval "m_eval_${MODEL_B}" columns="metadata:model_path" | sed -n 's#^\s*"\(gs://.*\)"#\1#p')

PATH_A=$(dirname "${MODEL_PATH_A}")
PATH_B=$(dirname "${MODEL_PATH_B}")

mkdir "${MODEL_A}" "${MODEL_B}"

gsutil rsync -r "${PATH_A}" "${MODEL_A}"
gsutil rsync -r "${PATH_B}" "${MODEL_B}"

# Mark things as executable
chmod a+x "${MODEL_A}/bin/bazel-bin/cc/gtp"
chmod a+x "${MODEL_B}/bin/bazel-bin/cc/gtp"

# Let python do the rest
python evaluator_ringmaster_wrapper.py
