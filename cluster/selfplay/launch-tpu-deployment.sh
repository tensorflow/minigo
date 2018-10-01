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

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source ${SCRIPT_DIR}/../common.sh

# envsubst doesn't exist for OSX. needs to be brew-installed
# via gettext. Should probably warn the user about that.
command -v envsubst >/dev/null 2>&1 || {
  echo >&2 "envsubst is required and not found. Aborting"
  if [[ "$OSTYPE" == "darwin"* ]]; then
    echo >&2 "------------------------------------------------"
    echo >&2 "If you're on OSX, you can install with brew via:"
    echo >&2 "  brew install gettext"
    echo >&2 "  brew link --force gettext"
  fi
  exit 1;
}

: ${BUCKET_NAME?"Need to set BUCKET_NAME"}
: ${SERVICE_ACCOUNT?"Need to set SERVICE_ACCOUNT"}

echo "-------------------------"
echo "  Launching TPU Cluster"
echo "-------------------------"
echo "Bucket:       $BUCKET_NAME"
echo "Service acct: $SERVICE_ACCOUNT"
echo "-------------------------"
cat ${SCRIPT_DIR}/tpu-player-deployment.yaml | envsubst | kubectl apply -f -
