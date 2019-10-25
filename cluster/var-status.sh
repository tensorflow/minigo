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
#
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "-------------------------------------------------------------"
echo "PROJECT:          $PROJECT"
echo "LOGGING_PROJECT:  $LOGGING_PROJECT"
echo "CLUSTER_NAME:     $CLUSTER_NAME"
echo "BOARD_SIZE:       $BOARD_SIZE"
echo "K8S_VERSION:      $K8S_VERSION"
echo "ZONE:             $ZONE"
echo "NUM_NODES:        $NUM_NODES"
echo "-------------------------------------------------------------"
echo "SERVICE_ACCOUNT:              $SERVICE_ACCOUNT"
echo "SERVICE_ACCOUNT_EMAIL:        $SERVICE_ACCOUNT_EMAIL"
echo "SERVICE_ACCOUNT_KEY_LOCATION: $SERVICE_ACCOUNT_KEY_LOCATION"
echo "-------------------------------------------------------------"
echo "VERSION_TAG:              $VERSION_TAG"
echo "EVAL_VERSION_TAG:         $EVAL_VERSION_TAG"
echo "GPU_PLAYER_CONTAINER:     $GPU_PLAYER_CONTAINER"
echo "CPU_PLAYER_CONTAINER:     $CPU_PLAYER_CONTAINER"
echo "MINIGUI_PY_CPU_CONTAINER: $MINIGUI_PY_CPU_CONTAINER"
echo "-------------------------------------------------------------"
echo "BUCKET_NAME:      $BUCKET_NAME"
echo "BUCKET_LOCATION:  $BUCKET_LOCATION"
echo "-------------------------------------------------------------"
echo "CBT_INSTANCE:          $CBT_INSTANCE"
echo "CBT_ZONE:              $CBT_ZONE"
echo "CBT_TABLE:             $CBT_TABLE"
echo "CBT_MODEL_TABLE:       $CBT_MODEL_TABLE"
echo "CBT_MODEL_EVAL_TABLE:  $CBT_MODEL_EVAL_TABLE"
echo "CBT_EVAL_TABLE:        $CBT_EVAL_TABLE"
echo "-------------------------------------------------------------"
