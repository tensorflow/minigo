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

# Reset all the envinornment variables used by the cluster configuration.
# To use, run: source unset-common.sh

unset PROJECT
unset LOGGING_PROJECT
unset CLUSTER_NAME
unset BOARD_SIZE
unset K8S_VERSION
unset ZONE
unset NUM_K8S_NODES

unset SERVICE_ACCOUNT
unset SERVICE_ACCOUNT_EMAIL
unset SERVICE_ACCOUNT_KEY_LOCATION

unset VERSION_TAG
unset GPU_PLAYER_CONTAINER
unset CPU_PLAYER_CONTAINER

unset BUCKET_NAME
unset BUCKET_LOCATION
