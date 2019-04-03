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

# Overrideable configuration parameters
# To override, simple do
#   export PARAM=value

export PROJECT=${PROJECT:-"minigo-pub"}
export LOGGING_PROJECT=${PROJECT:-"$PROJECT"}
export CLUSTER_NAME=${CLUSTER_NAME:-"minigo-v5"}
export BOARD_SIZE=${BOARD_SIZE:-"19"}
export K8S_VERSION=${K8S_VERSION:-"1.9"}
export ZONE=${ZONE:-"asia-east1-a"}
export NUM_K8S_NODES=${NUM_NODES:-"5"}

# Configuration for service accounts so that the cluster can do cloud-things.
export SERVICE_ACCOUNT=${SERVICE_ACCOUNT:-"${PROJECT}-${CLUSTER_NAME}-services"}
export SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT}@${PROJECT}.iam.gserviceaccount.com"
export SERVICE_ACCOUNT_KEY_LOCATION=${SERVICE_ACCOUNT_KEY_LOCATION:-"/tmp/${SERVICE_ACCOUNT}-key.json"}

# Constants for docker container creation
export VERSION_TAG=${VERSION_TAG:-"0.16"}
export EVAL_VERSION_TAG=${EVAL_VERSION_TAG:-"latest"}
export MINIGUI_PY_CPU_CONTAINER=${MINIGUI_PY_CPU_CONTAINER:-"minigui-py-cpu-v2"}

# Bucket names live in a single global namespace
# So, we prefix the project name to avoid collisions
#
# For more details, see https://cloud.google.com/storage/docs/best-practices
export BUCKET_NAME=${BUCKET_NAME:-"${PROJECT}-minigo-v5-${BOARD_SIZE}"}

# By default, buckets are created in us-east1, but for more performance, it's
# useful to have a region located near the GKE cluster.
# For more about locations, see
# https://cloud.google.com/storage/docs/bucket-locations
export BUCKET_LOCATION=${BUCKET_LOCATION:-"asia-east1"}

# Bigtable resources
export CBT_INSTANCE=${CBT_INSTANCE:-"minigo-instance"}
export CBT_ZONE=${CBT_ZONE:-"us-central1-b"}
export CBT_TABLE=${CBT_TABLE:-"games"}
export CBT_EVAL_TABLE=${CBT_EVAL_TABLE:-"eval_games"}
export CBT_MODEL_TABLE=${CBT_MODEL_TABLE:-"models"}

# Needed for Bigtable clients or any gRPC code running on a GCE VM
export GRPC_DEFAULT_SSL_ROOTS_FILE_PATH=/etc/ssl/certs/ca-certificates.crt
