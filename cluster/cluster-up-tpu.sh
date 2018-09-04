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

source ${SCRIPT_DIR}/common.sh
source ${SCRIPT_DIR}/utils.sh

export NUM_NODES=128
ZONE=us-central1-f

echo "TPU Cluster Creation"
echo "--------------------------------------"
echo "Using Project:      ${PROJECT}"
echo "Using Zone:         ${ZONE}"
echo "Using Cluster Name: ${CLUSTER_NAME}"

echo "Overriding num nodes to: $NUM_NODES"

check_gcloud_exists

#gcloud compute networks create $CLUSTER_NAME

# Create a Kubernetes cluster. This setup is designed for creating large clusters.
gcloud beta container clusters create \
    --project=$PROJECT \
    --zone=$ZONE \
    --cluster-version=1.10 \
    --scopes=cloud-platform \
    --network=$CLUSTER_NAME \
    --enable-ip-alias \
    --enable-tpu \
    --machine-type n1-standard-16 \
    --disk-size 45 \
    --num-nodes $NUM_NODES \
    $CLUSTER_NAME
    #--tpu-ipv4-cidr=/18 \

# Fetch its credentials so we can use kubectl locally
gcloud container clusters get-credentials $CLUSTER_NAME --project $PROJECT --zone $ZONE

#create_service_account_key

# Import the credentials into the cluster as a secret
kubectl create secret generic ${SERVICE_ACCOUNT}-creds --from-file=service-account.json=${SERVICE_ACCOUNT_KEY_LOCATION}


