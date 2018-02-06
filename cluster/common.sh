# Overrideable configuration parameters
# To override, simple do
#   export PARAM=value

export PROJECT=${PROJECT:-"minigo-pub"}
export LOGGING_PROJECT=${PROJECT:-"$PROJECT"}
export CLUSTER_NAME=${CLUSTER_NAME:-"minigo2"}
export BOARD_SIZE=${BOARD_SIZE:-"19"}
export K8S_VERSION=${K8S_VERSION:-"1.9.2-gke.0"}
export ZONE=${ZONE:-"asia-east1-a"}
export NUM_K8S_NODES=${NUM_K8S_NODES:-"5"}

# Configuration for service accounts so that the cluster can do cloud-things.
export SERVICE_ACCOUNT=${SERVICE_ACCOUNT:-"minigo-services2"}
export SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT}@${PROJECT}.iam.gserviceaccount.com"
export SERVICE_ACCOUNT_KEY_LOCATION=${SERVICE_ACCOUNT_KEY_LOCATION:-"/tmp/${SERVICE_ACCOUNT}-key.json"}

# Constants for docker container creation
export VERSION_TAG=${VERSION_TAG:-"0.14"}
export GPU_PLAYER_CONTAINER=${GPU_PLAYER_CONTAINER:-"minigo-gpu-player"}
export CPU_PLAYER_CONTAINER=${CPU_PLAYER_CONTAINER:-"minigo-player"}

# Bucket names live in a single global namespace
# So, we prefix the project name to avoid collisions
#
# For more details, see https://cloud.google.com/storage/docs/best-practices
export BUCKET_NAME=${BUCKET_NAME:-"${PROJECT}-minigo-v2"}

# By default, buckets are created in us-east1, but for more performance, it's
# useful to have a region located near the GKE cluster.
# For more about locations, see
# https://cloud.google.com/storage/docs/bucket-locations
export BUCKET_LOCATION=${BUCKET_LOCATION:-"asia-east1"}
