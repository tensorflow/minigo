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
# Utilies for working with clusters and google cloud


# Checks that the gsutil CLI exists.
function check_gsutil_exists() {
  command -v gsutil >/dev/null 2>&1 || {
    echo >&2 "gsutil command is not defined"
    echo >&2 "Install Google Cloud SDK first:"
    echo >&2 "   https://cloud.google.com/sdk/downloads"
    exit 1
  }
}


# Checks that the gcloud CLI exists.
function check_gcloud_exists() {
  command -v gcloud >/dev/null 2>&1 || {
    echo >&2 "gcloud command is not defined"
    echo >&2 "Install Google Cloud SDK first"
    echo >&2 "   https://cloud.google.com/sdk/downloads"
    exit 1
  }
}


# Creates a cloud bucket if it doesn't exist. Recall that cloud buckets are
# a global namespace.
#
# Globals:
#   BUCKET_NAME: The name of the cloud bucket
#   BUCKET_LOCATION: The location to create the cloud bucket
function create_gcs_bucket() {
  check_gsutil_exists
  if [[ -z "${BUCKET_NAME}" ]]; then
    echo >&2 "BUCKET_NAME is not defined"
    return 1
  fi
  if [[ -z "${BUCKET_LOCATION}" ]]; then
    echo >&2 "BUCKET_LOCATION is not defined"
    return 1
  fi
  gsutil ls -b gs://$BUCKET_NAME >/dev/null || {
    echo >&2 "Bucket $BUCKET_NAME does not exist. Creating."
    gsutil mb -l $BUCKET_LOCATION gs://$BUCKET_NAME
  }
}


# Creates a cluster service account if it doesn't exist.
# Globals:
#   PROJECT: The cloud project
#   SERVICE_ACCOUNT: The service account email to create.
function create_service_account_key() {
  check_gcloud_exists
  if [[ -z "${PROJECT}" ]]; then
    echo >&2 "PROJECT is not defined"
    return 1
  fi
  if [[ -z "${SERVICE_ACCOUNT}" ]]; then
    echo >&2 "SERVICE_ACCOUNT is not defined"
    return 1
  fi
  if [[ -z "${SERVICE_ACCOUNT_EMAIL}" ]]; then
    echo >&2 "SERVICE_ACCOUNT_EMAIL is not defined"
    return 1
  fi
  if [[ -z "${SERVICE_ACCOUNT_KEY_LOCATION}" ]]; then
    echo >&2 "SERVICE_ACCOUNT_KEY_LOCATION is not defined"
    return 1
  fi

  if ! gcloud iam service-accounts list --project=$PROJECT | grep -q ${SERVICE_ACCOUNT}; then
    echo >&2 "SERVICE_ACCOUNT doesn't exist: creating"
    gcloud iam service-accounts create $SERVICE_ACCOUNT --project=$PROJECT
  fi

  # Make sure the service account can actually read the existing model entries
  gsutil ls gs://$BUCKET_NAME/models >/dev/null 2>&1 && {
    gsutil -m acl ch -r -u "${SERVICE_ACCOUNT_EMAIL}":R gs://${BUCKET_NAME}/models
  }

  # Grant it write permissions on our bucket. Should be safe to do multiple times
  gsutil acl ch -u "${SERVICE_ACCOUNT_EMAIL}":W gs://${BUCKET_NAME}

  if [[ ! -f "${SERVICE_ACCOUNT_KEY_LOCATION}" ]]; then
    echo >&2 "Service account key file doesn't exist."
    echo >&2 "Creating new key file at ${SERVICE_ACCOUNT_KEY_LOCATION}"
    gcloud iam service-accounts keys create \
      "${SERVICE_ACCOUNT_KEY_LOCATION}" --iam-account "${SERVICE_ACCOUNT_EMAIL}"

    # Ensure the service account can write to GCS
    gcloud projects add-iam-policy-binding "${PROJECT}" \
      --member serviceAccount:"${SERVICE_ACCOUNT_EMAIL}" \
      --role roles/storage.objectAdmin \
      --project "${PROJECT}"
  fi
}
