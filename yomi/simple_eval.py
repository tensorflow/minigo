""" Simple job that reads from a storage bucket """

import os
import apiclient.discovery
import httplib2
import json
from oauth2client.service_account import ServiceAccountCredentials

SERVICE_ACCOUNT_KEY_LOCATION = os.environ['SERVICE_ACCOUNT_KEY_LOCATION']
BUCKET_NAME = os.environ['BUCKET_NAME']


def run():
    """ Talk to GCS. Do some things. """
    google_api_secret = json.load(open(SERVICE_ACCOUNT_KEY_LOCATION))

    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        SERVICE_ACCOUNT_KEY_LOCATION,
        ['https://www.googleapis.com/auth/devstorage.read_write'])

    # Make storage service object
    storage = apiclient.discovery.build(
        'storage', 'v1', http=credentials.authorize(httplib2.Http()))

    req = storage.buckets().get(bucket=BUCKET_NAME)
    resp = req.execute()
    print(json.dumps(resp))


def print_env():
    flags = {
        'BUCKET_NAME': BUCKET_NAME,
        'SERVICE_ACCOUNT_KEY_LOCATION': SERVICE_ACCOUNT_KEY_LOCATION,
    }
    print("Env variables are:")
    print('\n'.join('--{}={}'.format(flag, value)
                    for flag, value in flags.items()))


if __name__ == '__main__':
    print_env()
    run()
