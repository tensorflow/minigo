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

"""Looks up a TPU's address from its name.
Required because this functionality isn't present in the Cloud TPU C++ API."""

import socket
import logging
from absl import app, flags
from urllib.request import Request
from urllib.request import urlopen
from apiclient import discovery
from oauth2client.client import GoogleCredentials


flags.DEFINE_string('project', None, 'Project name. Defaults to the VM project.')
flags.DEFINE_string('zone', None, 'Zone. Defaults to the VM zone.')
flags.DEFINE_string('tpu', None, 'TPU name. Defaults to the VM hostname.')

FLAGS = flags.FLAGS


_GCE_METADATA_ENDPOINT = 'http://metadata.google.internal'


def _request_compute_metadata(path):
    req = Request('%s/computeMetadata/v1/%s' % (_GCE_METADATA_ENDPOINT, path),
                  headers={'Metadata-Flavor': 'Google'})
    return urlopen(req).read().decode('utf-8')


def main(argv):
    logging.getLogger().setLevel(logging.ERROR)

    del argv  # Unused
    tpu = FLAGS.tpu
    if tpu is None:
        tpu = socket.gethostname()

    project = FLAGS.project
    if project is None:
        project = _request_compute_metadata('project/project-id')

    zone = FLAGS.zone
    if zone is None:
        zone = _request_compute_metadata('instance/zone').split('/')[-1]

    credentials = GoogleCredentials.get_application_default()
    service = discovery.build(
        'tpu', 'v1alpha1', credentials=credentials, cache_discovery=False)
    full_name = 'projects/{}/locations/{}/nodes/{}'.format(project, zone, tpu)
    res = service.projects().locations().nodes().get(name=full_name).execute()
    addrs = []
    for endpoint in res['networkEndpoints']:
        addrs.append('grpc://{}:{}'.format(endpoint['ipAddress'],
                                           endpoint['port']))
    print(','.join(addrs))
        


if __name__ == '__main__':
    app.run(main)
