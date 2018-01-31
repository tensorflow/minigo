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

import google.cloud.logging as glog
import logging
import contextlib
import io
import sys
import os

LOGGING_PROJECT = os.environ.get('LOGGING_PROJECT', '')


def configure(project=LOGGING_PROJECT):
    if not project:
        print('!! Error: The $LOGGING_PROJECT enviroment '
              'variable is required in order to set up cloud logging. '
              'Cloud logging is disabled.')
        return

    logging.basicConfig(level=logging.INFO)
    try:
        # if this fails, redirect stderr to /dev/null so no startup spam.
        with contextlib.redirect_stderr(io.StringIO()):
            client = glog.Client(project)
            client.setup_logging(logging.INFO)
    except:
        print('!! Cloud logging disabled')
