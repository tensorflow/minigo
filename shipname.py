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

import random
import re
import petname

import go

MODEL_NUM_REGEX = "^\d{6}"
MODEL_NAME_REGEX = "^\d{6}(-\w+)+"

SHIP_FILE = "data/names.txt"

names = open(SHIP_FILE).read().split('\n')

def generate(model_num):
    if model_num == 0:
        new_name = 'bootstrap'
    elif go.N == 19:
        new_name = random.choice(names)
    else:
        new_name = petname.generate()
    full_name = "%06d-%s" % (model_num, new_name)
    return full_name

def detect_model_num(string):
    '''Takes a string related to a model name and extract its model number.

    For example:
        '000000-bootstrap.index' => 0
    '''
    match = re.match(MODEL_NUM_REGEX, string)
    if match:
        return int(match.group())
    else:
        return None

def detect_model_name(string):
    '''Takes a string related to a model name and extract its model number.

    For example:
        '000000-bootstrap.index' => '000000-bootstrap'
    '''
    match = re.match(MODEL_NAME_REGEX, string)
    if match:
        return match.group()
    else:
        return None
