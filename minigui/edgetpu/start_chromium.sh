#!/bin/bash
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
# You can pass a parameter as the start URL
killall chromium
export DISPLAY=:0
export GDK_BACKEND=x11
chromium --incognito $1 &
CHROMIUM_PID=$!
sleep 5
xte -x :0 "key F11"
xte -x :0 "keydown Control_L" "key 0" "keyup Control_L"
wait ${CHROMIUM_PID}
