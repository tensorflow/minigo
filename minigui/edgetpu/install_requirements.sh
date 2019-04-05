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

python3 -m pip install -r minigui/requirements.txt

# For Raspberry Pi
if grep -q "Raspberry Pi" /sys/firmware/devicetree/base/model; then
  sudo apt-get install chromium xautomation
# For DevBoard
elif grep -q "MX8MQ" /sys/firmware/devicetree/base/model; then
  sudo apt-get install chromium xautomation
else
  echo "Generic Linux system"
fi

cat << EOF | python3
try: import edgetpu
except ImportError: 
  print("\nNo EdgeTPU libs found.")
  print("Follow instructions at https://coral.withgoogle.com/tutorials/accelerator/ to install the EdgeTPU")
EOF

