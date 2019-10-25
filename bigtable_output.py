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

"""Helpers to write data to Bigtable.
"""

import sgf_wrapper

def process_game(path):
    """
    Get CBT metadata from a SGF file.

    Calling function should probably overwrite 'tool'.
    """
    with open(path) as f:
        sgf_contents = f.read()

    root_node = sgf_wrapper.get_sgf_root_node(sgf_contents)
    assert root_node.properties['FF'] == ['4'], ("Bad game record", path)

    result = root_node.properties['RE'][0]
    assert result.lower()[0] in 'bw', result
    assert result.lower()[1] == '+', result
    black_won = result.lower()[0] == 'b'

    length = 0
    node = root_node.next
    while node:
        props = node.properties
        length += 1 if props.get('B') or props.get('W') else 0
        node = node.next

    return {
        "black": root_node.properties['PB'][0],
        "white": root_node.properties['PW'][0],
        # All values are strings, "1" for true and "0" for false here
        "black_won": '1' if black_won else '0',
        "white_won": '0' if black_won else '1',
        "result": result,
        "length": str(length),
        "sgf": path,
        "tag": "",
        "tool": "bigtable_output",
    }
