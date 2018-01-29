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

import argparse
import argh
import os
import itertools
import re
import numpy as np

def crawl(sgf_directory='sgf', print_summary=True): 
    max_w_upset = {'value': 0}
    max_b_upset = {'value': 0}

    worst_qs = []
    for root, _, filenames in os.walk(sgf_directory):
        for filename in filenames:
            if not filename.endswith('.sgf'):
                continue

            data = open(os.path.join(root, filename)).read()
            result = re.search("RE\[([BWbw])\+", data)
            if not result:
                print("No result string found in sgf: ", filename)
                continue
            else:
                result = result.group(1)

            q_values = list(map(float, re.findall("C\[(-?\d.\d*)", data)))
            if result == "B":
                print("%s:%s+:%s" % (filename, result, min(q_values)))
                worst_qs.append(min(q_values))
                if min(q_values) < max_b_upset['value']:
                    max_b_upset = {"filename": os.path.join(root, filename),
                                   "value": min(q_values)}
            else:
                print("%s:%s+:%s" % (filename, result, max(q_values))) 
                worst_qs.append(max(q_values))
                if max(q_values) > max_w_upset['value']:
                    max_w_upset = {"filename": os.path.join(root, filename),
                                   "value": max(q_values)}


    if print_summary:
        b_upsets = np.array([q for q in worst_qs if q < 0])
        w_upsets = np.array([q for q in worst_qs if q > 0])
        both = np.array(list(map(abs, worst_qs)))
        print("Biggest w upset:", max_w_upset)
        print("Biggest b upset:", max_b_upset)

        print ("99th percentiles (both/w/b)")
        print(np.percentile(both, 99))
        print(np.percentile(b_upsets, 1))
        print(np.percentile(w_upsets, 99))


if __name__ == '__main__':
    argh.dispatch_command(crawl)
