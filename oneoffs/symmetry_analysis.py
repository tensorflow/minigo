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
import json
import os
import sys;
sys.path.insert(0, '.')

import numpy as np
import tensorflow as tf

import dual_net
import features
import sgf_wrapper
import symmetries


def analyze_symmetries(sgf_file, dual_network):
    with open(sgf_file) as f:
        sgf_contents = f.read()

    iterator = sgf_wrapper.replay_sgf(sgf_contents)
    differences = []
    stddevs = []

    # For every move in the game, get the corresponding network values for all
    # eight symmetries.
    for i, pwc in enumerate(iterator):
        feats = features.extract_features(pwc.position)
        variants = [symmetries.apply_symmetry_feat(s, feats)
                    for s in symmetries.SYMMETRIES]
        values = dual_network.sess.run(
            dual_network.inference_output['value_output'],
            feed_dict={dual_network.inference_input: variants})

        # Get the difference between the maximum and minimum outputs of the
        # value network over all eight symmetries; also get the standard
        # deviation of the eight values.
        differences.append(max(values) - min(values))
        stddevs.append(np.std(values))

    differences.sort()
    percentiles = [differences[i * len(differences) // 100] for i in range(100)]
    worst = differences[-1]
    avg_stddev = np.mean(stddevs)
    return (percentiles, worst, avg_stddev)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sgf-folder', type=str,
        help='Path to the folder containing the SGF game replays. The folder '
             'will be searched through recursively for SGF files.')
    parser.add_argument(
        '--load-file', type=str,
        help='Path to the trained model directory to use for analysis.')
    parser.add_argument(
        '--json-file', type=str, default=None,
        help='Optional path to the JSON file where data should be loaded '
             'and/or saved.')
    flags = parser.parse_args()

    if flags.json_file and os.path.isfile(flags.json_file):
        print('')
        print('Loading data from', flags.json_file)

        with open(flags.json_file, 'r') as json_file:
            data = json.load(json_file)
    else:
        data = {
            'percentiles': [],
            'median': [],
            'percentile90': [],
            'worst': [],
            'avg_stddev': []
        }

        dual_network = dual_net.DualNetwork(flags.load_file)

        # Find all .sgf files within flags.sgf_folder.
        for subdir, dirs, files in os.walk(flags.sgf_folder):
            for file in files:
                if file.endswith('.sgf'):
                    sgf_file_path = os.path.join(subdir, file)

                    try:
                        percentiles, worst, avg_stddev = analyze_symmetries(
                            sgf_file_path, dual_network)
                        data['percentiles'].append(percentiles)
                        data['median'].append(percentiles[50])
                        data['percentile90'].append(percentiles[90])
                        data['worst'].append(worst)
                        data['avg_stddev'].append(avg_stddev)
                    except Exception as e:
                        print('')
                        print('Error parsing %s: %s' % (sgf_file_path, e))
                        print('')

        if flags.json_file:
            print('')
            print('Saving data to', flags.json_file)
            os.makedirs(os.path.dirname(flags.json_file))

            with open(flags.json_file, 'w') as json_file:
                # We use a default conversion to float in order to convert numpy
                # floats into Python floats, which JSON can actually serialize.
                json.dump(data, json_file, default=float)

    print('')
    print('%d SGF files parsed. Summary:' % len(data['median']))

    print('Typical symmetry value difference (scale of 0-2):  %.3f' %
          np.mean(data['median']))
    print('Typical 90th percentile symmetry value difference: %.3f' %
          np.mean(data['percentile90']))
    print('Typical worst symmetry value difference:           %.3f' %
          np.mean(data['worst']))
    print('Typical standard deviation over all eight values:  %.3f' %
          np.mean(data['avg_stddev']))
