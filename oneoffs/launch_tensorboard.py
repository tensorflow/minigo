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
"""Look for estimator-like working dirs and launch tensorboard,

Scrapes a directory with many subdirectories of estimator-like results and calls
tensorboard with a formatted --logdir string for all found subdirectories that
have tensorboard logs in them.

E.g.
Given 'results', and results contains
  results/foo/work_dir
  results/bar/work_dir

It will run:
  tensorboard --logdir=foo:results/foo/work_dir,bar:results/bar/work_dir

Usage: python launch_tensorboard.py ${ROOT_DIR}
"""

from absl import app, flags
import os
import subprocess

flags.DEFINE_integer('port', 5001, 'Port for Tensorboard to listen on.')

FLAGS = flags.FLAGS

def main(argv):
    # It takes a couple of seconds to import anything from tensorflow, so only
    # do it if we need to read from GCS.
    root_dir = argv[1]
    pattern = os.path.join(root_dir, '*', 'work_dir')
    if root_dir.startswith('gs://'):
        from tensorflow import gfile
        dirs = gfile.Glob(pattern)
    else:
        import glob
        dirs = glob.glob(pattern)

    log_dirs = []
    for d in dirs:
        name = os.path.basename(os.path.dirname(d))
        log_dirs.append('{}:{}'.format(name, d))

    cmd = [
        'tensorboard',
        '--port={}'.format(FLAGS.port),
        '--logdir={}'.format(','.join(log_dirs))]
    print(' '.join(cmd))

    url = 'http://localhost:{}/#scalars&tagFilter=policy_(cost%7Centropy)%7Cvalue_(cost_normalized%7Cconfidence)'.format(FLAGS.port)
    print('\nVisit:\n {}\n'.format(url))
    subprocess.run(cmd)

if __name__ == '__main__':
    app.run(main)


