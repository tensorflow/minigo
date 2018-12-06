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
import os
import sys
import time
from absl import app, flags
sys.path.insert(0, '.')

from rl_loop import fsdb
import mask_flags

# From rl_loop/fsdb.py
flags.declare_key_flag('bucket_name')

# "_nr" signifies "No Resign", aka calibration game, which will use a different
# set of flags and will not update its flags from a remote flagfile.
flags.DEFINE_enum('mode', None, ['cc', 'tpu', 'tpu_nr'],
                  'Which setup to use: cc on GPU or cc/py on TPU.')

flags.DEFINE_string('output_bigtable', '', 'Bigtable output specification')

FLAGS = flags.FLAGS


def run_cc():
    _, model_name = fsdb.get_latest_model()
    num_games_finished = len(fsdb.get_games(model_name))
    if num_games_finished > 25000:
        print("{} has enough games! ({})".format(
            model_name, num_games_finished))
        time.sleep(10 * 60)
        sys.exit()

    mask_flags.checked_run([
        'bazel-bin/cc/main',
        '--model={}'.format(model_name),
        '--mode=selfplay',
        '--engine=tf',
        '--output_dir={}/{}'.format(
            fsdb.selfplay_dir(), model_name),
        '--holdout_dir={}/{}'.format(
            fsdb.holdout_dir(), model_name),
        '--sgf_dir={}/{}'.format(
            fsdb.sgf_dir(), model_name),
        '--flagfile=rl_loop/distributed_flags'])


def run_tpu(no_resign=False):
    os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = '/etc/ssl/certs/ca-certificates.crt'
    flagset = [
        'bazel-bin/cc/main',
        '--mode=selfplay',
        '--engine=tpu',
        '--model={}'.format(os.path.join(fsdb.working_dir(), 'model.ckpt-%d.pb')),
        '--output_dir={}'.format(fsdb.selfplay_dir()),
        '--holdout_dir={}'.format(fsdb.holdout_dir()),
        '--sgf_dir={}'.format(fsdb.sgf_dir()),
        '--run_forever=true',
        '--output_bigtable={}'.format(FLAGS.output_bigtable)]

    if 'KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS' in os.environ:
        flagset.append('--tpu_name={}'.format(os.environ['KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS']))

    if no_resign:
        flagset.extend([
            '--flagfile=rl_loop/distributed_flags_nr'])
    else:
        flagset.extend([
            '--flags_path={}'.format(fsdb.flags_path()),
            '--flagfile=rl_loop/distributed_flags'])

    mask_flags.checked_run(flagset)

def main(unused_argv):
    flags.mark_flags_as_required(['bucket_name', 'mode'])
    if FLAGS.mode == 'cc':
        run_cc()
    elif FLAGS.mode == 'tpu':
        run_tpu(no_resign=False)
    elif FLAGS.mode == 'tpu_nr':
        run_tpu(no_resign=True)


if __name__ == '__main__':
    app.run(main)
