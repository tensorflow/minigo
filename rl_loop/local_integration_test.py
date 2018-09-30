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

"""Runs a RL loop locally for integration testing.

A successful run will bootstrap, selfplay, shuffle selfplay data, train for
a while, then assert that the generated checkpoint is also playable.
"""

import os
import sys
import tempfile

sys.path.insert(0, '.')

from absl import app, flags
from tensorflow import gfile

import example_buffer as eb
import fsdb
import prep_flags
import shipname
import utils


def main(unused_argv):
    """Run the reinforcement learning loop."""
    utils.ensure_dir_exists(fsdb.models_dir())
    utils.ensure_dir_exists(fsdb.selfplay_dir())
    utils.ensure_dir_exists(fsdb.holdout_dir())
    utils.ensure_dir_exists(fsdb.sgf_dir())
    utils.ensure_dir_exists(fsdb.eval_dir())
    utils.ensure_dir_exists(fsdb.golden_chunk_dir())
    scratch_dir = os.path.join(flags.FLAGS.base_dir, 'scratch')
    utils.ensure_dir_exists(scratch_dir)

    bootstrap_name = shipname.generate(0)
    bootstrap_model_path = os.path.join(fsdb.models_dir(), bootstrap_name)
    prep_flags.checked_run([
        'python', 'bootstrap.py',
        '--export_path={}'.format(bootstrap_model_path),
        '--work_dir={}'.format(scratch_dir),
        '--flagfile=rl_loop/local_flags'])

    selfplay_cmd = [
        'python', 'selfplay.py',
        '--load_file={}'.format(bootstrap_model_path),
        '--selfplay_dir={}'.format(os.path.join(fsdb.selfplay_dir(), bootstrap_name)),
        '--holdout_dir={}'.format(os.path.join(fsdb.holdout_dir(), bootstrap_name)),
        '--sgf_dir={}'.format(fsdb.sgf_dir()),
        '--holdout_pct=0',
        '--flagfile=rl_loop/local_flags']

    # Selfplay twice
    prep_flags.checked_run(selfplay_cmd)
    prep_flags.checked_run(selfplay_cmd)
    # and once more to generate a held out game for validation
    # exploits flags behavior where if you pass flag twice, second one wins.
    prep_flags.checked_run(selfplay_cmd + ['--holdout_pct=100'])

    # Double check that at least one sgf has been generated.
    assert os.listdir(os.path.join(fsdb.sgf_dir(), 'full'))

    print("Making shuffled golden chunk from selfplay data...")
    # TODO(amj): refactor example_buffer so it can be called the same way
    # as everything else.
    eb.make_chunk_for(output_dir=fsdb.golden_chunk_dir(),
                      local_dir=scratch_dir,
                      game_dir=fsdb.selfplay_dir(),
                      model_num=1,
                      positions=64,
                      threads=8,
                      sampling_frac=1)

    tf_records = sorted(gfile.Glob(
        os.path.join(fsdb.golden_chunk_dir(), '*.tfrecord.zz')))

    trained_model_name = shipname.generate(1)
    trained_model_path = os.path.join(fsdb.models_dir(), trained_model_name)

    # Train on shuffled game data
    prep_flags.checked_run([
        'python', 'train.py', *tf_records, 
        '--work_dir={}'.format(scratch_dir),
        '--export_path={}'.format(trained_model_path),
        '--flagfile=rl_loop/local_flags'])

    # Validate the trained model on held out game
    prep_flags.checked_run([
        'python', 'validate.py',
        os.path.join(fsdb.holdout_dir(), bootstrap_name),
        '--work_dir={}'.format(scratch_dir),
        '--flagfile=rl_loop/local_flags'])

    # Verify that trained model works for selfplay
    # exploits flags behavior where if you pass flag twice, second one wins.
    prep_flags.checked_run(
        selfplay_cmd + ['--load_file={}'.format(trained_model_path)])

    prep_flags.checked_run([
        'python', 'evaluate.py',
        bootstrap_model_path, trained_model_path,
        '--games=1',
        '--eval_sgf_dir={}'.format(fsdb.eval_dir()),
        '--flagfile=rl_loop/local_flags'])
    print("Completed integration test!")

if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as base_dir:
        # Hack to create a temp directory and use it as our base_dir
        # before having fsdb.py parse the flags.
        sys.argv.append('--base_dir=' + base_dir)
        app.run(main)
