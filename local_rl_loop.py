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

"""Runs a RL loop locally. Mostly for integration testing purposes.

A successful run will bootstrap, selfplay, gather, and start training for
a while. You should see the combined_cost variable drop steadily, and ideally
overfit to a near-zero loss.
"""

import os
import tempfile

from absl import flags
import dual_net
import main
import selfplay
import example_buffer as eb
import subprocess


def rl_loop():
    """Run the reinforcement learning loop

    This is meant to be more of an integration test than a realistic way to run
    the reinforcement learning.
    """
    # TODO(brilee): move these all into appropriate local_flags file.
    # monkeypatch the hyperparams so that we get a quickly executing network.
    flags.FLAGS.conv_width = 8
    flags.FLAGS.fc_width = 16
    flags.FLAGS.trunk_layers = 1
    flags.FLAGS.train_batch_size = 16
    flags.FLAGS.shuffle_buffer_size = 1000
    dual_net.EXAMPLES_PER_GENERATION = 64

    flags.FLAGS.num_readouts = 10

    with tempfile.TemporaryDirectory() as base_dir:
        bootstrap_model = '000000-bootstrap'

        flags.FLAGS.base_dir = base_dir
        working_dir = os.path.join(base_dir, 'models_in_training')
        flags.FLAGS.model_dir = working_dir
        model_save_path = os.path.join(base_dir, 'models', bootstrap_model)
        local_eb_dir = os.path.join(base_dir, 'scratch')
        next_model_save_file = os.path.join(
            base_dir, 'models', '000001-nextmodel')
        selfplay_dir = os.path.join(base_dir, 'data', 'selfplay')
        model_selfplay_dir = os.path.join(selfplay_dir, bootstrap_model)
        gather_dir = os.path.join(base_dir, 'data', 'training_chunks')
        holdout_dir = os.path.join(
            base_dir, 'data', 'holdout', bootstrap_model)
        sgf_dir = os.path.join(base_dir, 'sgf', bootstrap_model)
        os.makedirs(os.path.join(base_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, bootstrap_model), exist_ok=True)

        print("Creating random initial weights...")
        main.bootstrap(working_dir, model_save_path)
        print("Playing some games...")
        # Do two selfplay runs to test gather functionality
        selfplay.run_game(
            load_file=model_save_path,
            holdout_dir=holdout_dir,
            selfplay_dir=model_selfplay_dir,
            sgf_dir=sgf_dir,
            holdout_pct=0)
        selfplay.run_game(
            load_file=model_save_path,
            holdout_dir=holdout_dir,
            selfplay_dir=model_selfplay_dir,
            sgf_dir=sgf_dir,
            holdout_pct=0)
        # Do one holdout run to test validation
        selfplay.run_game(
            load_file=model_save_path,
            holdout_dir=holdout_dir,
            selfplay_dir=model_selfplay_dir,
            sgf_dir=sgf_dir,
            holdout_pct=100)

        print("SGFs here:")
        sgf_listing = subprocess.check_output(["ls", "-l", sgf_dir + "/full"])
        print(sgf_listing.decode("utf-8"))
        assert len(os.listdir(os.path.join(sgf_dir, "full")))

        print("Gathering game output...")
        eb.make_chunk_for(output_dir=gather_dir,
                          local_dir=local_eb_dir,
                          game_dir=selfplay_dir,
                          model_num=1,
                          positions=dual_net.EXAMPLES_PER_GENERATION,
                          threads=8,
                          sampling_frac=1)

        print("Training on gathered game data...")
        main.train_dir(gather_dir,
                       next_model_save_file)
        print("Trying validate on 'holdout' game...")
        main.validate(holdout_dir)
        print("Verifying that new checkpoint is playable...")
        selfplay.run_game(
            load_file=next_model_save_file,
            holdout_dir=holdout_dir,
            selfplay_dir=model_selfplay_dir,
            sgf_dir=sgf_dir,
            holdout_pct=0)


if __name__ == '__main__':
    # horrible horrible hack to pass flag validation.
    # Problems come from local_rl_loop calling into main() as library calls
    # rather than subprocess calls. Subprocessing calls will allow us to pass
    # flags and have them parsed as normal.
    remaining_argv = flags.FLAGS(['', '--base_dir=foobar'], known_only=True)
    rl_loop()
