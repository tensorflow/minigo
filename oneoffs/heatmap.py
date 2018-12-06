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

"""
Used to plot a heatmap of the policy and value networks.
Check FLAGS for default values.

Usage:
python heatmap.py

"""
import sys
sys.path.insert(0, '.')

import os

import tensorflow as tf
from absl import app, flags
from tqdm import tqdm

import go
from rl_loop import fsdb
import oneoff_utils


flags.DEFINE_string("sgf_dir", "sgf/baduk_db/", "sgf database.")
flags.DEFINE_string("data_dir", "data/eval", "Where to save data.")
flags.DEFINE_integer("idx_start", 150, "Only take models after given idx.")
flags.DEFINE_integer("eval_every", 5, "Eval every k models.")

FLAGS = flags.FLAGS


def eval_policy(eval_positions):
    """Evaluate all positions with all models save the policy heatmaps as CSVs

    CSV name is "heatmap-<position_name>-<model-index>.csv"
    CSV format is: model number, value network output, policy network outputs

    position_name is taken from the SGF file
    Policy network outputs (19x19) are saved in flat order (see coord.from_flat)
    """

    model_paths = oneoff_utils.get_model_paths(fsdb.models_dir())

    idx_start = FLAGS.idx_start
    eval_every = FLAGS.eval_every

    print("Evaluating models {}-{}, eval_every={}".format(
          idx_start, len(model_paths), eval_every))

    player = None
    for i, idx in enumerate(tqdm(range(idx_start, len(model_paths), eval_every))):
        if player and i % 20 == 0:
            player.network.sess.close()
            tf.reset_default_graph()
            player = None

        if not player:
            player = oneoff_utils.load_player(model_paths[idx])
        else:
            oneoff_utils.restore_params(model_paths[idx], player)

        pos_names, positions = zip(*eval_positions)
        # This should be batched at somepoint.
        eval_probs, eval_values = player.network.run_many(positions)

        for pos_name, probs, value in zip(pos_names, eval_probs, eval_values):
            save_file = os.path.join(
                FLAGS.data_dir, "heatmap-{}-{}.csv".format(pos_name, idx))

            with open(save_file, "w") as data:
                data.write("{},  {},  {}\n".format(
                    idx, value, ",".join(map(str, probs))))


def positions_from_sgfs(sgf_files, include_empty=True):
    positions = []
    if include_empty:
        # sgf_replay doesn't like SGFs with no moves played.
        # Add the empty position for analysis manually.
        positions.append(("empty", go.Position(komi=7.5)))

    for sgf in sgf_files:
        sgf_name = os.path.basename(sgf).replace(".sgf", "")
        final = oneoff_utils.final_position_sgf(sgf)
        positions.append((sgf_name, final))
    return positions


def main(unusedargv):
    sgf_files = oneoff_utils.find_and_filter_sgf_files(FLAGS.sgf_dir)
    eval_positions = positions_from_sgfs(sgf_files)

    eval_policy(eval_positions)


if __name__ == "__main__":
    app.run(main)
