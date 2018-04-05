"""
Used to plot a heatmap of the policy and value networks.
Check FLAGS for default values for what models to load.

Usage:
python heatmap.py

"""
import sys
sys.path.insert(0, '.')

import itertools
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import go
import oneoff_utils


tf.app.flags.DEFINE_string(
    "sgf_dir", "sgf/baduk_db/", "Path to directory of sgf to analyze")

tf.app.flags.DEFINE_string("model_dir", "saved_models",
                           "Where the model files are saved")
tf.app.flags.DEFINE_string("data_dir", "data/eval", "Where to save data")
tf.app.flags.DEFINE_integer("idx_start", 150,
                            "Only take models after given idx")
tf.app.flags.DEFINE_integer("eval_every", 5,
                            "Eval every k models to generate the curve")

FLAGS = tf.app.flags.FLAGS


def eval_for_policy(eval_positions, model_dir, data_dir, idx_start, eval_every):
    """Evaluate all positions with all models save the policy heatmaps as CSVs

    CSV name is "heatmap-<position_name>-<model-index>.csv"
    CSV format is: model number, value network output, policy network outputs

    position_name is taken from the SGF file
    Policy network outputs (19x19) are saved in flat order (see coord.from_flat)
    """

    model_paths = oneoff_utils.get_model_paths(model_dir)

    print("Evaluating models {}-{}, eval_every={}".format(
          idx_start, len(model_paths), eval_every))
    for idx in tqdm(range(idx_start, len(model_paths), eval_every)):
        if idx == idx_start:
            player = oneoff_utils.load_player(model_paths[idx])
        else:
            oneoff_utils.restore_params(model_paths[idx], player)

        pos_names, positions = zip(*eval_positions)
        # This should be batched at somepoint.
        eval_probs, eval_values = player.network.run_many(positions)

        for pos_name, probs, value in zip(pos_names, eval_probs, eval_values):
            save_file = os.path.join(
                data_dir, "heatmap-{}-{}.csv".format(pos_name, idx))

            with open(save_file, "w") as data:
                data.write("{},  {},  {}\n".format(
                    idx, value, ",".join(map(str, probs))))


def positions_from_sgfs(sgf_files, include_empty_position=True):
    data = []
    if include_empty:
        # sgf_replay doesn't like SGFs with no moves played.
        # Add the empty position for analysis manually.
        data.append(("empty", go.Position(komi=7.5)))
    for sgf in sgf_files:
        sgf_name = os.path.basename(sgf).replace(".sgf", "")
        positions, moves, _ = oneoff_utils.parse_sgf(sgf)
        final = positions[-1].play_move(moves[-1])
        data.append((sgf_name, final))
    return data


def main(unusedargv):
    sgf_files = oneoff_utils.find_and_filter_sgf_files(FLAGS.sgf_dir)
    eval_positions = positions_from_sgfs(sgf_files)

    eval_for_policy(
        eval_positions,
        FLAGS.model_dir,
        FLAGS.data_dir,
        FLAGS.idx_start,
        FLAGS.eval_every)


FLAGS = tf.app.flags.FLAGS

if __name__ == "__main__":
    tf.app.run(main)
