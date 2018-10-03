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
Used to find PV for position.
Check FLAGS for default values.

Usage:
python position_pv.py

"""
import sys
sys.path.insert(0, '.')

import os

from absl import app, flags
import numpy as np
from tqdm import tqdm

import go
from rl_loop import fsdb
import oneoff_utils
import strategies


flags.DEFINE_string("sgf_dir", "sgf/baduk_db/", "sgf database.")
flags.DEFINE_string("data_dir", "data/eval", "Where to save data.")
flags.DEFINE_integer("idx_start", 150, "Only take models after given idx.")
flags.DEFINE_integer("eval_every", 5, "Eval every k models.")

FLAGS = flags.FLAGS


def eval_pv(eval_positions):
    model_paths = oneoff_utils.get_model_paths(fsdb.models_dir())

    idx_start = FLAGS.idx_start
    eval_every = FLAGS.eval_every

    print("Evaluating models {}-{}, eval_every={}".format(
          idx_start, len(model_paths), eval_every))
    for idx in tqdm(range(idx_start, len(model_paths), eval_every)):
        if idx == idx_start:
            player = oneoff_utils.load_player(model_paths[idx])
        else:
            oneoff_utils.restore_params(model_paths[idx], player)

        mcts = strategies.MCTSPlayer(
            player.network,
            resign_threshold=-1)

        for name, position in eval_positions:
            mcts.initialize_game(position)
            mcts.suggest_move(position)

            path = []
            node = mcts.root
            while node.children:
                node = node.children.get(node.best_child())
                path.append("{},{}".format(node.fmove, int(node.N)))

            save_file = os.path.join(
                FLAGS.data_dir, "pv-{}-{}".format(name, idx))
            with open(save_file, "w") as data:
                data.write("{},  {}\n".format(idx, ",".join(path)))


def positions_from_sgfs(sgf_files):
    # sgf_replay doesn't like empty sgf file
    data = [("empty", go.Position(komi=7.5))]
    for sgf in sgf_files:
        sgf_name = os.path.basename(sgf).replace(".sgf", "")
        positions, moves, _ = oneoff_utils.parse_sgf(sgf)
        final = positions[-1].play_move(moves[-1])
        data.append((sgf_name, final))
    return data


def main(unusedargv):
    sgf_files = oneoff_utils.find_and_filter_sgf_files(FLAGS.sgf_dir)
    eval_positions = positions_from_sgfs(sgf_files)

    eval_pv(eval_positions)


if __name__ == "__main__":
    app.run(main)
