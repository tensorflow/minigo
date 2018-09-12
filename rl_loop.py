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

"""Bookkeeping around other entry points to automate the rl pipeline."""
import argparse
import logging
import os
import sys
import time

from absl import flags
import argh
from tensorflow import gfile
import itertools
import random

import bootstrap
import cloud_logging
import dual_net
import fsdb
import selfplay as selfplay_lib
import shipname



# What percent of games to holdout from training per generation
HOLDOUT_PCT = 0.05



def selfplay():
    _, model_name = fsdb.get_latest_model()
    games = gfile.Glob(os.path.join(fsdb.selfplay_dir(), model_name, '*.zz'))
    if len(games) > MAX_GAMES_PER_GENERATION:
        print("{} has enough games ({})".format(model_name, len(games)))
        time.sleep(10 * 60)
        sys.exit(1)
    print("Playing a game with model {}".format(model_name))
    model_save_path = os.path.join(fsdb.models_dir(), model_name)
    selfplay_dir = os.path.join(fsdb.selfplay_dir(), model_name)
    game_holdout_dir = os.path.join(fsdb.holdout_dir(), model_name)
    sgf_dir = os.path.join(fsdb.sgf_dir(), model_name)
    selfplay_lib.run_game(
        load_file=model_save_path,
        selfplay_dir=selfplay_dir,
        holdout_dir=game_holdout_dir,
        sgf_dir=sgf_dir,
        holdout_pct=HOLDOUT_PCT
    )

def backfill():
    models = [m[1] for m in fsdb.get_models()]

    import dual_net
    import tensorflow as tf
    from tqdm import tqdm
    features, labels = dual_net.get_inference_input()
    dual_net.model_fn(features, labels, tf.estimator.ModeKeys.PREDICT,
                      dual_net.get_default_hyperparams())

    for model_name in tqdm(models):
        if model_name.endswith('-upgrade'):
            continue
        try:
            load_file = os.path.join(fsdb.models_dir(), model_name)
            dest_file = os.path.join(fsdb.models_dir(), model_name)
            main.convert(load_file, dest_file)
        except:
            print('failed on', model_name)
            continue


parser = argparse.ArgumentParser()

argh.add_commands(parser, [selfplay, backfill])

if __name__ == '__main__':
    cloud_logging.configure()
    remaining_argv = flags.FLAGS(sys.argv, known_only=True)
    argh.dispatch(parser, argv=remaining_argv[1:])
