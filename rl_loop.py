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

"""Wrapper scripts to ensure that main.py commands are called correctly."""
import argh
import argparse
import cloud_logging
import logging
import os
import main
import shipname
import sys
import time
from utils import timer
from tensorflow import gfile

# Pull in environment variables. Run `source ./cluster/common` to set these.
BUCKET_NAME = os.environ['BUCKET_NAME']
BOARD_SIZE = os.environ['BOARD_SIZE']

BASE_DIR = "gs://{}".format(BUCKET_NAME)
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SELFPLAY_DIR = os.path.join(BASE_DIR, 'data/selfplay')
HOLDOUT_DIR = os.path.join(BASE_DIR, 'data/holdout')
SGF_DIR = os.path.join(BASE_DIR, 'sgf')
TRAINING_CHUNK_DIR = os.path.join(BASE_DIR, 'data', 'training_chunks')

ESTIMATOR_WORKING_DIR = 'estimator_working_dir'

# How many games before the selfplay workers will stop trying to play more.
MAX_GAMES_PER_GENERATION = 10000

# What percent of games to holdout from training per generation
HOLDOUT_PCT = 0.05


def print_flags():
    flags = {
        'BUCKET_NAME': BUCKET_NAME,
        'BASE_DIR': BASE_DIR,
        'MODELS_DIR': MODELS_DIR,
        'SELFPLAY_DIR': SELFPLAY_DIR,
        'HOLDOUT_DIR': HOLDOUT_DIR,
        'SGF_DIR': SGF_DIR,
        'TRAINING_CHUNK_DIR': TRAINING_CHUNK_DIR,
        'ESTIMATOR_WORKING_DIR': ESTIMATOR_WORKING_DIR,
        'BOARD_SIZE': BOARD_SIZE,
    }
    print("Computed variables are:")
    print('\n'.join('--{}={}'.format(flag, value)
                    for flag, value in flags.items()))


def get_models():
    """Finds all models, returning a list of model number and names
    sorted increasing.

    Returns: [(13, 000013-modelname), (17, 000017-modelname), ...etc]
    """
    all_models = gfile.Glob(os.path.join(MODELS_DIR, '*.meta'))
    model_filenames = [os.path.basename(m) for m in all_models]
    model_numbers_names = sorted([
        (shipname.detect_model_num(m), shipname.detect_model_name(m))
        for m in model_filenames])
    return model_numbers_names


def get_latest_model():
    """Finds the latest model, returning its model number and name

    Returns: (17, 000017-modelname)
    """
    return get_models()[-1]


def get_model(model_num):
    models = {k: v for k, v in get_models()}
    if not model_num in models:
        raise ValueError("Model {} not found!".format(model_num))
    return models[model_num]


def game_counts(n_back=20):
    """Prints statistics for the most recent n_back models"""
    all_models = gfile.Glob(os.path.join(MODELS_DIR, '*.meta'))
    model_filenames = sorted([os.path.basename(m).split('.')[0]
                              for m in all_models], reverse=True)
    for m in model_filenames[:n_back]:
        games = gfile.Glob(os.path.join(SELFPLAY_DIR, m, '*.zz'))
        print(m, len(games))


def bootstrap():
    bootstrap_name = shipname.generate(0)
    bootstrap_model_path = os.path.join(MODELS_DIR, bootstrap_name)
    print("Bootstrapping with working dir {}\n Model 0 exported to {}".format(
        ESTIMATOR_WORKING_DIR, bootstrap_model_path))
    main.bootstrap(ESTIMATOR_WORKING_DIR, bootstrap_model_path)


def selfplay(readouts=1600, verbose=2, resign_threshold=0.99):
    _, model_name = get_latest_model()
    games = gfile.Glob(os.path.join(SELFPLAY_DIR, model_name, '*.zz'))
    if len(games) > MAX_GAMES_PER_GENERATION:
        print("{} has enough games ({})".format(model_name, len(games)))
        time.sleep(10*60)
        sys.exit(1)
    print("Playing a game with model {}".format(model_name))
    model_save_path = os.path.join(MODELS_DIR, model_name)
    game_output_dir = os.path.join(SELFPLAY_DIR, model_name)
    game_holdout_dir = os.path.join(HOLDOUT_DIR, model_name)
    sgf_dir = os.path.join(SGF_DIR, model_name)
    main.selfplay(
        load_file=model_save_path,
        output_dir=game_output_dir,
        holdout_dir=game_holdout_dir,
        output_sgf=sgf_dir,
        readouts=readouts,
        holdout_pct=HOLDOUT_PCT,
        resign_threshold=resign_threshold,
        verbose=verbose,
    )


def gather():
    print("Gathering game output...")
    main.gather(input_directory=SELFPLAY_DIR,
                output_directory=TRAINING_CHUNK_DIR)


def train():
    model_num, model_name = get_latest_model()
    print("Training on gathered game data, initializing from {}".format(model_name))
    new_model_name = shipname.generate(model_num + 1)
    print("New model will be {}".format(new_model_name))
    load_file = os.path.join(MODELS_DIR, model_name)
    save_file = os.path.join(MODELS_DIR, new_model_name)
    try:
        main.train(ESTIMATOR_WORKING_DIR, TRAINING_CHUNK_DIR, save_file,
                   generation_num=model_num + 1)
    except:
        print("Got an error training, muddling on...")
        logging.exception("Train error")


def validate(model_num=None, validate_name=None):
    """ Runs validate on the directories up to the most recent model, or up to
    (but not including) the model specified by `model_num`
    """
    if model_num is None:
        model_num, model_name = get_latest_model()
    else:
        model_num = int(model_num)
        model_name = get_model(model_num)

    # Model N was trained on games up through model N-2, so the validation set
    # should only be for models through N-2 as well, thus the (model_num - 1)
    # term.
    models = list(
        filter(lambda num_name: num_name[0] < (model_num - 1), get_models()))
    # Run on the most recent 50 generations,
    # TODO(brianklee): make this hyperparameter dependency explicit/not hardcoded
    holdout_dirs = [os.path.join(HOLDOUT_DIR, pair[1])
                    for pair in models[-50:]]

    main.validate(ESTIMATOR_WORKING_DIR, *holdout_dirs,
                  checkpoint_name=os.path.join(MODELS_DIR, model_name),
                  validate_name=validate_name)


def backfill():
    models = [m[1] for m in get_models()]

    import dual_net
    import tensorflow as tf
    from tensorflow.python.framework import meta_graph
    features, labels = dual_net.get_inference_input()
    dual_net.model_fn(features, labels, tf.estimator.ModeKeys.PREDICT,
                      dual_net.get_default_hyperparams())

    for model_name in models:
        if model_name.endswith('-upgrade'):
            continue
        try:
            load_file = os.path.join(MODELS_DIR, model_name)
            dest_file = os.path.join(MODELS_DIR, model_name)
            sess = tf.Session()

            # retrieve the global step as a python value
            ckpt = tf.train.load_checkpoint(load_file)
            global_step_value = ckpt.get_tensor('global_step')

            # restore all saved weights, except global_step
            meta_graph_def = meta_graph.read_meta_graph_file(
                load_file + '.meta')
            stored_var_names = set([n.name
                                    for n in meta_graph_def.graph_def.node
                                    if n.op == 'VariableV2'])
            stored_var_names.remove('global_step')
            var_list = [v for v in tf.global_variables()
                        if v.op.name in stored_var_names]
            tf.train.Saver(var_list=var_list).restore(sess, load_file)

            # manually set the global step
            global_step_tensor = tf.train.get_or_create_global_step()
            assign_op = tf.assign(global_step_tensor, global_step_value)
            sess.run(assign_op)
        except:
            print('failed on', model_name)
            continue

        # export a new savedmodel that has the right global step type
        tf.train.Saver().save(sess, dest_file)


def echo():
    pass  # flags printed in ifmain block below.


parser = argparse.ArgumentParser()

argh.add_commands(parser, [echo, train, selfplay, gather,
                           bootstrap, game_counts, validate, backfill])

if __name__ == '__main__':
    print_flags()
    cloud_logging.configure()
    argh.dispatch(parser)
