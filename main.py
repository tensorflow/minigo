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

import argh
import argparse
import os.path
import random
import socket
import sys
import tempfile
import time

import dual_net
import evaluation
import preprocessing
import selfplay_mcts
from gtp_wrapper import make_gtp_instance
import utils

import cloud_logging
import tensorflow as tf
from absl import flags
from tqdm import tqdm
from tensorflow import gfile

# How many positions we should aggregate per 'chunk'.
EXAMPLES_PER_RECORD = 10000

# How many positions to draw from for our training window.
# AGZ used the most recent 500k games, which, assuming 250 moves/game = 125M
WINDOW_SIZE = 125000000


def gtp(load_file: 'The path to the network model files'=None,
        cgos_mode: 'Whether to use CGOS time constraints'=False,
        kgs_mode: 'Whether to use KGS courtesy-pass'=False,
        verbose=1):
    engine = make_gtp_instance(load_file,
                               verbosity=verbose,
                               cgos_mode=cgos_mode,
                               kgs_mode=kgs_mode)
    print("GTP engine ready\n", file=sys.stderr, flush=True)
    for msg in sys.stdin:
        if not engine.handle_msg(msg.strip()):
            break


def bootstrap(
        working_dir: 'tf.estimator working directory. If not set, defaults to a random tmp dir'=None,
        model_save_path: 'Where to export the first bootstrapped generation'=None):
    if working_dir is None:
        with tempfile.TemporaryDirectory() as working_dir:
            utils.ensure_dir_exists(working_dir)
            utils.ensure_dir_exists(os.path.dirname(model_save_path))
            dual_net.bootstrap(working_dir)
            dual_net.export_model(working_dir, model_save_path)
    else:
        utils.ensure_dir_exists(working_dir)
        utils.ensure_dir_exists(os.path.dirname(model_save_path))
        dual_net.bootstrap(working_dir)
        dual_net.export_model(working_dir, model_save_path)
        freeze_graph(model_save_path)


def train_dir(
        working_dir: 'tf.estimator working directory.',
        chunk_dir: 'Directory where training chunks are.',
        model_save_path: 'Where to export the completed generation.'):
    tf_records = sorted(gfile.Glob(os.path.join(chunk_dir, '*.tfrecord.zz')))
    tf_records = tf_records[-1 * (WINDOW_SIZE // EXAMPLES_PER_RECORD):]

    train(working_dir, tf_records, model_save_path)


def train(
        working_dir: 'tf.estimator working directory.',
        tf_records: 'list of files of tf_records to train on',
        model_save_path: 'Where to export the completed generation.'):
    print("Training on:", tf_records[0], "to", tf_records[-1])
    with utils.logged_timer("Training"):
        dual_net.train(working_dir, tf_records) 
    print("== Training done.  Exporting model to ", model_save_path)
    dual_net.export_model(working_dir, model_save_path)
    freeze_graph(model_save_path)


def validate(
        working_dir: 'tf.estimator working directory',
        *tf_record_dirs: 'Directories where holdout data are',
        checkpoint_name: 'Which checkpoint to evaluate (None=latest)'=None,
        validate_name: 'Name for validation set (i.e., selfplay or human)'=None):
    tf_records = []
    with utils.logged_timer("Building lists of holdout files"):
        for record_dir in tf_record_dirs:
            tf_records.extend(gfile.Glob(os.path.join(record_dir, '*.zz')))

    first_record = os.path.basename(tf_records[0])
    last_record = os.path.basename(tf_records[-1])
    with utils.logged_timer("Validating from {} to {}".format(first_record, last_record)):
        dual_net.validate(
            working_dir, tf_records, checkpoint_name=checkpoint_name,
            name=validate_name)


def evaluate(
        black_model: 'The path to the model to play black',
        white_model: 'The path to the model to play white',
        output_dir: 'Where to write the evaluation results'='sgf/evaluate',
        games: 'the number of games to play'=16,
        verbose: 'How verbose the players should be (see selfplay)' = 1):
    utils.ensure_dir_exists(output_dir)

    with utils.logged_timer("Loading weights"):
        black_net = dual_net.DualNetwork(black_model)
        white_net = dual_net.DualNetwork(white_model)

    with utils.logged_timer("Playing game"):
        evaluation.play_match(
            black_net, white_net, games, output_dir, verbose)


def selfplay(
        load_file: "The path to the network model files",
        output_dir: "Where to write the games"="data/selfplay",
        holdout_dir: "Where to write the games"="data/holdout",
        output_sgf: "Where to write the sgfs"="sgf/",
        verbose: '>=2 will print debug info, >=3 will print boards' = 1,
        holdout_pct: 'how many games to hold out for validation' = 0.05):
    clean_sgf = os.path.join(output_sgf, 'clean')
    full_sgf = os.path.join(output_sgf, 'full')
    utils.ensure_dir_exists(clean_sgf)
    utils.ensure_dir_exists(full_sgf)
    utils.ensure_dir_exists(output_dir)
    utils.ensure_dir_exists(holdout_dir)

    with utils.logged_timer("Loading weights from %s ... " % load_file):
        network = dual_net.DualNetwork(load_file)

    with utils.logged_timer("Playing game"):
        player = selfplay_mcts.play(network, verbose)

    output_name = '{}-{}'.format(int(time.time()), socket.gethostname())
    game_data = player.extract_data()
    with gfile.GFile(os.path.join(clean_sgf, '{}.sgf'.format(output_name)), 'w') as f:
        f.write(player.to_sgf(use_comments=False))
    with gfile.GFile(os.path.join(full_sgf, '{}.sgf'.format(output_name)), 'w') as f:
        f.write(player.to_sgf())

    tf_examples = preprocessing.make_dataset_from_selfplay(game_data)

    # Hold out 5% of games for evaluation.
    if random.random() < holdout_pct:
        fname = os.path.join(holdout_dir, "{}.tfrecord.zz".format(output_name))
    else:
        fname = os.path.join(output_dir, "{}.tfrecord.zz".format(output_name))

    preprocessing.write_tf_examples(fname, tf_examples)


def convert(load_file, dest_file):
    from tensorflow.python.framework import meta_graph
    features, labels = dual_net.get_inference_input()
    dual_net.model_fn(features, labels, tf.estimator.ModeKeys.PREDICT,
                      dual_net.get_default_hyperparams())
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

    # export a new savedmodel that has the right global step type
    tf.train.Saver().save(sess, dest_file)
    sess.close()
    tf.reset_default_graph()


def freeze_graph(load_file):
    """ Loads a network and serializes just the inference parts for use by e.g. the C++ binary """
    n = dual_net.DualNetwork(load_file)
    out_graph = tf.graph_util.convert_variables_to_constants(
        n.sess, n.sess.graph.as_graph_def(), ["policy_output", "value_output"])
    with gfile.GFile(os.path.join(load_file + '.pb'), 'wb') as f:
        f.write(out_graph.SerializeToString())


parser = argparse.ArgumentParser()
argh.add_commands(parser, [gtp, bootstrap, train, train_dir, freeze_graph,
                           selfplay, evaluate, validate, convert])

if __name__ == '__main__':
    cloud_logging.configure()
    # Let absl.flags parse known flags from argv, then pass the remaining flags
    # into argh for dispatching.
    remaining_argv = flags.FLAGS(sys.argv, known_only=True)
    argh.dispatch(parser, argv=remaining_argv[1:])
