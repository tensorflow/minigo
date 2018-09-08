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
import sys
import tempfile

import dual_net
import evaluation
import utils

import cloud_logging
import tensorflow as tf
from absl import flags
from tensorflow import gfile

# How many positions we should aggregate per 'chunk'.
EXAMPLES_PER_RECORD = 10000

# How many positions to draw from for our training window.
# AGZ used the most recent 500k games, which, assuming 250 moves/game = 125M
WINDOW_SIZE = 125000000


def bootstrap(
        model_save_path: 'Where to export the first bootstrapped generation'=None):
    utils.ensure_dir_exists(os.path.dirname(model_save_path))
    dual_net.bootstrap()
    dual_net.export_model(model_save_path)


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


def convert(load_file, dest_file):
    from tensorflow.python.framework import meta_graph
    features, labels = dual_net.get_inference_input()
    dual_net.model_fn(features, labels, tf.estimator.ModeKeys.PREDICT)
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




parser = argparse.ArgumentParser()
argh.add_commands(parser, [bootstrap, evaluate, convert])

if __name__ == '__main__':
    cloud_logging.configure()
    # Let absl.flags parse known flags from argv, then pass the remaining flags
    # into argh for dispatching.
    remaining_argv = flags.FLAGS(sys.argv, known_only=True)
    argh.dispatch(parser, argv=remaining_argv[1:])
