import argparse
import argh
import os.path
import collections
import random
import re
import socket
import sys
import time
import cloud_logging
from tqdm import tqdm
import gzip
import numpy as np
import tensorflow as tf
from tensorflow import gfile

import go
import dual_net
from gtp_wrapper import make_gtp_instance, MCTSPlayer
import preprocessing
import selfplay_mcts
from utils import logged_timer as timer
import evaluation
import sgf_wrapper
import utils


def _ensure_dir_exists(directory):
    if directory.startswith('gs://'):
        return
    os.makedirs(directory, exist_ok=True)

def gtp(load_file: "The path to the network model files"=None,
        readouts: 'How many simulations to run per move'=100,
        cgos_mode: 'Whether to use CGOS time constraints'=False,
        verbose=1):
    engine = make_gtp_instance(load_file,
                               readouts_per_move=readouts,
                               verbosity=verbose,
                               cgos_mode=cgos_mode)
    sys.stderr.write("GTP engine ready\n")
    sys.stderr.flush()
    while not engine.disconnect:
        inpt = input()
        # handle either single lines at a time
        # or multiple commands separated by '\n'
        try:
            cmd_list = inpt.split("\n")
        except:
            cmd_list = [inpt]
        for cmd in cmd_list:
            engine_reply = engine.send(cmd)
            sys.stdout.write(engine_reply)
            sys.stdout.flush()

def bootstrap(save_file):
    dual_net.DualNetworkTrainer(save_file).bootstrap()

def train(chunk_dir, save_file, load_file=None, generation_num=0,
          logdir=None, num_steps=None):
    tf_records = gfile.Glob(os.path.join(chunk_dir, '*.tfrecord.zz'))
    tf_records = [f for f in tf_records
        if (generation_num - 50) < int(os.path.basename(f)[:6]) <= generation_num]

    n = dual_net.DualNetworkTrainer(save_file)
    with timer("Training"):
        n.train(tf_records, init_from=load_file, logdir=logdir, num_steps=num_steps)

def evaluate(
        black_model: 'The path to the model to play black',
        white_model: 'The path to the model to play white',
        output_dir: 'Where to write the evaluation results'='data/evaluate/sgf',
        readouts: 'How many readouts to make per move.'=400,
        games: 'the number of games to play'=16,
        verbose: 'How verbose the players should be (see selfplay)' = 1):

    black_model = os.path.abspath(black_model)
    white_model = os.path.abspath(white_model)

    with timer("Loading weights"):
        black_net = dual_net.DualNetwork(black_model)
        white_net = dual_net.DualNetwork(white_model)

    with timer("%d games" % games):
        players = evaluation.play_match(black_net, white_net, games, readouts, verbose)

    for idx,p in enumerate(players):
        fname ="{:s}-vs-{:s}-{:d}".format(black_net.name, white_net.name, idx)
        with open(os.path.join(output_dir, fname + '.sgf'), 'w') as f:
            f.write(sgf_wrapper.make_sgf(p[0].position.recent,
                                         p[0].make_result_string(p[0].position),
                                         black_name=os.path.basename(black_model),
                                         white_name=os.path.basename(white_model)))

def selfplay(
         load_file: "The path to the network model files",
         output_dir: "Where to write the games"="data/selfplay",
         output_sgf: "Where to write the sgfs"="sgf/",
         readouts: 'How many simulations to run per move'=100,
         verbose : '>=2 will print debug info, >=3 will print boards' = 1,
         resign_threshold : 'absolute value of threshold to resign at' = 0.95):
    _ensure_dir_exists(output_sgf)
    _ensure_dir_exists(output_dir)

    with timer("Loading weights from %s ... " % load_file):
        network = dual_net.DualNetwork(load_file)
        network.name = os.path.basename(load_file)

    with timer("Playing game"):
        player = selfplay_mcts.play(network, readouts, resign_threshold, verbose)

    output_name = '{}-{}'.format(int(time.time()), socket.gethostname())
    game_data = player.extract_data()
    with gfile.GFile(os.path.join(output_sgf, '{}.sgf'.format(output_name)), 'w') as f:
        f.write(player.to_sgf())

    fname = os.path.join(output_dir, "{}.tfrecord.zz".format(output_name))
    preprocessing.make_dataset_from_selfplay(game_data, fname)

def gather(
        input_directory: 'where to look for games'='data/selfplay/',
        output_directory: 'where to put collected games'='data/training_chunks/',
        examples_per_record: 'how many tf.examples to gather in each chunk'=20000):
    _ensure_dir_exists(output_directory)
    models = [model_dir.strip('/') for model_dir in gfile.ListDirectory(input_directory)]
    with timer("Finding existing tfrecords..."):
        model_gamedata = {
            model: gfile.Glob(
                os.path.join(input_directory, model, '**', '*.tfrecord.zz'))
            for model in models
        }
    print("Found %d models" % len(models))
    for model_name, record_files in sorted(model_gamedata.items()):
        print("    %s: %s files" % (model_name, len(record_files)))

    meta_file = os.path.join(output_directory, 'meta.txt')
    try:
        with gfile.GFile(meta_file, 'r') as f:
            already_processed = set(f.read().split())
    except tf.errors.NotFoundError:
        already_processed = set()

    num_already_processed = len(already_processed)

    for model_name, record_files in sorted(model_gamedata.items()):
        with timer("Processing %s" % model_name):
            if set(record_files) <= already_processed:
                print("%s is already fully processed" % model_name)
                continue
            for i, example_batch in enumerate(
                    tqdm(preprocessing.shuffle_tf_examples(examples_per_record, record_files))):
                output_record = os.path.join(output_directory,
                    '{}-{}.tfrecord.zz'.format(model_name, str(i)))
                preprocessing.write_tf_examples(output_record, example_batch, serialize=False)
            already_processed.update(record_files)

    print("Processed %s new files" % (len(already_processed) - num_already_processed))
    with gfile.GFile(meta_file, 'w') as f:
        f.write('\n'.join(sorted(already_processed)))

parser = argparse.ArgumentParser()
argh.add_commands(parser, [gtp, bootstrap, train, selfplay, gather, evaluate])

if __name__ == '__main__':
    cloud_logging.configure()
    argh.dispatch(parser)
