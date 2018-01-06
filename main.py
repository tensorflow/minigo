import argparse
import argh
import os.path
import random
import re
import sys
import time
import logging
import google.cloud.logging as glog
from tqdm import tqdm
import gzip
import numpy as np

from go import IllegalMove, replay_position
from gtp_wrapper import make_gtp_instance, MCTSPlayer
from load_data_sets import DataSetV2
import selfplay_mcts
from utils import logged_timer as timer
import ds_wrangler
import evaluation
import sgf_wrapper

def gtp(load_file: "The path to the network model files",
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

def train(processed_dir, load_file=None, save_file=None,
          epochs=1, logdir=None, batch_size=64):
    from dual_net import DualNetwork
    train_chunk_files = [os.path.join(processed_dir, fname)
                         for fname in os.listdir(processed_dir)
                         if fname.endswith('.gz')]
    save_file = os.path.join(os.getcwd(), save_file)
    if load_file is not None:
        load_file = os.path.join(os.getcwd(), os.path.abspath(load_file))
    n = DualNetwork()
    try:
        n.initialize_variables(load_file)
    except:
        n.initialize_variables(None)
    if logdir is not None:
        n.initialize_logging(logdir)
    last_save_checkpoint = 0
    with timer("Training"):
        for i in range(epochs):
            random.shuffle(train_chunk_files)
            for i, file_ in enumerate(tqdm(train_chunk_files)):
                train_dataset = DataSetV2.read(file_)
                train_dataset.shuffle()
                n.train(train_dataset, batch_size)
                if i % 10 == 0:
                    n.save_variables(save_file)
            n.save_variables(save_file)

def evaluate(
        black_model: 'The path to the model to play black',
        white_model: 'The path to the model to play white',
        output_dir: 'Where to write the evaluation results'='data/evaluate/sgf',
        readouts: 'How many readouts to make per move.'=400,
        games: 'the number of games to play'=16,
        use_cpu: 'passed to the network initializer'=False,
        verbose: 'How verbose the players should be (see selfplay)' = 1):
    from dual_net import DualNetwork

    black_model = os.path.join(os.getcwd(), os.path.abspath(black_model))
    white_model = os.path.join(os.getcwd(), os.path.abspath(white_model))

    import tensorflow as tf
    g1 = tf.Graph()
    g2 = tf.Graph()

    with timer("Loading weights"):
        try:
            with g1.as_default():
                black_net = DualNetwork(use_cpu=use_cpu)
                black_net.initialize_variables(black_model) 
            with g2.as_default():
                white_net = DualNetwork(use_cpu=use_cpu)
                white_net.initialize_variables(white_model)
        except:
            print("*** Unable to initialize networks")
            raise

    with timer("%d games" % games):
        players = evaluation.play_match(black_net, white_net, games, readouts, verbose)

    for idx,p in enumerate(players):
        try:
            fname ="{:s}-vs-{:s}-{:d}".format(black_net.name, white_net.name, idx)
            with open(os.path.join(output_dir, fname + '.sgf'), 'w') as f:
                f.write(sgf_wrapper.make_sgf(p[0].position.recent,
                                             p[0].make_result_string(p[0].position),
                                             black_name=os.path.basename(black_model),
                                             white_name=os.path.basename(white_model)))
        except IllegalMove:
            print("player in game #%d played an illegal move!" % idx)

def selfplay(
         load_file: "The path to the network model files",
         output_dir: "Where to write the games"="data/selfplay",
         output_sgf: "Where to write the sgfs"="sgf/",
         readouts: 'How many simulations to run per move'=100,
         games: 'Number of games to play' = 4,
         verbose : '>1 will print debug info, >2 will print boards' = 1,
         resign_threshold : 'absolute value of threshold to resign at' = 0.95,
         use_cpu: 'passed to the network initializer'=False):

    from dual_net import DualNetwork
    print ("Initializing network...", flush=True)
    network = DualNetwork(use_cpu=use_cpu)

    if load_file is not None:
        load_file = os.path.join(os.getcwd(), os.path.abspath(load_file))
    try:
        with timer("Loading weights from %s ... " % load_file):
            network.initialize_variables(load_file)
            network.name = os.path.basename(load_file)
    except:
        print("*** Unable to initialize network: %s" % load_file)
        network.initialize_variables(None)

    with timer("%d games" % games):
        players = selfplay_mcts.play(network, games, readouts, resign_threshold, verbose)

    ds = None
    for idx,p in enumerate(players):
        try:
            pwcs, pis, results = p.to_dataset()
            if idx == 0:
                ds = DataSetV2.from_positions_w_context(pwcs, pis, results)
            #TODO: this does a lot of needless copying (O(n^2))... fix eventually
            else:
                ds.extend(DataSetV2.from_positions_w_context(pwcs, pis, results))
            fname ="{:d}-{:d}".format(int(time.time()), idx)
            with open(os.path.join(output_sgf, fname + '.sgf'), 'w') as f:
                f.write(p.to_sgf())
        except IllegalMove:
            print("player #%d played an illegal move!" % idx)

    fname ="{:d}".format(int(time.time()))
    ds.write(os.path.join(output_dir, fname + '.gz'))
    ds.write_meta(os.path.join(output_dir, fname +'.meta'))

def gather(
        input_directory: 'where to look for games'='data/selfplay/',
        output_directory: 'where to put collected games'='data/training_chunks/',
        max_positions: 'how many positions before discarding games'= 125000000,
        positions_per_chunk: 'how many positions (samples) to collect per chunk.'=2048,
        chunks_to_make: 'how many chunks to create thru repeated sampling'=600):
    paths = [(root, dirs, files) for root, dirs, files in os.walk(input_directory)]

    meta_paths = []
    with timer("Walking list of gamedata directories"):
        for root, _, files in os.walk(input_directory):
            for f in files:
                if f.endswith('.meta'):
                    meta_paths.append(os.path.join(root,f))
    print("Found %d meta files" % len(meta_paths))

    paths_to_sizes = ds_wrangler.get_paths_to_num_positions(meta_paths, max_positions)

    if input_directory.endswith('/'):
        input_directory = input_directory.strip('/') # Support 'dir/' or 'dir' as params
    pattern = '%s/(.*?)/' % input_directory
    unique_models = set([re.search(pattern, k).group(1) for k in paths_to_sizes if re.search(pattern, k)])
    print("Got models: \n\t" + "\n\t".join(unique_models))

    cumulative_moves = np.cumsum([v for k,v in sorted(paths_to_sizes.items(), reverse=True)])
    reversed_paths = [(p.replace('.meta', '.gz'))
                     for p in sorted(paths_to_sizes, reverse=True)]
    # Now we've got a dictionary of {cumulative position number: /path/to/game.gz} items. 
    with timer("Choosing %d moves for %d chunks" % (positions_per_chunk, chunks_to_make)):
        paths_to_moves_by_chunk = ds_wrangler.choose_moves_for_chunks(
                cumulative_moves,
                reversed_paths,
                chunks_to_make,
                positions_per_chunk)

    with timer("Gathering moves from files"):
        ds_wrangler.gather_moves_and_write(
                paths_to_moves_by_chunk, chunks_to_make, output_directory)

parser = argparse.ArgumentParser()
argh.add_commands(parser, [gtp, train, selfplay, gather, evaluate])

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        client = glog.Client('tensor-go')
        client.setup_logging(logging.INFO)
    except:
        print('!! Cloud logging disabled')
    argh.dispatch(parser)
