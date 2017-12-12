"""
A bunch of scripty glue for running the training jobs in a loop.

This should mostly be reworked when we move to tf.Estimator's and a proper
dataset queue.
"""
import argh
import argparse
import subprocess
import os
import time
from tqdm import tqdm
import sys
import petname
import shipname
from load_data_sets import DataSetV2
import re
import google.cloud.logging as glog
import logging
from utils import logged_timer as timer
import go


BUCKET = os.environ['BUCKET_NAME'] # Did this die?  Set your bucket!
GAMES_BUCKET = "gs://%s/games/" % BUCKET
MODELS_BUCKET = "gs://%s/models" % BUCKET
MODEL_NUM_REGEX = "\d{6}"
GAME_DIRECTORY = "./data/selfplay/"
MODEL_DIRECTORY = "./saved_models"
TRAINING_DIRECTORY = "data/training_chunks"
TF_LOG_DIR = "logs/%s" % BUCKET

def bootstrap(filename):
    import dual_net
    n = dual_net.DualNetwork()
    n.initialize_variables()
    n.save_variables(filename)

def push_model(model_num, name):
    for f in os.listdir(MODEL_DIRECTORY):
        if f.startswith("{0:06d}".format(model_num)):
            arg =  'gsutil cp %s/%s %s/%06d-%s.%s' % (
                    MODEL_DIRECTORY,
                    f, MODELS_BUCKET, model_num, name, f.split('.')[1])
            subprocess.call(arg.split())

def dir_model_num(dir_path):
    'Returns the model number of a directory, if present, e.g. 000010-neat-model => 10'
    if not re.match(MODEL_NUM_REGEX, os.path.basename(dir_path)):
        return None
    return int(re.match(MODEL_NUM_REGEX, os.path.basename(dir_path)).group())


def smart_rsync(from_model_num=0, source_dir=GAMES_BUCKET, game_dir=None):
    game_dir = game_dir or GAME_DIRECTORY
    from_model_num = 0 if from_model_num < 0 else from_model_num
    seen_dirs = subprocess.check_output(('gsutil ls -d %s*' % source_dir).split()).split()
    seen_dirs = list(map(lambda d: d.decode('UTF-8').strip('/'), seen_dirs))
    model_dirs = [d for d in seen_dirs
            if dir_model_num(d) is not None and dir_model_num(d) >= from_model_num ]
    print("model_dirs:", model_dirs)
    for d in model_dirs:
        basename = os.path.basename(d)
        if not os.path.isdir(os.path.join(game_dir, basename)):
            os.mkdir(os.path.join(game_dir, basename))
        subprocess.call('gsutil -m rsync -r -c {0}{2} {1}{2}/'.format(
                    source_dir, game_dir, basename).split(),
                    stderr=open('.rsync_log', 'ab'))

def find_largest_modelnum(directory):
    return max([dir_model_num(f) or 0 for f in os.listdir(directory) ])

def gather_loop():
    # Check how many chunks there are.  Run rsync.  If there are no
    # chunks, run gather, else wait a bit, and repeat.
    while True:
        while True:
           num_chunks = sum([1 if p.endswith('.gz') else 0 for p in os.listdir(TRAINING_DIRECTORY)])
           if num_chunks == 0:
               break
           print('Found ', num_chunks, ' chunks.  Waiting for training job to use them.')
           time.sleep(30)

        # Create training chunks, do something with the data...
        with timer("=== Gather & write out chunks: "):
            failball = subprocess.call( ('python main.py gather').split())
            if failball:
                print(failball)
                sys.exit(1)

def rsync_loop():
    while True:
        maxnum = find_largest_modelnum(MODEL_DIRECTORY)
        print ("Oldest model: ", maxnum)
        with timer("=== Rsync new games"):
            smart_rsync(maxnum-6)
        logging.info("Rsync finished")
        time.sleep(300)

def train_loop():
    model_num = find_largest_modelnum(MODEL_DIRECTORY) or 0
    train_cmd = 'python main.py train --load-file {2}/{0:06d} -s {2}/{1:06d}' + \
                ' --logdir {3} {4}'

    while True:
        print(" ====== Model %d ======" % (model_num))

        print("Waiting for training chunks...")
        while True:
            num_chunks = sum([1 if fname.endswith('.gz') else 0 for fname in os.listdir(TRAINING_DIRECTORY)])
            if num_chunks != 0:
                break
            time.sleep(15)

        # Take a training step.
        bigarg = train_cmd.format(model_num, model_num+1, MODEL_DIRECTORY, TF_LOG_DIR, TRAINING_DIRECTORY)
        print("RUNNING '%s'" % bigarg)
        failball = subprocess.call(bigarg.split())
        if failball:
            print(failball)
            sys.exit(1)

        # Back up the chunks
        subprocess.call("gsutil cp {dirname}/*.gz gs://{bucket}/old_chunks/{num}/".format(
                {dirname: TRAINING_DIRECTORY,
                 bucket: BUCKET,
                 num: model_num}).split())
        # Wipe the training directory.
        for p in os.listdir(TRAINING_DIRECTORY):
            if p.endswith('.gz'):
                os.remove(os.path.join(TRAINING_DIRECTORY, p))

        # For now, always promote our new model.
        if go.N == 19:
            new_name = shipname.generate()
        else:
            new_name = petname.generate()
        print("A new champion! ", new_name)
        print("Pushing %06d-%s to saved_models" % (model_num, new_name))
        push_model(model_num, new_name)
        subprocess.call('./update-acls')
        model_num+=1

def consolidate(
        input_directory: 'where to look for games'='data/selfplay/',
        max_positions: 'how many positions before discarding games'= 250000,
        before_model: 'consolidate games only for models before this.'=1):

    import ds_wrangler, numpy as np
    paths = [(root, dirs, files) for root, dirs, files in os.walk(input_directory) 
             if dir_model_num(root) is not None and dir_model_num(root) < before_model]

    for model, _, _ in paths:
        metas = []
        for player, _, files in os.walk(model):
            for f in files:
                if f.endswith('.meta'):
                    metas.append(os.path.join(player, f))
        bigchunks = []
        current = []
        counter = 0
        for m in tqdm(metas):
            sz,err = ds_wrangler._read_meta(m)
            if err:
                print('Invalid:', m)
                continue
            counter += sz
            current.append(m)
            if counter > max_positions:
                print("%d positions in %d files" % (counter, len(current)))
                bigchunks.append(current)
                current = []
                counter = 0

        bigchunks.append(current)

        for idx, c in enumerate(tqdm(bigchunks)):
            datasets = [DataSetV2.read(filename.replace('.meta','.gz')) for filename in c]
            combined = DataSetV2(
                np.concatenate([d.pos_features for d in datasets]),
                np.concatenate([d.next_moves for d in datasets]),
                np.concatenate([d.results for d in datasets]))
            combined.write(os.path.join(model, "%s-combined-%d.gz" % (os.path.basename(model), idx)))


parser = argparse.ArgumentParser()
argh.add_commands(parser, [train_loop, gather_loop, rsync_loop, bootstrap, smart_rsync, consolidate])

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        pass
        #client = glog.Client('tensor-go')
        #client.setup_logging(logging.INFO)
    except:
        print('!! Cloud logging disabled')

    argh.dispatch(parser)
