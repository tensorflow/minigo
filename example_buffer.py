import argh
import argparse
import datetime as dt
import functools
import itertools
import multiprocessing as mp
import os
import random
import subprocess
import time
import tensorflow as tf
from tqdm import tqdm
from collections import deque

import preprocessing
import dual_net
from utils import timer, ensure_dir_exists
import fsdb


READ_OPTS = preprocessing.TF_RECORD_CONFIG

LOCAL_DIR = "data/"


def pick_examples_from_tfrecord(filename, samples_per_game=4):
    protos = list(tf.python_io.tf_record_iterator(filename, READ_OPTS))
    if len(protos) < 20:  # Filter games with less than 20 moves
        return []
    choices = random.sample(protos, min(len(protos), samples_per_game))

    def make_example(protostring):
        e = tf.train.Example()
        e.ParseFromString(protostring)
        return e
    return list(map(make_example, choices))


def choose(game, samples_per_game=4):
    examples = pick_examples_from_tfrecord(game, samples_per_game)
    timestamp = file_timestamp(game)
    return [(timestamp, ex) for ex in examples]


def file_timestamp(filename):
    return int(os.path.basename(filename).split('-')[0])


def _ts_to_str(timestamp):
    return dt.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


class ExampleBuffer():
    def __init__(self, max_size=2000000):
        self.examples = deque(maxlen=max_size)
        self.max_size = max_size

    def parallel_fill(self, games, threads=8, samples_per_game=4):
        games.sort(key=os.path.basename)
        if len(games) * samples_per_game > self.max_size:
            games = games[-1 * self.max_size // samples_per_game:]

        func = functools.partial(choose, samples_per_game=samples_per_game)

        with mp.Pool(threads) as pool:
            res = tqdm(pool.imap(func, games), total=len(games))
            self.examples.extend(itertools.chain(*res))

    def update(self, new_games, samples_per_game=4):
        """
        new_games is list of .tfrecord.zz files of new games
        """
        new_games.sort(key=os.path.basename)
        first_new_game = None
        for idx, game in enumerate(tqdm(new_games)):
            timestamp = file_timestamp(game)
            if timestamp <= self.examples[-1][0]:
                continue
            elif first_new_game is None:
                first_new_game = idx
                print(
                    "Found {}/{} new games".format(len(new_games) - idx, len(new_games)))

            choices = [(timestamp, ex) for ex in pick_examples_from_tfrecord(
                game, samples_per_game)]
            self.examples.extend(choices)

    def flush(self, path):
        with timer("Writing examples to " + path):
            random.shuffle(self.examples)
            preprocessing.write_tf_examples(
                path, [ex[1] for ex in self.examples])

    @property
    def count(self):
        return len(self.examples)

    def __str__(self):
        return "ExampleBuffer: {} positions sampled from {} to {}".format(
            self.count,
            _ts_to_str(self.examples[0][0]),
            _ts_to_str(self.examples[-1][0]))


def files_for_model(model):
    return tf.gfile.Glob(os.path.join(LOCAL_DIR, model[1], '*.zz'))


def smart_rsync(
        from_model_num=0,
        source_dir=None,
        dest_dir=LOCAL_DIR):
    source_dir = source_dir or fsdb.selfplay_dir()
    from_model_num = 0 if from_model_num < 0 else from_model_num
    models = [m for m in fsdb.get_models() if m[0] >= from_model_num]
    for _, model in models:
        _rsync_dir(os.path.join(
            source_dir, model), os.path.join(dest_dir, model))


def _rsync_dir(source_dir, dest_dir):
    ensure_dir_exists(dest_dir)
    with open('.rsync_log', 'ab') as rsync_log:
        subprocess.call(['gsutil', '-m', 'rsync', source_dir, dest_dir],
                        stderr=rsync_log)


def fill_and_wait(bufsize=dual_net.EXAMPLES_PER_GENERATION,
                  write_dir=None,
                  model_window=100,
                  threads=8,
                  skip_first_rsync=False):
    """ Fills a ringbuffer with positions from the most recent games, then
    continually rsync's and updates the buffer until a new model is promoted.
    Once it detects a new model, iit then dumps its contents for training to
    immediately begin on the next model.
    """
    write_dir = write_dir or fsdb.golden_chunk_dir()
    buf = ExampleBuffer(bufsize)
    models = fsdb.get_models()[-model_window:]
    # Last model is N.  N+1 is training.  We should gather games for N+2.
    chunk_to_make = os.path.join(write_dir, str(
        models[-1][0] + 2) + '.tfrecord.zz')
    while tf.gfile.Exists(chunk_to_make):
        print("Chunk for next model ({}) already exists.  Sleeping.".format(chunk_to_make))
        time.sleep(5 * 60)
        models = fsdb.get_models()[-model_window:]
    print("Making chunk:", chunk_to_make)
    if not skip_first_rsync:
        with timer("Rsync"):
            smart_rsync(models[-1][0] - 6)
    files = tqdm(map(files_for_model, models), total=len(models))
    buf.parallel_fill(list(itertools.chain(*files)), threads=threads)

    print("Filled buffer, watching for new games")
    while fsdb.get_latest_model()[0] == models[-1][0]:
        with timer("Rsync"):
            smart_rsync(models[-1][0] - 2)
        new_files = tqdm(map(files_for_model, models[-2:]), total=len(models))
        buf.update(list(itertools.chain(*new_files)))
        time.sleep(60)
    latest = fsdb.get_latest_model()

    print("New model!", latest[1], "!=", models[-1][1])
    print(buf)
    buf.flush(os.path.join(write_dir, str(latest[0] + 1) + '.tfrecord.zz'))


def make_chunk_for(output_dir=LOCAL_DIR,
                   local_dir=LOCAL_DIR,
                   game_dir=None,
                   model_num=1,
                   positions=dual_net.EXAMPLES_PER_GENERATION,
                   threads=8,
                   samples_per_game=4):
    """
    Explicitly make a golden chunk for a given model `model_num`
    (not necessarily the most recent one).

      While we haven't yet got enough samples (EXAMPLES_PER_GENERATION)
      Add samples from the games of previous model.
    """
    game_dir = game_dir or fsdb.selfplay_dir()
    ensure_dir_exists(output_dir)
    models = [(num, name)
              for num, name in fsdb.get_models() if num < model_num]
    buf = ExampleBuffer(positions)
    files = []
    for _, model in sorted(models, reverse=True):
        local_model_dir = os.path.join(local_dir, model)
        if not tf.gfile.Exists(local_model_dir):
            print("Rsyncing", model)
            _rsync_dir(os.path.join(
                game_dir, model), local_model_dir)
        files.extend(tf.gfile.Glob(os.path.join(local_model_dir, '*.zz')))
        if len(files) * samples_per_game > positions:
            break

    print("Filling from {} files".format(len(files)))

    buf.parallel_fill(files, threads=threads,
                      samples_per_game=samples_per_game)
    print(buf)
    output = os.path.join(output_dir, str(model_num) + '.tfrecord.zz')
    print("Writing to", output)
    buf.flush(output)


parser = argparse.ArgumentParser()
argh.add_commands(parser, [fill_and_wait, smart_rsync, make_chunk_for])

if __name__ == "__main__":
    import sys
    remaining_argv = flags.FLAGS(sys.argv, known_only=True)
    argh.dispatch(parser, argv=remaining_argv[1:])
