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

import datetime as dt
import fire
import functools
import itertools
import multiprocessing as mp
import os
import sys
import random
import subprocess
import time
from collections import deque

sys.path.insert(0, '.')

from absl import flags
import tensorflow as tf
from tqdm import tqdm
import numpy as np

import preprocessing
from utils import timer, ensure_dir_exists
from rl_loop import fsdb


READ_OPTS = preprocessing.TF_RECORD_CONFIG

LOCAL_DIR = "data/"

# How many positions to look at per generation.
# Per AGZ, 2048 minibatch * 1k = 2M positions/generation
EXAMPLES_PER_GENERATION = 2 ** 21

MINIMUM_NEW_GAMES = 12000
AVG_GAMES_PER_MODEL = 20000


def pick_examples_from_tfrecord(filename, sampling_frac=0.02):
    protos = list(tf.python_io.tf_record_iterator(filename, READ_OPTS))
    number_samples = np.random.poisson(len(protos) * sampling_frac)
    choices = random.sample(protos, min(len(protos), number_samples))
    return choices


def choose(game, sampling_frac=0.02):
    examples = pick_examples_from_tfrecord(game, sampling_frac)
    timestamp = file_timestamp(game)
    return [(timestamp, ex) for ex in examples]


def file_timestamp(filename):
    return int(os.path.basename(filename).split('-')[0])


def _ts_to_str(timestamp):
    return dt.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


class ExampleBuffer():
    def __init__(self, max_size=2**21, sampling_frac=0.02):
        self.examples = deque(maxlen=max_size)
        self.max_size = max_size
        self.sampling_frac = sampling_frac
        self.func = functools.partial(choose, sampling_frac=sampling_frac)
        self.total_updates = 0

    def parallel_fill(self, games, threads=8):
        """ games is a list of .tfrecord.zz game records. """
        games.sort(key=os.path.basename)
        # A couple extra in case parsing fails
        max_games = int(self.max_size / self.sampling_frac / 200) + 480
        if len(games) > max_games:
            games = games[-max_games:]

        with mp.Pool(threads) as pool:
            res = tqdm(pool.imap(self.func, games), total=len(games))
            self.examples.extend(itertools.chain.from_iterable(res))
        print("Got", len(self.examples), "examples")

    def update(self, new_games):
        """ new_games is a list of .tfrecord.zz new game records. """
        new_games.sort(key=os.path.basename)
        first_new_game = None
        for idx, game in enumerate(new_games):
            timestamp = file_timestamp(game)
            if timestamp <= self.examples[-1][0]:
                continue
            elif first_new_game is None:
                first_new_game = idx
                num_new_games = len(new_games) - idx
                print("Found {}/{} new games".format(
                    num_new_games, len(new_games)))
                self.total_updates += num_new_games
            self.examples.extend(self.func(game))
        if first_new_game is None:
            print("No new games", file_timestamp(
                new_games[-1]), self.examples[-1][0])

    def flush(self, path):
        # random.shuffle on deque is O(n^2) convert to list for O(n)
        self.examples = list(self.examples)
        random.shuffle(self.examples)
        with timer("Writing examples to " + path):
            preprocessing.write_tf_examples(
                path, [ex[1] for ex in self.examples], serialize=False)
        self.examples.clear()
        self.examples = deque(maxlen=self.max_size)

    @property
    def count(self):
        return len(self.examples)

    def __str__(self):
        if self.count == 0:
            return "ExampleBuffer: 0 positions"
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


def time_rsync(from_date,
               source_dir=None,
               dest_dir=LOCAL_DIR):
    source_dir = source_dir or fsdb.selfplay_dir()
    while from_date < dt.datetime.utcnow():
        src = os.path.join(source_dir, from_date.strftime("%Y-%m-%d-%H"))
        if tf.gfile.Exists(src):
            _rsync_dir(src, os.path.join(
                dest_dir, from_date.strftime("%Y-%m-%d-%H")))
        from_date = from_date + dt.timedelta(hours=1)


def _rsync_dir(source_dir, dest_dir):
    ensure_dir_exists(dest_dir)
    with open('.rsync_log', 'ab') as rsync_log:
        subprocess.call(['gsutil', '-m', 'rsync', source_dir, dest_dir],
                        stderr=rsync_log)


def _determine_chunk_to_make(write_dir):
    """
    Returns the full path of the chunk to make (gs://...)
    and a boolean, indicating whether we should wait for a new model
    or if we're 'behind' and should just write out our current chunk immediately
    True == write immediately.
    """
    models = fsdb.get_models()
    # Last model is N.  N+1 (should be) training.  We should gather games for N+2.
    chunk_to_make = os.path.join(write_dir, str(
        models[-1][0] + 1) + '.tfrecord.zz')
    if not tf.gfile.Exists(chunk_to_make):
        # N+1 is missing.  Write it out ASAP
        print("Making chunk ASAP:", chunk_to_make)
        return chunk_to_make, True
    chunk_to_make = os.path.join(write_dir, str(
        models[-1][0] + 2) + '.tfrecord.zz')
    while tf.gfile.Exists(chunk_to_make):
        print("Chunk for next model ({}) already exists. Sleeping.".format(
            chunk_to_make))
        time.sleep(5 * 60)
        models = fsdb.get_models()
        chunk_to_make = os.path.join(write_dir, str(
            models[-1][0] + 2) + '.tfrecord.zz')
    print("Making chunk:", chunk_to_make)

    return chunk_to_make, False


def get_window_size(chunk_num):
    """ Adjust the window size by how far we are through a run.
    At the start of the run, there's a benefit to 'expiring' the completely
    random games a little sooner, and scaling up to the 500k game window
    specified in the paper.
    """
    return min(500000, (chunk_num + 5) * (AVG_GAMES_PER_MODEL // 2))


def fill_and_wait_time(bufsize=EXAMPLES_PER_GENERATION,
                       write_dir=None,
                       threads=32,
                       start_from=None):
    start_from = start_from or dt.datetime.utcnow()
    write_dir = write_dir or fsdb.golden_chunk_dir()
    buf = ExampleBuffer(bufsize)
    chunk_to_make, fast_write = _determine_chunk_to_make(write_dir)

    hours = fsdb.get_hour_dirs()
    with timer("Rsync"):
        time_rsync(min(dt.datetime.strptime(
            hours[-1], "%Y-%m-%d-%H/"), start_from))
        start_from = dt.datetime.utcnow()

    hours = fsdb.get_hour_dirs()
    files = (tf.gfile.Glob(os.path.join(LOCAL_DIR, d, "*.zz"))
             for d in reversed(hours) if tf.gfile.Exists(os.path.join(LOCAL_DIR, d)))
    files = itertools.islice(files, get_window_size(chunk_to_make))

    models = fsdb.get_models()
    buf.parallel_fill(
        list(itertools.chain.from_iterable(files)), threads=threads)
    print("Filled buffer, watching for new games")

    while (fsdb.get_latest_model() == models[-1] or buf.total_updates < MINIMUM_NEW_GAMES):
        with timer("Rsync"):
            time_rsync(start_from - dt.timedelta(minutes=60))
        start_from = dt.datetime.utcnow()
        hours = sorted(fsdb.get_hour_dirs(LOCAL_DIR))
        new_files = list(map(lambda d: tf.gfile.Glob(
            os.path.join(LOCAL_DIR, d, '*.zz')), hours[-2:]))
        buf.update(list(itertools.chain.from_iterable(new_files)))
        if fast_write:
            break
        time.sleep(30)
        if fsdb.get_latest_model() != models[-1]:
            print("New model!  Waiting for games. Got",
                  buf.total_updates, "new games so far")

    latest = fsdb.get_latest_model()
    print("New model!", latest[1], "!=", models[-1][1])
    print(buf)
    buf.flush(chunk_to_make)


def fill_and_wait_models(bufsize=EXAMPLES_PER_GENERATION,
                         write_dir=None,
                         threads=8,
                         model_window=100,
                         skip_first_rsync=False):
    """ Fills a ringbuffer with positions from the most recent games, then
    continually rsync's and updates the buffer until a new model is promoted.
    Once it detects a new model, iit then dumps its contents for training to
    immediately begin on the next model.
    """
    write_dir = write_dir or fsdb.golden_chunk_dir()
    buf = ExampleBuffer(bufsize)
    models = fsdb.get_models()[-model_window:]
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
                   positions=EXAMPLES_PER_GENERATION,
                   threads=8,
                   sampling_frac=0.02):
    """
    Explicitly make a golden chunk for a given model `model_num`
    (not necessarily the most recent one).

      While we haven't yet got enough samples (EXAMPLES_PER_GENERATION)
      Add samples from the games of previous model.
    """
    game_dir = game_dir or fsdb.selfplay_dir()
    ensure_dir_exists(output_dir)
    models = [model for model in fsdb.get_models() if model[0] < model_num]
    buf = ExampleBuffer(positions, sampling_frac=sampling_frac)
    files = []
    for _, model in sorted(models, reverse=True):
        local_model_dir = os.path.join(local_dir, model)
        if not tf.gfile.Exists(local_model_dir):
            print("Rsyncing", model)
            _rsync_dir(os.path.join(game_dir, model), local_model_dir)
        files.extend(tf.gfile.Glob(os.path.join(local_model_dir, '*.zz')))
        print("{}: {} games".format(model, len(files)))
        if len(files) * 200 * sampling_frac > positions:
            break

    print("Filling from {} files".format(len(files)))

    buf.parallel_fill(files, threads=threads)
    print(buf)
    output = os.path.join(output_dir, str(model_num) + '.tfrecord.zz')
    print("Writing to", output)
    buf.flush(output)


if __name__ == "__main__":
    import sys
    remaining_argv = flags.FLAGS(sys.argv, known_only=True)
    fire.Fire({
        'fill_and_wait_models': fill_and_wait_models,
        'fill_and_wait_time': fill_and_wait_time,
        'smart_rsync': smart_rsync,
        'make_chunk_for': make_chunk_for,
    }, remaining_argv[1:])
