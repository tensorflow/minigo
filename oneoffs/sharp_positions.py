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
Used as a starting point for our generating hard position collection.

training_curve.py is a supervised approach, assuming the pro moves are correct
Performance is then measured against the played move and eventual outcome.

sharp_positions is unsupervised, it tries to determine what the correct outcome
and move is based on clustering of the strongest rated algorithms.

Step 1. Create collection.csv of sgf and move number
    export BOARD_SIZE=19
    SGF_DIR=data/sgf
    MODEL_DIR=models/
    python3 sharp_positions.py subsample --num_positions 10 --sgf_dir $SGF_DIR

Step 2. Create a directory of those sgfs truncated at the specified move number
    # from https://github.com/sethtroisi/go-scripts/
    ./truncate.sh

    # Remove any bad SGFS.
    cat badfiles.txt | tqdm | xargs rm

    # Remove original SGFs
    cat ../minigo/collection.csv | cut -f1 -d, | sort -u | tqdm | xargs rm

    # Rerun truncate successfully this time
    ./truncate.sh

Step 3. Get value & policy for all models for all positions
    python3 sharp_positions.py evaluate --sgf_dir <problem-collections2> --model_dir $MODEL_DIR

Step 4. Fit a model and minimize a set of positions to predict strength
    python3 sharp_positions.py minimize \
        --model_dir $MODEL_DIR --sgf_dir data/s \
        --rating_json ratings.json --results results.csv
"""

import sys
sys.path.insert(0, '.')

import itertools
import json
import multiprocessing
import os
import random
from collections import defaultdict, Counter

import fire
import numpy as np
import tensorflow as tf
from absl import flags
from sklearn import svm
from tqdm import tqdm

import oneoff_utils

# Filter tensorflow info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

flags.DEFINE_integer("num_positions", 1, "How many positions from each game.")
flags.DEFINE_integer("top_n", 3, "Policy moves to record per position.")
flags.DEFINE_integer("min_idx", 100, "Min model number to include.")
flags.DEFINE_integer("batch_size", 64, "Eval batch size.")

# Inputs
flags.DEFINE_string("sgf_dir", "data/s/", "input collection of SGFs")
flags.DEFINE_string("model_dir", "models/", "Folder of Minigo models")
flags.DEFINE_string("rating_json", "ratings.json", "Ratings of models")

# Outputs
flags.DEFINE_string("collection", "collection.csv", "subsampled csv file")
flags.DEFINE_string("results", "results.csv", "Evaluate results file")
flags.DEFINE_string("SVM_json", "SVM_data.json", "SVM data about positions")

FLAGS = flags.FLAGS


def grouper(n, iterable):
    """Itertools recipe
    >>> list(grouper(3, iter('ABCDEFG')))
    [['A', 'B', 'C'], ['D', 'E', 'F'], ['G']]
    """
    return (iterable[i:i + n] for i in range(0, len(iterable), n))


def get_final_positions():
    sgf_files = sorted(oneoff_utils.find_and_filter_sgf_files(FLAGS.sgf_dir))

    with multiprocessing.Pool() as pool:
        pos = list(pool.map(oneoff_utils.final_position_sgf, tqdm(sgf_files)))

    assert len(pos) > 0, "BOARD_SIZE != 19?"
    return sgf_files, pos


def get_model_idx(model_name):
    number = os.path.basename(model_name).split('-')[0]
    assert len(number) == 6 and 0 <= int(number) <= 1000, model_name
    return int(number)


def subsample():
    """Sample num_positions postions from each game in sgf_dir

    Usage:
        python3 sharp_positions.py subsample --num_positions 10 --sgf_dir data/s

    NOTE(sethtroisi): see link for a script to truncate SGFs at move number
        https://github.com/sethtroisi/go-scripts
    """

    sgf_files = oneoff_utils.find_and_filter_sgf_files(
        FLAGS.sgf_dir, None, None)

    with open(FLAG.collection, 'w') as collection:
        fails = 0
        for path in tqdm(sorted(sgf_files)):
            try:
                positions, moves, results = oneoff_utils.parse_sgf(path)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                fails += 1
                print("Fail {}, while parsing {}: {}".format(fails, path, e))
                continue

            moves = len(positions)
            indexes = random.sample(range(10, moves), FLAGS.num_positions)
            for index in sorted(indexes):
                collection.write('{}, {}\n'.format(path, index))


def evaluate():
    """Get Policy and Value for each network, for each position

    Usage:
        python3 sharp_positions.py evaluate --sgf_dir data/s --model_dir models/
    """

    def short_str(v):
        if isinstance(v, float):
            return "{.3f}".format(v)
        return str(v)

    # Load positons
    sgf_names, all_positions = get_final_positions()

    # Run and save some data about each position
    # Save to csv because that's easy
    model_paths = oneoff_utils.get_model_paths(FLAGS.model_dir)
    num_models = len(model_paths)
    print("Evaluating {} models: {} to {}".format(
        num_models, model_paths[0], model_paths[-1]))
    print()

    with open(FLAGS.results, "w") as results:
        results.write(",".join(sgf_names) + "\n")

        player = None
        for idx in tqdm(range(FLAGS.min_idx, num_models, 1), desc="model"):
            model = model_paths[idx]

            if player and idx % 50 == 0:
                player.network.sess.close()
                tf.reset_default_graph()
                player = None

            if player:
                oneoff_utils.restore_params(model, player)
            else:
                player = oneoff_utils.load_player(model)

            row = [model]
            for positions in grouper(FLAGS.batch_size, all_positions):
                probs, values = player.network.run_many(positions)
                # NOTE(sethtroisi): For now we store the top n moves to shrink
                # the size of the recorded data.

                top_n = FLAGS.top_n
                top_policy_move = np.fliplr(np.argsort(probs))[:,:top_n]
                top_policy_value = np.fliplr(np.sort(probs))[:,:top_n]

                # One position at a time
                for v, m, p in zip(values, top_policy_move, top_policy_value):
                    row.append(v)
                    row.extend(itertools.chain.from_iterable(zip(m, p)))

                if len(positions) > 10:
                    average_seen = top_policy_value.sum() / len(positions)
                    if average_seen < 0.3:
                        print("\t", average_seen, top_policy_value.sum(axis=-1))

            results.write(",".join(map(short_str, row)) + "\n")


def minimize():
    """Find a subset of problems that maximal explains rating.

    Usage:
        python3 sharp_positions.py minimize \
            --model_dir models --sgf_dir data/s
            --rating_json ratings.json --results results.csv
    """
    ########################### HYPER PARAMETERS ###############################

    # Stop when r2 is this much worse than full set of positions
    r2_stopping_percent = 0.96
    # for this many iterations
    stopping_iterations = 5

    # Limit SVM to a smaller number of positions to speed up code.
    max_positions_fit = 300
    # Filter any position that "contributes" less than this percent of max.
    filter_contribution_percent = 0.3
    # Never filter more than this many positions in one iterations
    filter_limit = 25

    ########################### HYPER PARAMETERS ###############################

    # Load positons
    model_paths = oneoff_utils.get_model_paths(FLAGS.model_dir)
    num_models = len(model_paths)
    assert num_models > 0, FLAGS.model_dir

    # Load model ratings
    # wget https://cloudygo.com/v12-19x19/json/ratings.json
    ratings = json.load(open(FLAGS.rating_json))
    raw_ratings = {int(r[0]): float(r[1]) for r in ratings}

    model_ratings = []
    for model in model_paths:
        model_idx = get_model_idx(model)
        if model_idx < FLAGS.min_idx:
            continue

        model_ratings.append(raw_ratings[model_idx])
    model_ratings = np.array(model_ratings)

    assert 0 < len(model_ratings) <= num_models, len(model_ratings)
    num_models = len(model_ratings)

    sgf_names, all_positions = get_final_positions()
    # Trim off common path prefix.
    common_path = os.path.commonpath(sgf_names)
    sgf_names = [name[len(common_path) + 1:] for name in sgf_names]

    print("Considering {} positions, {} models".format(
        len(all_positions), num_models))
    print()

    # Load model data
    top_n = FLAGS.top_n
    positions = defaultdict(list)
    with open(FLAGS.results) as results:
        headers = results.readline().strip()
        assert headers.count(",") + 1 == len(sgf_names)

        # Row is <model_name> + positions x [value, top_n x [move, move_policy]]
        for row in tqdm(results.readlines(), desc="result line"):
            data = row.split(",")
            model_idx = get_model_idx(data.pop(0))
            if model_idx < FLAGS.min_idx:
                continue

            data_per = 1 + top_n * 2
            assert len(data) % data_per == 0, len(data)

            for position, position_data in enumerate(grouper(data_per, data)):
                value = float(position_data.pop(0))
                moves = list(map(int, position_data[0::2]))
                move_policy = list(map(float, position_data[1::2]))

                positions[position].append([value, moves, move_policy])

    def one_hot(n, i):
        one_hot = [0] * n
        if 0 <= i < n:
            one_hot[i] += 1
        return one_hot

    # NOTE: top_n isn't the same semantic value here and can be increased.
    one_hot_moves = top_n
    num_features = 1 + 5 + (one_hot_moves + 1)

    # Features by position
    features = []
    pos_top_moves = []
    for position, data in tqdm(positions.items(), desc="featurize"):
        assert len(data) == num_models, len(data)

        top_moves = Counter([d[1][0] for d in data])
        top_n_moves = [m for m, c in top_moves.most_common(one_hot_moves)]
        if len(top_n_moves) < one_hot_moves:
            top_n_moves.extend([-1] * (one_hot_moves - len(top_n_moves)))
        assert len(top_n_moves) == one_hot_moves, "pad with dummy moves"
        pos_top_moves.append(top_n_moves)

        # Eventaully we want
        # [model 1 position 1 features, m1 p2 features, m1 p3 features, ... ]
        # [model 2 position 1 features, m2 p2 features, m2 p3 features, ... ]
        # [model 3 position 1 features, m3 p2 features, m3 p3 features, ... ]
        # ...
        # [model m position 1 features, mm p2 features, mm p3 features, ... ]

        # We'll do position selection by joining [model x position_feature]

        feature_columns = []
        for model, (v, m, mv) in enumerate(data):
            # Featurization (for each positions):
            #   * Value (-1 to 1), Bucketed value
            #   * Cluster all model by top_n moves (X,Y,Z or other)?
            #     * value of that move for model
            #   * policy value of top move
            model_features = []

            model_features.append(2 * v - 1)
            # NOTE(sethtroisi): Consider bucketize value by value percentiles.
            value_bucket = np.searchsorted((0.2, 0.4, 0.6, 0.8), v)
            model_features.extend(one_hot(5, value_bucket))

            # Policy weight for most common X moves (among all models).
            policy_weights = [0] * (one_hot_moves + 1)
            for move, policy_value in zip(m, mv):
                if move in top_n_moves:
                    policy_weights[top_n_moves.index(move)] = policy_value
                else:
                    policy_weights[-1] += policy_value
            model_features.extend(policy_weights)

            assert len(model_features) == num_features

            feature_columns.append(model_features)
        features.append(feature_columns)

    features = np.array(features)
    print("Feature shape", features.shape)
    print()

    # Split the models to test / train
    train_size = int(num_models * 0.9)
    train_models = sorted(np.random.permutation(num_models)[:train_size])
    test_models = sorted(set(range(num_models)) - set(train_models))
    assert set(train_models + test_models) == set(range(num_models))
    features_train = features[:, train_models, :]
    features_test  = features[:, test_models, :]

    labels_train = model_ratings[train_models]
    labels_test = model_ratings[test_models]

    # Choose some set of positions and see how well they explain ratings
    positions_to_use = set(positions.keys())
    linearSVM = svm.LinearSVR()
    best_test_r2 = 0
    below_threshold = 0

    for iteration in itertools.count(1):
        iter_positions = np.random.permutation(list(positions_to_use))
        iter_positions = sorted(iter_positions[:max_positions_fit])

        # Take this set of positions and build X
        X = np.concatenate(features_train[iter_positions], axis=1)
        Xtest = np.concatenate(features_test[iter_positions], axis=1)
        assert X.shape == (train_size, num_features * len(iter_positions))

        linearSVM.fit(X, labels_train)

        score_train = linearSVM.score(X, labels_train)
        score_test = linearSVM.score(Xtest, labels_test)
        print("iter {}, {}/{} included, R^2: {:.4f} train, {:.3f} test".format(
            iteration, len(iter_positions), len(positions_to_use),
            score_train, score_test))

        # Determine the most and least useful position:
        # TODO(amj,brilee): Validate this math.
        assert len(linearSVM.coef_) == num_features * len(iter_positions)

        # The intercepts tell us how much this contributes to overall rating
        # but coef tell us how much different answers differentiate rating.
        coef_groups = list(grouper(num_features, linearSVM.coef_))
        position_coefs = [abs(sum(c)) for c in coef_groups]

        pos_value_idx = np.argsort(position_coefs)
        max_pos = pos_value_idx[-1]
        most_value = position_coefs[max_pos]

        print("\tMost value {} => {:.1f} {}".format(
            max_pos, most_value, sgf_names[iter_positions[max_pos]]))

        # Drop any positions that aren't very useful
        for dropped, pos_idx in enumerate(pos_value_idx[:filter_limit], 1):
            contribution = position_coefs[pos_idx]
            positions_to_use.remove(iter_positions[pos_idx])
            print("\t\tdropping({}): {:.1f} {}".format(
                dropped, contribution, sgf_names[iter_positions[pos_idx]]))

            if contribution > filter_contribution_percent * most_value:
                break
        print()

        best_test_r2 = max(best_test_r2, score_test)
        if score_test > r2_stopping_percent * best_test_r2:
            below_threshold = 0
        else:
            below_threshold += 1
            if below_threshold == stopping_iterations:
                print("{}% decrease in R^2, stopping".format(
                    100 - int(100 * r2_stopping_percent)))
                break

    # Write down the differentiating positions and their answers.
    svm_data = []
    for position_idx in list(reversed(pos_value_idx)):
        coefs = coef_groups[position_idx]

        # Global position index.
        position = iter_positions[position_idx]
        sgf_name = sgf_names[position]
        top_moves = pos_top_moves[position]

        svm_data.append([sgf_name, [top_moves, coefs.tolist()]])

    with open(FLAGS.SVM_json, "w") as svm_json:
        json.dump(svm_data, svm_json)
    print("Dumped data about {} positions to {}".format(
        len(svm_data), FLAGS.SVM_json))


if __name__ == "__main__":
    remaining_argv = flags.FLAGS(sys.argv, known_only=True)
    fire.Fire({
        'subsample': subsample,
        'evaluate': evaluate,
        'minimize': minimize,
    }, remaining_argv[1:])
