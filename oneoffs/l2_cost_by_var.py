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

"""Plot the l2 cost of various tensors over a training run.

Example usage:
  python3 oneoffs/l2_cost_by_var.py --base_dir "gs://minigo-pub/v7-19x19/"
"""
import sys
sys.path.insert(0, '.')

import os.path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import re
from absl import app, flags
from collections import defaultdict
from tqdm import tqdm

import dual_net
from rl_loop import fsdb
import oneoff_utils

flags.DEFINE_string("plot_dir", "data/l2_cost", "Where to save the plots.")
flags.DEFINE_integer("idx_start", 200, "Only take models after given idx.")
flags.DEFINE_integer("eval_every", 10, "Eval every k models.")

FLAGS = flags.FLAGS


def reduce_var(var):
    return re.sub(r'_[0-9]+', '_<X>', var)


def reduce_and_print_vars(var_names):
    reduced_vars = sorted(set(map(reduce_var, var_names)))
    print('vars names({} reduced to {}):'.format(
        len(var_names), len(reduced_vars)))
    for v in reduced_vars:
        print('\t', v)
    return reduced_vars


def get_l2_cost_data(model_paths, idx_start, eval_every):

    print('Reading models {}-{}, eval_every={}'.format(
        idx_start, len(model_paths), eval_every))

    def l2_cost(tensor):
        return np.sum(np.square(tensor))

    var_names = tf.train.load_checkpoint(model_paths[1]) \
        .get_variable_to_dtype_map().keys()
    reduced_vars = reduce_and_print_vars(var_names)
    # Not a real var, sorry.
    reduced_vars.remove('global_step')

    df = pd.DataFrame()
    for idx in tqdm(range(idx_start, len(model_paths), eval_every)):
        model_path = model_paths[idx]
        ckpt = tf.train.load_checkpoint(model_path)

        row = defaultdict(float)
        row['model'] = idx
        for v in var_names:
            l2 = l2_cost(ckpt.get_tensor(v))
            row[v] = l2
            row[reduce_var(v)] += l2
        df = df.append(row, ignore_index=True)
    return df


def dual_net_list(model):
    dual = dual_net.DualNetwork(model)

    print("Dual Net will calculate L2 cost over these variables")
    with dual.sess.graph.as_default():
        var_names = [v.name for v in tf.trainable_variables()]
        reduce_and_print_vars(var_names)
    print()


def save_plots(data_dir, df):
    for column in sorted(df.columns.values):
        if column == 'model':
            continue

        if '<X>' in column:
            plt.figure()
            plt.plot(df['model'].astype('int64'), df[column])
            plt.xlabel('Model idx')
            plt.ylabel('l2_cost')
            plt.title('{} l2_cost over v7 run'.format(column))

            file_name = '{}.png'.format(column.replace('/', '-'))
            plot_path = os.path.join(data_dir, file_name)

            plt.savefig(plot_path)
            # Showing plots gets old really fast but can be uncommented if desired.
            # plt.show()
            plt.close()


def main(unusedargv):
    model_paths = oneoff_utils.get_model_paths(fsdb.models_dir())

    # List vars constructed when using dual_net.
    dual_net_list(model_paths[0])

    # Calculate l2 cost over a sequence of our models.
    df = get_l2_cost_data(model_paths, FLAGS.idx_start, FLAGS.eval_every)
    print(df)
    save_plots(FLAGS.plot_dir, df)


if __name__ == "__main__":
    app.run(main)
