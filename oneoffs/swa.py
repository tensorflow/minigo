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

# Example usage:
#   export BOARD_SIZE=19
#   python3 oneoffs/swa.py --bucket_name="minigo-pub/v7-19x19" --data_dir="data/swa"
#   python main.py freeze-graph data/swa/swa-2

import sys
sys.path.insert(0, '.')
import os

import tensorflow as tf
from absl import flags
from tensorflow.python.framework import meta_graph

import dual_net
import fsdb
from utils import ensure_dir_exists

flags.DEFINE_string("data_dir", "data/swa/", "Where to save swa models")
flags.DEFINE_integer("count", 2, "number of top models to aggregate")

FLAGS = flags.FLAGS


def swa():
    path_base = fsdb.models_dir()
    model_names = [
        "000393-lincoln",
        "000390-indus",
        "000404-hannibal",
        "000447-hawke",
        "000426-grief",
        "000431-lion",
        "000428-invincible",
        "000303-olympus",
        "000291-superb",
        "000454-victorious",
    ]
    model_names = model_names[:FLAGS.count]

    model_paths = [os.path.join(path_base, m) for m in model_names]

    # construct the graph
    features, labels = dual_net.get_inference_input()
    dual_net.model_fn(features, labels, tf.estimator.ModeKeys.PREDICT)

    # restore all saved weights
    meta_graph_def = meta_graph.read_meta_graph_file(model_paths[0] + '.meta')
    stored_var_names = set(
        [n.name for n in meta_graph_def.graph_def.node if n.op == 'VariableV2'])

    var_list = [v for v in tf.global_variables()
                if v.op.name in stored_var_names]
    var_list.sort(key=lambda v: v.op.name)

    print(stored_var_names)
    print(len(stored_var_names), len(var_list))

    sessions = [tf.Session() for _ in model_paths]
    saver = tf.train.Saver()
    for sess, model_path in zip(sessions, model_paths):
        saver.restore(sess, model_path)

    # Load all VariableV2s for each model.
    values = [sess.run(var_list) for sess in sessions]

    # Iterate over all variables average values from all models.
    all_assign = []
    for var, vals in zip(var_list, zip(*values)):
        print("{}x {}".format(len(vals), var))
        if var.name == "global_step:0":
            avg = vals[0]
            for val in vals:
                avg = tf.maximum(avg, val)
        else:
            avg = tf.add_n(vals) / len(vals)
            continue

        all_assign.append(tf.assign(var, avg))

    # Run all asign ops on an existing model (which has other ops and graph).
    sess = sessions[0]
    sess.run(all_assign)

    # Export a new saved model.
    ensure_dir_exists(FLAGS.data_dir)
    dest_path = os.path.join(FLAGS.data_dir, "swa-" + str(FLAGS.count))
    saver.save(sess, dest_path)


if __name__ == '__main__':
    remaining_argv = flags.FLAGS(sys.argv, known_only=True)
    swa()
