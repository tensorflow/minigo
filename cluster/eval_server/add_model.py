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
Add a model to the eval models cbt.

Usage:
    source cluster/common.sh

    python cluster/eval_server/add_model.py \\
        --cbt_project "$PROJECT" \\
        --cbt_instance "$CBT_INSTANCE" \\
        --cbt_table "$CBT_MODEL_EVAL_TABLE" \\
        <model_name> \\
        "gs://<model_path>/model.ckpt-30720.pb" \\
        "flags: --num_readouts=100 -virtual_losses=2" \\
        bazel-bin/cc/gtp

    cbt -project "$PROJECT" -instance "$CBT_INSTANCE" read "$CBT_MODEL_EVAL_TABLE"
"""

import sys
sys.path.insert(0, '.')

import datetime
import hashlib
import os
import pprint
import shutil

from absl import app, flags
from google.cloud import bigtable
from tensorflow import gfile
import fire

from bigtable_input import METADATA


flags.mark_flags_as_required([
    "cbt_project", "cbt_instance", "cbt_table"
])

FLAGS = flags.FLAGS


# TODO(sethtroisi): Unify with oneoffs
MODEL_ROW = "m_eval_{:0>10}"

PLAYER_FOLDER = "gs://minigo-pub/eval_server/models/"

GTP_BIN = "bazel-bin/cc/gtp"
LIB_TF_FW_SO = "cc/tensorflow/lib/libtensorflow_framework.so"
LIB_TF_CC_SO = "cc/tensorflow/lib/libtensorflow_cc.so"


def verify_params(model_path, model_flags, binary_path):
    # Verify model_path = .../<model>.pb
    assert model_path.endswith('.pb'), model_path

    assert model_flags, "Flags must be non-empty"
    assert model_flags.startswith("flags: "), "model_flags must start with \"flags: \""
    assert "-num_readouts=" in model_flags, "must set playouts"
    assert "-model" not in model_flags, "leave model unset"

    # Verify binary_path is to a gtp binary and it exists
    assert binary_path.endswith(GTP_BIN), binary_path
    assert os.path.isfile(binary_path)
    binary_base = binary_path.replace(GTP_BIN, "")
    assert os.path.isdir(binary_base)
    # $ ldd bazel-bin/cc/gtp
    #    libtensorflow_framework.so => (bazel path) => cc/tensorflow/lib/libtensorflow_framework.so
    #    libtensorflow_cc.so => (bazel path) => cc/tensorflow/lib/libtensorflow_cc.so
    #    libcuda.so.1
    #    libnvidia-fatbinaryloader.so.<version e.g. 390>.116
    assert os.path.isfile(os.path.join(binary_base, LIB_TF_FW_SO))
    assert os.path.isfile(os.path.join(binary_base, LIB_TF_CC_SO))

    return True

def copy_to_gcs(src, dst):
    assert gfile.Exists(src)
    assert not gfile.Exists(dst)

    with gfile.GFile(src, "rb") as src_f, gfile.GFile(dst, "wb") as dst_f:
        shutil.copyfileobj(src_f, dst_f)


def add_model(argv):
    _, name, model_path, model_flags, binary_path = argv
    """Add a player (model + flags) to models cbt
    name: name of the model (e.g. v17-990-p1 or 990-cormorant-p800)
    model_path: path to model
    model_flags: flags to be used with this model must start with flags:
    binary_path: path to gtp binary that can execute model
    """
    assert verify_params(model_path, model_flags, binary_path)

    now_date = datetime.datetime.now().isoformat(' ')

    # Unset parameters are to match model CBT table.
    metadata = {
        "model":     name,
        "model_num": "",
        "model_flags": model_flags,
        "run": "",
        "parent": "",
        "tag": "eval_server",
        "tool": "add_model.py",
        "trained_date": now_date,
    }

    hash_params = model_path.encode()
    hash_params += model_flags.replace(' ', '').replace('=', '').encode()
    # [:10] to match MODEL_ROW format
    player_hash = hashlib.md5(hash_params).hexdigest()[:10]
    player_name = MODEL_ROW.format(player_hash)

    # Check that table exists
    print(f"CBT: {FLAGS.cbt_project}:{FLAGS.cbt_instance}:{FLAGS.cbt_table}")
    assert FLAGS.cbt_project and FLAGS.cbt_instance and FLAGS.cbt_table, \
        "Must define --cbt_project, --cbt_instance, and --cbt_table"
    bt_table = (bigtable
                .Client(FLAGS.cbt_project, admin=True)
                .instance(FLAGS.cbt_instance)
                .table(FLAGS.cbt_table))
    assert bt_table.exists(), f"Table({FLAGS.cbt_table}) doesn't exist"
    # Check that model wasn't already uploaded.
    assert bt_table.read_row(player_name) is None, f"{player_name} already uploaded"

    # Upload model and binary
    gcs_player_folder = os.path.join(PLAYER_FOLDER, player_hash)
    new_model_path = os.path.join(gcs_player_folder, os.path.basename(model_path))
    new_binary_base = os.path.join(gcs_player_folder, "bin")
    binary_base = binary_path.replace(GTP_BIN, "")

    print(f"Saving to {gcs_player_folder!r}")

    # NOTE: Currently this only supports minigo models but will be extended when
    # Docker supports both openCL and CUDA at the same time.
    copy_to_gcs(model_path, new_model_path)
    for bin_file in [GTP_BIN, LIB_TF_FW_SO, LIB_TF_CC_SO]:
        copy_to_gcs(os.path.join(binary_base, bin_file),
                    os.path.join(new_binary_base, bin_file))

    metadata["model_path"] = new_model_path

    # Add model metadata to bigtable
    row = bt_table.row(player_name)
    for column, value in metadata.items():
        row.set_cell(METADATA, column, value)

    pprint.pprint(dict(metadata))

    response = bt_table.mutate_rows([row])

    # Validate that all rows were written successfully
    for status in response:
        print ("Status:", status.code, status)
        if status.code is not 0:
            print("Failed to write {}".format(i, status))


if __name__ == '__main__':
    app.run(add_model)
