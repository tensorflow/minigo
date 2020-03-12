# Copyright 2020 Google LLC
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
Transposes the convolutions in an NCHW minigo model to NHWC.

This keeps the input feature layout as NCHW.

Usage:
  python3 oneoffs/transpose_model.py \
      --src_path "$SRC_PATH"
      --dst_path "$DST_PATH"
"""

import sys
sys.path.insert(0, '.')  # nopep8

# Hide the GPUs from TF. This makes startup 2x quicker on some machines.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # nopep8

from absl import app, flags

from collections import defaultdict
from google.protobuf import text_format
import numpy as np
import json
import tensorflow as tf
import minigo_model

flags.DEFINE_string('src_path', None, 'Source model path.')
flags.DEFINE_string('dst_path', None, 'Destination model path.')
flags.DEFINE_string('mode', 'first_and_last',
                    'Transpose mode: either "first_and_last" or "all". '
                    '"first_and_last" inserts transpose ops before the first '
                    'convolution and the last convolution ops in the graph. '
                    '"all" inserts tranpose ops before and after every '
                    'convolution op in the graph.')

FLAGS = flags.FLAGS


# Template for a permutation constant op used by a tranpose.
PERMUTE_TMPL = """
    name: "%s"
    op: "Const"
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
            dim {
              size: %d
            }
          }
        }
      }
    }
"""


# Template for a tranpose op.
TRANSPOSE_TMPL = """
    name: "%s"
    op: "Transpose"
    input: "%s"
    input: "%s"
    attr {
      key: "T"
      value {}
    }
    attr {
      key: "Tperm"
      value {
        type: DT_INT32
      }
    }
"""


def traverse(graph_def, root, term_op):
    """Returns the depth-first traversal of a GraphDef.

    Args:
      graph_def: GraphDef to traverse.
      root: name of the node to start traversal at.
      term_op: operation type to stop traversal at.

    Returns:
      A (traversal, terms) pair of string lists. The first list contains
      the names of the nodes visited during traversal, the second list
      contains the names of the nodes of type term_op that stopped the
      traversal.
    """

    # Build a map from node name to all nodes that use it as an input, i.e. the
    # node's outputs.
    nodes = {}
    outputs = defaultdict(list)
    for node in graph_def.node:
        nodes[node.name] = node
        for input in node.input:
            outputs[input].append(node.name)

    traversal = []
    pending = [root]
    visited = set(pending)
    terms = set()
    while pending:
        node_name = pending.pop(0)
        if nodes[node_name].op != term_op:
            traversal.append(node_name)
            for output in outputs[node_name]:
                if output not in visited:
                    visited.add(node_name)
                    pending.append(output)
        else:
            terms.add(node_name)
    return traversal, sorted(terms)


def find_node(graph_def, name):
    for n in graph_def.node:
        if n.name == name:
            return n
    raise KeyError(name)


def make_transpose(transpose_name, input_name, input_type, perm):
    """Makes a transpose node.

    Args:
      transpose_name: name of the transpose op.
      input_name: name of the op to be the tranpose op's input.
      input_type: type of the input node.
      perm: permutation array, e.g. [0, 2, 3, 1] for NCHW to NHWC.

    Returns:
      A (transpose, permation) pair of NodeDefs to be added to a GraphDef.
    """

    perm_bytes = np.array(perm, dtype=np.int32).tobytes()
    perm_def = PERMUTE_TMPL % (transpose_name + '/perm', len(perm))
    perm_node = tf.compat.v1.NodeDef()
    text_format.Merge(perm_def, perm_node)
    perm_node.attr['value'].tensor.tensor_content = perm_bytes

    transpose_def = TRANSPOSE_TMPL % (
        transpose_name, input_name, perm_node.name)
    transpose_node = tf.compat.v1.NodeDef()
    text_format.Merge(transpose_def, transpose_node)
    transpose_node.attr['T'].type = input_type

    return transpose_node, perm_node


def transpose_conv(graph_def, conv_name):
    """Transpose the input and output of a Conv2D from NCHW to NHWC.

    Args:
      graph_def: a GraphDef
      conv_name: name of the Conv2D node.
    """

    conv_node = find_node(graph_def, conv_name)
    assert conv_node.attr['data_format'].s == b'NCHW'
    conv_node.attr['data_format'].s = b'NHWC'

    input_name = conv_node.input[0]
    input_node = find_node(graph_def, input_name)
    input_type = input_node.attr.get('T', None) or input_node.attr.get('DstT')
    assert input_type is not None, node
    input_type = input_type.type

    transpose_node_in, perm_node_in = make_transpose(
        conv_name + '_transpose_in', input_name, input_type, [0, 2, 3, 1])
    conv_node.input[0] = transpose_node_in.name

    transpose_node_out, perm_node_out = make_transpose(
        conv_name + '_transpose_out', conv_name, input_type, [0, 3, 1, 2])
    for n in graph_def.node:
        for i, input in enumerate(n.input):
            if input == conv_name:
                n.input[i] = transpose_node_out.name

    graph_def.node.append(transpose_node_in)
    graph_def.node.append(perm_node_in)
    graph_def.node.append(transpose_node_out)
    graph_def.node.append(perm_node_out)


def transpose_input(graph_def, node_name, input_idx, transpose_name, perm):
    """Transposes the input to an op.

    The transpose nodes are added to the graph def and all references to
    the input are replaced with the transpose.

    Args:
      graph_def: a GraphDef.
      node_name: name of the node whose input should be transposed.
      input_idx: index of the input to be transposed.
      transpose_name: name of the inserted transpose op.
      perm: permutation array, e.g. [0, 2, 3, 1] for NCHW to NHWC.
    """

    node = find_node(graph_def, node_name)
    input_name = node.input[input_idx]
    input_node = find_node(graph_def, input_name)
    input_type = node.attr.get('T', None) or node.attr.get('DstT')
    assert input_type is not None, node
    input_type = input_type.type

    transpose_node, perm_node = make_transpose(
        transpose_name, input_name, input_type, perm)

    for n in graph_def.node:
        for i, input in enumerate(n.input):
            if input == input_name:
                n.input[i] = transpose_name

    graph_def.node.append(perm_node)
    graph_def.node.append(transpose_node)


def transpose_all_convs(graph_def):
    """Transposes the inputs and output of all Conv2D nodes in a graph.

    Args:
        graph_def: a GraphDef.
    """

    for node in list(graph_def.node):
        if node.op == 'Conv2D':
            transpose_conv(graph_def, node.name)


def transpose_first_and_last_convs(graph_def, input_node, term_op):
    """Transposes the first input and last outputs of Conv2D nodes.

    This assumes all Conv2D nodes in the graph are in a residual tower,
    i.e. the only nodes between the first and last conv nodes are activations,
    batchnorms and skip connections.

    Args:
        graph_def: a GraphDef.
        input_node: name of the input node to the graph ('pos_tensor' for a
                    Minigo model).
        term_op: the op type at which to stop traversal ('Reshape' for a
                 Minigo model).
    """

    traversal, reshapes = traverse(graph_def, input_node, term_op)
    assert len(reshapes) == 2

    first_conv = None
    for node_name in traversal:
        node = find_node(graph_def, node_name)
        if node.op == 'Conv2D':
            first_conv = node
            break

    transpose_input(graph_def, first_conv.name, 0, 'to_nhwc', [0, 2, 3, 1])

    for node_name in reshapes:
        transpose_input(graph_def, node_name, 0, node_name + '_to_nchw',
                        [0, 3, 1, 2])

    # Change the data format for conv and batch norm nodes to NHWC.
    for node_name in traversal:
        node = find_node(graph_def, node_name)
        data_format = node.attr.get('data_format', None)
        if data_format is None:
            assert node.op not in ['Conv2D', 'FusedBatchNormV3']
        else:
            assert node.op in ['Conv2D', 'FusedBatchNormV3']
            assert data_format.s == b'NCHW'
            node.attr['data_format'].s = b'NHWC'


def main(unused_argv):
    metadata, model_bytes = minigo_model.read_model(FLAGS.src_path)
    assert metadata['input_layout'] == 'nchw'

    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(model_bytes)

    if FLAGS.mode == 'first_and_last':
        transpose_first_and_last_convs(graph_def, 'pos_tensor', 'Reshape')
    elif FLAGS.mode == 'all':
        transpose_all_convs(graph_def)
    else:
        raise ValueError('Unexpected transpose mode.')

    minigo_model.write_graph_def(graph_def, metadata, FLAGS.dst_path)


if __name__ == "__main__":
    app.run(main)
