# Copyright 2019 Google LLC
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
Library for writing Minigo model files.
"""

import json
import struct
import tensorflow as tf


MAGIC = '<minigo>'
MAGIC_SIZE = len(MAGIC)
HEADER_SIZE = 24


def _is_supported_metadata_type(value):
    for t in [int, str, float, bool]:
        if isinstance(value, t):
            return True
    return False


def write_graph_def(graph_def, metadata, dst_path):
    """Writes a TensorFlow GraphDef & metadata in Minigo format.

    Args:
      model_bytes: the serialized model.
      metadata: a dictionary of metadata to write to file.
      dst_path: destination path to write to.
    """
    write_model_bytes(graph_def.SerializeToString(), metadata, dst_path)


def write_model_bytes(model_bytes, metadata, dst_path):
    """Writes a serialized model & metadata in Minigo format.

    Args:
      model_bytes: the serialized model.
      metadata: a dictionary of metadata to write to file.
      dst_path: destination path to write to.
    """

    for key, value in metadata.items():
        assert isinstance(key, str), '%s is not a string' % key
        assert _is_supported_metadata_type(value), '%s: unsupported type %s' % (
            key, type(value).__name__)

    metadata_bytes = json.dumps(metadata, sort_keys=True,
                                separators=(',', ':')).encode()

    # If the destination path is on GCS, write there directly since GCS files
    # are immutable and a partially written file cannot be observed.
    # Otherwise, write to a temp file and rename. The temp file is written to
    # the same filesystem as dst_path on the assumption that the rename will be
    # atomic.
    if dst_path.startswith('gs://'):
        write_path = dst_path
    else:
        write_path = dst_path + '.tmp'

    # File header:
    #   char[8]: '<minigo>'
    #   uint64: version
    #   uint64: file size
    #   uint64: metadata size
    version = 1
    header_size = 32
    metadata_size = len(metadata_bytes)
    model_size = len(model_bytes)
    file_size = header_size + metadata_size + model_size
    with tf.io.gfile.GFile(write_path, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<QQQ', version, file_size, metadata_size))
        f.write(metadata_bytes)
        f.write(model_bytes)

    if write_path != dst_path:
        tf.io.gfile.rename(write_path, dst_path, overwrite=True)


def read_model(path):
    """Reads a serialized model & metadata in Minigo format.

    Args:
      path: the model path.

    Returns:
      A (metadata, model_bytes) pair of the model's metadata as a dictionary
      and the serialized model as bytes.
    """

    with tf.io.gfile.GFile(path, 'rb') as f:
        magic = f.read(MAGIC_SIZE).decode('utf-8')
        if magic != MAGIC:
            raise RuntimeError(
                'expected magic string %s, got %s' % (MAGIC, magic))

        version, file_size, metadata_size = struct.unpack(
            '<QQQ', f.read(HEADER_SIZE))
        if version != 1:
            raise RuntimeError('expected version == 1, got %d' % version)

        metadata_bytes = f.read(metadata_size).decode('utf-8')
        if len(metadata_bytes) != metadata_size:
            raise RuntimeError('expected %dB of metadata, read only %dB' % (
                metadata_size, len(metadata_bytes)))

        metadata = json.loads(metadata_bytes)
        model_bytes = f.read()
        model_size = len(model_bytes)

        bytes_read = MAGIC_SIZE + HEADER_SIZE + model_size + metadata_size
        if bytes_read != file_size:
            raise RuntimeError('expected %dB, read only %dB' %
                               (file_size, bytes_read))

    return metadata, model_bytes
