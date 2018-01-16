import itertools
import gzip
import numpy as np
import copy
import os
import struct
import sys
from collections import namedtuple
from tensorflow import gfile

from features import bulk_extract_features
import features
import go
from sgf_wrapper import replay_sgf

# Number of data points to store in a chunk on disk
CHUNK_SIZE = 4096
CHUNK_HEADER_FORMAT = "iii?"
CHUNK_HEADER_SIZE = struct.calcsize(CHUNK_HEADER_FORMAT)

DEBUG = False

def find_sgf_files(*dataset_dirs):
    for dataset_dir in dataset_dirs:
        full_dir = os.path.join(os.getcwd(), dataset_dir)
        dataset_files = [os.path.join(full_dir, name) for name in os.listdir(full_dir)]
        for f in dataset_files:
            if os.path.isfile(f) and f.endswith(".sgf"):
                yield f

def get_positions_from_sgf(file):
    with open(file) as f:
        for position_w_context in replay_sgf(f.read()):
            yield position_w_context

class DataSetV2(object):
    def __init__(self, pos_features, next_moves, results, is_test=False):
        '''
        pos_features, next_moves, results, must all be iterables of the same length, where each iterables items
        x[i] must correspond to the each others items y[i].
        pos_features -- array of features, uint8
        next_moves -- array of next move search probabilities, float32
        results -- array of final results, z, int8, either -1 or 1
        '''
        self.pos_features = pos_features
        self.next_moves = next_moves
        assert np.all([np.isclose(np.sum(nms), 1) for nms in self.next_moves])
        self.results = results
        self.is_test = is_test
        self.data_size = pos_features.shape[0]
        assert next_moves.shape[0] == self.data_size, "Next_moves array wrong size"
        assert results.shape[0] == self.data_size, "Results array wrong size"
        self.board_size = pos_features.shape[1]
        self.input_planes = pos_features.shape[-1]
        self._index_within_epoch = 0

    @staticmethod
    def from_positions_w_context(positions_w_context,
                                 searches,
                                 results,
                                 is_test=False,
                                 features=features.NEW_FEATURES):
        positions, _, _ = zip(*positions_w_context)
        extracted_features = bulk_extract_features(positions, features=features)

        return DataSetV2(extracted_features,
                         np.array(searches, dtype=np.float32),
                         results,
                         is_test=is_test)

    def write(self, filename):
        header_bytes = struct.pack(CHUNK_HEADER_FORMAT, self.data_size, self.board_size, self.input_planes, self.is_test)
        position_bytes = np.packbits(self.pos_features).tobytes()
        next_move_bytes = self.next_moves.tobytes()
        result_bytes = self.results.tobytes()

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with gzip.open(filename, "wb", compresslevel=6) as f:
            f.write(header_bytes)
            f.write(position_bytes)
            f.write(next_move_bytes)
            f.write(result_bytes)
        bytes_written = list(map(len, [header_bytes, position_bytes, next_move_bytes, result_bytes]))
        if filename.endswith('.gz'):
            self.write_meta(filename.replace('.gz', '.meta'))
        if DEBUG:
            print('''
            Wrote {0} positions ...
             {2} bytes header
             {3} bytes position
             {4} bytes next moves / search
             {5} bytes results
             {1} total
            '''.format(self.data_size, sum(bytes_written), *bytes_written))

    @staticmethod
    def read(filename):
        with gfile.Open(filename, 'rb') as gf:
            with gzip.open(gf, "rb") as f:
                header_bytes = f.read(CHUNK_HEADER_SIZE)
                data_size, board_size, input_planes, is_test = struct.unpack(
                        CHUNK_HEADER_FORMAT, header_bytes)

                position_dims = data_size * board_size * board_size * input_planes
                next_move_dims = data_size * ((board_size * board_size) +1) * 4
                result_dims = data_size

                # the +7 // 8 compensates for numpy's bitpacking padding
                packed_position_bytes = f.read((position_dims + 7) // 8)
                packed_next_move_bytes = f.read(next_move_dims)
                packed_result_bytes = f.read(result_dims)

                bytes_read = list(map(len, [header_bytes,
                                            packed_position_bytes,
                                            packed_next_move_bytes,
                                            packed_result_bytes]))
                if DEBUG:
                    print('''
                Reading {0} records...
                 {2} bytes header
                 {3} bytes position
                 {4} bytes next moves / search
                 {5} bytes results
                 {1} total
                    '''.format(data_size, sum(bytes_read), *bytes_read ))

                # should have cleanly finished reading all bytes from file!
                assert len(f.read()) == 0

                flat_position = np.unpackbits(np.fromstring(
                        packed_position_bytes, dtype=np.uint8))[:position_dims]
                flat_nextmoves = np.fromstring(
                        packed_next_move_bytes, dtype=np.float32)[:next_move_dims]
                flat_results = np.fromstring(packed_result_bytes, dtype=np.int8)[:result_dims]

                pos_features = flat_position.reshape(data_size, board_size, board_size, input_planes)
                next_moves = flat_nextmoves.reshape(data_size, (board_size * board_size) + 1)
                results = flat_results.reshape(-1)

        return DataSetV2(pos_features, next_moves, results, is_test=is_test)

    @property
    def size(self):
        return self.pos_features.shape[0]
