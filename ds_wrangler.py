'''
Operations for wrangling datasets.

Used by the `gather` command, mostly.
'''

from load_data_sets import DataSetV2
import collections
import numpy as np
import os
import random
from tqdm import tqdm

# Separate function for ease of testing.
def _read_meta(meta_path):
    try:
        if not os.path.exists(meta_path.replace('.meta', '.gz')):
            return 0, True
        with open(meta_path) as _file:
            sz = int(_file.read())
            return sz, None
    except:
        print("Error reading metadata for %s" % meta_path)
        return 0, True

def get_paths_to_num_positions(meta_paths, max_positions):
    '''
    Takes a list of paths to .meta files, and the total number of positions needed

    Returns a dict of { cumulative_total : path }
    '''

    # Just read as many .meta files as we need to reach our total...
    paths_to_sizes = {}
    tot = 0
    for p in sorted(meta_paths, reverse=True):
        if tot > max_positions:
            break
        sz, err = _read_meta(p)
        if sz < 15 or err: # Don't train on aborted games. 
            continue
        paths_to_sizes[p] = sz
        tot += sz
    print("Using %d files for %d positions" % (len(paths_to_sizes), tot))
    return paths_to_sizes

def choose_moves_for_chunks(
        cumulative_moves, # [130, 280, 345, ...]
        reversed_paths, # [path_to_game_with_130, path_to_game_with_150...]
        chunks_to_make,
        positions_per_chunk):
    paths_to_moves_by_chunk = collections.defaultdict(lambda:
            collections.defaultdict(set))
    for n in range(chunks_to_make):
        picked = 0
        while picked != positions_per_chunk:
            selection = random.randint(0, cumulative_moves[-1]-1) 

            index = cumulative_moves.searchsorted(selection, side="right")

            if index != 0:
                move = selection - cumulative_moves[index-1]
            else:
                move = selection
            path = reversed_paths[index]

            # Add the movenum to the set
            if not move in paths_to_moves_by_chunk[path][n]:
                paths_to_moves_by_chunk[path][n].add(move)
                picked += 1
    return paths_to_moves_by_chunk

# paths_to_moves_by_chunk is a mapping from each path to a dictionary, which in
# turn maps each numbered chunk to a set of moves to extract from the path.
# E.g.
#    'filename1': { 0: set([..moves..]), 1: set([...])},
#    'filename2': { 0:  ... },
# Let's open them up and pull out the training tuples. 
def gather_moves_and_write(
        paths_to_moves_by_chunk, chunks_to_make, output_directory):
    fname_to_dataset = {}

    # Pre-open all the files we'll need and load them as DataSets
    for filename in paths_to_moves_by_chunk:
        if paths_to_moves_by_chunk[filename]:
            fname_to_dataset[filename] = DataSetV2.read(filename)
    print ("Loaded %d files" % len(fname_to_dataset))

    for n in tqdm(range(chunks_to_make)):
        pos_features,next_moves,results = [], [], []
        for filename, moves_sets in paths_to_moves_by_chunk.items():
            if not n in moves_sets:
                continue # This file not referenced by chunk number n.
            moves = sorted(list(moves_sets[n])) # `np.take` requires a list.
            ds = fname_to_dataset[filename]
            pos_features.append(ds.pos_features.take(moves, axis=0))
            next_moves.append(ds.next_moves.take(moves, axis=0))
            results.append(ds.results.take(moves, axis=0))

        combined = DataSetV2(
                np.concatenate([p for p in pos_features]),
                np.concatenate([nm for nm in next_moves]),
                np.concatenate([r for r in results]))

        fname = os.path.join(output_directory, "training-{:d}.gz".format(n))
        combined.write(fname)

