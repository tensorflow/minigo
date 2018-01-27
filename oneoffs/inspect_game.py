import sys; sys.path.insert(0, '.')
import dual_net
import strategies
import sgf_wrapper
import numpy as np
import pdb
import random
import features
import symmetries

def initialize_game(sgf_file, load_file, move=1):
    with open(sgf_file) as f:
        sgf_contents = f.read()
    iterator = sgf_wrapper.replay_sgf(sgf_contents)
    for i in range(move):
        position_w_context = next(iterator)
    player = strategies.MCTSPlayerMixin(dual_net.DualNetwork(load_file))
    player.initialize_game(position_w_context.position)
    return player

def analyze_symmetries(sgf_file, load_file):
    with open(sgf_file) as f:
        sgf_contents = f.read()
    iterator = sgf_wrapper.replay_sgf(sgf_contents)
    net = dual_net.DualNetwork(load_file)
    for i, pwc in enumerate(iterator):
        if i < 200:
            continue
        feats = features.extract_features(pwc.position)
        variants = [symmetries.apply_symmetry_feat(s, feats) for s in symmetries.SYMMETRIES]
        values = net.sess.run(
            net.inference_output['value_output'],
            feed_dict={net.inference_input['pos_tensor']: variants})
        mean = np.mean(values)
        stdev = np.std(values)
        all_vals = sorted(zip(values, symmetries.SYMMETRIES))

        print("{:3d} {:.3f} +/- {:.3f} min {:.3f} {} max {:.3f} {}".format(
            i, mean, stdev, *all_vals[0], *all_vals[-1]))


