"""Runs a RL loop locally. Mostly for integration testing purposes.

A successful run will bootstrap, selfplay, gather, and start training for
a while. You should see the combined_cost variable drop steadily, and ideally
overfit to a near-zero loss.
"""

import os
import tempfile

import dual_net
import go
import main

go.set_board_size(9)

def rl_loop():
    # monkeypatch the hyperparams so that we get a quickly executing network.
    dual_net.get_default_hyperparams = lambda **kwargs: {'k': 1, 'fc_width': 2, 'num_shared_layers': 1, 'l2_strength': 2e-4,}

    with tempfile.TemporaryDirectory() as base_dir:
        model_save_file = os.path.join(base_dir, 'models', '000000-bootstrap')
        selfplay_dir = os.path.join(base_dir, 'data', 'selfplay')
        model_selfplay_dir = os.path.join(selfplay_dir, '000000-bootstrap', 'worker1')
        gather_dir = os.path.join(base_dir, 'data', 'training_chunks')
        sgf_dir = os.path.join(base_dir, 'sgf', '000000-bootstrap')
        os.mkdir(os.path.join(base_dir, 'data'))

        print("Creating random initial weights...")
        dual_net.DualNetworkTrainer(model_save_file).bootstrap()
        print("Playing some games...")
        # Do two selfplay runs to test gather functionality
        main.selfplay(
            load_file=model_save_file,
            output_dir=model_selfplay_dir,
            output_sgf=sgf_dir,
            readouts=5,
            games=2,
            n=9)
        main.selfplay(
            load_file=model_save_file,
            output_dir=model_selfplay_dir,
            output_sgf=sgf_dir,
            readouts=5,
            games=2,
            n=9)
        print("Gathering game output...")
        main.gather(input_directory=selfplay_dir, output_directory=gather_dir)
        print("Training on gathered game data... (ctrl+C to quit)")
        main.train(gather_dir, save_file=model_save_file, n=9, num_steps=10000)

if __name__ == '__main__':
    rl_loop()