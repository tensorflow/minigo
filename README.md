MuGo Zero: A minimalist Go engine modeled after AlphaGo Zero, built on MuGo
==================================================

This is a pure Python implementation of a neural-network based Go AI, using TensorFlow.

This is based off of Brian Lee's "MuGo" -- a pure Python implementation of the
first AlphaGo paper published in Nature, adapted and extended to the
architecture described in the second.

The goal of this project is to reproduce the results of the original paper
through an open-source implementation using the Google Cloud Platform and
open-source pipeline tools.  A secondary goal is to provide a clear set of
learning examples for establishing RL pipelines on various hardware
accelerators available in TensorFlow.

An explicit non-goal is to produce a competitive program, and the original
non-goal of MuGo -- "diving into the fiddly bits of optimizing" -- is preserved
in the service of the secondary goal above :)

Getting Started
===============

Install Tensorflow
------------------
Start by installing Tensorflow. This should be as simple as

```python
pip install -r requirements.txt
```

Optionally, you can install TensorFlow with GPU support, if you intend on training a network yourself.

Play against MuGo Zero
======================

If you just want to get MuGo Zero working... (TODO)


MuGo uses the GTP protocol, and you can use any gtp-compliant program with it. To invoke the raw policy network, use
```
python main.py gtp policy --read-file=saved_models/20170718
```

(An MCTS version of MuGo has been implemented, using the policy network to simulate games, but it's not that much better than just the raw policy network, because Python is slow at simulating full games.)

One way to play via GTP is to use gogui-display (which implements a UI that speaks GTP.) You can download the gogui set of tools at [http://gogui.sourceforge.net/](http://gogui.sourceforge.net/). See also [documentation on interesting ways to use GTP](http://gogui.sourceforge.net/doc/reference-twogtp.html).
```
gogui-twogtp -black 'python main.py gtp policy --read-file=saved_models/20170718' -white 'gogui-display' -size 19 -komi 7.5 -verbose -auto
```

Another way to play via GTP is to play against GnuGo, while spectating the games
```
BLACK="gnugo --mode gtp"
WHITE="python main.py gtp policy --read-file=saved_models/20170718"
TWOGTP="gogui-twogtp -black \"$BLACK\" -white \"$WHITE\" -games 10 \
  -size 19 -alternate -sgffile gnugo"
gogui -size 19 -program "$TWOGTP" -computer-both -auto
```

Another way to play via GTP is to connect to CGOS, the [Computer Go Online Server](http://yss-aya.com/cgos/). The CGOS server hosted by boardspace.net is actually abandoned; you'll want to connect to the CGOS server at yss-aya.com. 

After configuring your cgos.config file, you can connect to CGOS with `cgosGtp -c cgos.config` and spectate your own game with `cgosView yss-aya.com 6819`

Running unit tests
------------------
```python
python -m unittest discover tests
```

Running MugoZero Self-Play
=========================

(TODO)

...
As the network is trained, the current model will be saved at `--save-file`. If you reexecute the same command, the network will pick up training where it left off.

Additionally, you can follow along with the training progress with TensorBoard - if you give each run a different name (`logs/my_training_run`, `logs/my_training_run2`), you can overlay the runs on top of each other.
```
tensorboard --logdir=logs/
```

Running MugoZero on a Cluster
=============================

(TODO)


Running MuGoZero on a GPU-Accelerated Cluster
=============================================

(TODO)
