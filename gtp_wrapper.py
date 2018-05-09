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

from gtp_cmd_handlers import *
import gtp_engine
import os
from dual_net import DualNetwork
from strategies import MCTSPlayer, CGOSPlayer


def make_gtp_instance(read_file, readouts_per_move=100, verbosity=1, cgos_mode=False, kgs_mode=False):
    n = DualNetwork(read_file)
    if cgos_mode:
        player = CGOSPlayer(network=n, seconds_per_move=5, timed_match=True,
                            verbosity=verbosity, two_player_mode=True)
    else:
        player = MCTSPlayer(network=n, num_readouts=readouts_per_move,
                            verbosity=verbosity, two_player_mode=True)

    name = "Minigo-" + os.path.basename(read_file)
    version = "0.2"

    engine = gtp_engine.Engine()
    engine.add_cmd_handler(
        gtp_engine.EngineCmdHandler(engine, name, version))

    if kgs_mode:
        engine.add_cmd_handler(KgsCmdHandler(player))
    engine.add_cmd_handler(RegressionsCmdHandler(player))
    engine.add_cmd_handler(GoGuiCmdHandler(player))
    engine.add_cmd_handler(MiniguiCmdHandler(player, courtesy_pass=kgs_mode))

    return engine
