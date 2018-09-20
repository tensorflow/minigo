// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import {App, GameStateMsg} from './app'
import {Board} from './board'
import * as lyr from './layer'
import {heatMapN, heatMapDq} from './heat_map'
import {Log} from './log'
import {WinrateGraph} from './winrate_graph'

class KioskApp extends App {
  private winrateGraph = new WinrateGraph('winrate-graph');
  private log = new Log('log');

  constructor() {
    super();

    this.connect().then(() => {
      let mainBoard = new Board(
        'main-board', this.size,
        [lyr.Label, lyr.BoardStones, [lyr.Variation, 'pv'], lyr.Annotations]);

      let searchBoard = new Board(
        'search-board', this.size,
        [[lyr.Caption, 'search'], lyr.BoardStones, [lyr.Variation, 'search']]);

      let nBoard = new Board(
        'n-board', this.size,
        [[lyr.Caption, 'N'], [lyr.HeatMap, 'n', heatMapN], lyr.BoardStones]);

      let dqBoard = new Board(
        'dq-board', this.size,
        [[lyr.Caption, 'ΔQ'], [lyr.HeatMap, 'dq', heatMapDq], lyr.BoardStones]);

      this.init([mainBoard, searchBoard, nBoard, dqBoard]);

      this.gtp.onText((line: string) => { this.log.log(line, 'log-cmd'); });
      this.newGame();
    });
  }

  protected newGame() {
    super.newGame();
    this.log.clear();
    this.winrateGraph.clear();
  }

  protected onGameState(msg: GameStateMsg) {
    super.onGameState(msg);
    this.log.scroll();
    this.winrateGraph.setWinrate(msg.moveNum, msg.q);

    if (this.gameOver) {
      window.setTimeout(() => { this.newGame(); }, 3);
    } else {
      this.genmove();
    }
  }

  private genmove() {
    this.gtp.send('genmove').then((move: string) => {
      this.gtp.send('gamestate');
    });
  }

  protected onGameOver() {
    this.gtp.send('final_score').then((result: string) => {
      let prettyResult: string;
      if (result[0] == 'W') {
        prettyResult = 'White wins by ';
      } else {
        prettyResult = 'Black wins by ';
      }
      if (result[2] == 'R') {
        prettyResult += 'resignation';
      } else {
        prettyResult += result.substr(2) + ' points';
      }
      this.log.log(prettyResult);
      this.log.scroll();
    });
  }
}

new KioskApp();
