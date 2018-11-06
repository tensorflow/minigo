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

import {App} from './app'
import {Board} from './board'
import {heatMapN, heatMapDq} from './heat_map'
import * as lyr from './layer'
import {Log} from './log'
import {Position} from './position'
import {toPrettyResult} from './util'
import {WinrateGraph} from './winrate_graph'

class KioskApp extends App {
  private winrateGraph = new WinrateGraph('winrate-graph');
  private log = new Log('log');

  constructor() {
    super();

    this.connect().then(() => {
      let mainBoard = new Board('main-board', [
          new lyr.Label(),
          new lyr.BoardStones(),
          new lyr.Variation('pv'),
          new lyr.Annotations()]);

      let searchBoard = new Board('search-board', [
          new lyr.Caption('search'),
          new lyr.BoardStones(),
          new lyr.Variation('search')]);

      let nBoard = new Board('n-board', [
          new lyr.Caption('N'),
          new lyr.HeatMap('n', heatMapN),
          new lyr.BoardStones()]);

      let dqBoard = new Board('dq-board', [
          new lyr.Caption('ΔQ'),
          new lyr.HeatMap('dq', heatMapDq),
          new lyr.BoardStones()]);

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

  protected onPosition(position: Position) {
    this.log.scroll();
    this.winrateGraph.setWinrate(position.moveNum, position.q);
    this.updateBoards(position);

    if (this.gameOver) {
      window.setTimeout(() => { this.newGame(); }, 3000);
    } else {
      this.gtp.send('genmove').then((move: string) => {
        this.gtp.send('gamestate');
      });
    }
  }

  protected onGameOver() {
    this.gtp.send('final_score').then((result: string) => {
      this.log.log(toPrettyResult(result));
      this.log.scroll();
    });
  }
}

new KioskApp();
