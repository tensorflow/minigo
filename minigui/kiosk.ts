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
import * as lyr from './layer'
import {Log} from './log'
import {Position} from './position'
import {toPrettyResult} from './util'
import {WinrateGraph} from './winrate_graph'

class KioskApp extends App {
  private winrateGraph = new WinrateGraph('winrate-graph');
  private log = new Log('log');
  private boards: Board[] = [];

  constructor() {
    super();

    this.connect().then(() => {
      this.boards = [
        new Board('main-board', this.rootPosition, [
            new lyr.Label(),
            new lyr.BoardStones(),
            new lyr.Variation('pv'),
            new lyr.Annotations()]),
        new Board('search-board', this.rootPosition, [
            new lyr.Caption('search'),
            new lyr.BoardStones(),
            new lyr.Variation('search')]),
        new Board('n-board', this.rootPosition, [
            new lyr.Caption('N'),
            new lyr.VisitCountHeatMap(),
            new lyr.BoardStones()]),
        new Board('dq-board', this.rootPosition, [
            new lyr.Caption('ΔQ'),
            new lyr.DeltaQHeatMap(),
            new lyr.BoardStones()]),
      ];

      this.gtp.onText((line: string) => { this.log.log(line, 'log-cmd'); });
      this.newGame();
    });
  }

  protected newGame() {
    super.newGame();
    this.log.clear();
    this.winrateGraph.newGame(this.rootPosition);
  }

  protected onPositionUpdate(position: Position, update: Position.Update) {
    if (position != this.activePosition) {
      return;
    }
    for (let board of this.boards) {
      board.update(update);
    }
    this.winrateGraph.update(position);
  }

  protected onNewPosition(position: Position) {
    this.activePosition = position
    for (let board of this.boards) {
      board.setPosition(position);
    }
    this.winrateGraph.update(position);
    this.log.scroll();

    if (this.activePosition.gameOver) {
      window.setTimeout(() => { this.newGame(); }, 3000);
    } else {
      this.gtp.send('genmove');
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
