/*
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
import {COL_LABELS, Color, Move, N, Nullable, toKgs} from './base'
import {Board, ClickableBoard} from './board'
import {heatMapDq, heatMapN} from './heat_map'
import * as lyr from './layer'
import {Log} from './log'
import {Position} from './position'
import {getElement, toPrettyResult} from './util'
import {WinrateGraph} from './winrate_graph'

const HUMAN = 'Human';
const MINIGO = 'Minigo';

// Demo app implementation that's shared between full and lightweight demo UIs.
class DemoApp extends App {
  private mainBoard: ClickableBoard;
  private playerElems: HTMLElement[] = [];
  private winrateGraph = new WinrateGraph('winrate-graph');
  private log = new Log('log', 'console');

  constructor() {
    super();

    this.connect().then(() => {
      // Create boards for each of the elements in the UI.
      // The extra board views aren't available in the lightweight UI, so we
      // must check if the HTML elements exist.
      this.mainBoard = new ClickableBoard('main-board', [
          new lyr.Label(),
          new lyr.BoardStones(),
          new lyr.Variation('pv'),
          new lyr.Annotations()]);

      let boards: Board[] = [this.mainBoard];

      let searchElem = getElement('search-board');
      if (searchElem) {
        boards.push(new Board(searchElem, [
            new lyr.Caption('search'),
            new lyr.BoardStones(),
            new lyr.Variation('search')]));
      }

      let nElem = getElement('n-board');
      if (nElem) {
        boards.push(new Board(nElem, [
            new lyr.Caption('N'),
            new lyr.HeatMap('n', heatMapN),
            new lyr.BoardStones()]));
      }

      let dqElem = getElement('dq-board');
      if (dqElem) {
        boards.push(new Board('dq-board', [
            new lyr.Caption('ΔQ'),
            new lyr.HeatMap('dq', heatMapDq),
            new lyr.BoardStones()]));
      }

      this.init(boards);

      this.mainBoard.onClick((p) => {
        this.playMove(this.activePosition.toPlay, p);
      });

      this.initButtons();

      // Initialize log.
      this.log.onConsoleCmd((cmd: string) => {
        this.gtp.send(cmd).then(() => { this.log.scroll(); });
      });

      this.gtp.onText((line: string) => { this.log.log(line, 'log-cmd'); });
      this.newGame();
    });
  }

  private initButtons() {
    getElement('pass').addEventListener('click', () => {
      if (this.mainBoard.enabled) {
        this.playMove(this.activePosition.toPlay, 'pass');
      }
    });

    getElement('reset').addEventListener('click', () => {
      this.gtp.newSession();
      this.newGame();
    });

    let initPlayerButton = (color: Color, elemId: string) => {
      let elem = getElement(elemId);
      this.playerElems[color] = elem;
      elem.addEventListener('click', () => {
        if (elem.innerText == HUMAN) {
          elem.innerText = MINIGO;
        } else {
          elem.innerText = HUMAN;
        }
        this.onPlayerChanged();
      });
    };
    initPlayerButton(Color.Black, 'black-player');
    initPlayerButton(Color.White, 'white-player');
  }

  protected newGame() {
    super.newGame();
    this.log.clear();
    this.winrateGraph.clear();
  }

  private onPlayerChanged() {
    if (this.engineBusy || this.gameOver) {
      return;
    }

    if (this.playerElems[this.activePosition.toPlay].innerText == MINIGO) {
      this.mainBoard.enabled = false;
      this.engineBusy = true;
      this.gtp.send('genmove').then((move: string) => {
        this.engineBusy = false;
        this.gtp.send('gamestate');
      });
    } else {
      this.mainBoard.enabled = true;
    }
  }

  protected onPosition(position: Position) {
    this.updateBoards(position);
    this.log.scroll();
    this.winrateGraph.setWinrate(position.moveNum, position.q);
    this.onPlayerChanged();
  }

  private playMove(color: Color, move: Move) {
    let colorStr = color == Color.Black ? 'b' : 'w';
    let moveStr = toKgs(move);
    this.gtp.send(`play ${colorStr} ${moveStr}`).then(() => {
      this.gtp.send('gamestate');
    });
  }

  protected onGameOver() {
    this.gtp.send('final_score').then((result: string) => {
      this.log.log(toPrettyResult(result));
      this.log.scroll();
    });
  }
}

new DemoApp();
*/
