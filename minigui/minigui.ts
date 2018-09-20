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

import {App, Position, GameStateMsg} from './app'
import {Color, Move, Nullable} from './base'
import {Board, ClickableBoard, COL_LABELS} from './board'
import {getElement} from './util'
import * as lyr from './layer'
import {heatMapN, heatMapDq} from './heat_map'
import {Log} from './log'
import {WinrateGraph} from './winrate_graph'

const HUMAN = 'Human';
const MINIGO = 'Minigo';

class DemoApp extends App {
  private mainBoard: ClickableBoard;
  private playerElems: HTMLElement[] = [];
  private winrateGraph = new WinrateGraph('winrate-graph');
  private log = new Log('log', 'console');

  constructor() {
    super();

    this.connect().then(() => {
      this.mainBoard = new ClickableBoard(
        'main-board', this.size,
        [lyr.Label, lyr.BoardStones, [lyr.Variation, 'pv'], lyr.Annotations]);

      let boards: Board[] = [this.mainBoard];

      let searchElem = getElement('search-board');
      if (searchElem) {
        boards.push(new Board(
            searchElem, this.size,
            [[lyr.Caption, 'search'], lyr.BoardStones, [lyr.Variation, 'search']]));
      }

      let nElem = getElement('n-board');
      if (nElem) {
        boards.push(new Board(
          nElem, this.size,
          [[lyr.Caption, 'N'], [lyr.HeatMap, 'n', heatMapN], lyr.BoardStones]));
      }

      let dqElem = getElement('dq-board');
      if (dqElem) {
        boards.push(new Board(
          'dq-board', this.size,
          [[lyr.Caption, 'ΔQ'], [lyr.HeatMap, 'dq', heatMapDq], lyr.BoardStones]));
      }

      this.init(boards);

      this.mainBoard.onClick((p) => {
        this.playMove(this.toPlay, p);
      });

      this.initButtons();

      this.winrateGraph.onMoveChanged((moveNum: Nullable<number>) => {
        let position: Position;
        if (moveNum == null || moveNum < 0 ||
            moveNum >= this.positionHistory.length) {
          position = this.positionHistory[this.positionHistory.length - 1];
        } else {
          position = this.positionHistory[moveNum];
        }
        if (position == this.activePosition) {
          return;
        }
        this.activePosition = position;
        this.updateBoards(position);
      });

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
        this.playMove(this.toPlay, 'pass');
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

    if (this.playerElems[this.toPlay].innerText == MINIGO) {
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

  protected onGameState(msg: GameStateMsg) {
    super.onGameState(msg);
    this.log.scroll();
    this.winrateGraph.setWinrate(msg.moveNum, msg.q);
    this.onPlayerChanged();
  }

  private playMove(color: Color, move: Move) {
    let colorStr = color == Color.Black ? 'b' : 'w';
    let moveStr: string;
    if (move == 'pass') {
      moveStr = move;
    } else if (move == 'resign') {
      // TODO(tommadams): support resign moves.
      throw new Error('resign not yet supported');
    } else {
      let row = this.size - move.row;
      let col = COL_LABELS[move.col];
      moveStr = `${col}${row}`;
    }
    this.gtp.send(`play ${colorStr} ${moveStr}`).then(() => {
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

new DemoApp();
