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

import {App, GameStateMsg, Position} from './app'
import {COL_LABELS, Color, Move, N, Nullable} from './base'
import {Board, ClickableBoard} from './board'
import {heatMapDq, heatMapN} from './heat_map'
import * as lyr from './layer'
import {Log} from './log'
import {WinrateGraph} from './winrate_graph'
import {getElement, toPrettyResult} from './util'
import {VariationTree, testVariationTree} from './variation_tree'

const HUMAN = 'Human';
const MINIGO = 'Minigo';

testVariationTree();

// Demo app implementation that's shared between full and lightweight demo UIs.
class DemoApp extends App {
  private mainBoard: ClickableBoard;
  private readsBoard: Board;
  private playerElems: HTMLElement[] = [];
  private winrateGraph = new WinrateGraph('winrate-graph');
  private log = new Log('log', 'console');

  constructor() {
    super();

    this.connect().then(() => {
      // Create boards for each of the elements in the UI.
      // The extra board views aren't available in the lightweight UI, so we
      // must check if the HTML elements exist.
      this.mainBoard = new ClickableBoard(
        'main-board',
        [lyr.Label, lyr.BoardStones, [lyr.Variation, 'pv'], lyr.Annotations]);
      this.mainBoard.enabled = true;

      this.readsBoard = new Board(
        'reads-board',
        [lyr.BoardStones]);

      this.init([this.mainBoard, this.readsBoard]);

      this.mainBoard.onClick((p) => {
        this.playMove(this.toPlay, p);
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
        this.playMove(this.toPlay, 'pass');
      }
    });

    getElement('load-sgf-input').addEventListener('change', (e: any) => {
      let files: File[] = Array.prototype.slice.call(e.target.files);
      if (files.length != 1) {
        let names: string[] = [];
        files.forEach((f) => { names.push(`"${f.name}"`); });
        throw new Error(`Expected one file, got [${names.join(', ')}]`);
      }
      let reader = new FileReader();
      reader.onload = () => {
        this.newGame();
        let sgf = reader.result.replace(/\n/g, '\\n');
        this.gtp.send(`playsgf ${sgf}`).then(() => {
          console.log('ok!');
        });
      };
      reader.readAsText(files[0]);
    });
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
      let row = N - move.row;
      let col = COL_LABELS[move.col];
      moveStr = `${col}${row}`;
    }
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
