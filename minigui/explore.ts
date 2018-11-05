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
import {COL_LABELS, Color, Move, N, Nullable, Point, otherColor, toKgs} from './base'
import {Board, ClickableBoard} from './board'
import {heatMapDq, heatMapN} from './heat_map'
import * as lyr from './layer'
import {Log} from './log'
import {Position} from './position'
import {getElement, toPrettyResult} from './util'
import {VariationTree} from './variation_tree'
import {WinrateGraph} from './winrate_graph'

// Demo app implementation that's shared between full and lightweight demo UIs.
class DemoApp extends App {
  private mainBoard: ClickableBoard;
  private readsBoard: Board;
  private winrateGraph = new WinrateGraph('winrate-graph');
  private variationTree = new VariationTree('tree');
  private log = new Log('log', 'console');

  private pvLayer: lyr.Layer;
  private bestMovesLayer: lyr.BestMoves;

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

      this.pvLayer = this.mainBoard.getLayer(2);

      this.readsBoard = new Board(
        'reads-board',
        [lyr.BoardStones, lyr.BestMoves]);
      this.bestMovesLayer = this.readsBoard.getLayer(1) as lyr.BestMoves;

      this.init([this.mainBoard, this.readsBoard]);

      this.mainBoard.onClick((p: Point) => {
        this.mainBoard.enabled = false;
        this.bestMovesLayer.clear();
        this.playMove(this.activePosition.toPlay, p).then(() => {
          let parent = this.activePosition;
          this.gtp.send('gamestate').then(() => {
            this.variationTree.addChild(parent, this.activePosition);
          }).finally(() => {
            this.mainBoard.enabled = true;
          });
        });
      });

      this.initButtons();

      // Initialize log.
      this.log.onConsoleCmd((cmd: string) => {
        this.gtp.send(cmd).then(() => { this.log.scroll(); });
      });

      this.gtp.onText((line: string) => { this.log.log(line, 'log-cmd'); });
      this.newGame();

      this.variationTree.onClick((positions: Position[]) => {
        let moves = [];
        for (let position of positions) {
          if (position.lastMove != null) {
            moves.push(toKgs(position.lastMove));
          }
        }
        this.gtp.send('clear_board');
        if (moves.length > 0) {
          this.gtp.send(`play_multiple b ${moves.join(' ')}`);
        }
        this.gtp.send('gamestate');
        this.activePosition = positions[positions.length - 1];
      });
    });
  }

  private initButtons() {
    getElement('toggle-pv').addEventListener('click', (e: any) => {
      this.pvLayer.hidden = !this.pvLayer.hidden;
      if (this.pvLayer.hidden) {
        e.target.innerText = 'Show PV';
      } else {
        e.target.innerText = 'Hide PV';
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
          this.variationTree.draw();
        });
      };
      reader.readAsText(files[0]);
    });

    getElement('main-line').addEventListener('click', () => {
      let position = this.activePosition;
      while (position != this.rootPosition &&
             !position.isMainline && position.parent != null) {
        position = position.parent;
      }
      if (position != this.activePosition) {
        this.activePosition = position;
        this.variationTree.setActive(this.activePosition);
        this.updateBoards(this.activePosition);
      }
    });
  }

  protected newGame() {
    this.gtp.send('prune_nodes 1');
    super.newGame();
    this.bestMovesLayer.clear();
    this.variationTree.newGame(this.rootPosition);
    this.gtp.send('prune_nodes 0');
    this.log.clear();
    this.winrateGraph.clear();
  }

  protected onPosition(position: Position) {
    if (position.parent == this.activePosition) {
      this.activePosition = position;
      this.bestMovesLayer.clear();
    }
    this.updateBoards(position);
    this.log.scroll();
    this.winrateGraph.setWinrate(position.moveNum, position.q);
    if (position.parent != null) {
      this.variationTree.addChild(position.parent, position);
    }
  }

  private playMove(color: Color, move: Move) {
    let colorStr = color == Color.Black ? 'b' : 'w';
    let moveStr = toKgs(move);
    return this.gtp.send(`play ${colorStr} ${moveStr}`);
  }

  protected onGameOver() {
    this.gtp.send('final_score').then((result: string) => {
      this.log.log(toPrettyResult(result));
      this.log.scroll();
    });
  }
}

new DemoApp();
