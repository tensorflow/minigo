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

import {App, SearchMsg} from './app'
import {COL_LABELS, Color, Move, N, Nullable, Point, movesEqual, otherColor, toKgs} from './base'
import {Board, ClickableBoard} from './board'
import {heatMapDq, heatMapN} from './heat_map'
import {Socket} from './gtp_socket'
import * as lyr from './layer'
import {Log} from './log'
import {Annotation, Position} from './position'
import {getElement, parseMove, toPrettyResult} from './util'
import {VariationTree} from './variation_tree'
import {WinrateGraph} from './winrate_graph'

class ExploreBoard extends ClickableBoard {
  private _showSearch = true;
  get showSearch() {
    return this._showSearch;
  }
  set showSearch(x: boolean) {
    if (x != this._showSearch) {
      this._showSearch = x;
      if (x) {
        this.variationLayer.show = false;
        this.qLayer.show = true;
      } else {
        this.variationLayer.show = false;
        this.qLayer.show = false;
      }
      this.draw();
    }
  }

  get showNext() {
    return this.nextLayer.show;;
  }
  set showNext(x: boolean) {
    if (x != this.nextLayer.show) {
      this.nextLayer.show = x;
      this.draw();
    }
  }

  private qLayer: lyr.Q;
  private variationLayer: lyr.Variation;
  private nextLayer: lyr.Annotations;

  constructor(parentId: string, private gtp: Socket) {
    super(parentId, []);

    this.qLayer = new lyr.Q();
    this.variationLayer = new lyr.Variation('pv');
    this.nextLayer = new lyr.Annotations(
        'annotations', [Annotation.Shape.DashedCircle]);
    this.addLayers([
        new lyr.Label(),
        new lyr.BoardStones(),
        this.qLayer,
        this.variationLayer,
        this.nextLayer,
        new lyr.Annotations('annotations', [Annotation.Shape.Dot])]);
    this.variationLayer.show = false;
    this.enabled = true;

    this.ctx.canvas.addEventListener('mousemove', (e) => {
      if (this.showSearch) {
        let p = this.canvasToBoard(e.offsetX, e.offsetY, 0.45);
        if (p != null) {
          if (this.getStone(p) != Color.Empty || !this.qLayer.hasPoint(p)) {
            p = null;
          }
        }
        this.showVariation(p);
      }
    });

    this.ctx.canvas.addEventListener('mouseleave', () => {
      if (this.showSearch) {
        this.showVariation(null);
      }
    });

    this.onClick((p: Point) => {
      if (this.variationLayer.childVariation != null) {
        this.gtp.send('variation');
      }
      this.variationLayer.clear();
      this.variationLayer.show = false;
      this.qLayer.clear();
      this.qLayer.show = true;
    });
  }

  private showVariation(p: Nullable<Point>) {
    if (movesEqual(p, this.variationLayer.childVariation)) {
      return;
    }

    this.variationLayer.clear();
    this.variationLayer.show = p != null;
    this.qLayer.show = p == null;

    if (p != null) {
      this.gtp.send(`variation ${toKgs(p)}`);
    } else {
      this.gtp.send('variation');
    }
    this.variationLayer.childVariation = p;
  }
}

// Demo app implementation that's shared between full and lightweight demo UIs.
class ExploreApp extends App {
  private board: ExploreBoard;
  private winrateGraph = new WinrateGraph('winrate-graph');
  private variationTree = new VariationTree('tree');
  private log = new Log('log', 'console');
  private showSearch = true;
  private showNext = true;
  private showConsole = false;
  private moveElem = getElement('move');

  constructor() {
    super();
    this.connect().then(() => {
      this.board = new ExploreBoard('main-board', this.gtp);

      this.board.onClick((p: Point) => {
        this.playMove(this.activePosition.toPlay, p);
      });

      this.init([this.board]);
      this.initEventListeners();

      // Initialize log.
      this.log.onConsoleCmd((cmd: string) => {
        this.gtp.send(cmd).then(() => { this.log.scroll(); });
      });
      this.gtp.onText((line: string) => {
        this.log.log(line, 'log-cmd');
        if (this.showConsole) {
          this.log.scroll();
        }
      });

      this.newGame();

      this.variationTree.onClick((position: Position) => {
        this.selectPosition(position);
      });
    });
  }

  private initEventListeners() {
    // Global keyboard events.
    window.addEventListener('keydown', (e: KeyboardEvent) => {
      // Toggle the console.
      if (e.key == 'Escape') {
        this.showConsole = !this.showConsole;
        let containerElem = getElement('log-container');
        containerElem.style.top = this.showConsole ? '0' : '-40vh';
        if (this.showConsole) {
          this.log.focus();
          this.log.scroll();
        } else {
          this.log.blur();
        }
        e.preventDefault();
        return false;
      }

      // Don't do any special key handling if any text inputs have focus.
      for (let elem of [this.log.consoleElem, this.moveElem]) {
        if (document.activeElement == elem) {
          return;
        }
      }

      switch (e.key) {
        case 'ArrowUp':
        case 'ArrowLeft':
          this.selectPrevPosition();
          break;

        case 'ArrowRight':
        case 'ArrowDown':
          this.selectNextPosition();
          break;
      }
    });

    // Mouse wheel.
    window.addEventListener('wheel', (e: WheelEvent) => {
      if (this.showConsole) {
        return;
      }

      if (e.deltaY < 0) {
        this.selectPrevPosition();
      } else if (e.deltaY > 0) {
        this.selectNextPosition();
      }
    });

    // Toggle search display.
    let searchElem = getElement('toggle-search');
    searchElem.addEventListener('click', () => {
      this.showSearch = !this.showSearch;
      this.board.showSearch = this.showSearch;
      if (this.showSearch) {
        searchElem.innerText = 'Hide search';
      } else {
        searchElem.innerText = 'Show search';
      }
    });

    // Toggle variation display.
    let variationElem = getElement('toggle-variation');
    variationElem.addEventListener('click', () => {
      this.showNext = !this.showNext;
      this.board.showNext = this.showNext;
      if (this.showNext) {
        variationElem.innerText = 'Hide variation';
      } else {
        variationElem.innerText = 'Show variation';
      }
    });

    // Load an SGF file.
    let loadSgfElem = getElement('load-sgf-input') as HTMLInputElement;
    loadSgfElem.addEventListener('change', () => {
      let files: File[] = Array.prototype.slice.call(loadSgfElem.files);
      if (files.length != 1) {
        return;
      }
      let reader = new FileReader();
      reader.onload = () => {
        this.board.clear();
        this.newGame();
        let sgf = reader.result.replace(/\n/g, '\\n');

        this.board.enabled = false;
        this.board.showSearch = false;
        this.gtp.send('ponder 0');
        this.gtp.send(`playsgf ${sgf}`).finally(() => {
          this.board.enabled = true;
          this.board.showSearch = this.showSearch;
          this.gtp.send('ponder 1');
        });
      };
      reader.readAsText(files[0]);
    });

    // Return to main line.
    let mainLineElem = getElement('main-line');
    mainLineElem.addEventListener('click', () => {
      let position = this.activePosition;
      while (position != this.rootPosition &&
             !position.isMainline && position.parent != null) {
        position = position.parent;
      }
      this.selectPosition(position);
    });

    // Set move number.
    this.moveElem.addEventListener('keypress', (e: KeyboardEvent) => {
      // Prevent non-numeric characters being input.
      if (e.key < '0' || e.key > '9') {
        e.preventDefault();
        return false;
      }
    });
    this.moveElem.addEventListener('input', () => {
      let moveNum = parseInt(this.moveElem.innerText);
      if (isNaN(moveNum)) {
        return;
      }
      let position = this.rootPosition;
      while (position.moveNum != moveNum && position.children.length > 0) {
        position = position.children[0];
      }
      if (position.moveNum == moveNum) {
        this.selectPosition(position);
      }
    });
  }

  protected selectNextPosition() {
    if (this.activePosition.children.length > 0) {
      this.selectPosition(this.activePosition.children[0]);
    }
  }

  protected selectPrevPosition() {
    if (this.activePosition.parent != null) {
      let p = this.activePosition.parent;
      for (let i = 0; i < 10; ++i) {
        this.selectPosition(p);
      }
    }
  }

  protected selectPosition(position: Position) {
    if (position != this.activePosition) {
      this.activePosition = position;
      this.updateBoards(position);
      this.winrateGraph.setWinrate(position.moveNum, position.q);
      this.variationTree.setActive(position);
      let moveNumStr = position.moveNum.toString();
      if (this.moveElem.innerText != moveNumStr) {
        this.moveElem.innerText = moveNumStr;
        // If the user changes the current move using the scroll wheel while the
        // move element text field has focus, setting the innerText will mess up
        // the caret position. We'll just remove focus from the text field to
        // work around this. The UX is actually pretty good and this is waaay
        // easier than the "correct" solution.
        if (document.activeElement == this.moveElem) {
          this.moveElem.blur();
        }
      }
      this.gtp.sendOne(`select_position ${position.id}`).catch(() => {});
    }
  }

  protected newGame() {
    this.gtp.send('prune_nodes 1');
    super.newGame();
    this.gtp.send('prune_nodes 0');
    this.variationTree.newGame(this.rootPosition);
    this.log.clear();
    this.winrateGraph.clear();
  }

  protected onSearch(msg: SearchMsg) {
    super.onSearch(msg);
    if (msg.n != null) {
      let nSum = 0;
      for (let n of msg.n) {
        nSum += n;
      }
      getElement('reads').innerText = this.formatNumReads(nSum);
    }
  }

  protected formatNumReads(numReads: number) {
     if (numReads < 1000) {
       return numReads.toString();
     }
     numReads /= 1000;
     let places = Math.max(0, 2 - Math.floor(Math.log10(numReads)));
     return numReads.toFixed(places) + 'k';
  }

  protected onPosition(position: Position) {
    this.activePosition = position;
    this.updateBoards(position);
    this.winrateGraph.setWinrate(position.moveNum, position.q);
    if (position.parent != null) {
      this.variationTree.addChild(position.parent, position);
    }
  }

  private playMove(color: Color, move: Move) {
    let colorStr = color == Color.Black ? 'b' : 'w';
    let moveStr = toKgs(move);
    this.board.enabled = false;
    this.gtp.send(`play ${colorStr} ${moveStr}`).then(() => {
      this.gtp.send('gamestate');
    }).finally(() => {
      this.board.enabled = true;
    });
  }

  protected onGameOver() {
    this.gtp.send('final_score').then((result: string) => {
      this.log.log(toPrettyResult(result));
      this.log.scroll();
    });
  }
}

new ExploreApp();
