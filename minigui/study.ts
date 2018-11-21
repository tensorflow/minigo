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
import {COL_LABELS, Color, Move, N, Nullable, Point, movesEqual, otherColor, toKgs} from './base'
import {Board, ClickableBoard} from './board'
import {Socket} from './gtp_socket'
import * as lyr from './layer'
import {Log} from './log'
import {Annotation, Position} from './position'
import {getElement, parseMove, pixelRatio, toPrettyResult} from './util'
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
        this.searchLayer.show = true;
      } else {
        this.variationLayer.show = false;
        this.searchLayer.show = false;
      }
      this.draw();
    }
  }

  private _highlightedNextMove: Nullable<Move> = null;
  get highlightedNextMove() {
    return this._highlightedNextMove;
  }
  set highlightedNextMove(x: Nullable<Move>) {
    if (x != this._highlightedNextMove) {
      this._highlightedNextMove = x;
      this.draw();
    }
  }

  get variation() {
    return this.variationLayer.variation;
  }

  private searchLayer: lyr.Search;
  private variationLayer: lyr.Variation;
  private nextLayer: lyr.Annotations;

  constructor(parentElemId: string, position: Position, private gtp: Socket) {
    super(parentElemId, position, []);

    this.searchLayer = new lyr.Search();
    this.variationLayer = new lyr.Variation('pv');
    this.addLayers([
        new lyr.Label(),
        new lyr.BoardStones(),
        this.searchLayer,
        this.variationLayer,
        new lyr.Annotations()]);
    this.variationLayer.show = false;
    this.enabled = true;

    this.ctx.canvas.addEventListener('mousemove', (e) => {
      if (this.showSearch) {
        let p = this.canvasToBoard(e.offsetX, e.offsetY, 0.45);
        if (p != null) {
          if (this.getStone(p) != Color.Empty || !this.searchLayer.hasVariation(p)) {
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
      if (this.variationLayer.showVariation != 'pv') {
        this.gtp.send('variation');
      }
      this.variationLayer.showVariation = 'pv';
      this.variationLayer.clear();
      this.variationLayer.show = false;
      this.searchLayer.clear();
      this.searchLayer.show = true;
    });
  }

  setPosition(position: Position) {
    if (position != this.position) {
      this.showVariation(null);
      super.setPosition(position);
    }
  }

  drawImpl() {
    super.drawImpl();
    if (this.showSearch) {
      this.drawNextMoves();
    }
  }

  private drawNextMoves() {
    let sr = this.stoneRadius;
    let pr = pixelRatio();

    // Calculate a dash pattern that's close to [4, 5] (a four pixel
    // dash followed by a five pixel space) but also whose length
    // divides the circle's circumference exactly. This avoids the final
    // dash or space on the arc being a different size than all the rest.
    // I wish things like this didn't bother me as much as they do.
    let circum = 2 * Math.PI * sr;
    let numDashes = 9 * Math.round(circum / 9);
    let dashLen = 4 * circum / numDashes;
    let spaceLen = 5 * circum / numDashes;

    let colors: string[];
    if (this.position.toPlay == Color.Black) {
      colors = ['#000', '#fff'];
    } else {
      colors = ['#fff', '#000'];
    }

    let ctx = this.ctx;
    let lineDash = [dashLen, spaceLen];
    ctx.lineCap = 'round';
    ctx.setLineDash(lineDash);
    for (let pass = 0; pass < 2; ++pass) {
      ctx.strokeStyle = colors[pass];
      ctx.lineWidth = (3 - pass * 2) * pr;
      for (let child of this.position.children) {
        let move = child.lastMove;
        if (move == null || move == 'pass' || move == 'resign') {
          continue;
        }

        if (child.lastMove == this.highlightedNextMove) {
          ctx.setLineDash([]);
        }
        let c = this.boardToCanvas(move.row, move.col);
        ctx.beginPath();
        ctx.moveTo(c.x + 0.5 + sr, c.y + 0.5);
        ctx.arc(c.x + 0.5, c.y + 0.5, sr, 0, 2 * Math.PI);
        ctx.stroke();
        if (child.lastMove == this.highlightedNextMove) {
          ctx.setLineDash(lineDash);
        }
      }
    }
    ctx.setLineDash([]);
  }

  private showVariation(p: Nullable<Point>) {
    let moveStr: string;
    if (p == null) {
      moveStr = 'pv';
    } else {
      moveStr = toKgs(p);
    }
    if (moveStr == this.variationLayer.showVariation) {
      return;
    }

    this.variationLayer.showVariation = moveStr;
    this.variationLayer.clear();
    this.variationLayer.show = p != null;
    this.searchLayer.show = p == null;

    if (p != null) {
      this.gtp.send(`variation ${moveStr}`);
    } else {
      this.gtp.send('variation');
    }
  }
}

// Demo app implementation that's shared between full and lightweight demo UIs.
class ExploreApp extends App {
  private board: ExploreBoard;
  private winrateGraph = new WinrateGraph('winrate-graph');
  private variationTree = new VariationTree('tree');
  private log = new Log('log', 'console');
  private showSearch = true;
  private showConsole = false;
  private moveElem = getElement('move');
  private commentElem = getElement('comment');

  constructor() {
    super();
    this.connect().then(() => {
      this.board = new ExploreBoard('main-board', this.rootPosition, this.gtp);

      this.board.onClick((p: Point) => {
        this.playMove(this.activePosition.toPlay, p);
      });

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
        if (position != this.activePosition) {
          this.selectPosition(position);
        }
      });
      this.variationTree.onHover((position: Nullable<Position>) => {
        if (position != null) {
          this.board.highlightedNextMove = position.lastMove;
        } else {
          this.board.highlightedNextMove = null;
        }
      });

      // Repeatedly ponder for a few seconds at a time.
      // We do this instead of pondering permanently because if simply
      // enabled pondering and let it run, then if the user closed their
      // Minigui tab without killing the backend, we'd permanently peg their
      // accelerator.
      this.gtp.onData('mg-ponder', (result: string) => {
        // Pondering reports 'done' when it has finised the requested pondering
        // and reports 'failed' when it couldn't perform any reads (which is
        // possible in some edge cases where the board is almost full).
        if (result.trim().toLowerCase() == 'done') {
          this.gtp.send('ponder time 10');
        }
      });
      this.gtp.send('ponder time 10');
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

      // If the user is hovering over a searched node and displaying a
      // variation, pressing a number will play the move out the corresponding
      // position in the variation.
      if (e.key >= '0' && e.key <= '9' && this.board.variation != null) {
        let move = e.key.charCodeAt(0) - '0'.charCodeAt(0);
        if (move == 0) {
          move = 10;
        }
        if (move <= this.board.variation.length) {
          let color = this.board.position.toPlay;
          for (let i = 0; i < move; ++i) {
            this.playMove(color, this.board.variation[i]);
            color = otherColor(color);
          }
        }
      }

      switch (e.key) {
        case 'ArrowUp':
        case 'ArrowLeft':
          this.goBack(1);
          break;
        case 'ArrowRight':
        case 'ArrowDown':
          this.goForward(1);
          break;
        case 'PageUp':
          this.goBack(10);
          break;
        case 'PageDown':
          this.goForward(10);
          break;
        case 'Home':
          this.goBack(Infinity);
          break;
        case 'End':
          this.goForward(Infinity);
          break;
      }
    });

    // Mouse wheel.
    window.addEventListener('wheel', (e: WheelEvent) => {
      if (this.showConsole || e.target == this.commentElem) {
        return;
      }
      if (e.deltaY < 0) {
        this.goBack(1);
      } else if (e.deltaY > 0) {
        this.goForward(1);
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

    // Clear the board and start a new game.
    let clearElem = getElement('clear-board');
    clearElem.addEventListener('click', () => { this.newGame(); });

    // Load an SGF file.
    let loadSgfElem = getElement('load-sgf-input') as HTMLInputElement;
    loadSgfElem.addEventListener('change', () => {
      let files: File[] = Array.prototype.slice.call(loadSgfElem.files);
      if (files.length != 1) {
        return;
      }
      let reader = new FileReader();
      reader.onload = () => {
        this.newGame();
        let sgf = reader.result.replace(/\n/g, '\\n');

        this.board.enabled = false;
        this.board.showSearch = false;
        this.gtp.send(`playsgf ${sgf}`).catch((error) => {
          window.alert(error);
        }).finally(() => {
          this.board.enabled = true;
          this.board.showSearch = this.showSearch;
        });
      };
      reader.readAsText(files[0]);
      // Clear the input's value. If we don't do this then reloading the
      // currently loaded SGF won't work because we listen to the 'change' even
      // and the value doesn't change when choosing the previously loaded file.
      loadSgfElem.value = "";
    });

    // Return to main line.
    let mainLineElem = getElement('main-line');
    mainLineElem.addEventListener('click', () => {
      let position = this.activePosition;
      while (position != this.rootPosition &&
             !position.isMainLine && position.parent != null) {
        position = position.parent;
      }
      if (position != this.activePosition) {
        this.selectPosition(position);
      }
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
        if (position != this.activePosition) {
          this.selectPosition(position);
        }
      }
    });
  }

  protected goBack(n: number) {
    let position = this.activePosition;
    for (let i = 0; i < n && position.parent != null; ++i) {
      position = position.parent;
    }
    this.selectPosition(position);
  }

  protected goForward(n: number) {
    let position = this.activePosition;
    for (let i = 0; i < n && position.children.length > 0; ++i) {
      position = position.children[0];
    }
    this.selectPosition(position);
  }

  protected selectPosition(position: Position) {
    this.activePosition = position;
    this.board.setPosition(position);
    this.winrateGraph.setActive(position);
    this.variationTree.setActive(position);
    this.commentElem.innerText = position.comment;
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

  protected newGame() {
    super.newGame();
    this.variationTree.newGame(this.rootPosition);
    this.winrateGraph.newGame(this.rootPosition);
    this.board.newGame(this.rootPosition);
    this.log.clear();
  }

  protected onPositionUpdate(position: Position, update: Position.Update) {
    this.winrateGraph.update(position);
    if (position != this.activePosition) {
      return;
    }
    this.board.update(update);
    getElement('reads').innerText = this.formatNumReads(position.n);
  }

  protected formatNumReads(numReads: number) {
     if (numReads < 1000) {
       return numReads.toString();
     }
     numReads /= 1000;
     let places = Math.max(0, 2 - Math.floor(Math.log10(numReads)));
     return numReads.toFixed(places) + 'k';
  }

  protected onNewPosition(position: Position) {
    if (position.parent != null) {
      this.variationTree.addChild(position.parent, position);
    }
    this.selectPosition(position);
  }

  private playMove(color: Color, move: Move) {
    let colorStr = color == Color.Black ? 'b' : 'w';
    let moveStr = toKgs(move);
    this.board.enabled = false;
    this.gtp.send(`play ${colorStr} ${moveStr}`).finally(() => {
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
