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
import {COL_LABELS, Color, Move, N, Nullable, Point, movesEqual, otherColor, toGtp} from './base'
import {Board, ClickableBoard} from './board'
import {Socket} from './gtp_socket'
import * as lyr from './layer'
import {Log} from './log'
import {Annotation, Position} from './position'
import {getElement, parseMove, pixelRatio, toPrettyResult} from './util'
import {VariationTree} from './variation_tree'
import {WinrateGraph} from './winrate_graph'

class ExploreBoard extends ClickableBoard {
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
    return this.searchLyr.variation.length > 0 ? this.searchLyr.variation : null;
  }

  get showSearch() {
    return this.searchLyr.show;
  }
  set showSearch(x: boolean) {
    this.searchLyr.show = x;
  }

  private searchLyr: lyr.Search;

  constructor(parentElemId: string, position: Position, private gtp: Socket) {
    super(parentElemId, position, []);

    this.searchLyr = new lyr.Search();
    this.addLayers([
        new lyr.Label(),
        new lyr.BoardStones(),
        this.searchLyr,
        new lyr.Annotations()]);
    this.enabled = true;
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
  private searchElem = getElement('toggle-search');
  private blackCapturesElem = getElement('b-caps');
  private whiteCapturesElem = getElement('w-caps');
  private readsElem = getElement('reads');

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
        let moveNum = e.key.charCodeAt(0) - '0'.charCodeAt(0);
        if (moveNum == 0) {
          moveNum = 10;
        }
        if (moveNum <= this.board.variation.length) {
          let color = this.board.position.toPlay;
          for (let i = 0; i < moveNum; ++i) {
            this.playMove(color, this.board.variation[i]);
            color = otherColor(color);
          }
        }
      }

      switch (e.key) {
        case ' ':
          this.toggleSearch();
          break;
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

      let delta: number;
      if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
        delta = e.deltaX;
      } else {
        delta = e.deltaY;
      }

      if (delta < 0) {
        this.goBack(1);
      } else if (delta > 0) {
        this.goForward(1);
      }
    });

    // Toggle search display.
    this.searchElem.addEventListener('click', () => { this.toggleSearch(); });

    // Clear the board and start a new game.
    let clearElem = getElement('clear-board');
    clearElem.addEventListener('click', () => { this.newGame(); });

    // Load an SGF file. There's no way to get the selected file's path, so
    // we POST the file contents to the server, which writes the file to a
    // temporary location and responds with that path. We can then issue a
    // loadsgf command to load the file from the temporary location.
    let loadSgfElem = getElement('load-sgf-input') as HTMLInputElement;
    loadSgfElem.addEventListener('change', () => {
      let files: File[] = Array.prototype.slice.call(loadSgfElem.files);
      if (files.length != 1) {
        return;
      }
      let reader = new FileReader();
      reader.onload = () => {
        this.newGame();
        this.board.enabled = false;
        this.board.showSearch = false;
        this.uploadTmpFile(reader.result as string).then((path) => {
          return this.gtp.send(`loadsgf ${path}`);
        }).catch((error) => {
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

    // Count score.
    let countScoreElem = getElement('count-score');
    countScoreElem.addEventListener('click', () => {
      this.gtp.send(`final_score`).then((result: string) => {
        window.alert(result);
      });
    });

    // Set move number.
    this.moveElem.addEventListener('keypress', (e: KeyboardEvent) => {
      // Prevent non-numeric characters being input.
      if (e.key < '0' || e.key > '9') {
        e.preventDefault();
        return false;
      }
    });
    this.moveElem.addEventListener('blur', () => {
      this.moveElem.innerText = this.activePosition.moveNum.toString();
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
    this.blackCapturesElem.innerText = this.activePosition.captures[0].toString();
    this.whiteCapturesElem.innerText = this.activePosition.captures[1].toString();
    this.readsElem.innerText = this.formatNumReads(position.n);
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
    this.log.clear();
    this.winrateGraph.newGame();
    this.variationTree.newGame();
    return super.newGame().then(() => {
      this.board.newGame(this.rootPosition);
    });
  }

  protected onPositionUpdate(position: Position, update: Position.Update) {
    this.winrateGraph.update(position);
    if (position != this.activePosition) {
      return;
    }
    this.board.update(update);
    this.readsElem.innerText = this.formatNumReads(position.n);
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
    if (position.parent == null) {
      this.variationTree.setRoot(position);
    } else {
      this.variationTree.addChild(position.parent, position);
    }
    this.selectPosition(position);
  }

  private playMove(color: Color, move: Move) {
    let colorStr = color == Color.Black ? 'b' : 'w';
    let moveStr = toGtp(move);
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

  private toggleSearch() {
    this.showSearch = !this.showSearch;
    this.board.showSearch = this.showSearch;
    if (this.showSearch) {
      this.searchElem.innerText = 'Hide search';
    } else {
      this.searchElem.innerText = 'Show search';
    }
  }

  private uploadTmpFile(contents: string) {
    return fetch('write_tmp_file', {
      method: 'POST',
      headers: {'Content-Type': 'text/plain'},
      body: contents,
    }).then((response) => {
      return response.text();
    });
  }
}

new ExploreApp();
