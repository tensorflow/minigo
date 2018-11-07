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
      let p = this.canvasToBoard(e.offsetX, e.offsetY, 0.45);
      if (p != null) {
        if (this.getStone(p) != Color.Empty || !this.qLayer.hasPoint(p)) {
          p = null;
        }
      }
      this.showVariation(p);
    });

    this.ctx.canvas.addEventListener('mouseleave', () => {
      this.showVariation(null);
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

  private pendingSelectPosition: Nullable<Position> = null;

  constructor() {
    super();
    this.connect().then(() => {
      this.board = new ExploreBoard('main-board', this.gtp);

      this.board.onClick((p: Point) => {
        this.playMove(this.activePosition.toPlay, p);
      });

      this.init([this.board]);
      this.initButtons();

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

      window.addEventListener('keydown', (e: KeyboardEvent) => {
        // Toggle the console.
        if (e.key == 'Escape') {
          this.showConsole = !this.showConsole;
          let containerElem = getElement('log-container');
          containerElem.style.top = this.showConsole ? '0' : '-40vh';
          if (this.showConsole) {
            this.log.scroll();
          } else {
            this.log.blur();
          }
          e.preventDefault();
          return false;
        }

        // Don't do any special key handling if the console has focus.
        if (this.log.hasFocus) {
          return;
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

      window.addEventListener('wheel', (e: WheelEvent) => {
        if (e.deltaY < 0) {
          this.selectPrevPosition();
        } else if (e.deltaY > 0) {
          this.selectNextPosition();
        }
      });
    });
  }

  private initButtons() {
    getElement('toggle-search').addEventListener('click', (e: any) => {
      this.showSearch = !this.showSearch;
      this.board.showSearch = this.showSearch;
      if (this.showSearch) {
        e.target.innerText = 'Hide search';
      } else {
        e.target.innerText = 'Show search';
      }
    });

    getElement('toggle-variation').addEventListener('click', (e: any) => {
      this.showNext = !this.showNext;
      this.board.showNext = this.showNext;
      if (this.showNext) {
        e.target.innerText = 'Hide variation';
      } else {
        e.target.innerText = 'Show variation';
      }
    });

    getElement('load-sgf-input').addEventListener('change', (e: any) => {
      let files: File[] = Array.prototype.slice.call(e.target.files);
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

  protected selectNextPosition() {
    if (this.activePosition.children.length > 0) {
      this.selectPosition(this.activePosition.children[0]);
    }
  }

  protected selectPrevPosition() {
    if (this.activePosition.parent != null) {
      this.selectPosition(this.activePosition.parent);
    }
  }

  protected selectPosition(position: Position) {
    this.activePosition = position;
    this.updateBoards(position);
    this.winrateGraph.setWinrate(position.moveNum, position.q);
    if (position.parent != null) {
      this.variationTree.addChild(position.parent, position);
    }

    let impl = (position: Position) => {
      if (this.pendingSelectPosition == null) {
        this.gtp.send(`select_position ${position.id}`).then((id: string) => {
          if (this.pendingSelectPosition == null) {
            throw new Error('pending select position is null');
          }
          if (id != this.pendingSelectPosition.id) {
            impl(this.pendingSelectPosition);
          } else {
            this.pendingSelectPosition = null;
          }
        });
      }
      this.pendingSelectPosition = position;
    };
    impl(position);
  }

  protected newGame() {
    this.gtp.send('prune_nodes 1');
    super.newGame();
    this.gtp.send('prune_nodes 0');
    this.variationTree.newGame(this.rootPosition);
    this.log.clear();
    this.winrateGraph.clear();
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
