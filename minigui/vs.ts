// Copyright 2019 Google LLC
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

import {Annotation, Position} from './position'
import {Socket} from './gtp_socket'
import {Color, Move, N, Nullable, gtpColor, movesEqual, setBoardSize, stonesEqual, toGtp} from './base'
import {Graph} from './graph'
import * as lyr from './layer'
import {Log} from './log'
import {Board} from './board'
import * as util from './util'

class Player {
  gtp = new Socket();
  board: Board;
  log: Log;
  activePosition: Nullable<Position>;
  latestPosition: Nullable<Position>;
  numWins = 0;

  nameElem: HTMLElement;
  scoreElem: HTMLElement;
  capturesElem: HTMLElement;
  timeElapsedElem: HTMLElement;
  timeTotalElem: HTMLElement;

  private genmoveTimerId = 0;
  private genmoveStartTimeSecs = 0;
  private genmoveTotalTimeSecs = 0;

  private searchLayer: lyr.Layer;
  private pvLayer: lyr.Layer;
  private dummyPosition: Position;

  private bookmarks: Element[] = [];

  constructor(public color: Color, public name: string) {
    this.dummyPosition = new Position({
      id: `${name}-dummy-position`,
      moveNum: 0,
      toPlay: 'b',
    });
    let gtpCol = gtpColor(color);
    this.log = new Log(`${gtpCol}-log`, `${gtpCol}-console`);

    let elem = util.getElement(`${gtpCol}-board`);
    this.searchLayer = new lyr.Search();
    this.pvLayer = new lyr.Variation('pv');
    this.board = new Board(elem, this.dummyPosition, [
      new lyr.Label(),
      new lyr.BoardStones(),
      this.searchLayer,
      this.pvLayer,
      new lyr.Annotations()]);
    this.pvLayer.show = false;

    this.log.onConsoleCmd((cmd: string) => {
      this.gtp.send(cmd).then(() => { this.log.scroll(); });
    });
    this.gtp.onText((line: string) => {
      this.log.log(line);
      if (this.activePosition == this.latestPosition) {
        this.log.scroll();
      }
    });

    this.newGame()
  }

  newGame() {
    this.log.clear();
    this.bookmarks = [];
    this.activePosition = null;
    this.latestPosition = null;
    this.board.position = this.dummyPosition;
    this.genmoveTotalTimeSecs = 0;

    // Look up all HTML elements each game because the players swap sides.
    let gtpCol = gtpColor(this.color);
    this.nameElem = util.getElement(`${gtpCol}-name`);
    this.scoreElem = util.getElement(`${gtpCol}-score`);
    this.capturesElem = util.getElement(`${gtpCol}-captures`);
    this.timeElapsedElem = util.getElement(`${gtpCol}-time-elapsed`);
    this.timeTotalElem = util.getElement(`${gtpCol}-time-total`);
  }

  addPosition(position: Position) {
    if (this.latestPosition == null) {
      this.activePosition = position;
    } else {
      this.latestPosition.children.push(position);
      position.parent = this.latestPosition;
    }
    this.latestPosition = position;
  }

  updatePosition(update: Position.Update) {
    if (this.latestPosition == null) {
      return;
    }
    if (update.id != this.latestPosition.id) {
      console.log(`update ${update.id} doesn't match latest position ${this.latestPosition.id}`);
      return;
    }
    this.latestPosition.update(update);
    if (this.activePosition == this.latestPosition) {
      this.board.update(update);
    }
  }

  selectMove(moveNum: number) {
    if (this.activePosition == null) {
      return;
    }
    let position = this.activePosition;
    while (position.moveNum < moveNum && position.children.length > 0) {
      position = position.children[0];
    }
    while (position.moveNum > moveNum && position.parent != null) {
      position = position.parent;
    }

    this.activePosition = position;
    this.board.setPosition(position);
    this.pvLayer.show = this.color != position.toPlay;
    this.searchLayer.show = !this.pvLayer.show;
  }

  startGenmoveTimer() {
    this.genmoveStartTimeSecs = Date.now() / 1000;
    this.genmoveTimerId = window.setInterval(() => {
      this.updateTimerDisplay(
        Date.now() / 1000 - this.genmoveStartTimeSecs,
        this.genmoveTotalTimeSecs);
    }, 1000);
  }

  stopGenmoveTimer() {
    this.genmoveTotalTimeSecs += Date.now() / 1000 - this.genmoveStartTimeSecs;
    window.clearInterval(this.genmoveTimerId);
    this.genmoveTimerId = 0;
    this.updateTimerDisplay(0, this.genmoveTotalTimeSecs);
  }

  setBookmark(moveNum: number) {
    if (this.log.logElem.lastElementChild != null) {
      this.bookmarks[moveNum] = this.log.logElem.lastElementChild;
    }
  }

  scrollToBookmark(moveNum: number) {
    if (this.bookmarks[moveNum] != null) {
      this.bookmarks[moveNum].scrollIntoView();
    }
  }

  private updateTimerDisplay(elapsed: number, total: number) {
    this.timeElapsedElem.innerText = this.formatDuration(elapsed);
    this.timeTotalElem.innerText = this.formatDuration(total);
  }

  private formatDuration(durationSecs: number) {
    durationSecs = Math.floor(durationSecs);
    let s = durationSecs % 60;
    let m = Math.floor((durationSecs / 60)) % 60;
    let h = Math.floor(durationSecs / 3600);
    let ss = s < 10 ? '0' + s : s.toString();
    let mm = m < 10 ? '0' + m : m.toString();
    let hh = h < 10 ? '0' + h : m.toString();
    return `${hh}:${mm}:${ss}`;
  }
}

class ReadsGraph extends Graph {
  private plots: number[][] = [[], []];

  constructor(elem: string) {
    super(elem, {
      xStart: 0,
      xEnd: 10,
      yStart: 10,
      yEnd: 0,
      xTicks: true,
      yTicks: true,
      marginTop: 0.05,
      marginBottom: 0.05,
      marginLeft: 0.08,
      marginRight: 0.05,
    });
    this.draw();
  }

  newGame() {
    for (let plot of this.plots) {
      plot.length = 0;
    }
    this.xEnd = 10;
    this.yStart = 10;
    this.draw();
  }

  drawImpl() {
    this.xEnd = Math.max(this.xEnd, this.moveNum);

    super.drawImpl();

    // Draw a dotted line for the current move number.
    this.drawPlot(
      [[this.moveNum, this.yStart], [this.moveNum, this.yEnd]], {
      dash: [0, 3],
      width: 1,
      style: '#96928f',
      snap: true,
    });

    for (let i = 0; i < this.plots.length; ++i) {
      let plot = this.plots[i];
      let points: number[][] = [];
      for (let x = 0; x < plot.length; ++x) {
        if (plot[x] != null) {
          points.push([x, plot[x]]);
        }
      }
      this.drawPlot(points, {
        width: 3,
        style: i == 0 ? '#fff' : '#000'
      });
    }

    let pr = util.pixelRatio();
    let ctx = this.ctx;

    let textHeight = 0.06 * ctx.canvas.height;
    ctx.font = `${textHeight}px sans-serif`;
    ctx.fillStyle = '#96928f';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    let y = this.yTickPoints[this.yTickPoints.length - 1];
    ctx.fillText(y.toString(), -6 * pr, this.yScale * (y - this.yStart));
  }

  update(position: Position) {
    if (position.treeStats.maxDepth > 0) {
      let plot = position.toPlay == Color.Black ? this.plots[1] : this.plots[0];
      plot[position.moveNum] = position.treeStats.maxDepth;
      this.yStart = Math.max(this.yStart, position.treeStats.maxDepth);
    }
    this.xEnd = Math.max(this.xEnd, position.moveNum);
    this.draw();
  }
}

class WinrateGraph extends Graph {
  private plots: number[][] = [[], []];

  constructor(elem: string) {
    super(elem, {xStart: -1, xEnd: 1, yStart: 0, yEnd: 10, yTicks: true});
  }

  newGame() {
    for (let plot of this.plots) {
      plot.length = 0;
    }
    this.yEnd = 10;
    this.moveNum = 0;
    this.draw();
  }

  drawImpl() {
    this.yEnd = Math.max(this.yEnd, this.moveNum);

    super.drawImpl();

    let pr = util.pixelRatio();
    let ctx = this.ctx;

    // Draw a dotted line for the current move number.
    this.drawPlot(
      [[this.xStart, this.moveNum], [this.xEnd, this.moveNum]], {
      dash: [0, 3],
      width: 1,
      style: '#96928f',
      snap: true,
    });

    // Draw the plots.
    for (let i = 0; i < this.plots.length; ++i) {
      let plot = this.plots[i];
      let points: number[][] = [];
      for (let y = 0; y < plot.length; ++y) {
        if (plot[y] != null) {
          points.push([-plot[y], y]);
        }
      }
      this.drawPlot(points, {
        width: 3,
        style: i == 0 ? '#fff' : '#000',
      });
    }
  }

  update(position: Position) {
    if (position.n > 0) {
      // Q is only valid if we've performed at least one read.
      let plot = position.toPlay == Color.Black ? this.plots[1] : this.plots[0];
      if (position.q != null) {
        plot[position.moveNum] = position.q;
      }
    }
    this.yEnd = Math.max(this.yEnd, position.moveNum);
    this.draw();
  }
}

// App base class used by the different Minigui UI implementations.
class VsApp {
  private players: Player[] = [];
  private black: Player;  // = players[0]
  private white: Player;  // = players[1]
  private currPlayer: Player;
  private nextPlayer: Player;
  private winrateGraph = new WinrateGraph('winrate-graph');
  private readsGraph = new ReadsGraph('reads-graph');
  private activeMoveNum = 0;
  private maxMoveNum = 0;
  private paused = false;
  private engineThinking = false;
  private gameOver = false;
  private startNextGameTimerId: Nullable<number> = null;

  constructor() {
    this.connect().then(() => {
      this.initEventListeners();
      this.newGame();
    });
  }

  private initEventListeners() {
    for (let player of this.players) {
      player.gtp.onData('mg-update', (j: Position.Update) => {
        player.updatePosition(j);
        if (player.latestPosition != null) {
          this.winrateGraph.update(player.latestPosition);
          this.readsGraph.update(player.latestPosition);
        }
      });

      player.gtp.onData('mg-position', (j: Position.Definition | Position.Update) => {
        let position = new Position(j as Position.Definition);
        position.update(j);

        this.maxMoveNum = Math.max(this.maxMoveNum, position.moveNum);

        if (player.latestPosition != null && position.lastMove != null) {
          // Copy the principal variation from the parent node.
          let gtpMove = toGtp(position.lastMove);
          let variation = player.latestPosition.variations.get(gtpMove);
          if (variation != null && !position.variations.has(gtpMove)) {
            variation = {
              n: variation.n,
              q: variation.q,
              moves: variation.moves.slice(1),
            };
            position.variations.set(gtpMove, variation);
            position.variations.set('pv', variation);
          }
        }

        player.addPosition(position);
        this.winrateGraph.update(position);
        this.readsGraph.update(position);
        if (player.board.position.moveNum + 1 == position.moveNum) {
          this.selectMove(position.moveNum);
        }
      });
    }

    window.addEventListener('keydown', (e: KeyboardEvent) => {
      if (this.isInLog(document.activeElement)) {
        return;
      }

      let handled = true;
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
        default:
          handled = false;
          break;
      }
      if (handled) {
        e.preventDefault();
        return false;
      }
    });

    window.addEventListener('wheel', (e: WheelEvent) => {
      if (this.isInLog(e.target as Element)) {
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

    let pauseElem = util.getElement('pause');
    pauseElem.addEventListener('click', () => {
      this.paused = !this.paused;
      pauseElem.innerText = this.paused ? 'Resume' : 'Pause';
      if (!this.paused && !this.engineThinking) {
        if (this.gameOver) {
          if (this.startNextGameTimerId != null) {
            window.clearTimeout(this.startNextGameTimerId);
            this.startNextGameTimerId = null;
          }
          this.startNextGame();
        } else {
          this.genmove();
        }
      }
    });
  }

  private isInLog(elem: Nullable<Element>) {
    while (elem != null) {
      if (elem.classList.contains('log-container')) {
        return true;
      }
      elem = elem.parentElement;
    }
    return false;
  }

  private goBack(n: number) {
    this.selectMove(this.activeMoveNum - n);
  }

  private goForward(n: number) {
    this.selectMove(this.activeMoveNum + n);
  }

  private selectMove(moveNum: number) {
    this.activeMoveNum = Math.max(0, Math.min(moveNum, this.maxMoveNum));
    for (let player of this.players) {
      player.selectMove(this.activeMoveNum);
      if (player.color == player.board.position.toPlay) {
        player.nameElem.classList.add('underline');
      } else {
        player.nameElem.classList.remove('underline');
      }
      player.scrollToBookmark(moveNum);
    }
    this.winrateGraph.setMoveNum(this.activeMoveNum);
    this.readsGraph.setMoveNum(this.activeMoveNum);
    this.updatePositionInfo();
  }

  private onGameOver() {
    this.gameOver = true;
    this.currPlayer.gtp.send('final_score').then((score: string) => {
      if (score[0] == 'B') {
        this.black.numWins += 1;
      } else {
        this.white.numWins += 1;
      }
      this.updatePlayerDisplay();
    }).finally(() => {
      // Wait a few seconds, then start the next game.
      this.startNextGameTimerId = window.setTimeout(() => {
        if (!this.paused) {
          this.startNextGame();
        }
      }, 5000);
    });
  }

  private startNextGame() {
    // Swap which engine is black and which is white, making sure the black
    // board is always on the left.
    for (let prop of ['color', 'board', 'log']) {
      let b = this.black as any;
      let w = this.white as any;
      [b[prop], w[prop]] = [w[prop], b[prop]];
    }
    [this.black, this.white] = [this.white, this.black];
    [this.currPlayer, this.nextPlayer] = [this.black, this.white];
    this.newGame();
    this.updatePlayerDisplay();
  }

  private updatePlayerDisplay() {
    for (let player of this.players) {
      player.nameElem.innerText = player.name;
      player.scoreElem.innerText = player.numWins.toString();
    }
  }

  private updatePositionInfo() {
    let position = this.currPlayer.activePosition;
    if (position != null) {
      this.black.capturesElem.innerText = position.captures[0].toString();
      this.white.capturesElem.innerText = position.captures[1].toString();
      util.getElement('move-num').innerText = `move: ${this.activeMoveNum}`;
    }
  }

  private genmove() {
    let gtpCol = gtpColor(this.currPlayer.color);
    this.engineThinking = true;

    let currPosition = this.currPlayer.latestPosition;
    if (currPosition != null) {
      this.currPlayer.setBookmark(currPosition.moveNum);
    }
    this.currPlayer.startGenmoveTimer();
    this.currPlayer.gtp.send(`genmove ${gtpCol}`).then((gtpMove: string) => {
      if (gtpMove == 'resign') {
        this.onGameOver();
      }
      if (this.currPlayer.latestPosition != null &&
          this.currPlayer.latestPosition.gameOver) {
        this.onGameOver();
      } else {
        if (currPosition != null) {
          this.nextPlayer.setBookmark(currPosition.moveNum);
        }
        this.nextPlayer.gtp.send(`play ${gtpCol} ${gtpMove}`).then(() => {
          [this.currPlayer, this.nextPlayer] = [this.nextPlayer, this.currPlayer];
          if (!this.paused && !this.gameOver) {
            this.genmove();
          }
        });
      }
    }).finally(() => {
      this.engineThinking = false;
      this.currPlayer.stopGenmoveTimer();
    });
  }

  protected connect() {
    let uri = `http://${document.domain}:${location.port}/minigui`;
    let params = new URLSearchParams(window.location.search);
    let p = params.get('gtp_debug');
    let debug = (p != null) && (p == '' || p == '1' || p.toLowerCase() == 'true');
    return fetch('config').then((response) => {
      return response.json();
    }).then((cfg: any) => {
      // TODO(tommadams): Give cfg a real type.

      // setBoardSize sets the global variable N to the board size for the game
      // (as provided by the backend engine). The code uses N from hereon in.
      setBoardSize(cfg.boardSize);

      if (cfg.players.length != 2) {
        throw new Error(`expected 2 players, got ${cfg.players}`);
      }
      let promises = [];
      for (let i = 0; i < cfg.players.length; ++i) {
        let color = [Color.Black, Color.White][i];
        let name = cfg.players[i];
        let player = new Player(color, name);
        this.players.push(player);
        promises.push(player.gtp.connect(uri, name, debug));
      }
      [this.black, this.white] = this.players;
      [this.currPlayer, this.nextPlayer] = this.players;
      return Promise.all(promises);
    });
  }

  protected newGame() {
    this.gameOver = false;
    this.activeMoveNum = 0;
    this.maxMoveNum = 0;
    this.winrateGraph.newGame();
    this.readsGraph.newGame();
    this.updatePlayerDisplay();
    let promises: Promise<any>[] = [];
    for (let player of this.players) {
      player.newGame();
      promises.push(player.gtp.send('clear_board'));
    }
    Promise.all(promises).then(() => {
      this.selectMove(0);
      this.genmove();
    });
  }
}

(window as any)['app'] = new VsApp();
