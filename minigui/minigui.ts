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

import {BoardSize, Color, Move, Nullable} from './base'
import {Annotation, Board, ClickableBoard, COL_LABELS} from './board'
import * as util from './util'
import * as lyr from './layer'
import {heatMapN, heatMapDq} from './heat_map'
import {Log} from './log'
import * as gtpsock from './gtp_socket'
import {Graph} from './graph'

class BoardState {
  search: Move[] = [];
  pv: Move[] = [];
  n: Nullable<number[]> = null;
  dq: Nullable<number[]> = null;
  annotations: Annotation[] = [];

  constructor(public stones: Color[],
              public lastMove: Nullable<Move>,
              public toPlay: Color) {
    if (lastMove != null && lastMove != 'pass' && lastMove != 'resign') {
      this.annotations.push({
        p: lastMove,
        shape: Annotation.Shape.Dot,
        color: '#ef6c02',
      });
    }
  }
}

export interface SearchMsg {
  move: number;
  toPlay: string;
  search: string[];
  n: number[];
  dq: number[];
  pv?: string[];
}

export interface GameStateMsg {
  board: string | Color[];
  toPlay: string;
  lastMove?: string;
  n: number;
  q: number;
}

class App {
  private drawPending = false;

  private mainBoard: ClickableBoard;
  private searchBoard: Board;
  private nBoard: Board;
  private dqBoard: Board;
  private boards: Board[] = [];
  private history: BoardState[];
  private activeMove = 0;
  private numConsecutivePasses = 0;
  private gameOver = false;
  private minigoBusy = false;
  private playerElems: HTMLElement[] = [];
  private toPlay = Color.Black;

  constructor(private size: number) {
    this.history = [
      new BoardState(util.emptyBoard(this.size), null, Color.Black)
    ];

    this.mainBoard = new ClickableBoard(
      'main-board', this.size,
      [lyr.Label, lyr.BoardStones, [lyr.Variation, 'pv'], lyr.Annotations],
      {margin: 30});

    this.searchBoard = new Board(
      'search-board', this.size,
      [[lyr.Caption, 'search'], lyr.BoardStones, [lyr.Variation, 'search']]);

    this.nBoard = new Board(
      'n-board', this.size,
      [[lyr.Caption, 'N'], [lyr.HeatMap, 'n', heatMapN], lyr.BoardStones]);

    this.dqBoard = new Board(
      'dq-board', this.size,
      [[lyr.Caption, 'ΔQ'], [lyr.HeatMap, 'dq', heatMapDq], lyr.BoardStones]);

    this.boards = [
      this.mainBoard, this.searchBoard, this.nBoard, this.dqBoard
    ];

    this.mainBoard.onClick((p) => {
      this.playMove(this.toPlay, p);
    });

    gtp.onData('mg-search', this.onSearch.bind(this));
    gtp.onData('mg-gamestate', this.onGameState.bind(this));

    util.getElement('pass').addEventListener('click', () => {
      if (this.mainBoard.enabled) {
        this.playMove(this.toPlay, 'pass');
      }
    });

    util.getElement('reset').addEventListener('click', () => {
      log.clear();
      qGraph.clear();
      gtp.newSession();
      gtp.send('clear_board', () => { this.gameOver = false; });
      gtp.send('gamestate');
      gtp.send('report_search_interval 50');
      gtp.send('info');
    });

    let initPlayerButton = (color: Color, elemId: string) => {
      let elem = util.getElement(elemId);
      this.playerElems[color] = elem;
      elem.addEventListener('click', () => {
        if (elem.innerText == 'Human') {
          elem.innerText = 'Minigo';
          if (this.toPlay == color && !this.minigoBusy) {
            this.genMove();
          }
        } else {
          elem.innerText = 'Human';
        }
      });
    };
    initPlayerButton(Color.Black, 'black-player');
    initPlayerButton(Color.White, 'white-player');
  }

  private genMove() {
    this.minigoBusy = true;
    gtp.send('genmove', (result: string) => {
      this.minigoBusy = false;
      this.onMovePlayed(util.parseGtpMove(result, this.mainBoard.size));
    });
  }

  onSearch(msg: SearchMsg) {
    this.history[msg.move].n = msg.n;
    this.history[msg.move].dq = msg.dq;
    this.history[msg.move].search = util.parseMoves(msg.search, this.size);
    if (msg.pv) {
      this.history[msg.move].pv = util.parseMoves(msg.search, this.size);
    }
    // TODO(tommadams): This will redraw the main board, even if nothing has
    // changed.
    this.refresh();
  }

  onGameState(msg: GameStateMsg) {
    // Update stones on the board.
    let stoneMap: {[index: string]: Color} = {
      '.': Color.Empty,
      'X': Color.Black,
      'O': Color.White,
    };
    let stones = [];
    for (let i = 0; i < msg.board.length; ++i) {
      stones.push(stoneMap[msg.board[i]]);
    }

    this.toPlay = util.parseGtpColor(msg.toPlay);
    let lastMove = msg.lastMove ? util.parseGtpMove(msg.lastMove, this.size) : null;
    this.history[msg.n] = new BoardState(stones, lastMove, this.toPlay);
    if (msg.n == this.activeMove + 1) {
      this.activeMove = msg.n;
    }

    qGraph.setMoveScore(msg.n, msg.q);
    qGraph.draw();

    if (!this.gameOver && this.playerElems[this.toPlay].innerText == 'Minigo') {
      this.genMove();
    }

    this.refresh();
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
    gtp.send(`play ${colorStr} ${moveStr}`, (result: string, ok: boolean) => {
      if (ok) {
        this.onMovePlayed(move);
      }
    });
  }

  // TODO(tommadams): Make private
  // Callback invoked when either a human or Minigo plays a valid move.
  onMovePlayed(move: Move) {
    if (move == 'pass') {
      if (++this.numConsecutivePasses == 2) {
        this.gameOver = true;
      }
    } else {
      this.numConsecutivePasses = 0;
      if (move == 'resign') {
        this.gameOver = true;
      }
    }

    log.scroll();
    gtp.send('gamestate');
    if (this.gameOver) {
      gtp.send('final_score', (result: string, ok: boolean) => {
        if (!ok) {
          return;
        }
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
        log.log(prettyResult);
        log.scroll();
      });
    }
  }

  private refresh() {
    if (!this.drawPending) {
      this.drawPending = true;
      window.requestAnimationFrame(() => {
        this.drawPending = false;
        for (let board of this.boards) {
          board.update(this.history[this.activeMove]);
          board.draw();
        }
      });
    }
  }
}

let app: App;

// Initialize log.
let log = new Log('log', 'console');
let consoleElem = util.getElement('console')
consoleElem.addEventListener('keypress', (e) => {
  if (e.keyCode == 13) {
    let cmd = consoleElem.innerText.trim();
    if (cmd != '') {
      gtp.send(cmd, () => { log.scroll(); });
    }
    consoleElem.innerHTML = '';
    e.preventDefault();
    return false;
  }
});

// Initialize socket connection to the backend.
let gtp = new gtpsock.Socket(
  `http://${document.domain}:${location.port}/minigui`,
  (line: string) => { log.log(line, 'log-cmd'); });

function init(boardSize: number) {
  app = new App(boardSize);

  gtp.send('clear_board');
  gtp.send('gamestate');
  gtp.send('report_search_interval 250');
  gtp.send('info');
  gtp.send('ponder_limit 0');
  util.getElement('ui-container').classList.remove('hidden');
}

function tryBoardSize(sizes: BoardSize[]) {
  if (sizes.length == 0) {
    throw new Error('Couldn\'t find an acceptable board size');
  }
  gtp.send(`boardsize ${sizes[0]}`, (result, ok) => {
    if (ok) {
      init(sizes[0]);
    } else {
      tryBoardSize(sizes.slice(1));
    }
  });
}

gtp.onConnect(() => {
  if (!app) {
    tryBoardSize([BoardSize.Nine, BoardSize.Nineteen]);
  }
});

let qGraph = new Graph('q-graph');
// TODO(tommadams): do this
/*
qGraph.onMoveChanged((move: number | null) => {
  gameState.hoveredMove = move;
  for (let board of boards) {
    board.clearAnnotations();
  }

  if (move == null) {
    move = gameState.history.length - 1;
  }
  let h = gameState.history[move];
  if (h.lastMove != null && h.lastMove != 'pass' && h.lastMove != 'resign') {
    mainBoard.setMark(h.lastMove, '#ef6c02');
  }
  mainBoard.setStones(h.stones);
  dqBoard.setStones(h.stones);
  nBoard.setStones(h.stones);
  mainBoard.setVariation(h.pv);
  dqBoard.setHeatMap(h.q);
  nBoard.setHeatMap(h.n);

  for (let board of boards) {
    redraw.requestDraw(board);
  }
});
*/
