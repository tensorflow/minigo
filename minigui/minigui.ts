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

import * as board from './board'
import * as util from './util'
import {Log} from './log'
import * as gtpsock from './gtp_socket'
import {Graph} from './graph'

let N = 0;

namespace BoardState {
  export type Move = board.Point | 'pass' | 'resign';
}

class BoardState {
  stones: Array<board.Color>;
  principalVariation = new Array<board.Move>();
  n: Array<number>;
  q: Array<number>;
  lastMove: BoardState.Move | null = null;

  constructor(stones: Array<board.Color> | null, lastMove: BoardState.Move | null) {
    this.lastMove = lastMove;
    if (stones) {
      this.stones = stones;
    } else {
      this.stones = new Array<board.Color>(N * N);
      for (let i = 0; i < N * N; ++i) {
        this.stones[i] = board.Color.Empty;
      }
    }
    this.n = new Array<number>(N * N);
    this.q = new Array<number>(N * N);
    for (let i = 0; i < N * N; ++i) {
      this.n[i] = 0;
      this.q[i] = 0;
    }
  }
}

// Game state.
class GameState {
  numConsecutivePasses = 0;
  gameOver = true;
  hoveredMove: number | null = null;
  moveNumber = 0;
  history = [new BoardState(null, null)];
}
let gameState = new GameState();

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

let mainBoard: board.ClickableBoard;
let currentVariationBoard: board.Board;
let qBoard: board.Board;
let nBoard: board.Board;
let boards: Array<board.Board>;

// Initialize socket connection to the backend.
let gtp = new gtpsock.Socket(
  `http://${document.domain}:${location.port}/minigui`,
  (line: string) => { log.log(line, 'log-cmd'); });

function init(boardSize: number) {
  N = boardSize;
  mainBoard = new board.ClickableBoard(
      'main-board', N, {margin: 30, labelRowCol: true, starPointRadius: 4});
  currentVariationBoard = new board.Board('search-board', N, {caption: 'search'});
  qBoard = new board.Board('q-board', N, {caption: 'ΔQ'});
  nBoard = new board.Board('n-board', N, {caption: 'N'});

  boards = [
    mainBoard,
    currentVariationBoard,
    qBoard,
    nBoard,
  ];
  for (let board of boards) {
    board.draw();
  }

  gtp.send('clear_board');
  gtp.send('mg_gamestate');
  gtp.send('report_search_interval 50');
  gtp.send('info');
  gameState.gameOver = false;
  util.getElement('ui-container').classList.remove('hidden');

  mainBoard.onClick((p) => {
    mainBoard.clearVariation();
    currentVariationBoard.clearVariation();
    qBoard.clearHeatMap();
    nBoard.clearHeatMap();

    let player = mainBoard.toPlay == board.Color.Black ? 'b' : 'w';
    let row = N - p.row;
    let col = board.COL_LABELS[p.col];
    let move = `${col}${row}`;
    gtp.send(`play ${player} ${move}`, (cmd: string, result: string, ok: boolean) => {
      if (ok) {
        onMovePlayed(move);
      }
    });
  });
}

function tryBoardSize(sizes: board.BoardSize[]) {
  if (sizes.length == 0) {
    throw new Error('Couldn\'t find an acceptable board size');
  }
  gtp.send(`boardsize ${sizes[0]}`, (cmd, result, ok) => {
    if (ok) {
      init(sizes[0]);
    } else {
      tryBoardSize(sizes.slice(1));
    }
  });
}

gtp.onConnect(() => {
  if (!mainBoard) {
    tryBoardSize([board.BoardSize.Nine, board.BoardSize.Nineteen]);
  }
});

let qGraph = new Graph('q-graph');
qGraph.onMoveChanged((move: number | null) => {
  gameState.hoveredMove = move;
  if (move == null) {
    move = gameState.history.length - 1;
  }
  let h = gameState.history[move];
  mainBoard.clearMarks();
  if (h.lastMove != null && h.lastMove != 'pass' && h.lastMove != 'resign') {
    mainBoard.setMark(h.lastMove, '#ef6c02');
  }
  mainBoard.setStones(h.stones);
  qBoard.setStones(h.stones);
  nBoard.setStones(h.stones);
  mainBoard.setVariation(h.principalVariation);
  qBoard.setHeatMap(h.q);
  nBoard.setHeatMap(h.n);
});

enum Controller {
  Human,
  Minigo,
}

class Player {
  color: board.Color;
  controller = Controller.Human;
  thinking = false;

  constructor(color: board.Color, buttonId: string) {
    this.color = color;
    let button = util.getElement(buttonId);
    button.addEventListener('click', () => {
      if (this.controller == Controller.Human) {
        this.controller = Controller.Minigo;
      } else {
        this.controller = Controller.Human;
      }
      button.innerHTML = `${Controller[this.controller]}`;
      gtp.send('mg_gamestate');
    });
  }

  think() {
    if (this.controller != Controller.Minigo) {
      throw new Error(`Unexpected controller: ${Controller[this.controller]}`);
    }
    this.thinking = true;
    gtp.send('mg_genmove 20', (cmd: string, result: string) => {
      this.thinking = false;
      onMovePlayed(result);
    });
  }
}

// Callback invoked when either a human or Minigo plays a valid move.
function onMovePlayed(move: string) {
  let gameOver = false;
  if (move == 'pass') {
    if (++gameState.numConsecutivePasses == 2) {
      gameOver = true;
    }
  } else {
    gameState.numConsecutivePasses = 0;
    if (move == 'resign') {
      gameOver = true;
    }
  }

  log.scroll();
  gtp.send('mg_gamestate');

  if (gameOver && !gameState.gameOver) {
    gameState.gameOver = true;

    gtp.send('final_score', (cmd: string, result: string, ok: boolean) => {
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

    if (isSelfPlay()) {
      mainBoard.clearMarks();
      mainBoard.clearVariation();
      if (modeSelect.innerHTML == 'demo') {
        resetGame(10);
      }
    }
  }
}

// TODO(tommadams): Change the index type to [color in Color] once
// https://github.com/Microsoft/TypeScript/issues/13042 is fixed.
let players: {[index: number]: Player} = {
  [board.Color.Black]: new Player(board.Color.Black, 'black-player'),
  [board.Color.White]: new Player(board.Color.White, 'white-player'),
};

util.getElement('pass').addEventListener('click', () => {
  if (!mainBoard.enabled) {
    return;
  }
  let player = mainBoard.toPlay == board.Color.Black ? 'b' : 'w';
  gtp.send(`play ${player} pass`, () => { onMovePlayed('pass'); });
});

util.getElement('reset').addEventListener('click', () => { resetGame(); });

let modeSelect = util.getElement('mode');
modeSelect.addEventListener('click', () => {
  if (modeSelect.innerHTML == 'demo') {
    modeSelect.innerHTML = 'play';
  } else {
    modeSelect.innerHTML = 'demo';
  }
});

let pendingReset = -1;
function resetGame(delay?: number) {
  if (delay) {
    if (pendingReset != -1) {
      window.clearTimeout(pendingReset);
      pendingReset = -1;
    }
    pendingReset = window.setTimeout(resetGame, delay * 1000);
    return;
  }

  gameState = new GameState();

  log.clear();
  for (let color in players) {
    players[color].thinking = false;
  }
  qGraph.clear();
  mainBoard.clearVariation();
  currentVariationBoard.clearVariation();
  qBoard.clearHeatMap();
  nBoard.clearHeatMap();
  gtp.newSession();
  gtp.send('clear_board', () => { gameState.gameOver = false; });
  gtp.send('mg_gamestate');
  gtp.send('report_search_interval 50');
  gtp.send('info');
}

function isSelfPlay() {
  return (players[board.Color.Black].controller == Controller.Minigo &&
          players[board.Color.White].controller == Controller.Minigo);
}

function searchHandler(line: string) {
  let variation = util.parseVariation(line.trim(), N, mainBoard.toPlay);
  currentVariationBoard.setVariation(variation);
}

function principalVariationHandler(line: string) {
  let variation = util.parseVariation(line.trim(), N, mainBoard.toPlay);
  gameState.history[gameState.moveNumber].principalVariation = variation;
  if (gameState.hoveredMove == null) {
    mainBoard.setVariation(variation);
  }
}

function qHandler(line: string) {
  let heatMap = [];
  for (let s of line.trim().split(' ')) {
    let x = parseFloat(s);
    if (x > 0) {
      x = Math.sqrt(x);
    } else {
      x = -Math.sqrt(-x);
    }
    x = Math.max(-0.6, Math.min(x, 0.6));
    heatMap.push(x);
  }
  // TODO(tommadams): Store the raw heat map and add a transform function to the
  // HeatMapBoard.
  gameState.history[gameState.moveNumber].q = heatMap;
  if (gameState.hoveredMove == null) {
    qBoard.setHeatMap(heatMap);
  }
}

function nHandler(line: string) {
  let ns = new Array<number>();
  let nSum = 0;
  for (let s of line.trim().split(' ')) {
    let n = parseInt(s);
    nSum += n;
    ns.push(n);
  }
  for (let i = 0; i < ns.length; ++i) {
    ns[i] /= nSum;
  }

  let heatMap = [];
  for (let n of ns) {
    n = Math.min(Math.sqrt(n), 0.6);
    if (n > 0) {
      n = 0.1 + 0.9 * n;
    }
    heatMap.push(n);
  }
  // TODO(tommadams): Store the raw heat map and add a transform function to the
  // HeatMapBoard.
  gameState.history[gameState.moveNumber].n = heatMap;
  if (gameState.hoveredMove == null) {
    nBoard.setHeatMap(heatMap);
  }
}

interface GameState {
  session: string;
  board: string;
  lastMove: string;
  toPlay: string;
  n: number;
  q: number;
}
function gameStateHandler(line: string) {
  // Parse the response from the backend.
  let obj = JSON.parse(line) as GameState;

  // Update stones on the board.
  let stoneMap: {[index: string]: board.Color} = {
    '.': board.Color.Empty,
    'X': board.Color.Black,
    'O': board.Color.White,
  };
  let stones = [];
  for (let i = 0; i < obj.board.length; ++i) {
    stones.push(stoneMap[obj.board[i]]);
  }
  if (gameState.hoveredMove != null) {
    currentVariationBoard.setStones(stones);
  } else {
    for (let board of boards) {
      board.setStones(stones);
    }
  }

  let lastMove = obj.lastMove ? util.parseGtpPoint(obj.lastMove, N) : null;

  if (obj.n == gameState.history.length) {
    gameState.history.push(new BoardState(stones, lastMove));
  }

  // Set the last move marker.
  mainBoard.clearMarks();
  if (obj.lastMove) {
    let move = util.parseGtpPoint(obj.lastMove, N);
    if (move != 'pass' && move != 'resign') {
      if (gameState.hoveredMove == null) {
        mainBoard.setMark(move, '#ef6c02');
      }
    }
  }

  gameState.moveNumber = obj.n;

  qGraph.setMoveScore(obj.n, obj.q);

  // Redraw everything.
  for (let board of boards) {
    board.draw();
  }
  qGraph.draw();

  // Update whose turn it is.
  let toPlay = obj.toPlay == "Black" ? board.Color.Black : board.Color.White;
  for (let board of boards) {
    board.toPlay = toPlay;
  }

  if (players[toPlay].controller == Controller.Human) {
    mainBoard.enabled = !gameState.gameOver;
    mainBoard.clearVariation();
    currentVariationBoard.clearVariation();
  } else {
    mainBoard.enabled = false;
    if (!gameState.gameOver &&
        !players[board.Color.Black].thinking &&
        !players[board.Color.White].thinking) {
      players[toPlay].think();
    }
  }
}

function defaultStderrHandler(line: string) {
  if (line.indexOf(' ==> ') == -1) {
    log.log(line);
  }
}

const STDERR_HANDLERS: Array<[string, gtpsock.StderrHandler]> = [
  ['mg-search:', searchHandler],
  ['mg-pv:', principalVariationHandler],
  ['mg-q:', qHandler],
  ['mg-n:', nHandler],
  ['mg-gamestate:', gameStateHandler],
  ['', defaultStderrHandler],
];

for (let [prefix, handler] of STDERR_HANDLERS) {
  gtp.addStderrHandler(prefix, handler);
}
