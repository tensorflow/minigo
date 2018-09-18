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
import {Socket} from './gtp_socket'
import {WinrateGraph} from './winrate_graph'

const HUMAN = 'Human';
const MINIGO = 'Minigo';

class Position {
  search: Move[] = [];
  pv: Move[] = [];
  n: Nullable<number[]> = null;
  dq: Nullable<number[]> = null;
  annotations: Annotation[] = [];

  constructor(public moveNum: number,
              public stones: Color[],
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
  search: string[] | Move[];
  n: number[];
  dq: number[];
  pv?: string[] | Move[];
}

export interface GameStateMsg {
  board: string;
  toPlay: string;
  lastMove?: string;
  moveNum: number;
  q: number;
  gameOver: boolean;
}

class App {
  private size: number;
  private mainBoard: ClickableBoard;
  private searchBoard: Board;
  private nBoard: Board;
  private dqBoard: Board;
  private boards: Board[] = [];
  private positionHistory: Position[];
  private activePosition: Position;
  private numConsecutivePasses = 0;
  private gameOver = false;
  private minigoBusy = false;
  private playerElems: HTMLElement[] = [];
  private toPlay = Color.Black;
  private winrateGraph = new WinrateGraph('winrate-graph');

  private log = new Log('log', 'console');
  private gtp: Socket;

  constructor(uri: string) {
    this.gtp = new Socket();
    this.gtp.connect(uri).then((size: number) => { this.init(size); });
  }

  init(size: number) {
    this.size = size;

    this.activePosition = new Position(
      0, util.emptyBoard(this.size), null, Color.Black)
    this.positionHistory = [this.activePosition];

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

    this.gtp.onData('mg-search', this.onSearch.bind(this));
    this.gtp.onData('mg-gamestate', this.onGameState.bind(this));

    util.getElement('pass').addEventListener('click', () => {
      if (this.mainBoard.enabled) {
        this.playMove(this.toPlay, 'pass');
      }
    });

    util.getElement('reset').addEventListener('click', () => {
      this.log.clear();
      this.winrateGraph.clear();
      this.gtp.newSession();
      this.gtp.send('clear_board');
      this.gtp.send('gamestate');
      this.gtp.send('report_search_interval 50');
      this.gtp.send('info');
    });

    let initPlayerButton = (color: Color, elemId: string) => {
      let elem = util.getElement(elemId);
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
    this.log = new Log('log', 'console');
    this.log.onConsoleCmd((cmd: string) => {
      this.gtp.send(cmd).then(() => { this.log.scroll(); });
    });

    this.gtp.onText((line: string) => { this.log.log(line, 'log-cmd'); });
    this.gtp.send('clear_board');
    this.gtp.send('gamestate');
    this.gtp.send('report_search_interval 250');
    this.gtp.send('info');
    this.gtp.send('ponder_limit 0');
  }

  private onPlayerChanged() {
    if (this.minigoBusy || this.gameOver) {
      return;
    }

    if (this.playerElems[this.toPlay].innerText == MINIGO) {
      this.mainBoard.enabled = false;
      this.minigoBusy = true;
      this.gtp.send('genmove').then((move: string) => {
        this.minigoBusy = false;
        this.onMovePlayed(util.parseGtpMove(move, this.mainBoard.size));
      });
    } else {
      this.mainBoard.enabled = true;
    }
  }

  private onSearch(msg: SearchMsg) {
    // Parse move variations.
    msg.search = util.parseMoves(msg.search as string[], this.size);
    if (msg.pv) {
      msg.pv = util.parseMoves(msg.pv as string[], this.size);
    }

    // Update the board state with contents of the search.
    const props = ['n', 'dq', 'pv', 'search'];
    util.partialUpdate(msg, this.positionHistory[msg.move], props);

    // Update the boards.
    this.updateBoards(msg);
  }

  private onGameState(msg: GameStateMsg) {
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
    this.gameOver = msg.gameOver;

    let lastMove = msg.lastMove ? util.parseGtpMove(msg.lastMove, this.size) : null;
    let position = new Position(msg.moveNum, stones, lastMove, this.toPlay);
    this.positionHistory[msg.moveNum] = position;
    if (msg.moveNum == this.activePosition.moveNum + 1) {
      this.activePosition = position;
      this.updateBoards(this.activePosition);
    }

    this.winrateGraph.setMoveScore(msg.moveNum, msg.q);
    this.winrateGraph.draw();

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
      this.onMovePlayed(move);
    });
  }

  // Callback invoked when either a human or Minigo plays a valid move.
  private onMovePlayed(move: Move) {
    this.log.scroll();
    this.gtp.send('gamestate');
    if (this.gameOver) {
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

  private updateBoards(state: any) {
    for (let board of this.boards) {
      if (board.update(state)) {
        board.draw();
      }
    }
  }
}

new App(`http://${document.domain}:${location.port}/minigui`);

