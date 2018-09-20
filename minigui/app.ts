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

import {Socket} from './gtp_socket'
import {Nullable, Color, Move} from './base'
import {Annotation, Board} from './board'
import * as util from './util'

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

interface SearchMsg {
  moveNum: number;
  toPlay: string;
  search: string[] | Move[];
  n: number[];
  dq: number[];
  pv?: string[] | Move[];
}

interface GameStateMsg {
  board: string;
  toPlay: string;
  lastMove?: string;
  moveNum: number;
  q: number;
  gameOver: boolean;
}

abstract class App {
  protected gtp = new Socket();
  protected size: number;
  protected engineBusy = false;
  protected positionHistory: Position[];
  protected activePosition: Position;
  protected gameOver = true;
  protected toPlay = Color.Black;

  private boards: Board[] = [];

  protected abstract onGameOver(): void;

  constructor() {
    this.gtp.onData('mg-search', this.onSearch.bind(this));
    this.gtp.onData('mg-gamestate', this.onGameState.bind(this));
  }

  protected connect() {
    let uri = `http://${document.domain}:${location.port}/minigui`;
    return this.gtp.connect(uri).then((size: number) => {
      this.size = size;
    });
  }

  protected init(boards: Board[]) {
    this.boards = boards;
  }

  protected newGame() {
    this.activePosition = new Position(
      0, util.emptyBoard(this.size), null, Color.Black)
    this.positionHistory = [this.activePosition];

    this.gtp.send('clear_board');
    this.gtp.send('gamestate');
    this.gtp.send('info');

    // TODO(tommadams): Make this configurable.
    this.gtp.send('report_search_interval 250');
    this.gtp.send('ponder_limit 0');

    this.updateBoards(this.activePosition);
  }

  protected updateBoards(state: any) {
    for (let board of this.boards) {
      if (board.update(state)) {
        board.draw();
      }
    }
  }

  protected onSearch(msg: SearchMsg) {
    // Parse move variations.
    msg.search = util.parseMoves(msg.search as string[], this.size);
    if (msg.pv) {
      msg.pv = util.parseMoves(msg.pv as string[], this.size);
    }

    // Update the board state with contents of the search.
    const props = ['n', 'dq', 'pv', 'search'];
    util.partialUpdate(msg, this.positionHistory[msg.moveNum], props);

    // Update the boards.
    if (msg.moveNum == this.activePosition.moveNum) {
      this.updateBoards(msg);
    }
  }

  protected onGameState(msg: GameStateMsg) {
    // Parse the raw message.
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
    let lastMove =
        msg.lastMove ? util.parseGtpMove(msg.lastMove, this.size) : null;

    // Create a new position.
    let position = new Position(msg.moveNum, stones, lastMove, this.toPlay);
    this.positionHistory[msg.moveNum] = position;
    if (msg.moveNum == this.activePosition.moveNum + 1) {
      this.activePosition = position;
      this.updateBoards(this.activePosition);
    }

    if (this.gameOver) {
      this.onGameOver();
    }
  }
}

export {
  App,
  Position,
  SearchMsg,
  GameStateMsg,
}
