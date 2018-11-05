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

import {Annotation, Position} from './position'
import {Socket} from './gtp_socket'
import {Color, Move, N, Nullable, movesEqual, setBoardSize, stonesEqual, toKgs} from './base'
import {Board} from './board'
import * as util from './util'

interface SearchJson {
  id: string;
  moveNum: number;
  toPlay: string;
  search: string[];
  n?: number[];
  dq?: number[];
  pv?: string[];
  childQ: number[];
}

class SearchMsg {
  id: string;
  moveNum: number;
  toPlay: Color;
  search: Move[];
  n: Nullable<number[]> = null;
  dq: Nullable<number[]> = null;
  pv: Nullable<Move[]> = null;
  childQ: number[];

  constructor(j: SearchJson) {
    this.id = j.id;
    this.moveNum = j.moveNum;
    this.toPlay = util.parseGtpColor(j.toPlay as string);
    this.search = util.parseMoves(j.search as string[], N);
    this.childQ = j.childQ;
    if (j.n) {
      this.n = j.n;
    }
    if (j.dq) {
      this.dq = j.dq;
    }
    if (j.pv) {
      this.pv = util.parseMoves(j.pv as string[], N);
    }
  }
}

interface GameStateJson {
  id: string;
  parent?: string;
  board: string;
  toPlay: string;
  lastMove?: string;
  moveNum: number;
  q: number;
  gameOver: boolean;
}

// App base class used by the different Minigui UI implementations.
abstract class App {
  // WebSocket connection to the Miniui backend server.
  protected gtp = new Socket();

  // True if Minigo is processing a genmove command.
  protected engineBusy = false;

  // The root of the game tree, an empty board.
  protected rootPosition: Position;

  // The currently active position.
  protected activePosition: Position;

  // True when the backend reports that the game is over.
  protected gameOver = true;

  // List of Board views.
  private boards: Board[] = [];

  protected positionMap = new Map<string, Position>();

  protected abstract onGameOver(): void;

  protected abstract onPosition(position: Position): void;

  constructor() {
    this.gtp.onData('mg-search', (j: SearchJson) => {
      this.onSearch(new SearchMsg(j));
    });
    this.gtp.onData('mg-gamestate', (j: GameStateJson) => {
      let position = this.parseGameState(j);
      this.gameOver = j.gameOver;
      this.onPosition(position);
      if (j.gameOver) {
        this.onGameOver();
      }
    });
  }

  protected connect() {
    let uri = `http://${document.domain}:${location.port}/minigui`;
    return this.gtp.connect(uri).then((size: number) => { setBoardSize(size); });
  }

  protected init(boards: Board[]) {
    this.boards = boards;
  }

  protected newGame() {
    this.rootPosition = new Position(
        null, util.emptyBoard(), 0, null, Color.Black, true)
    this.activePosition = this.rootPosition;

    this.positionMap.clear();

    this.gtp.send('clear_board');
    this.gtp.send('gamestate');
    this.gtp.send('info');

    // Iterate over the data-* attributes attached to the main minigui container
    // element, looking for data-gtp-* attributes. Send any matching ones as GTP
    // commands to the backend.
    let containerElem = document.querySelector('.minigui') as HTMLElement;
    if (containerElem != null) {
      let dataset = containerElem.dataset;
      for (let key in dataset) {
        if (key.startsWith('gtp')) {
          let cmd = key.substr(3).toLowerCase();
          let args = dataset[key];
          this.gtp.send(`${cmd} ${args}`);
        }
      }
    }

    this.updateBoards(this.activePosition);
  }

  // Updates all layers of the App's boards from a state object. Typically the
  // static object is a JSON blob sent from the backend with things like the
  // stones on the board, the current principle variation, visit counts, etc.
  // Layers that derive from DataLayer will check the given state object for a
  // specific property. If the state object contains a matching property, the
  // layer will update its copy of the state and signal that the board should
  // be redrawn (by their update method returning true).
  protected updateBoards(state: any) {
    for (let board of this.boards) {
      if (board.update(state)) {
        board.draw();
      }
    }
  }

  protected onSearch(msg: SearchMsg) {
    // Update the board state with contents of the search.
    // Copies the properties named in `props` that are present in `msg` into the
    // current position.
    // TODO(tommadams): It would be more flexible to allow the position
    // to store any/all properties return in the SearchMsg, without having to
    // specify this property list.
    let position = this.positionMap.get(msg.id);
    if (!position) {
      // Ignore search messages for positions we don't know about.
      // This can happen when refreshing the page, for example.
      return;
    }
    const props = ['n', 'dq', 'pv', 'search', 'childQ'];
    util.partialUpdate(msg, position, props);

    if (position == this.activePosition) {
      this.updateBoards(position);
    }
  }

  private parseGameState(j: GameStateJson) {
    // Parse parts of the JSON object that need it.
    const stoneMap: {[index: string]: Color} = {
      '.': Color.Empty,
      'X': Color.Black,
      'O': Color.White,
    };
    let stones = [];
    for (let i = 0; i < j.board.length; ++i) {
      stones.push(stoneMap[j.board[i]]);
    }
    let toPlay = util.parseGtpColor(j.toPlay);

    let lastMove: Nullable<Move> = null;
    if (j.lastMove) {
      lastMove = util.parseGtpMove(j.lastMove, N);
    }

    let position = this.positionMap.get(j.id);
    if (position !== undefined) {
      // We've already seen this position, verify that the JSON matches
      // what we have.
      if (position.toPlay != toPlay) {
        throw new Error('toPlay doesn\'t match');
      }
      if (!movesEqual(position.lastMove, lastMove)) {
        throw new Error('lastMove doesn\'t match');
      }
      if (!stonesEqual(position.stones, stones)) {
        throw new Error('stones don\'t match');
      }
      if (j.parent !== undefined) {
        if (position.parent != this.positionMap.get(j.parent)) {
          throw new Error('parents don\'t match');
        }
      }
      return position;
    }

    if (j.parent === undefined) {
      // No parent, this must be the root.
      if (lastMove != null) {
        throw new Error('lastMove mustn\'t be set for root position');
      }
      position = this.rootPosition;
    } else {
      // The position has a parent, which must exist in the positionMap.
      let parent = this.positionMap.get(j.parent);
      if (parent === undefined) {
        throw new Error(
            `Can't find parent with id ${j.parent} for position ${j.id}`);
      }
      if (lastMove == null) {
        throw new Error('lastMove must be set for non-root position');
      }
      position = parent.addChild(lastMove, stones, j.q);
    }

    if (position.toPlay != toPlay) {
      throw new Error(`expected ${position.toPlay}, got ${toPlay}`);
    }

    this.positionMap.set(j.id, position);
    return position;
  }
}

export {
  App,
}
