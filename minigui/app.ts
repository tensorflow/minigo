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

import {Annotation, Position, rootPosition} from './position'
import {Socket} from './gtp_socket'
import {Color, Move, N, Nullable, movesEqual, setBoardSize, stonesEqual, toKgs} from './base'
import {Board} from './board'
import * as util from './util'

// Raw position update JSON object. Gets parsed into a PositionUpdate.
interface PositionUpdateJson {
  id: string;
  moveNum: number;
  n: number;
  q: number;
  toPlay?: string;
  parentId?: string;
  stones?: string;
  lastMove?: string;
  search?: string[];
  pv?: string[];
  childN?: number[];
  childQ?: number[];
  gameOver?: boolean;
}

class PositionUpdate {
  n: number;
  q: number;
  moveNum: Nullable<number> = null;
  toPlay: Nullable<Color> = null;
  stones: Nullable<Color[]> = null;
  lastMove: Nullable<Move> = null;
  search: Nullable<Move[]> = null;
  pv: Nullable<Move[]> = null;
  childN: Nullable<number[]> = null;
  childQ: Nullable<number[]> = null;
  gameOver: Nullable<boolean> = null;

  constructor(public position: Position j: PositionUpdateJson) {
    this.id = j.id;
    this.moveNum = j.moveNum;
    this.n = j.n;
    this.q = j.q;

    if (j.toPlay) {
      this.toPlay = util.parseColor(j.toPlay);
    }

    if (j.parentId) {
      this.parentId = j.parentId;
    }

    if (j.stones) {
      const stoneMap: {[index: string]: Color} = {
        '.': Color.Empty,
        'X': Color.Black,
        'O': Color.White,
      };
      this.stones = [];
      for (let i = 0; i < j.board.length; ++i) {
        this.stones.push(stoneMap[j.board[i]]);
      }
    }

    if (j.lastMove) {
      this.lastMove = util.parseMove(j.lastMove);
    }

    if (j.search) {
      this.search = util.parseMoves(j.search);
    }
    if (j.pv) {
      this.pv = util.parseMoves(j.pv as string[]);
    }
    if (j.childN) {
      this.childN = j.childN;
    }
    if (j.childQ) {
      this.childQ = [];
      this.dq = [];
      for (let q of j.childQ) {
        q /= 1000;
        this.childQ.push(q);
        this.dq.push(q - this.q);
      }
    }
  }
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

  protected abstract onPosition(update: PositionUpdate): void;

  constructor() {
    this.gtp.onData('mg-search', (j: PositionUpdateJson) => {
      this.onSearch(new PositionUpdate(j));
    });
    this.gtp.onData('mg-gamestate', (j: PositionUpdateJson) => {
      let update = new PositionUpdate(j);
      if (update.gameOver != null) {
        this.gameOver = update.gameOver;
      }
      this.gameOver = j.gameOver;
      this.onPosition(update);
      if (j.gameOver) {
        this.onGameOver();
      }
    });
  }

  private parsePositionUpdate(j: PositionUpdateJson) {
    let position = positionMap.get(j.id);
    if (position !== undefined) {
      if (position.parent != null) {
        if (position.parent != positionMap.get(j.parentId)) {
          throw new Error('parents don\'t match');
        }
      } else {
        if (j.parentId !== undefined) {
          throw new Error('parents don\'t match');
        }
      }
    } else {
      let parent = positionMap.get(j.parentId);
      if (parent == null) {
        throw new Error('can\'t find parent');
      }
      position = parent.addChild(j.id, j.lastMove, stones, j.q);
    }

    return update;
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
        'root', null, util.emptyBoard(), 0, null, Color.Black, true)
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
  protected updateBoards(position: Position) {
    for (let board of this.boards) {
      if (board.setPosition(position)) {
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
    const props = ['n', 'q', 'dq', 'pv', 'search', 'childN', 'childQ'];
    util.partialUpdate(msg, position, props);

    if (position == this.activePosition) {
      this.updateBoards(position);
    }
  }

  private parseGameState(j: GameStateJson) {
    // Parse parts of the JSON object that need it.
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
      position = parent.addChild(j.id, lastMove, stones, j.q);
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
  SearchMsg,
}
