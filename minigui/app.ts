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

interface PositionJson {
  id: string;
  parentId?: string;
  moveNum: number;
  toPlay: string;
  stones: string;
  move?: string;
  gameOver: boolean;
  comment?: string;
  caps?: number[];
}

interface PositionUpdateJson {
  id: string;
  n?: number;
  q?: number;
  pv?: string;
  variations?: string[][];
  search?: string[];
  childN?: number[];
  childQ?: number[];
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

  protected positionMap = new Map<string, Position>();

  protected abstract onGameOver(): void;

  protected abstract onPositionUpdate(position: Position, update: Position.Update): void;

  protected abstract onNewPosition(position: Position): void;

  constructor() {
    this.gtp.onData('mg-update', (j: PositionUpdateJson) => {
      let position = this.positionMap.get(j.id);
      if (position === undefined) {
        // Just after refreshing the page, the backend will still be sending
        // position updates if pondering is enabled. Ignore updates for any
        // positions we don't know about.
        return;
      }
      let update = this.newPositionUpdate(j);
      position.update(update);
      this.onPositionUpdate(position, update);
    });

    this.gtp.onData('mg-position', (j: PositionJson | PositionUpdateJson) => {
      let position = this.newPosition(j as PositionJson);
      let update = this.newPositionUpdate(j);
      position.update(update);
      this.onNewPosition(position);
      if (position.gameOver) {
        this.onGameOver();
      }
    });
  }

  private newPosition(j: PositionJson): Position {
    let position = this.positionMap.get(j.id);
    if (position !== undefined) {
      return position;
    }

    for (let prop of ['stones', 'toPlay', 'gameOver']) {
      if (!j.hasOwnProperty(prop)) {
        throw new Error(`missing required property: ${prop}`);
      }
    }

    let def = j as PositionJson;
    let stones: Color[] = [];
    const stoneMap: {[index: string]: Color} = {
      '.': Color.Empty,
      'X': Color.Black,
      'O': Color.White,
    };
    for (let i = 0; i < def.stones.length; ++i) {
      stones.push(stoneMap[def.stones[i]]);
    }
    let toPlay = util.parseColor(def.toPlay);
    let gameOver = def.gameOver;

    if (def.parentId === undefined) {
      // No parent, this must be the root.
      if (def.move != null) {
        throw new Error('move mustn\'t be set for root position');
      }
      position = this.rootPosition;
      position.id = def.id;
    } else {
      // The position has a parent, which must exist in the positionMap.
      let parent = this.positionMap.get(def.parentId);
      if (parent === undefined) {
        throw new Error(
            `Can't find parent with id ${def.parentId} for position ${def.id}`);
      }
      if (def.move == null) {
        throw new Error('new positions must specify move');
      }
      let move = util.parseMove(def.move);
      position = parent.addChild(def.id, move, stones, gameOver);
    }

    if (j.comment) {
      position.comment = j.comment;
    }

    if (j.caps !== undefined) {
      position.captures[0] = j.caps[0];
      position.captures[1] = j.caps[1];
    }

    if (position.toPlay != toPlay) {
      throw new Error(`expected ${position.toPlay}, got ${toPlay}`);
    }
    this.positionMap.set(position.id, position);
    return position;
  }

  private newPositionUpdate(j: PositionUpdateJson): Position.Update {
    let update: Position.Update = {}
    if (j.n != null) { update.n = j.n; }
    if (j.q != null) { update.q = j.q; }
    if (j.variations != null) {
      update.variations = {};
      for (let key in j.variations) {
        if (key == null) {
          continue;
        }
        update.variations[key] = util.parseMoves(j.variations[key]);
      }
    }
    if (j.childN != null) {
      update.childN = j.childN;
    }
    if (j.childQ != null) {
      update.childQ = [];
      for (let q of j.childQ) {
        update.childQ.push(q / 1000);
      }
    }
    return update;
  }

  protected connect() {
    let uri = `http://${document.domain}:${location.port}/minigui`;
    let params = new URLSearchParams(window.location.search);
    let p = params.get("gtp_debug");
    let debug = (p != null) && (p == "" || p == "1" || p.toLowerCase() == "true");
    return this.gtp.connect(uri, debug).then((size: number) => {
      // setBoardSize sets the global variable N to the board size for the game
      // (as provided by the backend engine). The code uses N from hereon in.
      setBoardSize(size);

      let stones = new Array<Color>(N * N);
      stones.fill(Color.Empty);
      this.rootPosition = new Position(
          'dummy-root', null, stones, null, Color.Black, false, true);
      this.activePosition = this.rootPosition;
    });
  }

  protected newGame() {
    this.positionMap.clear();
    this.rootPosition.children = [];
    this.activePosition = this.rootPosition;

    this.gtp.send('clear_board');
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
  }
}

export {
  App,
}
