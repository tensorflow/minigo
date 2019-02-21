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
import {Color, Move, N, setBoardSize} from './base'
import {Board} from './board'
import * as util from './util'

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

  protected abstract onPositionUpdate(position: Position, update: Position.Update): void;

  protected abstract onNewPosition(position: Position): void;

  // We track game over state separately from the gameOver property of the
  // latest position because when one player resigns, the game ends without a
  // new Position with gameOver == true being generated.
  protected gameOver = false;

  constructor() {
    this.gtp.onData('mg-update', (j: Position.Update) => {
      let position = this.positionMap.get(j.id);
      if (position === undefined) {
        // Just after refreshing the page, the backend will still be sending
        // position updates if pondering is enabled. Ignore updates for any
        // positions we don't know about.
        return;
      }
      position.update(j);
      this.onPositionUpdate(position, j);
    });

    this.gtp.onData('mg-position', (j: Position.Definition | Position.Update) => {
      let position: Position;
      let def = j as Position.Definition;
      if (def.move == null) {
        // No parent, this must be the root.
        let p = this.positionMap.get(def.id);
        if (p == null) {
          p = new Position(def);
          this.rootPosition = p;
        }
        position = this.rootPosition;
      } else {
        // Get the parent.
        if (def.parentId === undefined) {
          throw new Error('child node must have a valid parentId');
        }
        let parent = this.positionMap.get(def.parentId);
        if (parent == null) {
          throw new Error(`couldn't find parent ${def.parentId}`);
        }

        // See if we already have this position.
        let child = parent.getChild(util.parseMove(def.move));
        if (child != null) {
          position = child;
        } else {
          position = new Position(def);
          parent.addChild(position);
        }
      }
      position.update(j);
      this.onNewPosition(position);
      this.positionMap.set(position.id, position);
    });
  }

  protected onGameOver() {
    this.gameOver = true;
  }

  protected connect() {
    let uri = `http://${document.domain}:${location.port}/minigui`;
    let params = new URLSearchParams(window.location.search);
    let p = params.get("gtp_debug");
    let debug = (p != null) && (p == "" || p == "1" || p.toLowerCase() == "true");
    return fetch('config').then((response) => {
      return response.json();
    }).then((cfg: any) => {
      // TODO(tommadams): Give cfg a real type.

      // setBoardSize sets the global variable N to the board size for the game
      // (as provided by the backend engine). The code uses N from hereon in.
      setBoardSize(cfg.boardSize);
      let stones = new Array<Color>(N * N);
      stones.fill(Color.Empty);
      this.rootPosition = new Position({
        id: 'dummy-root',
        moveNum: 0,
        toPlay: 'b',
      });
      this.activePosition = this.rootPosition;

      if (cfg.players.length != 1) {
        throw new Error(`expected 1 player, got ${cfg.players}`);
      }
      return this.gtp.connect(uri, cfg.players[0], debug);
    });
  }

  protected newGame(): Promise<any> {
    this.gameOver = false;
    this.positionMap.clear();
    this.rootPosition.children = [];
    this.activePosition = this.rootPosition;

    // TODO(tommadams): Move this functionality into .ctl files.
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

    return this.gtp.send('clear_board');
  }
}

export {
  App,
}
