define(["require", "exports", "./position", "./gtp_socket", "./base", "./util"], function (require, exports, position_1, gtp_socket_1, base_1, util) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    class App {
        constructor() {
            this.gtp = new gtp_socket_1.Socket();
            this.engineBusy = false;
            this.positionMap = new Map();
            this.gtp.onData('mg-update', (j) => {
                let position = this.positionMap.get(j.id);
                if (position === undefined) {
                    return;
                }
                let update = this.newPositionUpdate(j);
                position.update(update);
                this.onPositionUpdate(position, update);
            });
            this.gtp.onData('mg-position', (j) => {
                let position = this.newPosition(j);
                let update = this.newPositionUpdate(j);
                position.update(update);
                this.onNewPosition(position);
                if (position.gameOver) {
                    this.onGameOver();
                }
            });
        }
        newPosition(j) {
            let position = this.positionMap.get(j.id);
            if (position !== undefined) {
                return position;
            }
            for (let prop of ['stones', 'toPlay', 'gameOver']) {
                if (!j.hasOwnProperty(prop)) {
                    throw new Error(`missing required property: ${prop}`);
                }
            }
            let def = j;
            let stones = [];
            const stoneMap = {
                '.': base_1.Color.Empty,
                'X': base_1.Color.Black,
                'O': base_1.Color.White,
            };
            for (let i = 0; i < def.stones.length; ++i) {
                stones.push(stoneMap[def.stones[i]]);
            }
            let toPlay = util.parseColor(def.toPlay);
            let gameOver = def.gameOver;
            if (def.parentId === undefined) {
                if (def.move != null) {
                    throw new Error('move mustn\'t be set for root position');
                }
                position = this.rootPosition;
                position.id = def.id;
            }
            else {
                let parent = this.positionMap.get(def.parentId);
                if (parent === undefined) {
                    throw new Error(`Can't find parent with id ${def.parentId} for position ${def.id}`);
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
            if (j.captures !== undefined) {
                position.captures[0] = j.captures[0];
                position.captures[1] = j.captures[1];
            }
            if (position.toPlay != toPlay) {
                throw new Error(`expected ${position.toPlay}, got ${toPlay}`);
            }
            this.positionMap.set(position.id, position);
            return position;
        }
        newPositionUpdate(j) {
            let update = {};
            if (j.n != null) {
                update.n = j.n;
            }
            if (j.q != null) {
                update.q = j.q;
            }
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
        connect() {
            let uri = `http://${document.domain}:${location.port}/minigui`;
            let params = new URLSearchParams(window.location.search);
            let p = params.get("gtp_debug");
            let debug = (p != null) && (p == "" || p == "1" || p.toLowerCase() == "true");
            return this.gtp.connect(uri, debug).then((size) => {
                base_1.setBoardSize(size);
                let stones = new Array(base_1.N * base_1.N);
                stones.fill(base_1.Color.Empty);
                this.rootPosition = new position_1.Position('dummy-root', null, stones, null, base_1.Color.Black, false, true);
                this.activePosition = this.rootPosition;
            });
        }
        newGame() {
            this.positionMap.clear();
            this.rootPosition.children = [];
            this.activePosition = this.rootPosition;
            this.gtp.send('clear_board');
            this.gtp.send('info');
            let containerElem = document.querySelector('.minigui');
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
    exports.App = App;
});
//# sourceMappingURL=app.js.map