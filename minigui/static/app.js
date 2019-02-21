define(["require", "exports", "./position", "./gtp_socket", "./base", "./util"], function (require, exports, position_1, gtp_socket_1, base_1, util) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    class App {
        constructor() {
            this.gtp = new gtp_socket_1.Socket();
            this.engineBusy = false;
            this.positionMap = new Map();
            this.gameOver = false;
            this.gtp.onData('mg-update', (j) => {
                let position = this.positionMap.get(j.id);
                if (position === undefined) {
                    return;
                }
                position.update(j);
                this.onPositionUpdate(position, j);
            });
            this.gtp.onData('mg-position', (j) => {
                let position;
                let def = j;
                if (def.move == null) {
                    let p = this.positionMap.get(def.id);
                    if (p == null) {
                        p = new position_1.Position(def);
                        this.rootPosition = p;
                    }
                    position = this.rootPosition;
                }
                else {
                    if (def.parentId === undefined) {
                        throw new Error('child node must have a valid parentId');
                    }
                    let parent = this.positionMap.get(def.parentId);
                    if (parent == null) {
                        throw new Error(`couldn't find parent ${def.parentId}`);
                    }
                    let child = parent.getChild(util.parseMove(def.move));
                    if (child != null) {
                        position = child;
                    }
                    else {
                        position = new position_1.Position(def);
                        parent.addChild(position);
                    }
                }
                position.update(j);
                this.onNewPosition(position);
                this.positionMap.set(position.id, position);
            });
        }
        onGameOver() {
            this.gameOver = true;
        }
        connect() {
            let uri = `http://${document.domain}:${location.port}/minigui`;
            let params = new URLSearchParams(window.location.search);
            let p = params.get("gtp_debug");
            let debug = (p != null) && (p == "" || p == "1" || p.toLowerCase() == "true");
            return fetch('config').then((response) => {
                return response.json();
            }).then((cfg) => {
                base_1.setBoardSize(cfg.boardSize);
                let stones = new Array(base_1.N * base_1.N);
                stones.fill(base_1.Color.Empty);
                this.rootPosition = new position_1.Position({
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
        newGame() {
            this.gameOver = false;
            this.positionMap.clear();
            this.rootPosition.children = [];
            this.activePosition = this.rootPosition;
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
            return this.gtp.send('clear_board');
        }
    }
    exports.App = App;
});
//# sourceMappingURL=app.js.map