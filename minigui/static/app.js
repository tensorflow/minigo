define(["require", "exports", "./position", "./gtp_socket", "./base", "./util"], function (require, exports, position_1, gtp_socket_1, base_1, util) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    class PositionUpdate {
        constructor(position, j) {
            this.position = position;
            this.moveNum = null;
            this.toPlay = null;
            this.stones = null;
            this.lastMove = null;
            this.search = null;
            this.pv = null;
            this.childN = null;
            this.childQ = null;
            this.gameOver = null;
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
                const stoneMap = {
                    '.': base_1.Color.Empty,
                    'X': base_1.Color.Black,
                    'O': base_1.Color.White,
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
                this.pv = util.parseMoves(j.pv);
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
    class App {
        constructor() {
            this.gtp = new gtp_socket_1.Socket();
            this.engineBusy = false;
            this.gameOver = true;
            this.boards = [];
            this.positionMap = new Map();
            this.gtp.onData('mg-search', (j) => {
                this.onSearch(new PositionUpdate(j));
            });
            this.gtp.onData('mg-gamestate', (j) => {
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
        parsePositionUpdate(j) {
            let position = positionMap.get(j.id);
            if (position !== undefined) {
                if (position.parent != null) {
                    if (position.parent != positionMap.get(j.parentId)) {
                        throw new Error('parents don\'t match');
                    }
                }
                else {
                    if (j.parentId !== undefined) {
                        throw new Error('parents don\'t match');
                    }
                }
            }
            else {
                let parent = positionMap.get(j.parentId);
                if (parent == null) {
                    throw new Error('can\'t find parent');
                }
                position = parent.addChild(j.id, j.lastMove, stones, j.q);
            }
            return update;
        }
        connect() {
            let uri = `http://${document.domain}:${location.port}/minigui`;
            return this.gtp.connect(uri).then((size) => { base_1.setBoardSize(size); });
        }
        init(boards) {
            this.boards = boards;
        }
        newGame() {
            this.rootPosition = new position_1.Position('root', null, util.emptyBoard(), 0, null, base_1.Color.Black, true);
            this.activePosition = this.rootPosition;
            this.positionMap.clear();
            this.gtp.send('clear_board');
            this.gtp.send('gamestate');
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
            this.updateBoards(this.activePosition);
        }
        updateBoards(position) {
            for (let board of this.boards) {
                if (board.setPosition(position)) {
                    board.draw();
                }
            }
        }
        onSearch(msg) {
            let position = this.positionMap.get(msg.id);
            if (!position) {
                return;
            }
            const props = ['n', 'q', 'dq', 'pv', 'search', 'childN', 'childQ'];
            util.partialUpdate(msg, position, props);
            if (position == this.activePosition) {
                this.updateBoards(position);
            }
        }
        parseGameState(j) {
            let position = this.positionMap.get(j.id);
            if (position !== undefined) {
                if (position.toPlay != toPlay) {
                    throw new Error('toPlay doesn\'t match');
                }
                if (!base_1.movesEqual(position.lastMove, lastMove)) {
                    throw new Error('lastMove doesn\'t match');
                }
                if (!base_1.stonesEqual(position.stones, stones)) {
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
                if (lastMove != null) {
                    throw new Error('lastMove mustn\'t be set for root position');
                }
                position = this.rootPosition;
            }
            else {
                let parent = this.positionMap.get(j.parent);
                if (parent === undefined) {
                    throw new Error(`Can't find parent with id ${j.parent} for position ${j.id}`);
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
    exports.App = App;
});
//# sourceMappingURL=app.js.map