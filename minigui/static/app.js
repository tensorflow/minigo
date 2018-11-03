define(["require", "exports", "./position", "./gtp_socket", "./base", "./util"], function (require, exports, position_1, gtp_socket_1, base_1, util) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.Position = position_1.Position;
    class SearchMsg {
        constructor(j) {
            this.n = null;
            this.dq = null;
            this.pv = null;
            this.id = j.id;
            this.moveNum = j.moveNum;
            this.toPlay = util.parseGtpColor(j.toPlay);
            this.search = util.parseMoves(j.search, base_1.N);
            this.childQ = j.childQ;
            if (j.n) {
                this.n = j.n;
            }
            if (j.dq) {
                this.dq = j.dq;
            }
            if (j.pv) {
                this.pv = util.parseMoves(j.pv, base_1.N);
            }
        }
    }
    exports.SearchMsg = SearchMsg;
    class App {
        constructor() {
            this.gtp = new gtp_socket_1.Socket();
            this.engineBusy = false;
            this.gameOver = true;
            this.boards = [];
            this.positionMap = new Map();
            this.gtp.onData('mg-search', (j) => {
                this.onSearch(new SearchMsg(j));
            });
            this.gtp.onData('mg-gamestate', (j) => {
                let position = this.parseGameState(j);
                this.gameOver = j.gameOver;
                this.onPosition(position);
                if (j.gameOver) {
                    this.onGameOver();
                }
            });
        }
        connect() {
            let uri = `http://${document.domain}:${location.port}/minigui`;
            return this.gtp.connect(uri).then((size) => { base_1.setBoardSize(size); });
        }
        init(boards) {
            this.boards = boards;
        }
        newGame() {
            this.rootPosition = new position_1.Position(null, 0, util.emptyBoard(), 0, null, base_1.Color.Black);
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
        updateBoards(state) {
            for (let board of this.boards) {
                if (board.update(state)) {
                    board.draw();
                }
            }
        }
        onSearch(msg) {
            let position = this.positionMap.get(msg.id);
            if (!position) {
                return;
            }
            const props = ['n', 'dq', 'pv', 'search', 'childQ'];
            util.partialUpdate(msg, position, props);
            if (position == this.activePosition) {
                this.updateBoards(position);
            }
        }
        onPosition(position) {
            if (position.parent == this.activePosition ||
                position.parent == null && this.activePosition == this.rootPosition) {
                this.activePosition = position;
            }
        }
        parseGameState(j) {
            const stoneMap = {
                '.': base_1.Color.Empty,
                'X': base_1.Color.Black,
                'O': base_1.Color.White,
            };
            let stones = [];
            for (let i = 0; i < j.board.length; ++i) {
                stones.push(stoneMap[j.board[i]]);
            }
            let toPlay = util.parseGtpColor(j.toPlay);
            let lastMove = null;
            if (j.lastMove) {
                lastMove = util.parseGtpMove(j.lastMove, base_1.N);
            }
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
                position = parent.addChild(lastMove, stones, j.q);
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