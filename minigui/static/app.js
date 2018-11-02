define(["require", "exports", "./position", "./gtp_socket", "./base", "./util"], function (require, exports, position_1, gtp_socket_1, base_1, util) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.Position = position_1.Position;
    class SearchMsg {
        constructor(j) {
            this.n = null;
            this.dq = null;
            this.pv = null;
            this.moveNum = j.moveNum;
            this.toPlay = util.parseGtpColor(j.toPlay);
            this.search = util.parseMoves(j.search, base_1.N);
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
    class GameStateMsg {
        constructor(j) {
            this.stones = [];
            this.lastMove = null;
            const stoneMap = {
                '.': base_1.Color.Empty,
                'X': base_1.Color.Black,
                'O': base_1.Color.White,
            };
            for (let i = 0; i < j.board.length; ++i) {
                this.stones.push(stoneMap[j.board[i]]);
            }
            this.toPlay = util.parseGtpColor(j.toPlay);
            if (j.lastMove) {
                this.lastMove = util.parseGtpMove(j.lastMove, base_1.N);
            }
            this.moveNum = j.moveNum;
            this.q = j.q;
            this.gameOver = j.gameOver;
        }
    }
    exports.GameStateMsg = GameStateMsg;
    class App {
        constructor() {
            this.gtp = new gtp_socket_1.Socket();
            this.engineBusy = false;
            this.gameOver = true;
            this.boards = [];
            this.gtp.onData('mg-search', (j) => {
                this.onSearch(new SearchMsg(j));
            });
            this.gtp.onData('mg-gamestate', (j) => {
                let msg = new GameStateMsg(j);
                this.gameOver = msg.gameOver;
                this.onGameState(msg);
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
            this.rootPosition = new position_1.Position(null, 0, util.emptyBoard(), null, base_1.Color.Black);
            this.activePosition = this.rootPosition;
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
            const props = ['n', 'dq', 'pv', 'search'];
            util.partialUpdate(msg, this.activePosition, props);
            if (this.activePosition.moveNum == msg.moveNum) {
                this.updateBoards(msg);
            }
        }
    }
    exports.App = App;
});
//# sourceMappingURL=app.js.map