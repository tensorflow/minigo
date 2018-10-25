define(["require", "exports", "./position", "./gtp_socket", "./base", "./util"], function (require, exports, position_1, gtp_socket_1, base_1, util) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.Position = position_1.Position;
    class App {
        constructor() {
            this.gtp = new gtp_socket_1.Socket();
            this.engineBusy = false;
            this.gameOver = true;
            this.toPlay = base_1.Color.Black;
            this.boards = [];
            this.gtp.onData('mg-search', this.onSearch.bind(this));
            this.gtp.onData('mg-gamestate', this.onGameState.bind(this));
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
            this.latestPosition = this.rootPosition;
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
            msg.search = util.parseMoves(msg.search, base_1.N);
            msg.toPlay = util.parseGtpColor(msg.toPlay);
            if (msg.pv) {
                msg.pv = util.parseMoves(msg.pv, base_1.N);
            }
            if (msg.moveNum != this.latestPosition.moveNum) {
                throw new Error(`Got a search msg for move ${msg.moveNum} but latest is ` +
                    `${this.latestPosition.moveNum}`);
            }
            const props = ['n', 'dq', 'pv', 'search'];
            util.partialUpdate(msg, this.latestPosition, props);
            if (this.activePosition.moveNum == msg.moveNum) {
                this.updateBoards(msg);
            }
        }
        onGameState(msg) {
            let stoneMap = {
                '.': base_1.Color.Empty,
                'X': base_1.Color.Black,
                'O': base_1.Color.White,
            };
            let stones = [];
            for (let i = 0; i < msg.board.length; ++i) {
                stones.push(stoneMap[msg.board[i]]);
            }
            this.toPlay = util.parseGtpColor(msg.toPlay);
            this.gameOver = msg.gameOver;
            let lastMove = msg.lastMove ? util.parseGtpMove(msg.lastMove, base_1.N) : null;
            if (lastMove == null) {
                if (msg.moveNum != 0) {
                    throw new Error(`moveNum == ${msg.moveNum} but don't have a lastMove`);
                }
                stones.forEach((color) => {
                    if (color != base_1.Color.Empty) {
                        throw new Error(`board isn't empty but don't have a lastMove`);
                    }
                });
            }
            else {
                if (msg.moveNum != this.latestPosition.moveNum + 1) {
                    throw new Error(`Expected game state for move ${this.latestPosition.moveNum + 1} ` +
                        `but got ${msg.moveNum}`);
                }
                this.latestPosition = this.latestPosition.addChild(lastMove, stones);
                if (this.activePosition == this.latestPosition.parent) {
                    this.activePosition = this.latestPosition;
                    this.updateBoards(this.activePosition);
                }
            }
            if (this.gameOver) {
                this.onGameOver();
            }
        }
    }
    exports.App = App;
});
//# sourceMappingURL=app.js.map