define(["require", "exports", "./gtp_socket", "./base", "./board", "./util"], function (require, exports, gtp_socket_1, base_1, board_1, util) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    class Position {
        constructor(moveNum, stones, lastMove, toPlay) {
            this.moveNum = moveNum;
            this.stones = stones;
            this.lastMove = lastMove;
            this.toPlay = toPlay;
            this.search = [];
            this.pv = [];
            this.n = null;
            this.dq = null;
            this.annotations = [];
            if (lastMove != null && lastMove != 'pass' && lastMove != 'resign') {
                this.annotations.push({
                    p: lastMove,
                    shape: board_1.Annotation.Shape.Dot,
                    color: '#ef6c02',
                });
            }
        }
    }
    exports.Position = Position;
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
            return this.gtp.connect(uri).then((size) => {
                this.size = size;
            });
        }
        init(boards) {
            this.boards = boards;
        }
        newGame() {
            this.activePosition = new Position(0, util.emptyBoard(this.size), null, base_1.Color.Black);
            this.positionHistory = [this.activePosition];
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
            msg.search = util.parseMoves(msg.search, this.size);
            msg.toPlay = util.parseGtpColor(msg.toPlay);
            if (msg.pv) {
                msg.pv = util.parseMoves(msg.pv, this.size);
            }
            const props = ['n', 'dq', 'pv', 'search'];
            util.partialUpdate(msg, this.positionHistory[msg.moveNum], props);
            if (msg.moveNum == this.activePosition.moveNum) {
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
            let lastMove = msg.lastMove ? util.parseGtpMove(msg.lastMove, this.size) : null;
            let position = new Position(msg.moveNum, stones, lastMove, this.toPlay);
            this.positionHistory[msg.moveNum] = position;
            if (msg.moveNum > this.activePosition.moveNum) {
                this.activePosition = position;
                this.updateBoards(this.activePosition);
            }
            if (this.gameOver) {
                this.onGameOver();
            }
        }
    }
    exports.App = App;
});
//# sourceMappingURL=app.js.map