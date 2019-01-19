define(["require", "exports", "./app", "./base", "./board", "./layer", "./log", "./util", "./winrate_graph"], function (require, exports, app_1, base_1, board_1, lyr, log_1, util_1, winrate_graph_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    const HUMAN = 'Human';
    const MINIGO = 'Minigo';
    class DemoApp extends app_1.App {
        constructor() {
            super();
            this.playerElems = [];
            this.winrateGraph = new winrate_graph_1.WinrateGraph('winrate-graph');
            this.log = new log_1.Log('log', 'console');
            this.boards = [];
            this.connect().then(() => {
                this.pvLayer = new lyr.Variation('pv');
                this.mainBoard = new board_1.ClickableBoard('main-board', this.rootPosition, [
                    new lyr.Label(),
                    new lyr.BoardStones(),
                    this.pvLayer,
                    new lyr.Annotations()
                ]);
                this.boards = [this.mainBoard];
                let searchElem = util_1.getElement('search-board');
                if (searchElem) {
                    this.boards.push(new board_1.Board(searchElem, this.rootPosition, [
                        new lyr.Caption('search'),
                        new lyr.BoardStones(),
                        new lyr.Variation('search')
                    ]));
                }
                let nElem = util_1.getElement('n-board');
                if (nElem) {
                    this.boards.push(new board_1.Board(nElem, this.rootPosition, [
                        new lyr.Caption('N'),
                        new lyr.VisitCountHeatMap(),
                        new lyr.BoardStones()
                    ]));
                }
                let dqElem = util_1.getElement('dq-board');
                if (dqElem) {
                    this.boards.push(new board_1.Board('dq-board', this.rootPosition, [
                        new lyr.Caption('ΔQ'),
                        new lyr.DeltaQHeatMap(),
                        new lyr.BoardStones()
                    ]));
                }
                this.mainBoard.onClick((p) => {
                    this.playMove(this.activePosition.toPlay, p);
                });
                this.initButtons();
                this.log.onConsoleCmd((cmd) => {
                    this.gtp.send(cmd).then(() => { this.log.scroll(); });
                });
                this.gtp.onText((line) => { this.log.log(line, 'log-cmd'); });
                this.newGame();
            });
        }
        initButtons() {
            util_1.getElement('pass').addEventListener('click', () => {
                if (this.mainBoard.enabled) {
                    this.playMove(this.activePosition.toPlay, 'pass');
                }
            });
            util_1.getElement('reset').addEventListener('click', () => {
                this.gtp.newSession();
                this.newGame();
            });
            let initPlayerButton = (color, elemId) => {
                let elem = util_1.getElement(elemId);
                this.playerElems[color] = elem;
                elem.addEventListener('click', () => {
                    if (elem.innerText == HUMAN) {
                        elem.innerText = MINIGO;
                    }
                    else {
                        elem.innerText = HUMAN;
                    }
                    if (!this.engineBusy) {
                        this.onPlayerChanged();
                    }
                });
            };
            initPlayerButton(base_1.Color.Black, 'black-player');
            initPlayerButton(base_1.Color.White, 'white-player');
        }
        newGame() {
            super.newGame();
            this.log.clear();
            this.winrateGraph.newGame(this.rootPosition);
        }
        onPlayerChanged() {
            if (this.activePosition.gameOver) {
                return;
            }
            if (this.playerElems[this.activePosition.toPlay].innerText == MINIGO) {
                this.genmove();
            }
            else {
                this.mainBoard.enabled = true;
                this.pvLayer.show = false;
            }
        }
        genmove() {
            if (this.activePosition.gameOver || this.engineBusy) {
                return;
            }
            this.mainBoard.enabled = false;
            this.pvLayer.show = true;
            this.engineBusy = true;
            let colorStr = this.activePosition.toPlay == base_1.Color.Black ? 'b' : 'w';
            this.gtp.send(`genmove ${colorStr}`).finally(() => {
                this.engineBusy = false;
                if (this.playerElems[this.activePosition.toPlay].innerText == MINIGO) {
                    this.genmove();
                }
            });
        }
        onPositionUpdate(position, update) {
            if (position != this.activePosition) {
                return;
            }
            for (let board of this.boards) {
                board.update(update);
            }
            this.winrateGraph.update(position);
        }
        onNewPosition(position) {
            this.activePosition = position;
            for (let board of this.boards) {
                board.setPosition(position);
            }
            this.winrateGraph.setActive(position);
            this.winrateGraph.update(position);
            this.log.scroll();
            this.onPlayerChanged();
        }
        playMove(color, move) {
            let colorStr = color == base_1.Color.Black ? 'b' : 'w';
            let moveStr = base_1.toKgs(move);
            this.gtp.send(`play ${colorStr} ${moveStr}`);
        }
        onGameOver() {
            this.gtp.send('final_score').then((result) => {
                this.log.log(util_1.toPrettyResult(result));
                this.log.scroll();
            });
        }
    }
    new DemoApp();
});
//# sourceMappingURL=demo.js.map