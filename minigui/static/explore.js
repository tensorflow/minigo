define(["require", "exports", "./app", "./base", "./board", "./layer", "./log", "./winrate_graph", "./util", "./variation_tree"], function (require, exports, app_1, base_1, board_1, lyr, log_1, winrate_graph_1, util_1, variation_tree_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    const HUMAN = 'Human';
    const MINIGO = 'Minigo';
    variation_tree_1.testVariationTree();
    class DemoApp extends app_1.App {
        constructor() {
            super();
            this.playerElems = [];
            this.winrateGraph = new winrate_graph_1.WinrateGraph('winrate-graph');
            this.log = new log_1.Log('log', 'console');
            this.connect().then(() => {
                this.mainBoard = new board_1.ClickableBoard('main-board', [lyr.Label, lyr.BoardStones, [lyr.Variation, 'pv'], lyr.Annotations]);
                this.mainBoard.enabled = true;
                this.readsBoard = new board_1.Board('reads-board', [lyr.BoardStones]);
                this.init([this.mainBoard, this.readsBoard]);
                this.mainBoard.onClick((p) => {
                    this.playMove(this.toPlay, p);
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
                    this.playMove(this.toPlay, 'pass');
                }
            });
            util_1.getElement('load-sgf-input').addEventListener('change', (e) => {
                let files = Array.prototype.slice.call(e.target.files);
                if (files.length != 1) {
                    let names = [];
                    files.forEach((f) => { names.push(`"${f.name}"`); });
                    throw new Error(`Expected one file, got [${names.join(', ')}]`);
                }
                let reader = new FileReader();
                reader.onload = () => {
                    this.newGame();
                    let sgf = reader.result.replace(/\n/g, '\\n');
                    this.gtp.send(`playsgf ${sgf}`).then(() => {
                        console.log('ok!');
                    });
                };
                reader.readAsText(files[0]);
            });
        }
        newGame() {
            super.newGame();
            this.log.clear();
            this.winrateGraph.clear();
        }
        onPlayerChanged() {
            if (this.engineBusy || this.gameOver) {
                return;
            }
        }
        onGameState(msg) {
            super.onGameState(msg);
            this.log.scroll();
            this.winrateGraph.setWinrate(msg.moveNum, msg.q);
            this.onPlayerChanged();
        }
        playMove(color, move) {
            let colorStr = color == base_1.Color.Black ? 'b' : 'w';
            let moveStr;
            if (move == 'pass') {
                moveStr = move;
            }
            else if (move == 'resign') {
                throw new Error('resign not yet supported');
            }
            else {
                let row = base_1.N - move.row;
                let col = base_1.COL_LABELS[move.col];
                moveStr = `${col}${row}`;
            }
            this.gtp.send(`play ${colorStr} ${moveStr}`).then(() => {
                this.gtp.send('gamestate');
            });
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
//# sourceMappingURL=explore.js.map