define(["require", "exports", "./app", "./base", "./board", "./layer", "./log", "./winrate_graph", "./util", "./variation_tree"], function (require, exports, app_1, base_1, board_1, lyr, log_1, winrate_graph_1, util_1, variation_tree_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    class DemoApp extends app_1.App {
        constructor() {
            super();
            this.winrateGraph = new winrate_graph_1.WinrateGraph('winrate-graph');
            this.variationTree = new variation_tree_1.VariationTree('tree');
            this.log = new log_1.Log('log', 'console');
            this.connect().then(() => {
                this.mainBoard = new board_1.ClickableBoard('main-board', [lyr.Label, lyr.BoardStones, [lyr.Variation, 'pv'], lyr.Annotations]);
                this.mainBoard.enabled = true;
                this.readsBoard = new board_1.Board('reads-board', [lyr.BoardStones]);
                this.init([this.mainBoard, this.readsBoard]);
                this.mainBoard.onClick((p) => {
                    this.playMove(this.activePosition.toPlay, p).then(() => {
                        this.gtp.send('gamestate');
                    });
                });
                this.initButtons();
                this.log.onConsoleCmd((cmd) => {
                    this.gtp.send(cmd).then(() => { this.log.scroll(); });
                });
                this.gtp.onText((line) => { this.log.log(line, 'log-cmd'); });
                this.newGame();
                this.variationTree.onClick((positions) => {
                    this.gtp.send('clear_board');
                    this.activePosition = this.rootPosition;
                    for (let position of positions) {
                        if (position.lastMove != null) {
                            this.playMove(base_1.otherColor(position.toPlay), position.lastMove);
                            this.gtp.send('gamestate');
                        }
                    }
                });
            });
        }
        initButtons() {
            util_1.getElement('pass').addEventListener('click', () => {
                if (this.mainBoard.enabled) {
                    this.playMove(this.activePosition.toPlay, 'pass');
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
                        this.variationTree.draw();
                    });
                };
                reader.readAsText(files[0]);
            });
        }
        newGame() {
            this.gtp.send('prune_nodes 1');
            super.newGame();
            this.variationTree.newGame(this.rootPosition);
            this.gtp.send('prune_nodes 0');
            this.log.clear();
            this.winrateGraph.clear();
        }
        onGameState(msg) {
            if (msg.lastMove != null) {
                this.activePosition = this.activePosition.addChild(msg.lastMove, msg.stones);
            }
            this.updateBoards(msg);
            this.log.scroll();
            this.winrateGraph.setWinrate(msg.moveNum, msg.q);
            this.variationTree.draw();
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
            return this.gtp.send(`play ${colorStr} ${moveStr}`);
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