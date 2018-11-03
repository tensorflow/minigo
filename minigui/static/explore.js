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
                this.pvLayer = this.mainBoard.getLayer(2);
                this.readsBoard = new board_1.Board('reads-board', [lyr.BoardStones, lyr.BestMoves]);
                this.bestMovesLayer = this.readsBoard.getLayer(1);
                this.init([this.mainBoard, this.readsBoard]);
                this.mainBoard.onClick((p) => {
                    this.bestMovesLayer.clear();
                    this.playMove(this.activePosition.toPlay, p).then(() => {
                        let parent = this.activePosition;
                        this.gtp.send('gamestate').then(() => {
                            this.variationTree.addChild(parent, this.activePosition);
                        });
                    });
                });
                this.initButtons();
                this.log.onConsoleCmd((cmd) => {
                    this.gtp.send(cmd).then(() => { this.log.scroll(); });
                });
                this.gtp.onText((line) => { this.log.log(line, 'log-cmd'); });
                this.newGame();
                this.variationTree.onClick((positions) => {
                    let moves = [];
                    for (let position of positions) {
                        if (position.lastMove != null) {
                            moves.push(base_1.toKgs(position.lastMove));
                        }
                    }
                    this.gtp.send('clear_board');
                    if (moves.length > 0) {
                        this.gtp.send(`play_multiple b ${moves.join(' ')}`);
                    }
                    this.gtp.send('gamestate');
                    this.activePosition = positions[positions.length - 1];
                });
            });
        }
        initButtons() {
            util_1.getElement('pass').addEventListener('click', () => {
                if (this.mainBoard.enabled) {
                    this.playMove(this.activePosition.toPlay, 'pass');
                }
            });
            util_1.getElement('toggle-pv').addEventListener('click', (e) => {
                this.pvLayer.hidden = !this.pvLayer.hidden;
                if (this.pvLayer.hidden) {
                    e.target.innerText = 'Show PV';
                }
                else {
                    e.target.innerText = 'Hide PV';
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
        onPosition(position) {
            super.onPosition(position);
            this.updateBoards(position);
            this.log.scroll();
            this.winrateGraph.setWinrate(position.moveNum, position.q);
            if (position.parent != null) {
                this.variationTree.addChild(position.parent, position);
            }
        }
        playMove(color, move) {
            let colorStr = color == base_1.Color.Black ? 'b' : 'w';
            let moveStr = base_1.toKgs(move);
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