define(["require", "exports", "./app", "./base", "./board", "./layer", "./log", "./util", "./variation_tree", "./winrate_graph"], function (require, exports, app_1, base_1, board_1, lyr, log_1, util_1, variation_tree_1, winrate_graph_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    class ExploreBoard extends board_1.ClickableBoard {
        constructor(parentId, gtp) {
            super(parentId, []);
            this.gtp = gtp;
            this._showSearch = false;
            this.qLayer = new lyr.Q();
            this.variationLayer = new lyr.Variation('pv');
            this.addLayers([
                new lyr.Label(),
                new lyr.BoardStones(),
                this.qLayer,
                this.variationLayer,
                new lyr.Annotations()
            ]);
            this.variationLayer.show = false;
            this.enabled = true;
            this.ctx.canvas.addEventListener('mousemove', (e) => {
                let p = this.canvasToBoard(e.offsetX, e.offsetY, 0.45);
                if (p != null) {
                    if (this.getStone(p) != base_1.Color.Empty ||
                        !this.qLayer.hasPoint(p)) {
                        p = null;
                    }
                }
                if (base_1.movesEqual(p, this.variationLayer.childVariation)) {
                    return;
                }
                this.variationLayer.clear();
                this.variationLayer.show = p != null;
                this.qLayer.show = p == null;
                if (p != null) {
                    this.gtp.send(`variation ${base_1.toKgs(p)}`);
                }
                else {
                    this.gtp.send('variation');
                }
                this.variationLayer.childVariation = p;
            });
            this.onClick((p) => {
                this.variationLayer.clear();
                this.variationLayer.show = false;
                this.qLayer.clear();
                this.qLayer.show = true;
                this.gtp.send('variation');
            });
        }
        get showSearch() {
            return this._showSearch;
        }
        set showSearch(x) {
            if (x != this._showSearch) {
                this._showSearch = x;
                if (x) {
                    this.variationLayer.show = false;
                    this.qLayer.show = true;
                }
                else {
                    this.variationLayer.show = false;
                    this.qLayer.show = false;
                }
                this.draw();
            }
        }
    }
    class ExploreApp extends app_1.App {
        constructor() {
            super();
            this.winrateGraph = new winrate_graph_1.WinrateGraph('winrate-graph');
            this.variationTree = new variation_tree_1.VariationTree('tree');
            this.log = new log_1.Log('log', 'console');
            this.connect().then(() => {
                this.board = new ExploreBoard('main-board', this.gtp);
                this.board.onClick((p) => {
                    this.playMove(this.activePosition.toPlay, p).then(() => {
                        let parent = this.activePosition;
                        this.board.enabled = false;
                        this.gtp.send('gamestate').then(() => {
                            this.variationTree.addChild(parent, this.activePosition);
                        }).finally(() => {
                            this.board.enabled = true;
                        });
                    });
                });
                this.init([this.board]);
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
            util_1.getElement('toggle-pv').addEventListener('click', (e) => {
                this.board.showSearch = !this.board.showSearch;
                if (!this.board.showSearch) {
                    e.target.innerText = 'Show search';
                }
                else {
                    e.target.innerText = 'Hide search';
                }
            });
            util_1.getElement('load-sgf-input').addEventListener('change', (e) => {
                let files = Array.prototype.slice.call(e.target.files);
                if (files.length != 1) {
                    return;
                }
                let reader = new FileReader();
                reader.onload = () => {
                    this.board.clear();
                    this.newGame();
                    let sgf = reader.result.replace(/\n/g, '\\n');
                    this.board.enabled = false;
                    this.gtp.send('ponder 0');
                    this.gtp.send(`playsgf ${sgf}`).finally(() => {
                        this.board.enabled = true;
                        this.gtp.send('ponder 1');
                    });
                };
                reader.readAsText(files[0]);
            });
            util_1.getElement('main-line').addEventListener('click', () => {
                let position = this.activePosition;
                while (position != this.rootPosition &&
                    !position.isMainline && position.parent != null) {
                    position = position.parent;
                }
                if (position != this.activePosition) {
                    this.activePosition = position;
                    this.variationTree.setActive(this.activePosition);
                    this.updateBoards(this.activePosition);
                }
            });
        }
        newGame() {
            this.gtp.send('prune_nodes 1');
            super.newGame();
            this.gtp.send('prune_nodes 0');
            this.variationTree.newGame(this.rootPosition);
            this.log.clear();
            this.winrateGraph.clear();
        }
        onPosition(position) {
            if (position.parent == this.activePosition) {
                this.activePosition = position;
            }
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
    new ExploreApp();
});
//# sourceMappingURL=explore.js.map