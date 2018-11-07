define(["require", "exports", "./app", "./base", "./board", "./layer", "./log", "./position", "./util", "./variation_tree", "./winrate_graph"], function (require, exports, app_1, base_1, board_1, lyr, log_1, position_1, util_1, variation_tree_1, winrate_graph_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    class ExploreBoard extends board_1.ClickableBoard {
        constructor(parentId, gtp) {
            super(parentId, []);
            this.gtp = gtp;
            this._showSearch = true;
            this.qLayer = new lyr.Q();
            this.variationLayer = new lyr.Variation('pv');
            this.nextLayer = new lyr.Annotations('annotations', [position_1.Annotation.Shape.DashedCircle]);
            this.addLayers([
                new lyr.Label(),
                new lyr.BoardStones(),
                this.qLayer,
                this.variationLayer,
                this.nextLayer,
                new lyr.Annotations('annotations', [position_1.Annotation.Shape.Dot])
            ]);
            this.variationLayer.show = false;
            this.enabled = true;
            this.ctx.canvas.addEventListener('mousemove', (e) => {
                let p = this.canvasToBoard(e.offsetX, e.offsetY, 0.45);
                if (p != null) {
                    if (this.getStone(p) != base_1.Color.Empty || !this.qLayer.hasPoint(p)) {
                        p = null;
                    }
                }
                this.showVariation(p);
            });
            this.ctx.canvas.addEventListener('mouseleave', () => {
                this.showVariation(null);
            });
            this.onClick((p) => {
                if (this.variationLayer.childVariation != null) {
                    this.gtp.send('variation');
                }
                this.variationLayer.clear();
                this.variationLayer.show = false;
                this.qLayer.clear();
                this.qLayer.show = true;
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
        get showNext() {
            return this.nextLayer.show;
            ;
        }
        set showNext(x) {
            if (x != this.nextLayer.show) {
                this.nextLayer.show = x;
                this.draw();
            }
        }
        showVariation(p) {
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
        }
    }
    class ExploreApp extends app_1.App {
        constructor() {
            super();
            this.winrateGraph = new winrate_graph_1.WinrateGraph('winrate-graph');
            this.variationTree = new variation_tree_1.VariationTree('tree');
            this.log = new log_1.Log('log', 'console');
            this.showSearch = true;
            this.showNext = true;
            this.showConsole = false;
            this.pendingSelectPosition = null;
            this.connect().then(() => {
                this.board = new ExploreBoard('main-board', this.gtp);
                this.board.onClick((p) => {
                    this.playMove(this.activePosition.toPlay, p);
                });
                this.init([this.board]);
                this.initButtons();
                this.log.onConsoleCmd((cmd) => {
                    this.gtp.send(cmd).then(() => { this.log.scroll(); });
                });
                this.gtp.onText((line) => {
                    this.log.log(line, 'log-cmd');
                    if (this.showConsole) {
                        this.log.scroll();
                    }
                });
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
                window.addEventListener('keydown', (e) => {
                    if (e.key == 'Escape') {
                        this.showConsole = !this.showConsole;
                        let containerElem = util_1.getElement('log-container');
                        containerElem.style.top = this.showConsole ? '0' : '-40vh';
                        if (this.showConsole) {
                            this.log.scroll();
                        }
                        else {
                            this.log.blur();
                        }
                        e.preventDefault();
                        return false;
                    }
                    if (this.log.hasFocus) {
                        return;
                    }
                    switch (e.key) {
                        case 'ArrowUp':
                        case 'ArrowLeft':
                            this.selectPrevPosition();
                            break;
                        case 'ArrowRight':
                        case 'ArrowDown':
                            this.selectNextPosition();
                            break;
                    }
                });
                window.addEventListener('wheel', (e) => {
                    if (e.deltaY < 0) {
                        this.selectPrevPosition();
                    }
                    else if (e.deltaY > 0) {
                        this.selectNextPosition();
                    }
                });
            });
        }
        initButtons() {
            util_1.getElement('toggle-search').addEventListener('click', (e) => {
                this.showSearch = !this.showSearch;
                this.board.showSearch = this.showSearch;
                if (this.showSearch) {
                    e.target.innerText = 'Hide search';
                }
                else {
                    e.target.innerText = 'Show search';
                }
            });
            util_1.getElement('toggle-variation').addEventListener('click', (e) => {
                this.showNext = !this.showNext;
                this.board.showNext = this.showNext;
                if (this.showNext) {
                    e.target.innerText = 'Hide variation';
                }
                else {
                    e.target.innerText = 'Show variation';
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
                    this.board.showSearch = false;
                    this.gtp.send('ponder 0');
                    this.gtp.send(`playsgf ${sgf}`).finally(() => {
                        this.board.enabled = true;
                        this.board.showSearch = this.showSearch;
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
        selectNextPosition() {
            if (this.activePosition.children.length > 0) {
                this.selectPosition(this.activePosition.children[0]);
            }
        }
        selectPrevPosition() {
            if (this.activePosition.parent != null) {
                this.selectPosition(this.activePosition.parent);
            }
        }
        selectPosition(position) {
            this.activePosition = position;
            this.updateBoards(position);
            this.winrateGraph.setWinrate(position.moveNum, position.q);
            if (position.parent != null) {
                this.variationTree.addChild(position.parent, position);
            }
            let impl = (position) => {
                if (this.pendingSelectPosition == null) {
                    this.gtp.send(`select_position ${position.id}`).then((id) => {
                        if (this.pendingSelectPosition == null) {
                            throw new Error('pending select position is null');
                        }
                        if (id != this.pendingSelectPosition.id) {
                            impl(this.pendingSelectPosition);
                        }
                        else {
                            this.pendingSelectPosition = null;
                        }
                    });
                }
                this.pendingSelectPosition = position;
            };
            impl(position);
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
            this.activePosition = position;
            this.updateBoards(position);
            this.winrateGraph.setWinrate(position.moveNum, position.q);
            if (position.parent != null) {
                this.variationTree.addChild(position.parent, position);
            }
        }
        playMove(color, move) {
            let colorStr = color == base_1.Color.Black ? 'b' : 'w';
            let moveStr = base_1.toKgs(move);
            this.board.enabled = false;
            this.gtp.send(`play ${colorStr} ${moveStr}`).then(() => {
                this.gtp.send('gamestate');
            }).finally(() => {
                this.board.enabled = true;
            });
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