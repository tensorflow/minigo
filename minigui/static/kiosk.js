define(["require", "exports", "./app", "./base", "./board", "./layer", "./log", "./util", "./winrate_graph"], function (require, exports, app_1, base_1, board_1, lyr, log_1, util_1, winrate_graph_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    class KioskApp extends app_1.App {
        constructor() {
            super();
            this.winrateGraph = new winrate_graph_1.WinrateGraph('winrate-graph');
            this.log = new log_1.Log('log');
            this.boards = [];
            this.connect().then(() => {
                this.boards = [
                    new board_1.Board('main-board', this.rootPosition, [
                        new lyr.Label(),
                        new lyr.BoardStones(),
                        new lyr.Variation('pv'),
                        new lyr.Annotations()
                    ]),
                    new board_1.Board('search-board', this.rootPosition, [
                        new lyr.Caption('live'),
                        new lyr.BoardStones(),
                        new lyr.Variation('live')
                    ]),
                    new board_1.Board('n-board', this.rootPosition, [
                        new lyr.Caption('N'),
                        new lyr.VisitCountHeatMap(),
                        new lyr.BoardStones()
                    ]),
                    new board_1.Board('dq-board', this.rootPosition, [
                        new lyr.Caption('ΔQ'),
                        new lyr.DeltaQHeatMap(),
                        new lyr.BoardStones()
                    ]),
                ];
                this.gtp.onText((line) => { this.log.log(line, 'log-cmd'); });
                this.newGame();
            });
        }
        newGame() {
            this.log.clear();
            this.winrateGraph.newGame();
            return super.newGame().then(() => {
                this.genmove();
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
        genmove() {
            let colorStr = this.activePosition.toPlay == base_1.Color.Black ? 'b' : 'w';
            this.gtp.send(`genmove ${colorStr}`).then((gtpMove) => {
                if (gtpMove == 'resign') {
                    this.onGameOver();
                }
                else {
                    this.genmove();
                }
            });
        }
        onNewPosition(position) {
            this.activePosition = position;
            for (let board of this.boards) {
                board.setPosition(position);
            }
            this.winrateGraph.setActive(position);
            this.log.scroll();
            if (position.gameOver) {
                this.onGameOver();
            }
        }
        onGameOver() {
            this.gtp.send('final_score').then((result) => {
                this.log.log(util_1.toPrettyResult(result));
                this.log.scroll();
                window.setTimeout(() => { this.newGame(); }, 3000);
            });
        }
    }
    new KioskApp();
});
//# sourceMappingURL=kiosk.js.map