define(["require", "exports", "./app", "./board", "./layer", "./log", "./util", "./winrate_graph"], function (require, exports, app_1, board_1, lyr, log_1, util_1, winrate_graph_1) {
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
                        new lyr.Caption('search'),
                        new lyr.BoardStones(),
                        new lyr.Variation('search')
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
            super.newGame();
            this.log.clear();
            this.winrateGraph.clear();
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
            this.winrateGraph.update(position);
            this.log.scroll();
            if (this.activePosition.gameOver) {
                window.setTimeout(() => { this.newGame(); }, 3000);
            }
            else {
                this.gtp.send('genmove');
            }
        }
        onGameOver() {
            this.gtp.send('final_score').then((result) => {
                this.log.log(util_1.toPrettyResult(result));
                this.log.scroll();
            });
        }
    }
    new KioskApp();
});
//# sourceMappingURL=kiosk.js.map