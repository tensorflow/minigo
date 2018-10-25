define(["require", "exports", "./app", "./board", "./heat_map", "./layer", "./log", "./util", "./winrate_graph"], function (require, exports, app_1, board_1, heat_map_1, lyr, log_1, util_1, winrate_graph_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    class KioskApp extends app_1.App {
        constructor() {
            super();
            this.winrateGraph = new winrate_graph_1.WinrateGraph('winrate-graph');
            this.log = new log_1.Log('log');
            this.connect().then(() => {
                let mainBoard = new board_1.Board('main-board', [lyr.Label, lyr.BoardStones, [lyr.Variation, 'pv'], lyr.Annotations]);
                let searchBoard = new board_1.Board('search-board', [[lyr.Caption, 'search'], lyr.BoardStones, [lyr.Variation, 'search']]);
                let nBoard = new board_1.Board('n-board', [[lyr.Caption, 'N'], [lyr.HeatMap, 'n', heat_map_1.heatMapN], lyr.BoardStones]);
                let dqBoard = new board_1.Board('dq-board', [[lyr.Caption, 'ΔQ'], [lyr.HeatMap, 'dq', heat_map_1.heatMapDq], lyr.BoardStones]);
                this.init([mainBoard, searchBoard, nBoard, dqBoard]);
                this.gtp.onText((line) => { this.log.log(line, 'log-cmd'); });
                this.newGame();
            });
        }
        newGame() {
            super.newGame();
            this.log.clear();
            this.winrateGraph.clear();
        }
        onGameState(msg) {
            super.onGameState(msg);
            this.log.scroll();
            this.winrateGraph.setWinrate(msg.moveNum, msg.q);
            if (this.gameOver) {
                window.setTimeout(() => { this.newGame(); }, 3000);
            }
            else {
                this.gtp.send('genmove').then((move) => {
                    this.gtp.send('gamestate');
                });
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