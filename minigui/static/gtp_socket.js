define(["require", "exports"], function (require, exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    class Socket {
        constructor() {
            this.sock = null;
            this.cmdQueue = [];
            this.handshakeComplete = false;
            this.connectCallback = null;
            this.lines = [];
            this.dataHandlers = [];
            this.textHandlers = [];
        }
        connect(uri) {
            this.sock = io.connect(uri);
            this.sock.on('json', (msg) => {
                let obj = JSON.parse(msg);
                if (obj.token != this.gameToken) {
                    console.log('ignoring', obj, `${obj.token} != ${this.gameToken}`);
                    return;
                }
                if (obj.stdout !== undefined) {
                    if (obj.stdout != '') {
                        this.lines.push(obj.stdout.trim());
                    }
                    else {
                        this.cmdHandler(this.lines.join('\n'));
                        this.lines = [];
                    }
                }
                else if (obj.stderr !== undefined) {
                    this.stderrHandler(obj.stderr);
                }
            });
            return new Promise((resolve) => {
                this.sock.on('connect', () => {
                    this.newSession();
                    this.send('boardsize 9')
                        .then(() => { resolve(9); })
                        .catch(() => {
                        this.send('boardsize 19').then(() => { resolve(19); });
                    });
                });
            });
        }
        onText(handler) {
            this.textHandlers.push(handler);
        }
        onData(prefix, handler) {
            this.dataHandlers.push({ prefix: prefix + ':', handler: handler });
        }
        send(cmd) {
            return new Promise((resolve, reject) => {
                this.cmdQueue.push({ cmd: cmd, resolve: resolve, reject: reject });
                if (this.cmdQueue.length == 1) {
                    this.sendNext();
                }
            });
        }
        newSession() {
            this.cmdQueue = [];
            let token = `session-id-${Date.now()}`;
            this.gameToken = token;
            this.send(`echo __NEW_TOKEN__ ${token}`);
        }
        cmdHandler(line) {
            let { cmd, resolve, reject } = this.cmdQueue[0];
            this.textHandler(`${cmd} ${line}`);
            if (line[0] == '=' || line[0] == '?') {
                this.cmdQueue = this.cmdQueue.slice(1);
                if (this.cmdQueue.length > 0) {
                    this.sendNext();
                }
            }
            let ok = line[0] == '=';
            let result = line.substr(1).trim();
            if (ok) {
                resolve(result);
            }
            else {
                reject(result);
            }
        }
        stderrHandler(line) {
            let handled = false;
            for (let { prefix, handler } of this.dataHandlers) {
                if (line.substr(0, prefix.length) == prefix) {
                    let stripped = line.substr(prefix.length);
                    let obj;
                    try {
                        obj = JSON.parse(stripped);
                    }
                    catch (e) {
                        obj = stripped;
                    }
                    handler(obj);
                    handled = true;
                }
            }
            if (!handled) {
                this.textHandler(line);
            }
        }
        textHandler(str) {
            for (let handler of this.textHandlers) {
                handler(str);
            }
        }
        sendNext() {
            let { cmd } = this.cmdQueue[0];
            if (this.textHandler) {
                this.textHandler(`${cmd}`);
            }
            this.sock.emit('gtpcmd', { data: cmd });
        }
    }
    exports.Socket = Socket;
});
//# sourceMappingURL=gtp_socket.js.map