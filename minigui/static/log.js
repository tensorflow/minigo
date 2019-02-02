define(["require", "exports", "./util"], function (require, exports, util_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    class Log {
        constructor(logElemId, consoleElemId = null) {
            this.consoleElem = null;
            this.cmdHandler = null;
            this.scrollPending = false;
            this.logElem = util_1.getElement(logElemId);
            if (consoleElemId) {
                this.consoleElem = util_1.getElement(consoleElemId);
                this.consoleElem.addEventListener('keypress', (e) => {
                    let elem = this.consoleElem;
                    if (e.keyCode == 13) {
                        let cmd = elem.innerText.trim();
                        if (cmd != '' && this.cmdHandler) {
                            this.cmdHandler(cmd);
                        }
                        elem.innerHTML = '';
                        e.preventDefault();
                        return false;
                    }
                });
            }
        }
        log(msg, className = '') {
            let child;
            if (typeof msg == 'string') {
                if (msg == '') {
                    msg = ' ';
                }
                child = document.createElement('div');
                child.innerText = msg;
                if (className != '') {
                    child.className = className;
                }
            }
            else {
                child = msg;
            }
            this.logElem.appendChild(child);
        }
        clear() {
            this.logElem.innerHTML = '';
        }
        scroll() {
            if (!this.scrollPending) {
                this.scrollPending = true;
                window.requestAnimationFrame(() => {
                    this.scrollPending = false;
                    if (this.logElem.lastElementChild) {
                        this.logElem.lastElementChild.scrollIntoView();
                    }
                });
            }
        }
        onConsoleCmd(cmdHandler) {
            this.cmdHandler = cmdHandler;
        }
        get hasFocus() {
            return this.consoleElem && document.activeElement == this.consoleElem;
        }
        focus() {
            if (this.consoleElem) {
                this.consoleElem.focus();
            }
        }
        blur() {
            if (this.consoleElem) {
                this.consoleElem.blur();
            }
        }
    }
    exports.Log = Log;
});
//# sourceMappingURL=log.js.map