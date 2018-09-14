// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Let's not worry about Socket.io type safety for now.
declare var io: any;

type DebugHandler = (msg: string) => void;
type CmdHandler = (result: string, ok: boolean) => void;
type StderrHandler = (obj: any) => void;
type ConnectCallback = () => void;

// The GtpSocket serializes all calls to send so that only one call is ever
// outstanding at a time. This isn't strictly necessary, but it makes reading
// the debug logs easier because we don't end up with request and result logs
// all jumbled up.
class Socket {
  private sock: any;
  private cmdQueue = new Array<{cmd: string, callback: CmdHandler | null}>();
  private gameToken: string;
  private handshakeComplete = false;
  private connectCallback: ConnectCallback | null = null;
  private lines = new Array<string>();

  private stderrHandlers = new Array<{prefix: string, handler: StderrHandler}>();

  constructor(uri: string, private debugHandler: DebugHandler) {
    this.sock = io.connect(uri)
    this.debugHandler = debugHandler;

    this.sock.on('connect', () => {
      this.newSession();
      if (this.connectCallback) {
        this.connectCallback();
      }
     });

    this.sock.on('json', (msg: string) => {
      let obj = JSON.parse(msg);
      if (obj.token != this.gameToken) {
        console.log('ignoring', obj, `${obj.token} != ${this.gameToken}`);
        return;
      }

      if (obj.stdout !== undefined) {
        if (obj.stdout != '') {
          this.lines.push(obj.stdout.trim());
        } else {
          this.cmdHandler(this.lines.join('\n'));
          this.lines = [];
        }
      } else if (obj.stderr !== undefined) {
        this.stderrHandler(obj.stderr);
      }
    });
  }

  onConnect(callback: ConnectCallback) {
    this.connectCallback = callback;
  }

  onData(prefix: string, handler: StderrHandler) {
    this.stderrHandlers.push({prefix: prefix + ':', handler: handler});
  }

  send(cmd: string, callback?: CmdHandler) {
    this.cmdQueue.push({cmd: cmd, callback: callback || null});
    if (this.cmdQueue.length == 1) {
      this.sendNext();
    }
  }

  newSession() {
    // Generates a new, unique session token and sends it to the server via
    // a special echo __NEW_TOKEN__ command. The server forwards this on to its
    // child Minigo process, which echos it back to the server after finishing
    // processing any other outstanding work. The server's stdout/stderr
    // handling logic in std_bg_thread looks for the __NEW_TOKEN__ string,
    // extracts the session token and then attaches that token to all subsequent
    // messages sent to the frontend.
    this.cmdQueue = [];
    let token = `session-id-${Date.now()}`;
    this.gameToken = token;
    this.send(`echo __NEW_TOKEN__ ${token}`);
  }

  private cmdHandler(line: string) {
    let {cmd, callback} = this.cmdQueue[0];

    if (this.debugHandler) {
      this.debugHandler(`${cmd} ${line}`);
    }

    if (line[0] == '=' || line[0] == '?') {
      // This line contains the response from a GTP command; pop the command off
      // the queue.
      this.cmdQueue = this.cmdQueue.slice(1);
      if (this.cmdQueue.length > 0) {
        this.sendNext();
      }
    }

    if (callback) {
      let ok = line[0] == '=';
      let result = line.substr(1).trim();
      callback(result, ok);
    }
  }

  private stderrHandler(line: string) {
    let handled = false;
    for (let {prefix, handler} of this.stderrHandlers) {
      if (line.substr(0, prefix.length) == prefix) {
        let stripped = line.substr(prefix.length);
        let obj;
        try {
          obj = JSON.parse(stripped);
        } catch (e) {
          obj = stripped;
        }
        handler(obj);
        handled = true;
      }
    }
    if (!handled && this.debugHandler) {
      this.debugHandler(line);
    }
  }

  private sendNext() {
    let {cmd, callback} = this.cmdQueue[0];
    if (this.debugHandler) {
      this.debugHandler(`${cmd}`);
    }
    this.sock.emit('gtpcmd', {data: cmd});
  }
}

export {
  CmdHandler,
  DebugHandler,
  Socket,
  StderrHandler,
};
