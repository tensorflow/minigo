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

import {Nullable} from './base';

type TextHandler = (msg: string) => void;
type CmdHandler = (result: string, ok: boolean) => void;
type DataHandler = (obj: any) => void;
type ConnectCallback = () => void;

function trimText(str: string, len: number) {
  if (str.length > len) {
    return `${str.substr(0, len - 3)}...`;
  }
  return str;
}

// The GtpSocket serializes all calls to send so that only one call is ever
// outstanding at a time. This isn't strictly necessary, but it makes reading
// the debug logs easier because we don't end up with request and result logs
// all jumbled up.
class Socket {
  private sock: any = null;
  private cmdQueue: {cmd: string, resolve: any, reject: any}[] = [];
  private token: string;
  private handshakeComplete = false;
  private connectCallback: Nullable<ConnectCallback> = null;
  private lines: string[] = [];

  private dataHandlers: {prefix: string, handler: DataHandler}[] = [];
  private textHandlers: TextHandler[] = [];
  private playerName = "";
  private debug = false;

  // Connects to the Minigui server at the given URI.
  // Returns a promise that gets resolved of the board size when the connection
  // is established.
  connect(uri: string, playerName: string, debug=false) {
    this.playerName = playerName;
    this.debug = debug;
    this.sock = io.connect(uri)

    this.sock.on('json', (msg: string) => {
      let obj = JSON.parse(msg);
      if (obj.token != this.token) {
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

    return new Promise((resolve) => {
      // Connect to the server.
      this.sock.on('connect', () => {
        this.newSession();
        resolve();
      });
    });
  }

  // Add a handler that will be invoked whenever the Minigo engine writes
  // anything to stdout, or writes something to stderr that isn't handled by
  // one of the data handlers registered via onData.
  onText(handler: TextHandler) {
    this.textHandlers.push(handler);
  }

  // Add a handler that accepts lines that begin with the given prefix plus ':'.
  // The matching handlers are invoked in the order they were registered.
  // Before the handlers are invoked, the contents of the line that follows the
  // matching prefix are parsed as a JSON object if possible and passed as the
  // handler argument. The raw line suffix is passed if it can't be parsed.
  onData(prefix: string, handler: DataHandler) {
    this.dataHandlers.push({prefix: prefix + ':', handler: handler});
  }

  // Sends a GTP command, returning a promise that is resolved when the
  // command succeeds or is rejected if the command fails.
  send(cmd: string): Promise<any> {
    return new Promise((resolve, reject) => {
      this.cmdQueue.push({cmd: cmd, resolve: resolve, reject: reject});
      if (this.cmdQueue.length == 1) {
        this.sendNext();
      }
    });
  }

  // Like send(cmd), but if the last call to sendOne is for the same GTP
  // command (different arguments are allowed) and that command has not yet
  // been sent, the pending command is rejected and replaced with this one.
  sendOne(cmd: string) {
    // Either the queue is empty (in which case we must just send) or there's
    // a single command currently being executed (which we must let finish).
    if (this.cmdQueue.length <= 1) {
      return this.send(cmd);
    }

    // If the last command in the queue doesn't match this one: just send.
    let lastCmd = this.cmdQueue[this.cmdQueue.length - 1].cmd;
    if (cmd.split(' ', 1)[0] != lastCmd.split(' ', 1)[0]) {
      return this.send(cmd);
    }

    // The last command in the queue matches this one: reject it and replace
    // it with ours.
    this.cmdQueue[this.cmdQueue.length - 1].reject('send one');
    this.cmdQueue.length -= 1;
    return this.send(cmd);
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
    let token = `${this.playerName}-${Date.now()}`;
    this.token = token;
    this.send(`echo __NEW_TOKEN__ ${token}`);
  }

  private cmdHandler(line: string) {
    let {cmd, resolve, reject} = this.cmdQueue[0];

    if (this.debug) {
      console.log(`### ${this.playerName} OUT ${cmd} ${line}`);
    }

    this.textHandler(`${trimText(line, 1024)}`);

    if (line[0] == '=' || line[0] == '?') {
      // This line contains the response from a GTP command; pop the command off
      // the queue.
      this.cmdQueue = this.cmdQueue.slice(1);
      if (this.cmdQueue.length > 0) {
        this.sendNext();
      }
    }

    let ok = line[0] == '=';
    let result = line.substr(1).trim();
    if (ok) {
      resolve(result);
    } else {
      reject(result);
    }
  }

  private stderrHandler(line: string) {
    let handled = false;

    if (this.debug) {
      console.log(`### ${this.playerName} ERR ${line}`);
    }

    for (let {prefix, handler} of this.dataHandlers) {
      if (line.substr(0, prefix.length) == prefix) {
        let stripped = line.substr(prefix.length);
        let obj;
        try {
          obj = JSON.parse(stripped);
        } catch (e) {
          obj = stripped;
        }
        try {
          handler(obj);
        } catch (e) {
          // Catch and log all exceptions to prevent them breaking the socket.
          console.log(`Error handling ${trimText(line, 1024)}`);
          console.log(e);
        }
        handled = true;
      }
    }
    if (!handled) {
      this.textHandler(line);
    }
  }

  private textHandler(str: string) {
    for (let handler of this.textHandlers) {
      handler(str);
    }
  }

  private sendNext() {
    let {cmd} = this.cmdQueue[0];
    if (this.textHandler) {
      this.textHandler(trimText(cmd, 1024));
    }
    if (this.debug) {
      console.log(`### ${this.playerName} SND ${cmd}`);
    }
    this.sock.emit('gtpcmd', {player: this.playerName, data: cmd});
  }
}

export {
  Socket,
};
