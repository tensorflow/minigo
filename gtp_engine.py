# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Version of the GTP specification used:
#   https://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html

import inspect
import re
import sys
import traceback


# List of canonical error messages as required by 6.3 in the specification.
_CANONICAL_ERRORS = {
    "boardsize": "unacceptable size",
    "play": "illegal move",
    "undo": "cannot undo",
    "final_score": "cannot score",
    "loadsgf": "cannot load file",
}

_GTP_CMD_DONE = "__GTP_CMD_DONE__"


def _preprocess(msg):
    # From the spec (3.1 Preprocessing):
    # 1. Remove all occurences of CR and other control characters except for
    #    HT and LF.
    # 2. For each line with a hash sign (#), remove all text following and
    #    including this character.
    # 3. Convert all occurences of HT to SPACE.
    # 4. Discard any empty or white-space only lines.

    # Control characters are defined in section 2.2 as dec 0-31 (oct 0-037)
    # and 127 (oct 177). We want to remove them all except \t (oct 011) and
    # \n (oct 12).
    msg = re.sub("r[\000-\010\013-\037\177]", "", msg)
    msg = msg.split("#", 1)[0]
    msg = msg.replace("\t", " ")
    return msg


def _parse(msg):
    msg = _preprocess(msg).strip()
    if not msg:
        return None, None, None
    parts = [x for x in msg.split(" ") if x]
    if len(parts) > 1 and parts[0].isdigit():
        msg_id = parts[0]
        parts = parts[1:]
    else:
        msg_id = None
    return msg_id, parts[0], parts[1:]


def _print_msg(result, msg_id, msg):
    msg_id = " {}".format(msg_id) if msg_id else ""
    if isinstance(msg, bool):
        msg = "true" if msg else "false"
    msg = " {}".format(msg) if msg else ""
    print("{}{}{}\n".format(result, msg_id, msg), flush=True)


def _print_error(msg_id, msg):
    print(_GTP_CMD_DONE, file=sys.stderr)
    _print_msg("?", msg_id, msg)


def _print_success(msg_id, msg):
    print(_GTP_CMD_DONE, file=sys.stderr)
    _print_msg("=", msg_id, msg)


def _handler_name(fn):
    return "{}.{}".format(fn.__self__.__class__.__name__, fn.__name__)


def _convert_args(handler, args):
    """Convert a list of command arguments to types specified by the handler.

    Args:
      handler: a command handler function.
      args: the list of string arguments to pass to handler.

    Returns:
      A new list containing `args` that have been converted to the expected type
      for `handler`. For each function parameter of `handler` that has either an
      explicit type annotation or a non-None default value, the corresponding
      element in `args` is converted to that type.
    """

    args = list(args)
    params = inspect.signature(handler).parameters
    for i, (arg, name) in enumerate(zip(args, params)):
        default = params[name].default
        annotation = params[name].annotation

        if annotation != inspect.Parameter.empty:
            if isinstance(annotation, type) and annotation != str:
                # The parameter is annotated with a type that isn't str: convert
                # the arg to that type.
                args[i] = annotation(arg)
        elif default != inspect.Parameter.empty:
            if default is not None and not isinstance(default, str):
                # The parameter has a default value that isn't None or a str:
                # convert the arg to the default value's type.
                args[i] = type(default)(arg)

    return args


class Engine(object):
    """A simple GTP engine.

    The engine by itself doesn't do anything: clients must register command
    handler objects using `add_cmd_handler`.
    """

    def __init__(self):
        self.cmds = {}

    def add_cmd_handler(self, handler_obj):
        """Registers a new command handler object.

        All methods on `handler_obj` whose name starts with "cmd_" are
        registered as a GTP command. For example, the method cmd_genmove will
        be invoked when the engine receives a genmove command.

        Args:
          handler_obj: the handler object to register.
        """
        for field in dir(handler_obj):
            if field.startswith("cmd_"):
                cmd = field[4:]
                fn = getattr(handler_obj, field)
                if cmd in self.cmds:
                    print('Replacing {} with {}'.format(
                        _handler_name(self.cmds[cmd]), _handler_name(fn)),
                        file=sys.stderr)
                self.cmds[cmd] = fn

    def handle_msg(self, msg):
        msg_id, cmd, args = _parse(_preprocess(msg))
        if not cmd:
            # Ignore empty lines.
            return True

        if cmd == "quit":
            _print_success(msg_id, "")
            return False

        sanitized_cmd = cmd.replace("-", "_")
        if sanitized_cmd not in self.cmds:
            _print_error(msg_id, "unknown command")
            return True

        try:
            handler = self.cmds[sanitized_cmd]
            args = _convert_args(handler, args)
            _print_success(msg_id, handler(*args))
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            if cmd in _CANONICAL_ERRORS:
                _print_error(msg_id, _CANONICAL_ERRORS[cmd])
            else:
                _print_error(msg_id, " ".join(map(str, e.args)))

        return True


class EngineCmdHandler(object):
    """Command handlers for basic engine stuff."""

    def __init__(self, engine, name, version):
        self._engine = engine
        self._name = name
        self._version = version

    def cmd_protocol_version(self):
        return 2

    def cmd_name(self):
        return self._name

    def cmd_version(self):
        return self._version

    def cmd_known_command(self, cmd):
        return cmd in self._engine.cmds

    def cmd_list_commands(self):
        return "\n".join(sorted(self._engine.cmds.keys()))
