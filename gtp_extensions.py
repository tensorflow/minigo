# extends gtp.py

import sys 

def parse_message(message):
    message = pre_engine(message).strip()
    first, rest = (message.split(" ", 1) + [None])[:2]
    if first.isdigit():
        message_id = int(first)
        if rest is not None:
            command, arguments = (rest.split(" ", 1) + [None])[:2]
        else:
            command, arguments = None, None
    else:
        message_id = None
        command, arguments = first, rest

    command = command.replace("-", "_") # for kgs extensions.
    return message_id, command, arguments


class KgsExtensionsMixin(gtp.Engine):

    def __init__(self, game_obj, name="gtp (python, kgs-chat extensions)", version="0.1"):
        super().__init__(game_obj=game_obj, name=name, version=version)
        self.known_commands += ["kgs-chat"]

    def send(self, message):
        message_id, command, arguments = parse_message(message)
        if command in self.known_commands:
            try:
                retval = getattr(self, "cmd_" + command)(arguments)
                response = format_success(message_id, retval)
                sys.stderr.flush()
                return response
            except ValueError as exception:
                return format_error(message_id, exception.args[0])
        else:
            return format_error(message_id, "unknown command: " + command)

    # Nice to implement this, as KGS sends it each move.
    def cmd_time_left(self, arguments):
        pass

    def cmd_kgs_chat(self, arguments):
        try:
            msg_type, sender, text = arguments.split()
        except ValueError:
            return "Unparseable message, args: %r" % arguments
        return self._game.chat(msg_type, sender, text)
