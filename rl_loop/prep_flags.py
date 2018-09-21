'''Filters flagfile to only pass in flags that are defined.

Having one big flagfile is great for seeing all the configuration at a glance.
However, absl.flags will throw an error if you pass an undefined flag.

To solve this problem, we mask the global flagfile by running 
    python some_module.py --helpfull
to generate a list of all flags that some_module.py accepts. Then, we can read in
the flagfile and pass in only those flags that are accepted by some_module.py
'''

import re
import subprocess
from absl import flags

# Matches both
#   --some_flag: Flag description
#   --[no]bool_flag: Flag description
FLAG_HELP_RE = re.compile(r'--((\[no\])?)([\w_-]+):')
FLAG_RE = re.compile(r'--[\w_-]+')


def parse_helpfull_output(help_output):
    '''Parses the output of --helpfull.
    Args:
        help_output: str, the full output of --helpfull.

    Returns a set of flags that are valid flags.'''
    valid_flags = set()
    for _, no_prefix, flag_name in FLAG_HELP_RE.findall(help_output):
        valid_flags.add('--' + flag_name)
        if no_prefix:
            valid_flags.add('--no' + flag_name)
    return valid_flags


assert parse_helpfull_output('''
    --some_flag: Flag description
    --[no]bool_flag: Flag description
    ''') == {'--some_flag', '--bool_flag', '--nobool_flag'}

def prepare_subprocess_cmd(subprocess_cmd):
    '''Prepares a subprocess command by running --helpfull and masking flags.

    Args:
        subprocess_cmd: List[str], what would be passed into subprocess.call()
            i.e. ['python', 'train.py', '--flagfile=flags']

    Returns:
        List[str], ['python', 'train.py', '--train_flag=blah', '--more_flags']
    '''
    help_cmd = subprocess_cmd + ['--helpfull']
    help_output = subprocess.run(help_cmd, stdout=subprocess.PIPE).stdout
    help_output = help_output.decode('ascii')
    valid_flags = parse_helpfull_output(help_output)
    parsed_flags = flags.FlagValues().read_flags_from_files(subprocess_cmd[1:])
    def valid_argv(argv):
        flagname_match = FLAG_RE.match(argv)
        if not flagname_match:
            return True
        else:
            flagname = flagname_match.group()
            return flagname in valid_flags
    filtered_flags = list(filter(valid_argv, parsed_flags))
    return [subprocess_cmd[0]] + filtered_flags

def run(cmd):
    '''Prepare and run a subprocess cmd, returning a CompletedProcess.'''
    cmd = prepare_subprocess_cmd(cmd)
    print("Running the following cmd", cmd)
    return subprocess.run(cmd, capture_output=True)

def checked_run(cmd):
    completed_process = run(cmd)
    if completed_process.returncode > 0:
        print("Command failed!")
        print("stdout:\n", completed_process.stdout)
        print("stderr:\n", completed_process.stderr)
    raise RuntimeError
