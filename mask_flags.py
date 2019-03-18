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

"""Filters flagfile to only pass in flags that are defined.

Having one big flagfile is great for seeing all the configuration at a glance.
However, absl.flags will throw an error if you pass an undefined flag.

To solve this problem, we filter the global flagfile by running
    python3 some_module.py --helpfull
to generate a list of all flags that some_module.py accepts. Then, we pass in
only those flags that are accepted by some_module.py and run as a subprocess

Usage example:
    import mask_flags
    mask_flags.run(['python3', 'train.py', '--custom_flag', '--flagfile=flags'])
    # will be transformed into
    subprocess.run(['python3', 'train.py', '--custom_flag',
                    '--train_only_flag=...', '--more_train_only=...''])

Command line usage example:
    python3 -m mask_flags train.py --custom_flag --flagfile=flags
"""

import re
import subprocess
import sys
import time
from absl import flags

# Matches both
#   --some_flag: Flag description
#   --[no]bool_flag: Flag description
FLAG_HELP_RE_PY = re.compile(r'--((\[no\])?)([\w_-]+):')
FLAG_HELP_RE_CC = re.compile(r'-((\[no\])?)([\w_-]+) \(')
FLAG_RE = re.compile(r'--[\w_-]+')


def parse_helpfull_output(help_output, regex=FLAG_HELP_RE_PY):
    """Parses the output of --helpfull.

    Args:
        help_output: str, the full output of --helpfull.

    Returns:
        A set of flags that are valid flags.
    """
    valid_flags = set()
    for _, no_prefix, flag_name in regex.findall(help_output):
        valid_flags.add('--' + flag_name)
        if no_prefix:
            valid_flags.add('--no' + flag_name)
    return valid_flags


def filter_flags(parsed_flags, valid_flags):
    """Return the subset of `parsed_flags` that are found in the list `valid_flags`"""
    def valid_argv(argv):
        """Figures out if a flag parsed from the flagfile matches a flag in
        the command about to be run."""
        flagname_match = FLAG_RE.match(argv)
        if not flagname_match:
            return True
        flagname = flagname_match.group()
        return flagname in valid_flags
    return list(filter(valid_argv, parsed_flags))


def prepare_subprocess_cmd(subprocess_cmd):
    """Prepares a subprocess command by running --helpfull and masking flags.

    Args:
        subprocess_cmd: List[str], what would be passed into subprocess.call()
            i.e. ['python', 'train.py', '--flagfile=flags']

    Returns:
        ['python', 'train.py', '--train_flag=blah', '--more_flags']
    """
    help_cmd = subprocess_cmd + ['--helpfull']
    help_output = subprocess.run(help_cmd, stdout=subprocess.PIPE).stdout
    help_output = help_output.decode('ascii')
    if 'python' in subprocess_cmd[0]:
        valid_flags = parse_helpfull_output(help_output)
    else:
        valid_flags = parse_helpfull_output(help_output, regex=FLAG_HELP_RE_CC)
    parsed_flags = flags.FlagValues().read_flags_from_files(subprocess_cmd[1:])

    filtered_flags = filter_flags(parsed_flags, valid_flags)
    return [subprocess_cmd[0]] + filtered_flags


def run(cmd):
    """Prepare and run a subprocess cmd, returning a CompletedProcess."""
    print("Preparing the following cmd:")
    cmd = prepare_subprocess_cmd(cmd)
    print("Running the following cmd:")
    print('\n'.join(cmd))
    return subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)


def checked_run(cmd):
    """Prepare and run a subprocess cmd, checking for successful completion."""
    completed_process = run(cmd)
    if completed_process.returncode > 0:
        print("Command failed!  Hanging around in case someone needs a "
              "docker connection. (Ctrl-C to quit now)")
        time.sleep(300)
        raise RuntimeError
    return completed_process


if __name__ == '__main__':
    sys.argv.pop(0)
    checked_run(sys.argv)
