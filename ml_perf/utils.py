# Copyright 2019 Google LLC
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

"""Utilities for the reinforcement trainer."""

import sys
sys.path.insert(0, '.')  # nopep8

import asyncio
import logging
import os

import mask_flags
from absl import flags
from utils import *


# Flags that take multiple values.
# For these flags, expand_cmd_str appends each value in order.
# For all other flags, expand_cmd_str takes the last value.
MULTI_VALUE_FLAGS = set(['--lr_boundaries', '--lr_rates'])


flag_cache = {}
flag_cache_lock = asyncio.Lock()


def is_python_cmd(cmd):
    return cmd[0] == 'python' or cmd[0] == 'python3'


def get_cmd_name(cmd):
    path = cmd[1] if is_python_cmd(cmd) else cmd[0]
    return os.path.splitext(os.path.basename(path))[0]


async def expand_cmd_str(cmd):
    n = 2 if is_python_cmd(cmd) else 1
    cmd = list(cmd)
    args = cmd[n:]
    process = cmd[:n]
    key = ' '.join(process)

    async with flag_cache_lock:
        valid_flags = flag_cache.get(key)
        if valid_flags is None:
            valid_flags = mask_flags.extract_valid_flags(cmd)
            flag_cache[key] = valid_flags

    parsed_args = flags.FlagValues().read_flags_from_files(args)
    flag_args = {}
    position_args = []
    for arg in parsed_args:
        if arg.startswith('--'):
            if '=' not in arg:
                flag_args[arg] = None
            else:
                flag, value = arg.split('=', 1)
                if flag in MULTI_VALUE_FLAGS:
                    if flag not in flag_args:
                        flag_args[flag] = []
                    flag_args[flag].append(value)
                else:
                    flag_args[flag] = value
        else:
            position_args.append(arg)

    flag_list = []
    for flag, value in flag_args.items():
        if value is None:
            flag_list.append(flag)
        elif type(value) == list:
            for v in value:
                flag_list.append('%s=%s' % (flag, v))
        else:
            flag_list.append('%s=%s' % (flag, value))

    flag_list = sorted(mask_flags.filter_flags(flag_list, valid_flags))
    return '  '.join(process + flag_list + position_args)


async def checked_run(*cmd):
    """Run the given subprocess command in a coroutine.

    Args:
        *cmd: the command to run and its arguments.

    Returns:
        The output that the command wrote to stdout & stderr.

    Raises:
        RuntimeError: if the command returns a non-zero result.
    """

    # Start the subprocess.
    logging.info('Running: %s', await expand_cmd_str(cmd))
    with logged_timer('{} finished'.format(get_cmd_name(cmd))):
        p = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT)

        # Stream output from the process stdout.
        lines = []
        while True:
            line = await p.stdout.readline()
            if not line:
                break
            line = line.decode()[:-1]
            lines.append(line)
            logging.info(line)

        # Wait for the process to finish, check it was successful & build stdout.
        await p.wait()
        output = '\n'.join(lines)[:-1]
        if p.returncode:
            raise RuntimeError('Return code {} from process: {}\n{}'.format(
                p.returncode, await expand_cmd_str(cmd), output))

        return output


def wait(aws):
    """Waits for all of the awaitable objects (e.g. coroutines) in aws to finish.

    All the awaitable objects are waited for, even if one of them raises an
    exception. When one or more awaitable raises an exception, the exception
    from the awaitable with the lowest index in the aws list will be reraised.

    Args:
        aws: a single awaitable, or list awaitables.

    Returns:
        If aws is a single awaitable, its result.
        If aws is a list of awaitables, a list containing the of each awaitable
        in the list.

    Raises:
        Exception: if any of the awaitables raises.
    """

    aws_list = aws if isinstance(aws, list) else [aws]
    results = asyncio.get_event_loop().run_until_complete(asyncio.gather(
        *aws_list, return_exceptions=True))
    # If any of the cmds failed, re-raise the error.
    for result in results:
        if isinstance(result, Exception):
            raise result
    return results if isinstance(aws, list) else results[0]
