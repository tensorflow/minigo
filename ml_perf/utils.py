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

from absl import flags
from utils import *


def expand_cmd_str(cmd):
    return '  '.join(flags.FlagValues().read_flags_from_files(cmd))


def get_cmd_name(cmd):
    if cmd[0] == 'python' or cmd[0] == 'python3':
        path = cmd[1]
    else:
        path = cmd[0]
    return os.path.splitext(os.path.basename(path))[0]


async def checked_run(*cmd):
    """Run the given subprocess command in a coroutine.

    Args:
        *cmd: the command to run and its arguments.

    Returns:
        The output that the command wrote to stdout.

    Raises:
        RuntimeError: if the command returns a non-zero result.
    """

    # Start the subprocess.
    logging.info('Running: %s', expand_cmd_str(cmd))
    with logged_timer('{} finished'.format(get_cmd_name(cmd))):
        p = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT)

        # Stream output from the process stdout.
        chunks = []
        while True:
            chunk = await p.stdout.read(16 * 1024)
            if not chunk:
                break
            chunks.append(chunk)

        # Wait for the process to finish, check it was successful & build stdout.
        await p.wait()
        stdout = b''.join(chunks).decode()[:-1]
        if p.returncode:
            raise RuntimeError('Return code {} from process: {}\n{}'.format(
                p.returncode, expand_cmd_str(cmd), stdout))

        return stdout


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
