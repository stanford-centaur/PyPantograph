import os
import os.path as osp
import asyncio
import warnings
from pathlib import Path
import collections as C
import functools as F

import pexpect
import re


def to_sync(func):
    @F.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper

async def check_output(*args, **kwargs):
    p = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **kwargs,
    )
    stdout_data, stderr_data = await p.communicate()
    if p.returncode == 0:
        return stdout_data

class Spwan(pexpect.spawn):
    async def send_async(self, s):
        s = self._coerce_send_string(s)
        self._log(s, 'send')

        b = self._encoder.encode(s, final=False)
        return os.write(self.child_fd, b)

    async def sendline_async(self, s=''):
        '''Wraps send(), sending string ``s`` to child process, with
        ``os.linesep`` automatically appended. Returns number of bytes
        written.  Only a limited number of bytes may be sent for each
        line in the default terminal mode, see docstring of :meth:`send`.
        '''
        s = self._coerce_send_string(s)
        return await self.send_async(s + self.linesep)

    async def read_async(self, size=-1):
        '''This reads at most "size" bytes from the file (less if the read hits
        EOF before obtaining size bytes). If the size argument is negative or
        omitted, read all data until EOF is reached. The bytes are returned as
        a string object. An empty string is returned when EOF is encountered
        immediately. '''

        if size == 0:
            return self.string_type()
        if size < 0:
            # delimiter default is EOF
            await self.expect(self.delimiter, async_=True)
            return self.before

        # I could have done this more directly by not using expect(), but
        # I deliberately decided to couple read() to expect() so that
        # I would catch any bugs early and ensure consistent behavior.
        # It's a little less efficient, but there is less for me to
        # worry about if I have to later modify read() or expect().
        # Note, it's OK if size==-1 in the regex. That just means it
        # will never match anything in which case we stop only on EOF.
        cre = re.compile(self._coerce_expect_string('.{%d}' % size), re.DOTALL)
        # delimiter default is EOF
        index = await self.expect([cre, self.delimiter], async_=True)
        if index == 0:
            ### FIXME self.before should be ''. Should I assert this?
            return self.after
        return self.before

    async def readline_async(self, size=-1):
        '''This reads and returns one entire line. The newline at the end of
        line is returned as part of the string, unless the file ends without a
        newline. An empty string is returned if EOF is encountered immediately.
        This looks for a newline as a CR/LF pair (\\r\\n) even on UNIX because
        this is what the pseudotty device returns. So contrary to what you may
        expect you will receive newlines as \\r\\n.
        See https://pexpect.readthedocs.io/en/latest/overview.html#find-the-end-of-line-cr-lf-conventions
        for more information.

        If the size argument is 0 then an empty string is returned. In all
        other cases the size argument is ignored, which is not standard
        behavior for a file-like object. '''

        if size == 0:
            return self.string_type()

        def blocking_expect():
            # We use `asyncio.to_thread` because passing `async_=True` to `expect()` sometimes hangs. See e.g.:
            # https://stackoverflow.com/questions/12647212/why-is-pexpect-intermittently-hanging-not-detecting-eof-after-executing-certai
            return self.expect([self.crlf, self.delimiter])

        index = await asyncio.to_thread(blocking_expect)
        if index == 0:
            return self.before + self.crlf
        else:
            return self.before

def _get_proc_cwd():
    return Path(__file__).parent

def _get_proc_path():
    return _get_proc_cwd() / "pantograph-repl"

async def get_lean_path_async(project_path):
    """
    Extracts the `LEAN_PATH` variable from a project path.
    """
    p = await check_output(
        'lake', 'env', 'printenv', 'LEAN_PATH',
        cwd=project_path,
    )
    return p

get_lean_path = to_sync(get_lean_path_async)
