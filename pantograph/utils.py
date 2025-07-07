import asyncio
from pathlib import Path
import functools as F

def get_event_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

def to_sync(func):
    loop = get_event_loop()
    @F.wraps(func)
    def wrapper(*args, **kwargs):
        return loop.run_until_complete(func(*args, **kwargs))
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
