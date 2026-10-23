"""Private Linux subprocess supervisor; never accepts network connections.

The liveness channel is separate from model traffic so a dead task is observable
even when a model stops reading. Only the executor unpickles trusted caller data.
"""

from __future__ import annotations

import ctypes
import os
import select
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import cast

_BUFFER_LIMIT = 1024 * 1024
_CLEANUP_SECONDS = 3.0
_stop_requested = False


def _request_stop(_signal: int, _frame: FrameType | None) -> None:
    global _stop_requested
    _stop_requested = True


def _become_subreaper() -> None:
    # Linux PR_SET_CHILD_SUBREAPER: detached grandchildren are adopted here,
    # rather than escaping to the service manager when their parent is killed.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot establish model subreaper")


def _kill_and_reap(executor: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + _CLEANUP_SECONDS
    try:
        os.killpg(executor.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    children_file = Path(f"/proc/self/task/{os.getpid()}/children")
    while True:
        # Killing each adopted child makes its own detached descendants become
        # our children. Repeating this closes double-fork and setsid escapes.
        for child in children_file.read_text().split():
            try:
                os.kill(int(child), signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            # Popen must not later try to reap a PID already reaped above.
            executor.returncode = 0
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Model descendants could not be reaped")
        time.sleep(0.01)


def _relay(
    parent: socket.socket, model: socket.socket, lifeline: socket.socket
) -> None:
    parent.setblocking(False)
    model.setblocking(False)
    pending: dict[socket.socket, bytearray] = {parent: bytearray(), model: bytearray()}
    other = {parent: model, model: parent}
    model_finished = False
    while not _stop_requested:
        if model_finished and not pending[parent]:
            return
        readers = [lifeline]
        for channel in (parent, model):
            if channel is model and model_finished:
                continue
            if len(pending[other[channel]]) < _BUFFER_LIMIT:
                readers.append(channel)
        writers = [channel for channel, data in pending.items() if data]
        ready_read, ready_write, _ = select.select(readers, writers, [], 0.1)
        if lifeline in ready_read:
            return
        for channel in ready_read:
            data = channel.recv(
                min(65536, _BUFFER_LIMIT - len(pending[other[channel]]))
            )
            if not data:
                if channel is parent:
                    return
                model_finished = True
                pending[model].clear()
            else:
                pending[other[channel]].extend(data)
        for channel in ready_write:
            if channel is model and model_finished:
                continue
            sent = channel.send(pending[channel])
            del pending[channel][:sent]


def _supervise(parent: socket.socket, lifeline: socket.socket) -> None:
    _become_subreaper()
    if _stop_requested or select.select([lifeline], [], [], 0)[0]:
        return
    upstream, downstream = socket.socketpair()
    try:
        executor = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "lx_anonymizer._model_supervisor",
                "executor",
                str(downstream.fileno()),
            ],
            pass_fds=(downstream.fileno(),),
            start_new_session=True,
        )
    finally:
        downstream.close()
    try:
        try:
            _relay(parent, upstream, lifeline)
        except (BrokenPipeError, ConnectionResetError):
            pass
    finally:
        upstream.close()
        _kill_and_reap(executor)


def _execute(channel: socket.socket) -> None:
    from lx_anonymizer.model_loading import _read, _serve

    factory = cast(Callable[[], Callable[[object], object]], _read(channel))
    if not callable(factory):
        raise TypeError("Model factory must be callable")
    _serve(factory, channel)


def main() -> int:
    try:
        with socket.socket(fileno=int(sys.argv[2])) as channel:
            if sys.argv[1] == "executor":
                _execute(channel)
            else:
                signal.signal(signal.SIGTERM, _request_stop)
                signal.signal(signal.SIGINT, _request_stop)
                with socket.socket(fileno=int(sys.argv[3])) as lifeline:
                    _supervise(channel, lifeline)
    except Exception as exc:  # noqa: BLE001 - private process integration boundary
        # Never print exception messages: model inputs can contain patient data.
        print(f"Model subprocess failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
