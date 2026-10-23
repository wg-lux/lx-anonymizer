"""Lazy, ordered model blocks with process-enforced loading/inference deadlines.

Factories must be importable in a fresh Python interpreter. Only trusted Python objects
cross the private socket; this is not a network or persisted pickle format.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import pickle
import socket
import struct
import subprocess
import sys
import threading
import time
import weakref
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, Self, cast

logger = logging.getLogger(__name__)
_MAX_MESSAGE = 128 * 1024 * 1024
_in_model_worker = False
# Host workers gate daemon-compatible isolation on this explicit contract.
MODEL_PROCESS_ISOLATION_VERSION = 2


class ModelUnavailableError(RuntimeError):
    """Every configured model failed or was unavailable."""


class ModelCleanupError(RuntimeError):
    """A worker could not be reaped; do not start another model."""


class ModelChainRuntime(Protocol):
    @property
    def used(self) -> Mapping[str, int]: ...

    @property
    def failed_models(self) -> Sequence[str]: ...

    def close(self) -> None: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class ModelBlock[Input, Output]:
    name: str
    load: Callable[[], Callable[[Input], Output]]
    timeout_seconds: float = 30.0
    accept: Callable[[Output], bool] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Model block name must not be empty")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Model block timeout must be finite and positive")


@dataclass(frozen=True)
class ModelAttempt:
    name: str
    status: str
    elapsed_seconds: float


def _remaining(deadline: float | None, cancel: threading.Event | None) -> float | None:
    if cancel is not None and cancel.is_set():
        raise InterruptedError("Model chain cancelled")
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Model deadline exceeded")
    return min(remaining, 0.05)


def _receive(
    channel: socket.socket,
    size: int,
    deadline: float | None,
    cancel: threading.Event | None,
) -> bytes:
    parts: list[bytes] = []
    while size:
        channel.settimeout(_remaining(deadline, cancel))
        try:
            part = channel.recv(min(size, 65536))
        except TimeoutError:
            continue
        if not part:
            raise EOFError("Model worker disconnected")
        parts.append(part)
        size -= len(part)
    return b"".join(parts)


def _read(
    channel: socket.socket,
    deadline: float | None = None,
    cancel: threading.Event | None = None,
) -> object:
    size = struct.unpack("!Q", _receive(channel, 8, deadline, cancel))[0]
    if size > _MAX_MESSAGE:
        raise ValueError("Model message exceeds the memory limit")
    return cast(object, pickle.loads(_receive(channel, size, deadline, cancel)))


def _write(
    channel: socket.socket,
    value: object,
    deadline: float | None = None,
    cancel: threading.Event | None = None,
) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > _MAX_MESSAGE:
        raise ValueError("Model message exceeds the memory limit")
    pending = memoryview(struct.pack("!Q", len(payload)) + payload)
    while pending:
        channel.settimeout(_remaining(deadline, cancel))
        try:
            sent = channel.send(pending)
        except TimeoutError:
            continue
        if sent == 0:
            raise EOFError("Model worker disconnected")
        pending = pending[sent:]


def _serve[Input, Output](
    load: Callable[[], Callable[[Input], Output]], channel: socket.socket
) -> None:
    global _in_model_worker
    _in_model_worker = True
    try:
        _write(channel, "ready")
        model = load()
        while True:
            request = cast(Input, _read(channel))
            result = model(request)
            _write(channel, (True, result))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:  # noqa: BLE001 - worker integration boundary
        # Integration boundary: transmit only the class, never patient text.
        try:
            _write(channel, (False, type(exc).__name__))
        except OSError:
            pass
    finally:
        channel.close()


def _dispose(
    process: subprocess.Popen[bytes] | None,
    channel: socket.socket | None,
    lifeline: socket.socket | None,
) -> None:
    if channel is not None:
        channel.close()
    if lifeline is not None:
        lifeline.close()
    if process is None:
        return
    try:
        status = process.wait(timeout=5.0)
    except subprocess.TimeoutExpired as exc:
        raise ModelCleanupError("Model supervisor could not be reaped") from exc
    if status != 0:
        raise ModelCleanupError("Model supervisor could not confirm descendant cleanup")


class ModelChain[Input, Output]:
    """One owner, one active worker, no background model downloads at startup.

    A failed block is disabled for this chain's lifetime. Construct a new chain
    for a new attempt. Empty valid results are accepted unless `accept` rejects
    them. Rejected results are not infrastructure failures and are not disabled.
    """

    def __init__(self, blocks: Sequence[ModelBlock[Input, Output]]) -> None:
        if not blocks or len({block.name for block in blocks}) != len(blocks):
            raise ValueError("Model chains require nonempty, uniquely named blocks")
        self.blocks = tuple(blocks)
        self.attempts: deque[ModelAttempt] = deque(maxlen=128)
        self.used: Counter[str] = Counter()
        self._disabled: set[str] = set()
        self._process: subprocess.Popen[bytes] | None = None
        self._lifeline: socket.socket | None = None
        self._channel: socket.socket | None = None
        self._active: str | None = None
        self._finalizer: (
            weakref.finalize[
                [
                    subprocess.Popen[bytes] | None,
                    socket.socket | None,
                    socket.socket | None,
                ],
                Self,
            ]
            | None
        ) = None
        self._lock = threading.Lock()
        self._cancel = threading.Event()

    def run(
        self, value: Input, *, accept: Callable[[Output], bool] | None = None
    ) -> Output:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("ModelChain cannot serve concurrent invocations")
        try:
            for block in self.blocks:
                if self._cancel.is_set():
                    raise InterruptedError("Model chain cancelled")
                if block.name in self._disabled:
                    continue
                started = time.monotonic()
                status = "failed"
                try:
                    deadline = started + block.timeout_seconds
                    self._start(block, deadline)
                    assert self._channel is not None
                    _write(self._channel, value, deadline, self._cancel)
                    reply = _read(self._channel, deadline, self._cancel)
                    if not isinstance(reply, tuple):
                        raise ModelUnavailableError("Model worker failed")
                    envelope = cast(tuple[object, ...], reply)
                    if len(envelope) == 2 and envelope == (
                        False,
                        "SensitiveMetaResolutionError",
                    ):
                        from lx_anonymizer.sensitive_meta_interface import (
                            SensitiveMetaResolutionError,
                        )

                        raise SensitiveMetaResolutionError(
                            "Model returned inconsistent sensitive metadata"
                        )
                    if len(envelope) != 2 or envelope[0] is not True:
                        logger.info(
                            "model_block name=%s failure_type=%s",
                            block.name,
                            envelope[1] if len(envelope) == 2 else "invalid_reply",
                        )
                        raise ModelUnavailableError("Model worker failed")
                    result = cast(Output, envelope[1])
                    self.used[block.name] += 1
                    validator = block.accept or accept
                    if validator is None or validator(result):
                        status = "used"
                        return result
                    status = "no_signal"
                except InterruptedError:
                    raise
                except (OSError, EOFError, ModelUnavailableError) as exc:
                    status = "timeout" if isinstance(exc, TimeoutError) else "failed"
                    self._disabled.add(block.name)
                finally:
                    self.attempts.append(
                        ModelAttempt(block.name, status, time.monotonic() - started)
                    )
                    if status != "used":
                        self._stop()
                    if status != "used" or self.used[block.name] == 1:
                        logger.info("model_block name=%s status=%s", block.name, status)
            raise ModelUnavailableError("All configured model blocks are unavailable")
        finally:
            self._lock.release()

    async def arun(
        self, value: Input, *, accept: Callable[[Output], bool] | None = None
    ) -> Output:
        task = asyncio.create_task(asyncio.to_thread(self.run, value, accept=accept))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            self._cancel.set()
            try:
                await task
            except (InterruptedError, ModelUnavailableError):
                pass
            raise

    def _start(self, block: ModelBlock[Input, Output], deadline: float) -> None:
        if _in_model_worker:
            raise ModelUnavailableError("Nested isolated model workers are forbidden")
        if self._active == block.name:
            return
        self._stop()
        parent, child = socket.socketpair()
        lifeline, monitor = socket.socketpair()
        self._channel, self._lifeline = parent, lifeline
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "lx_anonymizer._model_supervisor",
                    "supervisor",
                    str(child.fileno()),
                    str(monitor.fileno()),
                ],
                pass_fds=(child.fileno(), monitor.fileno()),
                start_new_session=True,
                env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
            )
            self._process = process
            self._finalizer = weakref.finalize(
                self, _dispose, process, parent, lifeline
            )
        finally:
            child.close()
            monitor.close()
        _write(parent, block.load, deadline, self._cancel)
        if _read(parent, deadline, self._cancel) != "ready":
            raise ModelUnavailableError("Model worker did not initialize")
        self._active = block.name

    def _stop(self) -> None:
        process, channel, lifeline = self._process, self._channel, self._lifeline
        # Preserve failed cleanup state: another call cannot start a new model
        # until this supervisor has confirmed that all descendants were reaped.
        _dispose(process, channel, lifeline)
        self._process = self._channel = self._active = None
        self._lifeline = None
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None

    def close(self) -> None:
        self._cancel.set()
        if not self._lock.acquire(timeout=1.0):
            raise ModelCleanupError("Model chain is still running")
        try:
            self._stop()
        finally:
            self._lock.release()

    def reset(self) -> None:
        """Explicitly start a fresh attempt; never revive a failed worker in-place."""
        self.close()
        self._cancel.clear()
        self._disabled.clear()
        self.attempts.clear()
        self.used.clear()

    @property
    def failed_models(self) -> list[str]:
        return sorted(self._disabled)

    def __enter__(self) -> ModelChain[Input, Output]:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
