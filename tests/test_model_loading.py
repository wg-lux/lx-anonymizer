# pyright: reportPrivateUsage=false
from __future__ import annotations

import asyncio
import multiprocessing
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from lx_anonymizer.config import Settings
from lx_anonymizer.model_loading import ModelBlock, ModelChain, ModelUnavailableError


@dataclass(frozen=True)
class FakeFactory:
    behavior: str = "ok"

    def __call__(self) -> Callable[[str], str]:
        if self.behavior == "load_hang":
            time.sleep(60)
        if self.behavior == "nested":
            with ModelChain([ModelBlock("nested", FakeFactory())]) as chain:
                chain.run("forbidden")
        if self.behavior == "load_fail":
            raise ImportError("optional backend is absent")
        return FakeModel(self.behavior)


@dataclass
class FakeModel:
    behavior: str
    calls: int = 0

    def __call__(self, value: str) -> str:
        self.calls += 1
        if self.behavior == "infer_hang":
            time.sleep(60)
        if self.behavior == "crash":
            os._exit(12)
        if self.behavior == "empty":
            return ""
        if self.behavior == "invalid_sensitive":
            from lx_anonymizer.sensitive_meta_interface import (
                SensitiveMetaResolutionError,
            )

            raise SensitiveMetaResolutionError("invalid test metadata")
        return f"{value}:{os.getpid()}:{self.calls}"


@dataclass(frozen=True)
class DescendantFactory:
    snapshot: str
    hang_during_load: bool = False

    def __call__(self) -> Callable[[str], str]:
        # Both generations detach from the model process group. The intermediate
        # process exits, exercising subreaper adoption before cancellation.
        script = (
            "import subprocess,sys; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], "
            "start_new_session=True); print(p.pid, flush=True)"
        )
        middle = subprocess.Popen(
            [sys.executable, "-c", script],
            start_new_session=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert middle.stdout is not None
        grandchild = int(middle.stdout.readline())
        middle.stdout.close()
        middle.wait(timeout=3)
        Path(self.snapshot).write_text(f"{os.getppid()} {os.getpid()} {grandchild}")
        if self.hang_during_load:
            time.sleep(60)
        return FakeModel("infer_hang")


def _daemon_model_task(
    result_path: str,
    descendants: bool = False,
    backpressure: bool = False,
) -> None:
    factory = (
        DescendantFactory(result_path, backpressure) if descendants else FakeFactory()
    )
    with ModelChain([ModelBlock("model", factory, 30.0)]) as chain:
        value = chain.run("x" * (8 * 1024 * 1024) if backpressure else "daemon")
        Path(result_path).write_text(value)


def _wait_for_snapshot(path: Path) -> list[int]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if path.exists() and (value := path.read_text()):
            return [int(pid) for pid in value.split()]
        time.sleep(0.02)
    raise AssertionError("Model descendant snapshot was not produced")


def _assert_reaped(pids: list[int]) -> None:
    deadline = time.monotonic() + 7
    while time.monotonic() < deadline:
        if all(not Path(f"/proc/{pid}").exists() for pid in pids):
            return
        time.sleep(0.02)
    raise AssertionError(f"Model descendants survived cleanup: {pids}")


def test_daemon_task_can_start_model_and_next_recycled_task_succeeds(
    tmp_path: Path,
) -> None:
    model_pids: list[str] = []
    for attempt in range(2):
        output = tmp_path / str(attempt)
        task = multiprocessing.get_context("spawn").Process(
            target=_daemon_model_task,
            args=(str(output),),
            daemon=True,
        )
        task.start()
        task.join(timeout=10)
        try:
            assert task.exitcode == 0
            value = output.read_text()
            assert value.startswith("daemon:")
            model_pids.append(value.split(":")[1])
        finally:
            if task.is_alive():
                task.kill()
                task.join(timeout=5)
            task.close()
    assert model_pids[0] != model_pids[1]


@pytest.mark.parametrize("loading", [False, True])
def test_model_timeout_reaps_detached_descendants(
    tmp_path: Path, loading: bool
) -> None:
    snapshot = tmp_path / "descendants"
    with (
        ModelChain(
            [ModelBlock("hanging", DescendantFactory(str(snapshot), loading), 2.0)]
        ) as chain,
        pytest.raises(ModelUnavailableError),
    ):
        chain.run("value")
    _assert_reaped(_wait_for_snapshot(snapshot))


@pytest.mark.parametrize("backpressure", [False, True])
def test_hard_killed_daemon_task_cleans_detached_descendants(
    tmp_path: Path,
    backpressure: bool,
) -> None:
    snapshot = tmp_path / "descendants"
    task = multiprocessing.get_context("spawn").Process(
        target=_daemon_model_task,
        args=(str(snapshot), True, backpressure),
        daemon=True,
    )
    task.start()
    try:
        pids = _wait_for_snapshot(snapshot)
        task.kill()
        task.join(timeout=5)
        assert task.exitcode == -signal.SIGKILL
        # The orphan supervisor is reaped by the service manager; it must reap
        # every executor/descendant itself before exiting.
        _assert_reaped(pids[1:])
    finally:
        if task.is_alive():
            task.kill()
            task.join(timeout=5)
        task.close()


@pytest.mark.parametrize("startup_delay", [0.0, 0.05, 0.2])
def test_parent_disappearing_during_supervisor_startup_is_clean(
    startup_delay: float,
) -> None:
    parent, child = socket.socketpair()
    lifeline, monitor = socket.socketpair()
    try:
        supervisor = subprocess.Popen(
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
        )
    finally:
        child.close()
        monitor.close()
    time.sleep(startup_delay)
    lifeline.close()
    parent.close()
    assert supervisor.wait(timeout=5) == 0


def nonempty(value: str) -> bool:
    return bool(value)


@pytest.mark.parametrize("failure", ["load_hang", "load_fail", "infer_hang", "crash"])
def test_failure_advances_and_disables_failed_block(failure: str) -> None:
    with ModelChain(
        [
            ModelBlock("preferred", FakeFactory(failure), 1.0),
            ModelBlock("fallback", FakeFactory(), 3.0),
        ]
    ) as chain:
        assert chain._process is None  # construction does not load anything
        started = time.monotonic()
        first = chain.run("first")
        assert time.monotonic() - started < 5.0
        second = chain.run("second")
        assert first.split(":")[1] == second.split(":")[1]
        assert second.endswith(":2")
        assert chain.failed_models == ["preferred"]
        assert chain.used == {"fallback": 2}
        process = chain._process
        assert process is not None and process.poll() is None
    assert chain._process is None


def test_reordering_and_no_signal_fallback() -> None:
    with ModelChain(
        [
            ModelBlock("empty", FakeFactory("empty"), 3.0, nonempty),
            ModelBlock("local", FakeFactory(), 3.0),
        ]
    ) as chain:
        assert chain.run("value").startswith("value:")
        assert [attempt.status for attempt in chain.attempts] == ["no_signal", "used"]
        assert chain.failed_models == []
    with ModelChain(
        [
            ModelBlock("local", FakeFactory(), 3.0),
            ModelBlock("never_loaded", FakeFactory("load_fail"), 3.0),
        ]
    ) as chain:
        chain.run("value")
        assert [attempt.name for attempt in chain.attempts] == ["local"]


def test_exhaustion_is_not_an_empty_success_and_reset_is_explicit() -> None:
    with ModelChain([ModelBlock("missing", FakeFactory("load_fail"), 3.0)]) as chain:
        with pytest.raises(ModelUnavailableError):
            chain.run("value")
        with pytest.raises(ModelUnavailableError):
            chain.run("value")
        assert len(chain.attempts) == 1
        chain.reset()
        with pytest.raises(ModelUnavailableError):
            chain.run("value")
        assert len(chain.attempts) == 1


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_rejected(timeout: float) -> None:
    with pytest.raises(ValueError):
        ModelBlock("invalid", FakeFactory(), timeout)


@pytest.mark.parametrize("order", [[], ["rapidocr", "rapidocr"], ["unknown"]])
def test_invalid_model_order_rejected(order: list[str]) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({"OCR_MODEL_ORDER": order})


@pytest.mark.parametrize("behavior", ["load_hang", "infer_hang"])
def test_async_wait_is_responsive_and_cancellation_reaps_worker(behavior: str) -> None:
    async def scenario() -> None:
        with ModelChain([ModelBlock("hanging", FakeFactory(behavior), 3.0)]) as chain:
            task = asyncio.create_task(chain.arun("value"))
            for _ in range(10):
                await asyncio.sleep(0.05)
            assert not task.done()
            with pytest.raises(RuntimeError, match="concurrent"):
                chain.run("second")
            cancelled_at = time.monotonic()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert chain._process is None
            assert time.monotonic() - cancelled_at < 1.5

    asyncio.run(scenario())


def test_integrity_failure_never_selects_a_weaker_model() -> None:
    from lx_anonymizer.sensitive_meta_interface import SensitiveMetaResolutionError

    with ModelChain(
        [
            ModelBlock("invalid", FakeFactory("invalid_sensitive"), 10.0),
            ModelBlock("fallback", FakeFactory(), 3.0),
        ]
    ) as chain:
        with pytest.raises(SensitiveMetaResolutionError):
            chain.run("value")
        assert not chain.used
        assert len(chain.attempts) == 1


def test_valid_empty_ocr_is_distinct_from_unavailable_models() -> None:
    with ModelChain([ModelBlock("empty", FakeFactory("empty"), 3.0)]) as chain:
        assert chain.run("") == ""
        assert chain.used == {"empty": 1}


@pytest.mark.parametrize("report", [False, True])
def test_packaged_spacy_loads_in_worker_and_does_not_reuse_patient_state(
    report: bool,
) -> None:
    from lx_dtypes.models.meta.ReportMeta import ReportReaderFlags

    from lx_anonymizer.model_adapters import MetadataFactory, ReportMetadataFactory

    config = Settings(SPACY_MODEL="de_core_news_sm", SPACY_AUTO_DOWNLOAD=False)
    factory = (
        ReportMetadataFactory(config, ReportReaderFlags())
        if report
        else MetadataFactory("spacy", config)
    )
    with ModelChain([ModelBlock("spacy", factory, 30.0)]) as chain:
        first = chain.run("Patient: Angela Merkel geb. 17.07.1954")
        empty = chain.run("The screen is blank.")
        assert first["dob"] == "1954-07-17"
        assert empty["dob"] is None
        assert empty["first_name"] != first["first_name"]


def test_nested_worker_is_rejected_before_spawning_and_falls_back() -> None:
    with ModelChain(
        [
            ModelBlock("recursive", FakeFactory("nested"), 10.0),
            ModelBlock("fallback", FakeFactory(), 10.0),
        ]
    ) as chain:
        assert chain.run("value").startswith("value:")
        assert chain.failed_models == ["recursive"]
        assert chain.used == {"fallback": 1}
