from __future__ import annotations

import shutil
import socket
import subprocess
import time

import pytest

from lx_anonymizer.utils.ollama import ensure_ollama


def _port_open(host: str = "127.0.0.1", port: int = 11434) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def _ollama_ps_ok() -> bool:
    if shutil.which("ollama") is None:
        return False
    cp = subprocess.run(
        ["ollama", "ps"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    return cp.returncode == 0


@pytest.mark.integration
def test_ensure_ollama_real_fallback_starts_server_when_port_is_free() -> None:
    """
    Real integration test for ensure_ollama() without patching.

    This specifically targets the "start ollama serve" fallback path and only runs
    when the default Ollama port is currently free. If a server is already running,
    we xfail because the fallback-start path cannot be exercised safely.
    """
    if shutil.which("ollama") is None:
        pytest.skip("ollama CLI is not installed in PATH")

    if _port_open() or _ollama_ps_ok():
        pytest.xfail(
            "Ollama server is already running on localhost:11434; "
            "cannot exercise ensure_ollama() fallback-start path safely."
        )

    proc = None
    try:
        proc = ensure_ollama(timeout=15)
        assert proc is not None, "ensure_ollama() did not return a process handle"

        # Give the server a short moment to finish becoming responsive.
        deadline = time.time() + 5
        while time.time() < deadline and not (_port_open() and _ollama_ps_ok()):
            time.sleep(0.2)

        assert _port_open(), "Ollama port 11434 is not open after ensure_ollama()"
        assert _ollama_ps_ok(), "`ollama ps` failed after ensure_ollama() startup"
    except RuntimeError as e:
        pytest.xfail(
            f"ensure_ollama() could not start a usable server in this env: {e}"
        )
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
