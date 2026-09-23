import json
import os
import subprocess
from pathlib import Path

import pytest
import requests

from lx_anonymizer import system_dependencies as deps
from lx_anonymizer.config import Settings, normalize_ollama_host, settings
from lx_anonymizer.llm.connection import request_options, resolve_connection
from lx_anonymizer.ocr import tessdata


@pytest.mark.parametrize(
    "layout",
    [
        "tessdata",
        "../share/tessdata",
        "../share/tesseract-ocr/5/tessdata",
        "../share/tesseract-ocr/4.00/tessdata",
    ],
)
def test_installation_layouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str
) -> None:
    binary = tmp_path / "installation with spaces" / "bin" / "tesseract"
    binary.parent.mkdir(parents=True)
    binary.touch()
    data = binary.parent / layout
    data.mkdir(parents=True)
    (data / "eng.traineddata").touch()
    monkeypatch.setattr(tessdata, "_PACKAGED_TESSDATA", tmp_path / "absent")

    def selected_binary(_: str) -> str:
        return str(binary)

    monkeypatch.setattr(tessdata, "which", selected_binary)
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    assert Path(tessdata.get_tessdata_path("eng")).resolve() == data.resolve()


@pytest.mark.parametrize(
    "language",
    ["../eng", r"..\eng", "eng/../../deu", "eng\x00", "eng+", "C:eng", "eng "],
)
def test_cross_platform_language_validation(language: str) -> None:
    with pytest.raises(ValueError):
        tessdata.get_tessdata_path(language)


def test_script_language(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "script").mkdir()
    (tmp_path / "script" / "Latin.traineddata").touch()
    monkeypatch.setattr(tessdata, "_PACKAGED_TESSDATA", tmp_path / "absent")
    monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path))
    assert tessdata.get_tessdata_path("script/Latin") == str(tmp_path)


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("localhost:11435", "http://127.0.0.1:11435"),
        ("0.0.0.0", "http://127.0.0.1:11434"),
        ("[::]:11435", "http://[::1]:11435"),
        ("[::1]", "http://[::1]:11434"),
        (" https://server.example/ ", "https://server.example:443"),
        ("http://localhost", "http://127.0.0.1:80"),
    ],
)
def test_ollama_host(host: str, expected: str) -> None:
    assert normalize_ollama_host(host.strip()) == expected
    config = Settings(LLM_BASE_URL="", OLLAMA_HOST=host)
    assert config.resolved_llm_base_url == expected


@pytest.mark.parametrize(
    "host",
    [
        "ftp://localhost",
        "host:99999",
        "host:bad",
        "host:0",
        "http://",
        "http://user:pass@host",
        "host/path",
        "host?token=secret",
        "host#fragment",
    ],
)
def test_invalid_ollama_host(host: str) -> None:
    with pytest.raises(ValueError):
        normalize_ollama_host(host)


def test_endpoint_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "OLLAMA_HOST", "localhost:11435")
    monkeypatch.setattr(settings, "LLM_BASE_URL", "")
    assert resolve_connection("ollama", None)[1] == "http://127.0.0.1:11435"
    assert resolve_connection("vllm", None)[1] == "http://127.0.0.1:8000"
    monkeypatch.setattr(settings, "LLM_BASE_URL", "http://127.0.0.1:11436")
    assert resolve_connection("ollama", None)[1].endswith(":11436")
    assert resolve_connection("ollama", "http://127.0.0.1:11437")[1].endswith(":11437")
    with pytest.raises(ValueError, match="HTTPS"):
        request_options(normalize_ollama_host("remote.example:11434"))


def test_doctor_missing_dependencies(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def missing_binary(_: str) -> None:
        return None

    monkeypatch.setattr(deps, "which", missing_binary)

    def missing_data(_: str) -> str:
        raise FileNotFoundError("Install eng and deu data")

    def missing_module(_: str) -> object:
        raise OSError("Missing shared library")

    monkeypatch.setattr(deps, "get_tessdata_path", missing_data)
    monkeypatch.setattr(deps.importlib, "import_module", missing_module)
    assert deps.main(["--video"]) == 1
    output = capsys.readouterr().out
    assert '"name": "ffprobe"' in output
    assert "Missing shared library" in output
    assert "Install eng and deu data" in output
    assert '"status": "skipped"' in output


@pytest.mark.parametrize(
    ("payload", "status_code", "expected"),
    [
        ({"models": [{"name": "test:latest"}]}, 200, "ok"),
        ({"models": [{"name": "other"}]}, 200, "error"),
        ({"models": "invalid"}, 200, "error"),
        ({"models": []}, 302, "error"),
    ],
)
def test_doctor_llm(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
    status_code: int,
    expected: str,
) -> None:
    monkeypatch.setattr(settings, "LLM_ENABLED", True)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "LLM_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setattr(settings, "LLM_MODEL", "test")

    def get(url: str, **kwargs: object) -> requests.Response:
        assert url.endswith("/api/tags")
        assert kwargs["timeout"] == 5
        assert kwargs["allow_redirects"] is False
        response = requests.Response()
        response.status_code = status_code
        response._content = json.dumps(payload).encode()
        return response

    monkeypatch.setattr(requests, "get", get)
    assert deps.check_llm().status == expected


def test_disabled_doctor_never_connects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "LLM_ENABLED", False)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Unexpected network request")

    monkeypatch.setattr(requests, "get", forbidden)
    assert deps.check_llm().status == "skipped"


@pytest.mark.parametrize("available", [True, False])
def test_provision_uses_one_endpoint_without_starting_service(
    tmp_path: Path, available: bool
) -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "provision_ollama_gemma4.sh"
    )
    log = tmp_path / "calls"
    binary = tmp_path / "ollama"
    binary.write_text(
        '#!/bin/sh\nprintf "%s %s\\n" "$OLLAMA_HOST" "$*" >> "$TEST_LOG"\n[ "$1" != list ] || [ "$TEST_AVAILABLE" = yes ]\n',
        encoding="utf-8",
    )
    binary.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        TEST_LOG=str(log),
        TEST_AVAILABLE="yes" if available else "no",
        LLM_BASE_URL="http://127.0.0.1:11439",
        OLLAMA_HOST="wrong:11434",
    )
    result = subprocess.run(
        ["bash", str(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == (0 if available else 1)
    calls = log.read_text().splitlines()
    assert len(calls) == (4 if available else 1)
    assert all(line.startswith("http://127.0.0.1:11439 ") for line in calls)
    assert not any("serve" in line for line in calls)
