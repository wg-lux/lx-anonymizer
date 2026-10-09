from pathlib import Path

import pytest

from lx_anonymizer.config import Settings, settings
from lx_anonymizer.llm.connection import request_options, resolve_connection
from lx_anonymizer.llm.factory import LLMFactory
from lx_anonymizer.llm.llm_service import LLMService, LLMServiceError


@pytest.mark.parametrize("provider,port", [("ollama", 11434), ("vllm", 8000)])
def test_provider_defaults(monkeypatch: pytest.MonkeyPatch, provider: str, port: int):
    monkeypatch.setattr(settings, "LLM_BASE_URL", "")
    assert resolve_connection(provider, None) == (provider, f"http://127.0.0.1:{port}")


def test_disabled_prevents_probe_and_direct_inference(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "LLM_ENABLED", False)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Disabled LLM attempted network access")

    monkeypatch.setattr("requests.get", forbidden)
    monkeypatch.setattr("requests.post", forbidden)
    assert LLMFactory.create_metadata_extractor().current_model is None
    with pytest.raises(LLMServiceError, match="disabled"):
        LLMService().correct_ocr_text("Synthetic OCR")


def test_environment_contract(monkeypatch: pytest.MonkeyPatch):
    for name, value in {
        "LLM_ENABLED": "false",
        "LLM_PROVIDER": "vllm",
        "LLM_BASE_URL": "https://llm.example",
        "LLM_MODEL": "glm",
        "LLM_TIMEOUT": "45",
    }.items():
        monkeypatch.setenv(name, value)
    config = Settings()
    assert not config.LLM_ENABLED
    assert config.LLM_PROVIDER == "vllm"
    assert config.resolved_llm_base_url == "https://llm.example"
    assert config.LLM_MODEL == "glm"
    assert config.LLM_TIMEOUT == 45


def test_missing_configured_model_is_unavailable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "LLM_ENABLED", True)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "vllm")
    monkeypatch.setattr(settings, "LLM_BASE_URL", "http://127.0.0.1:8001")
    monkeypatch.setattr(settings, "LLM_MODEL", "configured-glm")
    calls: list[str] = []

    class Response:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {"data": [{"id": "other-model"}]}

    def get(url: str, **kwargs: object) -> Response:
        calls.append(url)
        return Response()

    monkeypatch.setattr("requests.get", get)
    extractor = LLMFactory.create_metadata_extractor()
    assert calls == ["http://127.0.0.1:8001/v1/models"]
    assert extractor.current_model is None


@pytest.mark.parametrize(
    "url",
    [
        "http://llm.example:11434",
        "ftp://127.0.0.1",
        "https://user:password@llm.example",
        "https://llm.example?token=x",
    ],
)
def test_unsafe_transport_rejected(url: str):
    with pytest.raises(ValueError):
        request_options(url)


def test_remote_mtls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    paths: list[str] = []
    for setting in ("LLM_CA_FILE", "LLM_CLIENT_CERT_FILE", "LLM_CLIENT_KEY_FILE"):
        path = tmp_path / setting
        path.touch(mode=0o600)
        paths.append(str(path))
        monkeypatch.setattr(settings, setting, str(path))
    options = request_options("https://llm.example")
    assert options["verify"] == paths[0]
    assert options["cert"] == (paths[1], paths[2])
    assert options["allow_redirects"] is False
    Path(paths[2]).chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        request_options("https://llm.example")


@pytest.mark.parametrize("model", ["local-model", "lx-gemma4-e2b-json"])
def test_ollama_latest_alias_and_cloud_filter(monkeypatch: pytest.MonkeyPatch, model: str):
    monkeypatch.setattr(settings, "LLM_ENABLED", True)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "LLM_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setattr(settings, "LLM_MODEL", model)

    class Response:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {
                "models": [
                    {"name": f"{model}:latest"},
                    {"name": "remote-alias", "remote_host": "https://cloud.example"},
                ]
            }

    def get(url: str, **kwargs: object) -> Response:
        return Response()

    monkeypatch.setattr("requests.get", get)
    extractor = LLMFactory.create_metadata_extractor()
    assert extractor.current_model is not None
    assert extractor.current_model.name == f"{model}:latest"
    assert extractor.available_models == [f"{model}:latest"]


@pytest.mark.parametrize(
    "provider,ocr_model,expected",
    [
        ("ollama", "", "gemma4:e2b"),
        ("vllm", "", "configured-metadata"),
        ("ollama", "custom-vision", "custom-vision"),
        ("vllm", "custom-vision", "custom-vision"),
    ],
)
def test_ocr_model_selection(
    monkeypatch: pytest.MonkeyPatch, provider: str, ocr_model: str, expected: str
) -> None:
    monkeypatch.setattr(settings, "LLM_MODEL", "configured-metadata")
    monkeypatch.setattr(settings, "LLM_OCR_MODEL", ocr_model)
    service = LLMService(provider=provider)
    payload = service._build_payload("OCR", "Synthetic text")
    assert payload.model == expected
    assert LLMService(provider=provider, model_name="explicit").model_name == "explicit"


def test_blank_ocr_model_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "LLM_OCR_MODEL", "  ")
    with pytest.raises(ValueError, match="OCR model must not be empty"):
        LLMService(provider="ollama")


def test_luxnix_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("LLM_MODEL", "LLM_OCR_MODEL", "LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    config = Settings()
    assert config.LLM_MODEL == "lx-gemma4-e2b-json"
    assert config.llm_ocr_model_for(config.LLM_PROVIDER) == "gemma4:e2b"
