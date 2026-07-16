from __future__ import annotations

import base64
from io import BytesIO
import json
from typing import Mapping, cast

import pytest
from PIL import Image

from lx_anonymizer.llm import llm_service
from lx_anonymizer.llm.llm_service import LLMService, LLMServiceError


class _FakeResponse:
    def __init__(self, payload: Mapping[str, object]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Mapping[str, object]:
        return self.payload


def test_ollama_vision_ocr_sends_png_and_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return _FakeResponse(
            {
                "model": "gemma4:e2b",
                "done": True,
                "message": {
                    "role": "assistant",
                    "content": "ID 123",
                    "thinking": "internal reasoning",
                },
            }
        )

    monkeypatch.setattr(llm_service.requests, "post", fake_post)
    service = LLMService(
        provider="ollama",
        base_url="http://ollama.internal:11434",
        model_name="gemma4:e2b",
        timeout=12,
    )

    result = service.recognize_image(
        Image.new("RGB", (16, 8), "white"), candidate_text="1D 123"
    )

    assert result == "ID 123"
    assert captured["url"] == "http://ollama.internal:11434/api/chat"
    payload = cast(dict[str, object], captured["json"])
    assert payload["model"] == "gemma4:e2b"
    messages = cast(list[dict[str, object]], payload["messages"])
    assert "1D 123" in cast(str, messages[0]["content"])
    images = cast(list[str], messages[0]["images"])
    decoded = base64.b64decode(images[0], validate=True)
    with Image.open(BytesIO(decoded)) as encoded_image:
        assert encoded_image.size == (16, 8)
        assert encoded_image.format == "PNG"


def test_ollama_vision_ocr_normalizes_no_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(*args: object, **kwargs: object) -> _FakeResponse:
        return _FakeResponse({"message": {"role": "assistant", "content": "[NO_TEXT]"}})

    monkeypatch.setattr(
        llm_service.requests,
        "post",
        fake_post,
    )
    service = LLMService(provider="ollama", model_name="gemma4:e2b")

    assert service.recognize_image(Image.new("L", (4, 4))) == ""


def test_ollama_text_correction_serializes_typed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        json.dumps(kwargs["json"])
        return _FakeResponse(
            {
                "model": "gemma4:e2b",
                "done": True,
                "eval_count": 12,
                "message": {
                    "role": "assistant",
                    "content": "Patient Lux",
                    "thinking": "internal reasoning",
                },
            }
        )

    monkeypatch.setattr(llm_service.requests, "post", fake_post)
    service = LLMService(
        provider="ollama",
        base_url="http://ollama.internal:11434",
        model_name="gemma4:e2b",
    )

    result = service.correct_ocr_text("Patlent Lux")

    assert result == "Patient Lux"
    assert captured["url"] == "http://ollama.internal:11434/api/chat"
    payload = cast(dict[str, object], captured["json"])
    assert payload["model"] == "gemma4:e2b"
    assert isinstance(payload["messages"], list)
    assert isinstance(payload["options"], dict)


def test_vision_ocr_rejects_non_ollama_provider() -> None:
    service = LLMService(provider="vllm", model_name="gemma4:e2b")

    with pytest.raises(LLMServiceError, match="requires the Ollama provider"):
        service.recognize_image(Image.new("RGB", (4, 4)))
