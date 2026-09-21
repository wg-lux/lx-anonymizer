"""Provider wire contracts, using synthetic data only (no live clinical requests)."""

from typing import cast

import pytest

from lx_anonymizer.config import settings
from lx_anonymizer.llm.llm_extractor import LLMMetadataExtractor
from lx_anonymizer.llm.llm_service import LLMService
from lx_anonymizer.llm.responses import LLMServiceError, parse_chat_response


@pytest.mark.parametrize(
    "provider,payload",
    [
        (
            "ollama",
            {
                "model": "gemma4:e2b",
                "done": True,
                "done_reason": "stop",
                "eval_count": 10,
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "thinking": "private",
                },
            },
        ),
        (
            "vllm",
            {
                "id": "chat-1",
                "object": "chat.completion",
                "usage": {"total_tokens": 10},
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "logprobs": None,
                        "message": {
                            "role": "assistant",
                            "content": "answer",
                            "reasoning_content": "private",
                        },
                    }
                ],
            },
        ),
    ],
)
def test_realistic_response_envelopes(
    provider: str, payload: dict[str, object]
) -> None:
    parsed = parse_chat_response(payload, provider)
    assert parsed.message is not None
    assert parsed.message.content == "answer"


@pytest.mark.parametrize(
    "provider,payload",
    [
        ("ollama", {"done": False, "message": {"content": "partial"}}),
        ("ollama", {"done_reason": "length", "message": {"content": "partial"}}),
        ("ollama", {"message": {"content": "", "thinking": "not an answer"}}),
        ("ollama", {"message": {"content": 42}}),
        ("ollama", {"error": "sensitive backend details"}),
        (
            "vllm",
            {
                "choices": [
                    {"finish_reason": "length", "message": {"content": "partial"}}
                ]
            },
        ),
        (
            "vllm",
            {
                "choices": [
                    {
                        "finish_reason": "content_filter",
                        "message": {"content": "filtered"},
                    }
                ]
            },
        ),
        (
            "vllm",
            {
                "choices": [
                    {"message": {"content": None, "reasoning_content": "private"}}
                ]
            },
        ),
        ("vllm", {"choices": [{"message": {"content": "answer", "refusal": "no"}}]}),
        ("vllm", {"choices": []}),
        ("vllm", []),
    ],
)
def test_invalid_answers_fail_closed(provider: str, payload: object) -> None:
    with pytest.raises(LLMServiceError):
        parse_chat_response(payload, provider)


@pytest.mark.parametrize(
    "provider,model",
    [
        ("ollama", "lx-gemma4-e2b-json:latest"),
        ("ollama", "qwen3:8b"),
        ("vllm", "Qwen/Qwen2.5-1.5B-Instruct"),
        ("vllm", "glm-5.2"),
    ],
)
def test_discovery_and_inference_contract(
    monkeypatch: pytest.MonkeyPatch, provider: str, model: str
) -> None:
    monkeypatch.setattr(settings, "LLM_ENABLED", True)
    captured: dict[str, object] = {}

    class Response:
        status_code = 200

        def __init__(self, payload: object) -> None:
            self.payload = payload

        def json(self) -> object:
            return self.payload

    def get(url: str, **kwargs: object) -> Response:
        if provider == "ollama":
            return Response({"models": [{"name": model}]})
        assert url.endswith("/v1/models")
        return Response(
            {
                "object": "list",
                "data": [
                    {"id": model, "object": "model", "owned_by": "local", "created": 1}
                ],
            }
        )

    def post(url: str, **kwargs: object) -> Response:
        captured.update(kwargs)
        content = '{"first_name":null,"last_name":null,"dob":null,"casenumber":null,"examination_date":null}'
        if provider == "ollama":
            return Response(
                {
                    "model": model,
                    "done": True,
                    "message": {"content": content, "thinking": "private"},
                }
            )
        return Response(
            {
                "model": model,
                "usage": {},
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
            }
        )

    monkeypatch.setattr("requests.get", get)
    monkeypatch.setattr("requests.post", post)
    extractor = LLMMetadataExtractor(
        provider=provider, base_url="http://127.0.0.1:8000", preferred_model=model
    )
    assert extractor.current_model is not None
    assert extractor.current_model.name == model
    assert (
        extractor.extract_metadata("Synthetic report without personal details")
        is not None
    )
    payload = cast(dict[str, object], captured["json"])
    assert payload["model"] == model
    if provider == "ollama":
        schema = cast(dict[str, object], payload["format"])
    else:
        response_format = cast(dict[str, object], payload["response_format"])
        assert response_format["type"] == "json_schema"
        json_schema = cast(dict[str, object], response_format["json_schema"])
        schema = cast(dict[str, object], json_schema["schema"])
    assert schema["additionalProperties"] is False
    assert schema["required"] == [
        "first_name",
        "last_name",
        "dob",
        "casenumber",
        "examination_date",
    ]


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_chunk_size_invariant(chunk_size: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        LLMService().correct_ocr_text_in_chunks("text", chunk_size=chunk_size)


def test_disabled_chunked_inference_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "LLM_ENABLED", False)
    with pytest.raises(LLMServiceError, match="disabled"):
        LLMService().correct_ocr_text_in_chunks("text")


@pytest.mark.parametrize("status", [400, 401, 429, 503])
def test_http_failure_has_no_hidden_retry_or_cache_entry(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    monkeypatch.setattr(settings, "LLM_ENABLED", True)
    posts: list[str] = []

    class Response:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

        def json(self) -> dict[str, object]:
            return {"models": [{"name": "local-model"}]}

    def get(url: str, **kwargs: object) -> Response:
        return Response(200)

    def post(url: str, **kwargs: object) -> Response:
        posts.append(url)
        return Response(status)

    monkeypatch.setattr("requests.get", get)
    monkeypatch.setattr("requests.post", post)
    extractor = LLMMetadataExtractor(
        provider="ollama",
        base_url="http://127.0.0.1:11434",
        preferred_model="local-model",
    )
    assert extractor.extract_metadata("Synthetic report") is None
    assert len(posts) == 1
    assert extractor.cache is not None
    assert extractor.cache.get_stats().cache_size == 0
