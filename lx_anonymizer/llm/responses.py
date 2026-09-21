"""Validate provider envelopes before projecting into shared lx-dtypes contracts.

Wire envelopes deliberately tolerate provider telemetry and reasoning fields;
clinical content remains strict. No reasoning text is used as answer content.
"""

from lx_dtypes.models.contracts.llm_service import (
    LLMChatResponseMessagePayload,
    LLMChatResponsePayload,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class LLMServiceError(RuntimeError):
    """An inference response cannot safely be consumed."""


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)


class _Message(_WireModel):
    content: str | None = None
    refusal: str | None = None


class _Choice(_WireModel):
    message: _Message
    finish_reason: str | None = None


def _empty_choices() -> list[_Choice]:
    return []


class _ChatResponse(_WireModel):
    message: _Message | None = None
    choices: list[_Choice] = Field(default_factory=_empty_choices)
    done: bool | None = None
    done_reason: str | None = None
    error: str | None = None


class _Model(_WireModel):
    id: str = Field(min_length=1)


class _Models(_WireModel):
    data: list[_Model]


def parse_model_names(raw: object) -> list[str]:
    """Accept vLLM and llama.cpp model listings including server metadata."""
    return [model.id for model in _Models.model_validate(raw).data]


def parse_chat_response(raw: object, provider: str) -> LLMChatResponsePayload:
    """Return only a validated final answer, never a partial or reasoning answer."""
    try:
        response = _ChatResponse.model_validate(raw)
    except ValidationError:
        raise LLMServiceError("Malformed LLM response envelope") from None
    if response.error is not None:
        raise LLMServiceError("LLM server returned an error")
    if provider == "ollama":
        message = response.message
        if response.done is False or response.done_reason not in (None, "stop"):
            raise LLMServiceError("LLM response is incomplete")
    elif provider == "vllm":
        if len(response.choices) != 1:
            raise LLMServiceError("Expected exactly one LLM response choice")
        choice = response.choices[0]
        if choice.finish_reason not in (None, "stop"):
            raise LLMServiceError("LLM response is incomplete or refused")
        message = choice.message
    else:
        raise ValueError("Unsupported LLM provider")
    if message is None or message.refusal is not None:
        raise LLMServiceError("LLM response is missing or refused")
    if not message.content or not message.content.strip():
        raise LLMServiceError("LLM response has no final text")
    return LLMChatResponsePayload(
        message=LLMChatResponseMessagePayload(role="assistant", content=message.content)
    )
