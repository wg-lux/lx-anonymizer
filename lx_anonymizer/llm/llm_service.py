import base64
from io import BytesIO
import logging
from collections.abc import Sequence
from typing import Mapping, Optional, TypedDict, cast

import requests
from PIL import Image
from lx_anonymizer.config import settings
from lx_dtypes.models.contracts.llm_service import (
    LLMChatMessagePayload,
    LLMChatOllamaPayload,
    LLMChatOllamaOptionsPayload,
    LLMChatOpenAIPayload,
)

logger = logging.getLogger(__name__)


class _OllamaVisionMessagePayload(TypedDict):
    role: str
    content: str
    images: list[str]


class _OllamaVisionOptionsPayload(TypedDict):
    temperature: int
    num_ctx: int


class _OllamaVisionRequestPayload(TypedDict):
    model: str
    messages: list[_OllamaVisionMessagePayload]
    stream: bool
    options: _OllamaVisionOptionsPayload


class LLMServiceError(RuntimeError):
    """Raised when an LLM operation violates its provider contract."""


class LLMService:
    """Helper for OCR cleanup against either an OpenAI-compatible backend or Ollama."""

    def __init__(
        self,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.provider = (provider or settings.LLM_PROVIDER or "vllm").strip().lower()
        resolved_base_url = base_url or settings.resolved_llm_base_url
        self.base_url = resolved_base_url.rstrip("/")
        self.model_name = model_name or settings.LLM_MODEL
        self.timeout = timeout or settings.LLM_TIMEOUT

    def _chat(self, prompt: str) -> str:
        system_prompt = (
            "You correct OCR text from German medical reports. "
            "Return only the corrected text, preserving names, dates, "
            "and identifiers when they are readable."
        )
        request_payload = self._build_payload(system_prompt, prompt)
        response = requests.post(
            self._chat_endpoint(),
            timeout=self.timeout,
            headers={"Content-Type": "application/json"},
            json=cast(dict[str, object], request_payload.model_dump(mode="json")),
        )
        response.raise_for_status()
        payload: Mapping[str, object] = cast(Mapping[str, object], response.json())
        return self._extract_response_content(payload).strip()

    def _chat_endpoint(self) -> str:
        if self.provider == "ollama":
            return f"{self.base_url}/api/chat"
        return f"{self.base_url}/v1/chat/completions"

    def _build_payload(
        self, system_prompt: str, prompt: str
    ) -> LLMChatOllamaPayload | LLMChatOpenAIPayload:
        messages = [
            LLMChatMessagePayload(role="system", content=system_prompt),
            LLMChatMessagePayload(role="user", content=prompt),
        ]
        if self.provider == "ollama":
            return LLMChatOllamaPayload(
                model=self.model_name,
                messages=messages,
                stream=False,
                options=LLMChatOllamaOptionsPayload(temperature=0, num_ctx=8192),
            )
        return LLMChatOpenAIPayload(
            model=self.model_name,
            temperature=0.0,
            max_tokens=1024,
            top_p=1.0,
            messages=messages,
        )

    @staticmethod
    def _extract_response_content(payload: Mapping[str, object]) -> str:
        ollama_content = _response_message_content(payload.get("message"))
        if ollama_content is not None:
            return ollama_content

        choices = payload.get("choices")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
            return ""
        if not choices:
            return ""
        choice_values = cast(Sequence[object], choices)
        first_choice = choice_values[0]
        if not isinstance(first_choice, Mapping):
            return ""
        choice_mapping = cast(Mapping[object, object], first_choice)
        return _response_message_content(choice_mapping.get("message")) or ""

    def correct_ocr_text(self, text: str) -> str:
        if not text:
            return text
        prompt = f"Correct this OCR text and return only the corrected text:\n\n{text}"
        return self._chat(prompt) or text

    def correct_ocr_text_in_chunks(self, text: str, chunk_size: int = 2048) -> str:
        if not text:
            return text

        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        corrected_chunks: list[str] = []
        for chunk in chunks:
            try:
                corrected_chunks.append(self.correct_ocr_text(chunk))
            except Exception as exc:
                logger.warning("LLM OCR correction failed for one chunk: %s", exc)
                corrected_chunks.append(chunk)
        return "".join(corrected_chunks)

    def recognize_image(self, image: Image.Image, candidate_text: str = "") -> str:
        """Transcribe visible text from an image with Ollama's vision endpoint."""
        if self.provider != "ollama":
            raise LLMServiceError("Vision OCR currently requires the Ollama provider")

        prompt = (
            "Transcribe every visible character in this medical image. Preserve "
            "line breaks, names, dates, identifiers, punctuation, and original "
            "spelling. Do not summarize, translate, infer, or add text. Return "
            "only the transcription. Return [NO_TEXT] when no text is visible."
        )
        stripped_candidate = candidate_text.strip()
        if stripped_candidate:
            prompt += (
                " A conventional OCR engine produced the candidate below. Use the "
                "image as the source of truth and correct only recognition errors.\n\n"
                f"OCR_CANDIDATE:\n{stripped_candidate}"
            )

        payload = self._build_ollama_vision_payload(image, prompt)
        response = requests.post(
            self._chat_endpoint(),
            timeout=self.timeout,
            headers={"Content-Type": "application/json"},
            json=cast(dict[str, object], payload),
        )
        response.raise_for_status()
        raw_response = response.json()
        if not isinstance(raw_response, Mapping):
            raise LLMServiceError("Ollama vision response must be a JSON object")
        content = self._extract_response_content(
            cast(Mapping[str, object], raw_response)
        ).strip()
        if content == "[NO_TEXT]":
            return ""
        return content

    def _build_ollama_vision_payload(
        self, image: Image.Image, prompt: str
    ) -> _OllamaVisionRequestPayload:
        encoded_image = _encode_image_as_png(image)
        return {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [encoded_image],
                }
            ],
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 8192},
        }


def _encode_image_as_png(image: Image.Image) -> str:
    normalized = image.convert("RGB")
    buffer = BytesIO()
    normalized.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _response_message_content(value: object) -> str | None:
    """Project provider-specific message objects onto the stable content field."""
    if not isinstance(value, Mapping):
        return None
    message = cast(Mapping[object, object], value)
    content = message.get("content")
    return content if isinstance(content, str) else None
