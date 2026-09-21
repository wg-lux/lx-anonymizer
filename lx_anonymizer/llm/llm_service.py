import base64
import logging
from io import BytesIO
from typing import Optional, TypedDict, cast

import requests
from lx_dtypes.models.contracts.llm_service import (
    LLMChatMessagePayload,
    LLMChatOllamaOptionsPayload,
    LLMChatOllamaPayload,
    LLMChatOpenAIPayload,
)
from PIL import Image

from lx_anonymizer.config import settings
from lx_anonymizer.llm.connection import request_options, resolve_connection
from lx_anonymizer.llm.responses import LLMServiceError as LLMServiceError
from lx_anonymizer.llm.responses import parse_chat_response

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


class LLMService:
    """Helper for OCR cleanup against either an OpenAI-compatible backend or Ollama."""

    def __init__(
        self,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.provider, self.base_url = resolve_connection(provider, base_url)
        self.model_name = (
            settings.LLM_MODEL if model_name is None else model_name
        ).strip()
        self.timeout = settings.LLM_TIMEOUT if timeout is None else timeout
        if not self.model_name:
            raise ValueError("LLM_MODEL must not be empty")
        if not 1 <= self.timeout <= 120:
            raise ValueError("LLM timeout must be between 1 and 120 seconds")

    def _chat(self, prompt: str) -> str:
        if not settings.LLM_ENABLED:
            raise LLMServiceError("LLM functionality is disabled")
        system_prompt = (
            "You correct OCR text from German medical reports. "
            "Return only the corrected text, preserving names, dates, "
            "and identifiers when they are readable."
        )
        request_payload = self._build_payload(system_prompt, prompt)
        response = requests.post(
            self._chat_endpoint(),
            timeout=self.timeout,
            **request_options(self.base_url),
            headers={"Content-Type": "application/json"},
            json=cast(
                dict[str, object],
                request_payload.model_dump(mode="json", exclude_none=True),
            ),
        )
        response.raise_for_status()
        return self._extract_response_content(response.json()).strip()

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

    def _extract_response_content(self, payload: object) -> str:
        response = parse_chat_response(payload, self.provider)
        assert response.message is not None
        return response.message.content

    def correct_ocr_text(self, text: str) -> str:
        if not text:
            return text
        prompt = f"Correct this OCR text and return only the corrected text:\n\n{text}"
        return self._chat(prompt) or text

    def correct_ocr_text_in_chunks(self, text: str, chunk_size: int = 2048) -> str:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not settings.LLM_ENABLED:
            raise LLMServiceError("LLM functionality is disabled")
        if not text:
            return text

        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        corrected_chunks: list[str] = []
        for chunk in chunks:
            try:
                corrected_chunks.append(self.correct_ocr_text(chunk))
            except (requests.RequestException, LLMServiceError, ValueError) as exc:
                logger.warning(
                    "LLM OCR correction failed; preserving original chunk (%s)",
                    type(exc).__name__,
                )
                corrected_chunks.append(chunk)
        return "".join(corrected_chunks)

    def recognize_image(self, image: Image.Image, candidate_text: str = "") -> str:
        """Transcribe visible text using a configured vision-capable model."""
        if not settings.LLM_ENABLED:
            raise LLMServiceError("LLM functionality is disabled")

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

        payload = (
            cast(dict[str, object], self._build_ollama_vision_payload(image, prompt))
            if self.provider == "ollama"
            else self._build_compatible_vision_payload(image, prompt)
        )
        response = requests.post(
            self._chat_endpoint(),
            timeout=self.timeout,
            **request_options(self.base_url),
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        content = self._extract_response_content(response.json()).strip()
        if content == "[NO_TEXT]":
            return ""
        return content

    def _build_compatible_vision_payload(
        self, image: Image.Image, prompt: str
    ) -> dict[str, object]:
        return {
            "model": self.model_name,
            "stream": False,
            "temperature": 0,
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_encode_image_as_png(image)}"
                            },
                        },
                    ],
                }
            ],
        }

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
