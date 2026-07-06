import logging
from typing import Optional
from typing import Mapping, cast

import requests
from pydantic import ValidationError

from lx_anonymizer.config import settings
from lx_dtypes.models.contracts.llm_service import (
    LLMChatMessagePayload,
    LLMChatOllamaPayload,
    LLMChatOllamaOptionsPayload,
    LLMChatOpenAIPayload,
    LLMChatResponsePayload,
)

logger = logging.getLogger(__name__)


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
        response = requests.post(
            self._chat_endpoint(),
            timeout=self.timeout,
            headers={"Content-Type": "application/json"},
            json=cast(dict[str, object], self._build_payload(system_prompt, prompt)),
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
        try:
            parsed_payload = LLMChatResponsePayload.model_validate(payload)
        except ValidationError:
            return ""

        if parsed_payload.message is not None:
            return parsed_payload.message.content or ""

        if not parsed_payload.choices:
            return ""

        return parsed_payload.choices[0].message.content or ""

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
