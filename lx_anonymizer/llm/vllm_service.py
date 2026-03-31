import logging
from typing import Optional

import requests

from lx_anonymizer.config import settings

logger = logging.getLogger(__name__)


class VLLMService:
    """Small helper for OCR cleanup against an OpenAI-compatible vLLM endpoint."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.base_url = (base_url or settings.LLM_BASE_URL).rstrip("/")
        self.model_name = model_name or settings.LLM_MODEL
        self.timeout = timeout or settings.LLM_TIMEOUT

    def _chat(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            timeout=self.timeout,
            headers={"Content-Type": "application/json"},
            json={
                "model": self.model_name,
                "temperature": 0,
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You correct OCR text from German medical reports. "
                            "Return only the corrected text, preserving names, dates, "
                            "and identifiers when they are readable."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            },
        )
        response.raise_for_status()
        payload = response.json()
        return (
            payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

    def correct_ocr_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return text
        prompt = f"Correct this OCR text and return only the corrected text:\n\n{text}"
        return self._chat(prompt) or text

    def correct_ocr_text_in_chunks(self, text: str, chunk_size: int = 2048) -> str:
        if not text or not isinstance(text, str):
            return text

        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        corrected_chunks: list[str] = []
        for chunk in chunks:
            try:
                corrected_chunks.append(self.correct_ocr_text(chunk))
            except Exception as exc:
                logger.warning("vLLM OCR correction failed for one chunk: %s", exc)
                corrected_chunks.append(chunk)
        return "".join(corrected_chunks)
