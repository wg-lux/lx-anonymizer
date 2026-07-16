from __future__ import annotations

import pytest
from PIL import Image

from lx_anonymizer import report_reader as report_reader_module
from lx_anonymizer.llm.llm_service import LLMService
from lx_anonymizer.report_reader import ReportReader


def test_report_reader_uses_gemma4_vision_ocr_with_tesseract_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def conventional_ocr(image: Image.Image) -> tuple[str, list[object]]:
        return "Patlent Müller", []

    def recognize_image(
        self: LLMService, image: Image.Image, candidate_text: str = ""
    ) -> str:
        observed["candidate"] = candidate_text
        observed["size"] = image.size
        return "Patient Müller"

    monkeypatch.setattr(
        report_reader_module, "tesseract_full_image_ocr", conventional_ocr
    )
    monkeypatch.setattr(LLMService, "recognize_image", recognize_image)
    reader = ReportReader.__new__(ReportReader)
    reader.ollama_available = True

    result = reader._ocr_image(  # pyright: ignore[reportPrivateUsage]
        Image.new("RGB", (80, 40), "white"), page_num=1, use_ensemble=False
    )

    assert result == "Patient Müller"
    assert observed == {"candidate": "Patlent Müller", "size": (80, 40)}


def test_report_reader_keeps_tesseract_text_when_gemma4_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def conventional_ocr(image: Image.Image) -> tuple[str, list[object]]:
        return "Patient Müller", []

    monkeypatch.setattr(
        report_reader_module,
        "tesseract_full_image_ocr",
        conventional_ocr,
    )

    def fail_recognition(
        self: LLMService, image: Image.Image, candidate_text: str = ""
    ) -> str:
        raise ConnectionError("Ollama unavailable")

    monkeypatch.setattr(LLMService, "recognize_image", fail_recognition)
    reader = ReportReader.__new__(ReportReader)
    reader.ollama_available = True

    result = reader._ocr_image(  # pyright: ignore[reportPrivateUsage]
        Image.new("RGB", (80, 40), "white"), page_num=1, use_ensemble=False
    )

    assert result == "Patient Müller"
