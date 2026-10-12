from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest
from PIL import Image

from lx_anonymizer.report_reader import reader as report_reader_module
from lx_anonymizer.config import Settings
from lx_anonymizer.llm.llm_service import LLMService
from lx_anonymizer.model_adapters import ReportOCRFactory
from lx_anonymizer.report_reader import ReportReader


@pytest.mark.parametrize("ocr_model", ["", "custom-vision"])
def test_report_vision_block_uses_configured_model(
    monkeypatch: pytest.MonkeyPatch,
    ocr_model: str,
) -> None:
    observed: dict[str, object] = {}

    def recognize_image(
        self: LLMService, image: Image.Image, candidate_text: str = ""
    ) -> str:
        assert self.model_name == (ocr_model or "gemma4:e2b")
        observed["size"] = image.size
        return "Patient Müller"

    monkeypatch.setattr(LLMService, "recognize_image", recognize_image)
    monkeypatch.setattr(report_reader_module.settings, "LLM_OCR_MODEL", ocr_model)
    config = report_reader_module.settings.model_copy(deep=True)
    model = ReportOCRFactory("vision", config)()
    assert model(Image.new("RGB", (80, 40), "white")) == "Patient Müller"
    assert observed == {"size": (80, 40)}


def conventional_ocr(image: Image.Image) -> str:
    assert image.size == (80, 40)
    return "Patient Müller"


@dataclass(frozen=True)
class FailingVisionFactory:
    backend: str
    config: Settings

    def __call__(self) -> Callable[[Image.Image], str]:
        if self.backend == "vision":
            raise ConnectionError("Ollama unavailable")
        return conventional_ocr


def test_report_reader_uses_tesseract_when_vision_cannot_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(report_reader_module, "ReportOCRFactory", FailingVisionFactory)
    monkeypatch.setattr(report_reader_module.settings, "LLM_ENABLED", True)
    monkeypatch.setattr(report_reader_module.settings, "OLLAMA_OCR_ENABLED", True)
    reader = ReportReader.__new__(ReportReader)
    result = reader._ocr_image(  # pyright: ignore[reportPrivateUsage]
        Image.new("RGB", (80, 40), "white"), page_num=1, use_ensemble=False
    )
    assert result == "Patient Müller"
