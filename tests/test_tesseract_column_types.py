from collections.abc import Mapping

import numpy as np
import pytest
from PIL import Image

from lx_anonymizer.ocr.medical_pytesseract import (
    MedicalDocumentType,
    extract_medical_text,
)
from lx_anonymizer.ocr.ocr_multi_scale import pyramid_ocr


@pytest.mark.parametrize("confidence", [90, "90"])
def test_pyramid_accepts_numeric_and_string_confidence(
    monkeypatch: pytest.MonkeyPatch, confidence: int | str
) -> None:
    def image_to_string(*args: object, **kwargs: object) -> str:
        return "Patient"

    def image_to_data(*args: object, **kwargs: object) -> dict[str, object]:
        return {"text": ["Patient", ""], "conf": [confidence, -1]}

    monkeypatch.setattr("pytesseract.image_to_string", image_to_string)
    monkeypatch.setattr("pytesseract.image_to_data", image_to_data)
    text, metadata = pyramid_ocr(
        Image.new("RGB", (40, 40)), scales=[1.0], preprocessing_methods=[]
    )
    assert text == "Patient"
    assert metadata["selected_confidence"] == 90.0


@pytest.mark.parametrize("invalid_column", [42, "Patient", None])
def test_medical_ocr_rejects_non_column_payloads(invalid_column: object) -> None:
    def image_to_data(
        image: Image.Image, *, lang: str, config: str
    ) -> Mapping[str, object]:
        return {"text": invalid_column}

    with pytest.raises(TypeError, match="Tesseract column 'text' must be a sequence"):
        extract_medical_text(
            np.zeros((40, 120), dtype=np.uint8),
            MedicalDocumentType.VIDEO,
            image_to_data=image_to_data,
        )
