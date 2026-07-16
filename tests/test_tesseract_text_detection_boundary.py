from __future__ import annotations

import pytest

from lx_anonymizer.text_detection.tesseract_text_detection import (
    normalize_tesseract_ocr_data,
)


def test_normalize_tesseract_ocr_data_discards_standard_extra_columns() -> None:
    result = normalize_tesseract_ocr_data(
        {
            "level": [1],
            "page_num": [1],
            "block_num": [0],
            "par_num": [0],
            "line_num": [0],
            "word_num": [0],
            "left": [10],
            "top": [20],
            "width": [30],
            "height": [40],
            "conf": ["91.5"],
            "text": ["Patient"],
        }
    )

    assert result.left == [10]
    assert result.conf == [91.5]
    assert result.text == ["Patient"]


@pytest.mark.parametrize("payload", [None, [], "invalid"])
def test_normalize_tesseract_ocr_data_rejects_non_mapping(payload: object) -> None:
    with pytest.raises(TypeError):
        normalize_tesseract_ocr_data(payload)


def test_normalize_tesseract_ocr_data_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        normalize_tesseract_ocr_data({"text": []})
