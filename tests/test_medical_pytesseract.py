from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from PIL import Image

from lx_anonymizer.ocr.medical_pytesseract import (
    MedicalDocumentType,
    OcrRuntimeConfig,
    RoiPolicy,
    build_tesseract_invocation,
    extract_medical_text,
    normalize_ocr_text_for_nlp,
    preprocess_for_medical_ocr,
)


def test_tesseract_invocation_pins_layout_modes_and_dawgs() -> None:
    video_roi = build_tesseract_invocation(
        MedicalDocumentType.VIDEO,
        has_roi=True,
    )
    video_full = build_tesseract_invocation(
        MedicalDocumentType.VIDEO,
        has_roi=False,
    )
    report = build_tesseract_invocation(
        MedicalDocumentType.REPORT,
        has_roi=False,
    )

    assert "--oem 1" in video_roi.config
    assert "--psm 7" in video_roi.config
    assert "--psm 11" in video_full.config
    assert "--psm 6" in report.config
    assert "-c load_system_dawg=0" in report.config
    assert "-c load_freq_dawg=0" in report.config
    assert "-c tessedit_enable_doc_dict=0" in report.config


def test_normalize_ocr_text_reconnects_cut_up_letters_and_dates() -> None:
    messy = "P a t i e n t : J o h n D o e\nD O B : 2 0 2 4 - 0 1 - 1 5\n@@@"

    cleaned = normalize_ocr_text_for_nlp(messy)

    assert cleaned == "Patient: John Doe\nDOB: 2024-01-15"


def test_video_preprocessing_inverts_white_text_on_dark_background() -> None:
    frame = np.zeros((80, 260), dtype=np.uint8)
    cv2.putText(
        frame,
        "2024-01-15",
        (8, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        255,
        2,
        cv2.LINE_AA,
    )

    processed = preprocess_for_medical_ocr(
        frame,
        MedicalDocumentType.VIDEO,
        roi={"x": 0, "y": 0, "width": frame.shape[1], "height": frame.shape[0]},
    )
    arr = np.asarray(processed.image)

    assert processed.inverted is True
    assert processed.thresholding == "otsu_binary"
    assert float(arr.mean()) > 127.0


def test_extract_medical_text_uses_named_roi_fallback_and_records_config() -> None:
    captured: dict[str, object] = {}

    def fake_image_to_data(
        image: Image.Image, *, lang: str, config: str
    ) -> dict[str, list[object]]:
        captured["size"] = image.size
        captured["lang"] = lang
        captured["config"] = config
        return {
            "text": ["J", "o", "h", "n", "D", "o", "e", "2024", "-", "01", "-", "15"],
            "conf": [
                "90",
                "91",
                "89",
                "92",
                "91",
                "90",
                "88",
                "95",
                "93",
                "94",
                "93",
                "94",
            ],
            "left": [0] * 12,
            "top": [0] * 12,
            "width": [10] * 12,
            "height": [10] * 12,
            "block_num": [1] * 12,
            "par_num": [1] * 12,
            "line_num": [1] * 12,
        }

    frame = np.zeros((40, 120), dtype=np.uint8)
    result = extract_medical_text(
        frame,
        MedicalDocumentType.VIDEO,
        roi={"x": 200, "y": 0, "width": 20, "height": 20},
        roi_policy=RoiPolicy.FALLBACK_TO_FULL_IMAGE,
        runtime_config=OcrRuntimeConfig(language="deu+eng"),
        image_to_data=fake_image_to_data,
    )

    assert result.text == "John Doe 2024-01-15"
    assert result.roi_used is None
    assert result.roi_fallback_used is True
    assert result.roi_fallback_reason == "ROI starts outside image bounds"
    assert "--psm 11" in cast(str, captured["config"])
    assert "-c load_system_dawg=0" in result.tesseract_config
    assert captured["lang"] == "deu+eng"


def test_extract_medical_text_can_raise_on_invalid_roi() -> None:
    frame = np.zeros((40, 120), dtype=np.uint8)

    def fake_image_to_data(
        image: Image.Image, *, lang: str, config: str
    ) -> dict[str, list[object]]:
        return {}

    try:
        extract_medical_text(
            frame,
            MedicalDocumentType.VIDEO,
            roi={"x": -1, "y": 0, "width": 20, "height": 20},
            roi_policy=RoiPolicy.RAISE,
            image_to_data=fake_image_to_data,
        )
    except ValueError as exc:
        assert str(exc) == "ROI x and y must be non-negative"
    else:
        raise AssertionError("Expected invalid ROI to raise")
