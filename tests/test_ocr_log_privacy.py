"""OCR results remain available to callers without entering worker logs."""

import logging

import pytest
from PIL import Image

from lx_anonymizer.ocr import ocr


@pytest.mark.parametrize("backend_fails", [False, True])
def test_box_ocr_does_not_log_document_or_backend_payload(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    backend_fails: bool,
) -> None:
    sentinel = "PRIVATE_PATIENT_913 /private/patient-913.pdf"

    def recognize(*_args: object, **_kwargs: object) -> str:
        if backend_fails:
            raise RuntimeError(sentinel)
        return sentinel

    def confidence(*_args: object, **_kwargs: object) -> dict[str, list[int]]:
        return {"conf": [90]}

    monkeypatch.setattr(ocr, "_pytesseract_image_to_string", recognize)
    monkeypatch.setattr(ocr, "_pytesseract_image_to_data", confidence)
    box = (0, 0, 4, 4)
    with caplog.at_level(logging.DEBUG, logger=ocr.__name__):
        results, confidences = ocr.tesseract_on_boxes_pytesseract(
            Image.new("RGB", (4, 4)), [box]
        )

    assert results == [("" if backend_fails else sentinel, box)]
    assert confidences == [0.0 if backend_fails else 90.0]
    assert sentinel not in caplog.text
    assert "Tesseract OCR processing complete" in caplog.text
    assert (
        "Error processing box 1/1" if backend_fails else "Processed box 1/1"
    ) in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_ensemble_backend_failures_do_not_log_document_payload(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from lx_anonymizer.ocr import ocr_donut, ocr_ensemble

    sentinel = "PRIVATE_PATIENT_913 /private/patient-913.pdf"

    def fail(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError(sentinel)

    def identity(image: Image.Image, **_kwargs: object) -> Image.Image:
        return image

    monkeypatch.setattr(ocr_ensemble, "optimize_image_for_medical_text", identity)
    monkeypatch.setattr(ocr_ensemble, "preprocess_image", identity)
    monkeypatch.setattr(ocr_ensemble, "_image_to_string", fail)
    monkeypatch.setattr(ocr, "trocr_full_image_ocr", fail)
    monkeypatch.setattr(ocr_donut, "donut_full_image_ocr", fail)
    with caplog.at_level(logging.DEBUG, logger=ocr_ensemble.logger.name):
        result = ocr_ensemble.ensemble_ocr(Image.new("RGB", (4, 4)))
    assert result == ""
    assert sentinel not in caplog.text
    for phase in ("Tesseract OCR failed", "TrOCR failed", "Donut OCR failed"):
        assert phase in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_donut_image_open_failure_does_not_log_source_path(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from lx_anonymizer.ocr import ocr_donut

    sentinel = "/private/PRIVATE_PATIENT_913.pdf"

    def fail(*_args: object, **_kwargs: object) -> Image.Image:
        raise OSError(sentinel)

    monkeypatch.setattr(Image, "open", fail)
    with caplog.at_level(logging.DEBUG, logger=ocr_donut.logger.name):
        result = ocr_donut.donut_full_image_ocr(sentinel)
    assert result == ""
    assert sentinel not in caplog.text
    assert "Error in Donut OCR" in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
