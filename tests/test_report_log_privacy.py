"""Report diagnostics retain phases without exposing document or error payloads."""

import logging
from pathlib import Path

import pytest
from PIL import Image

from lx_anonymizer.anonymization.anonymizer import Anonymizer
from lx_anonymizer.anonymization.sensitive_region_cropper import SensitiveRegionCropper
from lx_anonymizer.report_reader.extraction import ReportReaderExtractionMixin

SENTINEL = "PRIVATE_SENTINEL_593"


def test_sensitive_region_matches_are_not_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
) -> None:
    cropper = object.__new__(SensitiveRegionCropper)
    cropper.margin = 0
    cropper.min_region_size = (1, 1)
    cropper.sensitive_patterns = {"sentinel": SENTINEL}

    def extract(_text: str) -> None:
        return None

    monkeypatch.setattr(cropper, "patient_extractor", extract, raising=False)
    with caplog.at_level(
        logging.DEBUG, logger="lx_anonymizer.anonymization.sensitive_region_cropper"
    ):
        regions = cropper.detect_sensitive_regions(
            Image.new("RGB", (100, 100)), [(SENTINEL, (1, 1, 20, 20))]
        )
    assert regions == [(1, 1, 20, 20)]
    assert "sensitive_region_pattern_matched" in caplog.text
    captured = capfd.readouterr()
    assert SENTINEL not in caplog.text + captured.out + captured.err


@pytest.mark.parametrize("phase", ["pdf_read", "image_redaction", "pdf_redaction"])
def test_backend_exception_and_source_filename_are_not_logged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
    phase: str,
) -> None:
    source = str(tmp_path / f"{SENTINEL}.pdf")
    output = str(tmp_path / f"{SENTINEL}-redacted.pdf")

    def fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError(SENTINEL)

    with caplog.at_level(logging.DEBUG):
        if phase == "pdf_read":
            monkeypatch.setattr(
                "lx_anonymizer.report_reader.extraction.pdfplumber.open", fail
            )
            assert ReportReaderExtractionMixin().read_pdf(source) == ""
            expected_event = "report_pdf_read_failed"
        else:
            anonymizer = object.__new__(Anonymizer)
            if phase == "image_redaction":
                monkeypatch.setattr(
                    "lx_anonymizer.anonymization.anonymizer.Image.open", fail
                )
                assert anonymizer.create_anonymized_image(source, output) is None
                expected_event = "redaction_image_failed"
            else:
                monkeypatch.setattr(
                    "lx_anonymizer.anonymization.anonymizer.convert_pdf_to_images", fail
                )
                assert (
                    anonymizer.create_anonymized_pdf_from_rois(source, {}, output)
                    is None
                )
                expected_event = "redaction_pdf_regions_failed"
    assert expected_event in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    captured = capfd.readouterr()
    assert SENTINEL not in caplog.text + captured.out + captured.err


def test_pdf_failure_traceback_omits_external_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import traceback

    from lx_anonymizer.report_contracts import AnonymizationArtifactError

    def fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError(SENTINEL)

    def tessdata(_language: str) -> str:
        return "/unused"

    monkeypatch.setattr(
        "lx_anonymizer.anonymization.anonymizer.get_tessdata_path", tessdata
    )
    monkeypatch.setattr(
        "lx_anonymizer.anonymization.anonymizer.convert_pdf_to_images", fail
    )
    anonymizer = object.__new__(Anonymizer)
    with pytest.raises(AnonymizationArtifactError) as caught:
        anonymizer.create_anonymized_pdf(
            str(tmp_path / f"{SENTINEL}.pdf"), str(tmp_path / "output.pdf")
        )
    rendered = "".join(traceback.format_exception(caught.value))
    assert SENTINEL not in rendered
    assert "Anonymized PDF creation failed" in rendered
