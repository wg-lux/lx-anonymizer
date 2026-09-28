import logging

import pytest

from lx_anonymizer.ner.frame_metadata_extractor import FrameMetadataExtractor


@pytest.mark.parametrize(
    ("text", "expected_dob", "expected_examination_date"),
    [
        (
            "15/02/2024 09:52:17 Temp. Pat.-ID ausgebe Lux, Thomas 29 21/03/1994",
            "1994-03-21",
            "2024-02-15",
        ),
        (
            "DOB: 21.03.1994 Untersuchung: 15.02.2024",
            "1994-03-21",
            "2024-02-15",
        ),
        (
            "Untersuchung: 15-02-2024 geboren: 21-03-1994",
            "1994-03-21",
            "2024-02-15",
        ),
        (
            "15.02.2024 09:52:17",
            None,
            "2024-02-15",
        ),
        (
            "Geburtsdatum: 21.03.1994",
            "1994-03-21",
            None,
        ),
    ],
)
def test_frame_dates_are_resolved_from_shared_candidates(
    text: str,
    expected_dob: str | None,
    expected_examination_date: str | None,
) -> None:
    metadata = FrameMetadataExtractor().extract_metadata_from_frame_text(text)

    assert metadata["dob"] == expected_dob
    assert metadata["examination_date"] == expected_examination_date


def test_one_date_is_never_assigned_to_both_roles() -> None:
    metadata = FrameMetadataExtractor().extract_metadata_from_frame_text(
        "DOB: 21.03.1994 Date: 21.03.1994"
    )

    assert metadata["dob"] == "1994-03-21"
    assert metadata["examination_date"] is None


def test_unlabelled_overlay_dates_use_older_date_as_dob() -> None:
    metadata = FrameMetadataExtractor().extract_metadata_from_frame_text(
        "15 02 2024 09:52:17 Lux, Thomas 21 03 1994"
    )

    assert metadata["dob"] == "1994-03-21"
    assert metadata["examination_date"] == "2024-02-15"


def test_metadata_merge_tracking_reports_fields_without_nullable_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    extractor = FrameMetadataExtractor()

    with caplog.at_level(logging.DEBUG):
        merged = extractor.merge_metadata(
            {"file_path": "video.mp4"},
            {"first_name": "Thomas"},
        )

    assert merged["first_name"] == "Thomas"
    tracking_messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Metadata merge completed:")
    ]
    assert len(tracking_messages) == 1
    assert "first_name" in tracking_messages[0]
    assert "None" not in tracking_messages[0]


def test_metadata_merge_projects_clinical_fields_from_ocr_diagnostics() -> None:
    merged = FrameMetadataExtractor().merge_metadata(
        {
            "backend": "rapidocr",
            "method": "rapidocr+roi",
            "roi_count": 2,
            "roi_0": "15/02/2024 09:51:09",
            "roi_0_elapse": 0.7,
            "text_regions": [{"text": "Thomas"}],
            "processing_time": 1.8,
        },
        {
            "first_name": "Thomas",
            "examination_date": "2024-02-15",
        },
    )

    assert merged["first_name"] == "Thomas"
    assert merged["examination_date"] == "2024-02-15"
    assert "backend" not in merged
    assert "text_regions" not in merged


def test_invalid_observation_does_not_partially_update_metadata(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from lx_anonymizer.sensitive_meta_interface import SensitiveMetaResolutionError

    extractor = FrameMetadataExtractor()
    before = extractor.meta.model_dump()

    def names(_: str) -> tuple[str, None]:
        return "PRIVATE_TEST_NAME", None

    def invalid_time(_: str) -> str:
        return "invalid_PRIVATE_TIME"

    monkeypatch.setattr(extractor, "_extract_patient_names", names)
    monkeypatch.setattr(extractor, "_extract_examination_time", invalid_time)
    with pytest.raises(SensitiveMetaResolutionError) as error:
        extractor.extract_metadata_from_frame_text("nonempty observation")
    assert extractor.meta.model_dump() == before
    assert "examination_time" in str(error.value)
    assert "PRIVATE" not in str(error.value)
    assert "PRIVATE" not in caplog.text


def test_unexpected_extractor_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    extractor = FrameMetadataExtractor()

    def broken(_: str) -> tuple[str, str]:
        raise RuntimeError("unexpected extractor failure")

    monkeypatch.setattr(extractor, "_extract_patient_names", broken)
    with pytest.raises(RuntimeError, match="unexpected extractor failure"):
        extractor.extract_metadata_from_frame_text("nonempty")


@pytest.mark.parametrize(
    "text,gender",
    [
        ("Geschlecht: M", "male"),
        ("Geschlecht: w", "female"),
        ("female", "female"),
        ("männlich", "male"),
        ("Geburtsdatum Untersuchung", "unknown"),
        ("Temp. Aufnahme", "unknown"),
    ],
)
def test_gender_tokens_use_shared_schema(text: str, gender: str) -> None:
    assert (
        FrameMetadataExtractor().extract_metadata_from_frame_text(text)["gender"]
        == gender
    )


def test_regex_configuration_error_is_not_suppressed() -> None:
    import re

    extractor = FrameMetadataExtractor()
    extractor.patient_patterns = ["("]
    with pytest.raises(re.error):
        extractor.extract_metadata_from_frame_text("nonempty")
