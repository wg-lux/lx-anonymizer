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
