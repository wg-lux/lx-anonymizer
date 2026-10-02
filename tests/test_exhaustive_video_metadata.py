# pyright: reportPrivateUsage=false
"""Exhaustive coverage with real decoding and deterministic OCR boundaries."""

import threading
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Protocol, cast
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from lx_dtypes.models.meta.VideoMeta import FrameProcessResult

from lx_anonymizer.frame_cleaner import (
    FrameCleaner,
    FrameCleanerSamplingProfile,
    VideoAnonymizationError,
)
from lx_anonymizer.ner.frame_metadata_extractor import FrameMetadataExtractor
from lx_anonymizer.runtime_types import ImageArray
from lx_anonymizer.video_processing.analysis_budget import FrameAnalysisBudgetExceeded


class _VideoWriter(Protocol):
    def isOpened(self) -> bool: ...
    def write(self, image: ImageArray) -> None: ...
    def release(self) -> None: ...


def _cleaner() -> FrameCleaner:
    cleaner = FrameCleaner.__new__(FrameCleaner)
    cleaner.sampling_profile = FrameCleanerSamplingProfile.from_quality_profile(
        "exhaustive"
    )
    cleaner.use_llm = False
    cleaner.frame_ocr = MagicMock()
    cleaner.frame_metadata_extractor = FrameMetadataExtractor()
    cleaner.region_detector = None
    cleaner._run_lock = threading.Lock()
    cleaner._reset_run_state()
    return cleaner


def _source(tmp_path: Path, overlay_index: int, count: int = 61) -> Path:
    path = tmp_path / "source.avi"
    writer = cast(
        _VideoWriter,
        cv2.VideoWriter(str(path), int.from_bytes(b"FFV1", "little"), 25.0, (64, 48)),
    )
    assert writer.isOpened(), "Lossless test encoder must be available"
    try:
        for index in range(count):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            if index == overlay_index:
                frame[2:10, 2:30] = 255
            writer.write(frame)
    finally:
        writer.release()
    return path


@pytest.mark.parametrize("overlay_index", [0, 1, 30, 60])
def test_exhaustive_analysis_collects_single_frame_overlay(
    tmp_path: Path, overlay_index: int
) -> None:
    # Arrange: a lossless source with exactly one patient overlay, including
    # frames the old stride and sample cap would never analyze.
    source = _source(tmp_path, overlay_index)
    cleaner = _cleaner()
    shape_seen: list[tuple[int, ...]] = []

    def ocr(frame: ImageArray) -> tuple[str, float, dict[str, object]]:
        shape_seen.append(frame.shape)
        return ("Anna Muster" if frame[2, 2, 0] else "", 0.95, {})

    with (
        patch.object(
            cleaner.frame_ocr, "extract_text_with_rapidocr", side_effect=ocr
        ) as recognize,
        patch.object(
            cleaner,
            "_unified_metadata_extract",
            return_value={"first_name": "Anna", "last_name": "Muster"},
        ) as extract,
        patch.object(
            cleaner, "_detect_phi_regions_for_frame", return_value=[]
        ) as detect,
    ):
        # Act: retain real decode, frame analysis, collection, and public output.
        result_path, metadata = cleaner.clean_video(
            source,
            {"x": 40, "y": 30, "width": 10, "height": 10},
            None,
            Fraction(25, 1),
            technique="extract_only",
        )

    # Assert: full-frame analysis finds the overlay despite an excluding ROI.
    assert result_path == source
    assert metadata["first_name"] == "Anna"
    assert metadata["last_name"] == "Muster"
    assert len(cleaner.frame_observations) == 61
    assert [item.frame_number for item in cleaner.frame_observations] == list(range(61))
    assert cleaner.frame_observations[overlay_index].ocr_text == "Anna Muster"
    assert cleaner.frame_observations[overlay_index].ocr_roi is None
    assert recognize.call_count == (2 if overlay_index in (0, 60) else 3)
    assert detect.call_count == 13
    assert cleaner._phi_frames_processed == 13
    assert all(shape == (48, 64, 3) for shape in shape_seen)
    extract.assert_called_once_with("Anna Muster")
    assert cleaner._previous_ocr_frame is None
    assert cleaner._previous_ocr_result is None


def test_changed_pixel_never_reuses_ocr_and_metadata_cache_is_bounded() -> None:
    # Arrange: unique pixels but repeated recognized text.
    cleaner = _cleaner()
    cleaner.sampling_profile = replace(cleaner.sampling_profile, max_retained_texts=2)
    with (
        patch.object(
            cleaner.frame_ocr,
            "extract_text_with_rapidocr",
            return_value=("Anna", 0.9, {}),
        ) as recognize,
        patch.object(
            cleaner, "_unified_metadata_extract", return_value={"first_name": "Anna"}
        ) as extract,
        patch.object(cleaner, "_detect_phi_regions_for_frame", return_value=[]),
    ):
        # Act: change one source pixel each time, without similarity thresholds.
        for value in range(4):
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            frame[0, 0, 0] = value
            cleaner._process_frame_result(frame, None, None)
        for text in ("different", "another"):
            cleaner._metadata_for_ocr_text(text, {})

    # Assert: OCR sees every changed image; extraction reuses only exact text.
    assert recognize.call_count == 4
    assert extract.call_count == 3
    assert len(cleaner._metadata_text_cache) == 2
    assert cleaner.sensitive_meta.first_name == "Anna"
    cleaner._reset_run_state()
    assert not cleaner._metadata_text_cache
    assert cleaner._previous_ocr_frame is None


@pytest.mark.parametrize("expected_count", [0, 2, 4])
def test_exhaustive_decoder_rejects_unknown_or_incomplete_frame_counts(
    expected_count: int,
) -> None:
    # Arrange: three decoded frames and incompatible declared source counts.
    cleaner = _cleaner()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.get.return_value = 25.0
    capture.read.side_effect = [(True, frame)] * 3 + [(False, frame)]
    # Act / Assert: partial decoding cannot become successful analysis.
    with (
        patch("cv2.VideoCapture", return_value=capture),
        pytest.raises((ValueError, RuntimeError), match="frame count|Incomplete"),
    ):
        list(cleaner._iter_video(Path("source.avi"), expected_count))
    capture.release.assert_called_once()


def test_ocr_failure_closes_decoder_and_does_not_return_metadata(
    tmp_path: Path,
) -> None:
    # Arrange.
    cleaner = _cleaner()
    source = tmp_path / "source.avi"
    source.write_bytes(b"source")
    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.get.return_value = 25.0
    capture.read.return_value = (True, np.zeros((8, 8, 3), dtype=np.uint8))
    with (
        patch("cv2.VideoCapture", return_value=capture),
        patch.object(cleaner, "_get_total_frames", return_value=3),
        patch.object(
            cleaner.frame_ocr,
            "extract_text_with_rapidocr",
            side_effect=RuntimeError("OCR failed"),
        ),
        pytest.raises(RuntimeError, match="OCR failed"),
    ):
        # Act / Assert.
        cleaner.clean_video(
            source, None, None, Fraction(25, 1), technique="extract_only"
        )
    capture.release.assert_called_once()
    assert cleaner._run_lock.acquire(blocking=False)
    cleaner._run_lock.release()


def test_analysis_cap_returns_metadata_and_releases_invocation(
    tmp_path: Path,
) -> None:
    # Arrange: a real decoder and a deadline crossed by the first OCR call.
    source = _source(tmp_path, 0, count=3)
    original_bytes = source.read_bytes()
    cleaner = _cleaner()
    cleaner.sampling_profile = replace(cleaner.sampling_profile, max_analysis_seconds=1)
    with (
        patch("lx_anonymizer.frame_cleaner.time.monotonic", side_effect=[0, 0, 2, 2]),
        patch.object(
            cleaner,
            "_process_frame_result",
            return_value=FrameProcessResult(
                is_sensitive=True,
                metadata={"first_name": "Anna"},
                ocr_text="Anna",
                ocr_confidence=0.95,
            ),
        ) as process,
    ):
        # Act: a cap preserves discovered metadata and declares incomplete coverage.
        result_path, metadata = cleaner.clean_video(
            source, None, None, Fraction(25, 1), technique="extract_only"
        )
    # Assert: only one frame was attempted; the immutable source is preserved.
    process.assert_called_once()
    assert result_path == source
    assert metadata["first_name"] == "Anna"
    coverage = cleaner._metadata_analysis
    assert coverage is not None
    assert coverage.complete is False
    assert coverage.inspected_frame_numbers == [0]
    assert coverage.cap_reason == "deadline_exceeded"
    assert metadata["metadata_analysis"] == coverage.model_dump(mode="json")
    assert source.read_bytes() == original_bytes
    assert cleaner._previous_ocr_frame is None
    assert cleaner._previous_ocr_result is None
    assert cleaner._run_lock.acquire(blocking=False)
    cleaner._run_lock.release()


def test_capped_analysis_cannot_drive_sensitive_frame_removal(tmp_path: Path) -> None:
    # Arrange / Act / Assert: this technique needs complete sensitive-frame discovery.
    source = _source(tmp_path, 0, count=3)
    cleaner = _cleaner()
    cleaner.sampling_profile = replace(cleaner.sampling_profile, max_analysis_seconds=1)
    with (
        patch("lx_anonymizer.frame_cleaner.time.monotonic", side_effect=[0, 2]),
        pytest.raises(FrameAnalysisBudgetExceeded),
    ):
        cleaner.clean_video(
            source,
            None,
            None,
            Fraction(25, 1),
            output_path=tmp_path / "candidate.mp4",
            technique="remove_frames",
        )
    assert not (tmp_path / "candidate.mp4").exists()


def test_projected_cap_probes_timeline_and_finds_name_at_last_frame(
    tmp_path: Path,
) -> None:
    # Arrange: a name beyond the first 32 frames, at the far end of a real source.
    source = _source(tmp_path, 259, count=260)
    cleaner = _cleaner()

    def guard(
        *,
        total_frames: int,
        completed_frames: int,
        elapsed_seconds: float,
        maximum_seconds: float,
        project_completion: bool,
    ) -> None:
        if project_completion and total_frames == 260 and completed_frames == 32:
            raise FrameAnalysisBudgetExceeded("projected_deadline_exceeded")

    def process(
        *,
        gray_frame: ImageArray,
        endoscope_image_roi: object,
        endoscope_data_roi_nested: object,
        frame_id: int,
        collect_for_batch: bool,
        high_quality_ocr: bool,
    ) -> FrameProcessResult:
        has_name = frame_id == 259
        return FrameProcessResult(
            is_sensitive=has_name,
            metadata={"first_name": "Anna"} if has_name else {},
            ocr_text="Anna" if has_name else "",
            ocr_confidence=0.95,
        )

    with (
        patch("lx_anonymizer.frame_cleaner.check_analysis_budget", side_effect=guard),
        patch.object(cleaner, "_process_frame_result", side_effect=process),
    ):
        # Act: infeasible exhaustive OCR becomes bounded timeline probing.
        _, metadata = cleaner.clean_video(
            source, None, None, Fraction(25, 1), technique="extract_only"
        )
    # Assert: useful metadata survives; no claim is made about uninspected frames.
    assert metadata["first_name"] == "Anna"
    coverage = cleaner._metadata_analysis
    assert coverage is not None
    assert coverage.mode == "interval_sampled"
    assert coverage.complete is False
    assert {0, 129, 259} <= set(coverage.inspected_frame_numbers)
    assert len(coverage.inspected_frame_numbers) <= 32 + 128


def test_observation_capacity_fails_loudly(tmp_path: Path) -> None:
    # Arrange.
    source = _source(tmp_path, 1, count=3)
    cleaner = _cleaner()
    cleaner.sampling_profile = replace(
        cleaner.sampling_profile, max_retained_observations=2
    )
    with (
        patch.object(
            cleaner.frame_ocr, "extract_text_with_rapidocr", return_value=("", 0.0, {})
        ),
        patch.object(cleaner, "_detect_phi_regions_for_frame", return_value=[]),
        pytest.raises(VideoAnonymizationError, match="capacity exceeded"),
    ):
        # Act / Assert: never silently truncate evidence and claim completion.
        cleaner.clean_video(
            source, None, None, Fraction(25, 1), technique="extract_only"
        )
    assert len(cleaner.frame_observations) == 2


@pytest.mark.parametrize(
    "updates",
    [
        {"max_retained_texts": 0},
        {"smart_early_stopping": True},
        {"high_quality_ocr": True},
    ],
)
def test_exhaustive_profile_rejects_invalid_limits_and_early_stopping(
    updates: dict[str, object],
) -> None:
    # Arrange.
    profile = FrameCleanerSamplingProfile.from_quality_profile("exhaustive")
    # Act / Assert.
    with pytest.raises(ValueError):
        replace(profile, **updates)


def test_bounded_llm_candidates_do_not_discard_collected_metadata() -> None:
    # Arrange: the only birth date appears in a low-confidence late candidate.
    cleaner = _cleaner()
    cleaner.sampling_profile = replace(cleaner.sampling_profile, max_retained_texts=2)
    predictions = [
        {"first_name": "Anna"},
        {"last_name": "Muster"},
        {"dob": "1980-02-03"},
    ]
    with patch.object(cleaner, "_unified_metadata_extract", side_effect=predictions):
        # Act: metadata is accumulated before ranking optional LLM candidates.
        for index, (text, confidence) in enumerate(
            zip(("name", "surname", "birth date"), (0.9, 0.8, 0.1), strict=True)
        ):
            metadata = cleaner._metadata_for_ocr_text(text, {})
            cleaner._collect_frame_for_batch(
                frame_id=index,
                ocr_text=text,
                ocr_conf=confidence,
                frame_metadata=metadata,
                is_sensitive=True,
                phi_regions=[],
            )

    # Assert: bounded candidate retention does not limit metadata coverage.
    assert cleaner.ocr_text_collection == ["name", "surname"]
    assert len(cleaner.frame_collection) == 2
    assert cleaner.sensitive_meta.first_name == "Anna"
    assert cleaner.sensitive_meta.last_name == "Muster"
    assert cleaner.sensitive_meta.dob is not None
    assert cleaner.sensitive_meta.dob.isoformat() == "1980-02-03"
