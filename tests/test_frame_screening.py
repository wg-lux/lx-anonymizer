# pyright: reportPrivateUsage=false
"""Colour screening reuses OCR only for frames positively screened as unchanged."""

import logging
import threading
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Protocol, cast
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from lx_anonymizer.frame_cleaner import FrameCleaner, FrameCleanerSamplingProfile
from lx_anonymizer.frame_cleaner.frame_screening import (
    OcrScreenReason,
    ScreenReference,
    is_inside_body,
    overlay_changed,
    overlay_pixels,
    screen_frame,
)
from lx_anonymizer.ner.frame_metadata_extractor import FrameMetadataExtractor
from lx_anonymizer.runtime_types import ImageArray
from lx_anonymizer.utils.roi_normalization import NormalizedRoi

ROI: NormalizedRoi = {"x": 16, "y": 8, "width": 40, "height": 36}
ROI_INPUT: dict[str, object] = {"x": 16, "y": 8, "width": 40, "height": 36}


class _VideoWriter(Protocol):
    def isOpened(self) -> bool: ...
    def write(self, image: ImageArray) -> None: ...
    def release(self) -> None: ...


def _frame(
    *,
    image_bgr: tuple[int, int, int] = (40, 40, 160),
    overlay_text: bool = False,
    sticker: bool = False,
) -> np.ndarray:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[8:44, 16:56] = image_bgr
    if overlay_text:
        frame[2:6, 2:14] = 255
    if sticker:
        frame[12:28, 20:36] = 255
        frame[18:22, 26:30] = 0
    return frame


def _reference(frame: np.ndarray, frames_since_ocr: int = 0) -> ScreenReference:
    return ScreenReference(
        frame_shape=frame.shape,
        overlay_pixels=overlay_pixels(frame, ROI),
        frames_since_ocr=frames_since_ocr,
    )


@pytest.mark.parametrize(
    ("image_bgr", "inside"),
    [
        ((40, 40, 160), True),  # saturated red mucosa
        ((120, 90, 230), True),  # pink mucosa, hue wraps around 0
        ((200, 200, 200), False),  # room / white paper
        ((60, 140, 60), False),  # green, NBI-like
        ((5, 5, 20), False),  # dark frame
    ],
)
def test_inside_body_classification(
    image_bgr: tuple[int, int, int], inside: bool
) -> None:
    assert is_inside_body(_frame(image_bgr=image_bgr), ROI) is inside


def test_overlay_change_detection_ignores_noise_and_sees_text() -> None:
    base = _frame()
    noisy = base.copy()
    noisy[0:8, :] = 30  # uniform low-level shift below the pixel delta
    reference = overlay_pixels(base, ROI)

    assert not overlay_changed(reference, overlay_pixels(base, ROI))
    assert not overlay_changed(reference, overlay_pixels(noisy, ROI))
    assert overlay_changed(reference, overlay_pixels(_frame(overlay_text=True), ROI))
    assert overlay_changed(reference, reference[:-1])


@pytest.mark.parametrize(
    ("frame", "reference_factory", "expected"),
    [
        (_frame(sticker=True), lambda: _reference(_frame()), OcrScreenReason.STICKER),
        (
            _frame(image_bgr=(200, 200, 200)),
            lambda: _reference(_frame()),
            OcrScreenReason.OUTSIDE_BODY,
        ),
        (_frame(), lambda: None, OcrScreenReason.NO_REFERENCE),
        (_frame(), lambda: _reference(_frame(), 4), OcrScreenReason.ANCHOR_INTERVAL),
        (
            _frame(overlay_text=True),
            lambda: _reference(_frame()),
            OcrScreenReason.OVERLAY_CHANGED,
        ),
        (_frame(), lambda: _reference(_frame(), 3), OcrScreenReason.REUSED),
    ],
)
def test_screen_frame_reasons(
    frame: np.ndarray,
    reference_factory: Callable[[], ScreenReference | None],
    expected: OcrScreenReason,
) -> None:
    decision = screen_frame(
        frame,
        ROI,
        sticker_detected=expected is OcrScreenReason.STICKER,
        reference=reference_factory(),
        max_reuse_frames=5,
    )

    assert decision.reason is expected
    assert decision.requires_ocr is (expected is not OcrScreenReason.REUSED)


def test_screen_frame_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="uint8 BGR"):
        screen_frame(
            np.zeros((48, 64), dtype=np.uint8),
            ROI,
            sticker_detected=False,
            reference=None,
            max_reuse_frames=5,
        )
    with pytest.raises(ValueError, match="outside"):
        is_inside_body(_frame(), {"x": 70, "y": 0, "width": 5, "height": 5})
    with pytest.raises(ValueError, match="positive"):
        screen_frame(
            _frame(), ROI, sticker_detected=False, reference=None, max_reuse_frames=0
        )
    with pytest.raises(ValueError, match="non-negative"):
        ScreenReference(
            frame_shape=(48, 64, 3), overlay_pixels=np.zeros(1), frames_since_ocr=-1
        )


def test_screened_profile_requires_every_frame() -> None:
    profile = FrameCleanerSamplingProfile.from_quality_profile("screened")
    assert profile.every_frame and profile.ocr_screening
    assert not FrameCleanerSamplingProfile.from_quality_profile(
        "exhaustive"
    ).ocr_screening
    with pytest.raises(ValueError, match="every decoded frame"):
        FrameCleanerSamplingProfile(
            max_frames_to_sample=12,
            high_quality_ocr=False,
            smart_early_stopping=False,
            early_stop_techniques=frozenset(),
            ocr_screening=True,
        )


def _cleaner() -> FrameCleaner:
    cleaner = FrameCleaner.__new__(FrameCleaner)
    cleaner.sampling_profile = FrameCleanerSamplingProfile.from_quality_profile(
        "screened"
    )
    cleaner.use_llm = False
    cleaner.frame_ocr = MagicMock()
    cleaner.frame_metadata_extractor = FrameMetadataExtractor()
    cleaner.region_detector = None
    cleaner._run_lock = threading.Lock()
    cleaner._reset_run_state()
    return cleaner


# Frame index -> content; every frame varies the mucosa so exact-pixel reuse
# never applies and the screening decision is exercised.
_OVERLAY_FRAME = 10
_STICKER_FRAME = 15
_OUTSIDE_FRAME = 20
_FRAME_COUNT = 30


def _source(tmp_path: Path) -> Path:
    path = tmp_path / "source.avi"
    writer = cast(
        _VideoWriter,
        cv2.VideoWriter(str(path), int.from_bytes(b"FFV1", "little"), 25.0, (64, 48)),
    )
    assert writer.isOpened(), "Lossless test encoder must be available"
    try:
        for index in range(_FRAME_COUNT):
            image_bgr = (
                (200, 200, 200) if index == _OUTSIDE_FRAME else (40, 40, 150 + index)
            )
            writer.write(
                _frame(
                    image_bgr=image_bgr,
                    overlay_text=index == _OVERLAY_FRAME,
                    sticker=index == _STICKER_FRAME,
                )
            )
    finally:
        writer.release()
    return path


def _ocr(frame: ImageArray) -> tuple[str, float, dict[str, object]]:
    text_bearing = frame[3, 3, 0] > 128 or frame[13, 21, 0] > 128
    return ("Anna Muster" if text_bearing else "", 0.95, {})


def test_screened_analysis_ocrs_informative_frames_and_covers_every_frame(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    cleaner = _cleaner()
    ocr_frames: list[int] = []

    def ocr(frame: ImageArray) -> tuple[str, float, dict[str, object]]:
        ocr_frames.append(len(cleaner.frame_observations))
        return _ocr(frame)

    with (
        patch.object(cleaner.frame_ocr, "extract_text_with_rapidocr", side_effect=ocr),
        patch.object(
            cleaner,
            "_unified_metadata_extract",
            return_value={"first_name": "Anna", "last_name": "Muster"},
        ),
        patch.object(cleaner, "_detect_phi_regions_for_frame", return_value=[]),
    ):
        result_path, metadata = cleaner.clean_video(
            source, ROI_INPUT, None, Fraction(25, 1), technique="extract_only"
        )

    # Overlay appears/disappears, sticker, outside, and the first clean frame
    # after each non-reusable frame receive OCR; nothing else does.
    assert ocr_frames == [0, 10, 11, 15, 16, 20, 21]
    assert result_path == source
    assert metadata["first_name"] == "Anna"
    assert [item.frame_number for item in cleaner.frame_observations] == list(
        range(_FRAME_COUNT)
    )
    assert cleaner.frame_observations[_OVERLAY_FRAME].ocr_text == "Anna Muster"
    assert cleaner.frame_observations[_STICKER_FRAME].ocr_text == "Anna Muster"
    assert cleaner.frame_observations[_OVERLAY_FRAME + 1].ocr_text == ""
    coverage = cleaner._metadata_analysis
    assert coverage is not None
    assert coverage.mode == "colour_screened"
    assert coverage.complete
    screening = cast(dict[str, object], metadata["frame_analysis"])
    assert screening["profile"] == "screened"
    decisions = cast(
        dict[str, int], cast(dict[str, object], screening["ocr_screening"])["decisions"]
    )
    assert decisions["reused"] == _FRAME_COUNT - len(ocr_frames)
    assert decisions["sticker"] == 1
    assert decisions["outside_body"] == 1
    assert cleaner._screen_reference is None


def test_screened_analysis_without_roi_ocrs_every_changed_frame(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = _source(tmp_path)
    cleaner = _cleaner()

    with (
        caplog.at_level(logging.WARNING),
        patch.object(
            cleaner.frame_ocr, "extract_text_with_rapidocr", side_effect=_ocr
        ) as recognize,
        patch.object(cleaner, "_unified_metadata_extract", return_value={}),
        patch.object(cleaner, "_detect_phi_regions_for_frame", return_value=[]),
    ):
        _, metadata = cleaner.clean_video(
            source, None, None, Fraction(25, 1), technique="extract_only"
        )

    assert recognize.call_count == _FRAME_COUNT
    assert '"event": "frame_analysis.ocr_screening_disabled"' in caplog.text
    coverage = cleaner._metadata_analysis
    assert coverage is not None and coverage.mode == "exhaustive"
    assert (
        cast(dict[str, object], metadata["frame_analysis"])["profile"] == "exhaustive"
    )
