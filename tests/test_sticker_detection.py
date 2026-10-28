import threading
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from lx_dtypes.models.meta.VideoMeta import FrameProcessResult

from lx_anonymizer.frame_cleaner import FrameCleaner, FrameCleanerSamplingProfile
from lx_anonymizer.frame_cleaner.cleaner import VideoAnonymizationError
from lx_anonymizer.frame_cleaner.sticker_detection import (
    StickerScore,
    score_sticker,
    white_mask,
)
from lx_anonymizer.ner.frame_metadata_extractor import FrameMetadataExtractor
from lx_anonymizer.runtime_types import ImageArray
from lx_anonymizer.sensitive_meta_interface import SensitiveMeta
from lx_anonymizer.utils.roi_normalization import NormalizedRoi

ROI: NormalizedRoi = {"x": 20, "y": 20, "width": 100, "height": 100}
MUCOSA_BGR = (40, 50, 190)


def _print_rows(size: int) -> range:
    return range(30 + 6, 30 + size - 6, 8)


def _print_pixels(size: int) -> int:
    """Dark text lines: 2 rows high, inset 6 px so the paper encloses them."""
    return len(_print_rows(size)) * 2 * (size - 12)


def _frame(
    sticker_size: int = 0,
    highlights: int = 0,
    sticker_bgr: tuple[int, int, int] = (245, 245, 245),
    printed: bool = True,
) -> ImageArray:
    """Red mucosa with an optional sticker and 2x2 specular highlights.

    ``printed=False`` gives a smooth white blob, like overexposure glare.
    """
    frame: ImageArray = np.zeros((140, 140, 3), dtype=np.uint8)
    frame[:] = MUCOSA_BGR
    for i in range(highlights):
        y, x = 25 + (i // 10) * 9, 25 + (i % 10) * 9
        frame[y : y + 2, x : x + 2] = 255
    if sticker_size:
        frame[30 : 30 + sticker_size, 30 : 30 + sticker_size] = sticker_bgr
        if printed:
            for y in _print_rows(sticker_size):
                frame[y : y + 2, 36 : 30 + sticker_size - 6] = (30, 30, 30)
    return frame


@pytest.mark.parametrize(
    ("frame", "expected_paper", "expected_print", "detected"),
    [
        (_frame(), 0, 0, False),
        (_frame(sticker_size=40), 1600 - _print_pixels(40), _print_pixels(40), True),
        (_frame(sticker_size=60), 3600 - _print_pixels(60), _print_pixels(60), True),
        # Below the minimum ROI fraction: too small to be a sticker.
        (_frame(sticker_size=10), 100 - _print_pixels(10), _print_pixels(10), False),
        # Specular highlights are white but removed as speckle.
        (_frame(highlights=60), 0, 0, False),
        (
            _frame(sticker_size=40, highlights=60),
            1600 - _print_pixels(40),
            _print_pixels(40),
            True,
        ),
        # Saturated bright colour (e.g. green light) is not paper white.
        (_frame(sticker_size=40, sticker_bgr=(60, 240, 60)), 0, 0, False),
        # Large smooth white glare without enclosed print is not a sticker.
        (_frame(sticker_size=60, printed=False), 3600, 0, False),
    ],
)
def test_score_sticker(
    frame: ImageArray, expected_paper: int, expected_print: int, detected: bool
) -> None:
    score = score_sticker(frame, ROI)
    assert score.largest_region_pixels == expected_paper
    assert score.print_pixels == expected_print
    assert score.roi_pixels == 100 * 100
    assert score.detected is detected
    assert score.ranking_pixels == (expected_paper if detected else 0)


def test_score_sticker_ignores_white_outside_roi() -> None:
    frame = _frame()
    frame[125:140, 125:140] = 255
    assert score_sticker(frame, ROI).largest_region_pixels == 0


def test_score_sticker_clips_roi_to_frame() -> None:
    roi: NormalizedRoi = {"x": 100, "y": 100, "width": 500, "height": 500}
    assert score_sticker(_frame(), roi).roi_pixels == 40 * 40


def test_grayscale_frames_use_brightness_only() -> None:
    gray: ImageArray = np.zeros((140, 140), dtype=np.uint8)
    gray[30:70, 30:70] = 200
    gray[40:42, 36:64] = 20
    score = score_sticker(gray, ROI)
    assert (score.largest_region_pixels, score.print_pixels) == (1600 - 56, 56)
    assert score.detected


def test_dark_pixels_outside_the_paper_are_not_print() -> None:
    frame = _frame(sticker_size=60, printed=False)
    frame[100:110, 30:90] = (20, 20, 20)
    assert score_sticker(frame, ROI).print_pixels == 0


@pytest.mark.parametrize(
    ("frame", "roi"),
    [
        (np.zeros((140, 140, 3), dtype=np.float32), ROI),
        (np.zeros((140, 140, 4), dtype=np.uint8), ROI),
        (np.zeros((140,), dtype=np.uint8), ROI),
        (_frame(), {"x": 0, "y": 0, "width": 0, "height": 4}),
        (_frame(), {"x": -1, "y": 0, "width": 4, "height": 4}),
        (_frame(), {"x": 500, "y": 500, "width": 4, "height": 4}),
    ],
)
def test_score_sticker_rejects_invalid_input(
    frame: ImageArray, roi: NormalizedRoi
) -> None:
    with pytest.raises(ValueError):
        score_sticker(frame, roi)


@pytest.mark.parametrize(
    ("largest", "print_pixels", "roi_pixels"),
    [(1, 0, 0), (-1, 0, 10), (0, -1, 10), (11, 0, 10), (6, 5, 10)],
)
def test_sticker_score_rejects_impossible_values(
    largest: int, print_pixels: int, roi_pixels: int
) -> None:
    with pytest.raises(ValueError):
        StickerScore(
            largest_region_pixels=largest,
            print_pixels=print_pixels,
            roi_pixels=roi_pixels,
        )


def test_white_mask_requires_low_saturation() -> None:
    pixels: ImageArray = np.array([[[250, 250, 250], [40, 50, 250]]], dtype=np.uint8)
    assert white_mask(pixels).tolist() == [[1, 0]]


def _cleaner_stub() -> FrameCleaner:
    cleaner = FrameCleaner.__new__(FrameCleaner)
    cleaner.frame_ocr = MagicMock()
    cleaner.frame_metadata_extractor = FrameMetadataExtractor()
    cleaner.sensitive_meta = SensitiveMeta()
    cleaner.sampling_profile = FrameCleanerSamplingProfile(
        max_frames_to_sample=3,
        high_quality_ocr=False,
        smart_early_stopping=False,
        early_stop_techniques=frozenset(),
    )
    cleaner.use_llm = False
    cleaner.enriched_extractor = None
    cleaner.frame_collection = []
    cleaner.frame_observations = []
    cleaner.ocr_text_collection = []
    cleaner.frame_sticker_pixels = []
    cleaner._model_failure_requires_mask = False  # pyright: ignore[reportPrivateUsage]
    cleaner._run_lock = threading.Lock()  # pyright: ignore[reportPrivateUsage]
    return cleaner


def _run_clean_video(
    tmp_path: Path,
    frames: list[ImageArray],
    confidences: tuple[float, ...],
    endoscope_image_roi: dict[str, object] | None,
) -> tuple[str, list[StickerScore | None], list[int]]:
    cleaner = _cleaner_stub()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"immutable source")
    results = [
        FrameProcessResult(
            is_sensitive=False,
            metadata={},
            ocr_text=f"Frame {i}",
            ocr_confidence=conf,
        )
        for i, conf in enumerate(confidences)
    ]
    with (
        patch.object(cleaner, "_get_total_frames", return_value=len(frames)),
        patch.object(
            cleaner,
            "_iter_video",
            return_value=[(i, frame, 1) for i, frame in enumerate(frames)],
        ),
        patch.object(cleaner, "_process_frame_result", side_effect=results) as process,
    ):
        _, metadata = cleaner.clean_video(
            video_path=source,
            endoscope_image_roi=endoscope_image_roi,
            endoscope_data_roi_nested=None,
            source_frame_rate=Fraction(25, 1),
            technique="extract_only",
        )
    calls = process.call_args_list
    return (
        str(metadata["text"]),
        [call.kwargs["sticker"] for call in calls],
        [call.kwargs["gray_frame"].ndim for call in calls],
    )


@pytest.mark.parametrize(
    ("sticker_sizes", "confidences", "expected_text"),
    [
        # The frame showing the largest sticker wins over higher OCR confidence.
        ((0, 60, 40), (0.9, 0.2, 0.8), "Frame 1"),
        ((50, 0, 30), (0.1, 0.9, 0.9), "Frame 0"),
        # Without any sticker, OCR confidence decides.
        ((0, 0, 0), (0.3, 0.9, 0.5), "Frame 1"),
    ],
)
def test_clean_video_prefers_frames_showing_a_sticker(
    tmp_path: Path,
    sticker_sizes: tuple[int, int, int],
    confidences: tuple[float, float, float],
    expected_text: str,
) -> None:
    frames = [_frame(sticker_size=size) for size in sticker_sizes]
    text, stickers, analysis_ndims = _run_clean_video(
        tmp_path, frames, confidences, {**ROI, "endoscope_image_x": None}
    )
    assert [s.ranking_pixels if s else None for s in stickers] == [
        size * size - _print_pixels(size) if size else 0 for size in sticker_sizes
    ]
    # Sampled OCR still receives equalized grayscale.
    assert analysis_ndims == [2, 2, 2]
    assert text == expected_text


@pytest.mark.parametrize(
    "endoscope_image_roi",
    [None, {}, {"endoscope_image_x": None, "endoscope_image_y": None}],
)
def test_clean_video_without_usable_roi_skips_sticker_detection(
    tmp_path: Path, endoscope_image_roi: dict[str, object] | None
) -> None:
    frames = [_frame(sticker_size=size) for size in (0, 60, 40)]
    text, stickers, _ = _run_clean_video(
        tmp_path, frames, (0.9, 0.2, 0.8), endoscope_image_roi
    )
    assert stickers == [None, None, None]
    assert text == "Frame 0"


DATA_ROI: dict[str, dict[str, int | None]] = {
    "patient_first_name": {"x": 0, "y": 0, "width": 10, "height": 10}
}
DETECTED = StickerScore(largest_region_pixels=5000, print_pixels=400, roi_pixels=10000)
NOT_DETECTED = StickerScore(largest_region_pixels=10, print_pixels=1, roi_pixels=10000)
GLARE = StickerScore(largest_region_pixels=5000, print_pixels=0, roi_pixels=10000)


def _process(
    cleaner: FrameCleaner,
    sticker: StickerScore | None,
    data_roi: dict[str, dict[str, int | None]] | None,
) -> FrameProcessResult:
    gray: ImageArray = np.zeros((140, 140), dtype=np.uint8)
    with (
        patch.object(cleaner, "_unified_metadata_extract", return_value={}),
        patch.object(cleaner, "_detect_phi_regions_for_frame", return_value=[]),
    ):
        return cleaner._process_frame_result(  # pyright: ignore[reportPrivateUsage]
            gray,
            endoscope_image_roi=dict(ROI),
            endoscope_data_roi_nested=data_roi,
            frame_id=3,
            collect_for_batch=True,
            sticker=sticker,
        )


def _ocr_by_roi(texts: dict[str, tuple[str, float]]) -> MagicMock:
    def extract(
        _frame: ImageArray, roi: dict[str, object], high_quality: bool = True
    ) -> tuple[str, float, dict[str, object]]:
        _ = high_quality
        key = "sticker" if "endoscope_image" in roi else "data"
        text, conf = texts[key]
        return text, conf, {}

    return MagicMock(side_effect=extract)


def test_detected_sticker_is_ocred_in_image_area() -> None:
    cleaner = _cleaner_stub()
    ocr = _ocr_by_roi({"data": ("Datum 01.02.2024", 0.6), "sticker": ("Anna", 0.8)})
    cleaner.frame_ocr.extract_text_from_frame = ocr

    result = _process(cleaner, DETECTED, DATA_ROI)

    assert ocr.call_count == 2
    sticker_roi = ocr.call_args_list[1].args[1]
    assert sticker_roi == {"endoscope_image": dict(ROI)}
    assert result.ocr_text == "Datum 01.02.2024\nAnna"
    assert result.ocr_confidence == 0.8
    assert cleaner.frame_sticker_pixels == [5000]


def test_glare_frame_does_not_outrank_printed_sticker(tmp_path: Path) -> None:
    frames = [
        _frame(sticker_size=80, printed=False),
        _frame(sticker_size=40),
    ]
    text, stickers, _ = _run_clean_video(
        tmp_path, frames, (0.9, 0.2), {**ROI, "endoscope_image_x": None}
    )
    assert [s.detected if s else None for s in stickers] == [False, True]
    assert text == "Frame 1"


def test_sticker_detection_does_not_decide_sensitivity() -> None:
    # Removal is decided in endoreg-db after annotation, not by this heuristic.
    cleaner = _cleaner_stub()
    cleaner.frame_ocr.extract_text_from_frame = _ocr_by_roi(
        {"data": ("", 0.0), "sticker": ("", 0.0)}
    )
    assert _process(cleaner, DETECTED, DATA_ROI).is_sensitive is False


@pytest.mark.parametrize(
    ("sticker", "data_roi"),
    [
        (None, DATA_ROI),
        (NOT_DETECTED, DATA_ROI),
        (GLARE, DATA_ROI),
        # Without a data ROI the main OCR already reads the endoscope image.
        (DETECTED, None),
    ],
)
def test_sticker_ocr_runs_only_when_image_area_was_not_read(
    sticker: StickerScore | None,
    data_roi: dict[str, dict[str, int | None]] | None,
) -> None:
    cleaner = _cleaner_stub()
    ocr = _ocr_by_roi({"data": ("text", 0.5), "sticker": ("text", 0.5)})
    cleaner.frame_ocr.extract_text_from_frame = ocr

    _process(cleaner, sticker, data_roi)

    assert ocr.call_count == 1


def _collect(cleaner: FrameCleaner, text: str, conf: float, pixels: int) -> None:
    cleaner._collect_frame_for_batch(  # pyright: ignore[reportPrivateUsage]
        frame_id=None,
        ocr_text=text,
        ocr_conf=conf,
        frame_metadata={},
        is_sensitive=False,
        phi_regions=[],
        sticker_pixels=pixels,
    )


def test_llm_candidates_prefer_sticker_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lx_anonymizer.frame_cleaner.cleaner.settings.LLM_MIN_TEXT_LENGTH", 1
    )
    cleaner = _cleaner_stub()
    _collect(cleaner, "confident overlay", 0.95, 0)
    _collect(cleaner, "large sticker", 0.4, 5000)
    _collect(cleaner, "small sticker", 0.6, 800)

    candidates = cleaner._select_llm_video_text_candidates()  # pyright: ignore[reportPrivateUsage]

    assert candidates == ["large sticker", "small sticker"]


def test_exhaustive_retention_evicts_frames_without_stickers_first() -> None:
    cleaner = _cleaner_stub()
    cleaner.sampling_profile = FrameCleanerSamplingProfile(
        max_frames_to_sample=1,
        high_quality_ocr=False,
        smart_early_stopping=False,
        early_stop_techniques=frozenset(),
        every_frame=True,
        max_retained_texts=2,
    )
    _collect(cleaner, "a", 0.9, 0)
    _collect(cleaner, "b", 0.1, 3000)
    _collect(cleaner, "c", 0.5, 2000)
    # Weaker than every retained frame: rejected.
    _collect(cleaner, "d", 0.99, 500)

    assert cleaner.ocr_text_collection == ["b", "c"]
    assert cleaner.frame_sticker_pixels == [3000, 2000]


def test_misaligned_sticker_scores_raise() -> None:
    cleaner = _cleaner_stub()
    _collect(cleaner, "text", 0.5, 3)
    cleaner.frame_sticker_pixels.clear()
    with pytest.raises(VideoAnonymizationError):
        cleaner._select_llm_video_text_candidates()  # pyright: ignore[reportPrivateUsage]


def test_collect_rejects_negative_sticker_pixels() -> None:
    with pytest.raises(ValueError):
        _collect(_cleaner_stub(), "text", 0.5, -1)
