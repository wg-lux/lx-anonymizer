from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from fractions import Fraction
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from lx_dtypes.models.meta.VideoMeta import FrameProcessResult, VideoMeta

from lx_anonymizer.frame_cleaner import (
    FrameCleaner,
    InvalidVideoInputError,
    VideoAnonymizationError,
    VideoOutputCollisionError,
)
from lx_anonymizer.ner.frame_metadata_extractor import FrameMetadataExtractor
from lx_anonymizer.ner.spacy_extractor import PatientDataExtractor
from lx_anonymizer.video_processing.video_encoder import VideoEncoder
from lx_anonymizer.video_processing.video_utils import (
    detect_video_format as real_detect_video_format,
)


def _run_media_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        stdin=subprocess.DEVNULL,
    )


def _create_source_video(path: Path) -> None:
    _run_media_command(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=white:size=160x120:rate=10:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:sample_rate=44100:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ]
    )


def _probe_streams(path: Path) -> list[dict[str, object]]:
    result = _run_media_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = cast(dict[str, object], json.loads(result.stdout))
    return cast(list[dict[str, object]], payload["streams"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _initialize_deterministic_core(cleaner: FrameCleaner) -> None:
    """Keep the media E2E lane independent from downloaded OCR/NER models."""
    cleaner.frame_ocr = MagicMock()
    cleaner.frame_metadata_extractor = FrameMetadataExtractor()
    cleaner.patient_data_extractor = cast(PatientDataExtractor, MagicMock())
    cleaner.roi_processor = MagicMock()
    cleaner.use_enhanced_ocr = True


@pytest.mark.video
@pytest.mark.ffmpeg
@pytest.mark.integration
def test_clean_video_masks_real_media_and_preserves_the_attempt_contract(
    tmp_path: Path,
    mock_central_video_format: MagicMock,
) -> None:
    # Arrange: create a decodable source with visible pixels and an audio stream.
    source = tmp_path / "immutable-source.mp4"
    candidate = tmp_path / "attempt-42" / "candidate.mp4"
    _create_source_video(source)
    source_digest = _sha256(source)
    mock_central_video_format.side_effect = real_detect_video_format
    clean_frame = FrameProcessResult(
        is_sensitive=False,
        metadata={},
        ocr_text="",
        ocr_confidence=0.0,
    )

    with (
        patch.object(VideoEncoder, "_detect_nvenc_support", return_value=False),
        patch.object(
            FrameCleaner,
            "_init_core_components",
            _initialize_deterministic_core,
        ),
        patch.object(FrameCleaner, "_log_hf_cache_status", return_value=None),
        patch.object(FrameCleaner, "_process_frame_result", return_value=clean_frame),
    ):
        cleaner = FrameCleaner(use_llm=False)

        # Act: cross the public FrameCleaner boundary and execute real decode/FFmpeg.
        result_path, raw_metadata = cleaner.clean_video(
            video_path=source,
            endoscope_image_roi={
                "x": 40,
                "y": 30,
                "width": 80,
                "height": 60,
                "image_width": 160,
                "image_height": 120,
            },
            endoscope_data_roi_nested=None,
            source_frame_rate=Fraction(10, 1),
            output_path=candidate,
            technique="mask_overlay",
        )

    # Assert: ownership, media integrity, anonymization, and typed metadata hold.
    assert result_path == candidate
    assert candidate.is_file() and candidate.stat().st_size > 0
    assert _sha256(source) == source_digest

    streams = _probe_streams(candidate)
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    assert len(video_streams) == 1
    assert not [s for s in streams if s.get("codec_type") == "audio"]
    assert video_streams[0]["width"] == 160
    assert video_streams[0]["height"] == 120
    assert video_streams[0]["pix_fmt"] == "yuv420p"
    assert video_streams[0]["avg_frame_rate"] == "10/1"
    assert video_streams[0]["nb_frames"] == "10"

    capture = cv2.VideoCapture(str(candidate))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    assert ok is True
    assert isinstance(frame, np.ndarray)
    assert float(frame[5:20, 5:20].mean()) < 8.0
    assert float(frame[40:80, 55:105].mean()) > 240.0

    metadata = VideoMeta.model_validate(raw_metadata)
    assert metadata.anonymizer_provenance is not None
    metrics = cast(dict[str, object], raw_metadata["paper_evaluation_metrics"])
    runtime = cast(dict[str, object], metrics["runtime"])
    assert runtime["technique"] == "mask_overlay"
    assert cast(int, runtime["frames_processed"]) > 0


@pytest.mark.parametrize("source_kind", ["missing", "empty", "directory", "symlink"])
def test_clean_video_rejects_invalid_source_at_the_public_boundary(
    tmp_path: Path,
    source_kind: str,
) -> None:
    # Arrange: construct each source shape prohibited by the immutable-input contract.
    source = tmp_path / "source.mp4"
    if source_kind == "empty":
        source.touch()
    elif source_kind == "directory":
        source.mkdir()
    elif source_kind == "symlink":
        target = tmp_path / "real-source.mp4"
        target.write_bytes(b"video")
        source.symlink_to(target)
    cleaner = FrameCleaner.__new__(FrameCleaner)
    cleaner._run_lock = threading.Lock()  # pyright: ignore[reportPrivateUsage]

    # Act and assert: rejection happens before any decoder or encoder is entered.
    with pytest.raises(InvalidVideoInputError):
        cleaner.clean_video(
            video_path=source,
            endoscope_image_roi=None,
            endoscope_data_roi_nested=None,
            source_frame_rate=Fraction(25, 1),
            output_path=tmp_path / "candidate.mp4",
        )


def test_clean_video_rejects_a_preexisting_candidate_without_mutating_either_file(
    tmp_path: Path,
) -> None:
    # Arrange: place caller and foreign-attempt bytes at the two ownership paths.
    source = tmp_path / "source.mp4"
    candidate = tmp_path / "candidate.mp4"
    source.write_bytes(b"immutable source")
    candidate.write_bytes(b"other attempt")
    cleaner = FrameCleaner.__new__(FrameCleaner)
    cleaner._run_lock = threading.Lock()  # pyright: ignore[reportPrivateUsage]
    owned_run = MagicMock()
    cleaner._clean_video_owned = owned_run  # pyright: ignore[reportPrivateUsage]

    # Act: attempt to reuse a path that this invocation does not own.
    with pytest.raises(VideoOutputCollisionError):
        cleaner.clean_video(
            video_path=source,
            endoscope_image_roi=None,
            endoscope_data_roi_nested=None,
            source_frame_rate=Fraction(25, 1),
            output_path=candidate,
        )

    # Assert: expensive work never starts and neither ownership domain is changed.
    owned_run.assert_not_called()
    assert source.read_bytes() == b"immutable source"
    assert candidate.read_bytes() == b"other attempt"


@pytest.mark.parametrize("candidate_kind", ["source", "suffixless"])
def test_clean_video_rejects_candidate_paths_that_cannot_be_attempt_owned(
    tmp_path: Path,
    candidate_kind: str,
) -> None:
    # Arrange: choose an output that aliases the source or has no media container.
    source = tmp_path / "source.mp4"
    source.write_bytes(b"immutable source")
    candidate = source if candidate_kind == "source" else tmp_path / "candidate"
    cleaner = FrameCleaner.__new__(FrameCleaner)
    cleaner._run_lock = threading.Lock()  # pyright: ignore[reportPrivateUsage]
    owned_run = MagicMock()
    cleaner._clean_video_owned = owned_run  # pyright: ignore[reportPrivateUsage]

    # Act: invoke the public boundary with an output that FFmpeg must not own.
    with pytest.raises(VideoOutputCollisionError):
        cleaner.clean_video(
            video_path=source,
            endoscope_image_roi=None,
            endoscope_data_roi_nested=None,
            source_frame_rate=Fraction(25, 1),
            output_path=candidate,
        )

    # Assert: validation is side-effect free and no processing begins.
    owned_run.assert_not_called()
    assert source.read_bytes() == b"immutable source"


def test_clean_video_rejects_non_positive_source_rate_before_processing(
    tmp_path: Path,
) -> None:
    # Arrange: provide valid owned paths but an invalid temporal contract.
    source = tmp_path / "source.mp4"
    source.write_bytes(b"immutable source")
    cleaner = FrameCleaner.__new__(FrameCleaner)
    cleaner._run_lock = threading.Lock()  # pyright: ignore[reportPrivateUsage]
    owned_run = MagicMock()
    cleaner._clean_video_owned = owned_run  # pyright: ignore[reportPrivateUsage]

    # Act: invoke the public boundary with a non-positive rational frame rate.
    with pytest.raises(ValueError, match="source_frame_rate must be a positive"):
        cleaner.clean_video(
            video_path=source,
            endoscope_image_roi=None,
            endoscope_data_roi_nested=None,
            source_frame_rate=Fraction(0, 1),
            output_path=tmp_path / "candidate.mp4",
        )

    # Assert: the invalid timeline never reaches video processing.
    owned_run.assert_not_called()


def test_clean_video_removes_only_its_partial_candidate_when_processing_fails(
    tmp_path: Path,
) -> None:
    # Arrange: simulate a lower video boundary writing an incomplete owned artifact.
    source = tmp_path / "source.mp4"
    candidate = tmp_path / "candidate.mp4"
    source.write_bytes(b"immutable source")
    cleaner = FrameCleaner.__new__(FrameCleaner)
    cleaner._run_lock = threading.Lock()  # pyright: ignore[reportPrivateUsage]

    def fail_after_partial_write(**_kwargs: object) -> tuple[Path, dict[str, object]]:
        candidate.write_bytes(b"partial")
        raise VideoAnonymizationError("injected encoder failure")

    cleaner._clean_video_owned = MagicMock(  # pyright: ignore[reportPrivateUsage]
        side_effect=fail_after_partial_write
    )

    # Act: inject a failure after candidate creation.
    with pytest.raises(VideoAnonymizationError, match="injected encoder failure"):
        cleaner.clean_video(
            video_path=source,
            endoscope_image_roi=None,
            endoscope_data_roi_nested=None,
            source_frame_rate=Fraction(25, 1),
            output_path=candidate,
        )

    # Assert: the owned partial is gone while the immutable source remains intact.
    assert not candidate.exists()
    assert source.read_bytes() == b"immutable source"


def test_frame_removal_failure_is_not_reported_as_a_candidate(tmp_path: Path) -> None:
    # Arrange: make the frame-removal subprocess boundary report failure.
    cleaner = FrameCleaner.__new__(FrameCleaner)
    cleaner.remove_frames_from_video_streaming = MagicMock(return_value=False)

    # Act and assert: an integrity failure raises instead of returning an output path.
    with pytest.raises(VideoAnonymizationError, match="Frame removal failed"):
        cleaner._apply_frame_removal(  # pyright: ignore[reportPrivateUsage]
            video_path=tmp_path / "source.mp4",
            output_video=tmp_path / "candidate.mp4",
            sensitive_idx=[3],
            total_frames=10,
        )
