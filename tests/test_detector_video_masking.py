import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from lx_anonymizer.anonymization.detector_video_masking import DetectorVideoMasker


def _write_test_video(path: Path, *, frames: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),  # pyright: ignore[reportUnknownMemberType]
        5.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for _ in range(frames):
            writer.write(  # pyright: ignore[reportUnknownMemberType]
                np.full((48, 64, 3), 255, dtype=np.uint8)
            )
    finally:
        writer.release()


def test_detector_video_masker_applies_regions_to_every_frame(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    masked = tmp_path / "masked.mp4"
    _write_test_video(source)
    calls: list[Image.Image] = []

    def detector(image: Image.Image) -> list[tuple[int, int, int, int]]:
        calls.append(image)
        return [(8, 6, 32, 20)]

    summary = DetectorVideoMasker(detector)._mask_frames(source, masked)

    assert summary.frames_processed == 3
    assert summary.frames_with_redactions == 3
    assert summary.redactions_applied == 3
    assert len(calls) == 3

    capture = cv2.VideoCapture(str(masked))
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded += 1
            assert float(frame[8:18, 10:30].mean()) < 20.0
    finally:
        capture.release()
    assert decoded == 3


def test_detector_video_masker_ignores_invalid_regions(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    masked = tmp_path / "masked.mp4"
    _write_test_video(source, frames=1)

    summary = DetectorVideoMasker(lambda _image: [(10, 10, 10, 20)])._mask_frames(
        source, masked
    )

    assert summary.frames_processed == 1
    assert summary.frames_with_redactions == 0
    assert summary.redactions_applied == 0


def test_detector_video_masker_finalizes_playable_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "anonymized.mp4"
    _write_test_video(source, frames=2)

    def copy_video_without_audio(
        masked_video: Path, _original_video: Path, output_video: Path
    ) -> None:
        shutil.copyfile(masked_video, output_video)

    monkeypatch.setattr(
        DetectorVideoMasker,
        "_mux_original_audio",
        staticmethod(copy_video_without_audio),
    )
    summary = DetectorVideoMasker(lambda _image: [(4, 4, 20, 16)]).mask_video(
        source, output
    )

    assert summary.frames_processed == 2
    assert output.is_file()
    capture = cv2.VideoCapture(str(output))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    assert ok is True
    assert frame is not None
