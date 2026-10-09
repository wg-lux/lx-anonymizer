import shutil
import subprocess
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from lx_anonymizer.anonymization.detector_video_masking import DetectorVideoMasker
from lx_anonymizer.runtime_types import VideoExecutionHooks


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

    written_indices: list[int] = []
    summary = DetectorVideoMasker(detector)._mask_frames(
        source, masked, written_indices.append
    )

    assert summary.frames_processed == 3
    assert summary.frames_with_redactions == 3
    assert summary.redactions_applied == 3
    assert len(calls) == 3
    assert written_indices == [0, 1, 2]

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

    def copy_video_without_audio(masked_video: Path, output_video: Path) -> None:
        shutil.copyfile(masked_video, output_video)

    monkeypatch.setattr(
        DetectorVideoMasker,
        "_finalize_video_only",
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


@pytest.mark.parametrize("finalization_fails", [False, True])
def test_observer_encloses_written_frames_and_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finalization_fails: bool
) -> None:
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    _write_test_video(source, frames=2)
    events: list[str | int] = []

    @contextmanager
    def observe(
        input_path: Path, output_path: Path
    ) -> Generator[Callable[[int], None], None, None]:
        assert (input_path, output_path) == (source, output)
        events.append("start")
        try:
            yield events.append
        except RuntimeError:
            events.append("failed")
            raise
        else:
            assert output.is_file()
            events.append("complete")

    def encode(_command: list[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(
            "Decoder masking must not invent a second encoder ancestry"
        )

    def finalize(masked: Path, destination: Path) -> None:
        events.append("finalize")
        if finalization_fails:
            raise RuntimeError("finalization failed")
        shutil.copyfile(masked, destination)

    monkeypatch.setattr(
        DetectorVideoMasker, "_finalize_video_only", staticmethod(finalize)
    )
    masker = DetectorVideoMasker(
        lambda _image: [], execution_hooks=VideoExecutionHooks(encode, observe)
    )
    if finalization_fails:
        with pytest.raises(RuntimeError, match="finalization failed"):
            masker.mask_video(source, output)
    else:
        assert masker.mask_video(source, output).frames_processed == 2
    assert events == [
        "start",
        0,
        1,
        "finalize",
        "failed" if finalization_fails else "complete",
    ]
