from __future__ import annotations

import threading
from fractions import Fraction
from pathlib import Path

import pytest

from lx_anonymizer.frame_cleaner import FrameCleaner


def test_frame_cleaner_rejects_concurrent_instance_reuse(tmp_path: Path) -> None:
    cleaner = FrameCleaner.__new__(FrameCleaner)
    cleaner._run_lock = threading.Lock()  # pyright: ignore[reportPrivateUsage]
    assert cleaner._run_lock.acquire(blocking=False)  # pyright: ignore[reportPrivateUsage]
    try:
        with pytest.raises(RuntimeError, match="single-run resources"):
            cleaner.clean_video(
                video_path=tmp_path / "input.mp4",
                endoscope_image_roi=None,
                endoscope_data_roi_nested=None,
                source_frame_rate=Fraction(25, 1),
            )
    finally:
        cleaner._run_lock.release()  # pyright: ignore[reportPrivateUsage]
