from __future__ import annotations

from pathlib import Path

import pytest

from lx_anonymizer import pipeline_manager
from lx_anonymizer.pipeline_manager import detect_combined_text_boxes


def _fixed_east_detection(
    image_path: str | Path,
    east_path: str | Path | None = None,
    min_confidence: float = 0.6,
    width: int = 320,
    height: int = 320,
) -> tuple[list[tuple[int, int, int, int]], str]:
    del image_path, east_path, min_confidence, width, height
    return [(1, 2, 3, 4)], "[]"


def _fixed_tesseract_detection(
    image_path: str | Path,
    min_confidence: float = 0.5,
    width: int = 320,
    height: int = 320,
) -> tuple[list[tuple[int, int, int, int]], str]:
    del image_path, min_confidence, width, height
    return [(5, 6, 7, 8)], "[]"


def test_combined_text_detection_skips_unavailable_craft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_manager, "CRAFT_AVAILABLE", False)
    monkeypatch.setattr(
        pipeline_manager,
        "east_text_detection",
        _fixed_east_detection,
    )
    monkeypatch.setattr(
        pipeline_manager,
        "tesseract_text_detection",
        _fixed_tesseract_detection,
    )

    def unexpected_craft(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("CRAFT must not be called when unavailable")

    monkeypatch.setattr(pipeline_manager, "craft_text_detection", unexpected_craft)

    boxes = detect_combined_text_boxes(Path("image.png"), "east.pb", 0.5, 320, 320)

    assert boxes == [(1, 2, 3, 4), (5, 6, 7, 8)]
