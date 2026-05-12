from dataclasses import dataclass
import threading
import time
from typing import Any

import numpy as np
import pytest

from lx_anonymizer.ocr import ocr_frame as ocr_mod
from lx_anonymizer.ocr.ocr_frame import FrameOCR


@dataclass
class FakeRapidOCROutput:
    boxes: np.ndarray
    txts: tuple[str, ...]
    scores: tuple[float, ...]
    elapse: float = 0.01


class FakeRapidOCREngine:
    def __init__(self, outputs: list[FakeRapidOCROutput]):
        self.outputs = outputs
        self.input_shapes: list[tuple[int, ...]] = []

    def __call__(self, image: np.ndarray) -> FakeRapidOCROutput:
        self.input_shapes.append(image.shape)
        return self.outputs.pop(0)


def _frame_ocr_with_engine(engine: Any) -> FrameOCR:
    frame_ocr = FrameOCR.__new__(FrameOCR)
    frame_ocr.rapidocr_engine = engine
    frame_ocr._rapidocr_lock = threading.Lock()
    return frame_ocr


def test_rapidocr_full_frame_normalizes_modern_output() -> None:
    engine = FakeRapidOCREngine(
        [
            FakeRapidOCROutput(
                boxes=np.array(
                    [
                        [[5, 20], [60, 20], [60, 30], [5, 30]],
                        [[5, 5], [80, 5], [80, 15], [5, 15]],
                    ],
                    dtype=np.float32,
                ),
                txts=("09:53:32", "Patient Mueller"),
                scores=(0.85, 0.95),
                elapse=0.12,
            )
        ]
    )
    frame_ocr = _frame_ocr_with_engine(engine)

    text, confidence, metadata = frame_ocr._extract_text_rapidocr(
        np.zeros((100, 200), dtype=np.uint8), roi=None
    )

    assert text == "Patient Mueller 09:53:32"
    assert confidence == pytest.approx(0.90)
    assert metadata["backend"] == "rapidocr"
    assert metadata["method"] == "rapidocr"
    assert metadata["regions"] == 2
    assert metadata["words"] == 3
    assert metadata["elapse"] == pytest.approx(0.12)
    assert engine.input_shapes == [(100, 200)]


def test_rapidocr_nested_rois_crop_and_offset_boxes() -> None:
    engine = FakeRapidOCREngine(
        [
            FakeRapidOCROutput(
                boxes=np.array([[[1, 2], [11, 2], [11, 8], [1, 8]]]),
                txts=("Patient Mueller",),
                scores=(0.8,),
            ),
            FakeRapidOCROutput(
                boxes=np.array([[[3, 4], [23, 4], [23, 10], [3, 10]]]),
                txts=("09:53:32",),
                scores=(0.6,),
            ),
        ]
    )
    frame_ocr = _frame_ocr_with_engine(engine)
    roi = {
        "patient": {"x": 20, "y": 30, "width": 40, "height": 20},
        "time": {"x": 100, "y": 50, "width": 50, "height": 25},
    }

    text, confidence, metadata = frame_ocr._extract_text_rapidocr(
        np.zeros((100, 200), dtype=np.uint8), roi=roi
    )

    assert text == "Patient Mueller\n09:53:32"
    assert confidence == pytest.approx(0.7)
    assert metadata["method"] == "rapidocr+roi"
    assert metadata["roi_count"] == 2
    assert metadata["roi_0"] == "Patient Mueller"
    assert metadata["roi_1"] == "09:53:32"
    assert metadata["text_regions"][0]["bbox"] == [21, 32, 31, 38]
    assert metadata["text_regions"][1]["bbox"] == [103, 54, 123, 60]
    assert engine.input_shapes == [(20, 40), (25, 50)]


def test_rapidocr_lazy_initialization_is_locked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_count = 0
    init_count_lock = threading.Lock()

    class LazyEngine:
        def __init__(self) -> None:
            nonlocal init_count
            time.sleep(0.02)
            with init_count_lock:
                init_count += 1

        def __call__(self, image: np.ndarray) -> FakeRapidOCROutput:
            return FakeRapidOCROutput(
                boxes=np.array([[[0, 0], [10, 0], [10, 5], [0, 5]]]),
                txts=("Patient Mueller",),
                scores=(0.9,),
            )

    monkeypatch.setattr(ocr_mod, "RAPIDOCR_AVAILABLE", True)
    monkeypatch.setattr(ocr_mod, "_RapidOCR", LazyEngine)

    frame_ocr = FrameOCR.__new__(FrameOCR)
    frame_ocr.rapidocr_engine = None
    frame_ocr._rapidocr_lock = threading.Lock()
    frame_ocr.tesserocr_processor = None
    frame_ocr._rapidocr_available = True

    barrier = threading.Barrier(4)
    results: list[tuple[str, float]] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait()
            text, confidence, _ = frame_ocr.extract_text_from_frame(
                np.zeros((20, 20), dtype=np.uint8), roi=None
            )
            results.append((text, confidence))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert init_count == 1
    assert results == [("Patient Mueller", 0.9)] * 4
