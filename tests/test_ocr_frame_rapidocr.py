# pyright: reportPrivateUsage=false

from dataclasses import dataclass
import threading
import time

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from lx_anonymizer.ocr import ocr_frame as ocr_mod
from lx_anonymizer.ocr.ocr_frame import FlatRoi, FrameOCR, NestedRoi
from lx_anonymizer.llm.llm_service import LLMService


@dataclass
class FakeRapidOCROutput:
    boxes: NDArray[np.float32]
    txts: tuple[str, ...]
    scores: tuple[float, ...]
    elapse: float = 0.01


class FakeRapidOCREngine:
    def __init__(self, outputs: list[FakeRapidOCROutput]):
        self.outputs = outputs
        self.input_shapes: list[tuple[int, ...]] = []

    def __call__(self, image: NDArray[np.uint8]) -> FakeRapidOCROutput:
        self.input_shapes.append(image.shape)
        return self.outputs.pop(0)


def test_gemma4_refines_conventional_ocr_and_preserves_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def recognize_image(
        self: LLMService, image: Image.Image, candidate_text: str = ""
    ) -> str:
        observed["size"] = image.size
        observed["candidate"] = candidate_text
        return "Patient Müller"

    monkeypatch.setattr(LLMService, "recognize_image", recognize_image)
    frame_ocr = FrameOCR.__new__(FrameOCR)
    conventional = (
        "Patient Muller",
        0.82,
        {"backend": "rapidocr", "method": "rapidocr"},
    )

    text, confidence, metadata = frame_ocr._extract_text_ollama(
        np.zeros((100, 200, 3), dtype=np.uint8),
        {"x": 20, "y": 30, "width": 40, "height": 20},
        conventional,
    )

    assert text == "Patient Müller"
    assert confidence == 0.82
    assert observed == {"size": (40, 20), "candidate": "Patient Muller"}
    assert metadata["backend"] == "ollama-gemma4"
    assert metadata["candidate_backend"] == "rapidocr"
    assert metadata["confidence_source"] == "conventional_ocr"


def _frame_ocr_with_engine(engine: FakeRapidOCREngine) -> FrameOCR:
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
    assert abs(confidence - 0.90) <= 1e-12
    assert metadata["backend"] == "rapidocr"
    assert metadata["method"] == "rapidocr"
    assert metadata["regions"] == 2
    assert metadata["words"] == 3
    assert abs(float(metadata["elapse"]) - 0.12) <= 1e-12
    assert engine.input_shapes == [(100, 200)]


def test_rapidocr_nested_rois_crop_and_offset_boxes() -> None:
    engine = FakeRapidOCREngine(
        [
            FakeRapidOCROutput(
                boxes=np.array(
                    [[[1, 2], [11, 2], [11, 8], [1, 8]]],
                    dtype=np.float32,
                ),
                txts=("Patient Mueller",),
                scores=(0.8,),
            ),
            FakeRapidOCROutput(
                boxes=np.array(
                    [[[3, 4], [23, 4], [23, 10], [3, 10]]],
                    dtype=np.float32,
                ),
                txts=("09:53:32",),
                scores=(0.6,),
            ),
        ]
    )
    frame_ocr = _frame_ocr_with_engine(engine)
    roi: NestedRoi = {
        "patient": {"x": 20, "y": 30, "width": 40, "height": 20},
        "time": {"x": 100, "y": 50, "width": 50, "height": 25},
    }

    text, confidence, metadata = frame_ocr._extract_text_rapidocr(
        np.zeros((100, 200), dtype=np.uint8), roi=roi
    )

    assert text == "Patient Mueller\n09:53:32"
    assert abs(confidence - 0.7) <= 1e-12
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
    monkeypatch.setattr(ocr_mod.settings, "OLLAMA_OCR_ENABLED", False)
    init_count = 0
    init_count_lock = threading.Lock()

    class LazyEngine:
        def __init__(self) -> None:
            nonlocal init_count
            time.sleep(0.02)
            with init_count_lock:
                init_count += 1

        def __call__(self, image: NDArray[np.uint8]) -> FakeRapidOCROutput:
            return FakeRapidOCROutput(
                boxes=np.array(
                    [[[0, 0], [10, 0], [10, 5], [0, 5]]],
                    dtype=np.float32,
                ),
                txts=("Patient Mueller",),
                scores=(0.9,),
            )

    monkeypatch.setattr(ocr_mod, "rapidocr_available", True)
    monkeypatch.setattr(ocr_mod, "_RapidOCRClass", LazyEngine)

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
            roi: FlatRoi | None = None
            text, confidence, _ = frame_ocr.extract_text_from_frame(
                np.zeros((20, 20), dtype=np.uint8), roi=roi
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


def test_rapidocr_init_params_request_cuda_when_provider_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def available_providers() -> tuple[str, str]:
        return (
            ocr_mod.CUDA_EXECUTION_PROVIDER,
            ocr_mod.CPU_EXECUTION_PROVIDER,
        )

    monkeypatch.delenv(ocr_mod.RAPIDOCR_ACCELERATION_ENV, raising=False)
    monkeypatch.setattr(
        FrameOCR,
        "_available_onnx_providers",
        staticmethod(available_providers),
    )

    params = FrameOCR._rapidocr_init_params()

    assert params["EngineConfig.onnxruntime.use_cuda"] is True


def test_rapidocr_cuda_request_raises_without_cuda_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def available_providers() -> tuple[str]:
        return (ocr_mod.CPU_EXECUTION_PROVIDER,)

    monkeypatch.setenv(ocr_mod.RAPIDOCR_ACCELERATION_ENV, "cuda")
    monkeypatch.setattr(
        FrameOCR,
        "_available_onnx_providers",
        staticmethod(available_providers),
    )

    with pytest.raises(RuntimeError, match=ocr_mod.CUDA_EXECUTION_PROVIDER):
        FrameOCR._rapidocr_init_params()


def test_rapidocr_init_params_use_latin_script_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ocr_mod.RAPIDOCR_ACCELERATION_ENV, "cpu")

    params = FrameOCR._rapidocr_init_params()

    assert params["Det.lang_type"].value == "en"
    assert params["Rec.lang_type"].value == "latin"
    assert params["EngineConfig.onnxruntime.use_cuda"] is False


def test_pytesseract_fallback_uses_default_config_on_partial_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_image_to_data(
        image: object,
        lang: str,
        config: str,
        output_type: object,
    ) -> dict[str, list[str]]:
        captured["image"] = image
        captured["lang"] = lang
        captured["config"] = config
        captured["output_type"] = output_type
        return {"text": ["Patient", "Mueller"], "conf": ["92", "88"]}

    monkeypatch.setattr(ocr_mod.pytesseract, "image_to_data", fake_image_to_data)

    frame_ocr = FrameOCR.__new__(FrameOCR)
    text, confidence, metadata = frame_ocr._extract_text_pytesseract(
        np.zeros((20, 80), dtype=np.uint8)
    )

    assert text == "Patient Mueller"
    assert abs(confidence - 0.9) <= 1e-12
    assert metadata == {"words": 2, "avg_conf": 0.9}
    assert captured["lang"] == "deu+eng"
    assert captured["config"] == "--oem 3 --psm 6 --dpi 300"
