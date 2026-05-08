from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from lx_anonymizer.text_detection import phi_region_detector
from lx_anonymizer.text_detection.phi_region_detector import (
    CustomPhiRegionDetector,
    PhiRegionDetectorConfig,
)


class _FakeNet:
    def __init__(self, output: np.ndarray):
        self.output = output
        self.input_shape: tuple[int, ...] | None = None

    def setInput(self, blob: np.ndarray) -> None:
        self.input_shape = tuple(blob.shape)

    def forward(self) -> np.ndarray:
        return self.output


def test_custom_phi_region_detector_parses_yolo_xywh_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "phi_detector.onnx"
    model_path.write_bytes(b"fake model")
    fake_net = _FakeNet(
        np.array(
            [
                [
                    [50.0, 50.0, 20.0, 10.0, 0.90],
                    [20.0, 20.0, 10.0, 10.0, 0.10],
                ]
            ],
            dtype=np.float32,
        )
    )

    monkeypatch.setattr(
        phi_region_detector.cv2_dnn,
        "readNet",
        lambda path: fake_net,
    )

    config = PhiRegionDetectorConfig(
        model_path=model_path,
        confidence_threshold=0.5,
        nms_threshold=0.45,
        input_size=100,
        box_format="yolo_xywh",
        score_format="class_scores",
        allowed_class_ids=frozenset(),
    )

    detector = CustomPhiRegionDetector(config)
    regions = detector.detect(Image.new("RGB", (100, 100), "white"))

    assert fake_net.input_shape == (1, 3, 100, 100)
    assert regions == [(40, 45, 60, 55)]


def test_custom_phi_region_detector_filters_class_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "phi_detector.onnx"
    model_path.write_bytes(b"fake model")
    fake_net = _FakeNet(
        np.array(
            [
                [
                    [50.0, 50.0, 20.0, 10.0, 0.95, 0.05],
                    [25.0, 25.0, 10.0, 10.0, 0.10, 0.90],
                ]
            ],
            dtype=np.float32,
        )
    )

    monkeypatch.setattr(
        phi_region_detector.cv2_dnn,
        "readNet",
        lambda path: fake_net,
    )

    config = PhiRegionDetectorConfig(
        model_path=model_path,
        confidence_threshold=0.5,
        nms_threshold=0.45,
        input_size=100,
        box_format="yolo_xywh",
        score_format="class_scores",
        allowed_class_ids=frozenset({1}),
    )

    detector = CustomPhiRegionDetector(config)
    regions = detector.detect(Image.new("RGB", (100, 100), "white"))

    assert regions == [(20, 20, 30, 30)]
