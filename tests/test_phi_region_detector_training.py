from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lx_anonymizer.text_detection.phi_region_detector_training import (
    PhiRegionDetectorTrainingConfig,
    PhiRegionDetectorTrainingError,
    train_phi_region_detector,
)


def test_phi_region_detector_training_rejects_missing_dataset_yaml(
    tmp_path: Path,
) -> None:
    with pytest.raises(PhiRegionDetectorTrainingError):
        PhiRegionDetectorTrainingConfig(
            dataset_yaml=tmp_path / "missing.yaml",
            output_dir=tmp_path / "runs",
        )


def test_phi_region_detector_training_exports_onnx_with_settings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text(
        "path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8"
    )

    class FakeYOLO:
        def __init__(self, model: str):
            self.model = model
            self.trainer = SimpleNamespace(save_dir=None)

        def train(self, **kwargs):
            run_dir = Path(kwargs["project"]) / kwargs["name"]
            weights_dir = run_dir / "weights"
            weights_dir.mkdir(parents=True)
            (weights_dir / "best.pt").write_bytes(b"checkpoint")
            self.trainer.save_dir = run_dir
            return SimpleNamespace(save_dir=run_dir)

        def export(self, **kwargs):
            onnx_path = Path(self.model).with_suffix(".onnx")
            onnx_path.write_bytes(b"onnx")
            return str(onnx_path)

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))

    result = train_phi_region_detector(
        PhiRegionDetectorTrainingConfig(
            dataset_yaml=dataset_yaml,
            output_dir=tmp_path / "runs",
            run_name="phi-test",
            epochs=1,
            batch_size=2,
            input_size=640,
            confidence_threshold=0.4,
            nms_threshold=0.5,
        )
    )

    model_path = Path(result["model_path"])
    assert model_path.name == "best.onnx"
    assert model_path.exists()
    assert result["settings"]["PHI_REGION_DETECTOR_MODEL_PATH"] == str(model_path)
    assert result["settings"]["PHI_REGION_DETECTOR_MODEL_SHA256"]
    assert result["settings"]["PHI_REGION_DETECTOR_INPUT_SIZE"] == 640
    assert Path(result["meta_path"]).exists()
    assert Path(result["training_result_path"]).exists()
