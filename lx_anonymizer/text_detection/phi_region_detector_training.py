from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


class PhiRegionDetectorTrainingError(RuntimeError):
    """Raised when PHI-region detector training cannot be started or completed."""


@dataclass(frozen=True)
class PhiRegionDetectorTrainingConfig:
    dataset_yaml: Path
    output_dir: Path
    base_model: str = "yolov8n.pt"
    run_name: str | None = None
    epochs: int = 50
    batch_size: int = 16
    input_size: int = 640
    device: str = "auto"
    workers: int = 4
    patience: int = 25
    export_onnx: bool = True
    confidence_threshold: float = 0.35
    nms_threshold: float = 0.45
    class_ids: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_yaml", _coerce_local_path(self.dataset_yaml))
        object.__setattr__(self, "output_dir", _coerce_local_path(self.output_dir))

        if not self.dataset_yaml.is_file():
            raise PhiRegionDetectorTrainingError(
                f"PHI detector dataset YAML does not exist: {self.dataset_yaml}"
            )
        if self.epochs < 1:
            raise PhiRegionDetectorTrainingError("epochs must be >= 1")
        if self.batch_size < 1:
            raise PhiRegionDetectorTrainingError("batch_size must be >= 1")
        if self.input_size < 32:
            raise PhiRegionDetectorTrainingError("input_size must be >= 32")
        if self.workers < 0:
            raise PhiRegionDetectorTrainingError("workers must be >= 0")
        if self.patience < 0:
            raise PhiRegionDetectorTrainingError("patience must be >= 0")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise PhiRegionDetectorTrainingError(
                "confidence_threshold must be between 0 and 1"
            )
        if not 0.0 <= self.nms_threshold <= 1.0:
            raise PhiRegionDetectorTrainingError(
                "nms_threshold must be between 0 and 1"
            )


def train_phi_region_detector(
    config: PhiRegionDetectorTrainingConfig,
) -> dict[str, Any]:
    """
    Train a YOLO-style PHI-region detector and export the OpenCV DNN ONNX artifact.

    The resulting ONNX file matches the runtime contract in
    `phi_region_detector.py`: RGB input, square image size, YOLO xywh boxes, and
    class-score rows. Training dependencies are intentionally optional so normal
    anonymizer deployments do not pull a full training stack.
    """

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise PhiRegionDetectorTrainingError(
            "PHI detector training requires the optional dependency `ultralytics`. "
            "Install the package with the training extra before starting this run."
        ) from exc

    run_name = config.run_name or _default_run_name()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(config.base_model)
    train_kwargs: dict[str, Any] = {
        "data": str(config.dataset_yaml),
        "epochs": config.epochs,
        "batch": config.batch_size,
        "imgsz": config.input_size,
        "project": str(config.output_dir),
        "name": run_name,
        "workers": config.workers,
        "patience": config.patience,
        "exist_ok": True,
    }
    if config.device.strip() and config.device != "auto":
        train_kwargs["device"] = config.device.strip()

    training_result = model.train(**train_kwargs)
    run_dir = _resolve_run_dir(training_result, model, config.output_dir, run_name)
    checkpoint_path = _find_checkpoint(run_dir)
    model_path = checkpoint_path
    onnx_path: Path | None = None

    if config.export_onnx:
        if checkpoint_path is None:
            raise PhiRegionDetectorTrainingError(
                f"Training completed but no best/last checkpoint was found in {run_dir}"
            )
        exported_model = YOLO(str(checkpoint_path))
        exported_path = exported_model.export(
            format="onnx",
            imgsz=config.input_size,
            simplify=False,
            opset=12,
        )
        onnx_path = Path(str(exported_path)).expanduser().resolve()
        model_path = onnx_path

    if model_path is None:
        raise PhiRegionDetectorTrainingError(
            f"Training completed but no model artifact was found in {run_dir}"
        )

    model_sha256 = _sha256_file(model_path)
    meta_path = run_dir / "phi_region_detector_meta.json"
    training_result_path = run_dir / "phi_region_detector_training_result.json"

    result: dict[str, Any] = {
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "onnx_path": str(onnx_path) if onnx_path else None,
        "meta_path": str(meta_path),
        "training_result_path": str(training_result_path),
        "run_dir": str(run_dir),
        "settings": {
            "PHI_REGION_DETECTOR_MODEL_PATH": str(model_path),
            "PHI_REGION_DETECTOR_MODEL_SHA256": model_sha256,
            "PHI_REGION_DETECTOR_CONFIDENCE": config.confidence_threshold,
            "PHI_REGION_DETECTOR_NMS_THRESHOLD": config.nms_threshold,
            "PHI_REGION_DETECTOR_INPUT_SIZE": config.input_size,
            "PHI_REGION_DETECTOR_BOX_FORMAT": "yolo_xywh",
            "PHI_REGION_DETECTOR_SCORE_FORMAT": "class_scores",
            "PHI_REGION_DETECTOR_CLASS_IDS": config.class_ids,
        },
        "config": _jsonable_config(config),
        "training_result": {
            "status": "success",
            "artifacts": [
                {
                    "kind": "checkpoint" if onnx_path is None else "model",
                    "path": str(model_path),
                    "checksum_sha256": model_sha256,
                    "bytes": model_path.stat().st_size,
                }
            ],
        },
    }
    if checkpoint_path and checkpoint_path != model_path:
        result["training_result"]["artifacts"].append(
            {
                "kind": "checkpoint",
                "path": str(checkpoint_path),
                "checksum_sha256": _sha256_file(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
            }
        )

    meta_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    training_result_path.write_text(
        json.dumps(result["training_result"], indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lx-anonymizer-train-phi",
        description="Train the lx-anonymizer PHI-region detector and export an ONNX model.",
    )
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--run-name")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--class-ids", default="")
    parser.add_argument(
        "--skip-onnx-export",
        action="store_true",
        help="Keep the PyTorch checkpoint as the primary artifact instead of exporting ONNX.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = PhiRegionDetectorTrainingConfig(
        dataset_yaml=args.dataset_yaml,
        output_dir=args.output_dir,
        base_model=args.base_model,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        input_size=args.input_size,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        export_onnx=not args.skip_onnx_export,
        confidence_threshold=args.confidence_threshold,
        nms_threshold=args.nms_threshold,
        class_ids=args.class_ids,
    )
    print(json.dumps(train_phi_region_detector(config), ensure_ascii=True))
    return 0


def _coerce_local_path(path: str | Path) -> Path:
    raw = str(path)
    if "://" in raw or raw.startswith("//"):
        raise PhiRegionDetectorTrainingError("remote paths and URLs are not accepted")
    return Path(path).expanduser().resolve()


def _default_run_name() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"phi-region-detector-{timestamp}"


def _resolve_run_dir(
    training_result: Any,
    model: Any,
    output_dir: Path,
    run_name: str,
) -> Path:
    candidates = [
        getattr(training_result, "save_dir", None),
        getattr(getattr(model, "trainer", None), "save_dir", None),
        output_dir / run_name,
    ]
    for candidate in candidates:
        if candidate:
            return Path(str(candidate)).expanduser().resolve()
    return (output_dir / run_name).resolve()


def _find_checkpoint(run_dir: Path) -> Path | None:
    for relative in ("weights/best.pt", "weights/last.pt", "best.pt", "last.pt"):
        path = run_dir / relative
        if path.is_file():
            return path.resolve()
    return None


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable_config(config: PhiRegionDetectorTrainingConfig) -> dict[str, Any]:
    data = asdict(config)
    data["dataset_yaml"] = str(config.dataset_yaml)
    data["output_dir"] = str(config.output_dir)
    return data


if __name__ == "__main__":
    raise SystemExit(main())
