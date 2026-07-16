from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, TypedDict, cast

import numpy as np
import pytesseract  # type: ignore[import-untyped]
from PIL import Image

from lx_anonymizer.text_detection.phi_region_detector import (
    CustomPhiRegionDetector,
    PhiRegionDetectorConfig,
)
from lx_anonymizer.text_detection.tesseract_text_detection import (
    normalize_tesseract_ocr_data,
)

Box = tuple[int, int, int, int]


class MidiBEvaluationError(RuntimeError):
    """Raised when MIDI-B inputs cannot support a valid evaluation."""


class RegionDetector(Protocol):
    name: str

    def detect(self, image: Image.Image) -> list[Box]: ...


class _PydicomModule(Protocol):
    def dcmread(self, path: str | Path, **kwargs: object) -> object: ...


class _BoxMetrics(TypedDict):
    annotations: int
    predictions: int
    true_positives: int
    best_iou_sum: float
    coverage_sum: float


@dataclass(frozen=True)
class MidiBPixelAnnotation:
    instance_uid: str
    text: str
    box: Box

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.box
        if not self.instance_uid:
            raise MidiBEvaluationError("pixel annotation has an empty instance UID")
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise MidiBEvaluationError(f"invalid MIDI-B pixel box: {self.box}")


@dataclass(frozen=True)
class MidiBInstanceResult:
    instance_uid: str
    modality: str
    annotation_count: int
    prediction_count: int
    true_positives: int
    mean_best_iou: float
    mean_ground_truth_coverage: float


@dataclass(frozen=True)
class MidiBMetricSummary:
    annotations: int
    predictions: int
    true_positives: int
    precision: float
    recall: float
    f1: float
    mean_best_iou: float
    mean_ground_truth_coverage: float


@dataclass(frozen=True)
class MidiBEvaluationReport:
    schema_version: int
    detector: str
    dataset_root: str
    answer_db: str
    iou_threshold: float
    annotated_instances: int
    evaluated_instances: int
    missing_instances: int
    failed_instances: int
    overall: MidiBMetricSummary
    by_modality: dict[str, MidiBMetricSummary]
    instances: list[MidiBInstanceResult]
    failures: list[str]

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


@dataclass
class _MetricAccumulator:
    annotations: int = 0
    predictions: int = 0
    true_positives: int = 0
    best_iou_sum: float = 0.0
    coverage_sum: float = 0.0

    def add(
        self,
        *,
        annotations: int,
        predictions: int,
        true_positives: int,
        best_iou_sum: float,
        coverage_sum: float,
    ) -> None:
        self.annotations += annotations
        self.predictions += predictions
        self.true_positives += true_positives
        self.best_iou_sum += best_iou_sum
        self.coverage_sum += coverage_sum

    def summarize(self) -> MidiBMetricSummary:
        precision = _safe_ratio(self.true_positives, self.predictions)
        recall = _safe_ratio(self.true_positives, self.annotations)
        f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
        return MidiBMetricSummary(
            annotations=self.annotations,
            predictions=self.predictions,
            true_positives=self.true_positives,
            precision=precision,
            recall=recall,
            f1=f1,
            mean_best_iou=_safe_ratio(self.best_iou_sum, self.annotations),
            mean_ground_truth_coverage=_safe_ratio(self.coverage_sum, self.annotations),
        )


class TesseractRegionDetector:
    name = "tesseract"

    def __init__(self, confidence_threshold: float = 0.5):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.confidence_threshold = confidence_threshold

    def detect(self, image: Image.Image) -> list[Box]:
        raw_payload = cast(
            object,
            pytesseract.image_to_data(  # pyright: ignore[reportUnknownMemberType]
                image,
                output_type=pytesseract.Output.DICT,
                config="--oem 3 --psm 11",
            ),
        )
        data = normalize_tesseract_ocr_data(raw_payload)
        minimum_confidence = self.confidence_threshold * 100.0
        boxes: list[Box] = []
        for index, text in enumerate(data.text):
            if not text.strip() or data.conf[index] < minimum_confidence:
                continue
            x1 = data.left[index]
            y1 = data.top[index]
            x2 = x1 + data.width[index]
            y2 = y1 + data.height[index]
            if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
        return boxes


class PhiModelRegionDetector:
    name = "phi-model"

    def __init__(self, config: PhiRegionDetectorConfig):
        self._detector = CustomPhiRegionDetector(config)

    def detect(self, image: Image.Image) -> list[Box]:
        return self._detector.detect(image)


def load_midi_b_pixel_annotations(answer_db: Path) -> list[MidiBPixelAnnotation]:
    answer_db = answer_db.expanduser().resolve()
    if not answer_db.is_file():
        raise MidiBEvaluationError(f"MIDI-B answer database not found: {answer_db}")

    with sqlite3.connect(answer_db) as connection:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(answer_data)").fetchall()
        }
        required_columns = {"SOPInstanceUID", "AnswerData"}
        if not required_columns.issubset(columns):
            raise MidiBEvaluationError(
                "answer_data must contain SOPInstanceUID and AnswerData columns"
            )
        rows = connection.execute(
            "SELECT SOPInstanceUID, AnswerData FROM answer_data"
        ).fetchall()

    annotations: list[MidiBPixelAnnotation] = []
    for raw_uid, raw_answer_data in rows:
        if not isinstance(raw_uid, str) or not isinstance(raw_answer_data, str):
            raise MidiBEvaluationError("answer_data contains non-text required values")
        uid = _strip_angle_brackets(raw_uid.strip())
        answer_payload = _load_json_mapping(raw_answer_data, "AnswerData")
        for raw_action in answer_payload.values():
            if not isinstance(raw_action, Mapping):
                raise MidiBEvaluationError("AnswerData action must be an object")
            action_mapping = cast(Mapping[str, object], raw_action)
            if action_mapping.get("action") != "<pixels_hidden>":
                continue
            action_text = action_mapping.get("action_text")
            if not isinstance(action_text, str):
                raise MidiBEvaluationError(
                    "pixels_hidden action_text must be a JSON string"
                )
            action = _load_json_mapping(
                _strip_angle_brackets(action_text.strip()), "pixels_hidden action_text"
            )
            annotations.append(
                MidiBPixelAnnotation(
                    instance_uid=uid,
                    text=_required_string(action, "text"),
                    box=_parse_annotation_box(action),
                )
            )

    if not annotations:
        raise MidiBEvaluationError(
            "the answer database contains no pixels_hidden annotations"
        )
    return sorted(annotations, key=lambda item: (item.instance_uid, item.box))


def evaluate_midi_b_pixel_detector(
    *,
    dataset_root: Path,
    answer_db: Path,
    detector: RegionDetector,
    iou_threshold: float = 0.5,
    max_instances: int | None = None,
) -> MidiBEvaluationReport:
    dataset_root = dataset_root.expanduser().resolve()
    answer_db = answer_db.expanduser().resolve()
    if not dataset_root.is_dir():
        raise MidiBEvaluationError(f"MIDI-B dataset root not found: {dataset_root}")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be greater than 0 and at most 1")
    if max_instances is not None and max_instances < 1:
        raise ValueError("max_instances must be at least 1")

    annotations = load_midi_b_pixel_annotations(answer_db)
    annotations_by_uid: dict[str, list[MidiBPixelAnnotation]] = defaultdict(list)
    for annotation in annotations:
        annotations_by_uid[annotation.instance_uid].append(annotation)

    selected_uids = sorted(annotations_by_uid)
    if max_instances is not None:
        selected_uids = selected_uids[:max_instances]
    selected_uid_set = set(selected_uids)
    dicom_paths = _index_dicom_instances(dataset_root, selected_uid_set)

    overall = _MetricAccumulator()
    modality_metrics: dict[str, _MetricAccumulator] = defaultdict(_MetricAccumulator)
    instance_results: list[MidiBInstanceResult] = []
    failures: list[str] = []
    missing_instances = 0
    failed_instances = 0

    pydicom = _load_pydicom()
    for uid in selected_uids:
        path = dicom_paths.get(uid)
        if path is None:
            missing_instances += 1
            failures.append(f"missing DICOM instance: {uid}")
            continue
        ground_truth = [item.box for item in annotations_by_uid[uid]]
        try:
            dataset = pydicom.dcmread(path)
            modality = _optional_text_attr(dataset, "Modality") or "UNKNOWN"
            image = _dicom_dataset_to_image(dataset)
            predictions = detector.detect(image)
            metrics = _evaluate_boxes(
                ground_truth, predictions, iou_threshold=iou_threshold
            )
        except (
            AttributeError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            failed_instances += 1
            failures.append(f"{uid}: {type(exc).__name__}: {exc}")
            continue

        overall.add(**metrics)
        modality_metrics[modality].add(**metrics)
        annotation_count = len(ground_truth)
        instance_results.append(
            MidiBInstanceResult(
                instance_uid=uid,
                modality=modality,
                annotation_count=annotation_count,
                prediction_count=len(predictions),
                true_positives=metrics["true_positives"],
                mean_best_iou=_safe_ratio(metrics["best_iou_sum"], annotation_count),
                mean_ground_truth_coverage=_safe_ratio(
                    metrics["coverage_sum"], annotation_count
                ),
            )
        )

    return MidiBEvaluationReport(
        schema_version=1,
        detector=detector.name,
        dataset_root=str(dataset_root),
        answer_db=str(answer_db),
        iou_threshold=iou_threshold,
        annotated_instances=len(selected_uids),
        evaluated_instances=len(instance_results),
        missing_instances=missing_instances,
        failed_instances=failed_instances,
        overall=overall.summarize(),
        by_modality={
            modality: accumulator.summarize()
            for modality, accumulator in sorted(modality_metrics.items())
        },
        instances=instance_results,
        failures=failures,
    )


def write_midi_b_report(report: MidiBEvaluationReport, output_path: Path) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lx-anonymizer-evaluate-midi-b",
        description="Evaluate PHI-region detection against MIDI-B pixel answer boxes.",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--answer-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--detector", choices=("phi-model", "tesseract"), default="phi-model"
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-instances", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    detector_name = cast(str, args.detector)
    if detector_name == "phi-model":
        model_path = cast(Path | None, args.model_path)
        if model_path is None:
            parser.error("--model-path is required for --detector phi-model")
        detector: RegionDetector = PhiModelRegionDetector(
            PhiRegionDetectorConfig(
                model_path=model_path.expanduser().resolve(),
                confidence_threshold=cast(float, args.confidence_threshold),
                nms_threshold=cast(float, args.nms_threshold),
                input_size=cast(int, args.input_size),
                box_format="yolo_xywh",
                score_format="class_scores",
                allowed_class_ids=frozenset(),
                resize_mode="letterbox",
                required=True,
            )
        )
    else:
        detector = TesseractRegionDetector(
            confidence_threshold=cast(float, args.confidence_threshold)
        )

    report = evaluate_midi_b_pixel_detector(
        dataset_root=cast(Path, args.dataset_root),
        answer_db=cast(Path, args.answer_db),
        detector=detector,
        iou_threshold=cast(float, args.iou_threshold),
        max_instances=cast(int | None, args.max_instances),
    )
    write_midi_b_report(report, cast(Path, args.output))
    print(json.dumps(report.to_dict(), ensure_ascii=True))
    return 0


def _load_pydicom() -> _PydicomModule:
    try:
        import pydicom  # type: ignore[import-untyped]
    except ImportError as exc:
        raise MidiBEvaluationError(
            "MIDI-B evaluation requires pydicom; install lx-anonymizer[evaluation]"
        ) from exc
    return cast(_PydicomModule, pydicom)


def index_midi_b_dicom_instances(
    dataset_root: Path, requested_uids: set[str]
) -> dict[str, Path]:
    """Index requested MIDI-B SOP instances without decoding pixel data."""
    return _index_dicom_instances(dataset_root, requested_uids)


def load_midi_b_dicom_image(path: Path) -> Image.Image:
    """Decode the first image frame using the evaluation rendering contract."""
    dataset = _load_pydicom().dcmread(path)
    return _dicom_dataset_to_image(dataset)


def _index_dicom_instances(
    dataset_root: Path, requested_uids: set[str]
) -> dict[str, Path]:
    pydicom = _load_pydicom()
    indexed: dict[str, Path] = {}
    for path in dataset_root.rglob("*.dcm"):
        try:
            dataset = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                specific_tags=["SOPInstanceUID"],
            )
            uid = _optional_text_attr(dataset, "SOPInstanceUID")
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            continue
        if uid in requested_uids:
            indexed[uid] = path
            if len(indexed) == len(requested_uids):
                break
    return indexed


def _dicom_dataset_to_image(dataset: object) -> Image.Image:
    raw_pixels = getattr(dataset, "pixel_array", None)
    if raw_pixels is None:
        raise MidiBEvaluationError("DICOM instance has no decodable pixel data")
    pixels = np.asarray(raw_pixels)
    if pixels.ndim == 4:
        pixels = pixels[0]
    elif pixels.ndim == 3 and pixels.shape[-1] not in (3, 4):
        pixels = pixels[0]
    if pixels.ndim not in (2, 3):
        raise MidiBEvaluationError(f"unsupported DICOM pixel shape: {pixels.shape}")

    if pixels.ndim == 3:
        rgb = _normalize_color_pixels(pixels)
        return Image.fromarray(rgb).convert("RGB")

    grayscale = _normalize_grayscale_pixels(pixels)
    if _optional_text_attr(dataset, "PhotometricInterpretation") == "MONOCHROME1":
        grayscale = 255 - grayscale
    return Image.fromarray(grayscale).convert("RGB")


def _normalize_grayscale_pixels(pixels: np.ndarray) -> np.ndarray:
    values = pixels.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise MidiBEvaluationError("DICOM pixels contain no finite values")
    lower, upper = np.percentile(finite, (0.5, 99.5))
    if upper <= lower:
        lower = float(np.min(finite))
        upper = float(np.max(finite))
    if upper <= lower:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (values - lower) * (255.0 / (upper - lower))
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def _normalize_color_pixels(pixels: np.ndarray) -> np.ndarray:
    color = pixels[..., :3]
    if color.dtype == np.uint8:
        return color
    values = color.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise MidiBEvaluationError("DICOM pixels contain no finite values")
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if upper <= lower:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.clip((values - lower) * (255.0 / (upper - lower)), 0.0, 255.0).astype(
        np.uint8
    )


def _evaluate_boxes(
    ground_truth: list[Box], predictions: list[Box], *, iou_threshold: float
) -> _BoxMetrics:
    pairs = sorted(
        (
            (_box_iou(gt_box, predicted_box), gt_index, prediction_index)
            for gt_index, gt_box in enumerate(ground_truth)
            for prediction_index, predicted_box in enumerate(predictions)
        ),
        reverse=True,
    )
    matched_ground_truth: set[int] = set()
    matched_predictions: set[int] = set()
    for iou, gt_index, prediction_index in pairs:
        if iou < iou_threshold:
            break
        if gt_index in matched_ground_truth or prediction_index in matched_predictions:
            continue
        matched_ground_truth.add(gt_index)
        matched_predictions.add(prediction_index)

    best_iou_sum = sum(
        max((_box_iou(gt_box, box) for box in predictions), default=0.0)
        for gt_box in ground_truth
    )
    coverage_sum = sum(
        _ground_truth_coverage(gt_box, predictions) for gt_box in ground_truth
    )
    return {
        "annotations": len(ground_truth),
        "predictions": len(predictions),
        "true_positives": len(matched_ground_truth),
        "best_iou_sum": best_iou_sum,
        "coverage_sum": coverage_sum,
    }


def _box_iou(first: Box, second: Box) -> float:
    intersection = _intersection_area(first, second)
    if intersection == 0:
        return 0.0
    union = _box_area(first) + _box_area(second) - intersection
    return _safe_ratio(intersection, union)


def _ground_truth_coverage(ground_truth: Box, predictions: list[Box]) -> float:
    """Return the union of predicted areas overlapping one ground-truth box."""
    clipped: list[Box] = []
    for prediction in predictions:
        intersection = (
            max(ground_truth[0], prediction[0]),
            max(ground_truth[1], prediction[1]),
            min(ground_truth[2], prediction[2]),
            min(ground_truth[3], prediction[3]),
        )
        if _box_area(intersection) > 0:
            clipped.append(intersection)
    if not clipped:
        return 0.0

    x_edges = sorted({edge for box in clipped for edge in (box[0], box[2])})
    covered_area = 0
    for left, right in zip(x_edges, x_edges[1:]):
        if right <= left:
            continue
        y_intervals = sorted(
            (box[1], box[3]) for box in clipped if box[0] < right and box[2] > left
        )
        if not y_intervals:
            continue
        merged_height = 0
        start, end = y_intervals[0]
        for next_start, next_end in y_intervals[1:]:
            if next_start > end:
                merged_height += end - start
                start, end = next_start, next_end
            else:
                end = max(end, next_end)
        merged_height += end - start
        covered_area += (right - left) * merged_height
    return _safe_ratio(covered_area, _box_area(ground_truth))


def _intersection_area(first: Box, second: Box) -> int:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def _box_area(box: Box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _load_json_mapping(raw: str, field_name: str) -> Mapping[str, object]:
    try:
        payload: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MidiBEvaluationError(f"{field_name} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise MidiBEvaluationError(f"{field_name} must be a JSON object")
    return cast(Mapping[str, object], payload)


def _parse_annotation_box(action: Mapping[str, object]) -> Box:
    top_left = _coordinate_pair(action.get("top_left"), "top_left")
    bottom_right = _coordinate_pair(action.get("bottom_right"), "bottom_right")
    return (top_left[0], top_left[1], bottom_right[0], bottom_right[1])


def _coordinate_pair(value: object, field_name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)):
        raise MidiBEvaluationError(f"{field_name} must contain two coordinates")
    coordinates = cast(list[object] | tuple[object, ...], value)
    if len(coordinates) != 2:
        raise MidiBEvaluationError(f"{field_name} must contain two coordinates")
    x, y = coordinates
    if not isinstance(x, int) or not isinstance(y, int):
        raise MidiBEvaluationError(f"{field_name} coordinates must be integers")
    return x, y


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise MidiBEvaluationError(f"{key} must be a string")
    return value


def _optional_text_attr(dataset: object, attribute: str) -> str:
    value = getattr(dataset, attribute, "")
    return str(value).strip() if value is not None else ""


def _strip_angle_brackets(value: str) -> str:
    if value.startswith("<") and value.endswith(">"):
        return value[1:-1]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
