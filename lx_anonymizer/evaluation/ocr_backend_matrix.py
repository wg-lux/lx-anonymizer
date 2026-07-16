from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TextIO, TypeAlias, TypedDict, cast

import cv2
import numpy as np
import numpy.typing as npt
import pytesseract  # type: ignore[import-untyped]
from PIL import Image

from lx_anonymizer.config import settings
from lx_anonymizer.image_processing.pdf_operations import convert_pdf_to_images
from lx_anonymizer.llm.llm_service import LLMService
from lx_anonymizer.ner.frame_metadata_extractor import FrameMetadataExtractor
from lx_anonymizer.ocr.ocr_ensemble import ensemble_ocr_with_details
from lx_anonymizer.ocr.ocr_frame import FrameOCR, RoiInput
from lx_anonymizer.ocr.ocr_frame_tesserocr import get_tesseocr_processor
from lx_anonymizer.ocr.ocr_preprocessing import optimize_image_for_medical_text
from lx_anonymizer.text_detection.phi_region_detector import (
    detect_phi_regions_from_settings,
)

if TYPE_CHECKING:
    from lx_anonymizer.report_reader import ReportReader

PhiField: TypeAlias = Literal[
    "first_name",
    "last_name",
    "dob",
    "casenumber",
    "examination_date",
]
FieldValues: TypeAlias = dict[PhiField, str | None]
JsonObject: TypeAlias = dict[str, object]
FrameArray: TypeAlias = npt.NDArray[np.uint8]
PipelineStatus: TypeAlias = Literal["ok", "failed", "skipped"]

PHI_FIELDS: tuple[PhiField, ...] = (
    "first_name",
    "last_name",
    "dob",
    "casenumber",
    "examination_date",
)

UTILITY_WEIGHTS: dict[str, float] = {
    "phi_field_recall": 0.35,
    "field_accuracy": 0.25,
    "text_score": 0.15,
    "box_coverage": 0.05,
    "over_redaction_score": 0.05,
    "speed_score": 0.10,
    "stability_score": 0.05,
}
PHI_RECALL_HARD_GATE = 0.95
DEFAULT_REPORT_NATIVE_MIN_CHARS = 50
DEFAULT_VIDEO_SPEED_TARGET_SECONDS = 0.50
DEFAULT_REPORT_SPEED_TARGET_SECONDS = 5.00
DEFAULT_MIN_SAMPLES_PER_MODALITY = 20
DEFAULT_MIN_SUBJECTS = 5
DEFAULT_MIN_SOURCE_GROUPS_PER_MODALITY = 3

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
_DIGIT_RE = re.compile(r"\d+")


class InputType(str, Enum):
    VIDEO_FRAME = "video_frame"
    TEXT_REPORT = "text_report_document"

    @property
    def label(self) -> str:
        if self is InputType.VIDEO_FRAME:
            return "Video Frame"
        return "Text Report Document"


class PipelineId(str, Enum):
    V1 = "V1"
    V2 = "V2"
    V3 = "V3"
    V4 = "V4"
    V5 = "V5"
    V6 = "V6"
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    R4 = "R4"


class TesseractData(TypedDict):
    text: list[str]
    left: list[int | str]
    top: list[int | str]
    width: list[int | str]
    height: list[int | str]
    conf: list[int | str]


class _TesseractOutput(Protocol):
    DICT: object


class _PytesseractModule(Protocol):
    Output: _TesseractOutput

    def image_to_string(
        self,
        image: Image.Image,
        *,
        config: str = "",
        lang: str | None = None,
    ) -> str | bytes: ...

    def image_to_data(
        self,
        image: Image.Image,
        *,
        config: str = "",
        output_type: object,
        lang: str | None = None,
    ) -> TesseractData: ...


class _Cv2Runtime(Protocol):
    IMREAD_COLOR: int
    COLOR_BGR2RGB: int

    def imread(self, filename: str, flags: int) -> FrameArray | None: ...

    def cvtColor(self, src: FrameArray, code: int) -> FrameArray: ...


_PYTESSERACT = cast(_PytesseractModule, pytesseract)
_CV2 = cast(_Cv2Runtime, cv2)


class PipelineRunner(Protocol):
    def __call__(self, item: "GoldenSetItem") -> "PipelinePrediction": ...


@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    page_index: int | None = None

    @classmethod
    def from_value(cls, value: object) -> "BoundingBox":
        if isinstance(value, Mapping):
            mapping = cast(Mapping[object, object], value)
            return cls._from_mapping(mapping)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            sequence = cast(Sequence[object], value)
            return cls._from_sequence(sequence)
        raise TypeError(
            f"Bounding box must be a mapping or 4-value sequence: {value!r}"
        )

    @classmethod
    def _from_mapping(cls, value: Mapping[object, object]) -> "BoundingBox":
        keys = {str(key): item for key, item in value.items()}
        page_index = _optional_int(keys.get("page_index", keys.get("page")))

        if {"x1", "y1", "x2", "y2"}.issubset(keys):
            return cls(
                x1=_required_int(keys.get("x1"), "x1"),
                y1=_required_int(keys.get("y1"), "y1"),
                x2=_required_int(keys.get("x2"), "x2"),
                y2=_required_int(keys.get("y2"), "y2"),
                page_index=page_index,
            ).validated()

        if {"x", "y", "width", "height"}.issubset(keys):
            x = _required_int(keys.get("x"), "x")
            y = _required_int(keys.get("y"), "y")
            width = _required_int(keys.get("width"), "width")
            height = _required_int(keys.get("height"), "height")
            return cls(
                x1=x,
                y1=y,
                x2=x + width,
                y2=y + height,
                page_index=page_index,
            ).validated()

        raise ValueError(
            "Bounding box mapping must use x1/y1/x2/y2 or x/y/width/height keys."
        )

    @classmethod
    def _from_sequence(cls, value: Sequence[object]) -> "BoundingBox":
        items = list(value)
        if len(items) != 4:
            raise ValueError(f"Bounding box sequence must contain 4 values: {value!r}")
        return cls(
            x1=_required_int(items[0], "x1"),
            y1=_required_int(items[1], "y1"),
            x2=_required_int(items[2], "x2"),
            y2=_required_int(items[3], "y2"),
        ).validated()

    def validated(self) -> "BoundingBox":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"Bounding box has non-positive area: {self.to_dict()}")
        return self

    def to_dict(self) -> JsonObject:
        data: JsonObject = {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
        }
        if self.page_index is not None:
            data["page_index"] = self.page_index
        return data

    def to_roi(self) -> JsonObject:
        return {
            "x": self.x1,
            "y": self.y1,
            "width": self.x2 - self.x1,
            "height": self.y2 - self.y1,
        }

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def iou(self, other: "BoundingBox") -> float:
        if (
            self.page_index is not None
            and other.page_index is not None
            and self.page_index != other.page_index
        ):
            return 0.0

        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        if inter_area == 0:
            return 0.0
        union_area = self.area + other.area - inter_area
        return inter_area / union_area if union_area else 0.0


@dataclass(frozen=True)
class EvaluationCanvas:
    width: int
    height: int
    page_index: int | None = None

    @classmethod
    def from_value(cls, value: object) -> "EvaluationCanvas":
        if not isinstance(value, Mapping):
            raise TypeError("ground_truth.image_dimensions entries must be objects")
        mapping = cast(Mapping[object, object], value)
        keys = {str(key): item for key, item in mapping.items()}
        width = _required_int(keys.get("width"), "width")
        height = _required_int(keys.get("height"), "height")
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        return cls(
            width=width,
            height=height,
            page_index=_optional_int(keys.get("page_index")),
        )

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class GroundTruth:
    fields: FieldValues
    bounding_boxes: tuple[BoundingBox, ...]
    text: str
    image_dimensions: tuple[EvaluationCanvas, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "GroundTruth":
        missing = [key for key in (*PHI_FIELDS, "bounding_boxes") if key not in value]
        if missing:
            raise ValueError(
                "ground_truth is missing required keys: " + ", ".join(missing)
            )

        raw_boxes = value["bounding_boxes"]
        if not isinstance(raw_boxes, Sequence) or isinstance(raw_boxes, (str, bytes)):
            raise TypeError("ground_truth.bounding_boxes must be a sequence")
        box_values = cast(Sequence[object], raw_boxes)

        fields: FieldValues = {
            "first_name": _optional_string(value.get("first_name")),
            "last_name": _optional_string(value.get("last_name")),
            "dob": _optional_string(value.get("dob")),
            "casenumber": _optional_string(value.get("casenumber")),
            "examination_date": _optional_string(value.get("examination_date")),
        }
        text = _optional_string(value.get("text")) or _fallback_ground_truth_text(
            fields
        )
        boxes = tuple(BoundingBox.from_value(item) for item in box_values)
        raw_dimensions = value.get("image_dimensions", ())
        if not isinstance(raw_dimensions, Sequence) or isinstance(
            raw_dimensions, (str, bytes)
        ):
            raise TypeError("ground_truth.image_dimensions must be a sequence")
        dimensions = tuple(
            EvaluationCanvas.from_value(item)
            for item in cast(Sequence[object], raw_dimensions)
        )
        return cls(
            fields=fields,
            bounding_boxes=boxes,
            text=text,
            image_dimensions=dimensions,
        )


@dataclass(frozen=True)
class GoldenSetItem:
    sample_id: str
    input_type: InputType
    source_path: Path
    ground_truth: GroundTruth
    roi: JsonObject | None
    tags: tuple[str, ...]
    subject_id: str | None = None
    source_group_id: str | None = None

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object], line_number: int
    ) -> "GoldenSetItem":
        sample_id = _required_string(value.get("sample_id"), "sample_id")
        input_type = _parse_input_type(
            _required_string(value.get("input_type"), "input_type")
        )

        path_value = _first_present(value, ("source_path", "path", "file_path"))
        source_path = Path(_required_string(path_value, "source_path")).expanduser()
        if not source_path.is_absolute():
            raise ValueError(
                f"Manifest line {line_number} source_path must be absolute: {source_path}"
            )
        if not source_path.exists():
            raise FileNotFoundError(
                f"Manifest line {line_number} source_path does not exist: {source_path}"
            )

        ground_truth_value = value.get("ground_truth")
        if not isinstance(ground_truth_value, Mapping):
            raise TypeError(
                f"Manifest line {line_number} ground_truth must be an object"
            )
        ground_truth_mapping = cast(Mapping[object, object], ground_truth_value)
        ground_truth = GroundTruth.from_mapping(
            _mapping_to_json_object(ground_truth_mapping)
        )

        roi = _optional_json_object(_first_present(value, ("roi", "fixed_roi")))
        tags = _string_tuple(value.get("tags"))
        return cls(
            sample_id=sample_id,
            input_type=input_type,
            source_path=source_path,
            ground_truth=ground_truth,
            roi=roi,
            tags=tags,
            subject_id=_optional_string(value.get("subject_id")),
            source_group_id=_optional_string(value.get("source_group_id")),
        )


@dataclass(frozen=True)
class GoldenSetAudit:
    sample_count: int
    subject_count: int
    source_group_count: int
    issues: tuple[str, ...]

    @property
    def benchmark_ready(self) -> bool:
        return not self.issues


def audit_golden_set(
    items: Sequence[GoldenSetItem],
    *,
    min_samples_per_modality: int = DEFAULT_MIN_SAMPLES_PER_MODALITY,
    min_subjects: int = DEFAULT_MIN_SUBJECTS,
    min_source_groups_per_modality: int = DEFAULT_MIN_SOURCE_GROUPS_PER_MODALITY,
) -> GoldenSetAudit:
    """Reject integration fixtures masquerading as independent model benchmarks."""
    if min_samples_per_modality <= 0 or min_subjects <= 0:
        raise ValueError("golden-set diversity thresholds must be positive")
    if min_source_groups_per_modality <= 0:
        raise ValueError("min_source_groups_per_modality must be positive")

    issues: list[str] = []
    missing_subjects = sum(item.subject_id is None for item in items)
    missing_groups = sum(item.source_group_id is None for item in items)
    if missing_subjects:
        issues.append(f"{missing_subjects} samples are missing subject_id")
    if missing_groups:
        issues.append(f"{missing_groups} samples are missing source_group_id")

    subject_ids = {item.subject_id for item in items if item.subject_id is not None}
    source_group_ids = {
        item.source_group_id for item in items if item.source_group_id is not None
    }
    if len(subject_ids) < min_subjects:
        issues.append(
            f"only {len(subject_ids)} independent subjects; require at least {min_subjects}"
        )

    for input_type in InputType:
        modality_items = [item for item in items if item.input_type is input_type]
        if not modality_items:
            continue
        if len(modality_items) < min_samples_per_modality:
            issues.append(
                f"{input_type.value} has {len(modality_items)} samples; "
                f"require at least {min_samples_per_modality}"
            )
        modality_groups = {
            item.source_group_id
            for item in modality_items
            if item.source_group_id is not None
        }
        if len(modality_groups) < min_source_groups_per_modality:
            issues.append(
                f"{input_type.value} has {len(modality_groups)} independent source groups; "
                f"require at least {min_source_groups_per_modality}"
            )

    return GoldenSetAudit(
        sample_count=len(items),
        subject_count=len(subject_ids),
        source_group_count=len(source_group_ids),
        issues=tuple(issues),
    )


@dataclass(frozen=True)
class PipelinePrediction:
    text: str
    fields: FieldValues
    bounding_boxes: tuple[BoundingBox, ...]
    details: JsonObject


@dataclass(frozen=True)
class PipelineSpec:
    pipeline_id: PipelineId
    input_type: InputType
    name: str
    description: str
    runner: PipelineRunner


@dataclass(frozen=True)
class UtilityMetrics:
    phi_field_recall: float
    field_accuracy: float
    text_score: float
    box_coverage: float
    false_positive_region_fraction: float
    non_phi_area_removed_fraction: float | None
    over_redaction_score: float
    speed_score: float
    stability_score: float
    utility_score: float
    hard_gate_applied: bool

    def to_dict(self) -> JsonObject:
        return {
            "phi_field_recall": self.phi_field_recall,
            "field_accuracy": self.field_accuracy,
            "text_score": self.text_score,
            "box_coverage": self.box_coverage,
            "false_positive_region_fraction": self.false_positive_region_fraction,
            "non_phi_area_removed_fraction": self.non_phi_area_removed_fraction,
            "over_redaction_score": self.over_redaction_score,
            "speed_score": self.speed_score,
            "stability_score": self.stability_score,
            "utility_score": self.utility_score,
            "hard_gate_applied": self.hard_gate_applied,
        }


@dataclass(frozen=True)
class EvaluationResult:
    sample_id: str
    input_type: InputType
    source_path: Path
    pipeline_id: PipelineId
    pipeline_name: str
    status: PipelineStatus
    latency_seconds: float
    rss_delta_mb: float | None
    metrics: UtilityMetrics
    prediction: PipelinePrediction | None
    error: str | None

    def to_json_dict(self, *, include_sensitive_output: bool = False) -> JsonObject:
        row: JsonObject = {
            "event": "sample_result",
            "sample_id": self.sample_id,
            "input_type": self.input_type.value,
            "input_type_label": self.input_type.label,
            "source_path": str(self.source_path),
            "pipeline_id": self.pipeline_id.value,
            "pipeline_name": self.pipeline_name,
            "status": self.status,
            "latency_ms": round(self.latency_seconds * 1000.0, 3),
            "rss_delta_mb": self.rss_delta_mb,
            "metrics": self.metrics.to_dict(),
            "error": self.error,
        }
        if self.prediction is None:
            return row

        predicted_field_presence: dict[str, bool] = {
            field: bool(
                _normalize_field_value(field, self.prediction.fields.get(field))
            )
            for field in PHI_FIELDS
        }
        row["recognized_text_chars"] = len(self.prediction.text)
        row["recognized_text_sha256"] = _hash_text(self.prediction.text)
        row["predicted_field_presence"] = predicted_field_presence
        row["predicted_bounding_boxes"] = [
            box.to_dict() for box in self.prediction.bounding_boxes
        ]
        if include_sensitive_output:
            row["recognized_text"] = self.prediction.text
            row["predicted_fields"] = dict(self.prediction.fields)
            # Backend details may contain raw OCR snippets (for example per-ROI
            # text and word-level recognition results), so they belong behind
            # the same explicit sensitive-output boundary.
            row["pipeline_details"] = self.prediction.details
        return row


@dataclass(frozen=True)
class SummaryRow:
    input_type: InputType
    pipeline_id: PipelineId
    pipeline_name: str
    attempted: int
    succeeded: int
    failed: int
    skipped: int
    mean_utility_score: float
    mean_phi_field_recall: float
    mean_field_accuracy: float
    mean_text_score: float
    mean_box_coverage: float
    mean_false_positive_region_fraction: float
    mean_non_phi_area_removed_fraction: float | None
    mean_over_redaction_score: float
    mean_speed_score: float
    mean_stability_score: float
    mean_latency_ms: float

    def to_json_dict(self) -> JsonObject:
        return {
            "event": "summary",
            "input_type": self.input_type.value,
            "input_type_label": self.input_type.label,
            "pipeline_id": self.pipeline_id.value,
            "pipeline_name": self.pipeline_name,
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "skipped": self.skipped,
            "mean_utility_score": self.mean_utility_score,
            "mean_phi_field_recall": self.mean_phi_field_recall,
            "mean_field_accuracy": self.mean_field_accuracy,
            "mean_text_score": self.mean_text_score,
            "mean_box_coverage": self.mean_box_coverage,
            "mean_false_positive_region_fraction": self.mean_false_positive_region_fraction,
            "mean_non_phi_area_removed_fraction": self.mean_non_phi_area_removed_fraction,
            "mean_over_redaction_score": self.mean_over_redaction_score,
            "mean_speed_score": self.mean_speed_score,
            "mean_stability_score": self.mean_stability_score,
            "mean_latency_ms": self.mean_latency_ms,
        }


class OcrBackendMatrixEvaluator:
    def __init__(
        self,
        *,
        report_native_min_chars: int = DEFAULT_REPORT_NATIVE_MIN_CHARS,
        video_speed_target_seconds: float = DEFAULT_VIDEO_SPEED_TARGET_SECONDS,
        report_speed_target_seconds: float = DEFAULT_REPORT_SPEED_TARGET_SECONDS,
    ) -> None:
        if report_native_min_chars < 0:
            raise ValueError("report_native_min_chars must be non-negative")
        if video_speed_target_seconds <= 0:
            raise ValueError("video_speed_target_seconds must be positive")
        if report_speed_target_seconds <= 0:
            raise ValueError("report_speed_target_seconds must be positive")

        self.report_native_min_chars = report_native_min_chars
        self.video_speed_target_seconds = video_speed_target_seconds
        self.report_speed_target_seconds = report_speed_target_seconds
        self._report_reader: ReportReader | None = None
        self._frame_ocr: FrameOCR | None = None
        self._frame_metadata_extractor: FrameMetadataExtractor | None = None

    def default_pipeline_specs(self) -> tuple[PipelineSpec, ...]:
        return (
            PipelineSpec(
                pipeline_id=PipelineId.V1,
                input_type=InputType.VIDEO_FRAME,
                name="TesseOCR + Fixed ROI",
                description="TesseOCR reader over manifest-provided fixed overlay ROIs.",
                runner=self._run_video_tesseocr_fixed_roi,
            ),
            PipelineSpec(
                pipeline_id=PipelineId.V2,
                input_type=InputType.VIDEO_FRAME,
                name="RapidOCR Detector + TesseOCR Reader",
                description="RapidOCR proposes text boxes; TesseOCR reads each patch.",
                runner=self._run_video_rapidocr_tesseocr,
            ),
            PipelineSpec(
                pipeline_id=PipelineId.V3,
                input_type=InputType.VIDEO_FRAME,
                name="PyTesseract Native Fallback",
                description="FrameOCR-style PyTesseract fallback with light preprocessing.",
                runner=self._run_video_pytesseract_fallback,
            ),
            PipelineSpec(
                pipeline_id=PipelineId.V4,
                input_type=InputType.VIDEO_FRAME,
                name="Full-Frame Baseline",
                description="Unmasked raw full-frame PyTesseract sparse text baseline.",
                runner=self._run_video_full_frame_baseline,
            ),
            PipelineSpec(
                pipeline_id=PipelineId.V5,
                input_type=InputType.VIDEO_FRAME,
                name="Production FrameOCR Cascade",
                description=(
                    "Exact public FrameOCR production call: RapidOCR recognition with "
                    "the configured TesseOCR/PyTesseract and optional vision-LLM fallbacks."
                ),
                runner=self._run_video_production_frame_ocr,
            ),
            PipelineSpec(
                pipeline_id=PipelineId.V6,
                input_type=InputType.VIDEO_FRAME,
                name="Production FrameOCR + PHI Detector",
                description=(
                    "Exact public FrameOCR production call with additive regions from "
                    "the configured production PHI detector."
                ),
                runner=self._run_video_production_frame_ocr_phi_detector,
            ),
            PipelineSpec(
                pipeline_id=PipelineId.R1,
                input_type=InputType.TEXT_REPORT,
                name="Native Extraction",
                description="Programmatic text extraction with pdfplumber or text-file read.",
                runner=self._run_report_native_extraction,
            ),
            PipelineSpec(
                pipeline_id=PipelineId.R2,
                input_type=InputType.TEXT_REPORT,
                name="PyTesseract + Morphological Preprocessing",
                description="Rasterized document OCR with binarization/contrast preprocessing.",
                runner=self._run_report_pytesseract_preprocessed,
            ),
            PipelineSpec(
                pipeline_id=PipelineId.R3,
                input_type=InputType.TEXT_REPORT,
                name="OCR Ensemble",
                description="Lightweight OCR ensemble agreement path for distorted text.",
                runner=self._run_report_ensemble,
            ),
            PipelineSpec(
                pipeline_id=PipelineId.R4,
                input_type=InputType.TEXT_REPORT,
                name="Post-OCR LLM Correction Error Boundary",
                description="OCR stream corrected by the configured LLM provider before parsing.",
                runner=self._run_report_llm_correction,
            ),
        )

    def evaluate(
        self,
        items: Sequence[GoldenSetItem],
        pipeline_ids: set[PipelineId] | None = None,
    ) -> list[EvaluationResult]:
        specs = [
            spec
            for spec in self.default_pipeline_specs()
            if pipeline_ids is None or spec.pipeline_id in pipeline_ids
        ]
        results: list[EvaluationResult] = []
        for item in items:
            for spec in specs:
                if spec.input_type is not item.input_type:
                    continue
                results.append(self.evaluate_one(item, spec))
        return results

    def evaluate_one(self, item: GoldenSetItem, spec: PipelineSpec) -> EvaluationResult:
        self._reset_sample_extractors()
        rss_before = _current_rss_mb()
        started_at = time.perf_counter()
        prediction: PipelinePrediction | None = None
        error: str | None = None
        status: PipelineStatus = "ok"
        try:
            prediction = spec.runner(item)
        except SkipPipeline as exc:
            status = "skipped"
            error = str(exc)
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"

        latency_seconds = time.perf_counter() - started_at
        rss_after = _current_rss_mb()
        rss_delta_mb = None
        if rss_before is not None and rss_after is not None:
            rss_delta_mb = round(rss_after - rss_before, 6)

        speed_target = (
            self.video_speed_target_seconds
            if item.input_type is InputType.VIDEO_FRAME
            else self.report_speed_target_seconds
        )
        if prediction is None:
            prediction = _empty_prediction()
        metrics = calculate_utility_metrics(
            prediction=prediction,
            ground_truth=item.ground_truth,
            latency_seconds=latency_seconds,
            speed_target_seconds=speed_target,
            process_succeeded=status == "ok",
            rss_delta_mb=rss_delta_mb,
        )
        if status != "ok":
            prediction_for_row: PipelinePrediction | None = None
        else:
            prediction_for_row = prediction
        return EvaluationResult(
            sample_id=item.sample_id,
            input_type=item.input_type,
            source_path=item.source_path,
            pipeline_id=spec.pipeline_id,
            pipeline_name=spec.name,
            status=status,
            latency_seconds=latency_seconds,
            rss_delta_mb=rss_delta_mb,
            metrics=metrics,
            prediction=prediction_for_row,
            error=error,
        )

    def _reset_sample_extractors(self) -> None:
        """Start each benchmark case with document-local metadata state."""
        self._report_reader = None
        self._frame_metadata_extractor = None

    def _report_reader_instance(self) -> ReportReader:
        if self._report_reader is None:
            from lx_anonymizer.report_reader import ReportReader

            self._report_reader = ReportReader()
        return self._report_reader

    def _frame_ocr_instance(self) -> FrameOCR:
        if self._frame_ocr is None:
            self._frame_ocr = FrameOCR()
        return self._frame_ocr

    def _frame_metadata_extractor_instance(self) -> FrameMetadataExtractor:
        if self._frame_metadata_extractor is None:
            self._frame_metadata_extractor = FrameMetadataExtractor()
        return self._frame_metadata_extractor

    def _run_video_tesseocr_fixed_roi(self, item: GoldenSetItem) -> PipelinePrediction:
        frame = _load_frame(item.source_path)
        if item.roi is None:
            raise SkipPipeline("V1 requires manifest roi or fixed_roi.")

        processor = get_tesseocr_processor(language="deu+eng")
        text_parts: list[str] = []
        details: JsonObject = {"roi_source": "manifest", "roi_count": 0}
        roi_boxes = _boxes_from_roi(item.roi)
        for index, roi_box in enumerate(roi_boxes):
            roi = roi_box.to_roi()
            text, confidence, metadata = processor.extract_text_from_frame(
                frame,
                roi=roi,
                high_quality=True,
            )
            if text:
                text_parts.append(text)
            details[f"roi_{index}_confidence"] = confidence
            details[f"roi_{index}_metadata"] = _json_safe_mapping(metadata)
        details["roi_count"] = len(roi_boxes)

        text = "\n".join(text_parts).strip()
        return self._frame_prediction(text=text, boxes=roi_boxes, details=details)

    def _run_video_rapidocr_tesseocr(self, item: GoldenSetItem) -> PipelinePrediction:
        frame = _load_frame(item.source_path)
        frame_ocr = self._frame_ocr_instance()
        if not frame_ocr._rapidocr_available:
            raise SkipPipeline("RapidOCR is not available.")
        frame_ocr._ensure_rapidocr_engine()
        rapid_text, rapid_confidence, rapid_metadata = frame_ocr._extract_text_rapidocr(
            frame,
            roi=None,
            high_quality=True,
        )
        proposed_boxes = _boxes_from_rapidocr_metadata(rapid_metadata)
        details: JsonObject = {
            "rapidocr_confidence": rapid_confidence,
            "rapidocr_region_count": len(proposed_boxes),
            "rapidocr_text_chars": len(rapid_text),
        }
        if not proposed_boxes:
            return self._frame_prediction(text=rapid_text, boxes=(), details=details)

        processor = get_tesseocr_processor(language="deu+eng")
        text_parts: list[str] = []
        for index, box in enumerate(proposed_boxes):
            roi = box.to_roi()
            text, confidence, metadata = processor.extract_text_from_frame(
                frame,
                roi=roi,
                high_quality=True,
            )
            if text:
                text_parts.append(text)
            details[f"patch_{index}_confidence"] = confidence
            details[f"patch_{index}_metadata"] = _json_safe_mapping(metadata)

        text = "\n".join(text_parts).strip()
        return self._frame_prediction(text=text, boxes=proposed_boxes, details=details)

    def _run_video_pytesseract_fallback(
        self, item: GoldenSetItem
    ) -> PipelinePrediction:
        frame = _load_frame(item.source_path)
        frame_ocr = self._frame_ocr_instance()
        text, confidence, metadata = frame_ocr._extract_text_pytesseract(
            frame,
            roi=None,
            high_quality=True,
        )
        boxes = _boxes_from_frame_metadata(metadata)
        details: JsonObject = {
            "confidence": confidence,
            "metadata": _json_safe_mapping(metadata),
        }
        return self._frame_prediction(text=text, boxes=boxes, details=details)

    def _run_video_full_frame_baseline(self, item: GoldenSetItem) -> PipelinePrediction:
        frame = _load_frame(item.source_path)
        image = _pil_from_frame(frame)
        text, boxes = _pytesseract_pil_ocr(
            image,
            config="--oem 3 --psm 11",
            lang="deu+eng",
            page_index=None,
        )
        details: JsonObject = {"psm": 11, "structural_layout_assistance": False}
        return self._frame_prediction(text=text, boxes=boxes, details=details)

    def _run_video_production_frame_ocr(
        self, item: GoldenSetItem
    ) -> PipelinePrediction:
        frame = _load_frame(item.source_path)
        frame_ocr = self._frame_ocr_instance()
        text, confidence, metadata = frame_ocr.extract_text_from_frame(
            frame,
            cast(RoiInput, item.roi),
            high_quality=True,
        )
        boxes = _boxes_from_frame_metadata(metadata)
        details: JsonObject = {
            "configuration": "production_frame_ocr_public_api",
            "confidence": confidence,
            "metadata": _json_safe_mapping(metadata),
        }
        return self._frame_prediction(text=text, boxes=boxes, details=details)

    def _run_video_production_frame_ocr_phi_detector(
        self, item: GoldenSetItem
    ) -> PipelinePrediction:
        frame = _load_frame(item.source_path)
        frame_ocr = self._frame_ocr_instance()
        text, confidence, metadata = frame_ocr.extract_text_from_frame(
            frame,
            cast(RoiInput, item.roi),
            high_quality=True,
        )
        ocr_boxes = _boxes_from_frame_metadata(metadata)
        phi_boxes = tuple(
            BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            for x1, y1, x2, y2 in detect_phi_regions_from_settings(
                self._frame_image_for_phi_detection(frame)
            )
        )
        details: JsonObject = {
            "configuration": "production_frame_ocr_plus_phi_detector",
            "confidence": confidence,
            "ocr_box_count": len(ocr_boxes),
            "phi_detector_box_count": len(phi_boxes),
            "metadata": _json_safe_mapping(metadata),
        }
        return self._frame_prediction(
            text=text,
            boxes=(*ocr_boxes, *phi_boxes),
            details=details,
        )

    @staticmethod
    def _frame_image_for_phi_detection(frame: FrameArray) -> Image.Image:
        if frame.ndim == 2:
            return Image.fromarray(frame).convert("RGB")
        return _pil_from_frame(frame)

    def _frame_prediction(
        self,
        *,
        text: str,
        boxes: Sequence[BoundingBox],
        details: JsonObject,
    ) -> PipelinePrediction:
        extractor = self._frame_metadata_extractor_instance()
        fields = _fields_from_meta(extractor.extract_metadata_from_frame_text(text))
        return PipelinePrediction(
            text=text,
            fields=fields,
            bounding_boxes=tuple(boxes),
            details=details,
        )

    def _run_report_native_extraction(self, item: GoldenSetItem) -> PipelinePrediction:
        text = self._load_native_report_text(item)
        if len(text.strip()) < self.report_native_min_chars:
            raise SkipPipeline(
                "Native extraction below threshold "
                f"({len(text.strip())} < {self.report_native_min_chars})."
            )
        return self._report_prediction(
            text=text,
            boxes=(),
            details={"native_text_chars": len(text), "shortcut_downstream_ocr": True},
            source_path=item.source_path,
        )

    def _run_report_pytesseract_preprocessed(
        self, item: GoldenSetItem
    ) -> PipelinePrediction:
        text, boxes, details = self._report_tesseract_preprocessed(item)
        return self._report_prediction(
            text=text,
            boxes=boxes,
            details=details,
            source_path=item.source_path,
        )

    def _run_report_ensemble(self, item: GoldenSetItem) -> PipelinePrediction:
        images = _load_report_images(item.source_path)
        if not images:
            raise SkipPipeline("OCR ensemble requires a PDF or image input.")

        text_parts: list[str] = []
        all_boxes: list[BoundingBox] = []
        page_details: list[JsonObject] = []
        for page_index, image in enumerate(images):
            ensemble_result = ensemble_ocr_with_details(image)
            text_parts.append(ensemble_result.text)
            page_details.append(
                {
                    "page_index": page_index,
                    "selected_engine": ensemble_result.selected_engine.value,
                    "normalized_selection_scores": {
                        engine.value: score
                        for engine, score in ensemble_result.normalized_scores.items()
                    },
                }
            )
            _, page_boxes = _pytesseract_pil_ocr(
                image,
                config="--oem 1 --psm 6",
                lang="deu+eng",
                page_index=page_index,
            )
            all_boxes.extend(page_boxes)
        text = "\n".join(part for part in text_parts if part).strip()
        details: JsonObject = {
            "page_count": len(images),
            "box_source": "pytesseract_reference_boxes",
            "ensemble_pages": page_details,
        }
        return self._report_prediction(
            text=text,
            boxes=tuple(all_boxes),
            details=details,
            source_path=item.source_path,
        )

    def _run_report_llm_correction(self, item: GoldenSetItem) -> PipelinePrediction:
        try:
            text, boxes, details = self._report_tesseract_preprocessed(item)
        except SkipPipeline:
            text = self._load_native_report_text(item)
            boxes = ()
            details: JsonObject = {"ocr_stream_source": "native_text_fallback"}

        if not text.strip():
            raise SkipPipeline("LLM correction requires non-empty OCR or native text.")

        service = LLMService(
            provider=settings.LLM_PROVIDER,
            base_url=settings.resolved_llm_base_url,
        )
        corrected_text = service.correct_ocr_text_in_chunks(text)
        details["llm_provider"] = settings.LLM_PROVIDER
        details["llm_model"] = settings.LLM_MODEL
        details["llm_correction_changed_text"] = corrected_text != text
        return self._report_prediction(
            text=corrected_text,
            boxes=boxes,
            details=details,
            source_path=item.source_path,
        )

    def _load_native_report_text(self, item: GoldenSetItem) -> str:
        suffix = item.source_path.suffix.lower()
        if suffix in {".txt", ".text", ".md"}:
            return item.source_path.read_text(encoding="utf-8")
        if suffix == ".pdf":
            return self._report_reader_instance().read_pdf(item.source_path)
        raise SkipPipeline("Native extraction supports text files and PDFs.")

    def _report_tesseract_preprocessed(
        self, item: GoldenSetItem
    ) -> tuple[str, tuple[BoundingBox, ...], JsonObject]:
        images = _load_report_images(item.source_path)
        if not images:
            raise SkipPipeline("Preprocessed OCR requires a PDF or image input.")

        text_parts: list[str] = []
        all_boxes: list[BoundingBox] = []
        for page_index, image in enumerate(images):
            processed = optimize_image_for_medical_text(image)
            text, boxes = _pytesseract_pil_ocr(
                processed,
                config="--oem 1 --psm 6",
                lang="deu+eng",
                page_index=page_index,
            )
            if text:
                text_parts.append(text)
            all_boxes.extend(boxes)

        details: JsonObject = {
            "page_count": len(images),
            "preprocessing": "optimize_image_for_medical_text",
            "oem": 1,
            "psm": 6,
        }
        return "\n".join(text_parts).strip(), tuple(all_boxes), details

    def _report_prediction(
        self,
        *,
        text: str,
        boxes: Sequence[BoundingBox],
        details: JsonObject,
        source_path: Path,
    ) -> PipelinePrediction:
        fields = _fields_from_meta(
            self._report_reader_instance().extract_report_meta(text, source_path)
        )
        return PipelinePrediction(
            text=text,
            fields=fields,
            bounding_boxes=tuple(boxes),
            details=details,
        )


class SkipPipeline(RuntimeError):
    pass


def load_golden_set(path: Path | str) -> list[GoldenSetItem]:
    manifest_path = Path(path)
    raw_text = manifest_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    raw_records = _load_manifest_records(raw_text, manifest_path)
    items: list[GoldenSetItem] = []
    for line_number, record in raw_records:
        items.append(GoldenSetItem.from_mapping(record, line_number=line_number))
    return items


def calculate_utility_metrics(
    *,
    prediction: PipelinePrediction,
    ground_truth: GroundTruth,
    latency_seconds: float,
    speed_target_seconds: float,
    process_succeeded: bool,
    rss_delta_mb: float | None,
) -> UtilityMetrics:
    phi_field_recall = _phi_field_recall(prediction.fields, ground_truth.fields)
    field_accuracy = _field_accuracy(prediction.fields, ground_truth.fields)
    text_score = _text_similarity_score(prediction.text, ground_truth.text)
    box_coverage = _box_coverage(prediction.bounding_boxes, ground_truth.bounding_boxes)
    false_positive_region_fraction, non_phi_area_removed_fraction = (
        _over_redaction_metrics(
            prediction.bounding_boxes,
            ground_truth.bounding_boxes,
            ground_truth.image_dimensions,
        )
    )
    over_redaction_score = 1.0 - false_positive_region_fraction
    if non_phi_area_removed_fraction is not None:
        over_redaction_score = (
            over_redaction_score + (1.0 - non_phi_area_removed_fraction)
        ) / 2.0
    speed_score = _speed_score(latency_seconds, speed_target_seconds)
    stability_score = _stability_score(
        process_succeeded=process_succeeded,
        rss_delta_mb=rss_delta_mb,
    )
    hard_gate_applied = phi_field_recall < PHI_RECALL_HARD_GATE
    if hard_gate_applied:
        utility_score = 0.0
    else:
        utility_score = (
            UTILITY_WEIGHTS["phi_field_recall"] * phi_field_recall
            + UTILITY_WEIGHTS["field_accuracy"] * field_accuracy
            + UTILITY_WEIGHTS["text_score"] * text_score
            + UTILITY_WEIGHTS["box_coverage"] * box_coverage
            + UTILITY_WEIGHTS["over_redaction_score"] * over_redaction_score
            + UTILITY_WEIGHTS["speed_score"] * speed_score
            + UTILITY_WEIGHTS["stability_score"] * stability_score
        )
    return UtilityMetrics(
        phi_field_recall=round(phi_field_recall, 6),
        field_accuracy=round(field_accuracy, 6),
        text_score=round(text_score, 6),
        box_coverage=round(box_coverage, 6),
        false_positive_region_fraction=round(false_positive_region_fraction, 6),
        non_phi_area_removed_fraction=(
            round(non_phi_area_removed_fraction, 6)
            if non_phi_area_removed_fraction is not None
            else None
        ),
        over_redaction_score=round(over_redaction_score, 6),
        speed_score=round(speed_score, 6),
        stability_score=round(stability_score, 6),
        utility_score=round(utility_score, 6),
        hard_gate_applied=hard_gate_applied,
    )


def aggregate_results(results: Sequence[EvaluationResult]) -> list[SummaryRow]:
    grouped: dict[tuple[InputType, PipelineId], list[EvaluationResult]] = {}
    for result in results:
        key = (result.input_type, result.pipeline_id)
        grouped.setdefault(key, []).append(result)

    rows: list[SummaryRow] = []
    for (input_type, pipeline_id), group in grouped.items():
        attempted_group = [result for result in group if result.status != "skipped"]
        scoring_group = attempted_group or group
        rows.append(
            SummaryRow(
                input_type=input_type,
                pipeline_id=pipeline_id,
                pipeline_name=group[0].pipeline_name,
                attempted=len(attempted_group),
                succeeded=sum(1 for result in group if result.status == "ok"),
                failed=sum(1 for result in group if result.status == "failed"),
                skipped=sum(1 for result in group if result.status == "skipped"),
                mean_utility_score=_mean_metric(
                    scoring_group, lambda result: result.metrics.utility_score
                ),
                mean_phi_field_recall=_mean_metric(
                    scoring_group, lambda result: result.metrics.phi_field_recall
                ),
                mean_field_accuracy=_mean_metric(
                    scoring_group, lambda result: result.metrics.field_accuracy
                ),
                mean_text_score=_mean_metric(
                    scoring_group, lambda result: result.metrics.text_score
                ),
                mean_box_coverage=_mean_metric(
                    scoring_group, lambda result: result.metrics.box_coverage
                ),
                mean_false_positive_region_fraction=_mean_metric(
                    scoring_group,
                    lambda result: result.metrics.false_positive_region_fraction,
                ),
                mean_non_phi_area_removed_fraction=_mean_optional_metric(
                    scoring_group,
                    lambda result: result.metrics.non_phi_area_removed_fraction,
                ),
                mean_over_redaction_score=_mean_metric(
                    scoring_group,
                    lambda result: result.metrics.over_redaction_score,
                ),
                mean_speed_score=_mean_metric(
                    scoring_group, lambda result: result.metrics.speed_score
                ),
                mean_stability_score=_mean_metric(
                    scoring_group, lambda result: result.metrics.stability_score
                ),
                mean_latency_ms=round(
                    _mean_metric(scoring_group, lambda result: result.latency_seconds)
                    * 1000.0,
                    3,
                ),
            )
        )

    return sorted(
        rows,
        key=lambda row: (
            row.input_type.value,
            -row.mean_utility_score,
            row.pipeline_id.value,
        ),
    )


def render_summary_table(rows: Sequence[SummaryRow]) -> str:
    if not rows:
        return "No OCR backend evaluation results."

    lines: list[str] = []
    for input_type in InputType:
        modality_rows = [row for row in rows if row.input_type is input_type]
        if not modality_rows:
            continue
        lines.append(f"{input_type.label}")
        lines.append(
            "rank pipeline utility recall field_acc text box overredact speed stability ok/fail/skip latency_ms"
        )
        for rank, row in enumerate(modality_rows, start=1):
            lines.append(
                " ".join(
                    [
                        str(rank),
                        row.pipeline_id.value,
                        f"{row.mean_utility_score:.3f}",
                        f"{row.mean_phi_field_recall:.3f}",
                        f"{row.mean_field_accuracy:.3f}",
                        f"{row.mean_text_score:.3f}",
                        f"{row.mean_box_coverage:.3f}",
                        f"{row.mean_over_redaction_score:.3f}",
                        f"{row.mean_speed_score:.3f}",
                        f"{row.mean_stability_score:.3f}",
                        f"{row.succeeded}/{row.failed}/{row.skipped}",
                        f"{row.mean_latency_ms:.1f}",
                    ]
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manage.py evaluate_ocr_backend_matrix",
        description="Evaluate OCR backend pipelines with anonymization utility scoring.",
    )
    parser.add_argument(
        "--golden-set",
        required=True,
        type=Path,
        help="Absolute-path JSONL or JSON manifest containing golden-set items.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Write structured JSONL sample and summary rows to this file. Defaults to stdout.",
    )
    parser.add_argument(
        "--pipelines",
        default=None,
        help="Comma-separated pipeline IDs to run, e.g. V1,V2,R1,R4.",
    )
    parser.add_argument(
        "--include-sensitive-output",
        action="store_true",
        help="Include recognized text and predicted PHI fields in JSONL rows.",
    )
    parser.add_argument(
        "--no-summary-table",
        action="store_true",
        help="Do not print the human-readable ranking table to stderr.",
    )
    parser.add_argument(
        "--report-native-min-chars",
        type=int,
        default=DEFAULT_REPORT_NATIVE_MIN_CHARS,
        help="Minimum native report text length before R1 is considered successful.",
    )
    parser.add_argument(
        "--allow-integration-fixture",
        action="store_true",
        help=(
            "Run despite golden-set diversity failures. Results are suitable for "
            "integration checks, not model ranking."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    pipeline_ids = _parse_pipeline_filter(cast(str | None, args.pipelines))
    items = load_golden_set(cast(Path, args.golden_set))
    if not items:
        parser.error("golden set manifest is empty")
    golden_set_audit = audit_golden_set(items)
    if not golden_set_audit.benchmark_ready and not cast(
        bool, args.allow_integration_fixture
    ):
        parser.error(
            "golden set is not benchmark-ready: "
            + "; ".join(golden_set_audit.issues)
            + ". Use --allow-integration-fixture only for smoke/integration runs."
        )

    evaluator = OcrBackendMatrixEvaluator(
        report_native_min_chars=cast(int, args.report_native_min_chars)
    )
    results = evaluator.evaluate(items, pipeline_ids=pipeline_ids)
    summaries = aggregate_results(results)

    output_path = cast(Path | None, args.output_jsonl)
    output_stream: TextIO
    should_close = False
    if output_path is None:
        output_stream = sys.stdout
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_stream = output_path.open("w", encoding="utf-8")
        should_close = True

    try:
        for result in results:
            _write_jsonl(
                output_stream,
                result.to_json_dict(
                    include_sensitive_output=cast(bool, args.include_sensitive_output)
                ),
            )
        for summary in summaries:
            _write_jsonl(output_stream, summary.to_json_dict())
    finally:
        if should_close:
            output_stream.close()

    if not cast(bool, args.no_summary_table):
        print(render_summary_table(summaries), file=sys.stderr)
    return 0


def _load_manifest_records(
    raw_text: str, manifest_path: Path
) -> list[tuple[int, Mapping[str, object]]]:
    if raw_text.startswith("["):
        parsed: object = json.loads(raw_text)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected JSON array in {manifest_path}")
        parsed_items = cast(list[object], parsed)
        records: list[tuple[int, Mapping[str, object]]] = []
        for index, item in enumerate(parsed_items, start=1):
            if not isinstance(item, Mapping):
                raise TypeError(f"Manifest item {index} is not an object")
            item_mapping = cast(Mapping[object, object], item)
            records.append((index, _mapping_to_json_object(item_mapping)))
        return records

    records: list[tuple[int, Mapping[str, object]]] = []
    for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parsed_line: object = json.loads(line)
        if not isinstance(parsed_line, Mapping):
            raise TypeError(f"Manifest line {line_number} is not an object")
        line_mapping = cast(Mapping[object, object], parsed_line)
        records.append((line_number, _mapping_to_json_object(line_mapping)))
    return records


def _parse_input_type(value: str) -> InputType:
    normalized = value.strip().casefold().replace("-", "_")
    if normalized in {"video", "frame", "video_frame", "image_frame"}:
        return InputType.VIDEO_FRAME
    if normalized in {
        "report",
        "text_report",
        "text_report_document",
        "pdf",
        "document",
    }:
        return InputType.TEXT_REPORT
    raise ValueError(f"Unsupported input_type: {value!r}")


def _parse_pipeline_filter(value: str | None) -> set[PipelineId] | None:
    if value is None or not value.strip():
        return None
    pipeline_ids: set[PipelineId] = set()
    for raw_part in value.split(","):
        part = raw_part.strip().upper()
        if not part:
            continue
        try:
            pipeline_ids.add(PipelineId(part))
        except ValueError as exc:
            valid = ", ".join(pipeline.value for pipeline in PipelineId)
            raise ValueError(
                f"Unknown pipeline ID {part!r}; valid IDs: {valid}"
            ) from exc
    return pipeline_ids


def _phi_field_recall(predicted: FieldValues, expected: FieldValues) -> float:
    supported_fields: list[PhiField] = []
    for field in PHI_FIELDS:
        if _normalize_field_value(field, expected[field]):
            supported_fields.append(field)
    if not supported_fields:
        return 1.0
    recalled = sum(
        1
        for field in supported_fields
        if _field_match(
            field=field,
            predicted=predicted.get(field),
            expected=expected.get(field),
        )
    )
    return recalled / len(supported_fields)


def _field_accuracy(predicted: FieldValues, expected: FieldValues) -> float:
    scores = [
        _field_similarity(
            field=field,
            predicted=predicted.get(field),
            expected=expected.get(field),
        )
        for field in PHI_FIELDS
    ]
    return sum(scores) / len(scores)


def _field_match(
    *, field: PhiField, predicted: str | None, expected: str | None
) -> bool:
    expected_norm = _normalize_field_value(field, expected)
    predicted_norm = _normalize_field_value(field, predicted)
    if not expected_norm:
        return not predicted_norm
    if not predicted_norm:
        return False
    if expected_norm in predicted_norm or predicted_norm in expected_norm:
        return True
    return (
        _field_similarity(field=field, predicted=predicted, expected=expected) >= 0.85
    )


def _field_similarity(
    *, field: PhiField, predicted: str | None, expected: str | None
) -> float:
    expected_norm = _normalize_field_value(field, expected)
    predicted_norm = _normalize_field_value(field, predicted)
    if not expected_norm and not predicted_norm:
        return 1.0
    if not expected_norm or not predicted_norm:
        return 0.0
    if field in {"dob", "examination_date", "casenumber"}:
        expected_digits = "".join(_DIGIT_RE.findall(expected_norm))
        predicted_digits = "".join(_DIGIT_RE.findall(predicted_norm))
        if expected_digits and predicted_digits:
            return _normalized_sequence_similarity(predicted_digits, expected_digits)
    return _normalized_sequence_similarity(predicted_norm, expected_norm)


def _normalize_field_value(field: PhiField, value: str | None) -> str:
    if value is None:
        return ""
    normalized = " ".join(value.strip().casefold().split())
    if normalized in {"", "none", "null", "n/a", "na", "unknown", "unbekannt", "-"}:
        return ""
    if field in {"dob", "examination_date"}:
        return normalized.replace("/", ".").replace("-", ".")
    return normalized


def _text_similarity_score(predicted_text: str, expected_text: str) -> float:
    expected = " ".join(expected_text.split())
    predicted = " ".join(predicted_text.split())
    if not expected and not predicted:
        return 1.0
    if not expected:
        return 0.0 if predicted else 1.0
    if not predicted:
        return 0.0

    cer = _normalized_edit_distance(predicted, expected)
    predicted_words = _WORD_RE.findall(predicted.casefold())
    expected_words = _WORD_RE.findall(expected.casefold())
    wer = _normalized_edit_distance_words(predicted_words, expected_words)
    return max(0.0, 1.0 - ((cer + wer) / 2.0))


def _box_coverage(
    predicted_boxes: Sequence[BoundingBox], expected_boxes: Sequence[BoundingBox]
) -> float:
    if not expected_boxes:
        return 1.0 if not predicted_boxes else 0.0
    if not predicted_boxes:
        return 0.0
    total = 0.0
    for expected in expected_boxes:
        total += max(predicted.iou(expected) for predicted in predicted_boxes)
    return total / len(expected_boxes)


def _over_redaction_metrics(
    predicted_boxes: Sequence[BoundingBox],
    expected_boxes: Sequence[BoundingBox],
    image_dimensions: Sequence[EvaluationCanvas],
) -> tuple[float, float | None]:
    predicted_area = _union_box_area(predicted_boxes)
    overlap_area = _union_box_area(_box_intersections(predicted_boxes, expected_boxes))
    false_positive_area = max(0, predicted_area - overlap_area)
    false_positive_fraction = (
        false_positive_area / predicted_area if predicted_area else 0.0
    )

    if not image_dimensions:
        return false_positive_fraction, None

    clipped_predictions = _clip_boxes_to_canvases(predicted_boxes, image_dimensions)
    clipped_expected = _clip_boxes_to_canvases(expected_boxes, image_dimensions)
    canvas_area = sum(canvas.area for canvas in image_dimensions)
    expected_area = _union_box_area(clipped_expected)
    non_phi_area = max(0, canvas_area - expected_area)
    clipped_false_positive_area = max(
        0,
        _union_box_area(clipped_predictions)
        - _union_box_area(_box_intersections(clipped_predictions, clipped_expected)),
    )
    removed_fraction = (
        min(1.0, clipped_false_positive_area / non_phi_area) if non_phi_area else 0.0
    )
    return false_positive_fraction, removed_fraction


def _box_intersections(
    left_boxes: Sequence[BoundingBox], right_boxes: Sequence[BoundingBox]
) -> tuple[BoundingBox, ...]:
    intersections: list[BoundingBox] = []
    for left in left_boxes:
        for right in right_boxes:
            if (
                left.page_index is not None
                and right.page_index is not None
                and left.page_index != right.page_index
            ):
                continue
            x1 = max(left.x1, right.x1)
            y1 = max(left.y1, right.y1)
            x2 = min(left.x2, right.x2)
            y2 = min(left.y2, right.y2)
            if x2 > x1 and y2 > y1:
                intersections.append(
                    BoundingBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        page_index=(
                            left.page_index
                            if left.page_index is not None
                            else right.page_index
                        ),
                    )
                )
    return tuple(intersections)


def _clip_boxes_to_canvases(
    boxes: Sequence[BoundingBox], canvases: Sequence[EvaluationCanvas]
) -> tuple[BoundingBox, ...]:
    clipped: list[BoundingBox] = []
    for box in boxes:
        for canvas in canvases:
            if (
                box.page_index is not None
                and canvas.page_index is not None
                and box.page_index != canvas.page_index
            ):
                continue
            x1 = max(0, box.x1)
            y1 = max(0, box.y1)
            x2 = min(canvas.width, box.x2)
            y2 = min(canvas.height, box.y2)
            if x2 > x1 and y2 > y1:
                clipped.append(
                    BoundingBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        page_index=canvas.page_index,
                    )
                )
    return tuple(clipped)


def _union_box_area(boxes: Sequence[BoundingBox]) -> int:
    total = 0
    pages = {box.page_index for box in boxes}
    for page_index in pages:
        page_boxes = [box for box in boxes if box.page_index == page_index]
        x_edges = sorted({edge for box in page_boxes for edge in (box.x1, box.x2)})
        for left, right in zip(x_edges, x_edges[1:], strict=False):
            if right <= left:
                continue
            intervals = sorted(
                (box.y1, box.y2)
                for box in page_boxes
                if box.x1 < right and box.x2 > left
            )
            if not intervals:
                continue
            covered_height = 0
            start, end = intervals[0]
            for next_start, next_end in intervals[1:]:
                if next_start > end:
                    covered_height += end - start
                    start, end = next_start, next_end
                else:
                    end = max(end, next_end)
            covered_height += end - start
            total += (right - left) * covered_height
    return total


def _speed_score(latency_seconds: float, target_seconds: float) -> float:
    if latency_seconds <= 0:
        return 1.0
    return min(1.0, target_seconds / latency_seconds)


def _stability_score(*, process_succeeded: bool, rss_delta_mb: float | None) -> float:
    if not process_succeeded:
        return 0.0
    if rss_delta_mb is None or rss_delta_mb <= 0:
        return 1.0
    leak_budget_mb = 256.0
    return max(0.0, min(1.0, 1.0 - (rss_delta_mb / leak_budget_mb)))


def _normalized_sequence_similarity(left: str, right: str) -> float:
    distance = _edit_distance(left, right)
    denominator = max(len(left), len(right), 1)
    return max(0.0, 1.0 - (distance / denominator))


def _normalized_edit_distance(left: str, right: str) -> float:
    distance = _edit_distance(left, right)
    return distance / max(len(left), len(right), 1)


def _normalized_edit_distance_words(left: Sequence[str], right: Sequence[str]) -> float:
    distance = _edit_distance_sequence(left, right)
    return distance / max(len(left), len(right), 1)


def _edit_distance(left: str, right: str) -> int:
    return _edit_distance_sequence(tuple(left), tuple(right))


def _edit_distance_sequence[T](left: Sequence[T], right: Sequence[T]) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            cost = 0 if left_item == right_item else 1
            current.append(
                min(
                    current[right_index - 1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def _pytesseract_pil_ocr(
    image: Image.Image,
    *,
    config: str,
    lang: str | None,
    page_index: int | None,
) -> tuple[str, tuple[BoundingBox, ...]]:
    image_for_tesseract = _pytesseract_prepare_image(image)
    if lang is None:
        data = _PYTESSERACT.image_to_data(
            image_for_tesseract,
            config=config,
            output_type=_PYTESSERACT.Output.DICT,
        )
    else:
        data = _PYTESSERACT.image_to_data(
            image_for_tesseract,
            config=config,
            output_type=_PYTESSERACT.Output.DICT,
            lang=lang,
        )

    boxes: list[BoundingBox] = []
    recognized_texts: list[str] = []
    for index, word in enumerate(data["text"]):
        text_value = str(word).strip()
        if not text_value:
            continue
        confidence = _float_or_none(data["conf"][index])
        if confidence is not None and confidence < 0:
            continue
        recognized_texts.append(text_value)
        x = _required_int(data["left"][index], "left")
        y = _required_int(data["top"][index], "top")
        width = _required_int(data["width"][index], "width")
        height = _required_int(data["height"][index], "height")
        if width <= 0 or height <= 0:
            continue
        boxes.append(
            BoundingBox(
                x1=x,
                y1=y,
                x2=x + width,
                y2=y + height,
                page_index=page_index,
            )
        )
    return " ".join(recognized_texts).strip(), tuple(boxes)


def _pytesseract_prepare_image(image: Image.Image) -> Image.Image:
    prepared = (
        image.convert("RGB")
        if ("A" in image.getbands() or image.mode not in {"L", "RGB"})
        else image
    )
    if prepared.format != "JPEG":
        prepared = prepared.copy()
        prepared.format = "JPEG"
    return prepared


def _load_frame(path: Path) -> FrameArray:
    image = _CV2.imread(str(path), _CV2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not load frame image: {path}")
    return np.asarray(image, dtype=np.uint8)


def _pil_from_frame(frame: FrameArray) -> Image.Image:
    if frame.ndim == 3:
        rgb = _CV2.cvtColor(frame, _CV2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    return Image.fromarray(frame)


def _load_report_images(path: Path) -> list[Image.Image]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return convert_pdf_to_images(path)
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
        return [Image.open(path).convert("RGB")]
    return []


def _boxes_from_roi(roi: Mapping[str, object]) -> tuple[BoundingBox, ...]:
    boxes: list[BoundingBox] = []

    def collect(value: object) -> None:
        if isinstance(value, Mapping):
            object_mapping = cast(Mapping[object, object], value)
            mapping = _mapping_to_json_object(object_mapping)
            if {"x", "y", "width", "height"}.issubset(mapping) or {
                "x1",
                "y1",
                "x2",
                "y2",
            }.issubset(mapping):
                boxes.append(BoundingBox.from_value(mapping))
                return
            for item in mapping.values():
                collect(item)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            sequence = cast(Sequence[object], value)
            for item in sequence:
                collect(item)

    collect(roi)
    return tuple(boxes)


def _boxes_from_rapidocr_metadata(
    metadata: Mapping[str, object],
) -> tuple[BoundingBox, ...]:
    regions = metadata.get("text_regions")
    if not isinstance(regions, Sequence) or isinstance(regions, (str, bytes)):
        return ()
    region_values = cast(Sequence[object], regions)
    boxes: list[BoundingBox] = []
    for region in region_values:
        if not isinstance(region, Mapping):
            continue
        object_mapping = cast(Mapping[object, object], region)
        region_mapping = _mapping_to_json_object(object_mapping)
        bbox = region_mapping.get("bbox")
        if bbox is None:
            continue
        try:
            boxes.append(BoundingBox.from_value(bbox))
        except (TypeError, ValueError):
            continue
    return tuple(boxes)


def _boxes_from_frame_metadata(
    metadata: Mapping[str, object],
) -> tuple[BoundingBox, ...]:
    boxes = _boxes_from_rapidocr_metadata(metadata)
    if boxes:
        return boxes
    return ()


def _fields_from_meta(meta: Mapping[str, object]) -> FieldValues:
    return {
        "first_name": _optional_string(meta.get("first_name")),
        "last_name": _optional_string(meta.get("last_name")),
        "dob": _optional_string(meta.get("dob")),
        "casenumber": _optional_string(meta.get("casenumber")),
        "examination_date": _optional_string(meta.get("examination_date")),
    }


def _json_safe_mapping(value: Mapping[str, object]) -> JsonObject:
    return {str(key): _json_safe_value(item) for key, item in value.items()}


def _json_safe_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, Mapping):
        object_mapping = cast(Mapping[object, object], value)
        return _json_safe_mapping(_mapping_to_json_object(object_mapping))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        sequence = cast(Sequence[object], value)
        return [_json_safe_value(item) for item in sequence]
    if hasattr(value, "isoformat"):
        isoformat = getattr(value, "isoformat")
        if callable(isoformat):
            return str(isoformat())
    return str(value)


def _current_rss_mb() -> float | None:
    statm_path = Path("/proc/self/statm")
    try:
        raw = statm_path.read_text(encoding="utf-8").split()
    except OSError:
        return None
    if len(raw) < 2:
        return None
    pages = _float_or_none(raw[1])
    if pages is None:
        return None
    return pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)


def _empty_prediction() -> PipelinePrediction:
    return PipelinePrediction(
        text="",
        fields={field: None for field in PHI_FIELDS},
        bounding_boxes=(),
        details={},
    )


def _fallback_ground_truth_text(fields: FieldValues) -> str:
    return " ".join(value for value in (fields[field] for field in PHI_FIELDS) if value)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mean_metric(
    results: Sequence[EvaluationResult],
    selector: Callable[[EvaluationResult], float],
) -> float:
    if not results:
        return 0.0
    return round(sum(selector(result) for result in results) / len(results), 6)


def _mean_optional_metric(
    results: Sequence[EvaluationResult],
    selector: Callable[[EvaluationResult], float | None],
) -> float | None:
    values = [value for result in results if (value := selector(result)) is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _first_present(mapping: Mapping[str, object], keys: Sequence[str]) -> object | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _optional_json_object(value: object | None) -> JsonObject | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("roi/fixed_roi must be an object when provided")
    object_mapping = cast(Mapping[object, object], value)
    return _mapping_to_json_object(object_mapping)


def _string_tuple(value: object | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("tags must be a sequence of strings")
    sequence = cast(Sequence[object], value)
    tags: list[str] = []
    for item in sequence:
        if not isinstance(item, str):
            raise TypeError("tags must contain only strings")
        tags.append(item)
    return tuple(tags)


def _mapping_to_json_object(value: Mapping[object, object]) -> JsonObject:
    return {str(key): item for key, item in value.items()}


def _required_string(value: object | None, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_string(value: object | None) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        isoformat = getattr(value, "isoformat")
        if callable(isoformat):
            return str(isoformat())
    text = str(value).strip()
    return text or None


def _required_int(value: object | None, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    raise TypeError(f"{name} must be an integer")


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    return _required_int(value, "optional integer")


def _float_or_none(value: object | None) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _write_jsonl(stream: TextIO, row: Mapping[str, object]) -> None:
    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


__all__ = [
    "BoundingBox",
    "EvaluationCanvas",
    "GoldenSetItem",
    "GoldenSetAudit",
    "GroundTruth",
    "InputType",
    "OcrBackendMatrixEvaluator",
    "PHI_RECALL_HARD_GATE",
    "PHI_FIELDS",
    "PipelineId",
    "PipelinePrediction",
    "SummaryRow",
    "UtilityMetrics",
    "aggregate_results",
    "audit_golden_set",
    "calculate_utility_metrics",
    "load_golden_set",
    "main",
    "render_summary_table",
]
