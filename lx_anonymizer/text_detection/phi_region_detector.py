from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence, cast

import cv2
import numpy as np
from PIL import Image

from lx_anonymizer.config import settings
from lx_anonymizer.setup.custom_logger import get_logger

logger = get_logger(__name__)
cv2_dnn = cast("_Cv2DnnModule", cv2.dnn)  # type: ignore[attr-defined]

BoxFormat = Literal["yolo_xywh", "xyxy"]
ScoreFormat = Literal["class_scores", "objectness"]
ResizeMode = Literal["letterbox", "stretch"]


class _DnnNet(Protocol):
    def setInput(self, blob: np.ndarray) -> None: ...

    def forward(self) -> object: ...


class _Cv2DnnModule(Protocol):
    def readNet(self, model_path: str) -> _DnnNet: ...

    def blobFromImage(
        self,
        image: np.ndarray,
        scalefactor: float,
        size: tuple[int, int],
        mean: tuple[float, float, float],
        swapRB: bool,
        crop: bool,
    ) -> np.ndarray: ...

    def NMSBoxes(
        self,
        bboxes: list[list[int]],
        scores: list[float],
        score_threshold: float,
        nms_threshold: float,
    ) -> np.ndarray: ...


class CustomPhiRegionDetectorError(RuntimeError):
    """Raised when a configured custom PHI detector cannot run."""


@dataclass(frozen=True)
class PhiRegionDetectorConfig:
    model_path: Path
    confidence_threshold: float
    nms_threshold: float
    input_size: int
    box_format: BoxFormat
    score_format: ScoreFormat
    allowed_class_ids: frozenset[int]
    resize_mode: ResizeMode = "letterbox"
    expected_sha256: str | None = None
    required: bool = False

    @classmethod
    def from_settings(cls) -> "PhiRegionDetectorConfig | None":
        model_path = settings.PHI_REGION_DETECTOR_MODEL_PATH.strip()
        if not model_path:
            return None

        return cls(
            model_path=Path(model_path).expanduser().resolve(),
            confidence_threshold=float(settings.PHI_REGION_DETECTOR_CONFIDENCE),
            nms_threshold=float(settings.PHI_REGION_DETECTOR_NMS_THRESHOLD),
            input_size=int(settings.PHI_REGION_DETECTOR_INPUT_SIZE),
            box_format=settings.PHI_REGION_DETECTOR_BOX_FORMAT,
            score_format=settings.PHI_REGION_DETECTOR_SCORE_FORMAT,
            allowed_class_ids=_parse_class_ids(settings.PHI_REGION_DETECTOR_CLASS_IDS),
            resize_mode=settings.PHI_REGION_DETECTOR_RESIZE_MODE,
            expected_sha256=(
                settings.PHI_REGION_DETECTOR_MODEL_SHA256.strip().lower() or None
            ),
            required=bool(settings.PHI_REGION_DETECTOR_REQUIRED),
        )


class CustomPhiRegionDetector:
    """
    Baseline local PHI-region detector.

    Contract:
    - model format: OpenCV DNN-readable ONNX
    - input: letterboxed RGB image, NCHW float32, 0..1. Letterboxing matches
      Ultralytics training and preserves the aspect ratio of medical images.
    - output rows: YOLO-like detections with box coords first
      - score_format=class_scores: [cx, cy, w, h, class_0, class_1, ...]
      - score_format=objectness: [cx, cy, w, h, objectness, class_0, ...]
    - box_format=yolo_xywh uses center x/y/width/height; xyxy uses corners.

    The detector returns final sensitive-region boxes. It is additive to the
    deterministic EAST/OCR redaction path; it should improve recall, not replace
    existing safeguards.
    """

    def __init__(self, config: PhiRegionDetectorConfig):
        self.config = config
        self._net: _DnnNet | None = None
        self._validate_config()

    def detect(self, image: Image.Image) -> list[tuple[int, int, int, int]]:
        if self._net is None:
            self._net = cv2_dnn.readNet(str(self.config.model_path))

        rgb = np.asarray(image.convert("RGB"))
        model_input, transform = _prepare_model_input(
            rgb,
            input_size=self.config.input_size,
            resize_mode=self.config.resize_mode,
        )
        blob = cv2_dnn.blobFromImage(
            model_input,
            scalefactor=1.0 / 255.0,
            size=(self.config.input_size, self.config.input_size),
            mean=(0.0, 0.0, 0.0),
            swapRB=False,
            crop=False,
        )

        self._net.setInput(blob)
        outputs = self._net.forward()
        rows = _as_detection_rows(outputs)
        boxes, scores = self._rows_to_boxes(rows, transform)
        return _nms_boxes(boxes, scores, self.config.nms_threshold)

    def _validate_config(self) -> None:
        if not self.config.model_path.exists() or not self.config.model_path.is_file():
            raise CustomPhiRegionDetectorError(
                f"PHI region detector model not found: {self.config.model_path}"
            )
        if self.config.input_size <= 0:
            raise CustomPhiRegionDetectorError(
                "PHI_REGION_DETECTOR_INPUT_SIZE must be positive"
            )
        if not 0.0 <= self.config.confidence_threshold <= 1.0:
            raise CustomPhiRegionDetectorError(
                "PHI_REGION_DETECTOR_CONFIDENCE must be between 0 and 1"
            )
        if not 0.0 <= self.config.nms_threshold <= 1.0:
            raise CustomPhiRegionDetectorError(
                "PHI_REGION_DETECTOR_NMS_THRESHOLD must be between 0 and 1"
            )
        if self.config.resize_mode not in ("letterbox", "stretch"):
            raise CustomPhiRegionDetectorError(
                "PHI_REGION_DETECTOR_RESIZE_MODE must be letterbox or stretch"
            )
        if self.config.expected_sha256:
            actual = _sha256_file(self.config.model_path)
            if actual.lower() != self.config.expected_sha256:
                raise CustomPhiRegionDetectorError(
                    "PHI region detector SHA-256 mismatch. "
                    f"Expected {self.config.expected_sha256}, got {actual}."
                )

    def _rows_to_boxes(
        self,
        rows: np.ndarray,
        transform: _InputTransform,
    ) -> tuple[list[tuple[int, int, int, int]], list[float]]:
        boxes: list[tuple[int, int, int, int]] = []
        scores: list[float] = []

        for row in rows:
            if row.shape[0] < 5:
                continue

            score, class_id = _score_row(row, self.config.score_format)
            if score < self.config.confidence_threshold:
                continue
            if (
                self.config.allowed_class_ids
                and class_id not in self.config.allowed_class_ids
            ):
                continue

            box = _convert_box(
                row[:4],
                transform=transform,
                box_format=self.config.box_format,
            )
            if box is None:
                continue

            boxes.append(box)
            scores.append(score)

        return boxes, scores


_cached_config: PhiRegionDetectorConfig | None = None
_cached_detector: CustomPhiRegionDetector | None = None


def detect_phi_regions_from_settings(
    image: Image.Image,
) -> list[tuple[int, int, int, int]]:
    try:
        config = PhiRegionDetectorConfig.from_settings()
    except Exception as exc:
        if settings.PHI_REGION_DETECTOR_REQUIRED:
            raise CustomPhiRegionDetectorError(
                "Invalid PHI region detector configuration"
            ) from exc
        logger.warning("Ignoring invalid PHI region detector configuration: %s", exc)
        return []

    if config is None:
        return []

    try:
        detector = _get_cached_detector(config)
        regions = detector.detect(image)
        if regions:
            logger.info("Custom PHI detector found %d regions", len(regions))
        return regions
    except Exception as exc:
        if config.required:
            raise CustomPhiRegionDetectorError(
                "Configured PHI region detector failed"
            ) from exc
        logger.warning("Custom PHI detector failed; continuing without it: %s", exc)
        return []


def _get_cached_detector(config: PhiRegionDetectorConfig) -> CustomPhiRegionDetector:
    global _cached_config, _cached_detector

    if _cached_detector is not None and _cached_config == config:
        return _cached_detector

    _cached_config = config
    _cached_detector = CustomPhiRegionDetector(config)
    return _cached_detector


def _parse_class_ids(value: str) -> frozenset[int]:
    if not value.strip():
        return frozenset()

    class_ids: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        class_ids.add(int(part))
    return frozenset(class_ids)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_detection_rows(outputs: object) -> np.ndarray:
    if isinstance(outputs, list):
        raw_outputs: Sequence[object] = cast(Sequence[object], outputs)
    elif isinstance(outputs, tuple):
        raw_outputs = cast(Sequence[object], outputs)
    else:
        raw_outputs = (outputs,)
    arrays: list[np.ndarray] = []

    for output in raw_outputs:
        arr = np.asarray(output, dtype=np.float32)
        if arr.size == 0:
            continue
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            continue
        if arr.shape[0] <= 256 and arr.shape[1] > 256:
            arr = arr.T
        arrays.append(arr)

    if not arrays:
        return np.empty((0, 0), dtype=np.float32)

    return np.concatenate(arrays, axis=0)


def _score_row(row: np.ndarray, score_format: ScoreFormat) -> tuple[float, int]:
    if score_format == "objectness" and row.shape[0] >= 6:
        class_scores = row[5:]
        class_idx = int(np.argmax(class_scores)) if class_scores.size else 0
        class_score = float(class_scores[class_idx]) if class_scores.size else 1.0
        return float(row[4]) * class_score, class_idx

    class_scores = row[4:]
    class_idx = int(np.argmax(class_scores)) if class_scores.size else 0
    score = float(class_scores[class_idx]) if class_scores.size else 0.0
    return score, class_idx


@dataclass(frozen=True)
class _InputTransform:
    image_width: int
    image_height: int
    input_size: int
    scale_x: float
    scale_y: float
    offset_x: int
    offset_y: int


def _prepare_model_input(
    rgb: np.ndarray,
    *,
    input_size: int,
    resize_mode: ResizeMode,
) -> tuple[np.ndarray, _InputTransform]:
    image_height, image_width = rgb.shape[:2]
    if image_width <= 0 or image_height <= 0:
        raise CustomPhiRegionDetectorError("PHI detector input image must not be empty")

    if resize_mode == "stretch":
        resized = np.asarray(
            Image.fromarray(rgb).resize(
                (input_size, input_size), Image.Resampling.BILINEAR
            )
        )
        return resized, _InputTransform(
            image_width=image_width,
            image_height=image_height,
            input_size=input_size,
            scale_x=input_size / image_width,
            scale_y=input_size / image_height,
            offset_x=0,
            offset_y=0,
        )

    scale = min(input_size / image_width, input_size / image_height)
    resized_width = max(1, min(input_size, round(image_width * scale)))
    resized_height = max(1, min(input_size, round(image_height * scale)))
    offset_x = (input_size - resized_width) // 2
    offset_y = (input_size - resized_height) // 2
    resized_image = Image.fromarray(rgb).resize(
        (resized_width, resized_height), Image.Resampling.BILINEAR
    )
    canvas = Image.new("RGB", (input_size, input_size), (114, 114, 114))
    canvas.paste(resized_image, (offset_x, offset_y))
    return np.asarray(canvas), _InputTransform(
        image_width=image_width,
        image_height=image_height,
        input_size=input_size,
        scale_x=scale,
        scale_y=scale,
        offset_x=offset_x,
        offset_y=offset_y,
    )


def _convert_box(
    coords: np.ndarray,
    *,
    transform: _InputTransform,
    box_format: BoxFormat,
) -> tuple[int, int, int, int] | None:
    values = [float(value) for value in coords]
    normalized = max(abs(value) for value in values) <= 1.5

    if box_format == "xyxy":
        x1, y1, x2, y2 = values
    else:
        cx, cy, width, height = values
        x1 = cx - width / 2
        y1 = cy - height / 2
        x2 = cx + width / 2
        y2 = cy + height / 2

    if normalized:
        x1 *= transform.input_size
        x2 *= transform.input_size
        y1 *= transform.input_size
        y2 *= transform.input_size

    x1 = (x1 - transform.offset_x) / transform.scale_x
    x2 = (x2 - transform.offset_x) / transform.scale_x
    y1 = (y1 - transform.offset_y) / transform.scale_y
    y2 = (y2 - transform.offset_y) / transform.scale_y

    left = max(0, min(int(round(x1)), transform.image_width))
    top = max(0, min(int(round(y1)), transform.image_height))
    right = max(0, min(int(round(x2)), transform.image_width))
    bottom = max(0, min(int(round(y2)), transform.image_height))

    if right <= left or bottom <= top:
        return None

    return left, top, right, bottom


def _nms_boxes(
    boxes: list[tuple[int, int, int, int]],
    scores: list[float],
    nms_threshold: float,
) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []

    cv_boxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]
    indices = cv2_dnn.NMSBoxes(
        cv_boxes,
        scores,
        score_threshold=0.0,
        nms_threshold=nms_threshold,
    )

    if len(indices) == 0:
        return []

    flattened = np.asarray(indices).reshape(-1)
    return [boxes[int(index)] for index in flattened]
