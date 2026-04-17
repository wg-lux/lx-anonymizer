from __future__ import annotations

import hashlib
import json
import shutil
import ssl
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import certifi
import cv2
import numpy as np
from typing import cast
from lx_anonymizer.region_processing.box_operations import extend_boxes_if_needed
from lx_anonymizer.setup.custom_logger import get_logger
from lx_anonymizer.setup.directory_setup import create_model_directory
from lx_anonymizer.text_detection.np_wrapper import load_image_into_np

logger = get_logger(__name__)
cv2_dnn = cast(Any, cv2.dnn)  # type: ignore[attr-defined]

# Official model source you currently use.
MODEL_URL = (
    "https://github.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV/raw/master/"
    "frozen_east_text_detection.pb"
)

# Replace this with the real SHA-256 of the exact model file you trust.
# Compute once with:
#   sha256sum frozen_east_text_detection.pb
EXPECTED_MODEL_SHA256 = "REPLACE_WITH_REAL_SHA256"

# Lower bound sanity check only. Integrity must come from SHA-256.
MIN_MODEL_SIZE_BYTES = 90_000_000
DOWNLOAD_TIMEOUT_SECONDS = 30

ALLOWED_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class Detection:
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    confidence: float

    @property
    def width(self) -> int:
        return self.end_x - self.start_x

    @property
    def height(self) -> int:
        return self.end_y - self.start_y

    def to_box(self) -> tuple[int, int, int, int]:
        return (self.start_x, self.start_y, self.end_x, self.end_y)

    def to_dict(self) -> dict[str, Any]:
        return {
            "startX": self.start_x,
            "startY": self.start_y,
            "endX": self.end_x,
            "endY": self.end_y,
            "confidence": self.confidence,
        }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_download(url: str, destination: Path, timeout: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    ssl_context = ssl.create_default_context(cafile=certifi.where())

    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=destination.name + ".tmp-",
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "lx-anonymizer/1.0",
            },
            method="GET",
        )

        with urllib.request.urlopen(
            req, context=ssl_context, timeout=timeout
        ) as response:
            if getattr(response, "status", 200) != 200:
                raise RuntimeError(
                    f"Model download failed with HTTP status {response.status}"
                )

            with tmp_path.open("wb") as out_f:
                shutil.copyfileobj(response, out_f)

        tmp_size = tmp_path.stat().st_size
        if tmp_size < MIN_MODEL_SIZE_BYTES:
            raise RuntimeError(
                f"Downloaded model too small ({tmp_size} bytes); refusing to use it."
            )

        if EXPECTED_MODEL_SHA256 == "REPLACE_WITH_REAL_SHA256":
            raise RuntimeError(
                "EXPECTED_MODEL_SHA256 is not configured. Refusing insecure model download."
            )

        actual_sha256 = sha256_file(tmp_path)
        if actual_sha256.lower() != EXPECTED_MODEL_SHA256.lower():
            raise RuntimeError(
                "Downloaded model SHA-256 mismatch. "
                f"Expected {EXPECTED_MODEL_SHA256}, got {actual_sha256}."
            )

        tmp_path.replace(destination)
        logger.info("Downloaded and verified EAST model at %s", destination)

    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            logger.warning("Failed to clean up temporary model file: %s", tmp_path)
        raise


def _default_east_model_path() -> Path:
    return Path(create_model_directory()) / "frozen_east_text_detection.pb"


def _validate_model_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False

    try:
        if path.stat().st_size < MIN_MODEL_SIZE_BYTES:
            logger.warning(
                "Model exists but is unexpectedly small: %s bytes", path.stat().st_size
            )
            return False

        if EXPECTED_MODEL_SHA256 == "REPLACE_WITH_REAL_SHA256":
            logger.warning(
                "Skipping existing model integrity check because SHA-256 is not configured."
            )
            return False

        actual_sha256 = sha256_file(path)
        if actual_sha256.lower() != EXPECTED_MODEL_SHA256.lower():
            logger.warning("Model SHA-256 mismatch for existing file at %s", path)
            return False

        return True
    except OSError as exc:
        logger.warning("Failed to validate model file %s: %s", path, exc)
        return False


def _ensure_east_model(east_path: str | Path | None = None) -> Path:
    resolved_path = (
        Path(east_path).expanduser().resolve()
        if east_path is not None
        else _default_east_model_path().resolve()
    )

    if _validate_model_file(resolved_path):
        return resolved_path

    if resolved_path.exists():
        logger.warning("Removing invalid or corrupted EAST model: %s", resolved_path)
        resolved_path.unlink(missing_ok=True)

    logger.info("Downloading verified EAST model to %s", resolved_path)
    try:
        atomic_download(MODEL_URL, resolved_path, DOWNLOAD_TIMEOUT_SECONDS)
    except urllib.error.URLError as exc:
        logger.error("Network error while downloading EAST model: %s", exc)
        raise RuntimeError(f"Failed to download EAST model: {exc}") from exc
    except Exception as exc:
        logger.error("Could not prepare EAST model: %s", exc)
        raise

    if not _validate_model_file(resolved_path):
        raise RuntimeError("Downloaded EAST model failed post-download validation.")

    return resolved_path


def validate_image_path(image_path: str | Path) -> Path:
    path = Path(image_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Image path is not a file: {path}")
    if path.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image type: {path.suffix}")

    return path


def non_max_suppression_with_indices(
    boxes: np.ndarray,
    probs: np.ndarray | None = None,
    overlap_thresh: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        selected_boxes: ndarray shape (N, 4)
        selected_indices: ndarray shape (N,)
    """
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.int32), np.empty((0,), dtype=np.int32)

    if boxes.dtype.kind in {"i", "u"}:
        boxes = boxes.astype(np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = np.maximum(0.0, x2 - x1 + 1) * np.maximum(0.0, y2 - y1 + 1)
    idxs = np.argsort(probs if probs is not None else y2)

    picked: list[int] = []

    while idxs.size > 0:
        last = idxs[-1]
        picked.append(int(last))
        idxs = idxs[:-1]

        if idxs.size == 0:
            break

        xx1 = np.maximum(x1[last], x1[idxs])
        yy1 = np.maximum(y1[last], y1[idxs])
        xx2 = np.minimum(x2[last], x2[idxs])
        yy2 = np.minimum(y2[last], y2[idxs])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        overlap = (w * h) / np.maximum(area[idxs], 1e-8)
        idxs = idxs[overlap <= overlap_thresh]

    picked_arr = np.array(picked, dtype=np.int32)
    return boxes[picked_arr].astype(np.int32), picked_arr


def clip_box_to_image(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    start_x, start_y, end_x, end_y = box

    start_x = max(0, min(start_x, image_width - 1))
    start_y = max(0, min(start_y, image_height - 1))
    end_x = max(0, min(end_x, image_width - 1))
    end_y = max(0, min(end_y, image_height - 1))

    if end_x <= start_x or end_y <= start_y:
        return None

    return (start_x, start_y, end_x, end_y)


def merge_close_detections(
    detections: list[Detection],
    horizontal_threshold: int = 10,
    line_threshold: int = 5,
) -> list[Detection]:
    if not detections:
        return []

    merged: list[Detection] = []
    current = detections[0]

    for det in detections[1:]:
        same_line = abs(det.start_y - current.start_y) < line_threshold
        close_horizontally = abs(det.start_x - current.end_x) < horizontal_threshold

        if same_line and close_horizontally:
            current = Detection(
                start_x=current.start_x,
                start_y=min(current.start_y, det.start_y),
                end_x=max(current.end_x, det.end_x),
                end_y=max(current.end_y, det.end_y),
                confidence=max(current.confidence, det.confidence),
            )
        else:
            merged.append(current)
            current = det

    merged.append(current)
    return merged


def sort_detections(
    detections: list[Detection],
    vertical_threshold: int = 10,
) -> list[Detection]:
    return sorted(
        detections,
        key=lambda d: (round(d.start_y / max(vertical_threshold, 1)), d.start_x),
    )


def east_text_detection(
    image_path: str | Path,
    east_path: str | Path | None = None,
    min_confidence: float = 0.6,
    width: int = 320,
    height: int = 320,
) -> tuple[list[tuple[int, int, int, int]], str]:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")
    if not (0.0 <= min_confidence <= 1.0):
        raise ValueError("min_confidence must be between 0.0 and 1.0")

    validated_image_path = validate_image_path(image_path)
    east_model_path = _ensure_east_model(east_path)

    logger.debug(
        "Using EAST model at %s (size=%s bytes)",
        east_model_path,
        east_model_path.stat().st_size,
    )

    orig = load_image_into_np(str(validated_image_path))
    if orig is None:
        raise RuntimeError(f"OpenCV failed to read image: {validated_image_path}")

    orig_h, orig_w = orig.shape[:2]
    new_w, new_h = width, height
    r_w = orig_w / float(new_w)
    r_h = orig_h / float(new_h)

    image = cv2.resize(orig, (new_w, new_h))
    H, W = image.shape[:2]

    layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    logger.debug("Loading EAST text detector")
    net = cv2_dnn.readNet(str(east_model_path))

    blob = cv2_dnn.blobFromImage(
        image,
        scalefactor=1.0,
        size=(W, H),
        mean=(123.68, 116.78, 103.94),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)
    scores, geometry = net.forward(layer_names)

    num_rows, num_cols = scores.shape[2:4]
    rects: list[tuple[int, int, int, int]] = []
    confidences: list[float] = []

    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(num_cols):
            score = float(scores_data[x])
            if score < min_confidence:
                continue

            offset_x, offset_y = x * 4.0, y * 4.0
            angle = float(angles_data[x])
            cos = float(np.cos(angle))
            sin = float(np.sin(angle))

            h = float(x_data0[x] + x_data2[x])
            w = float(x_data1[x] + x_data3[x])

            if w <= 0 or h <= 0:
                continue

            end_x = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
            end_y = int(offset_y - (sin * x_data1[x]) + (cos * x_data2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(score)

    if not rects:
        return [], "[]"

    rects_np = np.array(rects, dtype=np.int32)
    confidences_np = np.array(confidences, dtype=np.float32)

    nms_boxes, selected_indices = non_max_suppression_with_indices(
        rects_np,
        probs=confidences_np,
        overlap_thresh=0.2,
    )

    detections: list[Detection] = []

    min_width = 15
    max_width = int(orig_w * 0.3)
    min_height = 8
    max_height = int(orig_h * 0.1)

    for box, original_idx in zip(nms_boxes, selected_indices, strict=False):
        start_x, start_y, end_x, end_y = box.tolist()

        # Scale back to original image.
        scaled_box = (
            int(start_x * r_w),
            int(start_y * r_h),
            int(end_x * r_w),
            int(end_y * r_h),
        )

        clipped = clip_box_to_image(scaled_box, orig_w, orig_h)
        if clipped is None:
            continue

        c_start_x, c_start_y, c_end_x, c_end_y = clipped
        det = Detection(
            start_x=c_start_x,
            start_y=c_start_y,
            end_x=c_end_x,
            end_y=c_end_y,
            confidence=float(confidences_np[original_idx]),
        )

        width_px = det.width
        height_px = det.height

        if not (
            min_width <= width_px <= max_width
            and min_height <= height_px <= max_height
            and (width_px / max(height_px, 1)) < 15
        ):
            continue

        detections.append(det)

    if not detections:
        return [], "[]"

    vertical_threshold = max(2, int(0.03 * orig_h))
    horizontal_threshold = max(2, int(0.03 * orig_w))

    detections = sort_detections(detections, vertical_threshold=vertical_threshold)
    detections = merge_close_detections(
        detections,
        horizontal_threshold=horizontal_threshold,
        line_threshold=5,
    )

    # Your helper works on tuples, so convert back and forth.
    extended_boxes = extend_boxes_if_needed(
        orig,
        [d.to_box() for d in detections],
        extension_margin=2,
    )

    final_detections: list[Detection] = []
    for original_det, extended_box in zip(detections, extended_boxes, strict=False):
        clipped = clip_box_to_image(extended_box, orig_w, orig_h)
        if clipped is None:
            continue
        final_detections.append(
            Detection(
                start_x=clipped[0],
                start_y=clipped[1],
                end_x=clipped[2],
                end_y=clipped[3],
                confidence=original_det.confidence,
            )
        )

    output_boxes = [d.to_box() for d in final_detections]
    output_confidences = [d.to_dict() for d in final_detections]

    return output_boxes, json.dumps(output_confidences)
