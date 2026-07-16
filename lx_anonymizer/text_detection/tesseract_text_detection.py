import json
from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias, cast

import cv2
import numpy as np
import pytesseract  # type: ignore[import-untyped, reportMissingTypeStubs]
from pytesseract import Output  # type: ignore[import-untyped, reportMissingTypeStubs]

from lx_dtypes.models.contracts.text_detection import (
    TesseractOCRData,
    TesseractWordConfidence,
)
from lx_anonymizer.region_processing.box_operations import extend_boxes_if_needed
from lx_anonymizer.setup.custom_logger import get_logger
from lx_anonymizer.text_detection.np_wrapper import load_image_into_np

logger = get_logger(__name__)

Box: TypeAlias = tuple[int, int, int, int]


def tesseract_text_detection(
    image_path: str | Path,
    min_confidence: float = 0.5,
    width: int = 320,
    height: int = 320,
) -> tuple[list[Box], str]:
    """
    Detects text at word level using Tesseract OCR.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")

    # Load the input image
    image = load_image_into_np(image_path)
    logger.debug("Loading tesseract text detection")

    orig = image.copy()
    (H, W) = image.shape[:2]

    # Resize the image and grab the new image dimensions
    image = cv2.resize(image, (width, height))
    (rH, rW) = H / float(height), W / float(width)

    # Configure Tesseract for word-level detection
    custom_config = r"--oem 3 --psm 11"  # PSM 11 for sparse text with OEM 3 (default)

    # Detecting text using Tesseract with word-level configuration
    results = _load_tesseract_ocr_data(image, custom_config)

    output_boxes: list[Box] = []
    output_confidences: list[TesseractWordConfidence] = []

    # Add size filtering parameters
    min_width = 15  # Minimum width for a word
    max_width = int(W * 0.2)  # Maximum width (20% of image width)
    min_height = 8  # Minimum height for a word
    max_height = int(H * 0.1)  # Maximum height (10% of image height)
    aspect_ratio_threshold = 10.0  # Maximum width/height ratio

    # Loop over each of the individual text localizations
    for i in range(0, len(results.text)):
        # Extract the bounding box coordinates and confidence
        x, y, w, h = (
            results.left[i],
            results.top[i],
            results.width[i],
            results.height[i],
        )
        conf = float(results.conf[i])
        text = results.text[i].strip()

        # Filter out weak detections and empty text
        if (
            conf > min_confidence
            and text  # Check if text is not empty
            and not text.isspace()  # Check if text is not just whitespace
            and w >= min_width
            and w <= max_width  # Width constraints
            and h >= min_height
            and h <= max_height  # Height constraints
            and w / h <= aspect_ratio_threshold
        ):  # Aspect ratio check
            # Scale the bounding box coordinates back to the original image size
            startX = int(x * rW)
            startY = int(y * rH)
            endX = int((x + w) * rW)
            endY = int((y + h) * rH)

            # Additional check for minimum box area
            if (endX - startX) * (endY - startY) >= 100:  # Minimum area of 100 pixels
                box: Box = (startX, startY, endX, endY)
                output_boxes.append(box)
                output_confidences.append(
                    TesseractWordConfidence(
                        startX=startX,
                        startY=startY,
                        endX=endX,
                        endY=endY,
                        confidence=conf,
                        text=text,
                    )
                )

    # Merge very close boxes that might be parts of the same word
    output_boxes = merge_close_boxes(output_boxes)

    # Sort boxes with a smaller vertical threshold
    output_boxes = sort_boxes(output_boxes, vertical_threshold=5)

    # Extend boxes with minimal margins
    output_boxes = extend_boxes_if_needed(orig, output_boxes, extension_margin=2)

    logger.info("Tesseract text detection complete. Found %s words.", len(output_boxes))
    return output_boxes, json.dumps(
        [confidence.model_dump(by_alias=True) for confidence in output_confidences]
    )


def merge_close_boxes(boxes: list[Box], horizontal_threshold: int = 8) -> list[Box]:
    """Merge boxes that are horizontally very close and likely part of the same word."""
    if not boxes:
        return []
    if horizontal_threshold < 1:
        raise ValueError("horizontal_threshold must be a positive integer")

    merged: list[Box] = []
    current_box: list[int] = list(boxes[0])

    for box in boxes[1:]:
        # Check if boxes are close horizontally and on the same line
        if (
            abs(box[0] - current_box[2]) < horizontal_threshold
            and abs(box[1] - current_box[1]) < 5
        ):  # Vertical alignment threshold
            # Merge boxes
            current_box[2] = box[2]  # Extend to end of next box
            current_box[3] = max(current_box[3], box[3])  # Take max height
        else:
            merged_box: Box = (
                current_box[0],
                current_box[1],
                current_box[2],
                current_box[3],
            )
            merged.append(merged_box)
            current_box = list(box)

    merged_box = (
        current_box[0],
        current_box[1],
        current_box[2],
        current_box[3],
    )
    merged.append(merged_box)
    return merged


def sort_boxes(boxes: list[Box], vertical_threshold: int = 5) -> list[Box]:
    """Sort boxes by vertical position then horizontal position."""
    if vertical_threshold < 1:
        raise ValueError("vertical_threshold must be a positive integer")
    return sorted(boxes, key=lambda b: (round(b[1] / vertical_threshold), b[0]))


def _load_tesseract_ocr_data(image: np.ndarray, custom_config: str) -> TesseractOCRData:
    raw_payload: object = cast(
        object,
        pytesseract.image_to_data(  # pyright: ignore[reportUnknownMemberType]
            image, output_type=Output.DICT, config=custom_config
        ),
    )
    return normalize_tesseract_ocr_data(raw_payload)


def normalize_tesseract_ocr_data(raw_payload: object) -> TesseractOCRData:
    """Normalize pytesseract's larger dictionary at the integration boundary."""
    if not isinstance(raw_payload, Mapping):
        raise TypeError("pytesseract image_to_data output must be a mapping")

    required_columns = ("left", "top", "width", "height", "conf", "text")
    missing_columns = [key for key in required_columns if key not in raw_payload]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"pytesseract output is missing required columns: {missing}")

    normalized_payload: dict[str, object] = {
        key: raw_payload[key] for key in required_columns
    }
    return TesseractOCRData.model_validate(normalized_payload)
