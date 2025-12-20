"""
Box Operations

The functions in this script define operations on coordinate
bounding boxes in images.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

from lx_anonymizer.setup.custom_logger import get_logger

# Define a type alias for a bounding box (startX, startY, endX, endY)
Box = Tuple[int, int, int, int]
# Define a type alias for OCR results (text, box)
OcrResult = Tuple[str, Box]

logger = get_logger(__name__)


def filter_empty_boxes(
    ocr_results: List[OcrResult], min_text_len: int = 2
) -> List[OcrResult]:
    """
    Returns only entries where stripped text length >= min_text_len.
    """
    filtered: List[OcrResult] = []
    for text, box in ocr_results:
        if len(text.strip()) >= min_text_len:
            filtered.append((text, box))
    return filtered


def get_dominant_color(
    image: npt.NDArray[np.uint8], box: Optional[Box] = None
) -> Tuple[int, int, int]:
    """
    Get the dominant color in a given box region of the image.
    """
    if box is None:
        # Return average color of the whole image or a default shape
        return tuple(map(int, np.mean(image, axis=(0, 1))))[:3]  # type: ignore

    start_x, start_y, end_x, end_y = box
    region = image[start_y:end_y, start_x:end_x]

    if region.size == 0:
        return (255, 255, 255)

    # Convert to float32 for cv2.kmeans
    pixels = np.float32(region.reshape(-1, 3))

    n_colors = 1
    # cv2 constants often need 'type: ignore' if stubs are missing
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)  # type: ignore
    flags = cv2.KMEANS_RANDOM_CENTERS  # type: ignore

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]

    logger.debug("Found dominant color: %s", dominant)
    return (int(dominant[0]), int(dominant[1]), int(dominant[2]))


def make_box_from_name(
    image: npt.NDArray[np.uint8], name: str, padding: int = 2
) -> Box:
    """
    Create a bounding box around the given name based on font size.
    """
    # Get the text size of the name
    size_result = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_w, text_h = size_result[0]

    start_x = max(0, text_w - padding)
    start_y = max(0, text_h - padding)
    end_x = min(image.shape[1], text_w + padding)
    end_y = min(image.shape[0], text_h + padding)

    logger.debug(
        "Created bounding box for name '%s': (%s, %s, %s, %s)",
        name,
        start_x,
        start_y,
        end_x,
        end_y,
    )
    return (start_x, start_y, end_x, end_y)


def make_box_from_device_list(x: int, y: int, w: int, h: int) -> Box:
    """
    Generate box coordinates from x, y, width, height.
    """
    start_x, start_y = x, y
    end_x, end_y = x + w, y + h
    logger.debug(
        "Created bounding box from device list: (%s, %s, %s, %s)",
        start_x,
        start_y,
        end_x,
        end_y,
    )
    return (start_x, start_y, end_x, end_y)


def extend_boxes_if_needed(
    image: npt.NDArray[np.uint8],
    boxes: List[Box],
    extension_margin: int = 10,
    color_threshold: int = 30,
) -> List[Box]:
    """
    Extends the Box if surrounding colors differ significantly from dominant color.
    """
    logger.debug("Starting box extension to make room for names.")
    extended_boxes: List[Box] = []

    for box in boxes:
        start_x, start_y, end_x, end_y = box
        dominant_color = get_dominant_color(image, box)

        # Above
        if start_y - extension_margin > 0:
            upper_color = get_dominant_color(
                image, (start_x, start_y - extension_margin, end_x, start_y)
            )
            if (
                np.linalg.norm(np.array(upper_color) - np.array(dominant_color))
                > color_threshold
            ):
                start_y = max(start_y - extension_margin, 0)

        # Below
        if end_y + extension_margin < image.shape[0]:
            lower_color = get_dominant_color(
                image, (start_x, end_y, end_x, end_y + extension_margin)
            )
            if (
                np.linalg.norm(np.array(lower_color) - np.array(dominant_color))
                > color_threshold
            ):
                end_y = min(end_y + extension_margin, image.shape[0])

        # Left
        if start_x - extension_margin > 0:
            left_color = get_dominant_color(
                image, (start_x - extension_margin, start_y, start_x, end_y)
            )
            if (
                np.linalg.norm(np.array(left_color) - np.array(dominant_color))
                > color_threshold
            ):
                start_x = max(start_x - extension_margin, 0)

        # Right
        if end_x + extension_margin < image.shape[1]:
            right_color = get_dominant_color(
                image, (end_x, start_y, end_x + extension_margin, end_y)
            )
            if (
                np.linalg.norm(np.array(right_color) - np.array(dominant_color))
                > color_threshold
            ):
                end_x = min(end_x + extension_margin, image.shape[1])

        extended_boxes.append((start_x, start_y, end_x, end_y))

    logger.debug("Extended boxes to make room for names.")
    return extended_boxes


def find_or_create_close_box(
    phrase_box: Box, boxes: List[Box], image_width: int, min_offset: int = 20
) -> Box:
    """Dynamic box creation based on text length"""
    start_x, start_y, end_x, end_y = phrase_box
    same_line_boxes = [b for b in boxes if abs(b[1] - start_y) <= 10]

    box_width = end_x - start_x
    required_offset = max(box_width + min_offset, min_offset)

    if same_line_boxes:
        same_line_boxes.sort(key=lambda b: b[0])
        for b in same_line_boxes:
            if b[0] > end_x + required_offset:
                return b

    new_start_x = min(end_x + required_offset, image_width - box_width)
    new_end_x = min(new_start_x + box_width, image_width)
    return (new_start_x, start_y, new_end_x, end_y)


def combine_boxes(text_with_boxes: List[OcrResult]) -> List[OcrResult]:
    """Merges boxes on the same line that are horizontally close."""
    if not text_with_boxes:
        return text_with_boxes

    # Sort by Y then X
    sorted_items = sorted(text_with_boxes, key=lambda x: (x[1][1], x[1][0]))
    merged: List[OcrResult] = [sorted_items[0]]

    for current_text, current_box in sorted_items[1:]:
        last_text, last_box = merged[-1]

        l_sx, l_sy, l_ex, l_ey = last_box
        c_sx, c_sy, c_ex, c_ey = current_box

        if l_sy == c_sy and (c_sx - l_ex) <= 10:
            new_box = (min(l_sx, c_sx), l_sy, max(l_ex, c_ex), l_ey)
            new_text = f"{last_text} {current_text}"
            merged[-1] = (new_text, new_box)
        else:
            merged.append((current_text, current_box))

    return merged


def close_to_box(name_box: Box, phrase_box: Box) -> bool:
    """Checks if two boxes are within a 10px threshold."""
    return (
        abs(name_box[0] - phrase_box[0]) <= 10
        and abs(name_box[1] - phrase_box[1]) <= 10
    )
