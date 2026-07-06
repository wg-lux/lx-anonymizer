import difflib
from collections.abc import Sequence
from os import PathLike
from typing import Optional, cast

import cv2
import numpy as np

from lx_anonymizer._native import native as _native
from lx_anonymizer.setup.custom_logger import get_logger

logger = get_logger(__name__)


Box = tuple[int, int, int, int]


def fuzzy_match_snippet(
    snippet_text: str, candidates: Sequence[str], threshold: float = 0.7
) -> tuple[Optional[str], float]:
    """
    Find the best fuzzy match for 'snippet_text' among a list of 'candidates'.
    Returns (best_match, best_ratio).
    Only returns a valid match if best_ratio >= threshold; otherwise (None, best_ratio).
    """
    logger.debug(f"Fuzzy matching: snippet text: {snippet_text}")

    if _native is not None:
        return cast(
            tuple[Optional[str], float],
            _native.fuzzy_match_best(snippet_text, candidates, threshold),
        )

    best_match: Optional[str] = None
    best_ratio = 0.0

    for candidate in candidates:
        ratio = difflib.SequenceMatcher(None, snippet_text, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = candidate

    if best_ratio >= threshold:
        return best_match, best_ratio
    else:
        return None, best_ratio


def correct_box_for_new_text(
    image_path: str | PathLike[str],
    snippet_box: Box,
    old_text: str,
    new_text: str,
    extension_margin: int = 15,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1.0,
    font_thickness: int = 2,
) -> Box:
    """
    If the new (corrected) text is significantly longer than the old text,
    expand the bounding box accordingly. Uses some of the box_operations helpers.
    """
    image = cast(Optional[np.ndarray], cv2.imread(str(image_path)))
    if image is None:
        raise ValueError(f"Could not read image for box correction: {image_path}")

    # Calculate the text size of the old and new text
    old_size = cv2.getTextSize(old_text, font_face, font_scale, font_thickness)[0]
    new_size = cv2.getTextSize(new_text, font_face, font_scale, font_thickness)[0]

    (startX, startY, endX, endY) = snippet_box
    old_width = old_size[0]
    new_width = new_size[0]

    # If the new text is significantly wider, expand the box horizontally
    delta = new_width - old_width
    if delta > 0:
        # Add margin to the right side of the box
        endX += delta + extension_margin
        if endX > image.shape[1]:  # clamp to image boundaries
            endX = image.shape[1]

    # Optionally, you could adjust height if needed
    # For instance, if new text has more lines, etc.

    # Return the new bounding box
    return (startX, startY, endX, endY)
