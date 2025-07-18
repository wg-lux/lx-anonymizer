import difflib
import cv2
from .custom_logger import get_logger

logger = get_logger(__name__)

def fuzzy_match_snippet(snippet_text, candidates, threshold=0.7):
    """
    Find the best fuzzy match for 'snippet_text' among a list of 'candidates'.
    Returns (best_match, best_ratio).
    Only returns a valid match if best_ratio >= threshold; otherwise (None, best_ratio).
    """
    best_match = None
    best_ratio = 0.0
    
    logger.debug(f"Fuzzy matching: snippet text: {snippet_text}")

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
    image_path,
    snippet_box,
    old_text,
    new_text,
    extension_margin=15,
    font_face=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1.0,
    font_thickness=2
):
    """
    If the new (corrected) text is significantly longer than the old text,
    expand the bounding box accordingly. Uses some of the box_operations helpers.
    """
    image = cv2.imread(str(image_path))

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


