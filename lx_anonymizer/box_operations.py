import numpy as np
import cv2
from .custom_logger import get_logger

logger = get_logger(__name__)

"""
Box Operations

The functions in this script define operations on coordinate 
bounding boxes in images.


"""


def filter_empty_boxes(ocr_results, min_text_len=2):
    """
    ocr_results: List of (text, box)
    Returns only entries where text length >= min_text_len
    """
    filtered = []
    for text, box in ocr_results:
        if len(text.strip()) >= min_text_len:
            filtered.append((text, box))
    return filtered


def get_dominant_color(image, box=None):
    """
    Get the dominant color in a given box region of the image.
    :param image: The input image
    :param box: The bounding box (startX, startY, endX, endY)
    :return: The dominant color as a tuple (B, G, R)
    """
    if not box:
        return image.shape[0:3][::-1]  # Return white if box is invalid
    (startX, startY, endX, endY) = box
    region = image[startY:endY, startX:endX]
    pixels = np.float32(region.reshape(-1, 3))

    n_colors = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]

    logger.debug(f"Found dominant color: {dominant}")
    return tuple(map(int, dominant))


def make_box_from_name(image, name, padding=2):
    """
    Create a bounding box around the given name in the image.

    Parameters:
    image: ndarray
        The image in which to create the bounding box.
    name: str
        The name for which to create the bounding box.
    padding: int
        The padding to add around the name to create the bounding box.

    Returns:
    tuple
        The bounding box coordinates as a tuple of (startX, startY, endX, endY).
    """
    # Get the text size of the name
    text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]

    # Calculate the bounding box coordinates
    startX = max(0, text_size[0] - padding)
    startY = max(0, text_size[1] - padding)
    endX = min(image.shape[1], text_size[0] + padding)
    endY = min(image.shape[0], text_size[1] + padding)
    logger.debug(f"Created bounding box for name '{name}': ({startX}, {startY}, {endX}, {endY})")
    return (startX, startY, endX, endY)


def make_box_from_device_list(x, y, w, h):
    """
    This Function will read the x and y coordinates from the device list and generate box coordinates in a way
    that OpenCV can use them.

    Args:
        x (Int): X-Coordinate
        y (Int): Y-Coordinate
        w (Int): Width
        h (Int): Height

    """
    startX = x
    startY = y
    endX = x + w
    endY = y + h
    logger.debug(f"Created bounding box from device list: ({startX}, {startY}, {endX}, {endY})")
    return startX, startY, endX, endY


def extend_boxes_if_needed(image, boxes, extension_margin=10, color_threshold=30):
    """
    Extends the Box in one direction or the other. This ensures, that the Box coordinates fit over the names.

    Args:
        image (_type_): NumPy Array representing the image (as used in cv2)
        boxes (_type_): Array of the Box Coordinates, that were detected in the image
        extension_margin (int, optional): _description_. Length, that will be added to the side of the box. Defaults to 10.
        color_threshold (int, optional): _description_. The extension is decided based on if the name extends out of the side. This value decides, at what differentiation from the dominant color an extension will happen. Defaults to 30.

    Returns:
        _type_: _description_
    """
    logger.debug(f"Starting box extension to make room for names.")
    extended_boxes = []
    for box in boxes:
        (startX, startY, endX, endY) = box

        # Get the dominant color of the current box
        dominant_color = get_dominant_color(image, box)

        # Check the color signal around the box and decide if extension is needed
        # Check above the box
        if startY - extension_margin > 0:
            upper_region_color = get_dominant_color(image, (startX, startY - extension_margin, endX, startY))
            if np.linalg.norm(np.array(upper_region_color) - np.array(dominant_color)) > color_threshold:
                startY = max(startY - extension_margin, 0)

        # Check below the box
        if endY + extension_margin < image.shape[0]:
            lower_region_color = get_dominant_color(image, (startX, endY, endX, endY + extension_margin))
            if np.linalg.norm(np.array(lower_region_color) - np.array(dominant_color)) > color_threshold:
                endY = min(endY + extension_margin, image.shape[0])

        # Check left of the box
        if startX - extension_margin > 0:
            left_region_color = get_dominant_color(image, (startX - extension_margin, startY, startX, endY))
            if np.linalg.norm(np.array(left_region_color) - np.array(dominant_color)) > color_threshold:
                startX = max(startX - extension_margin, 0)

        # Check right of the box
        if endX + extension_margin < image.shape[1]:
            right_region_color = get_dominant_color(image, (endX, startY, endX + extension_margin, endY))
            if np.linalg.norm(np.array(right_region_color) - np.array(dominant_color)) > color_threshold:
                endX = min(endX + extension_margin, image.shape[1])

        # Add the possibly extended box to the list
        extended_boxes.append((startX, startY, endX, endY))

    logger.debug(f"Extended boxes to make room for names.")
    return extended_boxes


def find_or_create_close_box(phrase_box, boxes, image_width, min_offset=20):
    """Dynamic box creation based on text length"""
    (startX, startY, endX, endY) = phrase_box
    same_line_boxes = [box for box in boxes if abs(box[1] - startY) <= 10]

    # Calculate required width based on text length
    box_width = endX - startX
    required_offset = max(box_width + min_offset, min_offset)

    if same_line_boxes:
        same_line_boxes.sort(key=lambda box: box[0])
        for box in same_line_boxes:
            if box[0] > endX + required_offset:  # Use dynamic offset
                return box

    # Create new box with dynamic sizing
    new_startX = min(endX + required_offset, image_width - box_width)
    new_endX = min(new_startX + box_width, image_width)
    new_box = (new_startX, startY, new_endX, endY)
    return new_box


def combine_boxes(text_with_boxes):
    if not text_with_boxes:
        return text_with_boxes

    text_with_boxes = sorted(text_with_boxes, key=lambda x: (x[1][1], x[1][0]))

    merged_text_with_boxes = [text_with_boxes[0]]

    for current in text_with_boxes[1:]:
        last = merged_text_with_boxes[-1]

        current_text, current_box = current
        last_text, last_box = last

        (last_startX, last_startY, last_endX, last_endY) = last_box
        (current_startX, current_startY, current_endX, current_endY) = current_box

        if last_startY == current_startY and (current_startX - last_endX) <= 10:
            merged_box = (min(last_startX, current_startX), last_startY, max(last_endX, current_endX), last_endY)
            merged_text = last_text + " " + current_text
            merged_text_with_boxes[-1] = (merged_text, merged_box)
        else:
            merged_text_with_boxes.append(current)

    return merged_text_with_boxes


def close_to_box(name_box, phrase_box):
    (startX, startY, _, _) = phrase_box
    return abs(name_box[0] - startX) <= 10 and abs(name_box[1] - startY) <= 10
