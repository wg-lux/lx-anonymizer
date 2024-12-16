import numpy as np
import cv2
from custom_logger import get_logger

logger = get_logger(__name__)

'''
Box Operations

The functions in this script define operations on coordinate 
bounding boxes in images.


'''

def get_dominant_color(image, box):
    """
    Get the dominant color in a given box region of the image.
    :param image: The input image
    :param box: The bounding box (startX, startY, endX, endY)
    :return: The dominant color as a tuple (B, G, R)
    """
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

def make_box_from_name(image, name, padding=10):
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

def make_box_from_device_list(x,y,w,h):
    """
    This Function will read the x and y coordinates from the device list and generate box coordinates in a way
    that OpenCV can use them.

    Args:
        x (Int): X-Coordinate
        y (Int): Y-Coordinate
        w (Int): Width
        h (Int): Height

    """
    startX=x 
    startY=y 
    endX=x+w 
    endY=y+h
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
