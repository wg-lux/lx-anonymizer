from .custom_logger import get_logger
from pathlib import Path
logger=get_logger(__name__)

def expand_roi(startX, startY, endX, endY, expansion, image_shape):
    """
    Expand the ROI by a certain number of pixels in all directions and ensure it is within image boundaries.

    Parameters:
    startX, startY, endX, endY: int
        The starting and ending coordinates of the ROI.
    expansion: int
        The number of pixels to expand the ROI in all directions.
    image_shape: tuple
        The shape of the image to ensure the expanded ROI is within the bounds.

    Returns:
    tuple
        The expanded ROI coordinates.
    """
    startX = max(0, startX - expansion)
    startY = max(0, startY - expansion)
    endX = min(image_shape[1], endX + expansion)
    endY = min(image_shape[0], endY + expansion)
    logger.debug(f"Expanded ROI to ({startX}, {startY}, {endX}, {endY})")
    return (startX, startY, endX, endY)