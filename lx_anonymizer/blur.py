import cv2
import uuid
from directory_setup import create_temp_directory, create_blur_directory
from box_operations import get_dominant_color
from region_detector import expand_roi
from pathlib import Path
from custom_logger import get_logger

logger = get_logger(__name__)


temp_dir, base_dir, csv_dir = create_temp_directory()

def blur_function(image_path, box, background_color=None, expansion=5, blur_strength=(51, 51), rectangle_scale=0.8):
    
    """
    Apply a strong Gaussian blur to the specified ROI in the image and slightly extend the blur outside the ROI.

    Parameters:
    image_path: str
        The path to the image file on which to apply the blurring.
    box: tuple
        The bounding box as a tuple of (startX, startY, endX, endY).
    background_color: tuple
        The background color to fill the rectangle. Default is None.
    expansion: int
        The number of pixels to expand the blur beyond the ROI.
    blur_strength: tuple
        The size of the Gaussian kernel to use for blurring.
    rectangle_scale: float
        The scale of the rectangle relative to the expanded ROI.

    Returns:
    str
        The path to the saved output image.
    """
    logger.info("Applying blur to the specified region")
    blur_dir = create_blur_directory()
    if blur_dir is None:
        raise ValueError("Blur directory could not be created or accessed")
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    (startX, startY, endX, endY) = box

    # Expand the ROI to include a border around the detected region
    (startX, startY, endX, endY) = expand_roi(startX, startY, endX, endY, expansion, image.shape)

    # Extract the expanded ROI from the image
    roi = image[startY:endY, startX:endX]
    
    # Use the provided background color or default to the dominant color in the ROI
    if background_color is not None:
        dominant_color = background_color
    else:
        dominant_color = get_dominant_color(image, box)

    # Calculate the dimensions for the smaller rectangle
    rect_width = int((endX - startX) * rectangle_scale)
    rect_height = int((endY - startY) * rectangle_scale)
    rect_startX = startX + (endX - startX - rect_width) // 2
    rect_startY = startY + (endY - startY - rect_height) // 2

    # Draw the rectangle on the blurred image
    cv2.rectangle(image, (rect_startX, rect_startY), (rect_startX + rect_width, rect_startY + rect_height), dominant_color, -1)

    # Apply a strong Gaussian blur to the ROI
    blurred_roi = cv2.GaussianBlur(roi, blur_strength, 0)

    # Replace the original image's ROI with the blurred one
    image[startY:endY, startX:endX] = blurred_roi

    # Save the modified image to a file
    output_image_path = Path(blur_dir)/ f"{uuid.uuid4().hex}.png"
    logger.debug(f"Blurred Image will be saved to: {blur_dir}")
    cv2.imwrite(str(output_image_path), image)
    logger.info(f"Blurred Image saved to {blur_dir}")

    return output_image_path