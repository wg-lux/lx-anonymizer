from hezar.models import Model
from hezar.utils import load_image
from .custom_logger import get_logger
from pathlib import Path
import cv2
import json
import numpy as np
from .box_operations import extend_boxes_if_needed
import torch

# Import PIL.Image to check if input is already a PIL Image
from PIL import Image

logger = get_logger(__name__)


def craft_text_detection(image_input, min_confidence=0.5, width=320, height=320):
    """
    Performs CRAFT text detection on the input image.

    Accepts either:
      - A file path (str or pathlib.Path), or
      - A PIL.Image.Image object.

    The function converts the input to a format suitable for both OpenCV (for dimension calculations)
    and the text detection model.
    """
    try:
        # Determine if input is a file path or a PIL image, and load appropriately.
        if isinstance(image_input, (str, Path)):
            image_file_path = str(image_input)
            # Use OpenCV to load the image for dimension calculations.
            orig = cv2.imread(image_file_path)
            if orig is None:
                raise FileNotFoundError(f"Failed to load image: {image_input}")
            # Load the image using your custom load_image utility.
            image = load_image(image_file_path)
        elif isinstance(image_input, Image.Image):
            # Input is already a PIL image. Convert to an OpenCV compatible image.
            orig = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
            image = image_input  # Use the provided PIL image directly.
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Original image dimensions.
        (origH, origW) = orig.shape[:2]
        rW = origW / float(width)
        rH = origH / float(height)

        logger.info("Loading CRAFT model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Model.load("hezarai/CRAFT", device=device)

        logger.info("Running CRAFT text detection...")
        # Run the model prediction with your desired thresholds.
        outputs = model.predict(
            image,
            text_threshold=0.5,  # Higher threshold for more confident word detection
            link_threshold=0.3,  # Lower threshold to better separate words
            low_text=0.4,  # Balance between detection and separation
            poly=False,  # Use rectangles for simpler processing
        )
        logger.info("Detection complete.")

        # Filtering parameters for text regions.
        min_width = 15  # Minimum width for a word
        max_width = int(origW * 0.2)  # Maximum 20% of image width
        min_height = 8  # Minimum height for a word
        max_height = int(origH * 0.1)  # Maximum 10% of image height
        aspect_ratio_threshold = 10.0  # Maximum width/height ratio

        output_boxes = []
        output_confidences = []

        # Check that outputs have the expected format.
        if outputs and isinstance(outputs, list) and len(outputs) > 0 and "boxes" in outputs[0]:
            boxes = outputs[0]["boxes"]
            for box in boxes:
                points = np.array(box, dtype=np.int32)
                logger.debug(f"Box shape: {points.shape}")

                if len(points.shape) == 1 and points.shape[0] == 4:
                    # Handle flat array [x1, y1, x2, y2]
                    x_coords = points[::2]
                    y_coords = points[1::2]

                    # Calculate bounding box coordinates
                    startX = int(min(x_coords) * rW)
                    startY = int(min(y_coords) * rH)
                    endX = int(max(x_coords) * rW)
                    endY = int(max(y_coords) * rH)

                    # Calculate box dimensions
                    box_width = endX - startX
                    box_height = endY - startY

                    # Apply size and aspect ratio filters.
                    if min_width <= box_width <= max_width and min_height <= box_height <= max_height and box_width / box_height <= aspect_ratio_threshold:
                        # Ensure coordinates are within image bounds.
                        startX = max(0, min(startX, origW - 1))
                        startY = max(0, min(startY, origH - 1))
                        endX = max(0, min(endX, origW))
                        endY = max(0, min(endY, origH))

                        # Only add valid boxes with a sufficient area.
                        if startX < endX and startY < endY and (endX - startX) * (endY - startY) >= 100:
                            output_boxes.append((startX, startY, endX, endY))
                            output_confidences.append({"startX": startX, "startY": startY, "endX": endX, "endY": endY, "confidence": min_confidence})

        if output_boxes:
            logger.info(f"Detected {len(output_boxes)} text regions")
            # Merge boxes that are very close and likely parts of the same word.
            output_boxes = merge_close_boxes(output_boxes)
            # Sort boxes with a reduced vertical threshold.
            output_boxes = sort_boxes(output_boxes, vertical_threshold=5)
            # Optionally, extend boxes with minimal margins.
            output_boxes = extend_boxes_if_needed(orig, output_boxes, extension_margin=2)

        return output_boxes, json.dumps(output_confidences)

    except Exception as e:
        logger.error(f"Error in CRAFT text detection: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise


def merge_close_boxes(boxes, horizontal_threshold=8):
    """Merge boxes that are horizontally very close and likely part of the same word."""
    if not boxes:
        return boxes

    merged = []
    current_box = list(boxes[0])

    for box in boxes[1:]:
        # Check if boxes are close horizontally and on the same line.
        if abs(box[0] - current_box[2]) < horizontal_threshold and abs(box[1] - current_box[1]) < 5:  # Vertical alignment threshold.
            # Merge boxes.
            current_box[2] = box[2]  # Extend to the end of the next box.
            current_box[3] = max(current_box[3], box[3])  # Take the max height.
        else:
            merged.append(tuple(current_box))
            current_box = list(box)

    merged.append(tuple(current_box))
    return merged


def sort_boxes(boxes, vertical_threshold=5):
    """Sort boxes by vertical position then horizontal position with a tighter threshold."""
    boxes.sort(key=lambda b: (round(b[1] / vertical_threshold), b[0]))
    return boxes
